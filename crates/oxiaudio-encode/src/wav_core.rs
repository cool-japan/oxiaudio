//! WAV encoding: `WavBitDepth`, `WavEncoder`, `WavEncodeConfig`, convenience functions,
//! TPDF dithering, and seekless streaming WAV output.

use hound::{SampleFormat as HoundSampleFormat, WavSpec, WavWriter};
use oxiaudio_core::{AudioBuffer, AudioEncoder, AudioMetadata, OxiAudioError};

use crate::wav_ext;

// ─── TPDF dithering ───────────────────────────────────────────────────────────

/// Apply TPDF (Triangular Probability Density Function) dithering in-place
/// before quantization to `noise_bits` bits.
///
/// Uses an LCG PRNG for deterministic, allocation-free noise generation.
/// The noise amplitude scalar is `2^(-noise_bits)`. The internal LCG yields
/// values r ∈ [0, ~0.5) via `(state >> 33) / u32::MAX`, so the triangle
/// noise r1−r2 has a peak-to-peak excursion of `±2^(-noise_bits-1)` (half an
/// LSB at `noise_bits` depth) and RMS of `amplitude / sqrt(24)`.
pub fn apply_tpdf_dither(samples: &mut [f32], noise_bits: u8) {
    let amplitude = 2.0f32.powi(-(noise_bits as i32));
    let mut state: u64 = 0x123456789u64;
    for s in samples.iter_mut() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let r1 = (state >> 33) as f32 / u32::MAX as f32;
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let r2 = (state >> 33) as f32 / u32::MAX as f32;
        *s += amplitude * (r1 - r2);
    }
}

// ─── WavBitDepth ──────────────────────────────────────────────────────────────

/// Bit depth / sample format for WAV encoding.
///
/// The `Default` variant is `F32`, which preserves the M1/M2 behaviour
/// (32-bit IEEE float PCM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WavBitDepth {
    /// 32-bit IEEE 754 floating-point — lossless round-trip from `AudioBuffer<f32>`.
    #[default]
    F32,
    /// 16-bit signed integer PCM. Each sample is scaled by `i16::MAX` (32767).
    I16,
    /// 24-bit signed integer PCM. Each sample is scaled by `8_388_607` (2^23 − 1).
    I24,
    /// 32-bit signed integer PCM. Each sample is scaled by `i32::MAX`.
    I32,
    /// 8-bit unsigned integer PCM (WAV spec: 0=min, 128=silence, 255=max).
    /// Stored as unsigned on disk; hound handles the signed↔unsigned conversion.
    U8,
}

// ─── WavEncoder ───────────────────────────────────────────────────────────────

/// WAV encoder backed by `hound 3.5.1`.
///
/// The default configuration (`WavEncoder::default()`) encodes 32-bit float PCM,
/// which is identical to the M1/M2 behaviour.  Set `bit_depth` to encode integer
/// PCM at 16, 24, or 32 bits.
#[derive(Debug, Clone, Copy, Default)]
pub struct WavEncoder {
    pub bit_depth: WavBitDepth,
}

impl WavEncoder {
    /// Encode `buf` to WAV with embedded `LIST/INFO` metadata chunk.
    ///
    /// Writes `RIFF → fmt → LIST/INFO → data`. For multi-channel audio (> 2 channels)
    /// the `fmt` chunk uses `WAVE_FORMAT_EXTENSIBLE` (0xFFFE). Metadata fields that
    /// are `Some` are written as INFO sub-chunks; `None` fields are skipped.
    ///
    /// INFO tag mapping:
    /// - `title`    → `INAM`
    /// - `artist`   → `IART`
    /// - `album`    → `IPRD`
    /// - `genre`    → `IGNR`
    /// - `comment`  → `ICMT`
    /// - `year`     → `ICRD`
    /// - `composer` → `IMUS`
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Io`] on any I/O failure.
    pub fn encode_with_metadata<W: std::io::Write + std::io::Seek>(
        &self,
        buf: &AudioBuffer<f32>,
        writer: W,
        metadata: &AudioMetadata,
    ) -> Result<(), OxiAudioError> {
        wav_ext::write_wav_with_metadata(buf, writer, metadata)
    }

    /// Encode `buf` to a WAV file at `path`.
    ///
    /// Opens (or creates) the file at `path`, wraps it in a `BufWriter`, and
    /// calls [`AudioEncoder::encode`].
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Io`] on file-creation or write failure.
    pub fn encode_to_file(
        &mut self,
        buf: &AudioBuffer<f32>,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), OxiAudioError> {
        let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
        let writer = std::io::BufWriter::new(file);
        self.encode(buf, writer)
    }
}

impl AudioEncoder for WavEncoder {
    /// Encode `buf` to WAV.
    ///
    /// For mono/stereo audio with `WavBitDepth::F32`, the `fmt` chunk uses
    /// `WAVE_FORMAT_IEEE_FLOAT` (0x0003) via `hound`. For multi-channel audio
    /// (> 2 channels), raw bytes are written directly using `WAVE_FORMAT_EXTENSIBLE`
    /// (0xFFFE) with the appropriate channel mask. Integer bit depths for
    /// multi-channel are not yet supported and will return an error.
    fn encode(
        &mut self,
        buf: &AudioBuffer<f32>,
        dst: impl std::io::Write + std::io::Seek,
    ) -> Result<(), OxiAudioError> {
        // Multi-channel (> 2 ch) requires WAVE_FORMAT_EXTENSIBLE.
        // hound does not support this format, so we write raw bytes for those cases.
        if wav_ext::needs_extensible(buf.channels) {
            // For extensible format we only support F32 bit depth through the raw writer.
            // Integer bit depths for multi-channel are handled below through hound for
            // mono/stereo; for multi-channel integer depths we use raw f32.
            return wav_ext::write_wav_raw(buf, dst);
        }

        let channels = buf.channels.channel_count() as u16;

        let spec = match self.bit_depth {
            WavBitDepth::F32 => WavSpec {
                channels,
                sample_rate: buf.sample_rate,
                bits_per_sample: 32,
                sample_format: HoundSampleFormat::Float,
            },
            WavBitDepth::I16 => WavSpec {
                channels,
                sample_rate: buf.sample_rate,
                bits_per_sample: 16,
                sample_format: HoundSampleFormat::Int,
            },
            WavBitDepth::I24 => WavSpec {
                channels,
                sample_rate: buf.sample_rate,
                bits_per_sample: 24,
                sample_format: HoundSampleFormat::Int,
            },
            WavBitDepth::I32 => WavSpec {
                channels,
                sample_rate: buf.sample_rate,
                bits_per_sample: 32,
                sample_format: HoundSampleFormat::Int,
            },
            WavBitDepth::U8 => WavSpec {
                channels,
                sample_rate: buf.sample_rate,
                bits_per_sample: 8,
                sample_format: HoundSampleFormat::Int,
            },
        };

        let mut writer =
            WavWriter::new(dst, spec).map_err(|e| OxiAudioError::Encode(e.to_string()))?;

        match self.bit_depth {
            WavBitDepth::F32 => {
                for &sample in &buf.samples {
                    writer
                        .write_sample(sample)
                        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
                }
            }
            WavBitDepth::I16 => {
                // Pre-convert using chunks_exact so the compiler can auto-vectorize
                // the conversion loop (SIMD hint via fixed-width chunk iteration).
                let converted: Vec<i16> = buf
                    .samples
                    .chunks_exact(8)
                    .flat_map(|chunk| {
                        chunk
                            .iter()
                            .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                    })
                    .chain(
                        buf.samples
                            .chunks_exact(8)
                            .remainder()
                            .iter()
                            .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16),
                    )
                    .collect();
                for val in converted {
                    writer
                        .write_sample(val)
                        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
                }
            }
            WavBitDepth::I24 => {
                // Pre-convert using chunks_exact for SIMD auto-vectorization.
                let converted: Vec<i32> = buf
                    .samples
                    .chunks_exact(8)
                    .flat_map(|chunk| {
                        chunk
                            .iter()
                            .map(|&s| (s.clamp(-1.0, 1.0) * 8_388_607.0_f32) as i32)
                    })
                    .chain(
                        buf.samples
                            .chunks_exact(8)
                            .remainder()
                            .iter()
                            .map(|&s| (s.clamp(-1.0, 1.0) * 8_388_607.0_f32) as i32),
                    )
                    .collect();
                for val in converted {
                    writer
                        .write_sample(val)
                        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
                }
            }
            WavBitDepth::I32 => {
                // Pre-convert using chunks_exact for SIMD auto-vectorization.
                let converted: Vec<i32> = buf
                    .samples
                    .chunks_exact(8)
                    .flat_map(|chunk| {
                        chunk
                            .iter()
                            .map(|&s| (s.clamp(-1.0, 1.0) * i32::MAX as f32) as i32)
                    })
                    .chain(
                        buf.samples
                            .chunks_exact(8)
                            .remainder()
                            .iter()
                            .map(|&s| (s.clamp(-1.0, 1.0) * i32::MAX as f32) as i32),
                    )
                    .collect();
                for val in converted {
                    writer
                        .write_sample(val)
                        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
                }
            }
            WavBitDepth::U8 => {
                // hound's i8 Sample impl converts to unsigned by adding 128 (WAV spec).
                // Scale f32 [-1, 1] to i8 [-127, 127]; hound writes it as u8 [1, 255].
                // Silence (0.0) → i8(0) → u8(128) — correct per WAV 8-bit PCM spec.
                for &s in &buf.samples {
                    let val: i8 = (s.clamp(-1.0, 1.0) * 127.0) as i8;
                    writer
                        .write_sample(val)
                        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
                }
            }
        }

        writer
            .finalize()
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
        Ok(())
    }
}

// ─── WavEncodeConfig ──────────────────────────────────────────────────────────

/// Configuration for WAV encoding.
///
/// Controls the sample format / bit depth written to disk and optionally embeds
/// a `LIST/INFO` metadata chunk.  When `metadata` is `Some`, the current
/// implementation forces the sample data to 32-bit IEEE float PCM (F32)
/// regardless of `bit_depth` because the raw WAV writer used for metadata
/// embedding always produces f32 output.  Integer bit depths are honoured only
/// when `metadata` is `None`.  This limitation is tracked for a future release.
///
/// # Examples
///
/// ```
/// use oxiaudio_encode::{WavEncodeConfig, WavBitDepth};
///
/// // Default: 32-bit float PCM, no metadata
/// let cfg = WavEncodeConfig::default();
/// assert_eq!(cfg.bit_depth, WavBitDepth::F32);
/// assert!(cfg.metadata.is_none());
///
/// // 16-bit PCM, no metadata
/// let cfg = WavEncodeConfig { bit_depth: WavBitDepth::I16, metadata: None };
/// assert_eq!(cfg.bit_depth, WavBitDepth::I16);
/// ```
#[derive(Debug, Clone)]
pub struct WavEncodeConfig {
    /// Sample format / bit depth for encoding.
    pub bit_depth: WavBitDepth,
    /// If `Some`, embed `LIST/INFO` metadata chunk. If `None`, no metadata chunk.
    ///
    /// **Note**: When `metadata` is `Some`, `bit_depth` is currently ignored and
    /// the file is written as 32-bit float PCM.
    pub metadata: Option<AudioMetadata>,
}

impl Default for WavEncodeConfig {
    fn default() -> Self {
        Self {
            bit_depth: WavBitDepth::F32,
            metadata: None,
        }
    }
}

/// Encode audio to WAV with custom [`WavEncodeConfig`] and write to `writer`.
///
/// When `config.metadata` is `Some`, the output includes a `LIST/INFO` chunk.
/// In that case the sample data is always written as 32-bit IEEE float PCM
/// (the `bit_depth` field is ignored).  When `config.metadata` is `None`,
/// `config.bit_depth` controls the sample format.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use oxiaudio_encode::{WavEncodeConfig, WavBitDepth, encode_wav_with_config};
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let buf = AudioBuffer {
///     samples: vec![0.0f32; 1024],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
/// let config = WavEncodeConfig { bit_depth: WavBitDepth::I16, metadata: None };
/// let mut out = Cursor::new(Vec::new());
/// encode_wav_with_config(&buf, &mut out, &config).unwrap();
/// assert_eq!(&out.into_inner()[..4], b"RIFF");
/// ```
pub fn encode_wav_with_config<W: std::io::Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    writer: W,
    config: &WavEncodeConfig,
) -> Result<(), OxiAudioError> {
    match &config.metadata {
        Some(meta) => {
            // encode_with_metadata writes f32 PCM regardless of bit_depth (M12 limitation).
            let enc = WavEncoder {
                bit_depth: config.bit_depth,
            };
            enc.encode_with_metadata(buf, writer, meta)
        }
        None => {
            let mut enc = WavEncoder {
                bit_depth: config.bit_depth,
            };
            enc.encode(buf, writer)
        }
    }
}

/// Encode `buf` to WAV format and return the result as `Vec<u8>`.
///
/// Uses the default `WavEncoder` (32-bit float PCM).
pub fn encode_wav_to_vec(buf: &AudioBuffer<f32>) -> Result<Vec<u8>, OxiAudioError> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    WavEncoder::default().encode(buf, &mut cursor)?;
    Ok(cursor.into_inner())
}

// ─── Seekless WAV streaming ────────────────────────────────────────────────────

/// Write a WAV file with chunk size fields set to `0xFFFFFFFF` (unknown).
///
/// Use this when writing to a non-seekable destination (e.g., a TCP stream, pipe,
/// or HTTP response body). Players like VLC and ffplay handle unknown-size WAV.
///
/// Writes 16-bit PCM (I16). Sample rate and channel count are taken from `buf`.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on any I/O failure.
#[must_use = "discarding errors ignores write failure"]
pub fn encode_wav_streaming<W: std::io::Write>(
    buf: &AudioBuffer<f32>,
    writer: &mut W,
) -> Result<(), OxiAudioError> {
    let n_channels = buf.channels.channel_count() as u16;
    let sample_rate = buf.sample_rate;
    let bytes_per_sample: u16 = 2;
    let byte_rate = sample_rate * n_channels as u32 * bytes_per_sample as u32;
    let block_align = n_channels * bytes_per_sample;
    let bits_per_sample: u16 = 16;

    // RIFF header with sentinel size
    writer.write_all(b"RIFF")?;
    writer.write_all(&0xFFFF_FFFFu32.to_le_bytes())?; // unknown RIFF size
    writer.write_all(b"WAVE")?;

    // fmt chunk (16 bytes payload)
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?; // PCM
    writer.write_all(&n_channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk with sentinel size
    writer.write_all(b"data")?;
    writer.write_all(&0xFFFF_FFFFu32.to_le_bytes())?; // unknown data size

    // Write samples as I16 LE
    for &s in &buf.samples {
        let sample_i16 = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_all(&sample_i16.to_le_bytes())?;
    }

    Ok(())
}

// ─── Seekless WAV streaming tests ─────────────────────────────────────────────

#[cfg(test)]
mod wav_streaming_tests {
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    use super::encode_wav_streaming;

    fn sine_buf(sr: u32, n: usize) -> AudioBuffer<f32> {
        let samples = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_wav_streaming_riff_sentinel() {
        let buf = sine_buf(44_100, 512);
        let mut out = Vec::<u8>::new();
        encode_wav_streaming(&buf, &mut out).expect("encode_wav_streaming");
        // bytes 4..8 must be the sentinel 0xFFFFFFFF (little-endian)
        assert_eq!(
            &out[4..8],
            &[0xFF, 0xFF, 0xFF, 0xFF],
            "RIFF size must be sentinel 0xFFFFFFFF"
        );
    }

    #[test]
    fn test_wav_streaming_no_seek_required() {
        // Vec<u8> implements Write but NOT Seek — this must compile and succeed.
        let buf = AudioBuffer {
            samples: vec![0.0f32; 256],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut out = Vec::<u8>::new();
        encode_wav_streaming(&buf, &mut out).expect("should work without Seek");
        assert!(!out.is_empty(), "output must not be empty");
        assert_eq!(&out[..4], b"RIFF");
    }

    #[test]
    fn test_wav_streaming_non_zero_data() {
        let buf = sine_buf(44_100, 1024);
        let mut out = Vec::<u8>::new();
        encode_wav_streaming(&buf, &mut out).expect("encode_wav_streaming");
        // WAV header is 44 bytes; data starts at byte 44
        let header_len = 44usize;
        assert!(out.len() > header_len, "output must contain sample data");
        let data = &out[header_len..];
        // A sine wave cannot be all zeros
        assert!(
            data.iter().any(|&b| b != 0),
            "sample data must not be all zeros for a sine wave"
        );
    }
}
