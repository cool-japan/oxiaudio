#![forbid(unsafe_code)]

mod aac;
pub mod aac_m4a;
mod aiff;
mod apev2;
mod au;
mod flac_meta;
mod flac_picture;
mod flac_streaming;
mod id3;
pub mod ogg;
pub mod opus_celt;
pub mod opus_celt_tables;
pub mod opus_encoder;
pub mod opus_hybrid;
pub mod opus_hybrid_conform;
pub mod opus_mdct;
pub mod opus_pvq;
pub mod opus_range;
pub mod opus_silk;
pub mod opus_silk_conform;
pub mod vorbis;
mod wav_cue;
mod wav_ext;

pub use aac::{
    encode_aac, encode_aac_file, encode_aac_mode, encode_aac_mode_file, encode_aac_pns,
    encode_aac_tns, AacBitrateMode,
};
pub use aac_m4a::{encode_m4a, encode_m4a_file};
pub use aiff::{
    encode_aiff_with_metadata, write_aiff_with_chunks, write_aiffc, write_aiffc_file, AiffBitDepth,
    AiffStreamEncoder, AiffcCodec,
};
pub use apev2::{write_apev2, ApeItem};
pub use au::{encode_au, encode_au_file, AuEncoding};
pub use flac_meta::{
    encode_flac_with_md5, encode_flac_with_md5_file, encode_flac_with_metadata,
    encode_flac_with_seektable, encode_flac_with_seektable_file, inject_flac_md5, FlacMetaConfig,
};
pub use flac_picture::{
    encode_flac_with_album_art, encode_flac_with_album_art_file,
    encode_flac_with_metadata_and_picture, encode_flac_with_picture, encode_flac_with_picture_file,
    FlacPicture,
};
pub use flac_streaming::FlacStreamingEncoder;
pub use id3::Id3v24Tag;
pub use ogg::{ogg_crc32, write_ogg_page, write_vorbis_comment_packet, OggStream};
pub use opus_celt::encode_celt_frame_conformant;
pub use opus_encoder::{encode_opus, encode_opus_file, OpusEncodeConfig, OpusStreamEncoder};
pub use opus_hybrid::{encode_hybrid_frame, hybrid_toc, should_use_hybrid};
pub use opus_hybrid_conform::encode_hybrid_frame_conformant;
pub use opus_silk::{analyze_silk_frame, encode_silk_frame, SilkBandwidth, SilkLpcFrame};
pub use opus_silk_conform::encode_silk_frame_conformant;
pub use vorbis::{
    encode_vorbis, encode_vorbis_file, encode_vorbis_quality_file, encode_vorbis_with_quality,
    VorbisQuality,
};
pub use wav_cue::{encode_wav_with_cues, encode_wav_with_cues_file, CuePoint};
pub use wav_ext::{
    encode_wav_rf64, encode_wav_rf64_file, encode_wav_with_progress, EncodeProgressFn,
};

pub mod flac_core;
pub mod wav_core;

// Re-exports from wav_core
pub use wav_core::{
    apply_tpdf_dither, encode_wav_streaming, encode_wav_to_vec, encode_wav_with_config,
    WavBitDepth, WavEncodeConfig, WavEncoder,
};

// Re-exports from flac_core
pub use flac_core::{
    analyze_loudness_gain, encode_flac, encode_flac_parallel, encode_flac_to_vec,
    encode_flac_with_config, encode_flac_with_level, encode_flac_with_progress,
    encode_normalized_wav, encode_normalized_wav_file, FlacBitDepth, FlacConfig, FlacEncoder,
    LoudnessTarget,
};

use oxiaudio_core::{
    AudioBuffer, AudioEncoder, AudioSink, ChannelLayout, OxiAudioError, SampleFormat,
};

#[cfg(feature = "mp3")]
pub use oxiaudio_encode_mp3_lame::lame::{LameMode, LameMp3Encoder, VbrPreset};

// ─── AIFF writer ──────────────────────────────────────────────────────────────

/// Convert an `f64` sample rate to Apple's 80-bit IEEE 754 Extended (SANE) format.
///
/// The layout is: 1 sign bit, 15 exponent bits (biased by 16383), 64 explicit
/// mantissa bits (no implicit leading 1 — AIFF always stores the integer bit explicitly).
fn f64_to_extended(f: f64) -> [u8; 10] {
    let mut bytes = [0u8; 10];
    if f == 0.0 {
        return bytes;
    }
    let bits = f.to_bits();
    let sign = (bits >> 63) as u8;
    let exp = ((bits >> 52) & 0x7FF) as i32 - 1023;
    // Reconstruct the 53-bit mantissa with the implicit leading 1.
    let mantissa = (bits & 0x000F_FFFF_FFFF_FFFF) | 0x0010_0000_0000_0000;
    let ext_exp = (exp + 16383) as u16;
    bytes[0] = (sign << 7) | ((ext_exp >> 8) as u8);
    bytes[1] = ext_exp as u8;
    // Shift the 53-bit mantissa up to fill the 64-bit mantissa field.
    // The implicit-1 bit moves to bit 63 of the 64-bit field.
    let mantissa_shifted = mantissa << 11;
    bytes[2..10].copy_from_slice(&mantissa_shifted.to_be_bytes());
    bytes
}

/// Write `buf` (f32 `AudioBuffer`) as AIFF (16-bit signed big-endian PCM) to `writer`.
///
/// AIFF file structure:
/// ```text
/// FORM chunk (container)
///   COMM chunk  — channels, numSampleFrames, sampleSize, sampleRate (80-bit extended)
///   SSND chunk  — offset=0, blockSize=0, then interleaved big-endian i16 PCM data
/// ```
pub fn write_aiff<W: std::io::Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    writer: &mut W,
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count() as u16;
    let num_frames = (buf.samples.len() / channels as usize) as u32;
    let sample_rate_ext = f64_to_extended(buf.sample_rate as f64);

    // Convert f32 samples → i16 big-endian PCM bytes.
    let pcm_bytes: Vec<u8> = buf
        .samples
        .iter()
        .flat_map(|&s| {
            let v: i16 = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            v.to_be_bytes()
        })
        .collect();

    // COMM chunk: 4+4+2+4+2+10 = 26 bytes total (ck_id + ck_size + payload)
    // Payload: numChannels(2) + numSampleFrames(4) + sampleSize(2) + sampleRate(10) = 18 bytes
    const COMM_PAYLOAD: u32 = 18;

    // SSND chunk: 4+4+4+4 + pcm_bytes (ck_id + ck_size + offset + blockSize + data)
    // Payload: offset(4) + blockSize(4) + data = 8 + pcm_bytes.len()
    let ssnd_payload = 8u32 + pcm_bytes.len() as u32;

    // FORM payload = "AIFF"(4) + COMM(8 + 18) + SSND(8 + ssnd_payload - 8)
    // = 4 + 26 + 8 + ssnd_payload
    let form_payload = 4u32 + 8 + COMM_PAYLOAD + 8 + ssnd_payload;

    // ── FORM header ──
    writer.write_all(b"FORM").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&form_payload.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer.write_all(b"AIFF").map_err(OxiAudioError::Io)?;

    // ── COMM chunk ──
    writer.write_all(b"COMM").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&COMM_PAYLOAD.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&channels.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&num_frames.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    // sampleSize: 16 bits
    writer
        .write_all(&16u16.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&sample_rate_ext)
        .map_err(OxiAudioError::Io)?;

    // ── SSND chunk ──
    writer.write_all(b"SSND").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&ssnd_payload.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    // offset = 0
    writer
        .write_all(&0u32.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    // blockSize = 0
    writer
        .write_all(&0u32.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    // PCM data
    writer.write_all(&pcm_bytes).map_err(OxiAudioError::Io)?;

    Ok(())
}

/// Write `buf` as AIFF (16-bit signed big-endian PCM) to the file at `path`.
pub fn write_aiff_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
) -> Result<(), OxiAudioError> {
    let mut file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    write_aiff(buf, &mut file)
}

// ─── StreamEncoder trait ───────────────────────────────────────────────────────

/// Trait for streaming (chunk-by-chunk) audio encoders.
///
/// Implementations accumulate or immediately encode each chunk passed to
/// [`Self::write_chunk`], then flush all pending data when [`Self::finalize`] is
/// called.
pub trait StreamEncoder: Send {
    /// Write the next chunk of audio data.
    fn write_chunk(&mut self, chunk: &AudioBuffer<f32>) -> Result<(), OxiAudioError>;
    /// Finalize and flush all pending data.
    ///
    /// Must be called before dropping; the boxed form is required by the trait to
    /// allow object-safe consumption.
    fn finalize(self: Box<Self>) -> Result<(), OxiAudioError>;
}

// ─── WavStreamEncoder ─────────────────────────────────────────────────────────

/// Streaming WAV encoder: encodes [`AudioBuffer<f32>`] chunks without buffering the full file.
///
/// `new` writes a RIFF header placeholder; `encode_chunk` appends samples; `finalize` seeks
/// back and patches the true data sizes.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use oxiaudio_encode::{WavStreamEncoder, WavBitDepth};
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let mut enc = WavStreamEncoder::new(
///     Cursor::new(Vec::new()),
///     44_100,
///     ChannelLayout::Mono,
///     WavBitDepth::F32,
/// ).unwrap();
/// let buf = AudioBuffer {
///     samples: vec![0.0f32; 512],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
/// enc.encode_chunk(&buf).unwrap();
/// enc.finalize().unwrap();
/// ```
pub struct WavStreamEncoder<W: std::io::Write + std::io::Seek> {
    /// Hound writer wrapping a `BufWriter<W>` to batch small per-sample writes.
    writer: Option<hound::WavWriter<std::io::BufWriter<W>>>,
    bit_depth: WavBitDepth,
    frames_written: u64,
}

impl<W: std::io::Write + std::io::Seek> WavStreamEncoder<W> {
    /// Create a new `WavStreamEncoder` writing a RIFF header placeholder to `dst`.
    ///
    /// `dst` is wrapped in [`std::io::BufWriter`] to batch small per-sample writes.
    /// Header sizes are patched when [`Self::finalize`] is called.
    pub fn new(
        dst: W,
        sample_rate: u32,
        channels: ChannelLayout,
        bit_depth: WavBitDepth,
    ) -> Result<Self, OxiAudioError> {
        let n_ch = channels.channel_count() as u16;
        let (sf, bps) = match bit_depth {
            WavBitDepth::F32 => (hound::SampleFormat::Float, 32u16),
            WavBitDepth::I16 => (hound::SampleFormat::Int, 16u16),
            WavBitDepth::I24 => (hound::SampleFormat::Int, 24u16),
            WavBitDepth::I32 => (hound::SampleFormat::Int, 32u16),
            WavBitDepth::U8 => (hound::SampleFormat::Int, 8u16),
        };
        let spec = hound::WavSpec {
            channels: n_ch,
            sample_rate,
            bits_per_sample: bps,
            sample_format: sf,
        };
        let buf_writer = std::io::BufWriter::new(dst);
        let writer = hound::WavWriter::new(buf_writer, spec)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
        Ok(Self {
            writer: Some(writer),
            bit_depth,
            frames_written: 0,
        })
    }

    /// Encode one chunk of audio samples, appending them to the stream.
    ///
    /// Returns `Err` if the encoder has already been finalized.
    pub fn encode_chunk(&mut self, buf: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| OxiAudioError::Encode("WavStreamEncoder already finalized".into()))?;
        match self.bit_depth {
            WavBitDepth::F32 => {
                for &s in &buf.samples {
                    writer
                        .write_sample(s)
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
                for v in converted {
                    writer
                        .write_sample(v)
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
                for v in converted {
                    writer
                        .write_sample(v)
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
                for v in converted {
                    writer
                        .write_sample(v)
                        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
                }
            }
            WavBitDepth::U8 => {
                for &s in &buf.samples {
                    let v: i8 = (s.clamp(-1.0, 1.0) * 127.0) as i8;
                    writer
                        .write_sample(v)
                        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
                }
            }
        }
        let n_ch = buf.channels.channel_count();
        self.frames_written += (buf.samples.len() / n_ch) as u64;
        Ok(())
    }

    /// Returns the total number of PCM frames written so far.
    pub fn frames_written(&self) -> u64 {
        self.frames_written
    }

    /// Finalize the stream: seeks back to the RIFF header and patches the true data sizes.
    ///
    /// Consumes the encoder. After this call, the destination contains a valid WAV file.
    pub fn finalize(mut self) -> Result<(), OxiAudioError> {
        self.writer
            .take()
            .ok_or_else(|| OxiAudioError::Encode("WavStreamEncoder already finalized".into()))?
            .finalize()
            .map_err(|e| OxiAudioError::Encode(e.to_string()))
    }
}

impl<W: std::io::Write + std::io::Seek> AudioSink for WavStreamEncoder<W> {
    fn write_chunk(&mut self, buf: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
        self.encode_chunk(buf)
    }
}

impl<W: std::io::Write + std::io::Seek + Send> StreamEncoder for WavStreamEncoder<W> {
    fn write_chunk(&mut self, chunk: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
        self.encode_chunk(chunk)
    }

    fn finalize(self: Box<Self>) -> Result<(), OxiAudioError> {
        (*self).finalize()
    }
}

// ─── FlacStreamEncoder ────────────────────────────────────────────────────────

/// Streaming FLAC encoder with a buffered approach.
///
/// # Note
///
/// flacenc's API requires all samples to be available before encoding begins. This encoder
/// accumulates all chunks in memory and encodes them on [`FlacStreamEncoder::finalize`].
/// For very large files, prefer [`FlacEncoder::encode`] directly if you can hold the full buffer.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use oxiaudio_encode::FlacStreamEncoder;
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let mut enc = FlacStreamEncoder::new(
///     Cursor::new(Vec::new()),
///     44_100,
///     ChannelLayout::Mono,
///     5,
/// );
/// let buf = AudioBuffer {
///     samples: vec![0.0f32; 512],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
/// enc.encode_chunk(&buf).unwrap();
/// enc.finalize().unwrap();
/// ```
pub struct FlacStreamEncoder<W: std::io::Write + std::io::Seek> {
    accumulated: Vec<f32>,
    sample_rate: u32,
    channels: ChannelLayout,
    compression_level: u8,
    dst: Option<W>,
}

impl<W: std::io::Write + std::io::Seek> FlacStreamEncoder<W> {
    /// Create a new `FlacStreamEncoder`.
    ///
    /// `compression_level` is clamped to `[0, 8]`; level 5 is the default.
    pub fn new(dst: W, sample_rate: u32, channels: ChannelLayout, compression_level: u8) -> Self {
        Self {
            accumulated: Vec::new(),
            sample_rate,
            channels,
            compression_level: compression_level.min(8),
            dst: Some(dst),
        }
    }

    /// Accumulate a chunk of audio samples for later encoding.
    ///
    /// Returns `Err` if the encoder has already been finalized.
    pub fn encode_chunk(&mut self, buf: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
        if self.dst.is_none() {
            return Err(OxiAudioError::Encode(
                "FlacStreamEncoder already finalized".into(),
            ));
        }
        self.accumulated.extend_from_slice(&buf.samples);
        Ok(())
    }

    /// Finalize: encodes all accumulated samples via [`FlacEncoder`] and writes to `dst`.
    ///
    /// Consumes the encoder. After this call, the destination contains a valid FLAC file.
    pub fn finalize(mut self) -> Result<(), OxiAudioError> {
        let dst = self
            .dst
            .take()
            .ok_or_else(|| OxiAudioError::Encode("FlacStreamEncoder already finalized".into()))?;
        let accumulated_buf = AudioBuffer {
            samples: self.accumulated,
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: SampleFormat::F32,
        };
        FlacEncoder::new(self.compression_level).encode(&accumulated_buf, dst)
    }
}

impl<W: std::io::Write + std::io::Seek> AudioSink for FlacStreamEncoder<W> {
    fn write_chunk(&mut self, buf: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
        self.encode_chunk(buf)
    }
}

impl<W: std::io::Write + std::io::Seek + Send> StreamEncoder for FlacStreamEncoder<W> {
    fn write_chunk(&mut self, chunk: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
        self.encode_chunk(chunk)
    }

    fn finalize(self: Box<Self>) -> Result<(), OxiAudioError> {
        (*self).finalize()
    }
}

// ─── EncoderConfig builder ────────────────────────────────────────────────────

/// Builder-style configuration for audio encoding.
///
/// Provides a unified API for WAV and FLAC encoding with optional pre-processing
/// (TPDF dithering, peak normalization).
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use oxiaudio_encode::{EncoderConfig, WavBitDepth};
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let buf = AudioBuffer {
///     samples: vec![0.0f32; 1024],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
///
/// let mut out = Cursor::new(Vec::new());
/// EncoderConfig::new(44_100, 1)
///     .with_bit_depth(WavBitDepth::I16)
///     .with_dither(true)
///     .encode_wav(&buf, &mut out)
///     .unwrap();
/// assert!(out.into_inner().len() > 44);
/// ```
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: WavBitDepth,
    pub apply_dither: bool,
    pub flac_compression: u8,
    pub normalize_before_encode: bool,
}

impl EncoderConfig {
    /// Create a new `EncoderConfig` with sensible defaults.
    ///
    /// Default: 16-bit PCM WAV, no dithering, FLAC compression level 5, no normalization.
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
            bit_depth: WavBitDepth::I16,
            apply_dither: false,
            flac_compression: 5,
            normalize_before_encode: false,
        }
    }

    /// Set the WAV bit depth.
    pub fn with_bit_depth(mut self, bit_depth: WavBitDepth) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Enable or disable TPDF dithering before integer quantization.
    ///
    /// Has no effect when `bit_depth` is `F32` (no quantization step).
    pub fn with_dither(mut self, apply: bool) -> Self {
        self.apply_dither = apply;
        self
    }

    /// Set the FLAC compression level (clamped to `0..=8`).
    pub fn with_flac_compression(mut self, level: u8) -> Self {
        self.flac_compression = level.min(8);
        self
    }

    /// Enable peak normalization before encoding (scales all samples so the peak = 1.0).
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize_before_encode = normalize;
        self
    }

    /// Encode `buf` to WAV using this configuration and write to `writer`.
    pub fn encode_wav<W: std::io::Write + std::io::Seek>(
        &self,
        buf: &AudioBuffer<f32>,
        writer: &mut W,
    ) -> Result<(), OxiAudioError> {
        let samples = self.prepare_samples(buf);
        let prepared = AudioBuffer {
            samples,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        };
        WavEncoder {
            bit_depth: self.bit_depth,
        }
        .encode(&prepared, writer)
    }

    /// Encode `buf` to FLAC using this configuration and write to `writer`.
    pub fn encode_flac<W: std::io::Write + std::io::Seek>(
        &self,
        buf: &AudioBuffer<f32>,
        writer: &mut W,
    ) -> Result<(), OxiAudioError> {
        let samples = self.prepare_samples(buf);
        let prepared = AudioBuffer {
            samples,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        };
        FlacEncoder::new(self.flac_compression).encode(&prepared, writer)
    }

    /// Apply pre-processing (normalize, dither) and return the processed sample vector.
    fn prepare_samples(&self, buf: &AudioBuffer<f32>) -> Vec<f32> {
        let mut samples = buf.samples.clone();

        if self.normalize_before_encode {
            let peak = samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
            if peak > 0.0 {
                let scale = 1.0 / peak;
                for s in &mut samples {
                    *s *= scale;
                }
            }
        }

        // Apply TPDF dither only for integer quantization formats.
        if self.apply_dither && !matches!(self.bit_depth, WavBitDepth::F32) {
            let noise_bits = match self.bit_depth {
                WavBitDepth::U8 => 8u8,
                WavBitDepth::I16 => 16,
                WavBitDepth::I24 => 24,
                WavBitDepth::I32 => 32,
                WavBitDepth::F32 => 32, // unreachable due to outer guard
            };
            apply_tpdf_dither(&mut samples, noise_bits);
        }

        samples
    }
}
