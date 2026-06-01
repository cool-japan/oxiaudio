//! FLAC encoding: `FlacEncoder`, configuration types, convenience functions,
//! parallel encoding, and two-pass loudness normalization.

use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::error::Verify;
use flacenc::source::MemSource;
use oxiaudio_core::{AudioBuffer, AudioEncoder, OxiAudioError};

use crate::wav_ext;
use crate::FlacStreamEncoder;

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Snap an arbitrary bit-depth request to the nearest FLAC-supported value.
///
/// The `flacenc 0.5.1` backend caps `bits_per_sample` at 24, so requests above 24
/// are clamped down to 24 rather than erroring.
pub(crate) fn clamp_flac_bits(bits: u8) -> u8 {
    match bits {
        0..=17 => 16,
        18..=21 => 20,
        _ => 24,
    }
}

/// Full-scale positive integer value for a signed PCM depth (`2^(bits-1) - 1`).
pub(crate) fn flac_full_scale(bits: u8) -> f32 {
    ((1i64 << (bits as i64 - 1)) - 1) as f32
}

/// Maps a compression level in `0..=8` to an appropriate `block_size` for flacenc.
///
/// The mapping is designed so that level 5 produces 4096, matching flacenc's
/// `config::Encoder` default.  Lower levels use smaller blocks (lower latency,
/// less compression); higher levels use larger blocks (better compression).
#[inline]
pub(crate) fn block_size_for_level(level: u8) -> usize {
    // Lookup table: index = compression_level (0..=8)
    const SIZES: [usize; 9] = [256, 512, 512, 1024, 2048, 4096, 4096, 8192, 8192];
    let idx = level.min(8) as usize;
    SIZES[idx]
}

// ─── FlacBitDepth ─────────────────────────────────────────────────────────────

/// Bit depth for FLAC encoding.
///
/// The `flacenc` backend supports up to 24-bit PCM. Both variants map directly
/// to `FlacEncoder::with_bits_per_sample`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlacBitDepth {
    /// 16-bit signed integer PCM. Each sample is scaled by `i16::MAX` (32767).
    I16,
    /// 24-bit signed integer PCM. Each sample is scaled by `8_388_607` (2^23 − 1).
    I24,
}

impl FlacBitDepth {
    /// Returns the numeric bit depth (16 or 24).
    #[inline]
    pub fn bits(self) -> u8 {
        match self {
            FlacBitDepth::I16 => 16,
            FlacBitDepth::I24 => 24,
        }
    }
}

// ─── FlacConfig ───────────────────────────────────────────────────────────────

/// Configuration for FLAC encoding.
///
/// # Examples
///
/// ```
/// use oxiaudio_encode::{FlacConfig, FlacBitDepth};
///
/// // Default: level 5, 16-bit
/// let cfg = FlacConfig::default();
/// assert_eq!(cfg.compression, 5);
/// assert_eq!(cfg.bit_depth, FlacBitDepth::I16);
///
/// // 24-bit at maximum compression
/// let cfg = FlacConfig { compression: 8, bit_depth: FlacBitDepth::I24 };
/// assert_eq!(cfg.bit_depth.bits(), 24);
/// ```
#[derive(Debug, Clone)]
pub struct FlacConfig {
    /// Compression level 0–8 (0 = fastest, 8 = best compression). Default: 5.
    pub compression: u8,
    /// Bit depth for encoding. Default: [`FlacBitDepth::I16`].
    pub bit_depth: FlacBitDepth,
}

impl Default for FlacConfig {
    fn default() -> Self {
        Self {
            compression: 5,
            bit_depth: FlacBitDepth::I16,
        }
    }
}

// ─── FlacEncoder ──────────────────────────────────────────────────────────────

/// FLAC encoder backed by `flacenc 0.5.1`.
///
/// `compression_level` (0–8) is mapped to flacenc's `block_size` as follows:
///   - level 0 → 256 (fastest; largest blocks relative to overhead)
///   - level 5 → 4096 (default; matches `flacenc::config::Encoder` default)
///   - level 8 → 8192 (best compression; largest blocks for better LPC fit)
///
/// Mapping: `block_size = 256 * 2^(level / 2)`, clamped to `[32, 32767]`.
/// flacenc 0.5.1 does not expose a single "level" knob equivalent to the
/// reference FLAC encoder; block_size is the primary compression trade-off.
///
/// `bits_per_sample` selects the PCM resolution written to the FLAC stream. The
/// `flacenc 0.5.1` backend supports up to 24-bit PCM (its verifier rejects depths
/// above 25 bits), so the supported depths are 16, 20, and 24; the `f32` input is
/// scaled to the chosen depth's full-scale integer range. The default is 24-bit
/// (preserving the M2/M3 behaviour).
pub struct FlacEncoder {
    pub compression_level: u8,
    /// PCM bit depth written to the FLAC stream (16, 20, or 24). Other values are
    /// clamped to the nearest supported depth (anything above 24 maps to 24).
    pub bits_per_sample: u8,
}

impl Default for FlacEncoder {
    fn default() -> Self {
        Self {
            compression_level: 5,
            bits_per_sample: 24,
        }
    }
}

impl FlacEncoder {
    /// Create a FLAC encoder with the given compression level (0–8) and the
    /// default 24-bit depth.
    pub fn new(compression_level: u8) -> Self {
        Self {
            compression_level,
            bits_per_sample: 24,
        }
    }

    /// Set the PCM bit depth (16, 20, 24, or 32; clamped to the nearest valid).
    pub fn with_bits_per_sample(mut self, bits: u8) -> Self {
        self.bits_per_sample = clamp_flac_bits(bits);
        self
    }

    /// Encode `buf` to a FLAC file at `path`.
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

impl AudioEncoder for FlacEncoder {
    fn encode(
        &mut self,
        buf: &AudioBuffer<f32>,
        mut dst: impl std::io::Write + std::io::Seek,
    ) -> Result<(), OxiAudioError> {
        let channels = buf.channels.channel_count();

        // Map compression_level (0–8) to block_size.
        // level 5 → 4096 matches flacenc's default (4096 = 256 * 2^(5/2) ≈ 256 * 5.66, rounded).
        // We use a small lookup so the anchor at level 5 = 4096 is exact.
        let block_size = block_size_for_level(self.compression_level);

        let bits = clamp_flac_bits(self.bits_per_sample);
        let scale = flac_full_scale(bits);

        // Convert f32 samples in [-1.0, 1.0] to signed i32 interleaved PCM at the
        // configured bit depth.
        let pcm: Vec<i32> = buf
            .samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * scale) as i32)
            .collect();

        let source =
            MemSource::from_samples(&pcm, channels, bits as usize, buf.sample_rate as usize);

        let mut cfg = flacenc::config::Encoder::default();
        cfg.block_size = block_size;
        let cfg = cfg
            .into_verified()
            .map_err(|(_, e)| OxiAudioError::Encode(e.to_string()))?;

        let mut stream = flacenc::encode_with_fixed_block_size(&cfg, source, block_size)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

        // After encoding, flacenc's update_frame_info sets min_block_size to the last
        // (potentially partial) frame's block_size. This makes min_block_size != max_block_size,
        // which causes symphonia's strict frame header check to fail (it requires BySample
        // sequencing when is_fixed=false, but flacenc emits ByFrame fixed-blocksize frames).
        // Override both to block_size so the StreamInfo signals fixed-blocksize mode correctly.
        stream
            .stream_info_mut()
            .set_block_sizes(block_size, block_size)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

        let mut sink = ByteSink::with_capacity(stream.count_bits());
        stream
            .write(&mut sink)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

        dst.write_all(sink.as_slice()).map_err(OxiAudioError::Io)?;

        Ok(())
    }
}

// ─── Free-function convenience API ────────────────────────────────────────────

/// Encode `buf` to FLAC with the default compression level (5) and write to `writer`.
///
/// Equivalent to `FlacEncoder::default().encode(buf, writer)`.
pub fn encode_flac<W: std::io::Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    writer: &mut W,
) -> Result<(), OxiAudioError> {
    FlacEncoder::default().encode(buf, writer)
}

/// Encode `buf` to FLAC with the specified `compression_level` (0–8) and write to `writer`.
pub fn encode_flac_with_level<W: std::io::Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    writer: &mut W,
    compression_level: u8,
) -> Result<(), OxiAudioError> {
    FlacEncoder::new(compression_level).encode(buf, writer)
}

/// Encode `buf` to FLAC, invoking a progress callback after each chunk is accumulated.
///
/// Calls `progress(frames_done, total_frames)` after every `CHUNK_SIZE` frames.
/// Because [`FlacStreamEncoder`] accumulates all samples before encoding, the callback
/// fires during accumulation; encoding itself occurs in the final `finalize()` call.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on any encoding or I/O failure.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_with_progress<W: std::io::Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    writer: W,
    compression_level: u8,
    progress: wav_ext::EncodeProgressFn<'_>,
) -> Result<(), OxiAudioError> {
    const CHUNK_SIZE: usize = 4096; // frames per progress report

    let n_channels = buf.channels.channel_count();
    let total_frames = buf
        .samples
        .len()
        .checked_div(n_channels.max(1))
        .unwrap_or(0);

    let mut enc = FlacStreamEncoder::new(writer, buf.sample_rate, buf.channels, compression_level);

    let mut frames_accumulated: usize = 0;

    for chunk_samples in buf.samples.chunks(CHUNK_SIZE * n_channels.max(1)) {
        let chunk_frames = chunk_samples
            .len()
            .checked_div(n_channels.max(1))
            .unwrap_or(0);
        let chunk_buf = AudioBuffer {
            samples: chunk_samples.to_vec(),
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        };
        enc.encode_chunk(&chunk_buf)?;
        frames_accumulated = (frames_accumulated + chunk_frames).min(total_frames);
        progress(frames_accumulated, total_frames);
    }

    enc.finalize()?;

    // Ensure the final progress report signals exactly total_frames.
    if frames_accumulated != total_frames {
        progress(total_frames, total_frames);
    }

    Ok(())
}

/// Encode `buf` to FLAC with a custom [`FlacConfig`] and write to `writer`.
///
/// The `f32` samples in `buf` are converted to integer PCM at the configured bit depth:
/// - `I16`: scaled by `i16::MAX` (32 767)
/// - `I24`: scaled by `8_388_607` (2^23 − 1)
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use oxiaudio_encode::{FlacConfig, FlacBitDepth, encode_flac_with_config};
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let buf = AudioBuffer {
///     samples: vec![0.0f32; 4096],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
/// let mut out = Cursor::new(Vec::new());
/// encode_flac_with_config(&buf, &mut out, &FlacConfig::default()).unwrap();
/// assert_eq!(&out.into_inner()[..4], b"fLaC");
/// ```
pub fn encode_flac_with_config<W: std::io::Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    writer: &mut W,
    config: &FlacConfig,
) -> Result<(), OxiAudioError> {
    FlacEncoder::new(config.compression)
        .with_bits_per_sample(config.bit_depth.bits())
        .encode(buf, writer)
}

/// Encode `buf` to FLAC format and return the result as `Vec<u8>`.
pub fn encode_flac_to_vec(buf: &AudioBuffer<f32>) -> Result<Vec<u8>, OxiAudioError> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    FlacEncoder::default().encode(buf, &mut cursor)?;
    Ok(cursor.into_inner())
}

// ─── Parallel FLAC encoding ───────────────────────────────────────────────────

/// Encode an `AudioBuffer<f32>` to FLAC using rayon to parallelise the hot
/// f32 → i32 PCM sample-conversion step, then encode sequentially with flacenc.
///
/// ## Why parallel only for conversion?
///
/// `flacenc 0.5.x` exposes a one-shot API (`encode_with_fixed_block_size`) that
/// requires the full interleaved i32 PCM buffer up front.  FLAC frame encoding
/// itself is inherently sequential (each frame depends on the stream state), so
/// we cannot concatenate independent FLAC streams.  However, the f32 → integer
/// scaling loop is embarrassingly parallel and can dominate for large buffers at
/// high sample counts.
///
/// Rayon's `par_iter` splits the work across available CPU cores.  The resulting
/// i32 samples are passed to the same single-threaded flacenc path as the regular
/// [`FlacEncoder`].
///
/// # Arguments
///
/// * `buf`               – Source audio in f32.
/// * `writer`            – Destination (must implement `Write + Seek` for FLAC STREAMINFO backfill).
/// * `compression_level` – FLAC compression level 0 (fastest) to 8 (best).
/// * `bits_per_sample`   – PCM bit depth: 16, 20, or 24 (values are clamped).
///
/// # Errors
///
/// Returns [`OxiAudioError`] on any encoding or I/O failure.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_parallel<W: std::io::Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    writer: W,
    compression_level: u8,
    bits_per_sample: u8,
) -> Result<(), OxiAudioError> {
    use rayon::prelude::*;

    let clamped_bits = clamp_flac_bits(bits_per_sample);
    let scale = flac_full_scale(clamped_bits);
    let n_channels = buf.channels.channel_count().max(1);

    // ── Step 1: parallel f32 → i32 conversion ────────────────────────────────
    // Each sample is independently scaled + clamped, so this step is
    // embarrassingly parallel across the entire interleaved sample vector.
    let pcm_i32: Vec<i32> = buf
        .samples
        .par_iter()
        .map(|&s| {
            let scaled = s * scale;
            let max = scale - 1.0;
            let clamped = if scaled < -scale {
                -scale
            } else if scaled > max {
                max
            } else {
                scaled
            };
            clamped as i32
        })
        .collect();

    // ── Step 2: sequential FLAC encoding via flacenc ─────────────────────────
    let block_size = block_size_for_level(compression_level);

    let source = MemSource::from_samples(
        &pcm_i32,
        n_channels,
        clamped_bits as usize,
        buf.sample_rate as usize,
    );

    let mut cfg = flacenc::config::Encoder::default();
    cfg.block_size = block_size;
    let cfg = cfg
        .into_verified()
        .map_err(|(_, e)| OxiAudioError::Encode(e.to_string()))?;

    let mut stream = flacenc::encode_with_fixed_block_size(&cfg, source, block_size)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    // Fix min_block_size == max_block_size so symphonia accepts the stream.
    stream
        .stream_info_mut()
        .set_block_sizes(block_size, block_size)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    let mut sink = ByteSink::with_capacity(stream.count_bits());
    stream
        .write(&mut sink)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    let mut writer = writer;
    writer.write_all(sink.as_slice()).map_err(OxiAudioError::Io)
}

// ─── Two-pass loudness normalization ─────────────────────────────────────────
//
// oxiaudio-encode does not depend on oxiaudio-dsp; the K-weighting biquad and
// integrated LUFS measurement are inlined here as private helpers so that no
// new workspace dependency is introduced.

/// Private biquad filter coefficients (Direct Form II Transposed).
#[derive(Clone, Copy)]
struct LnBiquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl LnBiquad {
    /// High-shelf filter (RBJ Audio EQ Cookbook, S = 1).
    fn high_shelf(frequency: f32, gain_db: f32, sample_rate: u32) -> Self {
        let a = 10_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * frequency / sample_rate as f32;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 * 0.5 * (2.0_f32).sqrt();
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Highpass filter (RBJ Audio EQ Cookbook).
    fn highpass(frequency: f32, q: f32, sample_rate: u32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * frequency / sample_rate as f32;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Apply this filter to interleaved samples in-place, returning filtered output.
    fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let n_channels = buf.channels.channel_count();
        let mut z1 = vec![0.0_f32; n_channels];
        let mut z2 = vec![0.0_f32; n_channels];
        let mut out = Vec::with_capacity(buf.samples.len());
        for (i, &x) in buf.samples.iter().enumerate() {
            let ch = i % n_channels;
            let y = self.b0 * x + z1[ch];
            z1[ch] = self.b1 * x - self.a1 * y + z2[ch];
            z2[ch] = self.b2 * x - self.a2 * y;
            out.push(y);
        }
        AudioBuffer {
            samples: out,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }
}

/// Apply ITU-R BS.1770-4 K-weighting to `buf`.
fn ln_k_weight(buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
    let sr = buf.sample_rate;
    let stage1 = if sr == 48_000 {
        LnBiquad {
            b0: 1.535_124_9_f32,
            b1: -2.691_696_2_f32,
            b2: 1.198_392_9_f32,
            a1: -1.690_659_3_f32,
            a2: 0.732_480_77_f32,
        }
    } else {
        LnBiquad::high_shelf(1681.97, 4.0, sr)
    };
    let stage2 = if sr == 48_000 {
        LnBiquad {
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,
            a1: -1.990_047_5_f32,
            a2: 0.990_072_25_f32,
        }
    } else {
        LnBiquad::highpass(38.135, (0.5_f32).sqrt(), sr)
    };
    stage2.process(&stage1.process(buf))
}

/// Measure EBU R128 integrated loudness (private; mirrors `oxiaudio_dsp::loudness_integrated`).
///
/// Returns `f32::NEG_INFINITY` for silent or too-short signals.
fn ln_loudness_integrated(buf: &AudioBuffer<f32>) -> f32 {
    let weighted = ln_k_weight(buf);
    let sr = buf.sample_rate;
    let ch = buf.channels.channel_count();
    let frames = weighted.samples.len() / ch.max(1);
    let block_frames = (0.4 * sr as f64) as usize;
    let hop_frames = (0.1 * sr as f64) as usize;
    if block_frames == 0 || hop_frames == 0 || frames < block_frames {
        return f32::NEG_INFINITY;
    }
    let n_blocks = (frames - block_frames) / hop_frames + 1;
    let mut block_lufs: Vec<f32> = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let start = b * hop_frames;
        let end = (start + block_frames).min(frames);
        let count = (end - start) * ch;
        if count == 0 {
            continue;
        }
        let mean_sq: f32 = weighted.samples[start * ch..end * ch]
            .iter()
            .map(|&s| s * s)
            .sum::<f32>()
            / count as f32;
        let lufs = if mean_sq < 1e-12 {
            -200.0
        } else {
            -0.691 + 10.0 * mean_sq.log10()
        };
        block_lufs.push(lufs);
    }
    let gated1: Vec<f32> = block_lufs.iter().copied().filter(|&l| l > -70.0).collect();
    if gated1.is_empty() {
        return f32::NEG_INFINITY;
    }
    let ungated_power: f32 =
        gated1.iter().map(|&l| 10.0f32.powf(l / 10.0)).sum::<f32>() / gated1.len() as f32;
    let ungated_lufs = -0.691 + 10.0 * ungated_power.log10();
    let rel_thresh = ungated_lufs - 10.0;
    let gated2: Vec<f32> = gated1.iter().copied().filter(|&l| l > rel_thresh).collect();
    if gated2.is_empty() {
        return f32::NEG_INFINITY;
    }
    let final_power: f32 =
        gated2.iter().map(|&l| 10.0f32.powf(l / 10.0)).sum::<f32>() / gated2.len() as f32;
    -0.691 + 10.0 * final_power.log10()
}

// ─── LoudnessTarget ───────────────────────────────────────────────────────────

/// Target loudness levels for normalization.
#[derive(Debug, Clone, Copy)]
pub enum LoudnessTarget {
    /// -14 LUFS (Spotify, YouTube)
    Streaming,
    /// -16 LUFS (Apple Podcasts)
    Podcast,
    /// -23 LUFS (EBU R128, broadcast)
    Broadcast,
    /// Custom LUFS value.
    Custom(f32),
}

impl LoudnessTarget {
    /// Returns the target LUFS value.
    pub fn lufs(self) -> f32 {
        match self {
            Self::Streaming => -14.0,
            Self::Podcast => -16.0,
            Self::Broadcast => -23.0,
            Self::Custom(v) => v,
        }
    }
}

/// Analyze integrated loudness of `buf` and compute the linear gain needed to reach `target`.
///
/// Uses EBU R128 K-weighted integrated loudness measurement.
/// Returns the gain factor (linear; e.g. `1.2` means boost by 20%).
/// Returns `1.0` if the signal is silent or too short to measure (to avoid divide-by-zero).
///
/// # Errors
///
/// Returns [`OxiAudioError`] only on internal computation failure (currently infallible;
/// the `Result` wrapper is provided for forward-compatibility).
#[must_use = "discarding the gain result ignores encode failure"]
pub fn analyze_loudness_gain(
    buf: &AudioBuffer<f32>,
    target: LoudnessTarget,
) -> Result<f32, OxiAudioError> {
    let measured_lufs = ln_loudness_integrated(buf);
    if !measured_lufs.is_finite() {
        // Silent or too short: no adjustment
        return Ok(1.0);
    }
    let gain_db = target.lufs() - measured_lufs;
    let gain_linear = 10_f32.powf(gain_db / 20.0);
    Ok(gain_linear)
}

/// Apply a linear gain factor to all samples of `buf`, clamping to `[-1.0, 1.0]`.
fn apply_gain_clamp(buf: &AudioBuffer<f32>, gain: f32) -> AudioBuffer<f32> {
    let samples = buf
        .samples
        .iter()
        .map(|&s| (s * gain).clamp(-1.0, 1.0))
        .collect();
    AudioBuffer {
        samples,
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        format: buf.format,
    }
}

/// Two-pass loudness normalization: measure → gain → encode to WAV.
///
/// Pass 1: measure integrated LUFS.
/// Pass 2: apply gain to reach `target`, then encode.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on loudness analysis failure or WAV encode failure.
#[must_use = "discarding errors ignores encode failure"]
pub fn encode_normalized_wav<W: std::io::Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    writer: W,
    target: LoudnessTarget,
) -> Result<(), OxiAudioError> {
    let gain = analyze_loudness_gain(buf, target)?;
    let normalized = apply_gain_clamp(buf, gain);
    use oxiaudio_core::AudioEncoder;
    crate::wav_core::WavEncoder::default()
        .encode(&normalized, writer)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))
}

/// Two-pass loudness normalization: encode to a WAV file at `path`.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on file creation failure, loudness analysis failure,
/// or WAV encode failure.
#[must_use = "discarding errors ignores encode failure"]
pub fn encode_normalized_wav_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    target: LoudnessTarget,
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    encode_normalized_wav(buf, std::io::BufWriter::new(file), target)
}

// ─── Loudness normalization tests ─────────────────────────────────────────────

#[cfg(test)]
mod loudness_tests {
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    use super::{analyze_loudness_gain, encode_normalized_wav, LoudnessTarget};

    fn sine_buf(amplitude: f32, sr: u32, secs: f32) -> AudioBuffer<f32> {
        let n = (sr as f32 * secs) as usize;
        AudioBuffer {
            samples: (0..n)
                .map(|i| {
                    amplitude * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin()
                })
                .collect(),
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_loudness_target_lufs_values() {
        assert_eq!(LoudnessTarget::Streaming.lufs(), -14.0);
        assert_eq!(LoudnessTarget::Podcast.lufs(), -16.0);
        assert_eq!(LoudnessTarget::Broadcast.lufs(), -23.0);
        assert_eq!(LoudnessTarget::Custom(-18.5).lufs(), -18.5);
    }

    #[test]
    fn test_encode_normalized_wav_reduces_loud_signal() {
        // Peak amplitude 0.9 → loud signal; 1 second at 48 kHz (> 0.4 s block threshold)
        let buf = sine_buf(0.9, 48_000, 1.5);
        let mut cursor = Cursor::new(Vec::new());
        encode_normalized_wav(&buf, &mut cursor, LoudnessTarget::Broadcast)
            .expect("encode_normalized_wav");
        let bytes = cursor.into_inner();
        // Must be a valid WAV file (RIFF magic)
        assert!(!bytes.is_empty(), "output must not be empty");
        assert_eq!(&bytes[..4], b"RIFF", "output must start with RIFF magic");
    }

    #[test]
    fn test_analyze_loudness_gain_silent_returns_one() {
        // Silent buffer → gain should be 1.0 (no adjustment)
        let buf = AudioBuffer {
            samples: vec![0.0f32; 48_000],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let gain =
            analyze_loudness_gain(&buf, LoudnessTarget::Broadcast).expect("analyze_loudness_gain");
        assert_eq!(gain, 1.0, "silent signal should return gain of 1.0");
    }

    #[test]
    fn test_analyze_loudness_gain_is_positive() {
        // Any loud signal should return a positive gain factor
        let buf = sine_buf(0.5, 48_000, 1.5);
        let gain =
            analyze_loudness_gain(&buf, LoudnessTarget::Streaming).expect("analyze_loudness_gain");
        assert!(gain > 0.0, "gain must be positive, got {gain}");
    }
}

// ─── encode_flac_with_progress tests ──────────────────────────────────────────

#[cfg(test)]
mod flac_progress_tests {
    use std::cell::Cell;
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    use super::encode_flac_with_progress;

    fn mono_sine(sr: u32, n: usize) -> AudioBuffer<f32> {
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
    fn test_encode_flac_with_progress_callback_called() {
        // 0.2 s mono at 44100 Hz — callback must be called at least once.
        let buf = mono_sine(44_100, (44_100.0f32 * 0.2) as usize);
        let call_count = Cell::new(0usize);
        let mut out = Cursor::new(Vec::<u8>::new());
        encode_flac_with_progress(&buf, &mut out, 5, &|_done, _total| {
            call_count.set(call_count.get() + 1);
        })
        .expect("encode_flac_with_progress should succeed");
        assert!(
            call_count.get() >= 1,
            "callback must be called at least once"
        );
    }

    #[test]
    fn test_encode_flac_with_progress_final_count() {
        // Verify the last callback invocation passes total_frames for both arguments.
        let n = 44_100usize;
        let buf = mono_sine(44_100, n);
        let last_done = Cell::new(0usize);
        let last_total = Cell::new(0usize);
        let mut out = Cursor::new(Vec::<u8>::new());
        encode_flac_with_progress(&buf, &mut out, 5, &|done, total| {
            last_done.set(done);
            last_total.set(total);
        })
        .expect("encode_flac_with_progress should succeed");
        // total_frames = n (mono)
        assert_eq!(
            last_total.get(),
            n,
            "final callback total must equal total_frames"
        );
        assert_eq!(
            last_done.get(),
            n,
            "final callback done must equal total_frames"
        );
    }

    #[test]
    fn test_encode_flac_with_progress_produces_valid_flac() {
        let buf = mono_sine(44_100, 8192);
        let mut out = Cursor::new(Vec::<u8>::new());
        encode_flac_with_progress(&buf, &mut out, 5, &|_d, _t| {})
            .expect("encode_flac_with_progress should succeed");
        let bytes = out.into_inner();
        assert!(!bytes.is_empty(), "output must not be empty");
        assert_eq!(&bytes[..4], b"fLaC", "output must start with fLaC magic");
    }
}
