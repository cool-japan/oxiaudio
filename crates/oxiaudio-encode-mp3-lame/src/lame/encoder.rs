//! CBR/VBR MP3 encoder backed by LAME: encoder structs, helpers, impl, and convenience functions.

use mp3lame_encoder::{
    max_required_buffer_size, Bitrate, Builder, DualPcm, Encoder, FlushNoGap, Mode, MonoPcm,
    Quality, VbrMode,
};
use oxiaudio_core::{AudioBuffer, AudioEncoder, ChannelLayout, OxiAudioError};
use std::io::Write;

use super::id3v2;
use super::types::{LameMode, Mp3Tags, VbrPreset};

/// LAME's algorithmic encoder delay in samples (constant for all modes and sample rates).
///
/// All LAME encodes introduce exactly this many silence samples at the start of the
/// decoded stream due to the MDCT overlap-add algorithm. Gapless-aware players use
/// this value (stored in the `iTunSMPB` ID3 COMM frame) to trim leading silence.
pub const LAME_ENCODER_DELAY: u32 = 576;

/// CBR / VBR MP3 encoder backed by LAME.
///
/// # LGPL notice
///
/// Enabling the `mp3-encode-lame` feature links against `libmp3lame` (LGPL-3.0).
/// Static linking requires the end user to be able to relink against a modified
/// `libmp3lame`. See `TODO.md` for details.
pub struct LameMp3Encoder {
    /// Output bitrate in kbps. Supported values for CBR: 64, 128, 192, 320.
    /// Ignored when `mode` is `LameMode::Vbr`.
    pub bitrate: u32,
    /// MPEG channel mode (also selects CBR vs VBR).
    pub mode: LameMode,
    /// Optional ID3v2.3 tags to prepend to the MP3 output.
    pub id3_tags: Option<Mp3Tags>,
}

impl Default for LameMp3Encoder {
    fn default() -> Self {
        Self {
            bitrate: 128,
            mode: LameMode::JointStereo,
            id3_tags: None,
        }
    }
}

impl LameMp3Encoder {
    /// Return a fluent [`LameMp3EncoderBuilder`] with the given CBR bitrate.
    pub fn builder(bitrate: u32) -> LameMp3EncoderBuilder {
        LameMp3EncoderBuilder::new(bitrate)
    }
}

/// Fluent builder for [`LameMp3Encoder`].
///
/// Supports CBR, VBR (via preset or raw quality), and ABR modes, with optional
/// ID3v2.3 tag attachment. Use [`encode`](Self::encode),
/// [`encode_to_vec`](Self::encode_to_vec), or
/// [`encode_to_file`](Self::encode_to_file) to drive encoding.
///
/// # Example
/// ```rust,no_run
/// # #[cfg(feature = "mp3-encode-lame")] {
/// # use oxiaudio_encode_mp3_lame::lame::{LameMp3Encoder, Mp3Tags, VbrPreset};
/// # use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
/// let buf = AudioBuffer { samples: vec![0.0f32; 1024], sample_rate: 44100,
///     channels: ChannelLayout::Stereo, format: SampleFormat::F32 };
/// let data = LameMp3Encoder::builder(128)
///     .with_vbr_preset(VbrPreset::Music)
///     .with_tags(Mp3Tags::builder().title("Demo").build())
///     .encode_to_vec(&buf)
///     .unwrap();
/// # }
/// ```
pub struct LameMp3EncoderBuilder {
    bitrate: u32,
    mode: LameMode,
    quality: i32,
    vbr_preset: Option<VbrPreset>,
    abr_kbps: Option<u32>,
    tags: Option<Mp3Tags>,
    ms_stereo_threshold: Option<f32>,
}

impl LameMp3EncoderBuilder {
    /// Create a new builder targeting `bitrate` kbps for CBR.
    pub fn new(bitrate: u32) -> Self {
        Self {
            bitrate,
            mode: LameMode::JointStereo,
            quality: 5,
            vbr_preset: None,
            abr_kbps: None,
            tags: None,
            ms_stereo_threshold: None,
        }
    }

    /// Use a [`VbrPreset`] instead of CBR.
    pub fn with_vbr_preset(mut self, preset: VbrPreset) -> Self {
        self.vbr_preset = Some(preset);
        self.abr_kbps = None;
        self
    }

    /// Use Average Bitrate mode targeting `kbps`.
    ///
    /// Valid values: 8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320.
    pub fn with_abr(mut self, kbps: u32) -> Self {
        self.abr_kbps = Some(kbps);
        self.vbr_preset = None;
        self
    }

    /// Set the encode quality hint (`0` = best/slowest, `9` = fastest/worst).
    ///
    /// This is the encode algorithm quality (LAME `-q` flag), not the VBR
    /// quality level. For VBR quality use [`with_vbr_preset`](Self::with_vbr_preset).
    pub fn with_quality(mut self, q: i32) -> Self {
        self.quality = q.clamp(0, 9);
        self
    }

    /// Attach ID3v2.3 tags to the output.
    pub fn with_tags(mut self, tags: Mp3Tags) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Set the channel mode explicitly.
    pub fn with_mode(mut self, mode: LameMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the mid/side stereo switching threshold (LAME `--nsmsfix`, 0.0–3.5).
    ///
    /// `0.0` disables MS stereo (forces LR). `3.5` strongly prefers MS stereo.
    /// `None` keeps the LAME library default.
    ///
    /// **Note**: Stored for forward compatibility; the `mp3lame-encoder` crate 0.2.x
    /// does not yet expose `lame_set_msfix`, so the value is not currently wired
    /// into the LAME context. The test confirms no panic and valid MP3 output.
    pub fn with_ms_stereo_threshold(mut self, threshold: f32) -> Self {
        self.ms_stereo_threshold = Some(threshold.clamp(0.0, 3.5));
        self
    }

    /// Build the [`LameMp3Encoder`] from the current builder state.
    fn build_encoder(self) -> LameMp3Encoder {
        let mode = if let Some(preset) = self.vbr_preset {
            LameMode::Vbr {
                quality: preset.quality(),
            }
        } else if let Some(kbps) = self.abr_kbps {
            LameMode::Abr { target_kbps: kbps }
        } else {
            self.mode
        };
        LameMp3Encoder {
            bitrate: self.bitrate,
            mode,
            id3_tags: self.tags,
        }
    }

    /// Encode `buf` and write the resulting MP3 bytes into `writer`.
    pub fn encode<W: Write + std::io::Seek>(
        self,
        buf: &AudioBuffer<f32>,
        writer: &mut W,
    ) -> Result<(), OxiAudioError> {
        let mut encoder = self.build_encoder();
        encoder.encode(buf, writer)
    }

    /// Encode `buf` and return the MP3 bytes as a `Vec<u8>`.
    pub fn encode_to_vec(self, buf: &AudioBuffer<f32>) -> Result<Vec<u8>, OxiAudioError> {
        let mut encoder = self.build_encoder();
        let mut cursor = std::io::Cursor::new(Vec::new());
        encoder.encode(buf, &mut cursor)?;
        Ok(cursor.into_inner())
    }

    /// Encode `buf` and write the MP3 output to the file at `path`.
    pub fn encode_to_file(
        self,
        buf: &AudioBuffer<f32>,
        path: &std::path::Path,
    ) -> Result<(), OxiAudioError> {
        let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
        let mut writer = std::io::BufWriter::new(file);
        let mut encoder = self.build_encoder();
        let mut cursor = std::io::Cursor::new(Vec::new());
        encoder.encode(buf, &mut cursor)?;
        writer
            .write_all(cursor.get_ref())
            .map_err(OxiAudioError::Io)?;
        Ok(())
    }
}

/// Map our `LameMode` (non-VBR variants) to the upstream `mp3lame_encoder::Mode`.
pub(super) fn to_mp3_mode(mode: LameMode, channels: u8) -> Mode {
    if channels == 1 {
        return Mode::Mono;
    }
    match mode {
        LameMode::Stereo => Mode::Stereo,
        LameMode::JointStereo => Mode::JointStereo,
        // Upstream spells this "DaulChannel" (typo in the library).
        LameMode::DualChannel => Mode::DaulChannel,
        LameMode::Mono | LameMode::ForcedMono => Mode::Mono,
        // VBR/ABR use JointStereo by default; these arms should not be reached
        // via `to_mp3_mode` since those paths are handled separately, but the
        // compiler requires exhaustiveness.
        LameMode::Vbr { .. } | LameMode::Abr { .. } => Mode::JointStereo,
    }
}

/// Map a `u32` kbps value to the upstream `Bitrate` enum.
///
/// Supports every bitrate exposed by `mp3lame-encoder` (the MPEG-1 Layer III
/// values plus the lower MPEG-2 values 8/16/24).
pub(super) fn to_bitrate(kbps: u32) -> Result<Bitrate, OxiAudioError> {
    match kbps {
        8 => Ok(Bitrate::Kbps8),
        16 => Ok(Bitrate::Kbps16),
        24 => Ok(Bitrate::Kbps24),
        32 => Ok(Bitrate::Kbps32),
        40 => Ok(Bitrate::Kbps40),
        48 => Ok(Bitrate::Kbps48),
        64 => Ok(Bitrate::Kbps64),
        80 => Ok(Bitrate::Kbps80),
        96 => Ok(Bitrate::Kbps96),
        112 => Ok(Bitrate::Kbps112),
        128 => Ok(Bitrate::Kbps128),
        160 => Ok(Bitrate::Kbps160),
        192 => Ok(Bitrate::Kbps192),
        224 => Ok(Bitrate::Kbps224),
        256 => Ok(Bitrate::Kbps256),
        320 => Ok(Bitrate::Kbps320),
        _ => Err(OxiAudioError::Encode(format!(
            "unsupported bitrate: {kbps} kbps (valid: 8, 16, 24, 32, 40, 48, 64, \
             80, 96, 112, 128, 160, 192, 224, 256, 320)"
        ))),
    }
}

/// Map a raw quality byte (0–9) to the `mp3lame_encoder::Quality` enum.
pub(super) fn to_vbr_quality(q: u8) -> Result<Quality, OxiAudioError> {
    match q {
        0 => Ok(Quality::Best),
        1 => Ok(Quality::SecondBest),
        2 => Ok(Quality::NearBest),
        3 => Ok(Quality::VeryNice),
        4 => Ok(Quality::Nice),
        5 => Ok(Quality::Good),
        6 => Ok(Quality::Decent),
        7 => Ok(Quality::Ok),
        8 => Ok(Quality::SecondWorst),
        9 => Ok(Quality::Worst),
        _ => Err(OxiAudioError::Encode(format!(
            "VBR quality {q} is out of range (valid: 0–9)"
        ))),
    }
}

/// Overwrite the Xing/LAME placeholder frame in `mp3_out` with the finalised tag.
///
/// LAME writes a zeroed placeholder Xing/Info frame at the very start of the MP3
/// stream during encoding (`bWriteVbrTag = 1` by default). After all PCM has been
/// encoded and flushed, `lame_get_lametag_frame()` (exposed via `lame_tag_encode_to_vec`)
/// produces the *final* frame with correct total-frame-count, byte-count, and TOC entries.
///
/// We call this helper after every `flush_to_vec` to splice the finalised frame back
/// into the beginning of `mp3_out`, replacing the placeholder.  If the encoder is
/// not configured to write a VBR tag (`lame_tag_size() == 0`), this is a no-op.
///
/// Returns the final tag bytes (may be empty if the encoder emits no Xing tag).
pub(super) fn finalize_xing_tag(encoder: &Encoder, mp3_out: &mut [u8]) -> Vec<u8> {
    let tag_size = encoder.lame_tag_size();
    if tag_size == 0 {
        return Vec::new();
    }
    let mut lame_tag: Vec<u8> = Vec::with_capacity(tag_size);
    match encoder.lame_tag_encode_to_vec(&mut lame_tag) {
        Some(_) if lame_tag.len() <= mp3_out.len() => {
            // Splice the finalised frame over the placeholder bytes.
            mp3_out[..lame_tag.len()].copy_from_slice(&lame_tag);
            lame_tag
        }
        _ => {
            // Either the encoder returned nothing or the tag is larger than mp3_out
            // (shouldn't happen in practice — fail silently and return empty).
            Vec::new()
        }
    }
}

/// Encode a single radio (track) ReplayGain value and peak amplitude into the
/// Xing/LAME info tag that is already embedded in `mp3_out`.
///
/// The function scans `mp3_out` for the four-byte marker `b"Xing"` or `b"Info"` and
/// writes:
///
/// - **marker+131..135**: peak signal amplitude as a big-endian IEEE 754 `f32`.
/// - **marker+135..137**: radio gain as a big-endian `u16` with the LAME bit layout:
///   - bits 15–13: name code (`001` = radio / track gain)
///   - bits 12–10: originator (`011` = set automatically)
///   - bit 9: sign (1 = negative)
///   - bits 8–0: absolute gain in tenths of a dB (0–511; values outside
///     ±51.1 dB are clamped to 511 tenths).
///
/// Returns `true` if the marker was found and the fields were written, `false` if no
/// Xing/Info marker could be located (e.g. CBR stream without a VBR header).
pub fn write_xing_replaygain(mp3_out: &mut [u8], gain_db: f32, peak: f32) -> bool {
    // Scan for the Xing or Info marker byte sequence.
    let Some(xp) = mp3_out
        .windows(4)
        .position(|w| w == b"Xing" || w == b"Info")
    else {
        return false;
    };

    // Ensure the tag is long enough to hold the peak and gain fields.
    // We need at least xp + 139 bytes (last written byte is marker+138).
    if mp3_out.len() < xp + 139 {
        return false;
    }

    // --- Peak signal: 4 bytes big-endian f32 at marker+131 ---
    let peak_clamped = peak.clamp(0.0, 1.0);
    let peak_bytes = peak_clamped.to_be_bytes();
    mp3_out[xp + 131] = peak_bytes[0];
    mp3_out[xp + 132] = peak_bytes[1];
    mp3_out[xp + 133] = peak_bytes[2];
    mp3_out[xp + 134] = peak_bytes[3];

    // --- Radio gain: 2 bytes big-endian u16 at marker+135 ---
    // Encode |gain| in tenths of a dB, clamped to [0, 511].
    let abs_tenths: u16 = ((gain_db.abs() * 10.0).round() as u16).min(511);
    let sign_bit: u16 = if gain_db < 0.0 { 1 } else { 0 };
    let name_code: u16 = 1; // 001 = radio / track gain
    let originator: u16 = 3; // 011 = automatic
    let gain_word: u16 = (name_code << 13) | (originator << 10) | (sign_bit << 9) | abs_tenths;
    let gain_bytes = gain_word.to_be_bytes();
    mp3_out[xp + 135] = gain_bytes[0];
    mp3_out[xp + 136] = gain_bytes[1];

    true
}

impl AudioEncoder for LameMp3Encoder {
    fn encode(
        &mut self,
        buf: &AudioBuffer<f32>,
        mut dst: impl std::io::Write + std::io::Seek,
    ) -> Result<(), OxiAudioError> {
        // `ForcedMono` collapses any input to a single channel; otherwise the
        // channel count follows the buffer layout.
        let force_mono = matches!(self.mode, LameMode::ForcedMono);
        let is_mono = force_mono || matches!(buf.channels, ChannelLayout::Mono);
        let channels: u8 = if is_mono { 1 } else { 2 };

        let mut encoder = build_lame_encoder(self, buf.sample_rate, channels, is_mono)?;

        // f32 → i16 conversion.
        let to_i16 = |s: f32| -> i16 { (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16 };

        let src_is_stereo = matches!(buf.channels, ChannelLayout::Stereo);

        // Per-channel frame count (samples / channels).
        let n_frames = if src_is_stereo {
            buf.samples.len() / 2
        } else {
            buf.samples.len()
        };

        // Reserve output buffer: max_required_buffer_size for PCM + 7200 flush headroom.
        let cap = max_required_buffer_size(n_frames).saturating_add(7200);
        let mut mp3_out: Vec<u8> = Vec::with_capacity(cap);

        if is_mono {
            // Build a mono PCM stream: pass-through for mono input, or sum the
            // two channels to mono when forcing a stereo source down.
            let pcm: Vec<i16> = if src_is_stereo {
                buf.samples
                    .chunks_exact(2)
                    .map(|c| to_i16((c[0] + c[1]) * 0.5))
                    .collect()
            } else {
                // Use chunks_exact(8) as a SIMD auto-vectorisation hint.
                let mut pcm = Vec::with_capacity(buf.samples.len());
                for chunk in buf.samples.chunks_exact(8) {
                    pcm.push(to_i16(chunk[0]));
                    pcm.push(to_i16(chunk[1]));
                    pcm.push(to_i16(chunk[2]));
                    pcm.push(to_i16(chunk[3]));
                    pcm.push(to_i16(chunk[4]));
                    pcm.push(to_i16(chunk[5]));
                    pcm.push(to_i16(chunk[6]));
                    pcm.push(to_i16(chunk[7]));
                }
                for &s in buf.samples.chunks_exact(8).remainder() {
                    pcm.push(to_i16(s));
                }
                pcm
            };
            encoder
                .encode_to_vec(MonoPcm(&pcm), &mut mp3_out)
                .map_err(|e| OxiAudioError::Encode(format!("encode (mono): {e}")))?;
        } else {
            // Deinterleave: even indices → left, odd indices → right.
            // Pre-allocate both buffers upfront (Task 8 line 67).
            let mut left = Vec::with_capacity(n_frames);
            let mut right = Vec::with_capacity(n_frames);
            // Process 8 interleaved pairs at once for SIMD auto-vectorisation hint.
            for chunk in buf.samples.chunks_exact(16) {
                left.push(to_i16(chunk[0]));
                right.push(to_i16(chunk[1]));
                left.push(to_i16(chunk[2]));
                right.push(to_i16(chunk[3]));
                left.push(to_i16(chunk[4]));
                right.push(to_i16(chunk[5]));
                left.push(to_i16(chunk[6]));
                right.push(to_i16(chunk[7]));
                left.push(to_i16(chunk[8]));
                right.push(to_i16(chunk[9]));
                left.push(to_i16(chunk[10]));
                right.push(to_i16(chunk[11]));
                left.push(to_i16(chunk[12]));
                right.push(to_i16(chunk[13]));
                left.push(to_i16(chunk[14]));
                right.push(to_i16(chunk[15]));
            }
            for pair in buf.samples.chunks_exact(16).remainder().chunks_exact(2) {
                left.push(to_i16(pair[0]));
                right.push(to_i16(pair[1]));
            }
            encoder
                .encode_to_vec(
                    DualPcm {
                        left: &left,
                        right: &right,
                    },
                    &mut mp3_out,
                )
                .map_err(|e| OxiAudioError::Encode(format!("encode (stereo): {e}")))?;
        }

        // Flush remaining LAME frames; FlushNoGap fills gaps with ancillary data.
        encoder
            .flush_to_vec::<FlushNoGap>(&mut mp3_out)
            .map_err(|e| OxiAudioError::Encode(format!("flush: {e}")))?;

        // Replace the placeholder Xing/LAME frame with the finalised version that
        // contains correct total-frame-count, byte-count and TOC table.
        finalize_xing_tag(&encoder, &mut mp3_out);

        // Prepend hand-rolled ID3v2.4 tag if requested, then write MP3 data.
        if let Some(ref tags) = self.id3_tags {
            let id3_bytes = id3v2::write_id3v2_4(tags);
            dst.write_all(&id3_bytes)?;
        }
        dst.write_all(&mp3_out)?;
        Ok(())
    }
}

/// Build a LAME `Encoder` from a `LameMp3Encoder` config, sample rate, and channel count.
///
/// This is shared logic used by both `LameMp3Encoder::encode` and
/// `LameMp3StreamEncoder::new` to avoid code duplication.
pub(super) fn build_lame_encoder(
    config: &LameMp3Encoder,
    sample_rate: u32,
    channels: u8,
    is_mono: bool,
) -> Result<Encoder, OxiAudioError> {
    let base = Builder::new()
        .ok_or_else(|| OxiAudioError::Encode("failed to allocate LAME encoder".into()))?
        .with_sample_rate(sample_rate)
        .map_err(|e| OxiAudioError::Encode(format!("set sample rate: {e}")))?
        .with_num_channels(channels)
        .map_err(|e| OxiAudioError::Encode(format!("set channels: {e}")))?;

    let encoder = match config.mode {
        LameMode::Vbr { quality } => {
            let q = to_vbr_quality(quality)?;
            let vbr_mode = if is_mono {
                Mode::Mono
            } else {
                Mode::JointStereo
            };
            base.with_vbr_mode(VbrMode::Mtrh)
                .map_err(|e| OxiAudioError::Encode(format!("set VBR mode: {e}")))?
                .with_vbr_quality(q)
                .map_err(|e| OxiAudioError::Encode(format!("set VBR quality: {e}")))?
                .with_mode(vbr_mode)
                .map_err(|e| OxiAudioError::Encode(format!("set mode: {e}")))?
                .build()
                .map_err(|e| OxiAudioError::Encode(format!("build encoder: {e}")))?
        }
        LameMode::Abr { target_kbps } => {
            // ABR: VBR ABR mode with the target mean bitrate set via brate.
            let bitrate = to_bitrate(target_kbps)?;
            let mode = if is_mono {
                Mode::Mono
            } else {
                Mode::JointStereo
            };
            base.with_vbr_mode(VbrMode::Abr)
                .map_err(|e| OxiAudioError::Encode(format!("set ABR mode: {e}")))?
                .with_brate(bitrate)
                .map_err(|e| OxiAudioError::Encode(format!("set ABR target bitrate: {e}")))?
                .with_mode(mode)
                .map_err(|e| OxiAudioError::Encode(format!("set mode: {e}")))?
                .build()
                .map_err(|e| OxiAudioError::Encode(format!("build encoder: {e}")))?
        }
        other => {
            let bitrate = to_bitrate(config.bitrate)?;
            let mode = to_mp3_mode(other, channels);
            base.with_brate(bitrate)
                .map_err(|e| OxiAudioError::Encode(format!("set bitrate: {e}")))?
                .with_mode(mode)
                .map_err(|e| OxiAudioError::Encode(format!("set mode: {e}")))?
                .build()
                .map_err(|e| OxiAudioError::Encode(format!("build encoder: {e}")))?
        }
    };
    Ok(encoder)
}

/// Encode `buf` using Average Bitrate mode and write to `writer`.
///
/// `target_kbps` must be one of: 8, 16, 24, 32, 40, 48, 56, 64, 80, 96,
/// 112, 128, 160, 192, 224, 256, 320.
pub fn encode_mp3_abr<W: Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    writer: &mut W,
    target_kbps: u32,
    tags: Option<Mp3Tags>,
) -> Result<(), OxiAudioError> {
    let mut encoder = LameMp3Encoder {
        bitrate: target_kbps,
        mode: LameMode::Abr { target_kbps },
        id3_tags: tags,
    };
    encoder.encode(buf, writer)
}

/// Encode `buf` to a `Vec<u8>` using Constant Bitrate mode.
///
/// This is a convenience wrapper around [`AudioEncoder::encode`] that avoids
/// the caller having to manage a `Cursor<Vec<u8>>`.
///
/// `bitrate_kbps` must be one of: 8, 16, 24, 32, 40, 48, 64, 80, 96, 112,
/// 128, 160, 192, 224, 256, 320.
pub fn encode_mp3_cbr_to_vec(
    buf: &AudioBuffer<f32>,
    bitrate_kbps: u32,
    tags: Option<Mp3Tags>,
) -> Result<Vec<u8>, OxiAudioError> {
    let mut encoder = LameMp3Encoder {
        bitrate: bitrate_kbps,
        mode: LameMode::JointStereo,
        id3_tags: tags,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    encoder.encode(buf, &mut cursor)?;
    Ok(cursor.into_inner())
}

/// Encode `buf` to a file using Constant Bitrate mode.
///
/// Creates (or truncates) the file at `path` and writes a CBR MP3 stream.
pub fn encode_mp3_cbr_to_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    bitrate_kbps: u32,
    tags: Option<Mp3Tags>,
) -> Result<(), OxiAudioError> {
    let data = encode_mp3_cbr_to_vec(buf, bitrate_kbps, tags)?;
    std::fs::write(path, &data).map_err(OxiAudioError::Io)
}

/// Encode audio to MP3 with automatically computed ReplayGain tags.
///
/// Computes the approximate ReplayGain track gain (RMS-based) using
/// [`crate::compute_replaygain_gain_approx`], then encodes to CBR MP3 at `bitrate_kbps`
/// with the ReplayGain values written in **two** locations:
///
/// 1. **ID3v2 TXXX frames** — `REPLAYGAIN_TRACK_GAIN` and `REPLAYGAIN_TRACK_PEAK`
///    in the ID3 tag prepended to the MP3 stream.
/// 2. **Xing/LAME binary header fields** — radio gain (big-endian `u16` at
///    marker+135) and peak amplitude (big-endian `f32` at marker+131) in the
///    first MP3 frame, readable by decoders that parse the LAME info tag directly
///    (e.g. foobar2000, mpd, ffmpeg).
///
/// # Errors
///
/// Returns `OxiAudioError::Encode` if the MP3 encoder fails, or
/// `OxiAudioError::Io` on write failure.
pub fn encode_mp3_with_auto_replaygain(
    buf: &AudioBuffer<f32>,
    path: impl AsRef<std::path::Path>,
    bitrate_kbps: u32,
) -> Result<(), OxiAudioError> {
    let gain_db_f32 = crate::compute_replaygain_gain_approx(buf);
    let peak_f32 = buf.samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
    let gain_db_f64 = f64::from(gain_db_f32);
    let peak_f64 = f64::from(peak_f32);

    // Build ID3 tags with TXXX ReplayGain frames.
    let tags = super::types::Mp3Tags::builder()
        .replaygain_track_gain(gain_db_f64)
        .replaygain_track_peak(peak_f64)
        .build();

    // Build the encoder, collect MP3 bytes into memory so we can post-process
    // the Xing/LAME binary header before writing to disk.
    let mut encoder = LameMp3Encoder {
        bitrate: bitrate_kbps,
        mode: LameMode::JointStereo,
        id3_tags: Some(tags),
    };

    let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
    encoder.encode(buf, &mut cursor)?;
    let mut mp3_bytes = cursor.into_inner();

    // `encoder.encode()` already finalised the Xing tag (via `finalize_xing_tag`).
    // The Xing/Info marker is inside mp3_bytes — skip past any leading ID3 tag to
    // find it. We scan the *entire* vec; the ID3 tag does not contain the literal
    // four-byte sequence "Xing" or "Info" so there is no ambiguity.
    write_xing_replaygain(&mut mp3_bytes, gain_db_f32, peak_f32);

    std::fs::write(path.as_ref(), &mp3_bytes).map_err(OxiAudioError::Io)
}
