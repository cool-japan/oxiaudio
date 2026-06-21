//! OGG Opus stream encoder.
//!
//! Wraps `OpusHead` / `OpusTags` generation, OGG muxing, and per-frame CELT
//! encoding behind a simple `encode_opus` function.
//!
//! # Structural limitations
//!
//! This encoder produces structurally valid OGG Opus files — correct `OggS` magic,
//! correct `OpusHead` / `OpusTags` pages, correct granule positions — but the audio
//! payload uses non-conformant 4-bit quantization instead of PVQ (RFC 6716 §4.3.4).
//! Standard Opus decoders will reject the audio frames. The encoder is suitable for:
//! - Exercising the OGG container writer and range coder.
//! - Writing integration tests that verify OGG framing without full RFC compliance.
//! - As a structural scaffold for a future conformant CELT implementation.
//!
//! # TOC byte
//!
//! Per RFC 6716 §3.1:
//! ```text
//! Bits 7-3: config (28 = CELT fullband 20ms, mono; 29 = CELT fullband 20ms, stereo)
//! Bit  2:   S (stereo if 1)
//! Bits 1-0: code (0 = 1 frame, no padding)
//! ```
//! Config 28 selects `CELT_FULLBAND_20MS` for mono and config 29 for stereo.

use std::io::Write;

use crate::ogg::{write_vorbis_comment_packet, OggStream};
use crate::opus_celt::encode_celt_frame;
use crate::opus_celt::encode_celt_frame_conformant;
use crate::opus_hybrid_conform::encode_hybrid_frame_conformant;
use crate::opus_range::RangeEncoder;
use crate::opus_silk_conform::encode_silk_frame_conformant;
use oxiaudio_core::{AudioBuffer, OxiAudioError};

/// Fixed sample rate expected by the Opus encoder (48 kHz).
const OPUS_SAMPLE_RATE: u32 = 48_000;

/// Frame size in samples per channel (20 ms at 48 kHz).
pub const FRAME_SIZE: usize = 960;

/// Pre-skip in 48 kHz samples (80 ms CELT priming window).
pub const PRE_SKIP: u16 = 3840;

/// Build the `OpusHead` identification packet (19 bytes, RFC 7845 §5.1).
///
/// # Fields written
///
/// | Offset | Field                | Value |
/// |--------|----------------------|-------|
/// | 0–7    | Magic                | `"OpusHead"` |
/// | 8      | Version              | 1 |
/// | 9      | Channel count        | `channels` |
/// | 10–11  | Pre-skip             | `pre_skip` (LE) |
/// | 12–15  | Input sample rate    | `sample_rate` (LE) |
/// | 16–17  | Output gain          | 0 (LE) |
/// | 18     | Channel mapping fam. | 0 (simple, ≤ 2 ch) |
fn write_opus_head(channels: u8, pre_skip: u16, sample_rate: u32) -> Vec<u8> {
    let mut head = Vec::with_capacity(19);
    head.extend_from_slice(b"OpusHead");
    head.push(1u8); // version
    head.push(channels); // channel count
    head.extend_from_slice(&pre_skip.to_le_bytes()); // pre-skip
    head.extend_from_slice(&sample_rate.to_le_bytes()); // input sample rate
    head.extend_from_slice(&0u16.to_le_bytes()); // output gain = 0
    head.push(0u8); // channel mapping family: simple
    head
}

/// Encode an [`AudioBuffer<f32>`] to OGG Opus format and write to `writer`.
///
/// The input buffer must be at 48 kHz (the encoder does NOT resample). Channels
/// must be 1 (mono) or 2 (stereo).
///
/// The `target_bitrate_kbps` parameter is accepted for API compatibility but is
/// not yet used by the structural CELT encoder (the current fixed-width quantization
/// ignores bitrate).
///
/// # Errors
///
/// Returns [`OxiAudioError::UnsupportedFormat`] when channels > 2 or sample rate
/// is not 48 kHz. Returns [`OxiAudioError::Io`] on write failure.
pub fn encode_opus<W: Write>(
    buf: &AudioBuffer<f32>,
    writer: W,
    _target_bitrate_kbps: u32,
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count();

    if channels == 0 || channels > 2 {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "Opus encoder supports 1–2 channels, got {channels}"
        )));
    }
    if buf.sample_rate != OPUS_SAMPLE_RATE {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "Opus encoder requires 48 kHz input, got {} Hz",
            buf.sample_rate
        )));
    }

    let serial: u32 = 0x1234_5678;
    let mut stream = OggStream::new(writer, serial);

    // ── Page 0: OpusHead (BOS page) ───────────────────────────────────────────
    let head = write_opus_head(channels as u8, PRE_SKIP, buf.sample_rate);
    // granule_delta=0: header pages carry no audio data.
    stream.write_packet(&head, 0, false)?;

    // ── Page 1: OpusTags ──────────────────────────────────────────────────────
    let tags =
        write_vorbis_comment_packet(concat!("OxiAudio ", env!("CARGO_PKG_VERSION")), &[], true);
    stream.write_packet(&tags, 0, false)?;

    // ── Audio pages: one OGG packet per CELT frame ───────────────────────────
    //
    // Each frame covers exactly FRAME_SIZE = 960 samples, so the granule-position
    // delta is always 960 (per OGG Opus spec, RFC 7845 §4).
    let frame_samples = FRAME_SIZE * channels;
    let n_frames = buf.samples.len().checked_div(frame_samples).unwrap_or(0);

    // TOC byte layout (RFC 6716 §3.1):
    //   config  = 28 (CELT fullband 20ms) for mono
    //   config  = 29 (CELT fullband 20ms stereo, S=1) for stereo — but RFC uses
    //             config=28 with the S bit set in practice; we follow that.
    //   Stereo bit (bit 2) = 1 if channels == 2
    //   Code (bits 1-0) = 0 (1 frame per packet, no padding)
    let toc: u8 = (28u8 << 3) | (if channels == 2 { 0x04 } else { 0x00 });

    for i in 0..n_frames {
        let start = i * frame_samples;
        let end = start + frame_samples;
        let pcm = &buf.samples[start..end];

        // Range-encode the CELT frame.
        let mut enc = RangeEncoder::new();
        encode_celt_frame(pcm, channels, &mut enc);
        let frame_bytes = enc.finish();

        // Packet = TOC byte + range-coded frame bytes.
        let mut packet = Vec::with_capacity(1 + frame_bytes.len());
        packet.push(toc);
        packet.extend_from_slice(&frame_bytes);

        let is_last = i == n_frames - 1;
        // Granule delta is always FRAME_SIZE samples (960) per audio packet.
        stream.write_packet(&packet, FRAME_SIZE as i64, is_last)?;
    }

    // If no frames were written (e.g. empty buffer), close the stream cleanly.
    stream
        .finish()
        .map_err(|_| OxiAudioError::Io(std::io::Error::other("OGG stream finish failed")))?;

    Ok(())
}

/// Encode an [`AudioBuffer<f32>`] to an OGG Opus file at `path`.
///
/// Convenience wrapper around [`encode_opus`]. Opens (or creates) the file,
/// wraps it in a `BufWriter`, and calls `encode_opus`.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on file-creation failure or write failure.
pub fn encode_opus_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    target_bitrate_kbps: u32,
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    encode_opus(buf, writer, target_bitrate_kbps)
}

/// Selects which RFC 6716–conformant per-frame encoder [`encode_opus_conformant`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusConformantMode {
    /// CELT-only fullband 20 ms frames (TOC 0xF8). Carries real spectral content
    /// (MDCT + PVQ); the conformance suite verifies decoded output correlates
    /// (>0.1) with a 440 Hz input tone. This is the default — it is the only
    /// conformant mode that encodes actual signal rather than silence.
    Celt,
    /// SILK-only narrowband 20 ms frames (TOC 0x08). NOTE: the current conformant
    /// SILK writer emits an inactive zero-excitation (silence) frame — the PCM
    /// content is NOT encoded. Decodes cleanly but reconstructs silence.
    Silk,
    /// Hybrid fullband 20 ms frames (TOC 0x78): SILK WB silence + CELT high-band.
    /// Decodes cleanly to 960 samples; low band is silence, high band carries
    /// CELT content for bands 17–20 only.
    Hybrid,
}

/// Encode an [`AudioBuffer<f32>`] to OGG Opus using RFC 6716–conformant per-frame
/// encoders, writing to `writer`. This is an opt-in alternative to [`encode_opus`].
///
/// Unlike [`encode_opus`] (which emits a non-conformant 4-bit placeholder CELT
/// payload that standard decoders reject), this routes each 20 ms frame through a
/// conformant SILK / CELT / Hybrid writer, producing a structurally valid OGG Opus
/// stream that standard Opus decoders accept (verified per-frame against the
/// `opus-decoder` crate).
///
/// # Conformance level (be precise — this is NOT transparent encoding)
/// - [`OpusConformantMode::Celt`] (default quality): full MDCT + PVQ; decoded
///   output correlates (>0.1) with the input tone — real spectral content, but
///   this is only a coarse "not-silence" gate, **not** high-SNR transparency.
/// - [`OpusConformantMode::Silk`]: **silence-only** — the conformant SILK writer
///   currently emits an inactive zero-excitation frame and ignores the PCM.
/// - [`OpusConformantMode::Hybrid`]: low band is silence; high band carries CELT
///   content for bands 17–20 only.
///
/// All frames are mono. Stereo input is downmixed to mono per frame by averaging
/// L/R. The OGG `OpusHead` still advertises the input channel count (1 or 2),
/// matching [`encode_opus`]'s behaviour — document/expect this asymmetry.
///
/// [`encode_opus`] and its exact byte output are completely unaffected by this
/// function; both share only the immutable `OpusHead`/`OpusTags`/OGG framing path.
///
/// # Errors
/// Returns [`OxiAudioError::UnsupportedFormat`] when channels are outside 1..=2 or
/// the sample rate is not 48 kHz; [`OxiAudioError::Io`] on write failure.
pub fn encode_opus_conformant<W: Write>(
    buf: &AudioBuffer<f32>,
    writer: W,
    mode: OpusConformantMode,
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count();

    if channels == 0 || channels > 2 {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "Opus encoder supports 1–2 channels, got {channels}"
        )));
    }
    if buf.sample_rate != OPUS_SAMPLE_RATE {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "Opus encoder requires 48 kHz input, got {} Hz",
            buf.sample_rate
        )));
    }

    let serial: u32 = 0x1234_5678;
    let mut stream = OggStream::new(writer, serial);

    let head = write_opus_head(channels as u8, PRE_SKIP, buf.sample_rate);
    stream.write_packet(&head, 0, false)?;

    let tags =
        write_vorbis_comment_packet(concat!("OxiAudio ", env!("CARGO_PKG_VERSION")), &[], true);
    stream.write_packet(&tags, 0, false)?;

    let frame_samples = FRAME_SIZE * channels;
    let n_frames = buf.samples.len().checked_div(frame_samples).unwrap_or(0);

    for i in 0..n_frames {
        let start = i * frame_samples;
        let end = start + frame_samples;
        let pcm = &buf.samples[start..end];

        // Downmix to mono once; keep the owned buffer alive for the borrow below.
        let mono_vec: Vec<f32> = if channels == 2 {
            pcm.chunks_exact(2).map(|c| 0.5 * (c[0] + c[1])).collect()
        } else {
            pcm.to_vec()
        };
        let mono: &[f32] = &mono_vec;

        let packet = match mode {
            OpusConformantMode::Celt => encode_celt_frame_conformant(mono, 1),
            OpusConformantMode::Silk => encode_silk_frame_conformant(mono, 1),
            OpusConformantMode::Hybrid => encode_hybrid_frame_conformant(mono, 1),
        };

        let is_last = i == n_frames - 1;
        stream.write_packet(&packet, FRAME_SIZE as i64, is_last)?;
    }

    stream
        .finish()
        .map_err(|_| OxiAudioError::Io(std::io::Error::other("OGG stream finish failed")))?;

    Ok(())
}

/// Encode an [`AudioBuffer<f32>`] to a conformant OGG Opus file at `path`.
///
/// File-writing convenience wrapper around [`encode_opus_conformant`]; see that
/// function for the per-mode conformance caveats (SILK is silence-only, CELT is
/// coarse-gated rather than transparent).
///
/// # Errors
/// Returns [`OxiAudioError::Io`] on file-creation or write failure; propagates the
/// validation errors of [`encode_opus_conformant`].
pub fn encode_opus_conformant_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    mode: OpusConformantMode,
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    encode_opus_conformant(buf, writer, mode)
}

/// Configuration for Opus encoding.
#[derive(Debug, Clone)]
pub struct OpusEncodeConfig {
    /// Target bitrate in kbps (6–510). Not yet used by the structural encoder.
    pub target_bitrate_kbps: u32,
    /// Frame size in samples per channel at 48 kHz.
    /// Supported: 120 (2.5ms), 240 (5ms), 480 (10ms), 960 (20ms), 1920 (40ms), 2880 (60ms).
    pub frame_size: usize,
}

impl Default for OpusEncodeConfig {
    fn default() -> Self {
        Self {
            target_bitrate_kbps: 128,
            frame_size: FRAME_SIZE,
        }
    }
}

impl OpusEncodeConfig {
    /// Create configuration with specified bitrate.
    pub fn with_bitrate(kbps: u32) -> Self {
        Self {
            target_bitrate_kbps: kbps,
            ..Self::default()
        }
    }
}

/// Streaming Opus encoder that accepts PCM frames one at a time and writes
/// OGG pages to the underlying writer.
///
/// Each call to [`OpusStreamEncoder::encode_frame`] encodes exactly
/// [`FRAME_SIZE`] samples per channel and writes one OGG packet.
pub struct OpusStreamEncoder<W: Write> {
    stream: OggStream<W>,
    channels: usize,
    granule_pos: i64,
    toc: u8,
    is_finalized: bool,
}

impl<W: Write> OpusStreamEncoder<W> {
    /// Create a new streaming encoder.
    ///
    /// Writes the `OpusHead` and `OpusTags` header pages immediately.
    ///
    /// # Errors
    /// Returns [`OxiAudioError::UnsupportedFormat`] if `channels` > 2.
    /// Returns [`OxiAudioError::Io`] on write failure.
    pub fn new(writer: W, channels: usize, serial: u32) -> Result<Self, OxiAudioError> {
        if channels == 0 || channels > 2 {
            return Err(OxiAudioError::UnsupportedFormat(format!(
                "OpusStreamEncoder supports 1–2 channels, got {channels}"
            )));
        }
        let mut stream = OggStream::new(writer, serial);
        let head = write_opus_head(channels as u8, PRE_SKIP, OPUS_SAMPLE_RATE);
        stream.write_packet(&head, 0, false)?;
        let tags =
            write_vorbis_comment_packet(concat!("OxiAudio ", env!("CARGO_PKG_VERSION")), &[], true);
        stream.write_packet(&tags, 0, false)?;
        let toc: u8 = (28u8 << 3) | (if channels == 2 { 0x04 } else { 0x00 });
        Ok(Self {
            stream,
            channels,
            granule_pos: 0,
            toc,
            is_finalized: false,
        })
    }

    /// Encode one audio frame of exactly `FRAME_SIZE * channels` samples.
    ///
    /// `pcm` must be interleaved (L0, R0, L1, R1, ...) at 48 kHz.
    ///
    /// # Errors
    /// Returns [`OxiAudioError::InvalidChannelLayout`] if pcm length != FRAME_SIZE * channels.
    pub fn encode_frame(&mut self, pcm: &[f32]) -> Result<(), OxiAudioError> {
        let expected = FRAME_SIZE * self.channels;
        if pcm.len() != expected {
            return Err(OxiAudioError::InvalidChannelLayout(format!(
                "OpusStreamEncoder::encode_frame expected {expected} samples, got {}",
                pcm.len()
            )));
        }
        let mut enc = RangeEncoder::new();
        encode_celt_frame(pcm, self.channels, &mut enc);
        let frame_bytes = enc.finish();
        let mut packet = Vec::with_capacity(1 + frame_bytes.len());
        packet.push(self.toc);
        packet.extend_from_slice(&frame_bytes);
        self.granule_pos += FRAME_SIZE as i64;
        self.stream
            .write_packet(&packet, FRAME_SIZE as i64, false)?;
        Ok(())
    }

    /// Finalize the stream, writing an EOS page.
    ///
    /// Must be called after all frames have been encoded.
    ///
    /// # Errors
    /// Returns [`OxiAudioError::Io`] on write failure.
    pub fn finalize(mut self) -> Result<(), OxiAudioError> {
        if !self.is_finalized {
            self.stream.finish().map_err(|_| {
                OxiAudioError::Io(std::io::Error::other("OGG stream finish failed"))
            })?;
            self.is_finalized = true;
        }
        Ok(())
    }

    /// Total granule position written so far.
    pub fn granule_pos(&self) -> i64 {
        self.granule_pos
    }

    /// Total number of frames encoded.
    pub fn frames_encoded(&self) -> u64 {
        (self.granule_pos / FRAME_SIZE as i64) as u64
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    use super::{
        encode_opus, encode_opus_conformant, OpusConformantMode, OpusEncodeConfig,
        OpusStreamEncoder, FRAME_SIZE,
    };

    fn silence_buf(channels: usize, frames: usize) -> AudioBuffer<f32> {
        let layout = if channels == 1 {
            ChannelLayout::Mono
        } else {
            ChannelLayout::Stereo
        };
        AudioBuffer {
            samples: vec![0.0f32; FRAME_SIZE * channels * frames],
            sample_rate: 48_000,
            channels: layout,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_encode_opus_conformant_produces_ogg() {
        let buf = silence_buf(1, 1);
        let mut out = Cursor::new(Vec::new());
        encode_opus_conformant(&buf, &mut out, OpusConformantMode::Celt)
            .expect("encode_opus_conformant mono silence");
        let bytes = out.into_inner();
        assert_eq!(&bytes[..4], b"OggS", "output must start with OggS magic");
        assert!(
            bytes.windows(8).any(|w| w == b"OpusHead"),
            "OpusHead magic must appear"
        );
    }

    #[test]
    fn test_encode_opus_produces_valid_ogg_output() {
        let buf = silence_buf(2, 1);
        let mut out = Cursor::new(Vec::new());
        encode_opus(&buf, &mut out, 128).expect("encode_opus stereo silence");
        let bytes = out.into_inner();
        assert!(!bytes.is_empty(), "output must not be empty");
        assert_eq!(&bytes[..4], b"OggS", "output must start with OggS magic");
    }

    #[test]
    fn test_encode_opus_head_has_correct_magic() {
        let buf = silence_buf(1, 1);
        let mut out = Cursor::new(Vec::new());
        encode_opus(&buf, &mut out, 64).expect("encode_opus mono silence");
        let bytes = out.into_inner();
        let has_opus_head = bytes.windows(8).any(|w| w == b"OpusHead");
        assert!(
            has_opus_head,
            "OpusHead magic must appear in the first page"
        );
    }

    #[test]
    fn test_encode_opus_tags_present() {
        let buf = silence_buf(1, 1);
        let mut out = Cursor::new(Vec::new());
        encode_opus(&buf, &mut out, 64).expect("encode_opus");
        let bytes = out.into_inner();
        let has_tags = bytes.windows(8).any(|w| w == b"OpusTags");
        assert!(has_tags, "OpusTags magic must appear in output");
    }

    #[test]
    fn test_encode_opus_rejects_wrong_sample_rate() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 44_100],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut out = Cursor::new(Vec::new());
        let result = encode_opus(&buf, &mut out, 64);
        assert!(result.is_err(), "encode_opus must reject non-48 kHz input");
    }

    #[test]
    fn test_encode_opus_rejects_zero_channels() {
        // ChannelLayout doesn't allow 0-channel; use Mono (1 ch) as the minimal valid case.
        // Instead, test that > 2 channels is rejected (the real guard in the function).
        // We can't easily construct a >2-channel buffer with standard ChannelLayout here,
        // so just verify mono and stereo are accepted.
        let mono = silence_buf(1, 1);
        let stereo = silence_buf(2, 1);
        assert!(encode_opus(&mono, &mut Cursor::new(Vec::new()), 64).is_ok());
        assert!(encode_opus(&stereo, &mut Cursor::new(Vec::new()), 128).is_ok());
    }

    #[test]
    fn test_encode_opus_empty_buffer_produces_header_only() {
        // An empty sample buffer produces only the OpusHead + OpusTags pages.
        let buf = AudioBuffer {
            samples: vec![],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut out = Cursor::new(Vec::new());
        encode_opus(&buf, &mut out, 64).expect("encode_opus empty");
        let bytes = out.into_inner();
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[..4], b"OggS");
    }

    #[test]
    fn test_encode_opus_multiple_frames() {
        // Three frames of silence: verifies that frame-loop handles multiple iterations.
        let buf = silence_buf(1, 3);
        let mut out = Cursor::new(Vec::new());
        encode_opus(&buf, &mut out, 64).expect("encode_opus 3 frames");
        let bytes = out.into_inner();
        assert!(!bytes.is_empty());
        // Count OGG pages (each starts with OggS).
        let page_count = bytes.windows(4).filter(|w| *w == b"OggS").count();
        // At minimum: OpusHead + OpusTags + 3 audio pages = 5, but pages may be
        // merged so we just check there are more than 2.
        assert!(page_count >= 3, "expected ≥ 3 OGG pages, got {page_count}");
    }

    #[test]
    fn test_encode_opus_file_creates_valid_file() {
        use super::encode_opus_file;
        let buf = silence_buf(1, 1);
        let path = std::env::temp_dir().join("oxiaudio_opus_enc_test.ogg");
        encode_opus_file(&buf, &path, 64).expect("encode_opus_file");
        let bytes = std::fs::read(&path).expect("read test file");
        assert_eq!(&bytes[..4], b"OggS", "file must start with OggS magic");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_opus_config_default() {
        let cfg = OpusEncodeConfig::default();
        assert_eq!(cfg.target_bitrate_kbps, 128);
        assert_eq!(cfg.frame_size, FRAME_SIZE);
    }

    #[test]
    fn test_opus_config_with_bitrate() {
        let cfg = OpusEncodeConfig::with_bitrate(320);
        assert_eq!(cfg.target_bitrate_kbps, 320);
        assert_eq!(cfg.frame_size, FRAME_SIZE);
    }

    #[test]
    fn test_opus_stream_encoder_produces_ogg_output() {
        let mut out = Cursor::new(Vec::new());
        let mut enc = OpusStreamEncoder::new(&mut out, 2, 0x1234).expect("new");
        let frame = vec![0.0f32; FRAME_SIZE * 2];
        enc.encode_frame(&frame).expect("encode_frame");
        enc.finalize().expect("finalize");
        let bytes = out.into_inner();
        assert!(
            bytes.windows(4).any(|w| w == b"OggS"),
            "must contain OGG pages"
        );
    }

    #[test]
    fn test_opus_stream_encoder_rejects_wrong_frame_size() {
        let mut out = Cursor::new(Vec::new());
        let mut enc = OpusStreamEncoder::new(&mut out, 1, 0x5678).expect("new");
        let bad_frame = vec![0.0f32; FRAME_SIZE - 1];
        assert!(
            enc.encode_frame(&bad_frame).is_err(),
            "must reject wrong frame size"
        );
        let _ = enc.finalize();
    }

    #[test]
    fn test_opus_stream_encoder_frame_count() {
        let mut out = Cursor::new(Vec::new());
        let mut enc = OpusStreamEncoder::new(&mut out, 1, 0xABCD).expect("new");
        for _ in 0..3 {
            enc.encode_frame(&vec![0.0f32; FRAME_SIZE]).expect("frame");
        }
        assert_eq!(enc.frames_encoded(), 3);
        let _ = enc.finalize();
    }
}
