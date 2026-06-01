//! Pure-Rust OGG Opus decoder backed by the `opus-decoder` crate.
//!
//! This module wraps the `opus-decoder` pure-Rust implementation (RFC 8251 conformant,
//! no unsafe, no FFI) with OGG container demuxing per RFC 3533 / RFC 7845.
//!
//! # OGG Opus stream structure (RFC 7845)
//!
//! An OGG Opus file contains three logical sections:
//! 1. **OpusHead** packet — 19 bytes, magic `"OpusHead"`, version, channel count,
//!    pre_skip (samples to discard), input sample rate, output gain, mapping family.
//! 2. **OpusTags** packet — Vorbis comment metadata; discarded during decode.
//! 3. **Audio packets** — raw Opus packets passed directly to `opus_decoder::OpusDecoder`.
//!
//! The `pre_skip` value from OpusHead is used to trim leading encoder priming samples.

#[cfg(feature = "opus")]
use opus_decoder::OpusDecoder as InnerDecoder;

#[cfg(feature = "opus")]
use crate::ogg_reader::OggReader;
#[cfg(feature = "opus")]
use oxiaudio_core::{ChannelLayout, SampleFormat};

use oxiaudio_core::{AudioBuffer, OxiAudioError};
use std::io::Read;
use std::path::Path;

// ─── OpusHead parsing ────────────────────────────────────────────────────────

/// Parsed contents of an OGG Opus `OpusHead` identification header.
#[derive(Debug, Clone)]
pub struct OpusHead {
    /// Number of output channels (1 or 2 for simple streams).
    pub channels: u8,
    /// Number of 48 kHz samples to discard from the beginning of the decoded stream.
    pub pre_skip: u16,
    /// Original input sample rate (informational; output is always 48 kHz).
    pub input_sample_rate: u32,
    /// Output gain in Q7.8 format (dB).
    pub output_gain: i16,
    /// Channel mapping family (0 = simple; 1 = Vorbis-compatible multistream).
    pub mapping_family: u8,
}

/// Parse the `OpusHead` binary packet (RFC 7845 §5.1).
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] when the packet is too short, or the magic
/// `"OpusHead"` prefix is missing, or the version byte is not 1.
pub fn parse_opus_head(packet: &[u8]) -> Result<OpusHead, OxiAudioError> {
    // Minimum size: 8 (magic) + 1 (version) + 1 (channels) + 2 (pre_skip) + 4 (input_rate)
    // + 2 (output_gain) + 1 (mapping_family) = 19 bytes.
    if packet.len() < 19 {
        return Err(OxiAudioError::Decode(format!(
            "OpusHead too short: {} bytes (min 19)",
            packet.len()
        )));
    }
    if &packet[..8] != b"OpusHead" {
        return Err(OxiAudioError::Decode("OpusHead magic not found".into()));
    }
    let version = packet[8];
    if version != 1 {
        return Err(OxiAudioError::Decode(format!(
            "Unsupported OpusHead version: {version}"
        )));
    }
    let channels = packet[9];
    let pre_skip = u16::from_le_bytes([packet[10], packet[11]]);
    let input_sample_rate = u32::from_le_bytes([packet[12], packet[13], packet[14], packet[15]]);
    let output_gain = i16::from_le_bytes([packet[16], packet[17]]);
    let mapping_family = packet[18];
    Ok(OpusHead {
        channels,
        pre_skip,
        input_sample_rate,
        output_gain,
        mapping_family,
    })
}

// ─── Public OpusDecoder facade ────────────────────────────────────────────────

/// High-level OGG Opus decoder that exposes a per-packet decode interface.
///
/// Wraps `opus_decoder::OpusDecoder` (pure Rust, no FFI) with the pre-skip
/// book-keeping required by RFC 7845.
#[cfg(feature = "opus")]
pub struct OpusDecoder {
    inner: InnerDecoder,
    channels: usize,
    /// Remaining 48 kHz samples to discard from the start of decoded output.
    pre_skip_remaining: usize,
    /// Scratch buffer reused across decode calls (avoids per-call allocation).
    pcm_scratch: Vec<f32>,
}

#[cfg(feature = "opus")]
impl OpusDecoder {
    /// Create a new `OpusDecoder` for the given `sample_rate` and `channels`.
    ///
    /// `sample_rate` must be one of 8000, 12000, 16000, 24000, or 48000.
    /// `channels` must be 1 or 2.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Decode`] when `sample_rate` or `channels` is not supported
    /// by the underlying `opus-decoder` crate.
    pub fn new(sample_rate: u32, channels: usize) -> Result<Self, OxiAudioError> {
        let inner = InnerDecoder::new(sample_rate, channels)
            .map_err(|e| OxiAudioError::Decode(e.to_string()))?;
        Ok(Self {
            inner,
            channels,
            pre_skip_remaining: 0,
            pcm_scratch: Vec::new(),
        })
    }

    /// Create a new `OpusDecoder` using the pre_skip value from an [`OpusHead`] packet.
    ///
    /// This is the preferred constructor when decoding an OGG Opus file, as it
    /// automatically configures pre_skip trimming.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Decode`] on invalid `sample_rate` or `channels`.
    pub fn from_opus_head(head: &OpusHead, sample_rate: u32) -> Result<Self, OxiAudioError> {
        let mut dec = Self::new(sample_rate, head.channels as usize)?;
        dec.pre_skip_remaining = head.pre_skip as usize;
        Ok(dec)
    }

    /// Decode a single raw Opus packet to f32 PCM samples (interleaved if stereo).
    ///
    /// Pre-skip samples are consumed internally and do not appear in the output.
    /// Returns interleaved PCM samples (may be empty while pre-skip is being consumed).
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Decode`] on malformed packets or decode failures.
    pub fn decode_packet(&mut self, data: &[u8]) -> Result<Vec<f32>, OxiAudioError> {
        // Maximum frame size: 120 ms @ 48 kHz = 5760 samples per channel.
        let max_samples = InnerDecoder::MAX_FRAME_SIZE_48K * self.channels;
        if self.pcm_scratch.len() < max_samples {
            self.pcm_scratch.resize(max_samples, 0.0);
        }

        let samples_per_channel = self
            .inner
            .decode_float(data, &mut self.pcm_scratch[..max_samples], false)
            .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

        let total_samples = samples_per_channel * self.channels;
        let decoded = &self.pcm_scratch[..total_samples];

        // Apply pre_skip: skip the first `pre_skip_remaining` interleaved samples
        // (pre_skip is expressed in samples per channel at 48 kHz per RFC 7845).
        let skip_interleaved = (self.pre_skip_remaining * self.channels).min(total_samples);
        if skip_interleaved >= total_samples {
            self.pre_skip_remaining -= skip_interleaved / self.channels;
            return Ok(Vec::new());
        }
        self.pre_skip_remaining = 0;
        Ok(decoded[skip_interleaved..].to_vec())
    }
}

// ─── OGG stream decode ────────────────────────────────────────────────────────

/// Decode an OGG Opus file from any `Read` reader.
///
/// Parses the OGG container, reads the `OpusHead` header for channel count and
/// `pre_skip`, discards the `OpusTags` metadata packet, then decodes all audio
/// packets and returns the assembled interleaved f32 PCM in an [`AudioBuffer<f32>`].
///
/// The output sample rate is always 48 kHz (the native Opus decoding rate).
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] when the stream is not valid OGG Opus, or
/// [`OxiAudioError::Io`] on underlying I/O failures.
///
/// # Feature flag
///
/// Requires the `opus` feature (`oxiaudio-decode = { features = ["opus"] }`).
#[cfg(feature = "opus")]
pub fn decode_opus_reader<R: Read>(reader: R) -> Result<AudioBuffer<f32>, OxiAudioError> {
    const OUTPUT_SAMPLE_RATE: u32 = 48_000;

    let mut ogg = OggReader::new(reader);

    // Packet 1: OpusHead
    let head_packet = ogg
        .read_packet()?
        .ok_or_else(|| OxiAudioError::Decode("OGG stream is empty (no OpusHead)".into()))?;
    let head = parse_opus_head(&head_packet)?;

    // Packet 2: OpusTags — discard.
    let tags_packet = ogg.read_packet()?;
    if let Some(ref tags) = tags_packet {
        if tags.len() < 8 || &tags[..8] != b"OpusTags" {
            // Tolerate: some encoders may deviate slightly; log and continue.
            log::warn!("second OGG Opus packet does not have OpusTags magic; continuing anyway");
        }
    }
    // If the stream ended after OpusHead (malformed but not fatal), return silence.
    if tags_packet.is_none() {
        log::warn!("OGG Opus stream has no audio packets (only OpusHead)");
        let layout = ChannelLayout::from(head.channels as u16);
        return Ok(AudioBuffer {
            samples: Vec::new(),
            sample_rate: OUTPUT_SAMPLE_RATE,
            channels: layout,
            format: SampleFormat::F32,
        });
    }

    let channels = head.channels as usize;
    let layout = ChannelLayout::from(head.channels as u16);
    let mut decoder = OpusDecoder::from_opus_head(&head, OUTPUT_SAMPLE_RATE)?;

    let mut all_samples: Vec<f32> = Vec::new();

    // Decode all audio packets.
    while let Some(packet) = ogg.read_packet()? {
        // Skip empty or degenerate packets.
        if packet.is_empty() {
            continue;
        }

        match decoder.decode_packet(&packet) {
            Ok(pcm) => {
                all_samples.extend_from_slice(&pcm);
            }
            Err(OxiAudioError::Decode(msg)) => {
                // Soft decode error: skip and continue (tolerant decoding matches
                // the existing pattern in SymphoniaDecoder for DecodeError).
                log::warn!("Opus decode error (packet skipped): {msg}");
                continue;
            }
            Err(e) => return Err(e),
        }
    }

    // Sanity-check channel alignment.
    if channels > 0 && all_samples.len() % channels != 0 {
        log::warn!(
            "Opus: sample count {} not divisible by channels {}; truncating",
            all_samples.len(),
            channels
        );
        let remainder = all_samples.len() % channels;
        all_samples.truncate(all_samples.len() - remainder);
    }

    Ok(AudioBuffer {
        samples: all_samples,
        sample_rate: OUTPUT_SAMPLE_RATE,
        channels: layout,
        format: SampleFormat::F32,
    })
}

/// Decode an OGG Opus file at `path` to an [`AudioBuffer<f32>`].
///
/// Opens the file and delegates to [`decode_opus_reader`].
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened, or
/// [`OxiAudioError::Decode`] if the OGG Opus data is malformed.
///
/// # Feature flag
///
/// Requires the `opus` feature (`oxiaudio-decode = { features = ["opus"] }`).
#[cfg(feature = "opus")]
pub fn decode_opus_file(path: &Path) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    let reader = std::io::BufReader::new(file);
    decode_opus_reader(reader)
}

// ─── Stub API for when the opus feature is disabled ──────────────────────────

#[cfg(not(feature = "opus"))]
/// Placeholder: requires the `opus` feature to be enabled.
pub fn decode_opus_reader<R: Read>(_reader: R) -> Result<AudioBuffer<f32>, OxiAudioError> {
    Err(OxiAudioError::UnsupportedFormat(
        "Opus decoding requires the 'opus' feature flag".into(),
    ))
}

#[cfg(not(feature = "opus"))]
/// Placeholder: requires the `opus` feature to be enabled.
pub fn decode_opus_file(_path: &Path) -> Result<AudioBuffer<f32>, OxiAudioError> {
    Err(OxiAudioError::UnsupportedFormat(
        "Opus decoding requires the 'opus' feature flag".into(),
    ))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ogg_reader::OggReader;
    use std::io::Cursor;

    // ─── OpusHead parsing ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_opus_head_valid() {
        // Build a minimal valid OpusHead: version=1, channels=2, pre_skip=312, input_rate=48000,
        // output_gain=0, mapping_family=0.
        let mut head = Vec::new();
        head.extend_from_slice(b"OpusHead");
        head.push(1); // version
        head.push(2); // channels
        head.extend_from_slice(&312u16.to_le_bytes()); // pre_skip
        head.extend_from_slice(&48_000u32.to_le_bytes()); // input_rate
        head.extend_from_slice(&0i16.to_le_bytes()); // output_gain
        head.push(0); // mapping_family

        let parsed = parse_opus_head(&head).expect("parse must succeed");
        assert_eq!(parsed.channels, 2);
        assert_eq!(parsed.pre_skip, 312);
        assert_eq!(parsed.input_sample_rate, 48_000);
        assert_eq!(parsed.output_gain, 0);
        assert_eq!(parsed.mapping_family, 0);
    }

    #[test]
    fn test_parse_opus_head_too_short() {
        let short = b"OpusHead";
        let err = parse_opus_head(short).expect_err("must fail on short packet");
        assert!(
            matches!(err, OxiAudioError::Decode(_)),
            "expected Decode error, got {err:?}"
        );
    }

    #[test]
    fn test_parse_opus_head_bad_magic() {
        let mut bad = vec![0u8; 19];
        bad[..8].copy_from_slice(b"BADMAGIC");
        let err = parse_opus_head(&bad).expect_err("must fail on bad magic");
        assert!(matches!(err, OxiAudioError::Decode(_)));
    }

    #[test]
    fn test_parse_opus_head_bad_version() {
        let mut pkt = vec![0u8; 19];
        pkt[..8].copy_from_slice(b"OpusHead");
        pkt[8] = 2; // version != 1
        let err = parse_opus_head(&pkt).expect_err("must fail on bad version");
        assert!(matches!(err, OxiAudioError::Decode(_)));
    }

    // ─── OpusDecoder construction ─────────────────────────────────────────────

    #[cfg(feature = "opus")]
    #[test]
    fn test_opus_decoder_new_48khz_stereo() {
        let dec = OpusDecoder::new(48_000, 2);
        assert!(dec.is_ok(), "OpusDecoder::new(48000, 2) must succeed");
    }

    #[cfg(feature = "opus")]
    #[test]
    fn test_opus_decoder_new_48khz_mono() {
        let dec = OpusDecoder::new(48_000, 1);
        assert!(dec.is_ok(), "OpusDecoder::new(48000, 1) must succeed");
    }

    #[cfg(feature = "opus")]
    #[test]
    fn test_opus_decoder_invalid_sample_rate() {
        let dec = OpusDecoder::new(44_100, 2);
        assert!(
            dec.is_err(),
            "OpusDecoder must reject 44100 Hz (not a valid Opus output rate)"
        );
    }

    #[cfg(feature = "opus")]
    #[test]
    fn test_opus_decoder_invalid_channels() {
        let dec = OpusDecoder::new(48_000, 3);
        assert!(dec.is_err(), "OpusDecoder must reject 3 channels");
    }

    // ─── decode_opus_file with non-existent path ──────────────────────────────

    #[cfg(feature = "opus")]
    #[test]
    fn test_decode_opus_file_nonexistent_returns_error() {
        let result = decode_opus_file(Path::new("/nonexistent_opus_file_xyz.opus"));
        assert!(
            result.is_err(),
            "expected Err for non-existent path, got Ok"
        );
    }

    // ─── decode_opus_reader on empty stream ───────────────────────────────────

    #[cfg(feature = "opus")]
    #[test]
    fn test_decode_opus_reader_empty_stream_returns_error() {
        // An empty reader has no OGG pages, so no OpusHead → Decode error.
        let empty: &[u8] = &[];
        let result = decode_opus_reader(Cursor::new(empty));
        assert!(result.is_err(), "empty stream must return an error");
    }

    // ─── OGG reader integration ───────────────────────────────────────────────

    #[test]
    fn test_ogg_reader_parse_empty_stream_returns_none() {
        let mut reader = OggReader::new(Cursor::new(vec![]));
        let result = reader
            .read_packet()
            .expect("must not error on empty stream");
        assert!(result.is_none());
    }
}
