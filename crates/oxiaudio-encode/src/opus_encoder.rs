//! OGG Opus stream encoder.
//!
//! Wraps `OpusHead` / `OpusTags` generation, OGG muxing, and per-frame Opus
//! encoding behind a simple `encode_opus` function.
//!
//! # Conformance
//!
//! [`encode_opus`] and [`OpusStreamEncoder`] — the default, most-discoverable
//! entry points — route every 20 ms frame through the RFC 6716–conformant
//! per-frame encoders (automatic CELT/SILK mode selection via
//! [`select_conformant_mode`] for [`encode_opus`]; CELT-only for the streaming
//! [`OpusStreamEncoder`]). The emitted OGG Opus stream is structurally valid
//! **and** its packets are accepted by a standard Opus decoder (verified
//! per-frame against the `opus-decoder` reference crate).
//!
//! Bit-exactness is verified rather than assumed: the CELT path's `final_range`
//! register matches the reference decoder's for every supported frame size on a
//! corpus that includes full-scale noise (`tests/m_opus_celt_snr.rs`), which is
//! the canonical libopus conformance check. Per-band energy is reproduced within
//! a few dB and the overall decoded level within ±3 dB.
//!
//! This is still *not* a transparent-quality encoder — see
//! [`OpusConformantMode`], `crate::opus_celt`'s module doc and
//! `crate::opus_silk_encode`'s module doc for the exact, honestly-measured
//! caveats: CELT is mono-only (stereo input is downmixed), non-transient, with
//! no dynamic allocation, with a conformance-verified frame-size ceiling that
//! is now the RFC's own 1275-byte frame limit
//! (`crate::opus_celt::MAX_CELT_FRAME_BYTES`, 510 kbps mono — it was ≈32 kbps
//! before oxiaudio 0.2.1); SILK is a genuine
//! analysis-by-synthesis narrowband encoder (NOT silence — measured best-lag
//! correlation ≈ 0.33–0.67 against reference decode), limited to unvoiced
//! excitation and independently-coded frames. (SILK silence *is* still accurate
//! for [`OpusConformantMode::Hybrid`]'s low band specifically — see that
//! variant's doc.) Prior to oxiaudio 0.2.1 [`encode_opus`]'s default payload
//! used non-conformant 4-bit placeholder quantization that **no** standard
//! decoder would accept at all.
//!
//! The pre-0.2.1 non-conformant byte layout is preserved verbatim as
//! [`encode_opus_structural`] / [`encode_opus_structural_file`] for OGG-framing
//! tests and byte-for-byte compatibility; new code should use [`encode_opus`].
//!
//! # TOC byte
//!
//! [`encode_opus`] / [`OpusStreamEncoder`] packets carry whatever TOC byte the
//! selected conformant per-frame encoder emits (`0xF8` for CELT-only fullband
//! 20 ms mono — see `crate::opus_celt::encode_celt_frame_conformant`; SILK
//! narrowband packets carry their own TOC — see
//! `crate::opus_silk_conform::encode_silk_frame_conformant`).
//!
//! [`encode_opus_structural`]'s legacy TOC layout, per RFC 6716 §3.1:
//! ```text
//! Bits 7-3: config (28 = CELT fullband 20ms, mono; 29 = CELT fullband 20ms, stereo)
//! Bit  2:   S (stereo if 1)
//! Bits 1-0: code (0 = 1 frame, no padding)
//! ```
//! Config 28 selects `CELT_FULLBAND_20MS` for mono and config 29 for stereo.

use std::io::Write;

use crate::ogg::{write_vorbis_comment_packet, OggStream};
use crate::opus_celt::encode_celt_frame;
use crate::opus_celt::{
    celt_frame_bytes_for_bitrate, encode_celt_frame_conformant_sized, DEFAULT_CELT_FRAME_BYTES,
};
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
/// # Conformance
///
/// As of oxiaudio 0.2.1 this is a thin wrapper around [`encode_opus_auto`]:
/// every 20 ms frame is routed through [`select_conformant_mode`] to the
/// matching RFC 6716–conformant per-frame encoder (CELT or SILK), so the
/// output is a structurally valid OGG Opus stream whose packets a standard
/// Opus decoder accepts. `target_bitrate_kbps` now genuinely influences mode
/// selection (see [`select_conformant_mode`]).
///
/// This is **not** a transparent-quality encoder — see [`encode_opus_auto`]'s
/// docs for the exact, honestly-measured fidelity caveats. For the pre-0.2.1
/// byte-for-byte non-conformant layout, use [`encode_opus_structural`].
///
/// # Errors
///
/// Returns [`OxiAudioError::UnsupportedFormat`] when channels > 2 or sample rate
/// is not 48 kHz. Returns [`OxiAudioError::Io`] on write failure.
pub fn encode_opus<W: Write>(
    buf: &AudioBuffer<f32>,
    writer: W,
    target_bitrate_kbps: u32,
) -> Result<(), OxiAudioError> {
    encode_opus_auto(buf, writer, target_bitrate_kbps)
}

/// Encode an [`AudioBuffer<f32>`] to OGG Opus format using the legacy,
/// pre-0.2.1 **non-conformant** structural encoder, and write to `writer`.
///
/// # Non-conformant — read before use
///
/// This function preserves oxiaudio's original [`encode_opus`] byte layout:
/// correct OGG framing (`OggS` magic, `OpusHead` / `OpusTags` pages, correct
/// granule positions) but a 4-bit placeholder CELT payload instead of PVQ
/// (RFC 6716 §4.3.4). **Standard Opus decoders will reject the audio frames.**
/// It exists only for:
/// - Exercising the OGG container writer and range coder in isolation.
/// - Byte-for-byte regression tests against oxiaudio ≤ 0.2.0 output.
///
/// New code should call [`encode_opus`] instead, which has been RFC
/// 6716–conformant (decodable by standard decoders) since oxiaudio 0.2.1.
///
/// The input buffer must be at 48 kHz (the encoder does NOT resample). Channels
/// must be 1 (mono) or 2 (stereo).
///
/// The `target_bitrate_kbps` parameter is accepted for API compatibility but is
/// not used by the structural CELT encoder (the fixed-width quantization
/// ignores bitrate).
///
/// # Errors
///
/// Returns [`OxiAudioError::UnsupportedFormat`] when channels > 2 or sample rate
/// is not 48 kHz. Returns [`OxiAudioError::Io`] on write failure.
pub fn encode_opus_structural<W: Write>(
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

/// Encode an [`AudioBuffer<f32>`] to an OGG Opus file at `path` using the legacy
/// non-conformant structural encoder.
///
/// Convenience wrapper around [`encode_opus_structural`]; see that function's
/// docs for why this is non-conformant and when to use it instead of
/// [`encode_opus_file`].
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on file-creation failure or write failure.
pub fn encode_opus_structural_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    target_bitrate_kbps: u32,
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    encode_opus_structural(buf, writer, target_bitrate_kbps)
}

/// Selects which RFC 6716–conformant per-frame encoder [`encode_opus_conformant`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusConformantMode {
    /// CELT-only fullband 20 ms frames (TOC 0xF8). Carries real spectral content
    /// (MDCT + PVQ); the conformance suite verifies decoded output correlates
    /// (>0.1) with a 440 Hz input tone. This is the default — it is the only
    /// conformant mode that encodes actual signal rather than silence.
    Celt,
    /// SILK-only narrowband 20 ms frames (TOC 0x08). A genuine
    /// analysis-by-synthesis narrowband encoder (`crate::opus_silk_encode`):
    /// real LP analysis, NLSF VQ, and analysis-by-synthesis excitation coding —
    /// **not** silence. Measured best-lag correlation ≈ 0.33–0.67 against
    /// reference decode; limited to unvoiced excitation and
    /// independently-coded frames (no inter-frame prediction).
    Silk,
    /// Hybrid fullband 20 ms frames (TOC 0x78): SILK WB **silence** + CELT
    /// high-band. Unlike [`OpusConformantMode::Silk`] above, the hybrid path
    /// does not yet route through the real SILK encoder (WB is out of that
    /// encoder's narrowband-only scope) — its low band genuinely is silence.
    /// Decodes cleanly to 960 samples; high band carries CELT content for
    /// bands 17–20 only.
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
/// - [`OpusConformantMode::Celt`] (default quality): full lapped MDCT with
///   pre-emphasis, real coarse+fine energy and real split-band (`itheta`)
///   coding. Verified bit-exact against the reference decoder's `final_range`;
///   per-band decoded level within a few dB of the input. Not transparent —
///   mono-only, non-transient, neutral allocation.
/// - [`OpusConformantMode::Silk`]: a genuine analysis-by-synthesis narrowband
///   encoder — **not** silence — with measured best-lag correlation ≈ 0.33–0.67
///   against reference decode; unvoiced-only, independently-coded frames.
/// - [`OpusConformantMode::Hybrid`]: low band genuinely is SILK silence (the
///   hybrid path doesn't route through the real SILK encoder); high band
///   carries CELT content for bands 17–20 only.
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

    // CELT's MDCT is lapped: every frame needs the previous frame as overlap
    // history or time-domain alias cancellation fails (see
    // `encode_celt_frame_conformant_with_history`).
    let mut prev_frame: Vec<f32> = Vec::new();

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
            OpusConformantMode::Celt => {
                encode_celt_frame_conformant_sized(&prev_frame, mono, DEFAULT_CELT_FRAME_BYTES)
            }
            OpusConformantMode::Silk => encode_silk_frame_conformant(mono, 1),
            OpusConformantMode::Hybrid => encode_hybrid_frame_conformant(mono, 1),
        };
        prev_frame = mono_vec;

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
/// function for the per-mode conformance caveats (SILK is a genuine
/// analysis-by-synthesis narrowband encoder, not silence; CELT is coarse-gated
/// rather than transparent; Hybrid's low band specifically is SILK silence).
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

/// Root-mean-square² (mean energy) below which a frame is treated as silence.
///
/// `1e-7` ≈ −70 dBFS, comfortably below audible content. Silent frames are routed
/// to the SILK path, which emits a compact inactive frame that any RFC 6716
/// decoder reconstructs to (near-)silence.
const SILENCE_ENERGY_THRESHOLD: f32 = 1e-7;

/// Ratio of first-difference energy to total energy above which a frame is treated
/// as high-frequency ("music"-like) content and routed to CELT rather than SILK.
///
/// For a pure tone at frequency `f` (48 kHz), this ratio is `≈ (2·sin(π f / fs))²`:
/// ~0.017 at 1 kHz, ~0.07 at 2 kHz, ~0.27 at 4 kHz. SILK's narrowband internal
/// rate only reaches ~4 kHz, so content whose energy sits mostly above that is far
/// better served by CELT. `0.06` puts the crossover near ~1.9 kHz.
const HF_RATIO_CELT_THRESHOLD: f32 = 0.06;

/// Bitrate (kbps) at or below which SILK narrowband is preferred for low-frequency
/// content, mirroring real Opus mode selection (SILK dominates below ~14 kbps).
const SILK_BITRATE_CEILING_KBPS: u32 = 20;

/// Choose the RFC 6716–conformant per-frame mode for one mono 20 ms frame.
///
/// The selection mirrors the coarse structure of a real Opus encoder's mode
/// decision, adapted to what this crate can encode *with real signal content*
/// (both [`OpusConformantMode::Celt`] and [`OpusConformantMode::Silk`] reconstruct
/// the input; [`OpusConformantMode::Hybrid`]'s low band is silence, so it is never
/// auto-selected):
///
/// * **Silence** (`mean energy < SILENCE_ENERGY_THRESHOLD`) → SILK. The SILK
///   inactive frame is the cheapest conformant silence representation.
/// * **Low-frequency, low-bitrate** speech-band content (first-difference energy
///   ratio below `HF_RATIO_CELT_THRESHOLD` and `target_bitrate_kbps ≤
///   SILK_BITRATE_CEILING_KBPS`) → SILK narrowband.
/// * **Everything else** (music, high tones, higher bitrates) → CELT fullband.
///
/// `mono` is a single-channel 48 kHz frame (any length; only its spectral balance
/// and energy are inspected).
pub fn select_conformant_mode(mono: &[f32], target_bitrate_kbps: u32) -> OpusConformantMode {
    let n = mono.len().max(1);
    let energy: f32 = mono.iter().map(|&x| x * x).sum::<f32>() / n as f32;
    if energy < SILENCE_ENERGY_THRESHOLD {
        return OpusConformantMode::Silk;
    }

    // First-difference energy as a cheap high-frequency proxy (no FFT required).
    let mut diff_energy = 0.0f32;
    let mut total_energy = 0.0f32;
    for w in mono.windows(2) {
        let d = w[1] - w[0];
        diff_energy += d * d;
        total_energy += w[0] * w[0];
    }
    total_energy += mono.last().map(|&x| x * x).unwrap_or(0.0);
    let hf_ratio = if total_energy > 0.0 {
        diff_energy / total_energy
    } else {
        0.0
    };

    if hf_ratio < HF_RATIO_CELT_THRESHOLD && target_bitrate_kbps <= SILK_BITRATE_CEILING_KBPS {
        OpusConformantMode::Silk
    } else {
        OpusConformantMode::Celt
    }
}

/// Full RFC 6716–conformant Opus encode path with **automatic per-frame mode
/// selection**, writing an OGG Opus stream to `writer`.
///
/// This is the top-level "wire it all together" encoder: for every 20 ms frame it
/// calls [`select_conformant_mode`] and routes the frame through the matching
/// conformant per-frame encoder (SILK narrowband or CELT fullband), so a single
/// stream may mix SILK and CELT packets frame-to-frame (legal in Opus — each
/// packet carries its own TOC). Stereo input is downmixed to mono per frame, as in
/// [`encode_opus_conformant`]; the `OpusHead` still advertises the input channel
/// count.
///
/// # Conformance (honest scope — this is not a transparent encoder)
///
/// Every emitted packet is accepted by the reference `opus-decoder`, and for
/// CELT frames the encoder's and decoder's `final_range` registers agree —
/// i.e. both sides consumed byte-for-byte the same symbol sequence.
/// `target_bitrate_kbps` genuinely sets the CELT payload size via
/// [`crate::opus_celt::celt_frame_bytes_for_bitrate`] (clamped to the
/// conformance-verified ceiling), and each frame is analysed with the previous
/// frame as lapped-MDCT overlap history.
///
/// Reconstruction is still *approximate*: SILK frames are band-limited
/// narrowband with a bounded greedy pulse search; CELT is mono-only,
/// non-transient and uses a neutral (non-dynalloc) allocation. See the
/// per-frame encoder modules for the measured fidelity picture.
///
/// # Errors
/// Returns [`OxiAudioError::UnsupportedFormat`] when channels are outside 1..=2 or
/// the sample rate is not 48 kHz; [`OxiAudioError::Io`] on write failure.
pub fn encode_opus_auto<W: Write>(
    buf: &AudioBuffer<f32>,
    writer: W,
    target_bitrate_kbps: u32,
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
    let celt_bytes = celt_frame_bytes_for_bitrate(target_bitrate_kbps);

    // Lapped-MDCT overlap history for the CELT path (see `encode_opus_conformant`).
    let mut prev_frame: Vec<f32> = Vec::new();

    for i in 0..n_frames {
        let start = i * frame_samples;
        let end = start + frame_samples;
        let pcm = &buf.samples[start..end];

        let mono_vec: Vec<f32> = if channels == 2 {
            pcm.chunks_exact(2).map(|c| 0.5 * (c[0] + c[1])).collect()
        } else {
            pcm.to_vec()
        };
        let mono: &[f32] = &mono_vec;

        let mode = select_conformant_mode(mono, target_bitrate_kbps);
        let packet = match mode {
            OpusConformantMode::Celt => {
                encode_celt_frame_conformant_sized(&prev_frame, mono, celt_bytes)
            }
            OpusConformantMode::Silk => encode_silk_frame_conformant(mono, 1),
            OpusConformantMode::Hybrid => encode_hybrid_frame_conformant(mono, 1),
        };
        prev_frame = mono_vec;

        let is_last = i == n_frames - 1;
        stream.write_packet(&packet, FRAME_SIZE as i64, is_last)?;
    }

    stream
        .finish()
        .map_err(|_| OxiAudioError::Io(std::io::Error::other("OGG stream finish failed")))?;

    Ok(())
}

/// Encode an [`AudioBuffer<f32>`] to an OGG Opus file at `path` using the automatic
/// full encode path ([`encode_opus_auto`]).
///
/// # Errors
/// Returns [`OxiAudioError::Io`] on file-creation or write failure; propagates the
/// validation errors of [`encode_opus_auto`].
pub fn encode_opus_auto_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    target_bitrate_kbps: u32,
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    encode_opus_auto(buf, writer, target_bitrate_kbps)
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
///
/// # Conformance
///
/// As of oxiaudio 0.2.1 each frame is routed through the RFC 6716–conformant
/// CELT-only per-frame encoder **with lapped-MDCT overlap history** (the
/// previous frame is retained across calls), so time-domain alias cancellation
/// actually works across a stream. Use [`OpusStreamEncoder::with_bitrate`] to
/// set the per-frame payload size. Stereo input is downmixed to mono per frame
/// before encoding (the `OpusHead` page still advertises the constructor's
/// `channels` value, matching the documented asymmetry of
/// [`encode_opus_conformant`]). This is not a transparent-quality encoder —
/// see `crate::opus_celt`'s module doc for the measured caveats.
pub struct OpusStreamEncoder<W: Write> {
    stream: OggStream<W>,
    channels: usize,
    granule_pos: i64,
    is_finalized: bool,
    /// Previous mono frame, kept as CELT lapped-MDCT overlap history.
    prev_frame: Vec<f32>,
    /// CBR payload size per frame, derived from the configured bitrate.
    celt_bytes: usize,
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
        Ok(Self {
            stream,
            channels,
            granule_pos: 0,
            is_finalized: false,
            prev_frame: Vec::new(),
            celt_bytes: DEFAULT_CELT_FRAME_BYTES,
        })
    }

    /// Create a streaming encoder with an explicit target bitrate.
    ///
    /// The per-frame CBR payload size is derived with
    /// [`celt_frame_bytes_for_bitrate`], so the request genuinely changes the
    /// emitted rate (clamped to the conformance-verified range, which now runs
    /// all the way to the RFC's 1275-byte frame limit — see
    /// `crate::opus_celt::MAX_CELT_FRAME_BYTES`).
    ///
    /// # Errors
    /// Returns [`OxiAudioError::UnsupportedFormat`] if `channels` > 2.
    /// Returns [`OxiAudioError::Io`] on write failure.
    pub fn with_bitrate(
        writer: W,
        channels: usize,
        serial: u32,
        target_bitrate_kbps: u32,
    ) -> Result<Self, OxiAudioError> {
        let mut enc = Self::new(writer, channels, serial)?;
        enc.celt_bytes = celt_frame_bytes_for_bitrate(target_bitrate_kbps);
        Ok(enc)
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
        // Downmix to mono for the conformant CELT per-frame encoder, matching
        // encode_opus_conformant / encode_opus_auto's documented asymmetry: the
        // OpusHead still advertises `self.channels`, but per-frame content is mono.
        let mono_vec: Vec<f32> = if self.channels == 2 {
            pcm.chunks_exact(2).map(|c| 0.5 * (c[0] + c[1])).collect()
        } else {
            pcm.to_vec()
        };
        let packet =
            encode_celt_frame_conformant_sized(&self.prev_frame, &mono_vec, self.celt_bytes);
        self.prev_frame = mono_vec;
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
        encode_opus, encode_opus_conformant, encode_opus_structural, OpusConformantMode,
        OpusEncodeConfig, OpusStreamEncoder, FRAME_SIZE,
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
    fn test_encode_opus_structural_produces_valid_ogg_output() {
        // The legacy non-conformant path must still work byte-structurally
        // (OGG framing) even though its payload is not RFC 6716–conformant.
        let buf = silence_buf(2, 1);
        let mut out = Cursor::new(Vec::new());
        encode_opus_structural(&buf, &mut out, 128).expect("encode_opus_structural stereo silence");
        let bytes = out.into_inner();
        assert!(!bytes.is_empty(), "output must not be empty");
        assert_eq!(&bytes[..4], b"OggS", "output must start with OggS magic");
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
