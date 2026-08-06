//! RFC 6716–conformant Opus hybrid encoder: SILK WB low-band + CELT high-band.
//!
//! CELT hybrid layer uses `start_band=17` and omits silence flag — decoder returns `Ok(960)`.
//!
//! Hybrid mode (configs 12–15) splits audio across a SILK LP layer for low
//! frequencies and a CELT layer for high frequencies. Unlike SILK-only and
//! CELT-only modes, both layers share a **single** range-coder bitstream; the
//! SILK decoder reads forward symbols first, then the CELT decoder continues
//! from the same entropy-coder position.
//!
//! # Config 15 — Hybrid FB 20 ms
//!
//! ```text
//! TOC = 0x78: (15 << 3) | stereo=0 | code=0
//!   config=15 → Hybrid FB (48 kHz output), 20 ms frame
//!   code=0    → single-frame CBR packet
//! ```
//!
//! # Shared range-coder layout
//!
//! ```text
//! packet = [TOC=0x78 | range-coded bytes]
//!            ↑ 1 byte  ↑ shared ec stream (exactly `target_bytes` long)
//!
//! shared ec stream:
//!   [SILK WB silence (inactive, frame_length=320, 16 kHz)]
//!   [redundancy flag, logp=12 — always "no redundancy"]
//!   [CELT high-band frame (start_band=17, no silence flag)]
//! ```
//!
//! # Scope limitation — the low band is silence
//!
//! The SILK layer here is `encode_silk_wb_silence_into`: a conformant but
//! **inactive** wideband frame. The real analysis-by-synthesis SILK encoder
//! (`crate::opus_silk_encode`) is narrowband-only, so hybrid mode currently
//! carries no low-frequency audio at all — only the CELT high band (17–20,
//! i.e. 8–20 kHz) is real. `select_conformant_mode` therefore never chooses
//! [`crate::opus_encoder::OpusConformantMode::Hybrid`]; it is reachable only by
//! explicit caller request. Wiring a wideband SILK encoder in (16 kHz internal
//! rate, order-16 NLSF codebooks, 20 shell blocks) is tracked as remaining work.
//!
//! # Conformance
//!
//! The CELT layer is encoded via `encode_celt_hybrid_layer_into`, which uses
//! `start_band = 17` and omits the silence flag. This matches what the hybrid CELT
//! decoder expects: after SILK decodes, `ec.tell() > 1`, so the decoder falls into
//! the `else { false }` branch of the silence check, and post-filter is skipped
//! because `start != 0`. The shared [`RangeEncoder`] is correctly positioned
//! for the decoder to continue reading CELT symbols immediately after SILK.
//!
//! Two things had to be fixed in oxiaudio 0.2.1 before that claim was true, and
//! both were invisible to a "does it decode?" test — the packets decoded to 960
//! samples the whole time, they just decoded *different symbols*:
//!
//! * **The redundancy flag was missing.** Between the SILK and CELT layers a
//!   hybrid decoder reads a one-bit flag (`logp = 12`, guarded by
//!   `tell + 37 <= total_bits`) saying whether the packet carries a redundant
//!   0–8 kHz CELT frame. Not writing it shifted every following CELT symbol by
//!   one bit position, so the encoder's and decoder's `final_range` registers
//!   never matched at any frame size.
//! * **The CELT layer's bit budget was a constant.** It was written against a
//!   hardcoded 512 bits while the packet was assembled with a variable-length
//!   `finish()`, so the decoder's `total_bits = payload_len · 8` only matched by
//!   coincidence. The packet is now emitted as exact CBR at `target_bytes` and
//!   the layer is given `target_bytes · 8`.
//!
//! `tests/m_opus_celt_snr.rs::hybrid_packet_layer_budget_is_consistent` pins
//! `final_range` equality against the reference decoder across the whole
//! `[MIN_HYBRID_FRAME_BYTES, MAX_HYBRID_FRAME_BYTES]` range.

use crate::opus_range::RangeEncoder;
use crate::opus_silk_conform::encode_silk_wb_silence_into;

/// TOC byte for Hybrid FB 20 ms mono code=0.
///
/// Bit layout (RFC 6716 §3.1):
/// - Bits 7–3: config = 15 (Hybrid Fullband 20 ms)
/// - Bit 2: stereo = 0 (mono)
/// - Bits 1–0: frame count code = 0 (one CBR frame)
///
/// Value: `(15 << 3) | (0 << 2) | 0 = 0x78`.
const TOC_HYBRID_FB_20MS_MONO: u8 = 0x78;

/// Encode 960 mono (or interleaved stereo) samples as an Opus Hybrid FB 20 ms packet.
///
/// Produces an RFC 6716–conformant hybrid Opus packet with TOC byte `0x78`
/// (config 15 = Hybrid Fullband 20 ms, mono, single-frame CBR).
///
/// The packet contains a shared range-coder stream with:
/// 1. A **SILK WB silence layer** — inactive, zero-excitation WB frame at
///    16 kHz internal rate (4 subframes, 16th-order NLSF, 20 pulse blocks).
/// 2. A **CELT high-band layer** — encoded via `encode_celt_hybrid_layer_into`
///    with `start_band = 17`, no silence flag; covers bands 17–20 only.
///
/// # Arguments
///
/// * `pcm` — interleaved f32 PCM samples (960 mono or 1920 stereo).
///   Only the first 960 samples are used for CELT analysis.
/// * `channels` — 1 (mono) or 2 (stereo); currently unused (CELT is always mono).
///
/// # Returns
///
/// A `Vec<u8>` containing the complete Opus packet: `[TOC, range_coder_bytes…]`.
///
/// # Conformance
///
/// The TOC byte and packet structure match RFC 6716 §3.1 and §3.2. The SILK
/// silence layer is bit-exact with the decoder's expectations for config=15.
/// The CELT high-band layer uses `start_band=17` and omits the silence flag,
/// matching the hybrid CELT decoder which skips the silence-flag branch when
/// `tell > 1` (already consumed by SILK) and skips post-filter when `start != 0`.
pub fn encode_hybrid_frame_conformant(pcm: &[f32], channels: usize) -> Vec<u8> {
    encode_hybrid_frame_conformant_sized(pcm, channels, DEFAULT_HYBRID_FRAME_BYTES).0
}

/// Default hybrid payload size in bytes per 20 ms frame (≈25.6 kbps).
///
/// Matches the CELT-only default so the two paths are directly comparable.
pub const DEFAULT_HYBRID_FRAME_BYTES: usize = 64;

/// Smallest hybrid payload the writer will emit.
///
/// The SILK WB low band alone costs ~10 bytes of side information before any
/// CELT symbol is written, so anything below this cannot hold a well-formed
/// two-layer frame.
pub const MIN_HYBRID_FRAME_BYTES: usize = 20;

/// Largest hybrid payload the writer will emit (RFC 6716 §3.2.1 frame limit).
pub const MAX_HYBRID_FRAME_BYTES: usize = 1275;

/// [`encode_hybrid_frame_conformant`] at an explicit CBR payload size.
///
/// Returns `(packet, final_range)`. `target_bytes` is the payload length in
/// bytes **excluding** the TOC byte and is clamped to
/// `[MIN_HYBRID_FRAME_BYTES, MAX_HYBRID_FRAME_BYTES]`; the emitted payload is
/// exactly that many bytes, which is what makes the CELT layer's bit budget
/// (`target_bytes * 8`) equal to the `total_bits` the decoder derives from the
/// packet length.
///
/// As on the CELT-only path, a frame whose range-coded (front) and raw-bit
/// (back) halves collide is retried one byte smaller rather than shipped with
/// dropped bytes — the decoder re-derives its budget from the emitted length,
/// so a smaller frame is simply a lower-rate, still exactly conformant encode.
pub fn encode_hybrid_frame_conformant_sized(
    pcm: &[f32],
    channels: usize,
    target_bytes: usize,
) -> (Vec<u8>, u32) {
    let _ = channels; // currently unused; CELT layer is always mono
    let mut bytes = target_bytes.clamp(MIN_HYBRID_FRAME_BYTES, MAX_HYBRID_FRAME_BYTES);
    loop {
        let (packet, range, fits) = encode_hybrid_try(pcm, bytes);
        if fits || bytes <= MIN_HYBRID_FRAME_BYTES {
            return (packet, range);
        }
        bytes -= 1;
    }
}

/// One hybrid encode attempt at an exact payload size.
fn encode_hybrid_try(pcm: &[f32], target_bytes: usize) -> (Vec<u8>, u32, bool) {
    let mut enc = RangeEncoder::new();

    // ── SILK WB silence layer ─────────────────────────────────────────────────
    // Encodes a conformant WB (16 kHz) inactive SILK frame into the shared
    // range coder.  The decoder (config=15) calls silk_internal_fs_hz(15)=16000,
    // so it expects WB tables (order=16, 20 shell blocks, 320-sample frame).
    encode_silk_wb_silence_into(&mut enc);

    // ── Redundancy flag (RFC 6716 §4.5, hybrid only) ─────────────────────────
    // Between the SILK and CELT layers a hybrid decoder reads a one-bit
    // "packet carries a redundant 0–8 kHz CELT frame" flag with `logp = 12`,
    // guarded by `tell + 17 + 20 <= total_bits` (17 bits for the redundancy
    // header it would introduce, 20 for the smallest useful redundant frame).
    // Omitting it — which this encoder did before oxiaudio 0.2.1 — shifts every
    // subsequent CELT symbol by one bit position, so the packet still decoded
    // to 960 samples but the decoder's `final_range` never matched the
    // encoder's. We always signal "no redundancy".
    let total_bits = (target_bytes as i32).saturating_mul(8);
    if enc.tell() + 17 + 20 <= total_bits {
        enc.enc_bit_logp(false, 12);
    }

    // ── CELT high-band layer (start_band=17, no silence flag) ────────────────
    // Continues encoding into the SAME range coder.  The hybrid CELT decoder
    // sets start_band=17 and reads bands 17–20 from wherever SILK left off.
    // encode_celt_hybrid_layer_into omits the silence flag (which the decoder
    // skips in hybrid mode because tell > 1 after SILK) and post-filter flag
    // (skipped because start_band != 0), then encodes bands 17–20 only.
    let mut trace = crate::opus_celt::CeltEncodeTrace::default();
    crate::opus_celt::encode_celt_hybrid_layer_into(pcm, target_bytes, &mut enc, &mut trace);

    // ── Assemble packet ───────────────────────────────────────────────────────
    // CBR: `finish_to_size_checked` pads to exactly `target_bytes`, so the
    // decoder's `total_bits = payload_len * 8` equals the budget the CELT layer
    // was written against.
    let final_range = enc.final_range();
    let (payload, fits) = enc.finish_to_size_checked(target_bytes);
    let mut packet = Vec::with_capacity(1 + payload.len());
    packet.push(TOC_HYBRID_FB_20MS_MONO);
    packet.extend_from_slice(&payload);
    (packet, final_range, fits)
}

// ── Unit tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_440hz() -> Vec<f32> {
        (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.3)
            .collect()
    }

    #[test]
    fn hybrid_conform_toc_is_config_15() {
        let pcm = sine_440hz();
        let packet = encode_hybrid_frame_conformant(&pcm, 1);
        assert!(!packet.is_empty(), "packet must be non-empty");
        assert_eq!(
            packet[0], TOC_HYBRID_FB_20MS_MONO,
            "TOC must be 0x78 (config 15 = Hybrid FB 20 ms mono), got 0x{:02X}",
            packet[0]
        );
        let config = (packet[0] >> 3) & 0x1F;
        assert_eq!(config, 15, "config field must be 15 for Hybrid FB 20 ms");
    }

    #[test]
    fn hybrid_conform_packet_non_trivial() {
        let pcm = sine_440hz();
        let packet = encode_hybrid_frame_conformant(&pcm, 1);
        assert!(
            packet.len() >= 4,
            "packet must have TOC + at least 3 payload bytes, got {}",
            packet.len()
        );
    }

    #[test]
    fn hybrid_conform_silence_non_empty() {
        let pcm = vec![0.0f32; 960];
        let packet = encode_hybrid_frame_conformant(&pcm, 1);
        assert!(
            !packet.is_empty(),
            "silence hybrid packet must be non-empty"
        );
        assert_eq!(
            packet[0], TOC_HYBRID_FB_20MS_MONO,
            "silence TOC must be 0x78"
        );
    }

    #[test]
    fn hybrid_conform_deterministic() {
        let pcm = sine_440hz();
        let p1 = encode_hybrid_frame_conformant(&pcm, 1);
        let p2 = encode_hybrid_frame_conformant(&pcm, 1);
        assert_eq!(
            p1, p2,
            "repeated encoding of same PCM must be deterministic"
        );
    }
}
