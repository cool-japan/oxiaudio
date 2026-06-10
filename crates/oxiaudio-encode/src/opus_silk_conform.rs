//! Minimal conformant SILK encoder — produces RFC 6716–decodable NB 20 ms packets.
//!
//! `encode_silk_frame_conformant` encodes a SILK-only NB 20 ms Opus packet that any
//! RFC 6716–compliant decoder (`opus-decoder` crate included) can decode without error.
//!
//! # Packet layout
//!
//! ```text
//! Byte 0: TOC = 0x08
//!           config 1 = SILK-only NB 20 ms
//!           stereo   = 0 (mono)
//!           code     = 0 (1-frame CBR, self-delimited)
//! Bytes 1..: range-coded SILK payload
//!
//!   SILK header (parse_header: 1 channel, 1 internal frame)
//!     VAD flag         = false  (1 bit, logp=1)
//!     has_lbrr flag    = false  (1 bit, logp=1)
//!
//!   decode_indices (CondCoding::Independently, vad=false)
//!     signal/offset type  TYPE_OFFSET_NO_VAD_ICDF  symbol 0 → inactive
//!     gain high byte      GAIN_ICDF[0]             symbol 0
//!     gain low byte       UNIFORM8_ICDF            symbol 0
//!     delta gain subfr 1  DELTA_GAIN_ICDF          symbol 0
//!     delta gain subfr 2  DELTA_GAIN_ICDF          symbol 0
//!     delta gain subfr 3  DELTA_GAIN_ICDF          symbol 0
//!     NLSF stage-1 idx    NLSF_CB1_ICDF_NB_MB      symbol 0  (cb1_index=0)
//!     NLSF stage-2 ×10    NLSF_CB2_ROW0_NB_MB      symbol 4  (neutral, amplitude=0)
//!     interp factor       NLSF_INTERP_FACTOR_ICDF  symbol 4  (no interpolation)
//!     seed                UNIFORM4_ICDF             symbol 0
//!
//!   decode_pulses (signal_type=0, quant_offset=0, frame_length=160)
//!     rate level          RATE_LEVELS_ICDF[0]       symbol 0
//!     block sum ×10       PULSES_PER_BLOCK_ICDF[0]  symbol 0  (0 pulses = silence)
//! ```
//!
//! # Signal quality
//!
//! This encoder always produces a silence-like (zero-excitation, inactive-signal) frame.
//! Full SILK encoding with LP analysis, NLSF VQ, noise shaping, and LTP coding is
//! deferred to a future run.

use crate::opus_range::RangeEncoder;

// ── SILK entropy tables ────────────────────────────────────────────────────────
//
// Copied verbatim from RFC 6716 Appendix B / libopus `silk/tables_other.c`
// (BSD-3-Clause, Xiph.Org Foundation / Microsoft).  These are the same tables
// used by the `opus-decoder` entropy decoder so the encoder/decoder pair is
// bit-exact by construction.

/// Signal-type + quant-offset iCDF for non-VAD frames.
///
/// Symbols: 0 = inactive (signal_type=0, quant_offset=0),
///          1 = inactive (signal_type=0, quant_offset=1).
const TYPE_OFFSET_NO_VAD_ICDF: [u8; 2] = [230, 0];

/// Gain coding iCDF for signal_type=0 (inactive/unvoiced), high 3-bit field.
const GAIN_ICDF_INACTIVE: [u8; 8] = [224, 112, 44, 15, 3, 2, 1, 0];

/// Uniform 8-way iCDF for gain low-byte field.
const UNIFORM8_ICDF: [u8; 8] = [224, 192, 160, 128, 96, 64, 32, 0];

/// Delta-gain iCDF — 41 entries, symbols 0..40.
const DELTA_GAIN_ICDF: [u8; 41] = [
    250, 245, 234, 203, 71, 50, 42, 38, 35, 33, 31, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
    17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];

/// NLSF stage-1 iCDF for NB/MB codebook.
///
/// 64 entries (two halves of 32):
///   entries [0..32]  = iCDF for inactive/unvoiced (cb1_offset=0)
///   entries [32..64] = iCDF for voiced             (cb1_offset=32)
const NLSF_CB1_ICDF_NB_MB: [u8; 64] = [
    212, 178, 148, 129, 108, 96, 85, 82, 79, 77, 61, 59, 57, 56, 51, 49, 48, 45, 42, 41, 40, 38,
    36, 34, 31, 30, 21, 12, 10, 3, 1, 0, 255, 245, 244, 236, 233, 225, 217, 203, 190, 176, 175,
    161, 149, 136, 125, 114, 102, 91, 81, 71, 60, 52, 43, 35, 28, 20, 19, 18, 12, 11, 5, 0,
];

/// NLSF stage-2 iCDF, row 0 of the NB/MB codebook (9 entries, symbols 0..8).
///
/// For cb1_index=0 with signal_type=0, `unpack_nlsf_ec_ix` maps all 10 NB
/// dimensions to ec_ix[i]=0, so they all use this row.
///
/// Symbol 4 represents amplitude = 4 − NLSF_QUANT_MAX_AMPLITUDE(4) = 0 (neutral).
const NLSF_CB2_ROW0_NB_MB: [u8; 9] = [255, 254, 253, 238, 14, 3, 2, 1, 0];

/// NLSF interpolation-factor iCDF (5 entries, symbols 0..4).
///
/// Symbol 4 → interp_coef_q2=4 → no interpolation (use current NLSFs directly).
/// This is the most probable symbol and the correct choice for a first frame.
const NLSF_INTERP_FACTOR_ICDF: [u8; 5] = [243, 221, 192, 181, 0];

/// Uniform 4-way iCDF for excitation seed.
const UNIFORM4_ICDF: [u8; 4] = [192, 128, 64, 0];

/// Pulse rate-level iCDF for signal_type=0 (inactive/unvoiced branch, index 0).
const RATE_LEVELS_ICDF_INACTIVE: [u8; 9] = [241, 190, 178, 132, 87, 74, 41, 14, 0];

/// Pulses-per-block iCDF for rate_level=0 (18 entries, symbols 0..17).
///
/// Symbol 0 → sum_pulses=0 → zero-excitation (silence), no shell decode.
const PULSES_PER_BLOCK_ICDF_LVL0: [u8; 18] = [
    125, 51, 26, 18, 15, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];

// ── WB (16 kHz) entropy tables for hybrid mode ───────────────────────────────
//
// These are needed for encoding the SILK layer in config=15 (Hybrid FB 20 ms)
// packets.  The decoder uses internal_fs_hz=16000 for configs 12–15, which
// requires 16th-order NLSFs and 20 shell-coded blocks per 20 ms frame.
//
// Copied verbatim from RFC 6716 Appendix B / libopus `silk/tables_NLSF_CB_WB.c`
// and `opus-decoder-0.1.1/src/silk/entropy_tables.rs` (BSD-3-Clause).

/// NLSF stage-1 iCDF for WB (16 kHz) codebook.
///
/// 64 entries (two halves of 32):
///   entries [0..32]  = iCDF for inactive/unvoiced (cb1_offset=0)
///   entries [32..64] = iCDF for voiced             (cb1_offset=32)
const NLSF_CB1_ICDF_WB: [u8; 64] = [
    225, 204, 201, 184, 183, 175, 158, 154, 153, 135, 119, 115, 113, 110, 109, 99, 98, 95, 79, 68,
    52, 50, 48, 45, 43, 32, 31, 27, 18, 10, 3, 0, 255, 251, 235, 230, 212, 201, 196, 182, 167, 166,
    163, 151, 138, 124, 110, 104, 90, 78, 76, 70, 69, 57, 45, 34, 24, 21, 11, 6, 5, 4, 3, 0,
];

/// NLSF stage-2 iCDF, row 0 of the WB codebook (9 entries, symbols 0..8).
///
/// For cb1_index=0, signal_type=0 (inactive), WB: `unpack_nlsf_ec_ix` maps all
/// 16 WB dimensions to ec_ix[i]=0 because `NLSF_CB2_SELECT_WB[0..8]` are
/// either 0 or 1, and bits 1..3 and 5..7 of each entry are zero → row index 0.
///
/// Symbol 4 → amplitude = 4 − NLSF_QUANT_MAX_AMPLITUDE(4) = 0 (neutral/zero).
const NLSF_CB2_ROW0_WB: [u8; 9] = [255, 254, 253, 244, 12, 3, 2, 1, 0];

// ── Frame geometry ─────────────────────────────────────────────────────────────

/// TOC byte: config 1 = SILK-only NB 20 ms, mono, 1-frame CBR.
///
/// Decoding: `(1 << 3) | (0 << 2) | 0 = 0x08`.
const TOC_NB_20MS_MONO: u8 = 0x08;

/// Number of subframes in a 20 ms NB SILK frame (4 × 5 ms subframes).
const NB_SUBFR: usize = 4;

/// Number of shell-codec blocks for a 160-sample NB frame: `160 / 16 = 10`.
const SHELL_BLOCKS_NB_20MS: usize = 10;

/// NLSF order (= LPC order) for NB (narrowband, 8 kHz): 10.
const NLSF_ORDER_NB: usize = 10;

/// NLSF order (= LPC order) for WB (wideband, 16 kHz): 16.
const NLSF_ORDER_WB: usize = 16;

/// Number of shell-codec blocks for a 320-sample WB 20 ms frame: `320 / 16 = 20`.
const SHELL_BLOCKS_WB_20MS: usize = 20;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Encode PCM as a SILK-mode Opus packet (NB 20 ms, unvoiced, zero excitation).
///
/// Returns a packet that any RFC 6716–compliant decoder can decode without error.
///
/// # Arguments
///
/// * `pcm` — 960 mono f32 samples at 48 kHz (20 ms). Shorter slices are accepted;
///   longer slices are truncated. The actual PCM content is currently not used —
///   the encoder always outputs a silence-like inactive frame.
/// * `_channels` — reserved for future stereo support; currently ignored.
///
/// # Packet structure
///
/// TOC byte `0x08` followed by a range-coded SILK payload encoding an inactive
/// (non-VAD), zero-excitation frame with the first NLSF stage-1 codebook entry
/// and neutral (zero-amplitude) stage-2 residuals.
pub fn encode_silk_frame_conformant(_pcm: &[f32], _channels: usize) -> Vec<u8> {
    let mut enc = RangeEncoder::new();

    // ── SILK header: parse_header (1 channel, 1 internal frame) ─────────────
    // VAD flag for internal frame 0.
    enc.enc_bit_logp(false, 1);
    // has_lbrr flag for channel 0.
    enc.enc_bit_logp(false, 1);

    // ── Side information: decode_indices (Independently, vad=false) ──────────

    // Signal type + quant-offset (no-VAD branch, TYPE_OFFSET_NO_VAD_ICDF).
    // Symbol 0 → ix=0 → signal_type=0 (inactive), quant_offset_type=0.
    enc.enc_icdf(0, &TYPE_OFFSET_NO_VAD_ICDF, 8);

    // Gain for subframe 0 (Independently coded = high byte + low byte).
    enc.enc_icdf(0, &GAIN_ICDF_INACTIVE, 8); // high 3-bit field (symbol 0)
    enc.enc_icdf(0, &UNIFORM8_ICDF, 8); // low 3-bit residual  (symbol 0)

    // Delta gains for subframes 1 .. NB_SUBFR-1 (3 additional subframes).
    for _ in 1..NB_SUBFR {
        enc.enc_icdf(0, &DELTA_GAIN_ICDF, 8);
    }

    // NLSF stage-1 codebook index.
    // signal_type=0 → cb1_offset = (0>>1)*32 = 0 → decode from NLSF_CB1_ICDF_NB_MB[0..].
    // Symbol 0 → cb1_index=0 (valid: cb.n_vectors=32, 0 < 32).
    enc.enc_icdf(0, &NLSF_CB1_ICDF_NB_MB, 8);

    // NLSF stage-2 residuals for all NLSF_ORDER_NB=10 dimensions.
    // cb1_index=0 → unpack_nlsf_ec_ix yields ec_ix[i]=0 for all i →
    // every dimension uses NLSF_CB2_ROW0_NB_MB (the first 9-byte row of the table).
    // Symbol 4 → stage2=4 → amplitude = 4 − NLSF_QUANT_MAX_AMPLITUDE(4) = 0 (neutral),
    // avoids the extension branches (stage2≠0 and stage2≠8).
    for _ in 0..NLSF_ORDER_NB {
        enc.enc_icdf(4, &NLSF_CB2_ROW0_NB_MB, 8);
    }

    // NLSF interpolation factor (decoded only when nb_subfr == MAX_NB_SUBFR=4,
    // which is true for 20 ms frames).
    // Symbol 4 → interp_coef_q2=4 → no interpolation (use current NLSFs directly).
    enc.enc_icdf(4, &NLSF_INTERP_FACTOR_ICDF, 8);

    // No voiced-mode parameters (signal_type=0 ≠ TYPE_VOICED=2).

    // Excitation randomisation seed (uniform 4-way).
    enc.enc_icdf(0, &UNIFORM4_ICDF, 8);

    // ── Pulse coding: decode_pulses (signal_type=0, frame_length=160) ─────────

    // Rate level for inactive/unvoiced branch: RATE_LEVELS_ICDF[signal_type>>1=0].
    enc.enc_icdf(0, &RATE_LEVELS_ICDF_INACTIVE, 8);

    // Pulse sums for SHELL_BLOCKS_NB_20MS=10 shell-coded blocks (frame_length=160/16=10).
    // Symbol 0 → sum_pulses=0 → silence: no shell decode, no LSB extension, no sign decode.
    for _ in 0..SHELL_BLOCKS_NB_20MS {
        enc.enc_icdf(0, &PULSES_PER_BLOCK_ICDF_LVL0, 8);
    }

    // Finalise range coder and prepend TOC byte.
    let payload = enc.finish();
    let mut packet = Vec::with_capacity(1 + payload.len());
    packet.push(TOC_NB_20MS_MONO);
    packet.extend_from_slice(&payload);
    packet
}

// ── Hybrid helper ─────────────────────────────────────────────────────────────

/// Encode a SILK WB silence frame (inactive, zero-excitation) into `enc`.
///
/// This is used by the hybrid encoder to share a single [`RangeEncoder`] between
/// the SILK and CELT layers without creating a separate packet. The sequence is
/// bit-exact with what the SILK decoder expects for config=15 (Hybrid FB 20 ms):
///
/// * 1 channel, 1 internal frame, `frame_length=320`, `nb_subfr=4`,
///   `NLSF_ORDER=16`, `shell_blocks=20`.
/// * Signal type = inactive (0), quant_offset = 0, zero gains,
///   NLSF cb1_index=0 (neutral), 16 × neutral stage-2 residuals,
///   no interpolation, seed=0, 20 zero-pulse blocks.
///
/// The caller is responsible for prepending the hybrid TOC byte (`0x78`)
/// and calling [`RangeEncoder::finish`] after also encoding the CELT layer.
pub(crate) fn encode_silk_wb_silence_into(enc: &mut RangeEncoder) {
    // ── SILK header: parse_header (1 channel, 1 internal frame) ────────────
    // VAD flag for internal frame 0 (no voice activity).
    enc.enc_bit_logp(false, 1);
    // has_lbrr flag for channel 0 (no redundancy).
    enc.enc_bit_logp(false, 1);

    // ── Side information: decode_indices (Independently, vad=false) ─────────

    // Signal type + quant-offset (no-VAD branch, TYPE_OFFSET_NO_VAD_ICDF).
    // Symbol 0 → ix=0 → signal_type=0 (inactive), quant_offset_type=0.
    enc.enc_icdf(0, &TYPE_OFFSET_NO_VAD_ICDF, 8);

    // Gain subframe 0 (Independently coded = high byte + low byte).
    enc.enc_icdf(0, &GAIN_ICDF_INACTIVE, 8); // high 3-bit field (symbol 0)
    enc.enc_icdf(0, &UNIFORM8_ICDF, 8); // low 3-bit residual  (symbol 0)

    // Delta gains for subframes 1..NB_SUBFR-1 (nb_subfr=4 → 3 additional).
    for _ in 1..NB_SUBFR {
        enc.enc_icdf(0, &DELTA_GAIN_ICDF, 8);
    }

    // NLSF stage-1 codebook index (WB, 64-symbol iCDF for 16 kHz configs).
    // signal_type=0 → cb1_offset = (0>>1)*32 = 0 → decode from NLSF_CB1_ICDF_WB[0..].
    // Symbol 0 → cb1_index=0 (valid: n_vectors=32, 0 < 32).
    enc.enc_icdf(0, &NLSF_CB1_ICDF_WB, 8);

    // NLSF stage-2 residuals for all NLSF_ORDER_WB=16 dimensions.
    // cb1_index=0 → unpack_nlsf_ec_ix yields ec_ix[i]=0 for all i in WB
    // (NLSF_CB2_SELECT_WB[0..8] = [0,0,0,0,0,0,0,1]; bits extracting row
    // index are zero for all 8 pairs → row 0 of NLSF_CB2_ICDF_WB).
    // Symbol 4 → amplitude = 4 − NLSF_QUANT_MAX_AMPLITUDE(4) = 0 (neutral),
    // avoids the extension branches (stage2 ≠ 0 and stage2 ≠ 8).
    for _ in 0..NLSF_ORDER_WB {
        enc.enc_icdf(4, &NLSF_CB2_ROW0_WB, 8);
    }

    // NLSF interpolation factor (decoded only when nb_subfr == MAX_NB_SUBFR=4).
    // Symbol 4 → interp_coef_q2=4 → no interpolation (use current NLSFs directly).
    enc.enc_icdf(4, &NLSF_INTERP_FACTOR_ICDF, 8);

    // No voiced-mode parameters (signal_type=0 ≠ TYPE_VOICED=2).

    // Excitation randomisation seed (uniform 4-way).
    enc.enc_icdf(0, &UNIFORM4_ICDF, 8);

    // ── Pulse coding: decode_pulses (signal_type=0, frame_length=320) ────────

    // Rate level for inactive/unvoiced branch: RATE_LEVELS_ICDF[signal_type>>1=0].
    enc.enc_icdf(0, &RATE_LEVELS_ICDF_INACTIVE, 8);

    // Pulse sums for SHELL_BLOCKS_WB_20MS=20 shell-coded blocks (320/16=20).
    // Symbol 0 → sum_pulses=0 → silence: no shell decode, no LSB extension.
    for _ in 0..SHELL_BLOCKS_WB_20MS {
        enc.enc_icdf(0, &PULSES_PER_BLOCK_ICDF_LVL0, 8);
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Produce 960 mono samples of a 1 kHz sine at amplitude 0.5, 48 kHz.
    fn sine_1khz() -> Vec<f32> {
        (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect()
    }

    #[test]
    fn toc_byte_is_correct() {
        let packet = encode_silk_frame_conformant(&sine_1khz(), 1);
        assert!(!packet.is_empty(), "packet must be non-empty");
        assert_eq!(
            packet[0], TOC_NB_20MS_MONO,
            "TOC must be 0x{:02X} (config 1 = NB 20ms mono), got 0x{:02X}",
            TOC_NB_20MS_MONO, packet[0]
        );
        let config = (packet[0] >> 3) & 0x1F;
        assert!(
            config <= 11,
            "TOC config {config} must be in SILK range 0..=11"
        );
    }

    #[test]
    fn packet_is_non_trivial() {
        let packet = encode_silk_frame_conformant(&sine_1khz(), 1);
        // Must have TOC + at least a few range-coder bytes.
        assert!(
            packet.len() >= 3,
            "packet must have TOC + at least 2 payload bytes, got {} bytes",
            packet.len()
        );
    }

    #[test]
    fn silence_and_sine_produce_equal_structure() {
        // Since PCM content is not yet used, both should produce identical packets.
        let p_sine = encode_silk_frame_conformant(&sine_1khz(), 1);
        let p_silence = encode_silk_frame_conformant(&vec![0.0f32; 960], 1);
        assert_eq!(
            p_sine, p_silence,
            "minimal encoder produces PCM-independent packets (signal quality deferred)"
        );
    }
}
