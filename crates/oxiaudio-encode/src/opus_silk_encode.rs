//! Real SILK narrowband (NB, 8 kHz) encoder — RFC 6716 §5.
//!
//! This module produces a genuine, non-silent SILK-only NB 20 ms Opus packet
//! that any RFC 6716–compliant decoder (`opus-decoder` included) reconstructs to
//! an approximation of the input speech-band signal. Unlike the earlier
//! silence-only stub, it performs real LP analysis, quantizes the resulting
//! parameters through the exact codebooks the decoder inverts, and codes a real
//! excitation (shell-coded pulses) obtained by analysis-by-synthesis.
//!
//! # Pipeline
//!
//! 1. Down-sample the 48 kHz frame to 160 samples at 8 kHz (SILK NB internal
//!    rate) and convert to `i16` PCM (the decoder's internal domain).
//! 2. LP analysis (autocorrelation + Levinson-Durbin + LPC→NLSF), reusing
//!    [`crate::opus_silk::analyze_silk_frame`].
//! 3. NLSF two-stage VQ quantization — stage-1 codebook search plus predictive
//!    stage-2 residual quantization — reproducing the decoder's `nlsf_decode`
//!    bit-for-bit so the encoder and decoder share identical Q12 LPC coefficients.
//! 4. Gain selection (log-domain, 6 dB grid) sized so the excitation fits the
//!    shell coder without LSB extension.
//! 5. Analysis-by-synthesis excitation quantization: the exact fixed-point SILK
//!    synthesis loop is run in the encoder, and per-sample pulses (magnitude +
//!    sign, with the LCG dither accounted for) are chosen to track the target.
//! 6. Entropy coding of side information and pulses via [`RangeEncoder`], in the
//!    precise order the decoder consumes them.
//!
//! # Signal type
//!
//! Frames are coded as **unvoiced** (`signal_type = 1`, VAD active), which uses
//! short-term LPC prediction only (no LTP). This is appropriate for the general
//! speech-band content the round-trip tests exercise and keeps the excitation
//! model exact and stable.
//!
//! # Conformance limitations (honestly documented / measured)
//!
//! * **Decodability is exact.** Every field is verified bit-for-bit against the
//!   reference `opus-decoder` (see `tests/silk_internal_roundtrip.rs`, gated by
//!   the `silk_debug` feature): the decoded side-info, NLSF indices, per-block
//!   pulse sums, shell magnitudes and signs all match what the encoder wrote,
//!   and the reconstructed Q15 NLSFs / Q12 LPC match the decoder's own
//!   `nlsf_decode`/`nlsf2a` exactly.
//! * **Fidelity is approximate, not transparent.** Measured against the
//!   reference decoder on 20 ms sine tones (200 Hz–3 kHz), decoded output is
//!   clip-free and positively correlated with the input (best-lag Pearson
//!   correlation ≈ 0.33–0.67). This is a real, audible reconstruction — not a
//!   silence stub — but it is well short of a transparent SILK encoder: there is
//!   no perceptual noise shaping beyond the closed-loop quantizer, and the pulse
//!   search is a bounded greedy one, not rate-distortion optimal.
//! * Only the **narrowband** internal rate and **unvoiced** signal type are
//!   emitted; voiced/LTP coding and MB/WB/hybrid paths are out of scope here.
//!   Coding tonal/voiced content as unvoiced NB is itself lossy.
//! * A **bandwidth-expansion** step widens the quantized NLSFs toward a flat
//!   spectrum whenever the synthesis filter would be too resonant, because the
//!   decoder's mandatory excitation-offset dither otherwise resonates to
//!   clipping. This trades spectral sharpness for robustness.
//! * Frames are coded **independently** (no inter-frame gain/NLSF/LPC-state
//!   prediction). A single decoded packet is exact; in a multi-frame stream the
//!   decoder's carried state (gain floor, LPC history) diverges from the
//!   encoder's assumed reset state, so later frames lose some fidelity. The
//!   bitstream remains fully decodable throughout.
//!
//! Note: `RangeEncoder::enc_bit_logp` writes its argument verbatim as of
//! oxiaudio 0.2.1. Before that it emitted the logical complement (the stream
//! stayed byte-synchronized, so the defect was invisible to "does it decode?"
//! tests), and the SILK header flags here were written complemented to
//! compensate. Both sides are now correct and pinned by
//! `opus_range_dec::tests::roundtrip_bit_logp_is_not_inverted`.

use crate::opus_range::RangeEncoder;

// ── Frame geometry ──────────────────────────────────────────────────────────

/// TOC byte: config 1 = SILK-only NB 20 ms, mono, code 0 (single frame).
const TOC_NB_20MS_MONO: u8 = 0x08;
/// Internal NB frame length in samples (160 = 20 ms @ 8 kHz).
const FRAME_LEN: usize = 160;
/// Number of 5 ms subframes in a 20 ms frame.
const NB_SUBFR: usize = 4;
/// LPC order for NB.
const ORDER: usize = 10;
/// Shell-codec block length in samples.
const SHELL_LEN: usize = 16;
/// Number of shell blocks in a 160-sample frame.
const SHELL_BLOCKS: usize = FRAME_LEN / SHELL_LEN;
/// Down-sampling ratio 48 kHz → 8 kHz.
const DECIM: usize = 6;

// ── SILK constants (mirrors of the decoder) ─────────────────────────────────

const QUANT_LEVEL_ADJUST_Q10: i32 = 80;
const OFFSET_UVL_Q10: i32 = 100;
const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;
const NLSF_QUANT_LEVEL_ADJ_Q10: i32 = 102;
const NLSF_STAGE2_ROW: usize = (2 * NLSF_QUANT_MAX_AMPLITUDE + 1) as usize;
const QUANT_STEP_Q16_NB: i32 = 11_796;
const RAND_MULTIPLIER: i32 = 196_314_165;
const RAND_INCREMENT: i32 = 907_633_515;
const SILK_MAX_PULSES: i32 = 16;
const N_RATE_LEVELS: usize = 10;
/// Internal-rate peak target (i16); leaves headroom for resampler overshoot.
const PEAK_TARGET: i32 = 13_000;
/// Minimum inverse prediction gain (Q30) required of the quantized LPC.
///
/// Larger = flatter/less resonant. `≈ 2^26` caps the short-term prediction gain
/// near 16×, keeping the decoder's excitation-offset dither from resonating to
/// clipping. (`nlsf2a`'s own stability floor is only `≈ 107374`, i.e. ~2^-3×,
/// which permits far too resonant filters for a robust encoder.)
const MIN_INV_PRED_GAIN_Q30: i32 = 1 << 26;

// Gain dequantization constants (mirror `silk/gain.rs`).
const MIN_QGAIN_DB: i32 = 2;
const MAX_QGAIN_DB: i32 = 88;
const N_LEVELS_QGAIN: i32 = 64;
const GAIN_OFFSET: i32 = ((MIN_QGAIN_DB * 128) / 6) + 16 * 128;
const GAIN_INV_SCALE_Q16: i32 =
    (65_536 * (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6)) / (N_LEVELS_QGAIN - 1);

// ── Entropy tables (verbatim from RFC 6716 App. B / libopus, BSD-3-Clause) ──

/// VAD-aware signal type + quant-offset iCDF.
const TYPE_OFFSET_VAD_ICDF: [u8; 4] = [232, 158, 10, 0];
/// Gain iCDF for signal_type = 1 (unvoiced), high 3-bit field.
const GAIN_ICDF_UNVOICED: [u8; 8] = [254, 237, 192, 132, 70, 23, 4, 0];
/// Uniform 8-way iCDF (gain low field).
const UNIFORM8_ICDF: [u8; 8] = [224, 192, 160, 128, 96, 64, 32, 0];
/// Delta-gain iCDF (41 symbols).
const DELTA_GAIN_ICDF: [u8; 41] = [
    250, 245, 234, 203, 71, 50, 42, 38, 35, 33, 31, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
    17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];
/// Symbol whose delta maps to `ind_tmp = 0` (constant gain across subframes).
const DELTA_GAIN_NEUTRAL: usize = 4;
/// NLSF interpolation-factor iCDF (symbol 4 = no interpolation).
const NLSF_INTERP_FACTOR_ICDF: [u8; 5] = [243, 221, 192, 181, 0];
/// Uniform 4-way iCDF (excitation seed).
const UNIFORM4_ICDF: [u8; 4] = [192, 128, 64, 0];
/// Rate-level iCDF for the unvoiced/inactive branch (`signal_type >> 1 == 0`).
const RATE_LEVELS_ICDF_UV: [u8; 9] = [241, 190, 178, 132, 87, 74, 41, 14, 0];
/// Pulses-per-block iCDF bank, rows indexed by rate level.
const PULSES_PER_BLOCK_ICDF: [[u8; 18]; 10] = [
    [
        125, 51, 26, 18, 15, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    ],
    [
        198, 105, 45, 22, 15, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    ],
    [
        213, 162, 116, 83, 59, 43, 32, 24, 18, 15, 12, 9, 7, 6, 5, 3, 2, 0,
    ],
    [
        239, 187, 116, 59, 28, 16, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    ],
    [
        250, 229, 188, 135, 86, 51, 30, 19, 13, 10, 8, 6, 5, 4, 3, 2, 1, 0,
    ],
    [
        249, 235, 213, 185, 156, 128, 103, 83, 66, 53, 42, 33, 26, 21, 17, 13, 10, 0,
    ],
    [
        254, 249, 235, 206, 164, 118, 77, 46, 27, 16, 10, 7, 5, 4, 3, 2, 1, 0,
    ],
    [
        255, 253, 249, 239, 220, 191, 156, 119, 85, 57, 37, 23, 15, 10, 6, 4, 2, 0,
    ],
    [
        255, 253, 251, 246, 237, 223, 203, 179, 152, 124, 98, 75, 55, 40, 29, 21, 15, 0,
    ],
    [
        255, 254, 253, 247, 220, 162, 106, 67, 42, 28, 18, 12, 9, 6, 4, 3, 2, 0,
    ],
];
const SHELL_CODE_TABLE0: [u8; 152] = [
    128, 0, 214, 42, 0, 235, 128, 21, 0, 244, 184, 72, 11, 0, 248, 214, 128, 42, 7, 0, 248, 225,
    170, 80, 25, 5, 0, 251, 236, 198, 126, 54, 18, 3, 0, 250, 238, 211, 159, 82, 35, 15, 5, 0, 250,
    231, 203, 168, 128, 88, 53, 25, 6, 0, 252, 238, 216, 185, 148, 108, 71, 40, 18, 4, 0, 253, 243,
    225, 199, 166, 128, 90, 57, 31, 13, 3, 0, 254, 246, 233, 212, 183, 147, 109, 73, 44, 23, 10, 2,
    0, 255, 250, 240, 223, 198, 166, 128, 90, 58, 33, 16, 6, 1, 0, 255, 251, 244, 231, 210, 181,
    146, 110, 75, 46, 25, 12, 5, 1, 0, 255, 253, 248, 238, 221, 196, 164, 128, 92, 60, 35, 18, 8,
    3, 1, 0, 255, 253, 249, 242, 229, 208, 180, 146, 110, 76, 48, 27, 14, 7, 3, 1, 0,
];
const SHELL_CODE_TABLE1: [u8; 152] = [
    129, 0, 207, 50, 0, 236, 129, 20, 0, 245, 185, 72, 10, 0, 249, 213, 129, 42, 6, 0, 250, 226,
    169, 87, 27, 4, 0, 251, 233, 194, 130, 62, 20, 4, 0, 250, 236, 207, 160, 99, 47, 17, 3, 0, 255,
    240, 217, 182, 131, 81, 41, 11, 1, 0, 255, 254, 233, 201, 159, 107, 61, 20, 2, 1, 0, 255, 249,
    233, 206, 170, 128, 86, 50, 23, 7, 1, 0, 255, 250, 238, 217, 186, 148, 108, 70, 39, 18, 6, 1,
    0, 255, 252, 243, 226, 200, 166, 128, 90, 56, 30, 13, 4, 1, 0, 255, 252, 245, 231, 209, 180,
    146, 110, 76, 47, 25, 11, 4, 1, 0, 255, 253, 248, 237, 219, 194, 163, 128, 93, 62, 37, 19, 8,
    3, 1, 0, 255, 254, 250, 241, 226, 205, 177, 145, 111, 79, 51, 30, 15, 6, 2, 1, 0,
];
const SHELL_CODE_TABLE2: [u8; 152] = [
    129, 0, 203, 54, 0, 234, 129, 23, 0, 245, 184, 73, 10, 0, 250, 215, 129, 41, 5, 0, 252, 232,
    173, 86, 24, 3, 0, 253, 240, 200, 129, 56, 15, 2, 0, 253, 244, 217, 164, 94, 38, 10, 1, 0, 253,
    245, 226, 189, 132, 71, 27, 7, 1, 0, 253, 246, 231, 203, 159, 105, 56, 23, 6, 1, 0, 255, 248,
    235, 213, 179, 133, 85, 47, 19, 5, 1, 0, 255, 254, 243, 221, 194, 159, 117, 70, 37, 12, 2, 1,
    0, 255, 254, 248, 234, 208, 171, 128, 85, 48, 22, 8, 2, 1, 0, 255, 254, 250, 240, 220, 189,
    149, 107, 67, 36, 16, 6, 2, 1, 0, 255, 254, 251, 243, 227, 201, 166, 128, 90, 55, 29, 13, 5, 2,
    1, 0, 255, 254, 252, 246, 234, 213, 183, 147, 109, 73, 43, 22, 10, 4, 2, 1, 0,
];
const SHELL_CODE_TABLE3: [u8; 152] = [
    130, 0, 200, 58, 0, 231, 130, 26, 0, 244, 184, 76, 12, 0, 249, 214, 130, 43, 6, 0, 252, 232,
    173, 87, 24, 3, 0, 253, 241, 203, 131, 56, 14, 2, 0, 254, 246, 221, 167, 94, 35, 8, 1, 0, 254,
    249, 232, 193, 130, 65, 23, 5, 1, 0, 255, 251, 239, 211, 162, 99, 45, 15, 4, 1, 0, 255, 251,
    243, 223, 186, 131, 74, 33, 11, 3, 1, 0, 255, 252, 245, 230, 202, 158, 105, 57, 24, 8, 2, 1, 0,
    255, 253, 247, 235, 214, 179, 132, 84, 44, 19, 7, 2, 1, 0, 255, 254, 250, 240, 223, 196, 159,
    112, 69, 36, 15, 6, 2, 1, 0, 255, 254, 253, 245, 231, 209, 176, 136, 93, 55, 27, 11, 3, 2, 1,
    0, 255, 254, 253, 252, 239, 221, 194, 158, 117, 76, 42, 18, 4, 3, 2, 1, 0,
];
const SHELL_CODE_TABLE_OFFSETS: [u8; 17] = [
    0, 0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119, 135,
];
const SIGN_ICDF: [u8; 42] = [
    254, 49, 67, 77, 82, 93, 99, 198, 11, 18, 24, 31, 36, 45, 255, 46, 66, 78, 87, 94, 104, 208,
    14, 21, 32, 42, 51, 66, 255, 94, 104, 109, 112, 115, 118, 248, 53, 69, 80, 88, 95, 102,
];

/// NB/MB stage-1 NLSF iCDF (first 32 symbols = inactive/unvoiced branch).
const NLSF_CB1_ICDF_NB_MB: [u8; 64] = [
    212, 178, 148, 129, 108, 96, 85, 82, 79, 77, 61, 59, 57, 56, 51, 49, 48, 45, 42, 41, 40, 38,
    36, 34, 31, 30, 21, 12, 10, 3, 1, 0, 255, 245, 244, 236, 233, 225, 217, 203, 190, 176, 175,
    161, 149, 136, 125, 114, 102, 91, 81, 71, 60, 52, 43, 35, 28, 20, 19, 18, 12, 11, 5, 0,
];
/// NB/MB per-dimension stage-2 selector map.
const NLSF_CB2_SELECT_NB_MB: [u8; 160] = [
    16, 0, 0, 0, 0, 99, 66, 36, 36, 34, 36, 34, 34, 34, 34, 83, 69, 36, 52, 34, 116, 102, 70, 68,
    68, 176, 102, 68, 68, 34, 65, 85, 68, 84, 36, 116, 141, 152, 139, 170, 132, 187, 184, 216, 137,
    132, 249, 168, 185, 139, 104, 102, 100, 68, 68, 178, 218, 185, 185, 170, 244, 216, 187, 187,
    170, 244, 187, 187, 219, 138, 103, 155, 184, 185, 137, 116, 183, 155, 152, 136, 132, 217, 184,
    184, 170, 164, 217, 171, 155, 139, 244, 169, 184, 185, 170, 164, 216, 223, 218, 138, 214, 143,
    188, 218, 168, 244, 141, 136, 155, 170, 168, 138, 220, 219, 139, 164, 219, 202, 216, 137, 168,
    186, 246, 185, 139, 116, 185, 219, 185, 138, 100, 100, 134, 100, 102, 34, 68, 68, 100, 68, 168,
    203, 221, 218, 168, 167, 154, 136, 104, 70, 164, 246, 171, 137, 139, 137, 155, 218, 219, 139,
];
/// NB/MB stage-2 residual iCDF bank (8 rows × 9 symbols).
const NLSF_CB2_ICDF_NB_MB: [u8; 72] = [
    255, 254, 253, 238, 14, 3, 2, 1, 0, 255, 254, 252, 218, 35, 3, 2, 1, 0, 255, 254, 250, 208, 59,
    4, 2, 1, 0, 255, 254, 246, 194, 71, 10, 2, 1, 0, 255, 252, 236, 183, 82, 8, 2, 1, 0, 255, 252,
    235, 180, 90, 17, 2, 1, 0, 255, 248, 224, 171, 97, 30, 4, 1, 0, 255, 254, 236, 173, 95, 37, 7,
    1, 0,
];

// ── NLSF/LPC codebook data (NB/MB, from libopus `silk/tables_NLSF_CB_NB_MB.c`) ─

const NLSF_CB1_NB_MB_Q8: [u8; 320] = [
    12, 35, 60, 83, 108, 132, 157, 180, 206, 228, 15, 32, 55, 77, 101, 125, 151, 175, 201, 225, 19,
    42, 66, 89, 114, 137, 162, 184, 209, 230, 12, 25, 50, 72, 97, 120, 147, 172, 200, 223, 26, 44,
    69, 90, 114, 135, 159, 180, 205, 225, 13, 22, 53, 80, 106, 130, 156, 180, 205, 228, 15, 25, 44,
    64, 90, 115, 142, 168, 196, 222, 19, 24, 62, 82, 100, 120, 145, 168, 190, 214, 22, 31, 50, 79,
    103, 120, 151, 170, 203, 227, 21, 29, 45, 65, 106, 124, 150, 171, 196, 224, 30, 49, 75, 97,
    121, 142, 165, 186, 209, 229, 19, 25, 52, 70, 93, 116, 143, 166, 192, 219, 26, 34, 62, 75, 97,
    118, 145, 167, 194, 217, 25, 33, 56, 70, 91, 113, 143, 165, 196, 223, 21, 34, 51, 72, 97, 117,
    145, 171, 196, 222, 20, 29, 50, 67, 90, 117, 144, 168, 197, 221, 22, 31, 48, 66, 95, 117, 146,
    168, 196, 222, 24, 33, 51, 77, 116, 134, 158, 180, 200, 224, 21, 28, 70, 87, 106, 124, 149,
    170, 194, 217, 26, 33, 53, 64, 83, 117, 152, 173, 204, 225, 27, 34, 65, 95, 108, 129, 155, 174,
    210, 225, 20, 26, 72, 99, 113, 131, 154, 176, 200, 219, 34, 43, 61, 78, 93, 114, 155, 177, 205,
    229, 23, 29, 54, 97, 124, 138, 163, 179, 209, 229, 30, 38, 56, 89, 118, 129, 158, 178, 200,
    231, 21, 29, 49, 63, 85, 111, 142, 163, 193, 222, 27, 48, 77, 103, 133, 158, 179, 196, 215,
    232, 29, 47, 74, 99, 124, 151, 176, 198, 220, 237, 33, 42, 61, 76, 93, 121, 155, 174, 207, 225,
    29, 53, 87, 112, 136, 154, 170, 188, 208, 227, 24, 30, 52, 84, 131, 150, 166, 186, 203, 229,
    37, 48, 64, 84, 104, 118, 156, 177, 201, 230,
];
const NLSF_CB1_WGHT_NB_MB_Q9: [i16; 320] = [
    2897, 2314, 2314, 2314, 2287, 2287, 2314, 2300, 2327, 2287, 2888, 2580, 2394, 2367, 2314, 2274,
    2274, 2274, 2274, 2194, 2487, 2340, 2340, 2314, 2314, 2314, 2340, 2340, 2367, 2354, 3216, 2766,
    2340, 2340, 2314, 2274, 2221, 2207, 2261, 2194, 2460, 2474, 2367, 2394, 2394, 2394, 2394, 2367,
    2407, 2314, 3479, 3056, 2127, 2207, 2274, 2274, 2274, 2287, 2314, 2261, 3282, 3141, 2580, 2394,
    2247, 2221, 2207, 2194, 2194, 2114, 4096, 3845, 2221, 2620, 2620, 2407, 2314, 2394, 2367, 2074,
    3178, 3244, 2367, 2221, 2553, 2434, 2340, 2314, 2167, 2221, 3338, 3488, 2726, 2194, 2261, 2460,
    2354, 2367, 2207, 2101, 2354, 2420, 2327, 2367, 2394, 2420, 2420, 2420, 2460, 2367, 3779, 3629,
    2434, 2527, 2367, 2274, 2274, 2300, 2207, 2048, 3254, 3225, 2713, 2846, 2447, 2327, 2300, 2300,
    2274, 2127, 3263, 3300, 2753, 2806, 2447, 2261, 2261, 2247, 2127, 2101, 2873, 2981, 2633, 2367,
    2407, 2354, 2194, 2247, 2247, 2114, 3225, 3197, 2633, 2580, 2274, 2181, 2247, 2221, 2221, 2141,
    3178, 3310, 2740, 2407, 2274, 2274, 2274, 2287, 2194, 2114, 3141, 3272, 2460, 2061, 2287, 2500,
    2367, 2487, 2434, 2181, 3507, 3282, 2314, 2700, 2647, 2474, 2367, 2394, 2340, 2127, 3423, 3535,
    3038, 3056, 2300, 1950, 2221, 2274, 2274, 2274, 3404, 3366, 2087, 2687, 2873, 2354, 2420, 2274,
    2474, 2540, 3760, 3488, 1950, 2660, 2897, 2527, 2394, 2367, 2460, 2261, 3028, 3272, 2740, 2888,
    2740, 2154, 2127, 2287, 2234, 2247, 3695, 3657, 2025, 1969, 2660, 2700, 2580, 2500, 2327, 2367,
    3207, 3413, 2354, 2074, 2888, 2888, 2340, 2487, 2247, 2167, 3338, 3366, 2846, 2780, 2327, 2154,
    2274, 2287, 2114, 2061, 2327, 2300, 2181, 2167, 2181, 2367, 2633, 2700, 2700, 2553, 2407, 2434,
    2221, 2261, 2221, 2221, 2340, 2420, 2607, 2700, 3038, 3244, 2806, 2888, 2474, 2074, 2300, 2314,
    2354, 2380, 2221, 2154, 2127, 2287, 2500, 2793, 2793, 2620, 2580, 2367, 3676, 3713, 2234, 1838,
    2181, 2753, 2726, 2673, 2513, 2207, 2793, 3160, 2726, 2553, 2846, 2513, 2181, 2394, 2221, 2181,
];
const NLSF_PRED_NB_MB_Q8: [u8; 18] = [
    179, 138, 140, 148, 151, 149, 153, 151, 163, 116, 67, 82, 59, 92, 72, 100, 89, 92,
];
const NLSF_DELTA_MIN_NB_MB_Q15: [i16; 11] = [250, 3, 6, 3, 3, 3, 4, 3, 3, 3, 461];

/// Piecewise-linear cosine table in Q12 (`silk_LSFCosTab_FIX_Q12`).
const LSF_COS_TAB: [i16; 129] = [
    8192, 8190, 8182, 8170, 8152, 8130, 8104, 8072, 8034, 7994, 7946, 7896, 7840, 7778, 7714, 7644,
    7568, 7490, 7406, 7318, 7226, 7128, 7026, 6922, 6812, 6698, 6580, 6458, 6332, 6204, 6070, 5934,
    5792, 5648, 5502, 5352, 5198, 5040, 4880, 4718, 4552, 4382, 4212, 4038, 3862, 3684, 3502, 3320,
    3136, 2948, 2760, 2570, 2378, 2186, 1990, 1794, 1598, 1400, 1202, 1002, 802, 602, 402, 202, 0,
    -202, -402, -602, -802, -1002, -1202, -1400, -1598, -1794, -1990, -2186, -2378, -2570, -2760,
    -2948, -3136, -3320, -3502, -3684, -3862, -4038, -4212, -4382, -4552, -4718, -4880, -5040,
    -5198, -5352, -5502, -5648, -5792, -5934, -6070, -6204, -6332, -6458, -6580, -6698, -6812,
    -6922, -7026, -7128, -7226, -7318, -7406, -7490, -7568, -7644, -7714, -7778, -7840, -7896,
    -7946, -7994, -8034, -8072, -8104, -8130, -8152, -8170, -8182, -8190, -8192,
];

// ── Public entry point ──────────────────────────────────────────────────────

/// Encode 960 mono 48 kHz samples as a real SILK NB 20 ms Opus packet.
///
/// Shorter inputs are zero-padded; longer inputs are truncated. The returned
/// packet begins with TOC byte `0x08` and is decodable by any RFC 6716 decoder.
pub fn encode_silk_nb_packet(pcm48: &[f32]) -> Vec<u8> {
    let x = downsample_to_nb(pcm48);
    let plan = plan_frame(&x);

    let mut enc = RangeEncoder::new();

    // ── SILK header (1 channel, 1 internal frame) ──
    // `RangeEncoder::enc_bit_logp` writes the value verbatim (it was inverted
    // before oxiaudio 0.2.1 — see that function's docs and
    // `opus_range_dec::tests::roundtrip_bit_logp_is_not_inverted`).
    enc.enc_bit_logp(true, 1); // VAD = active
    enc.enc_bit_logp(false, 1); // has_lbrr = false

    // ── decode_indices (Independently, VAD) ──
    // signal_type = 1 (unvoiced), quant_offset_type = 0 → VAD symbol 0.
    enc.enc_icdf(0, &TYPE_OFFSET_VAD_ICDF, 8);
    // Gain subframe 0: high 3 bits + low 3 bits.
    enc.enc_icdf((plan.gain_index >> 3) as usize, &GAIN_ICDF_UNVOICED, 8);
    enc.enc_icdf((plan.gain_index & 7) as usize, &UNIFORM8_ICDF, 8);
    // Delta gains for subframes 1..3 (neutral → constant gain).
    for _ in 1..NB_SUBFR {
        enc.enc_icdf(DELTA_GAIN_NEUTRAL, &DELTA_GAIN_ICDF, 8);
    }
    // NLSF stage-1 index (offset 0 for signal_type >> 1 == 0).
    enc.enc_icdf(plan.cb1_index, &NLSF_CB1_ICDF_NB_MB, 8);
    // NLSF stage-2 residual indices.
    for i in 0..ORDER {
        let row = nlsf_ec_row(plan.cb1_index, i);
        let stage2 = (plan.nlsf_stage2[i] as i32 + NLSF_QUANT_MAX_AMPLITUDE) as usize;
        enc.enc_icdf(stage2, &NLSF_CB2_ICDF_NB_MB[row..], 8);
    }
    // NLSF interpolation factor: symbol 4 = no interpolation.
    enc.enc_icdf(4, &NLSF_INTERP_FACTOR_ICDF, 8);
    // Excitation seed (uniform 4-way).
    enc.enc_icdf(plan.seed as usize, &UNIFORM4_ICDF, 8);

    // ── decode_pulses ──
    enc.enc_icdf(plan.rate_level, &RATE_LEVELS_ICDF_UV, 8);
    // Per-block pulse sums.
    for b in 0..SHELL_BLOCKS {
        enc.enc_icdf(
            plan.block_sum[b] as usize,
            &PULSES_PER_BLOCK_ICDF[plan.rate_level],
            8,
        );
    }
    // Shell trees.
    for b in 0..SHELL_BLOCKS {
        if plan.block_sum[b] > 0 {
            encode_shell_block(
                &mut enc,
                &plan.magnitude[b * SHELL_LEN..(b + 1) * SHELL_LEN],
            );
        }
    }
    // Signs (unvoiced, quant_offset 0 → base = 7 * (0 + (1 << 1)) = 14).
    encode_signs(&mut enc, &plan.pulses, &plan.block_sum);

    let payload = enc.finish();
    let mut packet = Vec::with_capacity(1 + payload.len());
    packet.push(TOC_NB_20MS_MONO);
    packet.extend_from_slice(&payload);
    packet
}

// ── Frame planning (analysis + quantization) ────────────────────────────────

/// Everything the entropy coder needs to emit one NB frame.
struct FramePlan {
    gain_index: i32,
    cb1_index: usize,
    nlsf_stage2: [i8; ORDER],
    seed: i8,
    rate_level: usize,
    /// Signed excitation pulses as consumed by `decode_core`.
    pulses: [i16; FRAME_LEN],
    /// Absolute pulse magnitudes (shell-coded).
    magnitude: [i16; FRAME_LEN],
    /// Per-shell-block magnitude sums.
    block_sum: [i32; SHELL_BLOCKS],
}

/// Analyze one 8 kHz `i16` frame and quantize it into a [`FramePlan`].
fn plan_frame(x: &[i16; FRAME_LEN]) -> FramePlan {
    // 1. LP analysis → target NLSF in Q15.
    let x_f: Vec<f32> = x.iter().map(|&v| v as f32 / 32_768.0).collect();
    let lp =
        crate::opus_silk::analyze_silk_frame(&x_f, crate::opus_silk::SilkBandwidth::Narrowband);
    let mut target_nlsf_q15 = [0i32; ORDER];
    for (i, &n) in lp.nlsf.iter().take(ORDER).enumerate() {
        target_nlsf_q15[i] = ((n.clamp(0.0, 0.999_9) as f64) * 32_768.0).round() as i32;
    }

    // 2. NLSF two-stage VQ, with bandwidth expansion to cap the synthesis
    //    filter's prediction gain. A too-resonant quantized LPC makes the
    //    decoder's mandatory excitation offset dither resonate to clipping
    //    regardless of gain/attenuation, so widen the NLSFs toward a uniform
    //    (flat-spectrum) layout until the inverse prediction gain is safe.
    let (cb1_index, nlsf_stage2, a_q12, _nlsf_q15) =
        quantize_lpc_bandwidth_limited(&target_nlsf_q15);

    // 3–5. Gain selection + closed-loop excitation quantization.
    let (gain_index, pulses) = choose_gain_and_excitation(x, &a_q12);

    // Derive magnitudes, block sums, and best rate level.
    let mut magnitude = [0i16; FRAME_LEN];
    for i in 0..FRAME_LEN {
        magnitude[i] = pulses[i].unsigned_abs() as i16;
    }
    let mut block_sum = [0i32; SHELL_BLOCKS];
    for b in 0..SHELL_BLOCKS {
        block_sum[b] = magnitude[b * SHELL_LEN..(b + 1) * SHELL_LEN]
            .iter()
            .map(|&m| m as i32)
            .sum();
    }
    let rate_level = select_rate_level(&block_sum);

    FramePlan {
        gain_index,
        cb1_index,
        nlsf_stage2,
        seed: 0,
        rate_level,
        pulses,
        magnitude,
        block_sum,
    }
}

/// Debug snapshot of the encoder's frame plan (used by round-trip tests).
#[cfg(feature = "silk_debug")]
pub struct DebugPlan {
    pub gain_index: i32,
    pub cb1_index: usize,
    pub nlsf_stage2: [i8; ORDER],
    pub nlsf_q15: [i16; ORDER],
    pub a_q12: [i16; ORDER],
    pub seed: i8,
    pub rate_level: usize,
    pub block_sum: [i32; SHELL_BLOCKS],
    pub magnitude: [i16; FRAME_LEN],
    pub pulses: [i16; FRAME_LEN],
}

/// Produce a [`DebugPlan`] for the given 48 kHz frame (debug builds only).
#[cfg(feature = "silk_debug")]
pub fn debug_plan(pcm48: &[f32]) -> DebugPlan {
    let x = downsample_to_nb(pcm48);
    let x_f: Vec<f32> = x.iter().map(|&v| v as f32 / 32_768.0).collect();
    let lp =
        crate::opus_silk::analyze_silk_frame(&x_f, crate::opus_silk::SilkBandwidth::Narrowband);
    let mut target = [0i32; ORDER];
    for (i, &n) in lp.nlsf.iter().take(ORDER).enumerate() {
        target[i] = ((n.clamp(0.0, 0.999_9) as f64) * 32_768.0).round() as i32;
    }
    let (_cb1, _s2, a_q12, nlsf_q15) = quantize_lpc_bandwidth_limited(&target);
    let plan = plan_frame(&x);
    DebugPlan {
        gain_index: plan.gain_index,
        cb1_index: plan.cb1_index,
        nlsf_stage2: plan.nlsf_stage2,
        nlsf_q15,
        a_q12,
        seed: plan.seed,
        rate_level: plan.rate_level,
        block_sum: plan.block_sum,
        magnitude: plan.magnitude,
        pulses: plan.pulses,
    }
}

// ── NLSF quantization ───────────────────────────────────────────────────────

/// Quantize the target NLSFs with bandwidth expansion capped so the resulting
/// synthesis filter is not overly resonant.
///
/// Returns `(cb1_index, stage-2 indices, Q12 LPC)`. The NLSFs are progressively
/// widened toward a flat spectrum until the inverse prediction gain clears
/// [`MIN_INV_PRED_GAIN_Q30`] (or the widening list is exhausted, always yielding
/// a valid, decodable quantization).
fn quantize_lpc_bandwidth_limited(
    target_q15: &[i32; ORDER],
) -> (usize, [i8; ORDER], [i16; ORDER], [i16; ORDER]) {
    let mut result = None;
    for &w in &[0.0f64, 0.08, 0.18, 0.32, 0.5, 0.7] {
        let widened = widen_nlsf(target_q15, w);
        let cb1_index = select_cb1(&widened);
        let (stage2, nlsf_q15) = quantize_nlsf_stage2(cb1_index, &widened);
        let a_q12 = nlsf2a(&nlsf_q15);
        let good = lpc_inv_pred_gain(&a_q12) >= MIN_INV_PRED_GAIN_Q30;
        result = Some((cb1_index, stage2, a_q12, nlsf_q15));
        if good {
            break;
        }
    }
    // The widening list is non-empty, so `result` is always populated; the
    // fallback keeps the function total without a panic path.
    result.unwrap_or_else(|| {
        let cb1_index = select_cb1(target_q15);
        let (stage2, nlsf_q15) = quantize_nlsf_stage2(cb1_index, target_q15);
        (cb1_index, stage2, nlsf2a(&nlsf_q15), nlsf_q15)
    })
}

/// Blend target NLSFs toward a uniform (flat-spectrum) layout by weight `w`.
///
/// `w = 0` returns the target unchanged; `w = 1` returns evenly spaced NLSFs
/// (a flat spectrum, zero prediction gain). Intermediate values apply
/// bandwidth expansion, taming resonant peaks.
fn widen_nlsf(target_q15: &[i32; ORDER], w: f64) -> [i32; ORDER] {
    let mut out = [0i32; ORDER];
    for i in 0..ORDER {
        let uniform = ((i + 1) as f64 / (ORDER + 1) as f64 * 32_768.0).round() as i32;
        out[i] = ((1.0 - w) * target_q15[i] as f64 + w * uniform as f64).round() as i32;
    }
    out
}

/// Select the stage-1 NLSF codebook vector minimizing weighted squared error.
fn select_cb1(target_q15: &[i32; ORDER]) -> usize {
    let mut best = 0usize;
    let mut best_cost = i64::MAX;
    for v in 0..32 {
        let base = v * ORDER;
        let mut cost = 0i64;
        for i in 0..ORDER {
            let centroid_q15 = (NLSF_CB1_NB_MB_Q8[base + i] as i32) << 7;
            let w = NLSF_CB1_WGHT_NB_MB_Q9[base + i] as i64;
            let d = (target_q15[i] - centroid_q15) as i64;
            cost += (w * d * d) >> 16;
        }
        if cost < best_cost {
            best_cost = cost;
            best = v;
        }
    }
    best
}

/// Quantize the stage-2 predictive residual and reconstruct the Q15 NLSFs.
///
/// Returns the signed stage-2 indices (each in `[-3, 3]` to avoid the extension
/// symbols) and the stabilized reconstructed NLSFs, matching `nlsf_decode`.
fn quantize_nlsf_stage2(
    cb1_index: usize,
    target_q15: &[i32; ORDER],
) -> ([i8; ORDER], [i16; ORDER]) {
    let base = cb1_index * ORDER;
    let mut pred_q8 = [0u8; ORDER];
    unpack_predictors(&mut pred_q8, cb1_index);

    // Desired residual in Q10 to hit the target NLSFs exactly.
    let mut target_res_q10 = [0i32; ORDER];
    for i in 0..ORDER {
        let centroid_q15 = (NLSF_CB1_NB_MB_Q8[base + i] as i32) << 7;
        let w = NLSF_CB1_WGHT_NB_MB_Q9[base + i] as i64;
        target_res_q10[i] = (((target_q15[i] - centroid_q15) as i64 * w) >> 14) as i32;
    }

    // Predictive quantization mirroring `residual_dequant` (reverse order).
    let mut indices = [0i8; ORDER];
    let mut res_recon = [0i32; ORDER];
    let mut out_q10 = 0i32;
    for i in (0..ORDER).rev() {
        let pred_q10 = (out_q10 * pred_q8[i] as i32) >> 8;
        let want = target_res_q10[i] - pred_q10;
        // want ≈ (v * qstep) >> 16, v ≈ index * 1024 (± small adjust).
        let v_target = ((want as i64) << 16) / QUANT_STEP_Q16_NB as i64;
        let idx = ((v_target as f64 / 1024.0).round() as i32).clamp(-3, 3);
        indices[i] = idx as i8;
        let mut v = idx << 10;
        if v > 0 {
            v -= NLSF_QUANT_LEVEL_ADJ_Q10;
        } else if v < 0 {
            v += NLSF_QUANT_LEVEL_ADJ_Q10;
        }
        let recon = pred_q10 + (((v as i64) * QUANT_STEP_Q16_NB as i64) >> 16) as i32;
        res_recon[i] = recon;
        out_q10 = recon;
    }

    // Reconstruct NLSFs (same arithmetic as `nlsf_decode`) and stabilize.
    let mut nlsf_q15 = [0i16; ORDER];
    for i in 0..ORDER {
        let w = NLSF_CB1_WGHT_NB_MB_Q9[base + i] as i32;
        let weighted_q15 = ((res_recon[i]) << 14) / w;
        let centroid_q15 = (NLSF_CB1_NB_MB_Q8[base + i] as i32) << 7;
        nlsf_q15[i] = (weighted_q15 + centroid_q15).clamp(0, 32_767) as i16;
    }
    nlsf_stabilize(&mut nlsf_q15);
    (indices, nlsf_q15)
}

/// Unpack stage-2 predictor coefficients for `cb1_index` (mirror of decoder).
fn unpack_predictors(pred_q8: &mut [u8; ORDER], cb1_index: usize) {
    let base = cb1_index * ORDER / 2;
    for i in (0..ORDER).step_by(2) {
        let entry = NLSF_CB2_SELECT_NB_MB[base + i / 2];
        pred_q8[i] = NLSF_PRED_NB_MB_Q8[i + ((entry & 1) as usize) * (ORDER - 1)];
        pred_q8[i + 1] = NLSF_PRED_NB_MB_Q8[i + (((entry >> 4) & 1) as usize) * (ORDER - 1) + 1];
    }
}

/// Compute the stage-2 iCDF row offset for dimension `i` (mirror of decoder).
fn nlsf_ec_row(cb1_index: usize, i: usize) -> usize {
    let sel_base = cb1_index * ORDER / 2;
    let entry = NLSF_CB2_SELECT_NB_MB[sel_base + i / 2];
    if i % 2 == 0 {
        (((entry >> 1) & 7) as usize) * NLSF_STAGE2_ROW
    } else {
        (((entry >> 5) & 7) as usize) * NLSF_STAGE2_ROW
    }
}

/// Stabilize NLSFs with the canonical minimum-delta constraints (decoder port).
fn nlsf_stabilize(nlsf_q15: &mut [i16; ORDER]) {
    let delta_min = &NLSF_DELTA_MIN_NB_MB_Q15;
    for _ in 0..20 {
        let mut min_diff = nlsf_q15[0] as i32 - delta_min[0] as i32;
        let mut split = 0usize;
        for i in 1..ORDER {
            let diff = nlsf_q15[i] as i32 - (nlsf_q15[i - 1] as i32 + delta_min[i] as i32);
            if diff < min_diff {
                min_diff = diff;
                split = i;
            }
        }
        let tail = (1 << 15) - (nlsf_q15[ORDER - 1] as i32 + delta_min[ORDER] as i32);
        if tail < min_diff {
            min_diff = tail;
            split = ORDER;
        }
        if min_diff >= 0 {
            return;
        }
        if split == 0 {
            nlsf_q15[0] = delta_min[0];
        } else if split == ORDER {
            nlsf_q15[ORDER - 1] = ((1 << 15) - delta_min[ORDER] as i32) as i16;
        } else {
            let mut min_center = 0i32;
            for &d in &delta_min[..split] {
                min_center += d as i32;
            }
            min_center += (delta_min[split] as i32) >> 1;
            let mut max_center = 1 << 15;
            for &d in &delta_min[split + 1..=ORDER] {
                max_center -= d as i32;
            }
            max_center -= (delta_min[split] as i32) >> 1;
            let center = rshift_round(nlsf_q15[split - 1] as i32 + nlsf_q15[split] as i32, 1)
                .clamp(min_center, max_center);
            nlsf_q15[split - 1] = (center - ((delta_min[split] as i32) >> 1)) as i16;
            nlsf_q15[split] = (nlsf_q15[split - 1] as i32 + delta_min[split] as i32) as i16;
        }
    }
    nlsf_q15.sort_unstable();
    nlsf_q15[0] = nlsf_q15[0].max(delta_min[0]);
    for i in 1..ORDER {
        let min_allowed = nlsf_q15[i - 1].saturating_add(delta_min[i]);
        nlsf_q15[i] = nlsf_q15[i].max(min_allowed);
    }
    let max_last = ((1 << 15) - delta_min[ORDER] as i32) as i16;
    nlsf_q15[ORDER - 1] = nlsf_q15[ORDER - 1].min(max_last);
    for i in (0..ORDER - 1).rev() {
        let max_allowed = nlsf_q15[i + 1] - delta_min[i + 1];
        nlsf_q15[i] = nlsf_q15[i].min(max_allowed);
    }
}

// ── NLSF → LPC (nlsf2a, decoder port) ───────────────────────────────────────

/// Convert stabilized Q15 NLSFs to Q12 LPC coefficients (mirror of decoder).
fn nlsf2a(nlsf_q15: &[i16; ORDER]) -> [i16; ORDER] {
    const QA: usize = 16;
    let ordering: [usize; ORDER] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];
    let mut cos_lsf_qa = [0i32; ORDER];
    for k in 0..ORDER {
        let f_int = (nlsf_q15[k] as i32 >> 8) as usize;
        let f_frac = nlsf_q15[k] as i32 - ((f_int as i32) << 8);
        let cos_val = LSF_COS_TAB[f_int] as i32;
        let delta = LSF_COS_TAB[f_int + 1] as i32 - cos_val;
        let interp = (cos_val << 8) + delta * f_frac;
        cos_lsf_qa[ordering[k]] = rshift_round(interp, 20 - QA);
    }

    let dd = ORDER / 2;
    let mut p = [0i32; ORDER / 2 + 1];
    let mut q = [0i32; ORDER / 2 + 1];
    nlsf2a_find_poly(&mut p, &cos_lsf_qa, dd, QA);
    nlsf2a_find_poly(&mut q, &cos_lsf_qa[1..], dd, QA);

    let mut a32_qa1 = [0i32; ORDER];
    for k in 0..dd {
        let ptmp = p[k + 1] + p[k];
        let qtmp = q[k + 1] - q[k];
        a32_qa1[k] = -qtmp - ptmp;
        a32_qa1[ORDER - k - 1] = qtmp - ptmp;
    }

    let mut lpc_q12 = [0i16; ORDER];
    lpc_fit(&mut lpc_q12, &mut a32_qa1, QA + 1);
    for i in 0..16 {
        if lpc_inv_pred_gain(&lpc_q12) != 0 {
            break;
        }
        bw_expand_32(&mut a32_qa1, 65_536 - (2 << i));
        for k in 0..ORDER {
            lpc_q12[k] = rshift_round(a32_qa1[k], QA + 1 - 12) as i16;
        }
    }
    lpc_q12
}

fn nlsf2a_find_poly(out: &mut [i32; ORDER / 2 + 1], c_lsf_qa: &[i32], dd: usize, qa: usize) {
    out[0] = 1 << qa;
    out[1] = -c_lsf_qa[0];
    for k in 1..dd {
        let ftmp = c_lsf_qa[2 * k];
        out[k + 1] = (out[k - 1] << 1) - rshift_round64((ftmp as i64) * out[k] as i64, qa);
        for n in (2..=k).rev() {
            out[n] += out[n - 2] - rshift_round64((ftmp as i64) * out[n - 1] as i64, qa);
        }
        out[1] -= ftmp;
    }
}

fn lpc_fit(lpc_q12: &mut [i16; ORDER], a_qin: &mut [i32; ORDER], qin: usize) {
    let qout = 12usize;
    let mut peak_index = 0usize;
    for _ in 0..10 {
        let mut max_abs = 0i32;
        for (idx, &value) in a_qin.iter().enumerate() {
            let abs_value = value.saturating_abs();
            if abs_value > max_abs {
                max_abs = abs_value;
                peak_index = idx;
            }
        }
        let shifted = rshift_round(max_abs, qin - qout);
        if shifted <= i16::MAX as i32 {
            for k in 0..ORDER {
                lpc_q12[k] = rshift_round(a_qin[k], qin - qout) as i16;
            }
            return;
        }
        let capped = shifted.min(163_838);
        let numerator = ((capped - i16::MAX as i32) as i64) << 14;
        let denominator = ((capped as i64) * (peak_index as i64 + 1)) >> 2;
        let chirp_q16 = 65_470 - (numerator / denominator) as i32;
        bw_expand_32(a_qin, chirp_q16);
    }
    for k in 0..ORDER {
        let rounded = rshift_round(a_qin[k], qin - qout);
        let clipped = rounded.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        lpc_q12[k] = clipped;
        a_qin[k] = (clipped as i32) << (qin - qout);
    }
}

fn bw_expand_32(ar_qa1: &mut [i32; ORDER], chirp_q16: i32) {
    let mut chirp = chirp_q16;
    let chirp_minus_one = chirp - 65_536;
    for coeff in ar_qa1.iter_mut().take(ORDER - 1) {
        *coeff = ((*coeff as i64 * chirp as i64) >> 16) as i32;
        chirp += rshift_round(((chirp as i64) * (chirp_minus_one as i64)) as i32, 16);
    }
    ar_qa1[ORDER - 1] = ((ar_qa1[ORDER - 1] as i64 * chirp as i64) >> 16) as i32;
}

/// Inverse prediction gain; returns 0 for unstable filters (decoder port).
fn lpc_inv_pred_gain(lpc_q12: &[i16; ORDER]) -> i32 {
    const INV_PRED_GAIN_QA: usize = 24;
    const A_LIMIT: i32 = 16_773_022;
    const MAX_INV_Q30: i32 = 107_374;
    let mut a = [0i32; ORDER];
    let mut dc_resp = 0i32;
    for k in 0..ORDER {
        dc_resp += lpc_q12[k] as i32;
        a[k] = (lpc_q12[k] as i32) << (INV_PRED_GAIN_QA - 12);
    }
    if dc_resp >= 4096 {
        return 0;
    }
    let mut inv_gain_q30 = 1 << 30;
    for k in (1..ORDER).rev() {
        let coeff = a[k];
        if !(-A_LIMIT..=A_LIMIT).contains(&coeff) {
            return 0;
        }
        let rc_q31 = -(coeff << (31 - INV_PRED_GAIN_QA));
        let rc_mult1 = (1 << 30) - smmul(rc_q31, rc_q31);
        inv_gain_q30 = smmul(inv_gain_q30, rc_mult1) << 2;
        if inv_gain_q30 < MAX_INV_Q30 {
            return 0;
        }
        let mult2_q = 32 - (rc_mult1.unsigned_abs().leading_zeros() as usize);
        let rc_mult2 = inverse32_var_q(rc_mult1, mult2_q + 30);
        for n in 0..((k + 1) >> 1) {
            let tmp1 = a[n];
            let tmp2 = a[k - n - 1];
            let lhs = tmp1.saturating_sub(mul32_frac_q(tmp2, rc_q31, 31));
            let rhs = tmp2.saturating_sub(mul32_frac_q(tmp1, rc_q31, 31));
            a[n] = rshift_round64((lhs as i64) * rc_mult2 as i64, mult2_q);
            a[k - n - 1] = rshift_round64((rhs as i64) * rc_mult2 as i64, mult2_q);
        }
    }
    if !(-A_LIMIT..=A_LIMIT).contains(&a[0]) {
        return 0;
    }
    let rc_q31 = -(a[0] << (31 - INV_PRED_GAIN_QA));
    let rc_mult1 = (1 << 30) - smmul(rc_q31, rc_q31);
    inv_gain_q30 = smmul(inv_gain_q30, rc_mult1) << 2;
    if inv_gain_q30 < MAX_INV_Q30 {
        0
    } else {
        inv_gain_q30
    }
}

// ── Gain + excitation (analysis by synthesis) ───────────────────────────────

/// Choose a gain index and excitation pulses for the frame.
///
/// Uses closed-loop (noise-shaping) quantization: the exact SILK synthesis
/// filter is run in the encoder and each pulse is chosen against the
/// *reconstructed* history, so quantization error is corrected step by step
/// rather than resonating (open-loop diverges badly on peaky LPC filters).
///
/// Two independent knobs keep the frame well-formed:
/// * `gain_index` is raised when a shell block's pulse sum would exceed
///   `SILK_MAX_PULSES` (avoids the LSB-extension coding path).
/// * the target amplitude is attenuated when the exact decoder reconstruction
///   would peak too high; since the reconstruction tracks the (scaled) target,
///   scaling the target down is what actually reduces the decoded peak, leaving
///   headroom for the 8 kHz→48 kHz resampler's sinc overshoot. Attenuation is a
///   pure amplitude scale, so it does not affect waveform correlation.
fn choose_gain_and_excitation(
    x: &[i16; FRAME_LEN],
    a_q12: &[i16; ORDER],
) -> (i32, [i16; FRAME_LEN]) {
    // Initial gain estimate from the LPC residual peak.
    let a_f: Vec<f64> = a_q12.iter().map(|&c| c as f64 / 4096.0).collect();
    let mut peak = 1.0f64;
    for n in 0..FRAME_LEN {
        let mut pred = 0.0f64;
        for (j, &aj) in a_f.iter().enumerate() {
            if n > j {
                pred += aj * x[n - 1 - j] as f64;
            }
        }
        peak = peak.max((x[n] as f64 - pred).abs());
    }
    let want_gain_q10 = (1024.0 * peak / 8.0).max(4.0);
    let mut gain_index = nearest_gain_index(want_gain_q10 * 64.0);
    let mut atten = 1.0f64;

    for _ in 0..24 {
        let gain_q10 = gain_from_index(gain_index) >> 6;
        let mut x_scaled = [0i16; FRAME_LEN];
        for n in 0..FRAME_LEN {
            x_scaled[n] = (x[n] as f64 * atten).round().clamp(-32768.0, 32767.0) as i16;
        }
        let pulses = quantize_excitation(&x_scaled, a_q12, gain_q10);
        let max_block = (0..SHELL_BLOCKS)
            .map(|b| {
                pulses[b * SHELL_LEN..(b + 1) * SHELL_LEN]
                    .iter()
                    .map(|&p| p.unsigned_abs() as i32)
                    .sum::<i32>()
            })
            .max()
            .unwrap_or(0);
        if max_block > SILK_MAX_PULSES && gain_index < N_LEVELS_QGAIN - 1 {
            gain_index += 1;
            continue;
        }
        let peak_pcm = decode_sim_peak(&pulses, a_q12, gain_q10);
        if peak_pcm <= PEAK_TARGET || (gain_index >= N_LEVELS_QGAIN - 1 && atten < 0.02) {
            return (gain_index, pulses);
        }
        atten *= (PEAK_TARGET as f64 / peak_pcm as f64).min(0.9);
    }
    // Fallback: last computed configuration (guaranteed decodable).
    let gain_q10 = gain_from_index(gain_index) >> 6;
    let mut x_scaled = [0i16; FRAME_LEN];
    for n in 0..FRAME_LEN {
        x_scaled[n] = (x[n] as f64 * atten).round().clamp(-32768.0, 32767.0) as i16;
    }
    (gain_index, quantize_excitation(&x_scaled, a_q12, gain_q10))
}

/// Closed-loop (noise-shaping) excitation quantization.
///
/// The exact SILK synthesis is run in the encoder; at each sample the pulse
/// (magnitude + stored sign, accounting for the decoder's LCG dither) is chosen
/// to minimize the error against the reconstructed history. Per-block pulse sums
/// are capped at `SILK_MAX_PULSES`.
fn quantize_excitation(
    x: &[i16; FRAME_LEN],
    a_q12: &[i16; ORDER],
    gain_q10: i32,
) -> [i16; FRAME_LEN] {
    let adj = QUANT_LEVEL_ADJUST_Q10 << 4;
    let off = OFFSET_UVL_Q10 << 4;
    let mut pulses = [0i16; FRAME_LEN];
    let mut c = [0i32; FRAME_LEN]; // reconstructed current_q14 (s_lpc history)
    let mut rand_seed = 0i32; // seed = 0
    let mut block_running = [0i32; SHELL_BLOCKS];

    for n in 0..FRAME_LEN {
        // LPC prediction from the *reconstructed* history (decoder-exact).
        let mut lpc_pred_q10 = (ORDER as i32) >> 1;
        for j in 0..ORDER {
            let hist = if n > j { c[n - 1 - j] } else { 0 };
            lpc_pred_q10 = smlawb(lpc_pred_q10, hist, a_q12[j] as i32);
        }
        let pred_q14 = lpc_pred_q10 << 4;
        let target_c = (((x[n] as i64) << 24) / gain_q10 as i64) as i32;
        let want_exc = target_c.saturating_sub(pred_q14);

        rand_seed = silk_rand(rand_seed);
        let mag_est = (((want_exc.unsigned_abs() as i64) - off as i64).max(0) + 8192) / 16384;
        let block = n / SHELL_LEN;
        let cap = (SILK_MAX_PULSES - block_running[block]).max(0);
        let mut best_pulse = 0i16;
        let mut best_exc = 0i32;
        let mut best_err = i64::MAX;
        for m in mag_est.saturating_sub(1)..=mag_est + 1 {
            if m < 0 {
                continue;
            }
            let m = (m as i32).min(cap);
            for &sign in &[1i32, -1i32] {
                let stored = sign * m;
                let exc = excitation_value(stored, rand_seed, adj, off);
                let err = (exc as i64 - want_exc as i64).abs();
                if err < best_err {
                    best_err = err;
                    best_pulse = stored as i16;
                    best_exc = exc;
                }
                if m == 0 {
                    break;
                }
            }
        }
        pulses[n] = best_pulse;
        block_running[block] += best_pulse.unsigned_abs() as i32;
        rand_seed = rand_seed.wrapping_add(best_pulse as i32);
        c[n] = best_exc.saturating_add(pred_q14);
    }
    pulses
}

/// Simulate the exact SILK decoder synthesis and return the peak |PCM| value.
fn decode_sim_peak(pulses: &[i16; FRAME_LEN], a_q12: &[i16; ORDER], gain_q10: i32) -> i32 {
    let adj = QUANT_LEVEL_ADJUST_Q10 << 4;
    let off = OFFSET_UVL_Q10 << 4;
    let mut c = [0i32; FRAME_LEN];
    let mut rand_seed = 0i32;
    let mut peak = 0i32;
    for n in 0..FRAME_LEN {
        let mut lpc_pred_q10 = (ORDER as i32) >> 1;
        for j in 0..ORDER {
            let hist = if n > j { c[n - 1 - j] } else { 0 };
            lpc_pred_q10 = smlawb(lpc_pred_q10, hist, a_q12[j] as i32);
        }
        rand_seed = silk_rand(rand_seed);
        let exc = excitation_value(pulses[n] as i32, rand_seed, adj, off);
        rand_seed = rand_seed.wrapping_add(pulses[n] as i32);
        c[n] = exc.saturating_add(lpc_pred_q10 << 4);
        let pcm = rshift_round(smulww(c[n], gain_q10), 8);
        peak = peak.max(pcm.abs());
    }
    peak
}

/// Compute the excitation value `decode_core` produces for a stored pulse.
fn excitation_value(stored_pulse: i32, rand_seed: i32, adj: i32, off: i32) -> i32 {
    let mut sample = stored_pulse << 14;
    if sample > 0 {
        sample -= adj;
    } else if sample < 0 {
        sample += adj;
    }
    sample += off;
    if rand_seed < 0 {
        sample = -sample;
    }
    sample
}

/// SILK LCG update (mirror of `silk_rand`).
fn silk_rand(seed: i32) -> i32 {
    seed.wrapping_mul(RAND_MULTIPLIER)
        .wrapping_add(RAND_INCREMENT)
}

/// Compute the linear Q16 gain for a gain index (mirror of `decode_gains`).
fn gain_from_index(index: i32) -> i32 {
    let prev = index.clamp(0, N_LEVELS_QGAIN - 1);
    log2lin(smulwb(GAIN_INV_SCALE_Q16, prev) + GAIN_OFFSET)
}

/// Find the gain index whose linear Q16 gain is closest to `want_q16`.
fn nearest_gain_index(want_q16: f64) -> i32 {
    let mut best = 0i32;
    let mut best_err = f64::MAX;
    for idx in 0..N_LEVELS_QGAIN {
        let g = gain_from_index(idx);
        let err = (g as f64 - want_q16).abs();
        if err < best_err {
            best_err = err;
            best = idx;
        }
    }
    best
}

/// SILK log-domain to linear Q16 conversion (mirror of `log2lin`).
fn log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    }
    if in_log_q7 >= 3967 {
        return i32::MAX;
    }
    let mut out = 1i32 << (in_log_q7 >> 7);
    let frac_q7 = in_log_q7 & 0x7F;
    let curve = smlawb(frac_q7, smulbb(frac_q7, 128 - frac_q7), -174);
    if in_log_q7 < 2048 {
        out = out.saturating_add(rshift_round(out.saturating_mul(curve), 7));
    } else {
        out = out.saturating_add((out >> 7).saturating_mul(curve));
    }
    out
}

// ── Rate level + shell/sign entropy coding ──────────────────────────────────

/// Pick the pulses-per-block rate level minimizing the entropy-coded bit cost.
fn select_rate_level(block_sum: &[i32; SHELL_BLOCKS]) -> usize {
    let mut best = 0usize;
    let mut best_bits = f64::MAX;
    // Rate levels 0..=8 are the "normal" levels; 9 is reserved for extension.
    for (level, table) in PULSES_PER_BLOCK_ICDF
        .iter()
        .enumerate()
        .take(N_RATE_LEVELS - 1)
    {
        let mut bits = 0.0f64;
        for &sum in block_sum {
            let s = sum.clamp(0, 17) as usize;
            let hi = if s == 0 { 256u32 } else { table[s - 1] as u32 };
            let lo = table[s] as u32;
            let p = (hi - lo).max(1) as f64 / 256.0;
            bits -= p.log2();
        }
        if bits < best_bits {
            best_bits = bits;
            best = level;
        }
    }
    best
}

/// Encode one 16-sample shell block (inverse of `silk_shell_decoder`).
fn encode_shell_block(enc: &mut RangeEncoder, m: &[i16]) {
    // Prefix sums over magnitudes.
    let s = |a: usize, b: usize| -> i32 { m[a..b].iter().map(|&v| v as i32).sum() };
    let total = s(0, 16);
    if total == 0 {
        return;
    }
    let p30 = s(0, 8);
    let p31 = total - p30;
    let p20 = s(0, 4);
    let p21 = p30 - p20;
    let p10 = s(0, 2);
    let p11 = p20 - p10;
    let p12 = s(4, 6);
    let p13 = p21 - p12;
    let p22 = s(8, 12);
    let p23 = p31 - p22;
    let p14 = s(8, 10);
    let p15 = p22 - p14;
    let p16 = s(12, 14);
    let p17 = p23 - p16;

    // Encode order mirrors the decoder's split sequence exactly.
    encode_split(enc, total, p30, &SHELL_CODE_TABLE3);
    encode_split(enc, p30, p20, &SHELL_CODE_TABLE2);
    encode_split(enc, p20, p10, &SHELL_CODE_TABLE1);
    encode_split(enc, p10, m[0] as i32, &SHELL_CODE_TABLE0);
    encode_split(enc, p11, m[2] as i32, &SHELL_CODE_TABLE0);
    encode_split(enc, p21, p12, &SHELL_CODE_TABLE1);
    encode_split(enc, p12, m[4] as i32, &SHELL_CODE_TABLE0);
    encode_split(enc, p13, m[6] as i32, &SHELL_CODE_TABLE0);
    encode_split(enc, p31, p22, &SHELL_CODE_TABLE2);
    encode_split(enc, p22, p14, &SHELL_CODE_TABLE1);
    encode_split(enc, p14, m[8] as i32, &SHELL_CODE_TABLE0);
    encode_split(enc, p15, m[10] as i32, &SHELL_CODE_TABLE0);
    encode_split(enc, p23, p16, &SHELL_CODE_TABLE1);
    encode_split(enc, p16, m[12] as i32, &SHELL_CODE_TABLE0);
    encode_split(enc, p17, m[14] as i32, &SHELL_CODE_TABLE0);
}

/// Encode one shell split: `child1` given parent sum `p` (skips when `p == 0`).
fn encode_split(enc: &mut RangeEncoder, p: i32, child1: i32, table: &[u8]) {
    if p <= 0 {
        return;
    }
    let offset = SHELL_CODE_TABLE_OFFSETS[p as usize] as usize;
    enc.enc_icdf(child1 as usize, &table[offset..], 8);
}

/// Encode pulse signs (inverse of `silk_decode_signs`).
///
/// `signal_type = 1`, `quant_offset_type = 0` → sign iCDF base `7 * (0 + 2) = 14`.
fn encode_signs(
    enc: &mut RangeEncoder,
    pulses: &[i16; FRAME_LEN],
    block_sum: &[i32; SHELL_BLOCKS],
) {
    let base = 14usize;
    for b in 0..SHELL_BLOCKS {
        if block_sum[b] <= 0 {
            continue;
        }
        let idx = (block_sum[b] & 0x1F).min(6) as usize;
        let icdf = [SIGN_ICDF[base + idx], 0];
        for &p in &pulses[b * SHELL_LEN..(b + 1) * SHELL_LEN] {
            if p.unsigned_abs() > 0 {
                // sign symbol 1 = positive, 0 = negative.
                let sym = usize::from(p > 0);
                enc.enc_icdf(sym, &icdf, 8);
            }
        }
    }
}

// ── Down-sampling ───────────────────────────────────────────────────────────

/// Down-sample 48 kHz mono `f32` to 160 samples of 8 kHz `i16` (box-filtered).
fn downsample_to_nb(pcm48: &[f32]) -> [i16; FRAME_LEN] {
    let mut x = [0i16; FRAME_LEN];
    for (m, slot) in x.iter_mut().enumerate() {
        let start = m * DECIM;
        let mut acc = 0.0f32;
        let mut cnt = 0.0f32;
        for k in 0..DECIM {
            if let Some(&v) = pcm48.get(start + k) {
                acc += v;
                cnt += 1.0;
            }
        }
        let avg = if cnt > 0.0 { acc / cnt } else { 0.0 };
        *slot = (avg.clamp(-1.0, 1.0) * 32_767.0).round() as i16;
    }
    x
}

// ── Fixed-point primitives (decoder ports) ──────────────────────────────────

fn smulwb(a32: i32, b32: i32) -> i32 {
    let b16 = b32 as i16 as i32;
    let high = (a32 >> 16) * b16;
    let low = ((a32 & 0xFFFF) * b16) >> 16;
    high.wrapping_add(low)
}

fn smlawb(a32: i32, b32: i32, c32: i32) -> i32 {
    a32.wrapping_add(smulwb(b32, c32))
}

fn smulww(a32: i32, b32: i32) -> i32 {
    ((a32 as i64 * b32 as i64) >> 16) as i32
}

fn smulbb(a32: i32, b32: i32) -> i32 {
    (a32 as i16 as i32) * (b32 as i16 as i32)
}

fn smmul(a: i32, b: i32) -> i32 {
    (((a as i64) * (b as i64)) >> 32) as i32
}

fn mul32_frac_q(a: i32, b: i32, q: usize) -> i32 {
    rshift_round64((a as i64) * (b as i64), q)
}

fn rshift_round(value: i32, shift: usize) -> i32 {
    if shift == 1 {
        (value >> 1) + (value & 1)
    } else {
        ((value >> (shift - 1)) + 1) >> 1
    }
}

fn rshift_round64(value: i64, shift: usize) -> i32 {
    if shift == 1 {
        ((value >> 1) + (value & 1)) as i32
    } else {
        (((value >> (shift - 1)) + 1) >> 1) as i32
    }
}

fn abs32_for_clz(value: i32) -> i32 {
    if value == i32::MIN {
        i32::MAX
    } else {
        value.abs()
    }
}

fn lshift_sat32(value: i32, shift: usize) -> i32 {
    if shift >= 31 {
        if value > 0 {
            i32::MAX
        } else if value < 0 {
            i32::MIN
        } else {
            0
        }
    } else {
        value
            .clamp(i32::MIN >> shift, i32::MAX >> shift)
            .wrapping_shl(shift as u32)
    }
}

fn inverse32_var_q(value: i32, q_res: usize) -> i32 {
    debug_assert!(value != 0);
    debug_assert!(q_res > 0);
    let b_headrm = abs32_for_clz(value).leading_zeros() as usize - 1;
    let b32_nrm = value << b_headrm;
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    let mut result = b32_inv << 16;
    let err_q32 = ((1i32 << 29) - smulwb(b32_nrm, b32_inv)) << 3;
    result = result.wrapping_add(((err_q32 as i64 * b32_inv as i64) >> 16) as i32);
    let lshift = 61isize - b_headrm as isize - q_res as isize;
    if lshift <= 0 {
        lshift_sat32(result, (-lshift) as usize)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32) -> Vec<f32> {
        (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 48_000.0).sin() * 0.5)
            .collect()
    }

    #[test]
    fn packet_starts_with_nb_toc() {
        let p = encode_silk_nb_packet(&sine(1000.0));
        assert_eq!(p[0], TOC_NB_20MS_MONO);
        assert!(p.len() >= 3);
    }

    #[test]
    fn nonsilent_input_produces_pulses() {
        let x = downsample_to_nb(&sine(1000.0));
        let plan = plan_frame(&x);
        let total: i32 = plan.block_sum.iter().sum();
        assert!(total > 0, "a 1 kHz tone must code some pulses");
    }

    #[test]
    fn silence_input_is_valid() {
        let p = encode_silk_nb_packet(&vec![0.0f32; 960]);
        assert_eq!(p[0], TOC_NB_20MS_MONO);
    }

    #[test]
    fn deterministic() {
        let a = encode_silk_nb_packet(&sine(700.0));
        let b = encode_silk_nb_packet(&sine(700.0));
        assert_eq!(a, b);
    }

    #[test]
    fn block_sums_never_exceed_max() {
        for f in [200.0, 500.0, 1000.0, 2000.0, 3000.0] {
            let x = downsample_to_nb(&sine(f));
            let plan = plan_frame(&x);
            for &s in &plan.block_sum {
                assert!(s <= SILK_MAX_PULSES, "block sum {s} exceeds cap at {f} Hz");
            }
        }
    }

    #[test]
    fn nlsf_are_stable_and_ordered() {
        let x = downsample_to_nb(&sine(800.0));
        let x_f: Vec<f32> = x.iter().map(|&v| v as f32 / 32_768.0).collect();
        let lp =
            crate::opus_silk::analyze_silk_frame(&x_f, crate::opus_silk::SilkBandwidth::Narrowband);
        let mut target = [0i32; ORDER];
        for (i, &n) in lp.nlsf.iter().take(ORDER).enumerate() {
            target[i] = ((n.clamp(0.0, 0.999_9) as f64) * 32_768.0).round() as i32;
        }
        let cb1 = select_cb1(&target);
        let (_, nlsf_q15) = quantize_nlsf_stage2(cb1, &target);
        let a_q12 = nlsf2a(&nlsf_q15);
        // LPC must be stable (nonzero inverse prediction gain).
        assert_ne!(lpc_inv_pred_gain(&a_q12), 0);
        // NLSFs must be strictly increasing after stabilization.
        for i in 1..ORDER {
            assert!(nlsf_q15[i] > nlsf_q15[i - 1]);
        }
    }
}
