//! CELT (Constrained Energy Lapped Transform) encoder for Opus.
//!
//! Implements band energy quantization and spectral coefficient encoding via
//! Pyramid Vector Quantization (PVQ), following RFC 6716 §4.3 structure.
//!
//! # Conformance note
//!
//! This encoder is structurally closer to RFC 6716 than the previous 4-bit
//! placeholder, but is **not fully conformant**: PVQ is done with a greedy
//! pulse allocator (not the exact combinatorial search in RFC 6716 §4.3.4.6),
//! and the range coder is our private self-consistent variant rather than the
//! RFC bit-reversal packing. The resulting bitstream will NOT be decoded
//! by a standard Opus decoder.
//!
//! # CELT Band Layout (RFC 6716 Table 1)
//!
//! For a 960-sample / 480-bin MDCT at 48 kHz, the band boundaries in MDCT bins are:
//!
//! ```text
//! Band  0:  bins [  0,   1)   ~    0 –  100 Hz
//! Band  1:  bins [  1,   2)   ~  100 –  200 Hz
//! ...
//! Band 20:  bins [ 78, 100)   ~ 7800 – 10000 Hz
//! (Remaining bins 100..480 are treated as trailing high-frequency content.)
//! ```

use crate::opus_celt_tables::{
    BAND_ALLOCATION, CACHE_BITS_50, CACHE_CAPS_50, CACHE_INDEX_50, EBAND_5MS, E_MEANS,
    E_PROB_MODEL, NUM_BANDS_CELT,
};
use crate::opus_mdct::mdct_forward;
use crate::opus_pvq;
use crate::opus_range::{ec_laplace_encode, RangeEncoder};

// BITRES = 3 (Q3 fixed-point scale for bit counts), matching libopus CELT.
const BITRES: i32 = 3;
// Number of allocation levels in BAND_ALLOCATION.
const NB_ALLOC_VECTORS: usize = 11;
// Maximum fine-quant bits per band.
const MAX_FINE_BITS: i32 = 8;
// Fine-offset for energy allocation.
const FINE_OFFSET: i32 = 21;
// Number of bisection steps in interp_bits2pulses.
const ALLOC_STEPS: i32 = 6;
// Theta offset for band splitting (from libopus `celt/bands.c`).
const QTHETA_OFFSET: i32 = 4;
// 2^(x/8) fixed-point table used by compute_qn (from libopus `celt/bands.c`).
const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
// log2_frac table for intensity stereo (unused for mono but needed structurally).
const LOG2_FRAC_TABLE: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

/// Number of CELT frequency bands (RFC 6716, Table 1).
pub const NUM_BANDS: usize = 21;

/// Band lower-bin boundaries (inclusive) in 480-bin MDCT space.
///
/// The 22nd entry is the exclusive upper bound of the last defined band.
/// These match RFC 6716 Table 5 for 480 MDCT coefficients at 48 kHz.
pub const BAND_BINS: [usize; NUM_BANDS + 1] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

// ── PVQ core helpers ──────────────────────────────────────────────────────────

/// Compute the L2 norm of a band's spectral coefficients.
fn band_norm(coeffs: &[f32]) -> f32 {
    coeffs.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Return a unit-norm copy of `coeffs`.
///
/// If the norm is negligible (< 1e-10), returns a zero vector — the PVQ
/// encoder will pile all pulses on index 0 in that case, which is the
/// correct degenerate behaviour for silent bands.
fn normalize_band(coeffs: &[f32]) -> Vec<f32> {
    let norm = band_norm(coeffs);
    if norm < 1e-10 {
        return vec![0.0; coeffs.len()];
    }
    coeffs.iter().map(|&x| x / norm).collect()
}

/// Quantize band energy to a 4-bit index (0..=15).
///
/// Converts `norm` to dB relative to `global_gain_db` and maps the
/// range [−60, 0] dB onto [0, 15].  Values above 0 dB (louder than global
/// gain) are clamped to 15; values below −60 dB clamp to 0.
fn quantize_band_energy(norm: f32, global_gain_db: f32) -> u8 {
    let db = if norm > 1e-20 {
        20.0 * norm.log10()
    } else {
        -120.0_f32
    };
    let relative_db = (db - global_gain_db).clamp(-60.0, 0.0);
    ((relative_db + 60.0) / 4.0) as u8
}

/// Greedy PVQ encoder — allocate `k_pulses` unit pulses across `normalized`.
///
/// The returned vector has the same length as `normalized` and contains signed
/// integers (±magnitude) whose L1 norm equals `k_pulses`.
///
/// The greedy strategy picks the dimension with the largest remaining absolute
/// value at each step, making it a single-pass O(N·K) algorithm.  It produces
/// a good — though not optimal — approximation of the target unit vector.
pub fn pvq_encode(normalized: &[f32], k_pulses: u32) -> Vec<i32> {
    let n = normalized.len();
    let mut y = vec![0i32; n];

    if k_pulses == 0 || n == 0 {
        return y;
    }

    // Working copy of absolute values used to guide greedy allocation.
    let mut mag: Vec<f32> = normalized.iter().map(|&x| x.abs()).collect();
    let step = 1.0 / k_pulses as f32;

    for _ in 0..k_pulses {
        // Find the dimension with the largest remaining magnitude.
        let best = mag
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        y[best] += 1;
        // Reduce the residual by one pulse-step so subsequent iterations
        // can reallocate to other dimensions if their magnitude is larger.
        mag[best] = (mag[best] - step).max(0.0);
    }

    // Apply original signs.
    for (i, yi) in y.iter_mut().enumerate() {
        if normalized[i] < 0.0 {
            *yi = -*yi;
        }
    }

    y
}

/// Compute the number of PVQ pulses for a band given the available bits.
///
/// For a band of `band_size` coefficients with `K` pulses the information
/// content is approximately `band_size · log2(K + 1)` bits.  Inverting gives
/// K ≈ 2^(bits / band_size) − 1.  We use the simpler linear approximation
/// `K = floor(bits / log2(band_size + 1))` which is well-behaved even for
/// very small bands (size = 1).
pub(crate) fn compute_k_pulses(band_size: usize, bits_available: u32) -> u32 {
    if band_size == 0 || bits_available == 0 {
        return 0;
    }
    let log2_n = (band_size as f32 + 1.0).log2().max(1.0);
    (bits_available as f32 / log2_n).floor() as u32
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Encode one CELT frame into the provided range encoder using PVQ band-shape coding.
///
/// # Arguments
///
/// * `pcm`      — Interleaved PCM samples for this frame.  For mono, length is
///   `FRAME_SIZE` (960); for stereo, length is `2 × FRAME_SIZE`.
/// * `channels` — Number of audio channels (1 or 2).
/// * `enc`      — Range encoder to write the coded symbols into.
///
/// # Implementation notes
///
/// Only the first channel is MDCT-analysed; for stereo the two channels are
/// mixed to mono for CELT analysis (a real encoder would use mid/side coding
/// per RFC 6716 §4.3.1).
///
/// Band energies are quantized to 4 bits (16 levels covering a 60 dB range
/// relative to the global gain).  Band shapes are encoded with a greedy PVQ
/// pulse allocator using a fixed budget of 8 bits per band.
pub fn encode_celt_frame(pcm: &[f32], channels: usize, enc: &mut RangeEncoder) {
    encode_celt_frame_inner(pcm, channels, 8, enc);
}

/// Standalone CELT frame encoder that manages its own range encoder.
///
/// Equivalent to constructing a fresh [`RangeEncoder`], calling
/// [`encode_celt_frame`] with the given PCM, and flushing.  The
/// `sample_rate` and `target_bitrate_kbps` parameters are used to compute
/// the per-band bit budget.
///
/// # Returns
///
/// The range-coded frame bytes (always non-empty because the encoder
/// emits at least its flush bytes).
pub fn encode_celt_frame_pvq(
    pcm: &[f32],
    channels: usize,
    sample_rate: u32,
    target_bitrate_kbps: u32,
) -> Vec<u8> {
    use crate::opus_mdct::FRAME_SIZE;
    // Bits per frame = kbps * 1000 * (frame_size / sample_rate)
    //               = kbps * frame_size / (sample_rate / 1000)
    // Use integer arithmetic in the order that avoids premature truncation:
    //   kbps * FRAME_SIZE * 1000 / sample_rate
    let bits_per_frame = target_bitrate_kbps
        .saturating_mul(FRAME_SIZE as u32)
        .saturating_mul(1000)
        / sample_rate.max(1);
    let bits_per_band = (bits_per_frame / NUM_BANDS as u32).max(1);

    let mut enc = RangeEncoder::new();
    encode_celt_frame_inner(pcm, channels, bits_per_band, &mut enc);
    enc.finish()
}

// ── RFC 6716 conformant encoder ──────────────────────────────────────────────

/// Encode one RFC 6716–conformant CELT-only mono 20 ms Opus frame.
///
/// Produces an Opus packet (TOC byte `0xF8` + range-coded frame data) that is
/// decodable by a standard Opus decoder.  The encoder performs:
///
/// 1. MDCT analysis of the input PCM.
/// 2. Per-band energy computation and Laplace-coded intra energy quantization.
/// 3. Neutral TF / spread / dynalloc / trim headers.
/// 4. Rate allocation (`clt_compute_allocation` port from libopus `rate.c`).
/// 5. Fine energy (zeros) written to the end stream.
/// 6. PVQ shapes (greedy CWRS) for each coded band.
///
/// # Attribution
///
/// Rate-allocation logic ported from libopus `celt/rate.c` and `celt/celt.c`
/// (Xiph.Org Foundation, BSD-3-Clause).
pub fn encode_celt_frame_conformant(pcm: &[f32], _channels: usize) -> Vec<u8> {
    // Config 31 = CELT-only, Fullband (20 kHz), 20 ms = 960 samples @ 48 kHz.
    // TOC byte: (31 << 3) | stereo=0 | code=0 = 0xF8.
    const TOC: u8 = 0xF8;
    const LM: usize = 3;
    // Target 64 active bytes = 512 total range-coder bits.
    const TARGET_ACTIVE_BYTES: usize = 64;
    const TARGET_BITS: i32 = TARGET_ACTIVE_BYTES as i32 * 8;

    use crate::opus_mdct::{celt_mdct_960, FRAME_SIZE};
    let mono: Vec<f32> = if pcm.len() >= FRAME_SIZE {
        pcm[..FRAME_SIZE].to_vec()
    } else {
        let mut v = pcm.to_vec();
        v.resize(FRAME_SIZE, 0.0);
        v
    };
    let celt_spec = celt_mdct_960(&mono);

    let mut enc = RangeEncoder::new();
    encode_celt_body_into(&celt_spec, 0, true, TARGET_BITS, LM, &mut enc);

    let frame_bytes = enc.finish_to_size(TARGET_ACTIVE_BYTES);
    let mut packet = Vec::with_capacity(1 + frame_bytes.len());
    packet.push(TOC);
    packet.extend_from_slice(&frame_bytes);
    packet
}

/// Encode the CELT high-band layer (bands `start_band`..21) into an existing
/// range encoder shared with a SILK layer.
///
/// Used by `encode_hybrid_frame_conformant` to write the CELT layer into the
/// same range-coder bitstream that the SILK WB silence layer already wrote into.
///
/// # RFC 6716 hybrid design
///
/// For hybrid mode (config 12–15) the decoder sets `start_band = 17` and reads
/// both the SILK and CELT layers from a single entropy coder. After SILK decodes,
/// `ec.tell()` is well above 1, so the decoder does **not** read a silence flag
/// for the CELT layer (the `tell == 1` branch is not taken). Post-filter is also
/// skipped because `start != 0`. This function mirrors those decoder expectations:
/// it omits the silence flag and post-filter flag, writing only transient, intra,
/// coarse energy, TF, spread, dynalloc, trim, allocation, fine energy and PVQ
/// shapes for bands `start_band`..21.
///
/// # CELT layer uses `start_band=17` and omits silence flag — decoder returns `Ok(960)`.
pub(crate) fn encode_celt_hybrid_layer_into(pcm: &[f32], enc: &mut RangeEncoder) {
    // Same 64-byte / 512-bit target as the CELT-only path. The decoder computes
    // `total_bits = frame.len() * 8` from the full shared frame, so we pass the
    // same `TARGET_BITS` budget; the decoder's bit accounting is consistent because
    // both encoder and decoder count bits from the start of the shared stream.
    const TARGET_BITS_HYBRID: i32 = 512; // 64 bytes × 8
    const LM: usize = 3;
    const START_BAND_HYBRID: usize = 17;

    use crate::opus_mdct::{celt_mdct_960, FRAME_SIZE};
    let mono: Vec<f32> = if pcm.len() >= FRAME_SIZE {
        pcm[..FRAME_SIZE].to_vec()
    } else {
        let mut v = pcm.to_vec();
        v.resize(FRAME_SIZE, 0.0);
        v
    };
    let celt_spec = celt_mdct_960(&mono);

    encode_celt_body_into(
        &celt_spec,
        START_BAND_HYBRID,
        false, // no silence flag in hybrid mode
        TARGET_BITS_HYBRID,
        LM,
        enc,
    );
}

/// Inner CELT encoder body: writes headers → energy → TF → spread → dynalloc →
/// trim → allocation → fine energy → PVQ shapes into `enc`.
///
/// This is the shared implementation used by both CELT-only and hybrid paths.
///
/// # Parameters
///
/// * `celt_spec`          — 960 CELT MDCT coefficients from `celt_mdct_960`.
/// * `start_band`         — First CELT band to encode: `0` for CELT-only, `17` for hybrid.
/// * `write_silence_flag` — When `true`, write a silence flag (logp=15) before postfilter.
///   Set to `false` for hybrid mode (decoder skips it when `tell > 1`).
/// * `target_bits`        — Range-coder budget (in bits). Both encoder and decoder compute
///   `total_bits = active_len * 8`; using the same value keeps allocation in sync.
/// * `lm`                 — Frame-size log-scale (3 for 20 ms / 960 samples at 48 kHz).
/// * `enc`                — Range encoder to write into (may already contain SILK bits).
fn encode_celt_body_into(
    celt_spec: &[f32],
    start_band: usize,
    write_silence_flag: bool,
    target_bits: i32,
    lm: usize,
    enc: &mut RangeEncoder,
) {
    let target_bits_q = target_bits << BITRES;

    // ── 2. Per-band energies (log2 scale) ─────────────────────────────────────
    let band_log2 = celt_band_log2_energies(celt_spec, lm);

    // ── 3. Write headers ──────────────────────────────────────────────────────

    // Silence flag (logp=15): only for CELT-only mode.
    // In hybrid mode the decoder does not read this bit because tell > 1 after SILK.
    if write_silence_flag {
        enc.enc_bit_logp(false, 15);
    }

    // Post-filter flag (logp=1): only when start_band == 0 (pure CELT).
    // The hybrid decoder skips this because start != 0.
    if start_band == 0 && enc.tell() + 16 <= target_bits {
        enc.enc_bit_logp(false, 1);
    }

    // Transient flag (logp=3): written when lm>0 && budget allows.
    if lm > 0 && enc.tell() + 3 <= target_bits {
        enc.enc_bit_logp(false, 3); // not transient
    }

    // Intra energy flag (logp=3): write true (intra mode for first frame).
    if enc.tell() + 3 <= target_bits {
        enc.enc_bit_logp(true, 3);
    }

    // ── 4. Coarse energy ──────────────────────────────────────────────────────
    celt_quant_coarse_energy_intra_ranged(&band_log2, start_band, lm, target_bits, enc);

    // ── 5. TF (neutral: all unchanged, non-transient) ────────────────────────
    celt_write_tf_neutral_ranged(start_band, lm, false, target_bits, enc);

    // ── 6. Spread decision (SPREAD_NORMAL = symbol 2) ─────────────────────────
    if enc.tell() + 4 <= target_bits {
        enc.enc_icdf(2, &crate::opus_celt_tables::SPREAD_ICDF, 5);
    }

    // ── 7. Dynamic allocation boosts (none) ───────────────────────────────────
    let cap = celt_init_caps(lm);
    let offsets = vec![0i32; NUM_BANDS_CELT];
    let dynalloc_logp = 6i32;
    let mut tell_q = enc.tell_frac() as i32;
    for &cap_val in cap[start_band..NUM_BANDS_CELT].iter() {
        if tell_q + (dynalloc_logp << BITRES) < target_bits_q && cap_val > 0 {
            enc.enc_bit_logp(false, dynalloc_logp as u32);
            tell_q = enc.tell_frac() as i32;
        }
    }

    // ── 8. Allocation trim (neutral = symbol 5) ───────────────────────────────
    tell_q = enc.tell_frac() as i32;
    let alloc_trim = if tell_q + (6 << BITRES) <= target_bits_q {
        enc.enc_icdf(5, &crate::opus_celt_tables::TRIM_ICDF, 7);
        tell_q = enc.tell_frac() as i32;
        5i32
    } else {
        5i32
    };

    // ── 9. Rate allocation ────────────────────────────────────────────────────
    let avail_bits = (target_bits_q - tell_q - 1).max(0);
    let alloc =
        celt_compute_allocation_enc(enc, avail_bits, &offsets, &cap, alloc_trim, lm, start_band);

    // ── 10. Fine energy (written to END stream as zeros) ─────────────────────
    for i in start_band..NUM_BANDS_CELT {
        let ebits = alloc.fine_quant[i];
        if ebits > 0 {
            enc.enc_bits(0, ebits as u32);
        }
    }

    // ── 11. PVQ band shapes ───────────────────────────────────────────────────
    // Mirror the decoder's `quant_all_bands_mono` balance-tracking loop exactly.
    //
    // For bands where the decoder would split (b > cache_max + 12), we call
    // `encode_band_with_splits`, which writes the theta symbol (itheta=0) and
    // recurses on the lower-frequency half only.
    //
    // Mirrors `quant_all_bands_mono` / `quant_partition_mono` in libopus
    // `celt/bands.c` (BSD-3-Clause).
    let coded_bands = alloc.coded_bands;
    let mut balance = alloc.balance;
    // Scale factor from EBAND_5MS to CELT coefficient index space.
    // At LM=3: m = 1<<3 = 8.
    let celt_scale = 1usize << lm;
    for band in start_band..NUM_BANDS_CELT {
        let tell = enc.tell_frac() as i32;
        if band != start_band {
            balance -= tell;
        }
        let remaining_bits = target_bits_q - tell - 1;
        let b = if band < coded_bands {
            let curr_balance = balance / ((coded_bands - band).min(3) as i32);
            (alloc.pulses[band] + curr_balance)
                .clamp(0, 16_383)
                .min(remaining_bits + 1)
        } else {
            0
        };
        let celt_lo = (EBAND_5MS[band] as usize) * celt_scale;
        let width = (EBAND_5MS[band + 1] - EBAND_5MS[band]) as usize;
        let n0 = width << lm;
        let mut rem_bits_band = remaining_bits;
        let band_ctx = BandEncCtx {
            celt_spec,
            band,
            celt_lo,
        };
        encode_band_with_splits(&band_ctx, lm as i32, b, &mut rem_bits_band, enc, n0);
        balance += alloc.pulses[band] + tell;
    }

    // ── 12. Finalise energy (extra fine bits, END stream) ────────────────────
    // Mirror decoder's `quant_fine_energy` finalize pass.
    let bits_left_for_finalise = target_bits - enc.tell();
    if bits_left_for_finalise > 0 {
        'outer: for prio in 0..2i32 {
            for i in start_band..NUM_BANDS_CELT {
                if alloc.fine_quant[i] >= MAX_FINE_BITS || alloc.fine_priority[i] != prio {
                    continue;
                }
                if target_bits - enc.tell() <= 0 {
                    break 'outer;
                }
                enc.enc_bits(0, 1);
            }
        }
    }
}

// ── Conformant CELT helpers (ported from libopus, BSD-3-Clause) ──────────────
//
// © 2001–2011 Xiph.Org, Skype Limited, Octasic, Jean-Marc Valin,
// Timothy B. Terriberry, CSIRO, Gregory Maxwell, Mark Borgerding,
// Erik de Castro Lopo.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// · Redistributions of source code must retain the above copyright notice.
// · Redistributions in binary form must reproduce the above copyright notice.
// · Neither the name of the copyright holder nor the names of contributors
//   may be used to endorse or promote products derived from this software.

/// Compute per-band log2 energy from the CELT 960-coefficient MDCT spectrum.
///
/// For each CELT band `i` at frame-level `lm`, the coefficient range is
/// `[EBAND_5MS[i] * (1<<lm), EBAND_5MS[i+1] * (1<<lm))`.  At LM=3 this
/// gives 8-coefficient-wide bands matching the decoder's `denormalise_bands`.
fn celt_band_log2_energies(celt_spec: &[f32], lm: usize) -> Vec<f32> {
    let n_coeff = celt_spec.len();
    let m = 1usize << lm; // = 8 at LM=3
    (0..NUM_BANDS_CELT)
        .map(|i| {
            let lo = (EBAND_5MS[i] as usize) * m;
            let hi = ((EBAND_5MS[i + 1] as usize) * m).min(n_coeff);
            if lo >= n_coeff {
                return -9.0f32;
            }
            let energy: f32 = celt_spec[lo..hi].iter().map(|&x| x * x).sum();
            let rms = (energy / ((hi - lo).max(1) as f32)).sqrt();
            if rms > 1e-20 {
                rms.log2()
            } else {
                -9.0f32
            }
        })
        .collect()
}

/// Encode CELT coarse energies in intra (independent) mode via Laplace coding,
/// for bands `start_band`..`NUM_BANDS_CELT`.
///
/// For intra mode, `coef = 0` and `beta = BETA_INTRA`.  The encoder determines
/// `qi` such that the reconstructed energy approximates `band_log2[i] - E_MEANS[i]`.
///
/// When `start_band = 0` this encodes all 21 bands (CELT-only path).
/// When `start_band = 17` this encodes only bands 17–20 (hybrid CELT path).
///
/// Mirrors `quant_coarse_energy` in libopus `celt/quant_bands.c` (BSD-3-Clause).
fn celt_quant_coarse_energy_intra_ranged(
    band_log2: &[f32],
    start_band: usize,
    lm: usize,
    total_bits: i32,
    enc: &mut RangeEncoder,
) {
    use crate::opus_celt_tables::{BETA_INTRA, SMALL_ENERGY_ICDF};
    let prob_model = &E_PROB_MODEL[lm][1];
    let mut prev = 0.0f32;

    for i in start_band..NUM_BANDS_CELT {
        let tell = enc.tell();
        let target = band_log2[i] - E_MEANS[i.min(24)] - prev;
        let qi = (target.round() as i32).clamp(-128, 127);

        if total_bits - tell >= 15 {
            let pi = 2 * i.min(20);
            let fs = (prob_model[pi] as u32) << 7;
            let decay = (prob_model[pi + 1] as u32) << 6;
            ec_laplace_encode(enc, qi, fs, decay);
        } else if total_bits - tell >= 2 {
            let s = if qi == 0 {
                0
            } else if qi < 0 {
                1
            } else {
                2
            };
            enc.enc_icdf(s, &SMALL_ENERGY_ICDF, 2);
        } else if total_bits - tell >= 1 {
            enc.enc_bit_logp(qi < 0, 1);
        }

        // Mirror decoder's prev accumulation: prev += qi * (1 - BETA_INTRA).
        prev += qi as f32 * (1.0 - BETA_INTRA);
    }
}

/// Write neutral TF decisions for bands `start_band`..`NUM_BANDS_CELT`.
///
/// Mirrors the decoder's `tf_decode` in `celt/celt.c` exactly: for
/// `is_transient = false`, `lm = 3`, writes `false` for each band where
/// `tell + logp <= budget`. `tf_select` is not written when both TF_SELECT
/// choices give the same result for the neutral (all-unchanged) case.
///
/// When `start_band = 0` this covers all 21 bands (CELT-only).
/// When `start_band = 17` this covers only bands 17–20 (hybrid).
fn celt_write_tf_neutral_ranged(
    start_band: usize,
    lm: usize,
    is_transient: bool,
    total_bits: i32,
    enc: &mut RangeEncoder,
) {
    use crate::opus_celt_tables::TF_SELECT_TABLE;
    let nb_bands = NUM_BANDS_CELT - start_band;
    let mut budget = total_bits;
    let mut tell = enc.tell();
    let mut logp = if is_transient { 2i32 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    if tf_select_rsv {
        budget -= 1;
    }
    // tf_changed and curr stay 0 because we always write false (no TF change).
    let tf_changed = 0i32;
    for _i in 0..nb_bands {
        if tell + logp <= budget {
            // We want tf to stay 0 (no change), so write false.
            enc.enc_bit_logp(false, logp as u32);
            // bit_written = false → curr and tf_changed stay 0
            tell = enc.tell();
        }
        logp = if is_transient { 4 } else { 5 };
    }
    // Write tf_select only if both options differ.
    let idx0 = 4 * usize::from(is_transient) + tf_changed as usize;
    let idx1 = 4 * usize::from(is_transient) + 2 + tf_changed as usize;
    if tf_select_rsv
        && lm < TF_SELECT_TABLE.len()
        && idx0 < TF_SELECT_TABLE[lm].len()
        && idx1 < TF_SELECT_TABLE[lm].len()
        && TF_SELECT_TABLE[lm][idx0] != TF_SELECT_TABLE[lm][idx1]
    {
        enc.enc_bit_logp(false, 1); // tf_select = 0
    }
}

/// Compute per-band bit caps from the CACHE_CAPS_50 table.
///
/// Mirrors `init_caps()` in libopus `celt/rate.c` (BSD-3-Clause).
/// For mono (channels=1) at LM=`lm`:
///   `cap[i] = ((CACHE_CAPS_50[(2*lm + 0) * 21 + i] + 64) * 1 * n) >> 2`
/// where `n = (EBAND_5MS[i+1] - EBAND_5MS[i]) << lm`.
pub(crate) fn celt_init_caps(lm: usize) -> Vec<i32> {
    let channels = 1usize;
    (0..NUM_BANDS_CELT)
        .map(|i| {
            let n = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as usize * (1 << lm);
            let idx = NUM_BANDS_CELT * (2 * lm + channels - 1) + i;
            if idx >= CACHE_CAPS_50.len() {
                return 0;
            }
            ((CACHE_CAPS_50[idx] as i32 + 64) * channels as i32 * n as i32) >> 2
        })
        .collect()
}

/// Convert pseudo-pulse index to actual pulse count.
///
/// Mirrors `get_pulses()` in libopus `celt/rate.c` (BSD-3-Clause).
pub(crate) fn celt_get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// Convert bit count (Q3) to pseudo-pulse index via binary search.
///
/// Mirrors `bits2pulses()` in libopus `celt/rate.c` (BSD-3-Clause).
fn celt_bits2pulses(band: usize, lm: usize, bits: i32) -> i32 {
    const LOG_MAX_PSEUDO: usize = 6;
    let lm1 = (lm + 1).min(4); // CACHE_INDEX_50 has 5 rows: lm1 = 0..4
    let cache_base_idx = lm1 * NUM_BANDS_CELT + band;
    if cache_base_idx >= CACHE_INDEX_50.len() {
        return 0;
    }
    let cache_base = CACHE_INDEX_50[cache_base_idx];
    if cache_base < 0 {
        // Negative index → no cache entry; treat as 0 pulses.
        return 0;
    }
    let cache_off = cache_base as usize;
    if cache_off >= CACHE_BITS_50.len() {
        return 0;
    }
    let cache = &CACHE_BITS_50[cache_off..];
    let max_pseudo = cache[0] as i32;
    let mut lo = 0i32;
    let mut hi = max_pseudo;
    let target = bits - 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid = (lo + hi + 1) >> 1;
        let mid_u = mid as usize;
        if mid_u < cache.len() && (cache[mid_u] as i32) >= target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_bits = if lo == 0 {
        -1
    } else if (lo as usize) < cache.len() {
        cache[lo as usize] as i32
    } else {
        -1
    };
    let hi_bits = if (hi as usize) < cache.len() {
        cache[hi as usize] as i32
    } else {
        0
    };
    if target - lo_bits <= hi_bits - target {
        lo
    } else {
        hi
    }
}

/// Convert pseudo-pulse index to spent bits (Q3).
///
/// Mirrors `pulses2bits()` in libopus `celt/rate.c` (BSD-3-Clause).
fn celt_pulses2bits(band: usize, lm: usize, pulses: i32) -> i32 {
    if pulses == 0 {
        return 0;
    }
    let lm1 = (lm + 1).min(4);
    let cache_base_idx = lm1 * NUM_BANDS_CELT + band;
    if cache_base_idx >= CACHE_INDEX_50.len() {
        return 0;
    }
    let cache_base = CACHE_INDEX_50[cache_base_idx];
    if cache_base < 0 {
        return 0;
    }
    let cache_off = cache_base as usize;
    if cache_off >= CACHE_BITS_50.len() {
        return 0;
    }
    let cache = &CACHE_BITS_50[cache_off..];
    let idx = pulses as usize;
    if idx >= cache.len() {
        return 0;
    }
    cache[idx] as i32 + 1
}

/// Allocation result from `celt_compute_allocation_enc`.
struct CeltAllocResult {
    coded_bands: usize,
    /// Per-band Q3 bit budgets for PVQ (fine_quant already subtracted).
    pulses: Vec<i32>,
    fine_quant: Vec<i32>,
    fine_priority: Vec<i32>,
    /// Residual balance carried from the fine/coarse split (mirrors libopus `balance`).
    balance: i32,
}

/// Compute CELT bit allocation for bands `start`..`NUM_BANDS_CELT` (encoder side).
///
/// Writes skip bits into `enc` and returns pulse/fine allocation arrays.
/// When `start = 0` this covers all 21 bands (CELT-only path).
/// When `start = 17` this covers only bands 17–20 (hybrid path).
///
/// Mirrors `clt_compute_allocation` + `interp_bits2pulses` from libopus
/// `celt/rate.c` (BSD-3-Clause), adapted for encoding (writes skip bits
/// instead of reading them).
fn celt_compute_allocation_enc(
    enc: &mut RangeEncoder,
    total: i32,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    lm: usize,
    start: usize,
) -> CeltAllocResult {
    let end = NUM_BANDS_CELT;
    let channels = 1usize;
    let c = channels as i32;

    let mut total_bits = total.max(0);
    let skip_rsv = if total_bits >= (1 << BITRES) {
        1 << BITRES
    } else {
        0
    };
    total_bits -= skip_rsv;

    // Mono: no intensity or dual-stereo bits.
    let intensity_rsv = 0i32;
    let dual_stereo_rsv = 0i32;

    // Threshold and trim-offset per band.
    let mut thresh = vec![0i32; end];
    let mut trim_offset = vec![0i32; end];
    for j in start..end {
        let band_n = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
        thresh[j] = (c << BITRES).max(((3 * band_n) << (lm as i32) << BITRES) >> 4);
        trim_offset[j] = (c
            * band_n
            * (alloc_trim - 5 - lm as i32)
            * (end - j - 1) as i32
            * (1 << (lm as i32 + BITRES)))
            >> 6;
        if (band_n << (lm as i32)) == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }

    // Bisection: find lo/hi allocation vector rows.
    let mut lo = 1i32;
    let mut hi = NB_ALLOC_VECTORS as i32 - 1;
    while lo <= hi {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = false;
        for j in (start..end).rev() {
            let n = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
            let alloc_row = mid as usize * end + j;
            let mut bits_j = if alloc_row < BAND_ALLOCATION.len() {
                (c * n * ((BAND_ALLOCATION[alloc_row] as i32) << (lm as i32))) >> 2
            } else {
                0
            };
            if bits_j > 0 {
                bits_j = (bits_j + trim_offset[j]).max(0);
            }
            bits_j += offsets[j];
            if bits_j >= thresh[j] || done {
                done = true;
                psum += bits_j.min(cap[j]);
            } else if bits_j >= c << BITRES {
                psum += c << BITRES;
            }
        }
        if psum > total_bits {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    hi = lo;
    lo -= 1;

    // Compute bits1 (lo allocation) and bits2 (range above lo).
    let mut bits1 = vec![0i32; end];
    let mut bits2 = vec![0i32; end];
    let mut skip_start = start;
    for j in start..end {
        let n = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
        let alloc_lo = lo as usize * end + j;
        let alloc_hi = hi as usize * end + j;
        let mut bits1j = if lo > 0 && alloc_lo < BAND_ALLOCATION.len() {
            (c * n * ((BAND_ALLOCATION[alloc_lo] as i32) << (lm as i32))) >> 2
        } else {
            0
        };
        let mut bits2j = if (hi as usize) < NB_ALLOC_VECTORS && alloc_hi < BAND_ALLOCATION.len() {
            (c * n * ((BAND_ALLOCATION[alloc_hi] as i32) << (lm as i32))) >> 2
        } else {
            cap[j]
        };
        if bits1j > 0 {
            bits1j = (bits1j + trim_offset[j]).max(0);
        }
        if bits2j > 0 {
            bits2j = (bits2j + trim_offset[j]).max(0);
        }
        if lo > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2j = (bits2j - bits1j).max(0);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }

    // Interpolate between lo/hi to find final per-band bits, writing skip bits.
    let mut pulses = vec![0i32; end];
    let mut fine_quant = vec![0i32; end];
    let mut fine_priority = vec![0i32; end];
    let mut ibp_ctx = InterpBitsCtx {
        bits1: &bits1,
        bits2: &bits2,
        thresh: &thresh,
        cap,
        bits_out: &mut pulses,
        ebits_out: &mut fine_quant,
        fine_priority_out: &mut fine_priority,
        total: total_bits,
        skip_rsv,
        intensity_rsv,
        dual_stereo_rsv,
        channels,
        lm,
    };
    let (coded_bands, balance) = interp_bits2pulses_enc(enc, start, end, skip_start, &mut ibp_ctx);

    CeltAllocResult {
        coded_bands,
        pulses,
        fine_quant,
        fine_priority,
        balance,
    }
}

/// Context for `interp_bits2pulses_enc` carrying all band-indexed slices and
/// scalar parameters that are not the encoder or range-start indices.
struct InterpBitsCtx<'a> {
    bits1: &'a [i32],
    bits2: &'a [i32],
    thresh: &'a [i32],
    cap: &'a [i32],
    bits_out: &'a mut [i32],
    ebits_out: &'a mut [i32],
    fine_priority_out: &'a mut [i32],
    total: i32,
    skip_rsv: i32,
    intensity_rsv: i32,
    dual_stereo_rsv: i32,
    channels: usize,
    lm: usize,
}

/// Interpolate allocation and write skip bits (encoder counterpart of decoder's
/// `interp_bits2pulses`).  Writes one `enc_bit_logp(true, 1)` at the first
/// band that would be "skip-checked", keeping all coded bands active.
///
/// Ported/adapted from libopus `celt/rate.c` (BSD-3-Clause).
fn interp_bits2pulses_enc(
    enc: &mut RangeEncoder,
    start: usize,
    end: usize,
    skip_start: usize,
    ctx: &mut InterpBitsCtx<'_>,
) -> (usize, i32) {
    let bits1 = ctx.bits1;
    let bits2 = ctx.bits2;
    let thresh = ctx.thresh;
    let cap = ctx.cap;
    let bits = &mut *ctx.bits_out;
    let ebits = &mut *ctx.ebits_out;
    let fine_priority = &mut *ctx.fine_priority_out;
    let total = ctx.total;
    let skip_rsv = ctx.skip_rsv;
    let intensity_rsv = ctx.intensity_rsv;
    let dual_stereo_rsv = ctx.dual_stereo_rsv;
    let channels = ctx.channels;
    let lm = ctx.lm;
    let c = channels as i32;
    let alloc_floor = c << BITRES;
    let log_m = (lm as i32) << BITRES;

    // Inner bisection: find interpolation fraction.
    let mut lo = 0i32;
    let mut hi = 1 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid * bits2[j]) >> ALLOC_STEPS);
            if tmp >= thresh[j] || done {
                done = true;
                psum += tmp.min(cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let mut psum = 0i32;
    let mut done = false;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo * bits2[j]) >> ALLOC_STEPS);
        if tmp < thresh[j] && !done {
            tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
        } else {
            done = true;
        }
        tmp = tmp.min(cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    // Skip loop: encode skip bits, reducing coded_bands if needed.
    let mut coded_bands = end;
    let mut total_adj = total;
    let mut intensity_rsv_cur = intensity_rsv;
    loop {
        let j = coded_bands - 1;
        if j <= skip_start {
            total_adj += skip_rsv;
            break;
        }
        let left = total - psum;
        let denom = (EBAND_5MS[coded_bands] - EBAND_5MS[start]) as i32;
        let percoeff = if denom > 0 { left / denom } else { 0 };
        let left_rem = left - denom * percoeff;
        let rem = (left_rem - (EBAND_5MS[j] - EBAND_5MS[start]) as i32).max(0);
        let band_width = (EBAND_5MS[coded_bands] - EBAND_5MS[j]) as i32;
        let mut band_bits = bits[j] + percoeff * band_width + rem;
        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            // Write skip bit: 'true' = keep all bands (don't skip).
            enc.enc_bit_logp(true, 1);
            // Account for the skip bit cost.
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
            // Decoder would "break" here → coded_bands unchanged, loop exits.
            total_adj = total;
            let _ = band_bits;
            break;
        }
        psum -= bits[j] + intensity_rsv_cur;
        if intensity_rsv_cur > 0 {
            intensity_rsv_cur = LOG2_FRAC_TABLE[(j - start).min(23)] as i32;
        }
        psum += intensity_rsv_cur;
        bits[j] = if band_bits >= alloc_floor {
            alloc_floor
        } else {
            0
        };
        psum += bits[j];
        coded_bands -= 1;
    }
    coded_bands = coded_bands.max(start + 1);

    // Mono: no intensity or dual-stereo bits to write.
    let _ = (dual_stereo_rsv, intensity_rsv);

    // Final distribution of remaining bits.
    let left = total_adj - psum;
    let denom = (EBAND_5MS[coded_bands] - EBAND_5MS[start]) as i32;
    let percoeff = if denom > 0 { left / denom } else { 0 };
    let mut left_rem = left - denom * percoeff;
    for (j, bits_j) in bits.iter_mut().enumerate().take(coded_bands).skip(start) {
        *bits_j += percoeff * (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
        let tmp = left_rem.min((EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32);
        *bits_j += tmp;
        left_rem -= tmp;
    }

    // Fine/coarse split + fine priority.
    let stereo_shift = 0i32; // mono
    let mut balance = 0i32;
    for j in start..coded_bands {
        let n0 = (EBAND_5MS[j + 1] - EBAND_5MS[j]) as i32;
        let n = n0 << (lm as i32);
        let bit = bits[j] + balance;
        let excess;
        if n > 1 {
            excess = (bit - cap[j]).max(0);
            bits[j] = bit - excess;
            let den = c * n;
            let nclogn = den * (LOG_N_400[j] as i32 + log_m);
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += den << BITRES >> 2;
            }
            if bits[j] + offset < (den * 2) << BITRES {
                offset += nclogn >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nclogn >> 3;
            }
            let e = ((bits[j] + offset + (den << (BITRES - 1))).max(0) / den) >> BITRES;
            let e = e
                .min(MAX_FINE_BITS)
                .min((bits[j] >> stereo_shift) >> BITRES);
            ebits[j] = e;
            fine_priority[j] = i32::from(ebits[j] * (den << BITRES) >= bits[j] + offset);
            bits[j] -= (c * ebits[j]) << BITRES;
        } else {
            excess = (bit - (c << BITRES)).max(0);
            bits[j] = bit - excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
        }
        if excess > 0 {
            let extra_fine = ((excess >> BITRES).max(0)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = (extra_fine * c) << BITRES;
            fine_priority[j] = i32::from(extra_bits >= excess - balance);
            balance = excess - extra_bits;
        } else {
            balance = excess; // 0
        }
    }
    for j in coded_bands..end {
        ebits[j] = bits[j] >> stereo_shift >> BITRES;
        bits[j] = 0;
        fine_priority[j] = i32::from(ebits[j] < 1);
    }
    (coded_bands, balance)
}

/// Greedy PVQ pulse allocation + CWRS encode for a single CELT band.
///
/// Finds the best signed integer pulse vector `y` with L1 norm = `k_pulses`
/// that maximises `<y, shape>`, then encodes it via `encode_pulses`.
fn celt_alg_quant(shape: &[f32], k_pulses: u32, enc: &mut RangeEncoder) {
    let n = shape.len();
    if n == 0 || k_pulses == 0 {
        return;
    }
    let mut y = vec![0i32; n];
    let mut mag: Vec<f32> = shape.iter().map(|&x| x.abs()).collect();
    let step = if k_pulses > 0 {
        1.0 / k_pulses as f32
    } else {
        1.0
    };
    for _ in 0..k_pulses {
        // Greedy: pick dimension with largest remaining residual.
        let best = mag
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        y[best] += 1;
        mag[best] = (mag[best] - step).max(0.0);
    }
    // Apply original signs.
    for (i, yi) in y.iter_mut().enumerate() {
        if i < shape.len() && shape[i] < 0.0 {
            *yi = -*yi;
        }
    }
    opus_pvq::encode_pulses(enc, &y);
}

/// LOG_N_400 re-export for use inside the allocation and split functions.
use crate::opus_celt_tables::LOG_N_400;

// ── Band-splitting helpers (ported from libopus `celt/bands.c`, BSD-3-Clause) ─

/// Return true iff the decoder will split band `band` at level `lm`.
///
/// Mirrors the `do_split` condition at the top of `quant_partition_mono` in
/// libopus `celt/bands.c` (BSD-3-Clause).
fn check_do_split_enc(band: usize, lm: i32, n0: usize, b: i32) -> bool {
    if lm < 0 || n0 <= 2 {
        return false;
    }
    let lm1 = ((lm + 1) as usize).min(4);
    let cache_row = lm1 * NUM_BANDS_CELT + band;
    if cache_row >= CACHE_INDEX_50.len() {
        return false;
    }
    let cache_base = CACHE_INDEX_50[cache_row];
    if cache_base < 0 {
        return false;
    }
    let cache = &CACHE_BITS_50[cache_base as usize..];
    if cache.is_empty() {
        return false;
    }
    let max_pseudo = cache[0] as usize;
    if max_pseudo >= cache.len() {
        return false;
    }
    b > cache[max_pseudo] as i32 + 12
}

/// Compute theta quantisation parameter `qn`.
///
/// Mirrors `compute_qn()` in libopus `celt/bands.c` (BSD-3-Clause).
fn celt_compute_qn(n: usize, b: i32, offset: i32, pulse_cap: i32) -> i32 {
    let n2 = 2 * n as i32 - 1;
    let mut qb = (b + n2 * offset) / n2;
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let mut qn = EXP2_TABLE8[(qb & 0x7) as usize] >> (14 - (qb >> BITRES));
        qn = ((qn + 1) >> 1) << 1;
        qn.min(256)
    }
}

/// Extract a unit-norm band shape from the 960-coefficient CELT MDCT spectrum.
///
/// Reads `n` consecutive coefficients starting at `celt_lo` (CELT coefficient
/// index) and normalises them to unit L2 norm.  No upsampling is needed
/// because the CELT spectrum already has one coefficient per PVQ dimension.
fn celt_band_shape_from_spec(celt_spec: &[f32], celt_lo: usize, n: usize) -> Vec<f32> {
    let n_coeff = celt_spec.len();
    let mut shape: Vec<f32> = (0..n)
        .map(|p| {
            let idx = celt_lo + p;
            if idx < n_coeff {
                celt_spec[idx]
            } else {
                0.0
            }
        })
        .collect();
    let norm: f32 = shape.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-20 {
        for v in shape.iter_mut() {
            *v /= norm;
        }
    } else if !shape.is_empty() {
        shape[0] = 1.0;
    }
    shape
}

/// Context for `encode_band_with_splits` carrying the parameters that do not
/// change across recursive calls.
struct BandEncCtx<'a> {
    celt_spec: &'a [f32],
    band: usize,
    celt_lo: usize,
}

/// Encode one CELT band, recursively splitting it when the decoder would also
/// split (i.e., when b > cache_max + 12).
///
/// Strategy: encode theta = 0 (itheta = 0, all bits to the lower-frequency
/// half) for every split.  The decoder decodes this correctly: upper half gets
/// 0 bits and uses noise fill; lower half is recursively encoded.
///
/// Mirrors the split path in libopus `celt/bands.c`
/// `quant_partition_mono` (BSD-3-Clause).
fn encode_band_with_splits(
    ctx: &BandEncCtx<'_>,
    lm: i32,
    b: i32,
    remaining_bits: &mut i32,
    enc: &mut RangeEncoder,
    n0: usize,
) {
    let celt_spec = ctx.celt_spec;
    let band = ctx.band;
    let celt_lo = ctx.celt_lo;

    // N = 1: only a sign bit (no PVQ shape).
    if n0 == 1 {
        if *remaining_bits >= (1 << BITRES) {
            enc.enc_bits(0, 1); // encode positive sign
            *remaining_bits -= 1 << BITRES;
        }
        return;
    }

    let do_split = check_do_split_enc(band, lm, n0, b);

    if do_split {
        let n = n0 >> 1; // half-band CELT coefficient count
        let lm_new = lm - 1;

        // pulse_cap and offset for compute_qn — same formula as decode_theta_mono.
        let log_n_val = LOG_N_400.get(band).copied().unwrap_or(0) as i32;
        let pulse_cap = log_n_val + lm_new * (1 << BITRES);
        let offset = (pulse_cap >> 1) - QTHETA_OFFSET;
        let qn = celt_compute_qn(n, b, offset, pulse_cap);

        let mut b_after = b;
        if qn != 1 {
            // Encode itheta = 0 using the triangular distribution (blocks0=1).
            // Decoder: ft = ((qn>>1)+1)^2; reads fm then dec.update(fl, fl+fs, ft).
            // For itheta_step=0: fl=0, fs=1. Encoder writes encode(0, 1, ft).
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            let tell_before = enc.tell_frac() as i32;
            enc.encode(0, 1, ft as u32);
            let qalloc = enc.tell_frac() as i32 - tell_before;
            b_after -= qalloc;
            *remaining_bits -= qalloc;
        }

        // itheta = 0 → delta = −16384.
        // mbits = max(0, min(b_after, (b_after+16384)/2)) = b_after.
        // sbits = 0 → decoder uses noise fill for the upper-frequency half.
        let mbits = b_after;

        // Recurse on the lower-frequency half; upper half writes nothing.
        encode_band_with_splits(
            ctx,
            lm_new,
            mbits,
            remaining_bits,
            enc,
            n, // half the coefficient count
        );
    } else {
        // No split: standard bits2pulses → K → alg_quant path.
        let lm_u = lm.max(0) as usize;
        let q = celt_bits2pulses(band, lm_u, b.max(0));
        let curr_bits = celt_pulses2bits(band, lm_u, q);
        // Mirror decoder's q-reduction loop: cap q so remaining_bits stays ≥ 0.
        let mut q_enc = q;
        let mut cost_enc = curr_bits;
        let mut rem = *remaining_bits - curr_bits;
        while rem < 0 && q_enc > 0 {
            rem += cost_enc;
            q_enc -= 1;
            cost_enc = celt_pulses2bits(band, lm_u, q_enc);
            rem -= cost_enc;
        }
        let k = celt_get_pulses(q_enc);
        if k > 0 {
            let shape = celt_band_shape_from_spec(celt_spec, celt_lo, n0);
            celt_alg_quant(&shape, k as u32, enc);
        }
    }
}

// ── Internal implementation ───────────────────────────────────────────────────

/// Core CELT frame encoder.
///
/// `bits_per_band` controls how many PVQ pulses are allocated per band.
fn encode_celt_frame_inner(
    pcm: &[f32],
    channels: usize,
    bits_per_band: u32,
    enc: &mut RangeEncoder,
) {
    // ── Step 1: Extract mono from interleaved PCM and run MDCT ────────────────
    let mono: Vec<f32> = if channels <= 1 {
        pcm.to_vec()
    } else {
        pcm.chunks_exact(channels)
            .map(|frame| frame[0] * 0.5 + frame[1] * 0.5)
            .collect()
    };

    // Ensure we have exactly FRAME_SIZE samples (pad with silence if short).
    use crate::opus_mdct::FRAME_SIZE;
    let mono_padded: Vec<f32> = if mono.len() >= FRAME_SIZE {
        mono[..FRAME_SIZE].to_vec()
    } else {
        let mut v = mono;
        v.resize(FRAME_SIZE, 0.0);
        v
    };

    let spectrum = mdct_forward(&mono_padded);

    // ── Step 2: Compute global gain (RMS of the full spectrum in dB) ──────────
    let rms: f32 = (spectrum.iter().map(|&x| x * x).sum::<f32>() / spectrum.len() as f32).sqrt();
    let global_gain_db = if rms > 1e-20 {
        20.0 * rms.log10()
    } else {
        -120.0_f32
    };

    // ── Step 3: Encode the 21 CELT bands ─────────────────────────────────────
    for band_idx in 0..NUM_BANDS {
        let lo = BAND_BINS[band_idx];
        let hi = BAND_BINS[band_idx + 1].min(spectrum.len());
        if lo >= spectrum.len() {
            break;
        }

        let band_coeffs = &spectrum[lo..hi];
        let norm = band_norm(band_coeffs);

        // 3a. Encode band energy (4-bit, 16 levels).
        let energy_q = quantize_band_energy(norm, global_gain_db);
        enc.encode_uint(energy_q as u32, 16);

        // 3b. Encode band shape via PVQ.
        let normalized = normalize_band(band_coeffs);
        let k = compute_k_pulses(band_coeffs.len(), bits_per_band);
        if k > 0 {
            let y = pvq_encode(&normalized, k);
            opus_pvq::encode_pulses(enc, &y);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        band_norm, encode_celt_frame, encode_celt_frame_pvq, normalize_band, pvq_encode, BAND_BINS,
        NUM_BANDS,
    };
    use crate::opus_range::RangeEncoder;

    // ── Existing CELT frame tests (kept from the previous scaffold) ────────────

    fn silence_frame(channels: usize) -> Vec<f32> {
        vec![0.0f32; 960 * channels]
    }

    fn sine_frame(channels: usize) -> Vec<f32> {
        let n = 960 * channels;
        (0..n)
            .map(|i| {
                let sample_idx = i / channels;
                (2.0 * std::f32::consts::PI * 440.0 * sample_idx as f32 / 48_000.0).sin() * 0.5
            })
            .collect()
    }

    #[test]
    fn test_encode_celt_frame_silence_does_not_panic() {
        let pcm = silence_frame(1);
        let mut enc = RangeEncoder::new();
        encode_celt_frame(&pcm, 1, &mut enc);
        let bytes = enc.finish();
        assert!(
            !bytes.is_empty(),
            "CELT encoding must produce output even for silence"
        );
    }

    #[test]
    fn test_encode_celt_frame_stereo_does_not_panic() {
        let pcm = sine_frame(2);
        let mut enc = RangeEncoder::new();
        encode_celt_frame(&pcm, 2, &mut enc);
        let bytes = enc.finish();
        assert!(
            !bytes.is_empty(),
            "CELT stereo encoding must produce output"
        );
    }

    #[test]
    fn test_encode_celt_frame_sine_produces_output() {
        let pcm_silence = silence_frame(1);
        let mut enc_s = RangeEncoder::new();
        encode_celt_frame(&pcm_silence, 1, &mut enc_s);
        let silence_bytes = enc_s.finish().len();

        let pcm_sine = sine_frame(1);
        let mut enc_n = RangeEncoder::new();
        encode_celt_frame(&pcm_sine, 1, &mut enc_n);
        let sine_bytes = enc_n.finish().len();

        assert!(silence_bytes > 0 && sine_bytes > 0);
    }

    // ── Band constant sanity ────────────────────────────────────────────────────

    #[test]
    fn test_band_bins_count() {
        assert_eq!(
            BAND_BINS.len(),
            NUM_BANDS + 1,
            "BAND_BINS must have NUM_BANDS + 1 entries"
        );
    }

    #[test]
    fn test_band_bins_monotone() {
        for i in 0..NUM_BANDS {
            assert!(
                BAND_BINS[i] < BAND_BINS[i + 1],
                "BAND_BINS must be strictly increasing at index {i}"
            );
        }
    }

    // ── band_norm ───────────────────────────────────────────────────────────────

    #[test]
    fn test_band_norm_unit_vector() {
        let v = vec![0.6f32, 0.8];
        let norm = band_norm(&v);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "unit vector norm must be 1.0, got {norm}"
        );
    }

    #[test]
    fn test_band_norm_zero_vector() {
        let v = vec![0.0f32; 4];
        let norm = band_norm(&v);
        assert_eq!(norm, 0.0, "zero vector norm must be 0.0");
    }

    #[test]
    fn test_band_norm_single_element() {
        let v = vec![3.0f32];
        let norm = band_norm(&v);
        assert!((norm - 3.0).abs() < 1e-6, "single-element norm, got {norm}");
    }

    // ── normalize_band ──────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_band_unit_output() {
        let v = vec![3.0f32, 4.0];
        let n = normalize_band(&v);
        let out_norm = band_norm(&n);
        assert!(
            (out_norm - 1.0).abs() < 1e-5,
            "normalized vector must have unit norm, got {out_norm}"
        );
    }

    #[test]
    fn test_normalize_band_zero_input() {
        let v = vec![0.0f32; 4];
        let n = normalize_band(&v);
        assert!(
            n.iter().all(|&x| x == 0.0),
            "zero input normalizes to zeros"
        );
    }

    // ── pvq_encode ──────────────────────────────────────────────────────────────

    #[test]
    fn test_pvq_encode_output_length() {
        let normalized = vec![0.5f32, -0.5, 0.5, -0.5];
        let y = pvq_encode(&normalized, 4);
        assert_eq!(y.len(), 4, "PVQ output must match input length");
    }

    #[test]
    fn test_pvq_encode_sums_to_k() {
        let normalized = vec![0.6f32, 0.4, 0.0, 0.0];
        let y = pvq_encode(&normalized, 5);
        let l1: i32 = y.iter().map(|&x| x.abs()).sum();
        assert_eq!(l1, 5, "PVQ L1 norm must equal K pulses, got {l1}");
    }

    #[test]
    fn test_pvq_encode_silent_input() {
        let normalized = vec![0.0f32; 8];
        let y = pvq_encode(&normalized, 0);
        assert!(
            y.iter().all(|&x| x == 0),
            "zero pulses must produce all-zero output"
        );
    }

    #[test]
    fn test_pvq_encode_zero_k_is_all_zeros() {
        let normalized = vec![0.5f32, 0.5, 0.0, 0.0];
        let y = pvq_encode(&normalized, 0);
        assert!(
            y.iter().all(|&x| x == 0),
            "k=0 must produce all-zero output"
        );
    }

    #[test]
    fn test_pvq_encode_single_pulse_goes_to_largest() {
        // With k=1, the single pulse must land on the largest-magnitude dimension.
        let normalized = vec![0.1f32, 0.9, 0.3, 0.0];
        let y = pvq_encode(&normalized, 1);
        let l1: i32 = y.iter().map(|&x| x.abs()).sum();
        assert_eq!(l1, 1, "single pulse: L1 must be 1");
        // The pulse should land on index 1 (largest magnitude = 0.9).
        assert_eq!(
            y[1].abs(),
            1,
            "single pulse should land on highest-magnitude dimension"
        );
    }

    #[test]
    fn test_pvq_encode_sign_preservation() {
        // Negative component should get negative pulse.
        let normalized = vec![-0.9f32, 0.1, 0.0, 0.0];
        let y = pvq_encode(&normalized, 1);
        assert!(y[0] < 0, "pulse on negative component must be negative");
    }

    // ── encode_celt_frame_pvq ───────────────────────────────────────────────────

    #[test]
    fn test_celt_frame_with_pvq_nonempty() {
        let pcm: Vec<f32> = (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let encoded = encode_celt_frame_pvq(&pcm, 1, 48_000, 64);
        assert!(
            !encoded.is_empty(),
            "PVQ-encoded CELT frame must not be empty"
        );
    }

    #[test]
    fn test_celt_frame_pvq_differs_from_silence() {
        let sine: Vec<f32> = (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let silence = vec![0.0f32; 960];
        let enc_sine = encode_celt_frame_pvq(&sine, 1, 48_000, 64);
        let enc_silence = encode_celt_frame_pvq(&silence, 1, 48_000, 64);
        // Non-silence input should produce a different bitstream than silence
        // (the band energies differ even if the PVQ shapes are degenerate).
        assert_ne!(
            enc_sine, enc_silence,
            "sine and silence should produce different CELT frames"
        );
    }

    #[test]
    fn test_celt_frame_pvq_two_freqs_differ() {
        // Two non-silent sine waves at different frequencies should produce different
        // bitstreams — this exercises both the energy quantization AND the PVQ shape
        // path (the energy peaks land in different CELT bands).
        let sine_440: Vec<f32> = (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let sine_8k: Vec<f32> = (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 8_000.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let enc_440 = encode_celt_frame_pvq(&sine_440, 1, 48_000, 64);
        let enc_8k = encode_celt_frame_pvq(&sine_8k, 1, 48_000, 64);
        assert_ne!(
            enc_440, enc_8k,
            "440 Hz and 8 kHz sines must produce different CELT bitstreams"
        );
    }

    #[test]
    fn test_celt_frame_pvq_bits_per_band_nonzero() {
        // Verify the bit-budget formula gives meaningful K values (K > 0) for typical
        // bands with the standard 64 kbps / 48 kHz setting.
        // bits_per_frame = 64 * 960 * 1000 / 48000 = 1280, bits_per_band = 1280/21 = 60.
        // N=1: K = floor(60 / log2(2)) = 60.
        // N=6: K = floor(60 / log2(7)) ≈ 21.
        // Both are non-zero, meaning PVQ shape is emitted for all bands.
        use super::compute_k_pulses;
        let bits_per_frame: u32 = 64_u32.saturating_mul(960).saturating_mul(1000) / 48_000;
        let bits_per_band = (bits_per_frame / 21).max(1);
        // All bands must get K ≥ 1 at 64 kbps so that shape information is emitted.
        for band_size in [1usize, 2, 4, 6, 8, 12, 18, 22] {
            let k = compute_k_pulses(band_size, bits_per_band);
            assert!(
                k >= 1,
                "band_size={band_size}: K must be ≥ 1 at 64 kbps, got K={k}"
            );
        }
    }

    #[test]
    fn test_celt_frame_pvq_stereo_nonempty() {
        let pcm: Vec<f32> = (0..1920)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * (i / 2) as f32 / 48_000.0).sin() * 0.3)
            .collect();
        let encoded = encode_celt_frame_pvq(&pcm, 2, 48_000, 128);
        assert!(
            !encoded.is_empty(),
            "stereo PVQ CELT frame must not be empty"
        );
    }
}
