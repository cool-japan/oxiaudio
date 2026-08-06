//! In-crate RFC 6716 CELT bitstream **verifier**.
//!
//! [`parse_celt_frame`] re-reads a frame written by
//! [`crate::opus_celt::encode_celt_frame_conformant_sized`] using this crate's
//! own [`RangeDecoder`], walking the field order of RFC 6716 §4.3 exactly as a
//! standard decoder does: silence flag, post-filter flag, transient flag, intra
//! flag, Laplace coarse energy, TF decisions, spread, dynamic-allocation
//! boosts, allocation trim, the rate allocation (including the band-skip bit),
//! fine-energy raw bits, the PVQ band shapes with their recursive `itheta`
//! splits, and the final fine-energy refinement pass.
//!
//! It performs **no synthesis** — it exists to prove entropy-level
//! conformance. Two properties are checked by callers:
//!
//! 1. Every field the encoder wrote reads back with the value it intended.
//! 2. The range coder's `final_range` after parsing equals the encoder's, which
//!    is the canonical libopus conformance test (`OPUS_GET_FINAL_RANGE`): it can
//!    only match if both sides consumed byte-for-byte the same symbol sequence.
//!
//! # Two things `final_range` cannot see
//!
//! Both bit oxiaudio in practice, so tests must check them separately:
//!
//! * **Dropped raw bits.** `final_range` is the *range* coder's register;
//!   `dec_bits` never touches it. A frame whose range-coded (front) and raw-bit
//!   (back) halves collided — so the writer had to drop end bytes — still
//!   reports a matching range while silently losing fine-energy and sign bits.
//!   Assert [`CeltEncodeTrace::fits`](crate::opus_celt::CeltEncodeTrace::fits)
//!   as well.
//! * **A rule this verifier gets wrong the same way the encoder does.** The
//!   allocator, the pulse cache and the split-decision helpers are *shared*
//!   between encoder and verifier, so a mis-ported rule in any of them is
//!   invisible here — the two sides agree because they are the same code. That
//!   is exactly how the `lm == -1` pulse-cache row survived until oxiaudio
//!   0.2.1. The reference `opus-decoder` round trip is the only witness for that
//!   class of defect and must stay in the conformance gate.
//!
//! # Attribution
//!
//! Field order and arithmetic ported from libopus `celt/celt_decoder.c`,
//! `celt/quant_bands.c`, `celt/bands.c` and `celt/rate.c` (© 2001–2011
//! Xiph.Org Foundation et al., BSD-3-Clause).

use crate::opus_celt_bands::{celt_compute_qn, check_do_split_enc, SPREAD_NORMAL};
use crate::opus_celt_rate::{
    celt_bits2pulses, celt_compute_allocation_enc, celt_get_pulses, celt_init_caps,
    celt_pulses2bits, SkipBitCoder, BITRES, MAX_FINE_BITS,
};
use crate::opus_celt_tables::{
    EBAND_5MS, LOG_N_400, NUM_BANDS_CELT, SMALL_ENERGY_ICDF, SPREAD_ICDF, TF_SELECT_TABLE,
    TRIM_ICDF,
};
use crate::opus_pvq::ncwrs_urow;
use crate::opus_range_dec::RangeDecoder;

/// Everything [`parse_celt_frame`] recovered from a CELT frame.
#[derive(Debug, Clone)]
pub struct CeltFrameParse {
    /// Silence flag (`true` means the whole frame decodes to zeros).
    pub silence: bool,
    /// Post-filter flag (always read when `start_band == 0`).
    pub post_filter: bool,
    /// Transient flag.
    pub transient: bool,
    /// Intra-energy flag.
    pub intra: bool,
    /// Per-band coarse-energy deltas, indexed by CELT band.
    pub coarse_qi: Vec<i32>,
    /// Per-band TF decisions (all zero for a neutral frame).
    pub tf_res: Vec<i32>,
    /// Spread-decision symbol.
    pub spread: i32,
    /// Allocation-trim symbol.
    pub alloc_trim: i32,
    /// Number of bands the allocator kept.
    pub coded_bands: usize,
    /// Per-band fine-energy bit counts.
    pub fine_quant: Vec<i32>,
    /// Per-band fine-energy values actually read.
    pub fine_bits: Vec<u32>,
    /// Number of PVQ leaves decoded (splits included).
    pub pvq_leaves: usize,
    /// Number of `itheta` symbols read.
    pub theta_symbols: usize,
    /// Range-coder position (Q3) after each named stage, for bisecting a desync.
    pub stage_tells: Vec<(&'static str, u32)>,
    /// Final range register — must equal the encoder's.
    pub final_range: u32,
    /// `true` if the decoder ran past the end of the buffer or hit a bad symbol.
    pub error: bool,
}

/// Decoder-side skip-bit source for the shared allocation routine.
struct SkipBitDecoder<'a, 'b> {
    dec: &'a mut RangeDecoder<'b>,
}

impl SkipBitCoder for SkipBitDecoder<'_, '_> {
    fn skip_bit(&mut self) -> bool {
        self.dec.dec_bit_logp(1)
    }
}

/// Laplace symbol decoder (`ec_laplace_decode`, libopus `celt/laplace.c`).
fn ec_laplace_decode(dec: &mut RangeDecoder<'_>, fs0: u32, decay: u32) -> i32 {
    const LAPLACE_MINP: u32 = 1;
    const LAPLACE_NMIN: u32 = 16;
    let freq1 = |fs: u32| -> u32 {
        let ft = 32_768u32
            .saturating_sub(LAPLACE_MINP * (2 * LAPLACE_NMIN))
            .saturating_sub(fs);
        ((ft as u64 * (16_384u32.saturating_sub(decay)) as u64) >> 15) as u32
    };
    let mut val = 0i32;
    let fm = dec.decode(32_768);
    let mut fl = 0u32;
    let mut fs_cur = fs0;
    if fm >= fs_cur {
        val += 1;
        fl = fs_cur;
        fs_cur = freq1(fs_cur) + LAPLACE_MINP;
        while fs_cur > LAPLACE_MINP && fm >= fl + 2 * fs_cur {
            fs_cur *= 2;
            fl += fs_cur;
            fs_cur =
                (((fs_cur - 2 * LAPLACE_MINP) as u64 * decay as u64) >> 15) as u32 + LAPLACE_MINP;
            val += 1;
        }
        if fs_cur <= LAPLACE_MINP {
            let di = ((fm - fl) >> 1) as i32;
            val += di;
            fl += 2 * di as u32 * LAPLACE_MINP;
        }
        if fm < fl + fs_cur {
            val = -val;
        } else {
            fl += fs_cur;
        }
    }
    let fh = (fl + fs_cur).min(32_768);
    dec.update(fl, fh, 32_768);
    val
}

/// Read one PVQ codeword index for a leaf of `n` coefficients and `k` pulses.
///
/// Mirrors `decode_pulses`: `V(N, K)` in wrapping `u32`, read as
/// `dec_uint(nc.max(2))`.
fn read_pvq_leaf(dec: &mut RangeDecoder<'_>, n: usize, k: u32) {
    if k == 0 || n == 0 {
        return;
    }
    let (_u_row, v32, _overflowed) = ncwrs_urow(n, k as usize);
    let _ = dec.dec_uint(v32.max(2));
}

/// Running counters for the recursive band reader.
struct BandReadStats {
    /// Number of PVQ leaves decoded.
    leaves: usize,
    /// Number of `itheta` symbols read.
    thetas: usize,
}

/// Recursively read one band's shape, mirroring `quant_partition_mono`.
fn read_band_with_splits(
    dec: &mut RangeDecoder<'_>,
    band: usize,
    n0: usize,
    lm: i32,
    b: i32,
    remaining_bits: &mut i32,
    stats: &mut BandReadStats,
) {
    if n0 == 1 {
        if *remaining_bits >= (1 << BITRES) {
            let _ = dec.dec_bits(1);
            *remaining_bits -= 1 << BITRES;
        }
        stats.leaves += 1;
        return;
    }

    if check_do_split_enc(band, lm, n0, b) {
        let n = n0 >> 1;
        let lm_new = lm - 1;
        let log_n_val = LOG_N_400.get(band).copied().unwrap_or(0) as i32;
        let pulse_cap = log_n_val + lm_new * (1 << BITRES);
        let offset = (pulse_cap >> 1) - crate::opus_celt_bands::QTHETA_OFFSET;
        let qn = celt_compute_qn(n, b, offset, pulse_cap);

        let mut b_after = b;
        let mut itheta = 0i32;
        if qn != 1 {
            let tell_before = dec.tell_frac() as i32;
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            let fm = dec.decode(ft as u32) as i32;
            let (fl, fs, ith) = if fm < (((qn >> 1) * ((qn >> 1) + 1)) >> 1) {
                let ith = ((isqrt32((8 * fm + 1) as u32) as i32) - 1) >> 1;
                ((ith * (ith + 1)) >> 1, ith + 1, ith)
            } else {
                let ith = (2 * (qn + 1) - (isqrt32((8 * (ft - fm - 1) + 1) as u32) as i32)) >> 1;
                (
                    ft - (((qn + 1 - ith) * (qn + 2 - ith)) >> 1),
                    qn + 1 - ith,
                    ith,
                )
            };
            dec.update(fl as u32, (fl + fs) as u32, ft as u32);
            let qalloc = dec.tell_frac() as i32 - tell_before;
            b_after -= qalloc;
            *remaining_bits -= qalloc;
            itheta = (ith * 16384) / qn;
            stats.thetas += 1;
        }

        let delta = theta_delta(itheta, n);
        let mut mbits = 0.max(b_after.min((b_after - delta) / 2));
        let mut sbits = b_after - mbits;
        let rebalance_before = *remaining_bits;
        if mbits >= sbits {
            read_band_with_splits(dec, band, n, lm_new, mbits, remaining_bits, stats);
            let rebalance = mbits - (rebalance_before - *remaining_bits);
            if rebalance > (3 << BITRES) && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            read_band_with_splits(dec, band, n, lm_new, sbits, remaining_bits, stats);
        } else {
            read_band_with_splits(dec, band, n, lm_new, sbits, remaining_bits, stats);
            let rebalance = sbits - (rebalance_before - *remaining_bits);
            if rebalance > (3 << BITRES) && itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            read_band_with_splits(dec, band, n, lm_new, mbits, remaining_bits, stats);
        }
        return;
    }

    // `lm` stays signed here for the same reason as in the encoder: a four-deep
    // split reaches `lm == -1`, which must select cache row 0.
    let mut q = celt_bits2pulses(band, lm, b.max(0));
    let mut curr_bits = celt_pulses2bits(band, lm, q);
    *remaining_bits -= curr_bits;
    while *remaining_bits < 0 && q > 0 {
        *remaining_bits += curr_bits;
        q -= 1;
        curr_bits = celt_pulses2bits(band, lm, q);
        *remaining_bits -= curr_bits;
    }
    if q != 0 {
        let k = celt_get_pulses(q);
        if k > 0 {
            read_pvq_leaf(dec, n0, k as u32);
        }
    }
    stats.leaves += 1;
}

/// Integer square root used by the triangular `itheta` distribution.
fn isqrt32(v: u32) -> u32 {
    if v == 0 {
        return 0;
    }
    let mut x = v;
    let mut y = (x + 1) >> 1;
    while y < x {
        x = y;
        y = (x + v / x) >> 1;
    }
    x
}

/// `delta` for a decoded `itheta` (see `decode_theta_mono`).
fn theta_delta(itheta: i32, n: usize) -> i32 {
    crate::opus_celt_bands::celt_theta_params(itheta, n).0
}

/// Parse a CELT-only frame body (the packet without its TOC byte).
///
/// `start_band` is `0` for CELT-only and `17` for the hybrid high band; `lm` is
/// the frame-size log scale (3 for 20 ms at 48 kHz).
pub fn parse_celt_frame(frame: &[u8], start_band: usize, lm: usize) -> CeltFrameParse {
    use crate::opus_celt_tables::E_PROB_MODEL;

    let mut dec = RangeDecoder::new(frame);
    let total_bits = frame.len() as i32 * 8;
    let total_bits_q = total_bits << BITRES;
    let mut stage_tells: Vec<(&'static str, u32)> = Vec::new();

    let silence = if start_band == 0 {
        dec.dec_bit_logp(15)
    } else {
        false
    };
    stage_tells.push(("silence", dec.tell_frac()));

    let post_filter = if start_band == 0 && dec.tell() + 16 <= total_bits {
        dec.dec_bit_logp(1)
    } else {
        false
    };
    stage_tells.push(("postfilter", dec.tell_frac()));

    let transient = if lm > 0 && dec.tell() + 3 <= total_bits {
        dec.dec_bit_logp(3)
    } else {
        false
    };
    stage_tells.push(("transient", dec.tell_frac()));

    let intra = if dec.tell() + 3 <= total_bits {
        dec.dec_bit_logp(3)
    } else {
        false
    };
    stage_tells.push(("intra", dec.tell_frac()));

    // ── Coarse energy ─────────────────────────────────────────────────────────
    let prob_model = &E_PROB_MODEL[lm][usize::from(intra)];
    let mut coarse_qi = vec![0i32; NUM_BANDS_CELT];
    for (i, qi) in coarse_qi
        .iter_mut()
        .enumerate()
        .take(NUM_BANDS_CELT)
        .skip(start_band)
    {
        let tell = dec.tell();
        *qi = if total_bits - tell >= 15 {
            let pi = 2 * i.min(20);
            ec_laplace_decode(
                &mut dec,
                (prob_model[pi] as u32) << 7,
                (prob_model[pi + 1] as u32) << 6,
            )
        } else if total_bits - tell >= 2 {
            let q = dec.dec_icdf(&SMALL_ENERGY_ICDF, 2);
            (q >> 1) ^ -(q & 1)
        } else if total_bits - tell >= 1 {
            -i32::from(dec.dec_bit_logp(1))
        } else {
            -1
        };
    }
    stage_tells.push(("coarse", dec.tell_frac()));

    // ── TF decisions ──────────────────────────────────────────────────────────
    let mut tf_res = vec![0i32; NUM_BANDS_CELT];
    {
        let mut budget = total_bits;
        let mut tell = dec.tell();
        let mut logp = if transient { 2i32 } else { 4 };
        let tf_select_rsv = lm > 0 && tell + logp < budget;
        if tf_select_rsv {
            budget -= 1;
        }
        let mut curr = 0i32;
        let mut tf_changed = 0i32;
        for tf in tf_res.iter_mut().take(NUM_BANDS_CELT).skip(start_band) {
            if tell + logp <= budget {
                curr ^= i32::from(dec.dec_bit_logp(logp as u32));
                tell = dec.tell();
                tf_changed |= curr;
            }
            *tf = curr;
            logp = if transient { 4 } else { 5 };
        }
        let idx0 = 4 * usize::from(transient) + tf_changed as usize;
        let idx1 = 4 * usize::from(transient) + 2 + tf_changed as usize;
        if tf_select_rsv
            && lm < TF_SELECT_TABLE.len()
            && idx0 < TF_SELECT_TABLE[lm].len()
            && idx1 < TF_SELECT_TABLE[lm].len()
            && TF_SELECT_TABLE[lm][idx0] != TF_SELECT_TABLE[lm][idx1]
        {
            let _ = dec.dec_bit_logp(1);
        }
    }
    stage_tells.push(("tf", dec.tell_frac()));

    // ── Spread ────────────────────────────────────────────────────────────────
    let spread = if dec.tell() + 4 <= total_bits {
        dec.dec_icdf(&SPREAD_ICDF, 5)
    } else {
        SPREAD_NORMAL
    };
    stage_tells.push(("spread", dec.tell_frac()));

    // ── Dynamic allocation (no boosts written by this encoder) ───────────────
    let cap = celt_init_caps(lm);
    let offsets = vec![0i32; NUM_BANDS_CELT];
    let dynalloc_logp = 6i32;
    let mut tell_q = dec.tell_frac() as i32;
    for &cap_val in cap[start_band..NUM_BANDS_CELT].iter() {
        if tell_q + (dynalloc_logp << BITRES) < total_bits_q && cap_val > 0 {
            let _ = dec.dec_bit_logp(dynalloc_logp as u32);
            tell_q = dec.tell_frac() as i32;
        }
    }
    stage_tells.push(("dynalloc", dec.tell_frac()));

    // ── Allocation trim ───────────────────────────────────────────────────────
    tell_q = dec.tell_frac() as i32;
    let alloc_trim = if tell_q + (6 << BITRES) <= total_bits_q {
        let t = dec.dec_icdf(&TRIM_ICDF, 7);
        tell_q = dec.tell_frac() as i32;
        t
    } else {
        5
    };
    stage_tells.push(("trim", dec.tell_frac()));

    // ── Rate allocation ───────────────────────────────────────────────────────
    let avail_bits = (total_bits_q - tell_q - 1).max(0);
    let alloc = celt_compute_allocation_enc(
        &mut SkipBitDecoder { dec: &mut dec },
        avail_bits,
        &offsets,
        &cap,
        alloc_trim,
        lm,
        start_band,
    );
    stage_tells.push(("alloc", dec.tell_frac()));

    // ── Fine energy ───────────────────────────────────────────────────────────
    let mut fine_bits = vec![0u32; NUM_BANDS_CELT];
    for (i, slot) in fine_bits
        .iter_mut()
        .enumerate()
        .take(NUM_BANDS_CELT)
        .skip(start_band)
    {
        let ebits = alloc.fine_quant[i];
        if ebits > 0 {
            *slot = dec.dec_bits(ebits as u32);
        }
    }
    stage_tells.push(("fine", dec.tell_frac()));

    // ── PVQ band shapes ───────────────────────────────────────────────────────
    let mut stats = BandReadStats {
        leaves: 0,
        thetas: 0,
    };
    let coded_bands = alloc.coded_bands;
    let mut balance = alloc.balance;
    let celt_scale = 1usize << lm;
    for band in start_band..NUM_BANDS_CELT {
        let tell = dec.tell_frac() as i32;
        if band != start_band {
            balance -= tell;
        }
        let remaining_bits = total_bits_q - tell - 1;
        let b = if band < coded_bands {
            let curr_balance = balance / ((coded_bands - band).min(3) as i32);
            (alloc.pulses[band] + curr_balance)
                .clamp(0, 16_383)
                .min(remaining_bits + 1)
        } else {
            0
        };
        let celt_lo = (EBAND_5MS[band] as usize) * celt_scale;
        let _ = celt_lo;
        let width = (EBAND_5MS[band + 1] - EBAND_5MS[band]) as usize;
        let n0 = width << lm;
        let mut rem_bits_band = remaining_bits;
        read_band_with_splits(
            &mut dec,
            band,
            n0,
            lm as i32,
            b,
            &mut rem_bits_band,
            &mut stats,
        );
        balance += alloc.pulses[band] + tell;
    }
    stage_tells.push(("bands", dec.tell_frac()));

    // ── Finalise ──────────────────────────────────────────────────────────────
    let mut bits_left = total_bits - dec.tell();
    'outer: for prio in 0..2i32 {
        for i in start_band..NUM_BANDS_CELT {
            if bits_left < 1 {
                break 'outer;
            }
            if alloc.fine_quant[i] >= MAX_FINE_BITS || alloc.fine_priority[i] != prio {
                continue;
            }
            let _ = dec.dec_bits(1);
            bits_left -= 1;
        }
    }
    stage_tells.push(("finalise", dec.tell_frac()));

    CeltFrameParse {
        silence,
        post_filter,
        transient,
        intra,
        coarse_qi,
        tf_res,
        spread,
        alloc_trim,
        coded_bands,
        fine_quant: alloc.fine_quant.clone(),
        fine_bits,
        pvq_leaves: stats.leaves,
        theta_symbols: stats.thetas,
        stage_tells,
        final_range: dec.final_range(),
        error: dec.is_error(),
    }
}
