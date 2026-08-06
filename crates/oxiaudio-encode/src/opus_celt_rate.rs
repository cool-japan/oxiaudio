//! CELT bit-allocation (`celt/rate.c` port) for the conformant Opus encoder.
//!
//! Splits out of `opus_celt.rs` so both files stay well under the 2000-line
//! COOLJAPAN limit. Everything here mirrors the *decoder* side exactly — the
//! encoder must reproduce `clt_compute_allocation` / `interp_bits2pulses`
//! bit-for-bit or the two sides disagree about how many pulses each band gets
//! and the bitstream desynchronises.
//!
//! # Attribution
//!
//! Ported from libopus `celt/rate.c` (© 2001–2011 Xiph.Org Foundation,
//! Jean-Marc Valin, Timothy B. Terriberry et al., BSD-3-Clause).

use crate::opus_celt_tables::{
    BAND_ALLOCATION, CACHE_BITS_50, CACHE_CAPS_50, CACHE_INDEX_50, EBAND_5MS, LOG_N_400,
    NUM_BANDS_CELT,
};
use crate::opus_range::RangeEncoder;

/// Coder-direction abstraction for the one conditional bit the CELT band-skip
/// loop exchanges (`interp_bits2pulses` in libopus `celt/rate.c`).
///
/// The allocation itself is identical on both sides — it *must* be, or encoder
/// and decoder disagree about how many pulses each band gets — so the loop is
/// written once and parameterised over how the skip bit is transferred.
pub(crate) trait SkipBitCoder {
    /// Transfer one skip bit. Returns `true` when all remaining bands are kept
    /// (the decoder's `break` case).
    fn skip_bit(&mut self) -> bool;
}

/// Encoder side: always keeps every band (writes `1`).
pub(crate) struct SkipBitEncoder<'a>(pub &'a mut RangeEncoder);

impl SkipBitCoder for SkipBitEncoder<'_> {
    fn skip_bit(&mut self) -> bool {
        self.0.enc_bit_logp(true, 1);
        true
    }
}

/// Q3 fixed-point scale for bit counts, matching libopus CELT `BITRES`.
pub(crate) const BITRES: i32 = 3;
/// Number of allocation levels in `BAND_ALLOCATION`.
const NB_ALLOC_VECTORS: usize = 11;
/// Maximum fine-quant bits per band.
pub(crate) const MAX_FINE_BITS: i32 = 8;
/// Fine-offset for energy allocation.
const FINE_OFFSET: i32 = 21;
/// Number of bisection steps in `interp_bits2pulses`.
const ALLOC_STEPS: i32 = 6;
/// `log2_frac` table for intensity stereo (unused for mono but needed structurally).
const LOG2_FRAC_TABLE: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

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

/// Cache row index for a partition's frame-size scale.
///
/// `lm` is the **current partition's** scale, which the split recursion
/// decrements once per level and which therefore reaches `-1` for a band that
/// splits four times (e.g. band 20 at LM = 3: 176 → 88 → 44 → 22 → 11 bins).
/// libopus indexes `cache.index` with `LM+1`, so `lm = -1` selects row 0 — the
/// 2.5 ms cache. Clamping `lm` to 0 *before* the `+1` (as this code did before
/// oxiaudio 0.2.1) selects row 1 instead, giving those leaves a different
/// pulse count than the decoder derives and desynchronising the bitstream on
/// exactly the high-rate frames where four-deep splits occur.
fn cache_row(lm: i32) -> usize {
    (lm + 1).clamp(0, 4) as usize
}

/// Convert bit count (Q3) to pseudo-pulse index via binary search.
///
/// `lm` is the partition scale and may be `-1`; see [`cache_row`].
///
/// Mirrors `bits2pulses()` in libopus `celt/rate.c` (BSD-3-Clause).
pub(crate) fn celt_bits2pulses(band: usize, lm: i32, bits: i32) -> i32 {
    const LOG_MAX_PSEUDO: usize = 6;
    let lm1 = cache_row(lm); // CACHE_INDEX_50 has 5 rows: lm1 = 0..4
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
/// `lm` is the partition scale and may be `-1`; see [`cache_row`].
///
/// Mirrors `pulses2bits()` in libopus `celt/rate.c` (BSD-3-Clause).
pub(crate) fn celt_pulses2bits(band: usize, lm: i32, pulses: i32) -> i32 {
    if pulses == 0 {
        return 0;
    }
    let lm1 = cache_row(lm);
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
pub(crate) struct CeltAllocResult {
    pub(crate) coded_bands: usize,
    /// Per-band Q3 bit budgets for PVQ (fine_quant already subtracted).
    pub(crate) pulses: Vec<i32>,
    pub(crate) fine_quant: Vec<i32>,
    pub(crate) fine_priority: Vec<i32>,
    /// Residual balance carried from the fine/coarse split (mirrors libopus `balance`).
    pub(crate) balance: i32,
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
pub(crate) fn celt_compute_allocation_enc(
    enc: &mut impl SkipBitCoder,
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
    enc: &mut impl SkipBitCoder,
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
            // The decoder breaks *before* charging the skip bit to `psum`
            // (libopus `celt/rate.c`): only the "keep going / skip this band"
            // branch pays for it. Charging it on the break path too inflates
            // `psum`, shrinks `left` in the final distribution below, and hands
            // every band fewer pulses than the decoder expects — a silent
            // whole-frame bitstream desynchronisation.
            if enc.skip_bit() {
                break;
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{cache_row, celt_bits2pulses, celt_get_pulses, celt_pulses2bits, NUM_BANDS_CELT};

    /// `cache_row` must reproduce libopus's `LM + 1` indexing, including the
    /// `lm == -1` case a four-deep band split produces.
    #[test]
    fn cache_row_matches_lm_plus_one() {
        assert_eq!(cache_row(-1), 0, "lm = -1 selects the 2.5 ms cache row");
        assert_eq!(cache_row(0), 1);
        assert_eq!(cache_row(1), 2);
        assert_eq!(cache_row(2), 3);
        assert_eq!(cache_row(3), 4);
        // Out-of-range inputs saturate instead of indexing past the table.
        assert_eq!(cache_row(-5), 0);
        assert_eq!(cache_row(9), 4);
    }

    /// Rows 0 and 1 of the pulse cache are genuinely different, so clamping
    /// `lm` to 0 before the `+1` (the pre-0.2.1 behaviour) really did change the
    /// emitted pulse count — this is the bug that capped the encoder at
    /// 80 bytes/frame.
    #[test]
    fn lm_minus_one_and_lm_zero_disagree() {
        let mut differing = 0usize;
        for band in 0..NUM_BANDS_CELT {
            for bits in [16i32, 48, 96, 200, 400] {
                let q_m1 = celt_bits2pulses(band, -1, bits);
                let q_0 = celt_bits2pulses(band, 0, bits);
                if celt_get_pulses(q_m1) != celt_get_pulses(q_0)
                    || celt_pulses2bits(band, -1, q_m1) != celt_pulses2bits(band, 0, q_0)
                {
                    differing += 1;
                }
            }
        }
        assert!(
            differing > 0,
            "cache rows 0 and 1 must differ somewhere, otherwise the lm = -1 fix is untestable"
        );
    }

    /// More budget must never buy fewer pulses, and the chosen pseudo-pulse
    /// index must always stay inside its own cache row (a positive `q` whose
    /// `celt_pulses2bits` is 0 would mean the index ran off the row and the
    /// decoder would price the leaf differently).
    #[test]
    fn bits2pulses_is_monotone_and_stays_in_row() {
        for lm in [-1i32, 0, 1, 2, 3] {
            for band in 0..NUM_BANDS_CELT {
                let mut prev_q = 0i32;
                for bits in (0i32..=512).step_by(8) {
                    let q = celt_bits2pulses(band, lm, bits);
                    assert!(
                        q >= prev_q,
                        "band {band} lm {lm}: bits2pulses({bits}) = {q} < previous {prev_q}"
                    );
                    if q > 0 {
                        assert!(
                            celt_pulses2bits(band, lm, q) > 0,
                            "band {band} lm {lm}: q={q} prices at 0 bits — index left its row"
                        );
                        assert!(
                            celt_get_pulses(q) > 0,
                            "band {band} lm {lm}: q={q} must map to at least one pulse"
                        );
                    }
                    prev_q = q;
                }
            }
        }
    }
}
