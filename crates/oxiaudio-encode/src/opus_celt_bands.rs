//! CELT band-shape coding: PVQ search, spreading rotation and RFC 6716
//! §4.3.4.3 split-band (`itheta`) coding for the conformant Opus encoder.
//!
//! Split out of `opus_celt.rs` to keep both files under the 2000-line
//! COOLJAPAN limit.
//!
//! # Split-band coding
//!
//! When the per-band budget exceeds the PVQ cache maximum the decoder splits a
//! band into two halves and reads an angle symbol `itheta` that says how the
//! band's energy divides between them: the lower half is reconstructed with
//! gain `cos(θ)` and the upper half with `sin(θ)`, where
//! `θ = itheta · (π/2)/16384`. Each half is then coded recursively.
//!
//! `encode_band_with_splits` implements the *encoder* side of exactly that
//! recursion, including the real angle (measured from the two halves' actual
//! energies, quantised to `qn` levels and range-coded with the triangular
//! distribution), the `delta`-driven `mbits`/`sbits` split, the visit order
//! (`mbits ≥ sbits` decides which half is coded first) and the post-recursion
//! rebalance. Before oxiaudio 0.2.1 the encoder always wrote `itheta = 0`,
//! which forced every split band's upper half to reconstruct as zeros.
//!
//! # Attribution
//!
//! Ported from libopus `celt/bands.c` and `celt/vq.c` (© 2001–2011 Xiph.Org
//! Foundation, Jean-Marc Valin, Timothy B. Terriberry et al., BSD-3-Clause).

use crate::opus_celt::CeltEncodeTrace;
use crate::opus_celt_rate::{celt_bits2pulses, celt_get_pulses, celt_pulses2bits, BITRES};
use crate::opus_celt_tables::{CACHE_BITS_50, CACHE_INDEX_50, LOG_N_400, NUM_BANDS_CELT};
use crate::opus_pvq;
use crate::opus_range::RangeEncoder;

/// CELT spreading mode signalled in the bitstream (`SPREAD_NORMAL` = symbol 2).
///
/// Must match the value written by `encode_celt_body_into` so the encoder's
/// forward `exp_rotation` is the exact inverse of the decoder's.
pub(crate) const SPREAD_NORMAL: i32 = 2;
/// Per-mode spreading factor table (libopus `celt/vq.c`), indexed by `spread-1`.
const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
/// Theta offset for band splitting (libopus `celt/bands.c`).
pub(crate) const QTHETA_OFFSET: i32 = 4;
/// `2^(x/8)` fixed-point table used by `compute_qn` (libopus `celt/bands.c`).
const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];

/// One exponential-rotation stage (libopus `celt/vq.c::exp_rotation1`).
///
/// Ported verbatim from `opus-decoder-0.1.1/src/celt/vq.rs::exp_rotation1` so
/// that the encoder's forward rotation is bit-for-bit the inverse of the
/// decoder's stage. Operates in place on `x`.
fn exp_rotation1(x: &mut [f32], stride: usize, c: f32, s: f32) {
    let len = x.len();
    if len <= stride {
        return;
    }
    for i in 0..(len - stride) {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
    if len <= 2 * stride {
        return;
    }
    for i in (0..=(len - 2 * stride - 1)).rev() {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
}

/// Apply the CELT spreading rotation (libopus `celt/vq.c::exp_rotation`).
///
/// `dir = 1` is the forward (encoder) rotation; `dir = -1` is the inverse used
/// by the decoder. The angle/stride derivation is identical to the decoder's
/// `exp_rotation`, so `exp_rotation(x, 1, …)` followed by the decoder's
/// `exp_rotation(x, -1, …)` (same `stride`, `k`, `spread`) is the identity.
///
/// The `2*k >= N` early-out and the `spread == SPREAD_NONE` early-out match the
/// decoder exactly, keeping both sides in lock-step for every band.
fn exp_rotation(x: &mut [f32], dir: i32, stride: usize, k: i32, spread: i32) {
    let n = x.len();
    if 2 * k >= n as i32 || spread <= 0 {
        return;
    }
    let factor = SPREAD_FACTOR[(spread - 1) as usize];
    let gain = n as f32 / (n as f32 + factor as f32 * k as f32);
    let theta = 0.5 * gain * gain;
    let angle = core::f32::consts::FRAC_PI_2 * theta;
    let c = angle.cos();
    let s = angle.sin();
    let mut stride2 = 0usize;
    if n >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < n {
            stride2 += 1;
        }
    }
    let len = n / stride;
    for i in 0..stride {
        let xs = &mut x[i * len..(i + 1) * len];
        if dir < 0 {
            if stride2 > 0 {
                exp_rotation1(xs, stride2, s, c);
            }
            exp_rotation1(xs, 1, c, s);
        } else {
            exp_rotation1(xs, 1, c, -s);
            if stride2 > 0 {
                exp_rotation1(xs, stride2, s, -c);
            }
        }
    }
}

/// Exact libopus PVQ codebook search (`celt/vq.c::op_pvq_search`, float path).
///
/// Returns the signed integer pulse vector `iy` with L1 norm exactly `k` that
/// maximises the reconstruction quality metric `<x, iy>² / <iy, iy>` — i.e. the
/// squared cosine similarity between the pulse vector and the (already rotated)
/// target `x`. This is the true rate-distortion-optimal shape the decoder will
/// reconstruct, replacing the previous greedy largest-residual heuristic.
///
/// The algorithm mirrors libopus:
/// 1. Strip signs; work on `|x|`.
/// 2. When `k > N/2`, project onto the pyramid (`iy[j] = floor(k·|x[j]|/Σ|x|)`)
///    to place most pulses cheaply, leaving only a few for refinement.
/// 3. Add each remaining pulse to the position that maximises `Rxy²/Ryy`, where
///    `Rxy = <x, iy>` and `Ryy = <iy, iy>` including the candidate pulse.
/// 4. Re-apply the original signs.
///
/// The returned vector always has L1 norm exactly `k`, so `encode_pulses`
/// produces a bitstream the standard decoder accepts, independent of the shape.
fn op_pvq_search(x: &[f32], k: u32) -> Vec<i32> {
    let n = x.len();
    let mut iy = vec![0i32; n];
    if n == 0 || k == 0 {
        return iy;
    }

    let signx: Vec<bool> = x.iter().map(|&v| v < 0.0).collect();
    let mut xabs: Vec<f32> = x.iter().map(|&v| v.abs()).collect();
    // `y[j]` stores `2 * iy[j]` (libopus optimisation avoiding a ×2 in the loop).
    let mut y = vec![0f32; n];
    let mut xy = 0f32;
    let mut yy = 0f32;
    let mut pulses_left = k as i32;

    // Pre-search: project onto the pyramid when K is large relative to N.
    if (k as usize) > (n >> 1) {
        let mut sum: f32 = xabs.iter().sum();
        // Degenerate (all-zero or overflow-prone) input: pin a unit on bin 0.
        if !(sum > 1e-15 && sum < 64.0) {
            xabs[0] = 1.0;
            sum = 1.0;
        }
        let rcp = k as f32 / sum;
        for j in 0..n {
            let pulses = (rcp * xabs[j]).floor() as i32;
            iy[j] = pulses;
            y[j] = pulses as f32;
            yy += y[j] * y[j];
            xy += xabs[j] * y[j];
            y[j] *= 2.0;
            pulses_left -= pulses;
        }
    }

    // Safety valve (e.g. near-silence): dump any surplus onto bin 0.
    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as f32;
        yy += tmp * tmp + tmp * y[0];
        iy[0] += pulses_left;
        pulses_left = 0;
    }

    // Add remaining pulses one at a time, each maximising Rxy²/Ryy.
    for _ in 0..pulses_left {
        yy += 1.0; // the +1 term of (iy+1)² is common to every candidate.
        let mut best_id = 0usize;
        let mut best_ratio = f32::NEG_INFINITY;
        for j in 0..n {
            let rxy = xy + xabs[j];
            let ryy = yy + y[j];
            // ryy is always > 0 here (yy ≥ 1), and rxy ≥ 0 (signs stripped).
            let ratio = rxy * rxy / ryy;
            if ratio > best_ratio {
                best_ratio = ratio;
                best_id = j;
            }
        }
        xy += xabs[best_id];
        yy += y[best_id];
        y[best_id] += 2.0;
        iy[best_id] += 1;
    }

    // Restore the original signs.
    for j in 0..n {
        if signx[j] {
            iy[j] = -iy[j];
        }
    }
    iy
}

/// Rate-distortion-optimal PVQ shape selection + CWRS encode for one CELT band.
///
/// `shape` is the unit-norm target for the band (or split leaf). The decoder
/// reconstructs `exp_rotation(normalize(iy), dir=-1)`, so to hit the target the
/// encoder forward-rotates the target, runs the exact libopus `op_pvq_search`,
/// and range-codes the resulting pulse index via `encode_pulses`. `blocks` is
/// the current MDCT block count (1 for a non-transient 20 ms frame), matching
/// the `blocks` the decoder passes to `alg_unquant`.
///
/// Returns `true` when `V(N, K)` wrapped `u32` for this leaf (see
/// [`CeltEncodeTrace::pvq_index_overflows`]).
fn celt_alg_quant(shape: &[f32], k_pulses: u32, blocks: usize, enc: &mut RangeEncoder) -> bool {
    let n = shape.len();
    if n == 0 || k_pulses == 0 {
        return false;
    }
    // Forward-rotate a working copy of the target, exactly inverting the
    // decoder's post-decode `exp_rotation(x, -1, blocks, k, SPREAD_NORMAL)`.
    let mut xr = shape.to_vec();
    exp_rotation(&mut xr, 1, blocks.max(1), k_pulses as i32, SPREAD_NORMAL);
    let y = op_pvq_search(&xr, k_pulses);
    opus_pvq::encode_pulses(enc, &y)
}

// ── Split-band (`itheta`) coding ─────────────────────────────────────────────

/// Bit-exact `FRAC_MUL16` (libopus `celt/bands.c`).
fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + ((a as i16 as i32) * (b as i16 as i32))) >> 15
}

/// Integer `EC_ILOG`: `floor(log2(v)) + 1` for `v > 0`, else 0.
fn ec_ilog(v: i32) -> i32 {
    if v <= 0 {
        0
    } else {
        32 - (v as u32).leading_zeros() as i32
    }
}

/// Bit-exact cosine approximation (libopus `bitexact_cos`).
///
/// `x` is an angle in Q14 over `[0, 16384]` (i.e. `[0, π/2]`); the result is
/// `≈ 32768·cos(x·π/2/16384)`, clamped to `[1, 32767]`.
///
/// libopus asserts its intermediate `x2 <= 32766` (so the return is at most
/// 32767) and relies on never being called with `|x| < 64`: both `itheta == 0`
/// and `itheta == 16384` are short-circuited by [`celt_theta_params`], and every
/// other reachable `itheta` is a multiple of `16384 / qn` with `qn <= 256`.
/// Enforcing the range here rather than assuming it matters because
/// [`bitexact_log2tan`] shifts by `15 - EC_ILOG(value)`: a return of 32768 makes
/// that shift `-1`, which is a panic in a debug build. The clamp is
/// unobservable for every reachable input (all of which give `tmp >= 1`), so it
/// costs no bit-exactness — see the full-range `final_range` sweep in
/// `tests/m_opus_celt_snr.rs`.
fn bitexact_cos(x: i32) -> i32 {
    let x2 = ((4096 + x.saturating_mul(x)) >> 13).clamp(0, 32767);
    let poly = (32767 - x2) + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2)));
    1 + poly.clamp(0, 32766)
}

/// Bit-exact `log2(tan())` approximation used for split bit rebalancing.
fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ec_ilog(icos.max(1));
    let ls = ec_ilog(isin.max(1));
    // Both arguments come from `bitexact_cos`, which is bounded by 32767, so
    // `EC_ILOG <= 15` and the shift counts are non-negative. `.max(0)` makes
    // that structural rather than a precondition — Rust panics on a negative
    // shift in debug builds, and this function is reachable from the public
    // encoder API.
    let icos_n = icos << (15 - lc).max(0);
    let isin_n = isin << (15 - ls).max(0);
    (ls - lc) * (1 << 11) + frac_mul16(isin_n, frac_mul16(isin_n, -2597) + 7932)
        - frac_mul16(icos_n, frac_mul16(icos_n, -2597) + 7932)
}

/// Return true iff the decoder will split band `band` at level `lm`.
///
/// Mirrors the `do_split` condition at the top of `quant_partition_mono` in
/// libopus `celt/bands.c`.
pub(crate) fn check_do_split_enc(band: usize, lm: i32, n0: usize, b: i32) -> bool {
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

/// Compute the theta quantisation resolution `qn`.
///
/// Mirrors `compute_qn()` in libopus `celt/bands.c` (mono: `stereo = false`).
pub(crate) fn celt_compute_qn(n: usize, b: i32, offset: i32, pulse_cap: i32) -> i32 {
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

/// `(delta, imid, iside)` for a decoded `itheta`, mirroring `decode_theta_mono`.
pub(crate) fn celt_theta_params(itheta: i32, n: usize) -> (i32, i32, i32) {
    if itheta == 0 {
        return (-16384, 32767, 0);
    }
    if itheta == 16384 {
        return (16384, 0, 32767);
    }
    let imid = bitexact_cos(itheta);
    let iside = bitexact_cos(16384 - itheta);
    let delta = frac_mul16((n as i32 - 1) << 7, bitexact_log2tan(iside, imid));
    (delta, imid, iside)
}

/// Range-code one `itheta` index with the triangular distribution the decoder
/// uses for `blocks0 == 1` (non-transient frames).
///
/// Mirrors the `encode` half of `compute_theta` in libopus `celt/bands.c`:
/// symbol `i` in `[0, qn]` has frequency `fs = i+1` below the midpoint and
/// `qn+1-i` above it, over `ft = ((qn>>1)+1)²`.
fn encode_itheta_triangular(enc: &mut RangeEncoder, itheta_q: i32, qn: i32) {
    let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
    let (fl, fs) = if itheta_q <= (qn >> 1) {
        ((itheta_q * (itheta_q + 1)) >> 1, itheta_q + 1)
    } else {
        (
            ft - (((qn + 1 - itheta_q) * (qn + 2 - itheta_q)) >> 1),
            qn + 1 - itheta_q,
        )
    };
    enc.encode(fl.max(0) as u32, (fl + fs).max(1) as u32, ft.max(1) as u32);
}

/// Measure the split angle between a partition's two halves, in Q14.
///
/// Mirrors `stereo_itheta()` in libopus `celt/bands.c` for the mono case:
/// `itheta = round(16384 · (2/π) · atan2(‖upper‖, ‖lower‖))`, so `0` means all
/// the energy is in the lower half and `16384` means all of it is in the upper
/// half.
fn measure_itheta_q14(lower: f32, upper: f32) -> i32 {
    const EPSILON: f32 = 1e-15;
    let mid = (lower * lower + EPSILON).sqrt();
    let side = (upper * upper + EPSILON).sqrt();
    let angle = side.atan2(mid); // [0, π/2]
    let q14 = (16384.0 * core::f32::consts::FRAC_2_PI * angle).round();
    (q14 as i32).clamp(0, 16384)
}

/// Mutable sinks threaded through the band-split recursion: the range encoder,
/// the running Q3 budget the decoder mirrors, and the per-frame trace counters.
///
/// Bundled into one struct so the recursive encoder stays inside the argument
/// budget the COOLJAPAN lint set enforces.
pub(crate) struct BandEncSink<'a> {
    /// Range encoder every symbol is written to.
    pub(crate) enc: &'a mut RangeEncoder,
    /// Per-frame trace counters (stage tells are recorded by the caller).
    pub(crate) trace: &'a mut CeltEncodeTrace,
    /// Running Q3 budget for the band currently being coded.
    pub(crate) remaining_bits: i32,
}

/// Context for [`encode_band_with_splits`]: the spectrum and the band index,
/// which do not change across the recursion.
pub(crate) struct BandEncCtx<'a> {
    /// The full 960-coefficient CELT MDCT spectrum.
    pub celt_spec: &'a [f32],
    /// CELT band index (used for the cache/`log_n` lookups).
    pub band: usize,
}

impl BandEncCtx<'_> {
    /// Spectrum coefficient at CELT index `i` (0 outside the buffer).
    fn coeff(&self, i: usize) -> f32 {
        self.celt_spec.get(i).copied().unwrap_or(0.0)
    }

    /// L2 norm of `n` coefficients starting at CELT index `lo`.
    fn norm(&self, lo: usize, n: usize) -> f32 {
        (0..n)
            .map(|p| {
                let v = self.coeff(lo + p);
                v * v
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Unit-norm shape of `n` coefficients starting at CELT index `lo`.
    ///
    /// A silent partition degenerates to `[1, 0, 0, …]`, which is what the PVQ
    /// search would pick anyway and keeps the vector normalisable.
    fn shape(&self, lo: usize, n: usize) -> Vec<f32> {
        let mut shape: Vec<f32> = (0..n).map(|p| self.coeff(lo + p)).collect();
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
}

/// Encode one CELT band (or split partition), recursing exactly where the
/// decoder's `quant_partition_mono` recurses.
///
/// * `lo` — CELT coefficient index of this partition's first bin.
/// * `n0` — partition width in coefficients.
/// * `lm` — current frame-size log scale (decremented at every split).
/// * `b`  — Q3 bit budget for this partition.
/// * `sink` — the range encoder, the running Q3 budget shared with the
///   decoder's mirror, and the per-frame [`CeltEncodeTrace`] counters.
pub(crate) fn encode_band_with_splits(
    ctx: &BandEncCtx<'_>,
    sink: &mut BandEncSink<'_>,
    lo: usize,
    n0: usize,
    lm: i32,
    b: i32,
) {
    sink.trace.min_partition_lm = sink.trace.min_partition_lm.min(lm);
    // N = 1: only a sign bit (no PVQ shape). The decoder reconstructs
    // `x[0] = if sign != 0 { -1.0 } else { 1.0 }`, so write the real sign.
    if n0 == 1 {
        if sink.remaining_bits >= (1 << BITRES) {
            sink.enc.enc_bits(u32::from(ctx.coeff(lo) < 0.0), 1);
            sink.remaining_bits -= 1 << BITRES;
        }
        sink.trace.pvq_leaves += 1;
        return;
    }

    if check_do_split_enc(ctx.band, lm, n0, b) {
        let n = n0 >> 1;
        let lm_new = lm - 1;

        // `pulse_cap` / `offset` use the *decremented* LM, matching
        // `decode_theta_mono`, which is called after `lm -= 1`.
        let log_n_val = LOG_N_400.get(ctx.band).copied().unwrap_or(0) as i32;
        let pulse_cap = log_n_val + lm_new * (1 << BITRES);
        let offset = (pulse_cap >> 1) - QTHETA_OFFSET;
        let qn = celt_compute_qn(n, b, offset, pulse_cap);

        let mut b_after = b;
        let mut itheta = 0i32;
        if qn != 1 {
            let itheta_q14 = measure_itheta_q14(ctx.norm(lo, n), ctx.norm(lo + n, n));
            let itheta_q = (((itheta_q14 * qn) + 8192) >> 14).clamp(0, qn);
            let tell_before = sink.enc.tell_frac() as i32;
            encode_itheta_triangular(sink.enc, itheta_q, qn);
            let qalloc = sink.enc.tell_frac() as i32 - tell_before;
            b_after -= qalloc;
            sink.remaining_bits -= qalloc;
            itheta = (itheta_q * 16384) / qn;
            sink.trace.theta_symbols += 1;
        }

        // `blocks0 == 1` for a non-transient 20 ms frame, so the decoder's
        // `blocks0 > 1` delta adjustment never fires here.
        let (delta, _imid, _iside) = celt_theta_params(itheta, n);
        let mut mbits = 0.max(b_after.min((b_after - delta) / 2));
        let mut sbits = b_after - mbits;

        let rebalance_before = sink.remaining_bits;
        if mbits >= sbits {
            encode_band_with_splits(ctx, sink, lo, n, lm_new, mbits);
            let rebalance = mbits - (rebalance_before - sink.remaining_bits);
            if rebalance > (3 << BITRES) && itheta != 0 {
                sbits += rebalance - (3 << BITRES);
            }
            encode_band_with_splits(ctx, sink, lo + n, n, lm_new, sbits);
        } else {
            encode_band_with_splits(ctx, sink, lo + n, n, lm_new, sbits);
            let rebalance = sbits - (rebalance_before - sink.remaining_bits);
            if rebalance > (3 << BITRES) && itheta != 16384 {
                mbits += rebalance - (3 << BITRES);
            }
            encode_band_with_splits(ctx, sink, lo, n, lm_new, mbits);
        }
        return;
    }

    // No split: standard bits2pulses → K → alg_quant path, mirroring the
    // decoder's q-reduction loop so `remaining_bits` never goes negative.
    //
    // `lm` is passed through **signed**: a four-deep split leaves `lm == -1`,
    // which selects cache row 0 in `celt_bits2pulses`. Clamping it to 0 here
    // (as this code did before oxiaudio 0.2.1) picked row 1 and gave the leaf a
    // different pulse count than `quant_partition_mono` derives.
    let mut q_enc = celt_bits2pulses(ctx.band, lm, b.max(0));
    let mut cost_enc = celt_pulses2bits(ctx.band, lm, q_enc);
    sink.remaining_bits -= cost_enc;
    while sink.remaining_bits < 0 && q_enc > 0 {
        sink.remaining_bits += cost_enc;
        q_enc -= 1;
        cost_enc = celt_pulses2bits(ctx.band, lm, q_enc);
        sink.remaining_bits -= cost_enc;
    }
    if q_enc != 0 {
        let k = celt_get_pulses(q_enc);
        if k > 0 {
            let shape = ctx.shape(lo, n0);
            // Non-transient 20 ms CELT frame ⇒ blocks = 1 at every split leaf,
            // matching the `blocks` value the decoder passes to `alg_unquant`.
            sink.trace.max_leaf_pulses = sink.trace.max_leaf_pulses.max(k as u32);
            if celt_alg_quant(&shape, k as u32, 1, sink.enc) {
                sink.trace.pvq_index_overflows += 1;
            }
        }
    }
    sink.trace.pvq_leaves += 1;
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        bitexact_cos, celt_theta_params, exp_rotation, measure_itheta_q14, op_pvq_search,
        SPREAD_NORMAL,
    };

    #[test]
    fn bitexact_cos_endpoints() {
        // `bitexact_cos` is only ever called for `0 < itheta < 16384`; the two
        // extremes are short-circuited by `celt_theta_params`, and 16384 would
        // overflow the i16 domain the fixed-point polynomial works in (exactly
        // as in libopus). Check the interior instead: cos(0) ≈ 32768 and the
        // value decreases monotonically towards 0 as itheta → 16384.
        assert!((bitexact_cos(0) - 32768).abs() <= 2, "{}", bitexact_cos(0));
        assert!(
            bitexact_cos(16383) < 32,
            "cos near π/2 must be ≈0, got {}",
            bitexact_cos(16383)
        );
        let mut prev = bitexact_cos(0);
        for x in (512..16384).step_by(512) {
            let c = bitexact_cos(x);
            assert!(
                c < prev,
                "bitexact_cos must decrease: {x} → {c} (prev {prev})"
            );
            prev = c;
        }
    }

    #[test]
    fn measure_itheta_matches_energy_split() {
        // All energy in the lower half → itheta 0; all in the upper → 16384.
        assert_eq!(measure_itheta_q14(1.0, 0.0), 0);
        assert_eq!(measure_itheta_q14(0.0, 1.0), 16384);
        // Equal energy → 45° → half of full scale.
        let half = measure_itheta_q14(1.0, 1.0);
        assert!(
            (half - 8192).abs() <= 2,
            "equal split must map near 8192, got {half}"
        );
    }

    #[test]
    fn theta_params_preserve_unit_energy() {
        // mid² + side² must stay ≈ 1 so a split band keeps its total energy.
        for itheta in [1, 2000, 4096, 8192, 12000, 16383] {
            let (_d, imid, iside) = celt_theta_params(itheta, 8);
            let mid = imid as f32 / 32768.0;
            let side = iside as f32 / 32768.0;
            let e = mid * mid + side * side;
            assert!(
                (e - 1.0).abs() < 0.02,
                "itheta {itheta}: mid²+side² = {e}, expected ≈1"
            );
        }
    }

    // ── op_pvq_search (exact libopus RDO shape search) ────────────────────────

    /// The search must always emit a pulse vector whose L1 norm equals K
    /// exactly — this is what makes the CWRS index (and therefore the whole
    /// bitstream) decodable regardless of the input shape.
    #[test]
    fn test_op_pvq_search_l1_norm_equals_k() {
        let cases: &[(&[f32], u32)] = &[
            (&[1.0, 0.0, 0.0, 0.0], 3),
            (&[0.5, 0.5, 0.5, 0.5], 5),
            (&[0.9, -0.1, 0.3, -0.2, 0.05], 7),
            (&[-1.0, 0.0], 1),
            (&[0.1; 16], 20), // K > N/2 ⇒ exercises the pyramid pre-search
            (&[0.0; 8], 4),   // degenerate all-zero input
        ];
        for &(x, k) in cases {
            let y = op_pvq_search(x, k);
            let l1: u32 = y.iter().map(|v| v.unsigned_abs()).sum();
            assert_eq!(l1, k, "op_pvq_search({x:?}, {k}) L1 norm {l1} != {k}");
            assert_eq!(y.len(), x.len(), "output length must match input");
        }
    }

    /// For a single dominant component the search must place every pulse on that
    /// bin with the correct sign — the optimal PVQ solution.
    #[test]
    fn test_op_pvq_search_concentrates_on_peak() {
        let x = [0.02f32, -0.98, 0.05, -0.01];
        let y = op_pvq_search(&x, 4);
        assert_eq!(y, vec![0, -4, 0, 0], "all pulses belong on the peak bin");
    }

    /// The RDO search must never score worse than the greedy largest-residual
    /// heuristic on the optimisation metric `<x,y>²/<y,y>` (higher = better).
    /// This is the property that lets the SNR gate be tightened.
    #[test]
    fn test_op_pvq_search_beats_or_ties_greedy() {
        fn metric(x: &[f32], y: &[i32]) -> f32 {
            let xy: f32 = x.iter().zip(y).map(|(&a, &b)| a * b as f32).sum();
            let yy: f32 = y.iter().map(|&b| (b * b) as f32).sum();
            if yy == 0.0 {
                0.0
            } else {
                xy * xy / yy
            }
        }
        fn greedy(x: &[f32], k: u32) -> Vec<i32> {
            let mut y = vec![0i32; x.len()];
            let mut mag: Vec<f32> = x.iter().map(|v| v.abs()).collect();
            let step = 1.0 / k as f32;
            for _ in 0..k {
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
            for (i, yi) in y.iter_mut().enumerate() {
                if x[i] < 0.0 {
                    *yi = -*yi;
                }
            }
            y
        }
        let shapes: &[&[f32]] = &[
            &[0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05],
            &[0.7, -0.5, 0.3, -0.25, 0.2, -0.1],
            &[0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        ];
        for x in shapes {
            for k in 1u32..=10 {
                let opt = op_pvq_search(x, k);
                let grd = greedy(x, k);
                let m_opt = metric(x, &opt);
                let m_grd = metric(x, &grd);
                assert!(
                    m_opt >= m_grd - 1e-6,
                    "RDO metric {m_opt:.6} < greedy {m_grd:.6} for x={x:?} k={k}"
                );
            }
        }
    }

    /// `exp_rotation` forward then inverse (same stride/k/spread) is the
    /// identity, guaranteeing the encoder's rotation exactly inverts the
    /// decoder's post-decode rotation.
    #[test]
    fn test_exp_rotation_forward_inverse_identity() {
        let orig = [0.1f32, -0.3, 0.5, 0.2, -0.4, 0.15, 0.05, -0.25];
        for k in 1i32..=3 {
            let mut v = orig.to_vec();
            exp_rotation(&mut v, 1, 1, k, SPREAD_NORMAL);
            exp_rotation(&mut v, -1, 1, k, SPREAD_NORMAL);
            for (a, b) in v.iter().zip(orig.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "rotation round-trip drift: {a} vs {b} (k={k})"
                );
            }
        }
    }

    #[test]
    fn pvq_search_hits_exact_pulse_count() {
        let x = [0.5f32, -0.3, 0.7, 0.1, -0.4, 0.2];
        for k in 1..12u32 {
            let y = op_pvq_search(&x, k);
            let l1: i32 = y.iter().map(|v| v.abs()).sum();
            assert_eq!(l1 as u32, k, "K={k} must place exactly K pulses");
        }
    }
}
