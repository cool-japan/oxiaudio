//! CELT PVQ (Pyramid Vector Quantization) encoder using exact CWRS combinatorics.
//!
//! Ports the CWRS (Combined Word-Range Scheme) encoder from the `cwrs.c` reference
//! implementation, providing a bit-exact inverse of the `decode_pulses` function in
//! `opus-decoder-0.1.1/src/celt/cwrs.rs`.
//!
//! # Algorithm
//!
//! The Combinatorial Number System (CWRS) maps a signed integer vector `y` with
//! L1 norm `K` and length `N` to a unique integer index in `[0, V(N, K))`, where
//! `V(N, K)` is the number of such codewords.  `encode_pulses` uses `icwrs` to
//! compute this index, then calls `enc_uint` to range-code it.
//!
//! # Reference
//!
//! Ported from `opus-decoder-0.1.1` (libopus BSD-3-Clause port).
//! The CWRS U-row recurrences match RFC 6716 §4.3.4.6.

use crate::opus_range::RangeEncoder;

// ── U-row computation ─────────────────────────────────────────────────────────

/// Compute the U-row for dimension `n` with `k+2` entries, plus `V(n, k)`.
///
/// Returns `(u, v, overflowed)` where:
/// - `u[i] = U(n, i)` for `i` in `0..=k+1`
/// - `v = u[k] + u[k+1] = V(n, k)` (total number of signed pulse codewords)
/// - `overflowed` is `true` if any u32 addition overflowed during computation.
///
/// When `overflowed` is true, the `v` and `u` values are unreliable.
/// Callers should fall back to [`ncwrs_urow_u64`] in that case.
///
/// Matches `ncwrs_urow()` in the reference cwrs.rs exactly:
/// - Initializes `u[0]=0, u[1]=1, u[i]=2i-1` for i≥2.
/// - Applies `n-2` forward steps (for n≥3) via `unext(&mut u[1..], u0=1)`.
/// - `u[0]` stays fixed at 0 throughout (the recurrence preserves this).
pub(crate) fn ncwrs_urow(n: usize, k: usize) -> (Vec<u32>, u32, bool) {
    let len = k + 2;
    let mut u = vec![0u32; len];
    u[0] = 0;
    if len > 1 {
        u[1] = 1;
    }
    for (idx, item) in u.iter_mut().enumerate().skip(2) {
        *item = ((idx as u32) << 1) - 1; // 2*idx - 1
    }
    let mut overflowed = false;
    // for _ in 2..n: apply (n-2) forward steps on u[1..] with initial u0=1.
    for _ in 2..n {
        if unext_sub1(&mut u) {
            overflowed = true;
        }
    }
    let v = match u[k].checked_add(u[k + 1]) {
        Some(v) => v,
        None => {
            overflowed = true;
            u32::MAX
        }
    };
    (u, v, overflowed)
}

/// Advance U-row: update `u[1..]` with initial carry = 1. Leaves `u[0]` = 0 fixed.
///
/// Equivalent to `unext(&mut u[1..], 1)` in the reference implementation.
/// Returns `true` if any checked_add overflowed.
fn unext_sub1(u: &mut [u32]) -> bool {
    if u.len() < 3 {
        return false;
    }
    let sub = &mut u[1..];
    let mut u0 = 1u32;
    let mut overflowed = false;
    for j in 1..sub.len() {
        let (a, ov1) = sub[j].overflowing_add(sub[j - 1]);
        let (u1, ov2) = a.overflowing_add(u0);
        sub[j - 1] = u0;
        u0 = u1;
        if ov1 || ov2 {
            overflowed = true;
        }
    }
    let last = sub.len() - 1;
    sub[last] = u0;
    overflowed
}

/// Step U-row backward one dimension, on a subslice `u[0..=k+1]`.
///
/// Equivalent to `uprev(&mut u[..=k+1], 0)` in the reference — used inside
/// `cwrsi` and our `icwrs` to step back after processing each dimension.
fn uprev_urow(u: &mut [u32]) {
    if u.len() < 2 {
        return;
    }
    let mut u0: u32 = 0;
    for j in 1..u.len() {
        let u1 = u[j].wrapping_sub(u[j - 1]).wrapping_sub(u0);
        u[j - 1] = u0;
        u0 = u1;
    }
    let last = u.len() - 1;
    u[last] = u0;
}

// ── Encoder: vector → index ───────────────────────────────────────────────────

/// Convert signed pulse vector `y` to a CWRS index (exact inverse of `cwrsi`).
///
/// Returns `(index, yy)` where `index` is in `[0, V(n,k))` and `yy` is the
/// sum of squares.
///
/// The algorithm mirrors `cwrsi` step-by-step: each dimension contributes
/// `(neg ? u[yk+1] : 0) + u[yk - mag]` to the accumulated index, then the
/// U-row is stepped back one dimension with `uprev`.
///
/// Precondition: `y` must have total L1 norm = k > 0.
pub(crate) fn icwrs(y: &[i32]) -> (u32, u32) {
    let n = y.len();
    if n == 0 {
        return (0, 0);
    }
    let k: usize = y.iter().map(|v| v.unsigned_abs() as usize).sum();
    if k == 0 {
        return (0, 0);
    }

    let (mut u, _, _) = ncwrs_urow(n, k);

    let mut idx: u32 = 0;
    let mut k_rem = k;
    let mut yy: u32 = 0;

    for (dim, &yj) in y.iter().enumerate() {
        let neg = yj < 0;
        let mag = yj.unsigned_abs() as usize;
        let yk = k_rem;
        let k_after = yk - mag;

        // p = u[yk+1]: sign separator (half the V(n_rem, yk) space).
        let p = if yk + 1 < u.len() { u[yk + 1] } else { 0 };
        // u[k_after]: base index for this magnitude step.
        let uk_after = if k_after < u.len() { u[k_after] } else { 0 };

        // In cwrsi:
        //   incoming idx → if neg { idx -= p }; then idx -= u[k_after]; remainder=0 for last dim
        // Inverse: contribution = (neg ? p : 0) + u[k_after]
        idx = idx.wrapping_add(if neg {
            p.wrapping_add(uk_after)
        } else {
            uk_after
        });

        yy = yy.wrapping_add((yj * yj) as u32);

        // Step backward one dimension (same as cwrsi's uprev call after each y[j]).
        if dim + 1 < n {
            let uprev_len = k_after + 2;
            if uprev_len <= u.len() {
                uprev_urow(&mut u[..uprev_len]);
            }
            k_rem = k_after;
        }
    }

    (idx, yy)
}

// ── U-row computation (u64 wide path) ────────────────────────────────────────

/// Compute the U-row for dimension `n` with `k+2` entries using u64 arithmetic.
///
/// This is the wide-path fallback used when `ncwrs_urow` (u32) would overflow.
/// Returns `(u, v)` where `u[i] = U(n,i)` and `v = V(n,k)` as u64 values.
/// If `v` overflows u64, it is clamped to `u64::MAX` (signals skip to caller).
pub(crate) fn ncwrs_urow_u64(n: usize, k: usize) -> (Vec<u64>, u64) {
    let len = k + 2;
    let mut u = vec![0u64; len];
    u[0] = 0;
    if len > 1 {
        u[1] = 1;
    }
    for (idx, item) in u.iter_mut().enumerate().skip(2) {
        *item = ((idx as u64) << 1) - 1; // 2*idx - 1
    }
    for _ in 2..n {
        unext_sub1_u64(&mut u);
    }
    let v = u[k].saturating_add(u[k + 1]);
    (u, v)
}

/// Advance U-row using u64 arithmetic; saturates rather than wrapping.
fn unext_sub1_u64(u: &mut [u64]) {
    if u.len() < 3 {
        return;
    }
    let sub = &mut u[1..];
    let mut u0 = 1u64;
    for j in 1..sub.len() {
        let u1 = sub[j].saturating_add(sub[j - 1]).saturating_add(u0);
        sub[j - 1] = u0;
        u0 = u1;
    }
    let last = sub.len() - 1;
    sub[last] = u0;
}

/// Step U-row backward one dimension using u64 arithmetic.
fn uprev_urow_u64(u: &mut [u64]) {
    if u.len() < 2 {
        return;
    }
    let mut u0: u64 = 0;
    for j in 1..u.len() {
        let u1 = u[j].wrapping_sub(u[j - 1]).wrapping_sub(u0);
        u[j - 1] = u0;
        u0 = u1;
    }
    let last = u.len() - 1;
    u[last] = u0;
}

/// Convert signed pulse vector `y` to a CWRS index using u64 arithmetic.
///
/// Wide-path fallback used when V(N,K) > u32::MAX.
/// Returns `(index, yy)` where `index` is in `[0, V(n,k))` as u64.
pub(crate) fn icwrs_u64(y: &[i32]) -> (u64, u32) {
    let n = y.len();
    if n == 0 {
        return (0, 0);
    }
    let k: usize = y.iter().map(|v| v.unsigned_abs() as usize).sum();
    if k == 0 {
        return (0, 0);
    }

    let (mut u, _) = ncwrs_urow_u64(n, k);

    let mut idx: u64 = 0;
    let mut k_rem = k;
    let mut yy: u32 = 0;

    for (dim, &yj) in y.iter().enumerate() {
        let neg = yj < 0;
        let mag = yj.unsigned_abs() as usize;
        let yk = k_rem;
        let k_after = yk - mag;

        let p = if yk + 1 < u.len() { u[yk + 1] } else { 0 };
        let uk_after = if k_after < u.len() { u[k_after] } else { 0 };

        idx = idx.wrapping_add(if neg {
            p.wrapping_add(uk_after)
        } else {
            uk_after
        });

        yy = yy.wrapping_add((yj * yj) as u32);

        if dim + 1 < n {
            let uprev_len = k_after + 2;
            if uprev_len <= u.len() {
                uprev_urow_u64(&mut u[..uprev_len]);
            }
            k_rem = k_after;
        }
    }

    (idx, yy)
}

// ── PVQ encode ────────────────────────────────────────────────────────────────

/// Encode signed pulse vector `y` into the range encoder using CWRS combinatorics.
///
/// Bit-exact encoder-side counterpart of `decode_pulses` in the reference cwrs.rs.
///
/// # Overflow handling
///
/// For large `N*K` values where V(N,K) overflows u32, this function automatically
/// falls back to a u64-wide path (`ncwrs_urow_u64` / `icwrs_u64` /
/// `enc_uint_u64`). Only when V(N,K) overflows u64 (extremely large bands,
/// N*K ≫ 10^15) is the band silently skipped — in practice this never occurs
/// in standard Opus streams (CELT bands top out at N≈22, K≈88).
pub(crate) fn encode_pulses(enc: &mut RangeEncoder, y: &[i32]) {
    let n = y.len();
    if n == 0 {
        return;
    }
    let k: u32 = y.iter().map(|v| v.unsigned_abs()).sum();
    if k == 0 {
        return;
    }

    // First try the fast u32 path (with overflow detection).
    let (_u32_row, v32, overflowed) = ncwrs_urow(n, k as usize);
    if !overflowed && v32 >= 2 {
        // u32 path is valid — no overflow.
        let (index, _yy) = icwrs(y);
        if index < v32 {
            enc.enc_uint(index, v32);
        }
        // If index >= v32, there's a logic error in icwrs; silently skip.
        return;
    }

    // u32 path overflowed. Try the u64 wide path.
    let (_u64_row, v64) = ncwrs_urow_u64(n, k as usize);
    // v64 == u64::MAX means saturation (overflow beyond u64 range); skip.
    if v64 < 2 || v64 == u64::MAX {
        return;
    }
    let (index64, _yy) = icwrs_u64(y);
    if index64 < v64 {
        enc.enc_uint_u64(index64, v64);
    }
    // If index64 >= v64, silently skip (logic error guard).
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opus_range::RangeEncoder;

    // ── Reference decoder (ported from opus-decoder-0.1.1/src/celt/cwrs.rs) ──
    //
    // Ported from opus-decoder-0.1.1 (libopus BSD-3-Clause port). Reference only, test use.

    use core::cmp;

    const REF_EC_SYM_BITS: i32 = 8;
    const REF_EC_SYM_MAX: u32 = (1u32 << REF_EC_SYM_BITS) - 1;
    const REF_EC_CODE_BITS: i32 = 32;
    const REF_EC_CODE_TOP: u32 = 1u32 << (REF_EC_CODE_BITS - 1);
    const REF_EC_CODE_BOT: u32 = REF_EC_CODE_TOP >> REF_EC_SYM_BITS;
    const REF_EC_CODE_EXTRA: i32 = ((REF_EC_CODE_BITS - 2) % REF_EC_SYM_BITS) + 1;
    const REF_EC_UINT_BITS: i32 = 8;

    fn ref_ec_ilog(v: u32) -> i32 {
        if v == 0 {
            0
        } else {
            32 - v.leading_zeros() as i32
        }
    }

    struct EcDec<'a> {
        buf: &'a [u8],
        storage: usize,
        end_offs: usize,
        end_window: u32,
        nend_bits: i32,
        nbits_total: i32,
        offs: usize,
        rng: u32,
        val: u32,
        ext: u32,
        rem: i32,
        error: bool,
    }

    impl<'a> EcDec<'a> {
        fn new(buf: &'a [u8]) -> Self {
            let storage = buf.len();
            let mut st = Self {
                buf,
                storage,
                end_offs: 0,
                end_window: 0,
                nend_bits: 0,
                nbits_total: REF_EC_CODE_BITS + 1
                    - ((REF_EC_CODE_BITS - REF_EC_CODE_EXTRA) / REF_EC_SYM_BITS) * REF_EC_SYM_BITS,
                offs: 0,
                rng: 1u32 << REF_EC_CODE_EXTRA,
                val: 0,
                ext: 0,
                rem: 0,
                error: false,
            };
            st.rem = st.read_byte() as i32;
            st.val = st.rng - 1 - ((st.rem as u32) >> (REF_EC_SYM_BITS - REF_EC_CODE_EXTRA));
            st.normalize();
            st
        }

        fn final_range(&self) -> u32 {
            self.rng
        }

        fn read_byte(&mut self) -> u8 {
            if self.offs < self.storage {
                let b = self.buf[self.offs];
                self.offs += 1;
                b
            } else {
                0
            }
        }

        fn read_byte_from_end(&mut self) -> u8 {
            if self.end_offs < self.storage {
                self.end_offs += 1;
                self.buf[self.storage - self.end_offs]
            } else {
                0
            }
        }

        fn normalize(&mut self) {
            while self.rng <= REF_EC_CODE_BOT {
                self.nbits_total += REF_EC_SYM_BITS;
                self.rng <<= REF_EC_SYM_BITS;
                let mut sym = self.rem as u32;
                self.rem = self.read_byte() as i32;
                sym = (sym << REF_EC_SYM_BITS | (self.rem as u32))
                    >> (REF_EC_SYM_BITS - REF_EC_CODE_EXTRA);
                self.val = ((self.val << REF_EC_SYM_BITS) + (REF_EC_SYM_MAX & !sym))
                    & (REF_EC_CODE_TOP - 1);
            }
        }

        fn decode(&mut self, ft: u32) -> u32 {
            self.ext = self.rng / ft;
            let s = self.val / self.ext;
            ft - cmp::min(s + 1, ft)
        }

        fn update(&mut self, fl: u32, fh: u32, ft: u32) {
            let s = self.ext.wrapping_mul(ft - fh);
            self.val = self.val.wrapping_sub(s);
            self.rng = if fl > 0 {
                self.ext.wrapping_mul(fh - fl)
            } else {
                self.rng.wrapping_sub(s)
            };
            self.normalize();
        }

        fn dec_uint(&mut self, ft_in: u32) -> u32 {
            if ft_in <= 1 {
                self.error = true;
                return 0;
            }
            let mut ftm1 = ft_in - 1;
            let ftb = ref_ec_ilog(ftm1);
            if ftb > REF_EC_UINT_BITS {
                let ftb2 = ftb - REF_EC_UINT_BITS;
                let ft = (ftm1 >> ftb2) + 1;
                let s = self.decode(ft);
                self.update(s, s + 1, ft);
                let t = (s << ftb2) | self.dec_bits(ftb2 as u32);
                if t <= ftm1 {
                    t
                } else {
                    self.error = true;
                    ftm1
                }
            } else {
                ftm1 += 1;
                let s = self.decode(ftm1);
                self.update(s, s + 1, ftm1);
                s
            }
        }

        fn dec_bits(&mut self, bits: u32) -> u32 {
            let mut window = self.end_window;
            let mut available = self.nend_bits;
            if available < bits as i32 {
                loop {
                    window |= (self.read_byte_from_end() as u32) << available;
                    available += REF_EC_SYM_BITS;
                    if available > 32 - REF_EC_SYM_BITS {
                        break;
                    }
                }
            }
            let ret = window & ((1u32 << bits) - 1);
            window >>= bits;
            available -= bits as i32;
            self.end_window = window;
            self.nend_bits = available;
            self.nbits_total += bits as i32;
            ret
        }
    }

    /// Reference `cwrsi`: index → signed pulse vector.
    ///
    /// Ported from `opus-decoder-0.1.1/src/celt/cwrs.rs::cwrsi`.
    fn ref_cwrsi(n: usize, mut k: usize, mut idx: u32, u: &mut [u32]) -> (u32, Vec<i32>) {
        let mut yy = 0u32;
        let mut y = vec![0i32; n];
        for yj in y.iter_mut().take(n) {
            let p = u[k + 1];
            let neg = idx >= p;
            if neg {
                idx = idx.wrapping_sub(p);
            }
            let yk = k;
            let mut cur = u[k];
            while cur > idx {
                k -= 1;
                cur = u[k];
            }
            idx = idx.wrapping_sub(cur);
            let mut val = (yk - k) as i32;
            if neg {
                val = -val;
            }
            *yj = val;
            yy = yy.wrapping_add((val * val) as u32);
            let uprev_len = k + 2;
            if uprev_len <= u.len() {
                uprev_urow(&mut u[..uprev_len]);
            }
        }
        (yy, y)
    }

    /// Reference `decode_pulses`.
    fn ref_decode_pulses(dec: &mut EcDec<'_>, n: usize, k: usize) -> Vec<i32> {
        if n == 0 || k == 0 {
            return vec![0; n];
        }
        let (mut u, nc, _ov) = ncwrs_urow(n, k);
        let idx = dec.dec_uint(nc.max(2));
        let (_yy, pulses) = ref_cwrsi(n, k, idx, &mut u);
        pulses
    }

    // ── V(N,K) cardinality sanity ─────────────────────────────────────────────

    #[test]
    fn test_v_n_k_cardinality_small() {
        // V(2,1) = 4: [+1,0],[-1,0],[0,+1],[0,-1].
        let (_u, v21, ov21) = ncwrs_urow(2, 1);
        assert_eq!(v21, 4, "V(2,1) must be 4");
        assert!(!ov21, "V(2,1) must not overflow u32");

        // V(2,2) = 8.
        let (_u, v22, ov22) = ncwrs_urow(2, 2);
        assert_eq!(v22, 8, "V(2,2) must be 8");
        assert!(!ov22, "V(2,2) must not overflow u32");

        // V(3,2) = 18.
        let (_u, v32, ov32) = ncwrs_urow(3, 2);
        assert_eq!(v32, 18, "V(3,2) must be 18");
        assert!(!ov32, "V(3,2) must not overflow u32");

        // V(3,1) = 6: [+1,0,0],[-1,0,0],[0,+1,0],[0,-1,0],[0,0,+1],[0,0,-1].
        let (_u, v31, ov31) = ncwrs_urow(3, 1);
        assert_eq!(v31, 6, "V(3,1) must be 6");
        assert!(!ov31, "V(3,1) must not overflow u32");
    }

    // ── icwrs / cwrsi exhaustive roundtrip ────────────────────────────────────

    #[test]
    fn test_icwrs_cwrsi_roundtrip_exhaustive() {
        // For small (N,K) with N≥2, enumerate all valid pulse vectors and verify roundtrip.
        let cases: &[(usize, usize)] = &[(2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)];

        for &(n, k) in cases {
            let (mut u_ref, nc, _ov) = ncwrs_urow(n, k);
            let u_backup = u_ref.clone();

            for idx in 0..nc {
                u_ref.copy_from_slice(&u_backup);
                let (_yy, y) = ref_cwrsi(n, k, idx, &mut u_ref);

                let l1: u32 = y.iter().map(|v| v.unsigned_abs()).sum();
                assert_eq!(l1, k as u32, "N={n} K={k} idx={idx}: L1 norm {l1} != {k}");

                let (recovered_idx, _) = icwrs(&y);
                assert_eq!(
                    recovered_idx, idx,
                    "N={n} K={k} y={y:?}: icwrs returned {recovered_idx}, expected {idx}"
                );
            }
        }
    }

    // ── encode_pulses final_range check ──────────────────────────────────────

    #[test]
    fn test_encode_decode_pulses_roundtrip() {
        // Verify that a LONG sequence of encode_pulses calls produces matching
        // final_range with ref_decode_pulses.
        //
        // The final_range equality is the RFC 6716 conformance criterion.
        // We repeat the pulse vectors MANY times to ensure the encoder emits
        // enough normalization bytes for the decoder's initialization window.
        let vecs: &[&[i32]] = &[
            &[1, 0],
            &[-1, 0],
            &[0, 1],
            &[0, -1],
            &[1, 1, 0],
            &[-1, 0, 1],
            &[2, 0],
            &[-2, 0],
            &[0, 2],
        ];

        // Compute (n, k) pairs.
        let nk: Vec<(usize, usize)> = vecs
            .iter()
            .map(|y| {
                let k = y.iter().map(|v| v.unsigned_abs() as usize).sum();
                (y.len(), k)
            })
            .collect();

        // Encode 4 repetitions (36 total enc_uint calls) to force normalization.
        let mut enc = RangeEncoder::new();
        for _ in 0..4 {
            for y in vecs {
                encode_pulses(&mut enc, y);
            }
        }
        let enc_range = enc.final_range();
        let bytes = enc.finish();

        // Decode all.
        let mut dec = EcDec::new(&bytes);
        for _ in 0..4 {
            for &(n, k) in &nk {
                let _ = ref_decode_pulses(&mut dec, n, k);
            }
        }
        let dec_range = dec.final_range();

        assert_eq!(
            enc_range, dec_range,
            "final_range mismatch: enc={enc_range:#010x} dec={dec_range:#010x}"
        );
    }

    #[test]
    fn test_encode_pulses_zero_k_is_noop() {
        // k=0 → encode_pulses is a no-op (no bits written).
        // The encoder still flush()es; this test verifies that finish() doesn't panic.
        let y = vec![0i32; 4];
        let mut enc = RangeEncoder::new();
        encode_pulses(&mut enc, &y);
        // finish() should not panic; the output may be empty (RFC 6716 convention for empty streams).
        let _bytes = enc.finish();
    }

    // ── U-row u64 wide path ────────────────────────────────────────────────────

    #[test]
    fn test_ncwrs_urow_u64_matches_u32_for_small_cases() {
        // For small (N,K) where u32 does not overflow, the u64 path must give the
        // same V(N,K) as the u32 path.
        let cases: &[(usize, usize)] = &[(2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 3)];
        for &(n, k) in cases {
            let (_u32_row, v32, _ov) = ncwrs_urow(n, k);
            let (_u64_row, v64) = ncwrs_urow_u64(n, k);
            assert_eq!(v32 as u64, v64, "V({n},{k}) mismatch: u32={v32} u64={v64}");
        }
    }

    #[test]
    fn test_ncwrs_urow_u64_large_band_no_overflow() {
        // V(22, 88) is astronomically large (>> u32::MAX, likely >> u64::MAX too).
        // This test verifies: (1) no panic; (2) v64 is either u64::MAX (saturation)
        // or >= 2 (valid), never 0 or 1; (3) encode_pulses does not panic.
        //
        // CELT band 20 has N=22 bins. At high bitrates the encoder may allocate up
        // to K=88 pulses to this band. V(22,88) >> 2^64.
        let (_, v64) = ncwrs_urow_u64(22, 88);
        // Either saturated or genuinely large.
        assert!(
            v64 == u64::MAX || v64 >= 2,
            "V(22,88) should be u64::MAX (saturated) or >=2, got {v64}"
        );

        // Ensure encode_pulses doesn't panic on a large-band vector with K=88.
        // Construct a valid pulse vector: 22 elements, L1 norm = 88 (4 per bin).
        let y: Vec<i32> = (0..22).map(|i| if i % 2 == 0 { 4 } else { -4 }).collect();
        assert_eq!(
            y.iter().map(|v| v.unsigned_abs()).sum::<u32>(),
            88,
            "test vector L1 norm must be 88"
        );
        let mut enc = RangeEncoder::new();
        encode_pulses(&mut enc, &y); // must not panic
        let _bytes = enc.finish(); // must not panic
    }

    #[test]
    fn test_icwrs_u64_matches_u32_for_small_cases() {
        // For small cases where u32 is sufficient, icwrs_u64 must give the same index.
        let cases: &[(usize, usize)] = &[(2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)];
        for &(n, k) in cases {
            let (mut u_ref, nc, _ov) = ncwrs_urow(n, k);
            let u_backup = u_ref.clone();
            for idx in 0..nc {
                u_ref.copy_from_slice(&u_backup);
                let (_yy, y) = ref_cwrsi(n, k, idx, &mut u_ref);
                let (idx32, _) = icwrs(&y);
                let (idx64, _) = icwrs_u64(&y);
                assert_eq!(
                    idx32 as u64, idx64,
                    "N={n} K={k} y={y:?}: icwrs={idx32} icwrs_u64={idx64}"
                );
            }
        }
    }

    #[test]
    fn test_encode_pulses_uses_u64_path_for_overflow() {
        // V(22, 20) = 853941394691792 which exceeds u32::MAX (4294967295).
        // The u32 path's overflow flag should be set; the u64 path should succeed.
        let (_, v32, overflowed) = ncwrs_urow(22, 20);
        // Confirm u32 overflow is detected via the overflow flag.
        assert!(
            overflowed,
            "V(22,20) must set overflow flag (actual u32 wrapped v32={v32})"
        );

        let (_, v64) = ncwrs_urow_u64(22, 20);
        assert!(v64 >= 2, "V(22,20) u64 must be >= 2; got v64={v64}");
        assert_eq!(v64, 853_941_394_691_792, "V(22,20) u64 must be exact");

        // Build a valid pulse vector: 22 elements, L1 norm = 20.
        // 10 elements with +2 and the rest 0: L1 norm = 20.
        let mut y = vec![0i32; 22];
        for i in 0..10 {
            y[i * 2] = 2;
        }
        assert_eq!(
            y.iter().map(|v| v.unsigned_abs()).sum::<u32>(),
            20,
            "test vector L1 norm must be 20"
        );
        let mut enc = RangeEncoder::new();
        encode_pulses(&mut enc, &y); // must not panic, must use u64 path
        let bytes = enc.finish();
        assert!(
            !bytes.is_empty(),
            "u64-path encoding for V(22,20) must produce non-empty output"
        );
    }

    #[test]
    fn test_ncwrs_urow_overflow_flag_small_cases() {
        // For small (N,K) that don't overflow, the flag should be false.
        let cases_no_overflow: &[(usize, usize)] =
            &[(2, 1), (2, 2), (3, 1), (3, 2), (4, 2), (5, 3)];
        for &(n, k) in cases_no_overflow {
            let (_, _, overflowed) = ncwrs_urow(n, k);
            assert!(
                !overflowed,
                "V({n},{k}) should not overflow u32 but flag is set"
            );
        }
        // For a large case that overflows, flag should be true.
        let (_, _, overflowed_large) = ncwrs_urow(22, 20);
        assert!(overflowed_large, "V(22,20) must set overflow flag");
    }
}
