//! RFC 6716 §4.1 range (arithmetic) encoder — exact inverse of libopus `ec_enc`.
//!
//! This is a faithful port of the libopus `ec_enc` encoder from `celt/entenc.c`,
//! bit-exact with the `EcDec` decoder in `opus-decoder-0.1.1/src/entropy.rs`.
//!
//! # Design
//!
//! The encoder maintains:
//! - A range interval `[val, val+rng)` that is narrowed for each coded symbol.
//! - A carry buffer (`rem` + `ext`) that defers byte output until carry resolution.
//! - A raw-bit window (`end_window`, `end_buf`) that packs bits from the physical
//!   end of the buffer (LSB-first), mirroring the decoder's `dec_bits()` / `read_byte_from_end()`.
//!
//! On [`RangeEncoder::finish`] the two byte streams are stitched: range bytes at the
//! front, raw-bit bytes at the back — matching the decoder's bidirectional reads.
//!
//! # ICDF convention
//!
//! `enc_icdf` expects a **top-cumulative** inverse CDF table where `icdf[s]` is the
//! probability mass **above** symbol `s` in units of `1/2^ftb`. The final entry must
//! be `0`. This matches `ec_dec_icdf` in the reference decoder exactly.
//!
//! # Porting note
//!
//! The carry-buffer mechanism (`carry_out` + `rem` + `ext`) is the key correctness
//! feature absent in the previous self-consistent variant. Without it, carries from
//! the bottom byte would corrupt the already-emitted bytes above it.

// Constants from `celt/mfrngcod.h` and `celt/entcode.h`.
const EC_SYM_BITS: u32 = 8;
const EC_SYM_MAX: u32 = (1u32 << EC_SYM_BITS) - 1; // 0xFF
const EC_CODE_BITS: u32 = 32;
const EC_CODE_TOP: u32 = 1u32 << (EC_CODE_BITS - 1); // 0x8000_0000
const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS; // 0x0080_0000
/// Bit position of the top byte to shift out in `carry_out`.
///
/// Mirrors `EC_CODE_SHIFT` from libopus `celt/entcode.h`:
/// `EC_CODE_BITS - EC_SYM_BITS - 1 = 23`.  `val` is kept in the low 31 bits,
/// so the top byte to emit is bits `[23..=30]` — i.e. `val >> 23`, **not**
/// `val >> 24`.  Using `>> 24` would drop bit 23 of every emitted byte,
/// corrupting the bitstream while leaving `rng` (final range) untouched.
const EC_CODE_SHIFT: u32 = EC_CODE_BITS - EC_SYM_BITS - 1; // 23
const EC_UINT_BITS: u32 = 8;

/// RFC 6716 range encoder (exact inverse of `EcDec`).
///
/// Encode symbols via [`encode`](Self::encode), [`enc_uint`](Self::enc_uint),
/// [`enc_bits`](Self::enc_bits), etc., then call [`finish`](Self::finish) to
/// produce the packet bytes.
pub struct RangeEncoder {
    /// Range-coded bytes accumulated at the front of the output.
    buf: Vec<u8>,
    /// Current range width. Initialized to `EC_CODE_TOP`.
    rng: u32,
    /// Low end of the current coding interval. Initialized to 0.
    val: u32,
    /// Number of deferred `0xFF` bytes awaiting carry resolution.
    ext: u32,
    /// Buffered byte awaiting carry resolution. -1 means empty.
    rem: i32,
    /// Raw-bit window (LSB-first accumulator for end-packed bits).
    end_window: u32,
    /// Number of valid bits currently in `end_window`.
    nend_bits: i32,
    /// Raw-bit bytes flushed from `end_window` (appended after `buf` on finish).
    end_buf: Vec<u8>,
    /// Total bits logically consumed (including deferred carry bytes).
    ///
    /// Initialised to `EC_CODE_BITS + 1 = 33`, incremented by `EC_SYM_BITS = 8`
    /// in each `enc_normalize` iteration.  Mirrors `nbits_total` in libopus
    /// `celt/entenc.c` (BSD-3-Clause).
    nbits_total: i32,
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self {
            buf: Vec::new(),
            rng: EC_CODE_TOP,
            val: 0,
            ext: 0,
            rem: -1,
            end_window: 0,
            nend_bits: 0,
            end_buf: Vec::new(),
            nbits_total: EC_CODE_BITS as i32 + 1,
        }
    }
}

impl RangeEncoder {
    /// Create a new, empty range encoder.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Internal carry-buffer mechanism ─────────────────────────────────────────

    /// Emit one byte (or defer it) with carry propagation.
    ///
    /// Mirrors `ec_enc_carry_out()` in `celt/entenc.c`.
    ///
    /// `c` is the candidate byte value (may be ≥ 256 when a carry arrived):
    /// - If `c == 0xFF` (as a u32): another deferred byte, increment `ext`.
    /// - Otherwise: flush `rem`, flush `ext` copies of `(0xFF + carry) & 0xFF`,
    ///   then buffer the new low byte.
    fn carry_out(&mut self, c: i32) {
        if (c as u32) != EC_SYM_MAX {
            let carry = (c >> EC_SYM_BITS) as u8; // 0 or 1
            if self.rem >= 0 {
                self.buf.push((self.rem as u8).wrapping_add(carry));
            }
            if self.ext > 0 {
                let v = (EC_SYM_MAX.wrapping_add(carry as u32) & EC_SYM_MAX) as u8;
                for _ in 0..self.ext {
                    self.buf.push(v);
                }
                self.ext = 0;
            }
            self.rem = c & (EC_SYM_MAX as i32);
        } else {
            self.ext += 1;
        }
    }

    /// Renormalize the encoder by shifting out top bytes when `rng <= EC_CODE_BOT`.
    ///
    /// Mirrors `ec_enc_normalize()` in `celt/entenc.c`.
    fn enc_normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.carry_out((self.val >> EC_CODE_SHIFT) as i32);
            // Keep only the bottom 31 bits (below EC_CODE_TOP).
            self.val = (self.val << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            self.rng <<= EC_SYM_BITS;
            self.nbits_total += EC_SYM_BITS as i32;
        }
    }

    // ── Core coding primitives ────────────────────────────────────────────────

    /// Encode symbol in `[fl, fh)` of total `ft` (exact inverse of `ec_dec_update`).
    ///
    /// Mirrors `ec_encode()` in `celt/entenc.c`.
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        let r = self.rng / ft;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(ft - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(ft - fh));
        }
        self.enc_normalize();
    }

    /// Encode symbol with a power-of-two total (`ft == 1<<bits`).
    ///
    /// Mirrors `ec_encode_bin()` in `celt/entenc.c`.
    pub fn encode_bin(&mut self, fl: u32, fh: u32, bits: u32) {
        let r = self.rng >> bits;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul((1u32 << bits) - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul((1u32 << bits) - fh));
        }
        self.enc_normalize();
    }

    /// Encode a single bit with `P(1) = 1/2^logp`.
    ///
    /// Mirrors `ec_enc_bit_logp()` in `celt/entenc.c` **exactly**:
    ///
    /// ```text
    /// r = rng;  l = val;  s = r >> logp;  r -= s;
    /// if value { val = l + r; }
    /// rng = if value { s } else { r };
    /// ```
    ///
    /// The Opus range coder stores symbols in *reversed* frequency order (see
    /// [`encode`](Self::encode): `fl > 0` moves `val` **up**), so the "1" symbol
    /// occupies `[ft-1, ft)` — the top of the frequency range — which the
    /// decoder recognises as `val < rng>>logp`. Writing `rng = s` without the
    /// matching `val += r` encodes the *complement*: prior to oxiaudio 0.2.1
    /// this function emitted `!value`, which silently inverted every CELT
    /// header flag (silence, intra, transient, skip) and every SILK VAD/LBRR
    /// flag. `opus_range_dec::tests::roundtrip_bit_logp_is_not_inverted` pins
    /// the corrected behaviour.
    pub fn enc_bit_logp(&mut self, value: bool, logp: u32) {
        let rng0 = self.rng;
        let s = rng0 >> logp;
        let r = rng0 - s;
        if value {
            self.val = self.val.wrapping_add(r);
            self.rng = s;
        } else {
            self.rng = r;
        }
        self.enc_normalize();
    }

    /// Encode a symbol from a top-cumulative inverse CDF table.
    ///
    /// `icdf[s]` is the probability mass **above** symbol `s` (top-cumulative),
    /// scaled by `1/2^ftb`. The final entry must be `0`.
    ///
    /// Mirrors `ec_enc_icdf()` in `celt/entenc.c`.
    pub fn enc_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(icdf[s - 1] as u32)));
            self.rng = r.wrapping_mul((icdf[s - 1] as u32) - (icdf[s] as u32));
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(icdf[s] as u32));
        }
        self.enc_normalize();
    }

    /// Encode an unsigned integer `fl` in `[0, ft)`.
    ///
    /// Mirrors `ec_enc_uint()` in `celt/entenc.c`.
    pub fn enc_uint(&mut self, fl: u32, ft: u32) {
        debug_assert!(ft > 1);
        let ftm1 = ft - 1;
        let ftb = ec_ilog(ftm1);
        if ftb > EC_UINT_BITS {
            let ftb2 = ftb - EC_UINT_BITS;
            let ft_hi = (ftm1 >> ftb2) + 1;
            let fl_hi = fl >> ftb2;
            self.encode(fl_hi, fl_hi + 1, ft_hi);
            let mask = if ftb2 >= 32 {
                u32::MAX
            } else {
                (1u32 << ftb2) - 1
            };
            self.enc_bits(fl & mask, ftb2);
        } else {
            self.encode(fl, fl + 1, ftm1 + 1);
        }
    }

    /// Pack `bits` raw bits (LSB-first) into the end-window for end-of-packet packing.
    ///
    /// Mirrors `ec_enc_bits()` in `celt/entenc.c`.
    /// These bits are physically placed at the back of the packet; the decoder reads
    /// them with `dec_bits()` / `read_byte_from_end()`.
    ///
    /// `nbits_total` **must** be advanced here, exactly as `ec_dec_bits()` does
    /// on the decoder side: CELT's rate allocation and per-band budgets are
    /// driven by [`tell`](Self::tell) / [`tell_frac`](Self::tell_frac), so an
    /// encoder that "forgets" raw bits reports a smaller budget consumption
    /// than the decoder and the two sides drift apart from the first raw-bit
    /// field onward (fine energy). Pinned by
    /// `opus_range_dec::tests::tell_matches_after_raw_bits`.
    pub fn enc_bits(&mut self, fval: u32, bits: u32) {
        debug_assert!(bits <= 25);
        let mask = if bits >= 32 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        };
        self.end_window |= (fval & mask) << self.nend_bits;
        self.nend_bits += bits as i32;
        while self.nend_bits >= EC_SYM_BITS as i32 {
            self.end_buf.push((self.end_window & EC_SYM_MAX) as u8);
            self.end_window >>= EC_SYM_BITS;
            self.nend_bits -= EC_SYM_BITS as i32;
        }
        self.nbits_total += bits as i32;
    }

    /// Bytes already committed to the **front** (range-coded) stream, including
    /// the byte still held in the carry buffer and any deferred `0xFF` run.
    ///
    /// Together with [`raw_bytes`](Self::raw_bytes) this is what a CBR packer
    /// needs to know whether a target frame size can still hold the stream:
    /// [`finish_to_size`](Self::finish_to_size) has to drop bytes when the two
    /// halves collide, which silently corrupts the packet.
    pub fn range_bytes(&self) -> usize {
        self.buf.len() + usize::from(self.rem >= 0) + self.ext as usize
    }

    /// Bytes already committed to the **back** (raw-bit) stream.
    pub fn raw_bytes(&self) -> usize {
        self.end_buf.len() + usize::from(self.nend_bits > 0)
    }

    // ── Bitstream position ────────────────────────────────────────────────────

    /// Return the number of bits consumed so far (from the range-coder side).
    ///
    /// Mirrors `ec_tell()` in `celt/entcode.c`.  Uses `nbits_total` which
    /// accumulates with each normalization byte, so this correctly tracks
    /// the total bits logically consumed even after many bytes are emitted.
    pub fn tell(&self) -> i32 {
        self.nbits_total - ec_ilog(self.rng) as i32
    }

    /// Return the Q3 fractional bit count consumed by the range coder.
    ///
    /// Mirrors `ec_tell_frac()` in `celt/entcode.c`.  Returns a value in Q3
    /// format (i.e., multiply by 1/8 to get bits).  Used to track the
    /// bit budget for CELT rate allocation.
    ///
    /// Ported from libopus `celt/entcode.c` (BSD-3-Clause).
    pub fn tell_frac(&self) -> u32 {
        const CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
        let nbits = (self.nbits_total as u32) << 3; // Q3
        let l = ec_ilog(self.rng);
        if l < 16 {
            // Defensive: rng is effectively zero, all bits consumed.
            return nbits;
        }
        let r = self.rng >> (l - 16);
        let b_raw = (r >> 12).saturating_sub(8) as usize;
        let b_idx = b_raw.min(7);
        let mut b = b_idx as u32;
        if r > CORRECTION[b_idx] {
            b += 1;
        }
        let l_q3 = (l << 3) + b;
        nbits.saturating_sub(l_q3)
    }

    /// Return the total number of bits written to the end (raw-bit) stream.
    ///
    /// This includes fully-flushed bytes in `end_buf` plus any remaining
    /// bits in `end_window`.  Used to compute padding for fixed-size frames.
    pub fn end_bits_used(&self) -> usize {
        self.end_buf.len() * (EC_SYM_BITS as usize) + self.nend_bits.max(0) as usize
    }

    /// Return the final range value for conformance checking.
    ///
    /// The RFC 6716 test vector checker compares `enc.final_range()` to
    /// `dec.final_range()` after a round-trip.
    pub fn final_range(&self) -> u32 {
        self.rng
    }

    // ── Compat wrappers for existing callers ─────────────────────────────────

    /// Encode a uniform integer in `[0, n)` — compatibility wrapper for `enc_uint`.
    ///
    /// Callers in `opus_silk.rs` and `opus_celt.rs` use this name.
    pub fn encode_uint(&mut self, val: u32, n: u32) {
        if n <= 1 {
            return;
        }
        self.enc_uint(val, n);
    }

    /// Encode a uniform integer `fl` in `[0, ft)` where `ft` may exceed `u32::MAX`.
    ///
    /// Mirrors the multi-pass split strategy of `ec_enc_uint()` for large `ft`:
    /// - If `ft` fits in u32, delegates to [`enc_uint`](Self::enc_uint).
    /// - Otherwise, splits into high-word (top `EC_UINT_BITS` bits range-coded)
    ///   and low-word (remaining bits as raw end-packed bits), recursively.
    ///
    /// Used by the CWRS encoder for large-band CELT PVQ where V(N,K) > 2^32.
    pub fn enc_uint_u64(&mut self, fl: u64, ft: u64) {
        if ft <= 1 {
            return;
        }
        // Fast path: if ft fits in u32, use the standard enc_uint.
        if ft <= u32::MAX as u64 {
            self.enc_uint(fl as u32, ft as u32);
            return;
        }
        // ft > u32::MAX: split into high and low parts.
        // High part: top EC_UINT_BITS bits of (ft-1), range-coded.
        // Low part: remaining bits, raw-packed at end of packet.
        let ftm1 = ft - 1;
        let ftb = 64 - ftm1.leading_zeros(); // ilog64(ftm1)
        let ftb2 = ftb - EC_UINT_BITS; // low bits count
        let ft_hi = ((ftm1 >> ftb2) + 1) as u32;
        let fl_hi = (fl >> ftb2) as u32;
        self.encode(fl_hi, fl_hi + 1, ft_hi);
        // Recursively encode the low bits as raw end-packed bits.
        // For ftb2 > 25 we need multiple enc_bits calls (max 25 bits per call).
        let mut low = fl & ((1u64 << ftb2) - 1);
        let mut bits_left = ftb2;
        while bits_left > 0 {
            let chunk = bits_left.min(25);
            self.enc_bits((low & ((1u64 << chunk) - 1)) as u32, chunk);
            low >>= chunk;
            bits_left -= chunk;
        }
    }

    /// Pack raw bits from the end of the packet — compatibility wrapper for `enc_bits`.
    ///
    /// Unlike the old `encode_bits_raw`, these bits now go to the physical end of the
    /// packet (matching the RFC 6716 decoder's `dec_bits()`).
    pub fn encode_bits_raw(&mut self, val: u32, bits: u32) {
        if bits == 0 {
            return;
        }
        self.enc_bits(val, bits);
    }

    /// Encode a symbol from a top-cumulative inverse CDF — compatibility wrapper.
    ///
    /// `s` is the symbol index, `icdf` is top-cumulative (last entry = 0),
    /// `ft_bits` is the log2 of the total probability mass.
    pub fn encode_icdf(&mut self, s: u32, icdf: &[u8], ft_bits: u8) {
        self.enc_icdf(s as usize, icdf, ft_bits as u32);
    }

    // ── Finalisation ─────────────────────────────────────────────────────────

    /// Flush all state and return the encoded packet bytes.
    ///
    /// Mirrors `ec_enc_done()` in `celt/entenc.c`.
    ///
    /// The output is `buf` (range-coded bytes, front) followed by `end_buf`
    /// (raw-bit bytes, back) — exactly as the decoder expects.
    pub fn finish(mut self) -> Vec<u8> {
        // 1. Compute the shortest terminating val that falls within the current interval.
        // Mirrors `ec_enc_done()` in `celt/entenc.c` exactly.
        let mut l = EC_CODE_BITS - ec_ilog(self.rng);
        let mut msk = (EC_CODE_TOP - 1) >> l;
        let mut end = (self.val.wrapping_add(msk)) & !msk;
        if (end | msk) >= self.val.wrapping_add(self.rng) {
            // Need one more bit: increment l and recompute.
            l += 1;
            msk >>= 1;
            end = (self.val.wrapping_add(msk)) & !msk;
        }

        // 2. Flush the terminating val through carry_out, shifting out one byte at a time.
        let mut l_rem = l as i32;
        while l_rem > 0 {
            self.carry_out((end >> EC_CODE_SHIFT) as i32);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l_rem -= EC_SYM_BITS as i32;
        }

        // 3. Flush any remaining buffered byte (rem) and deferred-0xFF count (ext).
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }

        // 4. Handle the partial end-window byte.
        // In libopus, `ec_enc_bits` writes full bytes to `buf[--storage]` (from the BACK).
        // The partial byte (in `ec_enc_done`) is also written to `buf[--storage]`, landing
        // at a LOWER index than all full bytes.
        //
        // Physical packet layout (address ascending):
        //   [range_bytes...] [...padding...] [partial_byte] [full_n-1] ... [full_0]
        //
        // The decoder reads from the HIGH end: full_0 first, ..., full_{n-1}, partial_byte.
        //
        // Our end_buf has [full_0, full_1, ..., full_{n-1}] (in flush order).
        // The partial byte must be PREPENDED to the reversed end_buf in the output.
        let partial: Option<u8> = if self.nend_bits > 0 {
            // Store partial end-window data in the LOW bits (LSB-first packing).
            // The decoder reads bytes and ORs them into window at the current `available`
            // position, then extracts with `window & ((1<<bits)-1)` (low bits).
            Some((self.end_window & EC_SYM_MAX) as u8)
        } else {
            None
        };

        // 5. Stitch: range bytes at front; partial byte (if any); full end bytes reversed.
        // Full end bytes reversed so that end_buf[0] (first flushed) is at the physical end.
        let mut out = self.buf;
        if let Some(p) = partial {
            out.push(p);
        }
        out.extend(self.end_buf.into_iter().rev());
        out
    }

    /// Flush all state and return exactly `target` bytes, matching the
    /// libopus fixed-buffer layout used by Opus decoders.
    ///
    /// Range-coded bytes fill from index 0 upward; end-packed bytes fill
    /// from index `target-1` downward (highest address first, matching the
    /// decoder's `read_byte_from_end()`).  Any unused middle bytes are zero.
    ///
    /// If the encoded content exceeds `target` (e.g., due to range-coder
    /// termination overhead), range bytes are clamped — the last symbol may
    /// be partially corrupted — but this is preferable to a wrong packet size
    /// that would cause total allocation mismatch in the decoder.
    ///
    /// Mirrors the fixed-buffer contract of `ec_enc_done()` in libopus
    /// `celt/entenc.c` (BSD-3-Clause).
    pub fn finish_to_size(self, target: usize) -> Vec<u8> {
        let (packet, _fit) = self.finish_to_size_checked(target);
        packet
    }

    /// [`finish_to_size`](Self::finish_to_size) plus a flag saying whether the
    /// stream actually fitted.
    ///
    /// The Opus range coder grows a range-coded stream from the front of the
    /// packet and a raw-bit stream from the back. `ec_tell()` bounds their
    /// combined *logical* size, but the termination flush (up to two extra
    /// bytes) and the rounding of the raw-bit stream up to a byte boundary can
    /// still push the two halves into each other by a couple of bytes on a
    /// near-full CBR frame. When that happens bytes must be dropped, which
    /// silently corrupts the packet — a standard decoder then reads different
    /// symbols than were written.
    ///
    /// The second return value is `false` in exactly that case, so callers can
    /// retry at a smaller frame size (which keeps encoder and decoder in
    /// agreement, because the decoder derives its bit budget from the emitted
    /// packet length) instead of shipping a corrupt frame.
    pub fn finish_to_size_checked(mut self, target: usize) -> (Vec<u8>, bool) {
        // ── Step 1: compute range-coder termination (same as finish()) ──────────
        let mut l = EC_CODE_BITS - ec_ilog(self.rng);
        let mut msk = (EC_CODE_TOP - 1) >> l;
        let mut end = (self.val.wrapping_add(msk)) & !msk;
        if (end | msk) >= self.val.wrapping_add(self.rng) {
            l += 1;
            msk >>= 1;
            end = (self.val.wrapping_add(msk)) & !msk;
        }
        let mut l_rem = l as i32;
        while l_rem > 0 {
            self.carry_out((end >> EC_CODE_SHIFT) as i32);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l_rem -= EC_SYM_BITS as i32;
        }
        // The termination wrote `l` bits into `ceil(l/8)` bytes; whatever is left
        // over in the last of them is free (and guaranteed zero, because
        // `end = (val + msk) & !msk` clears exactly those low bits).
        let free_bits_in_last = (-l_rem).clamp(0, 7) as u32;
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }

        // ── Step 2: build output buffer ─────────────────────────────────────────
        //
        // Mirrors the tail of libopus `ec_enc_done()`: the gap between the two
        // streams is zeroed and the **partial** raw-bit window is `|=`-merged
        // into `buf[storage - end_offs - 1]`. When a padding gap remains that
        // index is a zero byte, so the merge is a plain store; when the two
        // streams meet exactly it is the *last range byte*, whose low
        // `free_bits_in_last` bits the termination left free — libopus
        // deliberately shares that byte.
        //
        // Before oxiaudio 0.2.1 the partial byte was written at its own index,
        // so a frame always needed one byte more than libopus does. Since the
        // CELT rate allocator fills the budget to the last bit, that phantom
        // byte made `fits` false for ~80 % of frame sizes and the CBR writer
        // retried a byte smaller, giving up to 36 bytes/frame (≈14 kbps) of
        // requested rate for no reason.
        let offs = self.buf.len();
        let end_offs = self.end_buf.len();
        let used = self.nend_bits.clamp(0, 7) as u32;

        // Fit condition, matching libopus's own error conditions:
        //   * the two streams must not overlap (`offs + end_offs <= target`), and
        //   * a partial window must land either in a padding byte or inside the
        //     free low bits of the shared boundary byte.
        let mut fits = offs + end_offs <= target && end_offs < target;
        if fits && used > 0 && offs + end_offs == target {
            fits = used <= free_bits_in_last;
        }

        let mut out = vec![0u8; target];

        // Range bytes at the front (clamped to target).
        let range_len = offs.min(target);
        out[..range_len].copy_from_slice(&self.buf[..range_len]);

        // Whole end bytes at the back: end_buf[0] (first flushed) at the highest
        // address (index target-1), matching read_byte_from_end() ordering.
        let mut back = target;
        for &b in &self.end_buf {
            if back == 0 || back <= range_len {
                break;
            }
            back -= 1;
            out[back] = b;
        }

        // Partial raw-bit window merged into the boundary byte.
        if used > 0 && end_offs < target {
            let idx = target - end_offs - 1;
            let mut window = self.end_window & EC_SYM_MAX;
            if offs + end_offs >= target && free_bits_in_last < used {
                // Busted: keep the range-coder data intact and drop the raw bits
                // that do not fit, exactly as libopus does.
                window &= (1u32 << free_bits_in_last) - 1;
            }
            out[idx] |= window as u8;
        }

        (out, fits)
    }
}

// ── Laplace energy encoder ────────────────────────────────────────────────────
//
// Ported from libopus `celt/laplace.c` (BSD-3-Clause, Xiph.Org Foundation).
// See `opus_celt_tables` module for the full BSD-3-Clause attribution.

/// Minimum per-bucket probability mass for the Laplace model tails.
const LAPLACE_MINP: u32 = 1;
/// Number of minimum-probability tail buckets.
const LAPLACE_NMIN: u32 = 16;

/// Compute frequency mass of the first non-zero bucket (mirrors `laplace_get_freq1`).
///
/// `fs0` is the zero-symbol frequency mass; `decay` is the exponential decay
/// parameter (both from the `E_PROB_MODEL` table via appropriate shifts).
fn laplace_get_freq1(fs0: u32, decay: u32) -> u32 {
    let ft = 32_768u32
        .saturating_sub(LAPLACE_MINP * (2 * LAPLACE_NMIN))
        .saturating_sub(fs0);
    ((ft as u64 * (16_384u32.saturating_sub(decay)) as u64) >> 15) as u32
}

/// Encode a Laplace-distributed coarse-energy delta into the range coder.
///
/// Exact inverse of `ec_laplace_decode` in libopus `celt/laplace.c` (and of the
/// reference decoder's `ec_laplace_decode`): the bucket layout for magnitude
/// `m ≥ 1` is `[fl_m, fl_m + fs_m)` for `−m` and `[fl_m + fs_m, fl_m + 2·fs_m)`
/// for `+m`, with
///
/// ```text
/// fl_1  = fs0,                      fs_1 = laplace_get_freq1(fs0, decay) + MINP
/// fl_k+1 = fl_k + 2·fs_k,           fs_k+1 = ((2·fs_k − 2·MINP)·decay >> 15) + MINP
/// ```
///
/// and a minimum-probability tail (`fs == MINP`) where every further magnitude
/// step costs exactly `2·MINP`.
///
/// # Returns
///
/// The value the decoder will actually reconstruct. The 15-bit frequency space
/// cannot represent arbitrarily large `|qi|`, so the magnitude is **clamped**
/// to the largest representable bucket exactly as libopus's
/// `ec_laplace_encode` does (it writes back through its `int *value`
/// parameter). Callers must use the returned value — not the requested one —
/// for any state they keep in lock-step with the decoder (the coarse-energy
/// `prev` accumulator and the fine-energy residual). Ignoring it silently
/// desynchronises the reconstructed band energies.
pub fn ec_laplace_encode(enc: &mut RangeEncoder, qi: i32, fs0: u32, decay: u32) -> i32 {
    const FT: u32 = 32_768;
    if qi == 0 {
        enc.encode_bin(0, fs0.min(FT), 15);
        return 0;
    }
    let neg = qi < 0;
    let want = qi.unsigned_abs();

    let mut fl = fs0.min(FT);
    let mut fs_cur = laplace_get_freq1(fs0, decay) + LAPLACE_MINP;
    let mut mag = 1u32;

    // Walk the decaying part of the PDF, stopping if another level would not
    // fit inside the 15-bit frequency space.
    while mag < want && fs_cur > LAPLACE_MINP {
        let doubled = fs_cur.saturating_mul(2);
        if fl.saturating_add(doubled).saturating_add(2 * LAPLACE_MINP) >= FT {
            break;
        }
        fl += doubled;
        fs_cur = (((doubled - 2 * LAPLACE_MINP) as u64 * decay as u64) >> 15) as u32 + LAPLACE_MINP;
        mag += 1;
    }

    // Minimum-probability tail: each further magnitude costs 2·MINP.
    if fs_cur <= LAPLACE_MINP && mag < want {
        let room = FT.saturating_sub(fl);
        let steps_available = (room / (2 * LAPLACE_MINP)).saturating_sub(1);
        let di = (want - mag).min(steps_available);
        fl += 2 * di * LAPLACE_MINP;
        mag += di;
    }

    let (lo, hi) = if neg {
        (fl, fl.saturating_add(fs_cur))
    } else {
        (
            fl.saturating_add(fs_cur),
            fl.saturating_add(fs_cur.saturating_mul(2)),
        )
    };
    let lo = lo.min(FT - 1);
    let hi = hi.min(FT).max(lo + 1);
    enc.encode_bin(lo, hi, 15);

    if neg {
        -(mag as i32)
    } else {
        mag as i32
    }
}

// ── Internal utilities ────────────────────────────────────────────────────────

/// Integer log2 — returns `floor(log2(v)) + 1` for v > 0, else 0.
///
/// Matches `EC_ILOG()` semantics from `celt/entcode.h`.
fn ec_ilog(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        32 - v.leading_zeros()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Reference decoder (ported from opus-decoder-0.1.1/src/entropy.rs) ─────
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

        /// Decode a symbol from a top-cumulative inverse CDF table.
        ///
        /// Ported verbatim from `ec_dec_icdf` in
        /// `opus-decoder-0.1.1/src/entropy.rs`, the exact inverse of
        /// [`RangeEncoder::enc_icdf`].
        fn dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> i32 {
            let s0 = self.rng;
            let d = self.val;
            let r = s0 >> ftb;
            let mut ret: i32 = -1;
            let mut s = s0;
            let mut t;
            loop {
                t = s;
                ret += 1;
                let idx = ret as usize;
                if idx >= icdf.len() {
                    self.error = true;
                    return icdf.len() as i32 - 1;
                }
                s = r.wrapping_mul(icdf[idx] as u32);
                if d >= s {
                    break;
                }
            }
            self.val = d - s;
            self.rng = t - s;
            self.normalize();
            ret
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────
    //
    // NOTE on encoder/decoder roundtrip semantics:
    //
    // The RFC 6716 range coder is designed for multi-symbol streams.  After
    // encoding only 1-2 symbols, the encoder may not have flushed enough bytes
    // to prime the decoder's 4-byte initialization window; the decoder then
    // reads zeros past end-of-buffer, which overestimates the decoder `val`.
    //
    // For this reason, single-symbol roundtrip tests only work when the
    // encoded symbol is fl=0 (which maps to the maximum decoder val) OR when
    // the stream is long enough to have triggered encoder normalization.
    //
    // The authoritative conformance check is `final_range()` equality:
    // enc.final_range() (BEFORE finish) must equal dec.final_range() (AFTER
    // decoding the same sequence).  This is the RFC 6716 test-vector check.

    /// Encode a stream of uint-8 symbols (forces encoder normalization).
    /// Chosen to force at least one carry_out byte (rng shrinks below EC_CODE_BOT).
    fn encode_long_stream_uint(vals: &[(u32, u32)]) -> (Vec<u8>, u32) {
        let mut enc = RangeEncoder::new();
        for &(v, n) in vals {
            enc.enc_uint(v, n);
        }
        let enc_range = enc.final_range();
        let bytes = enc.finish();
        (bytes, enc_range)
    }

    #[test]
    fn test_ec_enc_final_range_matches_decoder_short() {
        // Encode 3 symbols, final_range must match the decoder's range after decoding.
        let vals: &[(u32, u32)] = &[(3, 8), (0, 4), (7, 8)];
        let (bytes, enc_range) = encode_long_stream_uint(vals);

        let mut dec = EcDec::new(&bytes);
        for &(_, n) in vals {
            let _ = dec.dec_uint(n);
        }
        let dec_range = dec.final_range();
        assert_eq!(
            enc_range, dec_range,
            "final_range mismatch: enc={enc_range:#010x} dec={dec_range:#010x}"
        );
    }

    #[test]
    fn test_ec_enc_final_range_matches_decoder_long() {
        // Encode many symbols to force normalization; verify final_range.
        let vals: &[(u32, u32)] = &[
            (3, 8),
            (0, 4),
            (7, 8),
            (1, 5),
            (15, 16),
            (0, 3),
            (2, 3),
            (100, 256),
            (255, 256),
            (0, 2),
        ];
        let (bytes, enc_range) = encode_long_stream_uint(vals);

        let mut dec = EcDec::new(&bytes);
        for &(v, n) in vals {
            // Assert bit-exact value recovery, not just final_range: for a
            // uniform coder `final_range` is symbol-value-independent, so it
            // alone would not catch a corrupted byte stream.
            let got = dec.dec_uint(n);
            assert_eq!(got, v, "dec_uint({n}) recovered {got}, expected {v}");
        }
        let dec_range = dec.final_range();
        assert_eq!(
            enc_range, dec_range,
            "final_range mismatch (long): enc={enc_range:#010x} dec={dec_range:#010x}"
        );
    }

    #[test]
    fn test_ec_enc_final_range_20_symbols() {
        // Encode 20 symbols to force multiple normalization steps, then recover
        // every symbol bit-exactly (not merely check final_range, which for a
        // uniform coder is independent of the symbol values encoded).
        let mut enc = RangeEncoder::new();
        let symbols: &[(u32, u32, u32)] = &[
            (0, 1, 4),
            (2, 3, 4),
            (1, 2, 4),
            (3, 4, 4),
            (0, 1, 4),
            (2, 3, 4),
            (1, 2, 4),
            (3, 4, 4),
            (0, 1, 4),
            (1, 2, 4),
            (0, 1, 4),
            (2, 3, 4),
            (1, 2, 4),
            (3, 4, 4),
            (0, 1, 4),
            (2, 3, 4),
            (1, 2, 4),
            (3, 4, 4),
            (0, 1, 4),
            (1, 2, 4),
        ];
        for &(fl, fh, ft) in symbols {
            enc.encode(fl, fh, ft);
        }
        let enc_range = enc.final_range();
        let bytes = enc.finish();

        let mut dec = EcDec::new(&bytes);
        for &(fl, _fh, ft) in symbols {
            let sym = dec.decode(ft);
            dec.update(sym, sym + 1, ft);
            assert_eq!(sym, fl, "20-symbol stream: decoded {sym}, expected {fl}");
        }
        let dec_range = dec.final_range();
        assert_eq!(
            enc_range, dec_range,
            "final_range (20 symbols): enc={enc_range:#010x} dec={dec_range:#010x}"
        );
    }

    #[test]
    fn test_ec_enc_decode_roundtrip_bits() {
        // enc_bits/dec_bits roundtrip: bits are end-packed, NOT affected by
        // the decoder's initialization window, so single-value tests work.
        let cases: &[(u32, u32)] = &[(0b1011, 4), (0b101, 3), (0b11, 2), (0b0, 1)];
        let mut enc = RangeEncoder::new();
        for &(val, bits) in cases {
            enc.enc_bits(val, bits);
        }
        let bytes = enc.finish();

        let mut dec = EcDec::new(&bytes);
        for &(val, bits) in cases {
            let got = dec.dec_bits(bits);
            assert_eq!(got, val, "dec_bits({bits}) → {got:#b} expected {val:#b}");
        }
    }

    #[test]
    fn test_ec_enc_bits_various_widths() {
        let cases: &[(u32, u32)] = &[(0b1111, 4), (0b000, 3), (0b10, 2), (0xFF, 8), (0b10101, 5)];
        for &(val, bits) in cases {
            let mut enc = RangeEncoder::new();
            enc.enc_bits(val, bits);
            let bytes = enc.finish();
            let mut dec = EcDec::new(&bytes);
            let got = dec.dec_bits(bits);
            assert_eq!(
                got, val,
                "enc_bits({val:#b}, {bits}) roundtrip failed: got {got:#b}"
            );
        }
    }

    #[test]
    fn test_ec_enc_empty_finish_valid() {
        // An empty encoder's finish() may produce 0 bytes (valid for libopus convention).
        // The decoder can handle this by reading zeros (graceful overread).
        let enc = RangeEncoder::new();
        let bytes = enc.finish();
        // No assertion on non-empty — the RFC 6716 encoder produces 0 bytes for an
        // empty stream. This test verifies that finish() doesn't panic.
        let _ = bytes;
    }

    #[test]
    fn test_ec_enc_final_range_matches_uint_and_bits() {
        // Mix of range-coded uint and end-packed bits, verify final_range.
        let mut enc = RangeEncoder::new();
        // Encode enough range symbols to force normalization.
        for _ in 0..8 {
            enc.enc_uint(3, 8);
        }
        enc.enc_bits(0b110, 3);
        enc.enc_bits(0b10101, 5);
        let enc_range = enc.final_range();
        let bytes = enc.finish();

        let mut dec = EcDec::new(&bytes);
        for _ in 0..8 {
            let _ = dec.dec_uint(8);
        }
        let b0 = dec.dec_bits(3);
        let b1 = dec.dec_bits(5);
        assert_eq!(b0, 0b110, "bits 3-wide roundtrip");
        assert_eq!(b1, 0b10101, "bits 5-wide roundtrip");
        let dec_range = dec.final_range();
        assert_eq!(enc_range, dec_range, "final_range with mixed coding");
    }

    #[test]
    fn test_ec_enc_final_range_uint_long_stream() {
        // Encode a sequence that exercises the multi-byte enc_uint path (n > 256).
        let vals: &[(u32, u32)] = &[
            (0, 1000),
            (999, 1000),
            (500, 1000),
            (0, 256),
            (255, 256),
            (128, 256),
        ];
        let (bytes, enc_range) = encode_long_stream_uint(vals);
        let mut dec = EcDec::new(&bytes);
        for &(_, n) in vals {
            let _ = dec.dec_uint(n);
        }
        let dec_range = dec.final_range();
        assert_eq!(enc_range, dec_range, "final_range for large-range uints");
    }

    #[test]
    fn test_ec_enc_final_range_repeated_symbols() {
        // Repeat a sequence many times so the encoder emits enough bytes
        // for the decoder's initialization window. Then verify final_range.
        // This directly tests the PVQ-style enc_uint sequence.
        let base_vals: &[(u32, u32)] = &[
            (0, 4),
            (3, 4),
            (1, 4),
            (2, 4),
            (1, 18),
            (15, 18),
            (0, 8),
            (5, 8),
            (3, 8),
        ];
        // Repeat 4 times (36 symbols total) to ensure normalization.
        let mut all_vals = Vec::new();
        for _ in 0..4 {
            all_vals.extend_from_slice(base_vals);
        }
        let (bytes, enc_range) = encode_long_stream_uint(&all_vals);
        let mut dec = EcDec::new(&bytes);
        for &(_, n) in &all_vals {
            let _ = dec.dec_uint(n);
        }
        let dec_range = dec.final_range();
        assert_eq!(
            enc_range, dec_range,
            "repeated PVQ seq: enc={enc_range:#010x} dec={dec_range:#010x}"
        );
    }

    // ── Bit-exact symbol-recovery round-trips ───────────────────────────────
    //
    // Unlike the `final_range` checks above (which only prove the encoder and
    // decoder walked the same total interval), these tests decode every symbol
    // back and assert it equals what was encoded — the RFC 6716 §4.1 guarantee
    // that the range coder is losslessly invertible symbol-for-symbol.

    #[test]
    fn test_ec_range_encode_symbol_roundtrip_bitexact() {
        // A long, varied stream of (fl, fh, ft) triples through `encode()`.
        // Decoded via `decode()` + `update()`; every recovered symbol must match.
        let mut symbols: Vec<(u32, u32)> = Vec::new(); // (symbol, ft)
                                                       // Deterministic pseudo-random-ish sequence over several alphabets.
        let alphabets = [4u32, 8, 3, 16, 5, 256, 18, 2];
        let mut state: u32 = 0x1234_5678;
        for i in 0..200 {
            let ft = alphabets[i % alphabets.len()];
            // xorshift for a spread of symbol values.
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let sym = state % ft;
            symbols.push((sym, ft));
        }

        let mut enc = RangeEncoder::new();
        for &(sym, ft) in &symbols {
            enc.encode(sym, sym + 1, ft);
        }
        let enc_range = enc.final_range();
        let bytes = enc.finish();

        let mut dec = EcDec::new(&bytes);
        for (i, &(sym, ft)) in symbols.iter().enumerate() {
            let got = dec.decode(ft);
            dec.update(got, got + 1, ft);
            assert_eq!(
                got, sym,
                "symbol {i} mismatch (ft={ft}): decoded {got} expected {sym}"
            );
        }
        assert!(!dec.error, "decoder flagged an error mid-stream");
        assert_eq!(
            enc_range,
            dec.final_range(),
            "final_range mismatch after bit-exact symbol recovery"
        );
    }

    #[test]
    fn test_ec_range_icdf_roundtrip_bitexact() {
        // Encode a stream through `enc_icdf` and decode with `dec_icdf`.
        // icdf tables are top-cumulative with a trailing 0 (RFC 6716 convention).
        // ftb = 8 → ft = 256.
        let tables: &[(&[u8], u32)] = &[
            (&[224, 160, 96, 32, 0], 8), // 5-symbol table
            (&[128, 0], 8),              // binary
            (&[200, 100, 50, 20, 8, 0], 8),
        ];
        let mut plan: Vec<(usize, usize)> = Vec::new(); // (table_idx, symbol)
        let mut state: u32 = 0x9E37_79B9;
        for i in 0..180 {
            let table_idx = i % tables.len();
            let nsym = tables[table_idx].0.len() - 1;
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let sym = (state as usize) % nsym;
            plan.push((table_idx, sym));
        }

        let mut enc = RangeEncoder::new();
        for &(ti, sym) in &plan {
            let (icdf, ftb) = tables[ti];
            enc.enc_icdf(sym, icdf, ftb);
        }
        let enc_range = enc.final_range();
        let bytes = enc.finish();

        let mut dec = EcDec::new(&bytes);
        for (i, &(ti, sym)) in plan.iter().enumerate() {
            let (icdf, ftb) = tables[ti];
            let got = dec.dec_icdf(icdf, ftb);
            assert_eq!(
                got as usize, sym,
                "icdf symbol {i} mismatch (table {ti}): decoded {got} expected {sym}"
            );
        }
        assert!(!dec.error, "decoder flagged an error mid-stream");
        assert_eq!(
            enc_range,
            dec.final_range(),
            "final_range mismatch after bit-exact icdf recovery"
        );
    }

    #[test]
    fn test_ec_range_mixed_roundtrip_bitexact() {
        // Interleave range-coded symbols, uints, and end-packed raw bits, then
        // recover all three streams bit-exactly in the same decode order.
        let uints: &[(u32, u32)] = &[
            (0, 1000),
            (999, 1000),
            (500, 1000),
            (7, 8),
            (0, 256),
            (255, 256),
            (42, 64),
        ];
        let raw: &[(u32, u32)] = &[(0b1011, 4), (0x1F, 5), (0xAA, 8), (0b1, 1), (0x155, 9)];

        let mut enc = RangeEncoder::new();
        // Encode all range-coded uints first, then all raw end-packed bits.
        // `enc_uint` for ft > 256 itself end-packs low bits, so the raw-bit
        // stream is: [uint low bits in order][explicit raw bits in order].
        // The decoder must consume raw bits in the SAME order (FIFO), which it
        // does by running all `dec_uint` first, then all `dec_bits`.
        for &(v, ft) in uints {
            enc.enc_uint(v, ft);
        }
        for &(v, bits) in raw {
            enc.enc_bits(v, bits);
        }
        let enc_range = enc.final_range();
        let bytes = enc.finish();

        let mut dec = EcDec::new(&bytes);
        for &(v, ft) in uints {
            let got = dec.dec_uint(ft);
            assert_eq!(
                got, v,
                "uint roundtrip: decoded {got} expected {v} (ft={ft})"
            );
        }
        for &(v, bits) in raw {
            let got = dec.dec_bits(bits);
            assert_eq!(
                got, v,
                "raw-bit roundtrip: decoded {got:#x} expected {v:#x}"
            );
        }
        assert!(!dec.error, "decoder flagged an error mid-stream");
        assert_eq!(
            enc_range,
            dec.final_range(),
            "final_range mismatch after mixed bit-exact recovery"
        );
    }
}
