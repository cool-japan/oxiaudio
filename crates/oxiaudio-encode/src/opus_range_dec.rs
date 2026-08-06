//! RFC 6716 §4.1 range (arithmetic) **decoder** — the in-crate inverse of
//! [`crate::opus_range::RangeEncoder`].
//!
//! oxiaudio ships a conformant Opus *encoder*; proving that the bytes it emits
//! are the ones a standard decoder will read back requires an RFC-structured
//! decoder on this side of the wire too. [`RangeDecoder`] is that decoder: a
//! faithful port of libopus `celt/entdec.c` (`ec_dec`), field-for-field
//! equivalent to the `EcDec` in the reference `opus-decoder` crate.
//!
//! It is used for two things:
//!
//! 1. **Entropy round-trip tests** — every [`RangeEncoder`](crate::opus_range::RangeEncoder)
//!    primitive is verified to decode back to exactly the symbol that was
//!    written (`tests` module below plus `tests/m_opus_entropy_roundtrip.rs`).
//! 2. **Bitstream conformance tests** — the CELT/SILK frame writers can be
//!    re-parsed field by field in the exact order a standard decoder consumes
//!    them, so a desynchronised bitstream fails loudly instead of silently
//!    decoding to noise.
//!
//! # Attribution
//!
//! Ported from libopus `celt/entdec.c` / `celt/entcode.c`
//! (© 2001–2011 Xiph.Org Foundation, Jean-Marc Valin, Timothy B. Terriberry
//! et al., BSD-3-Clause).

// Constants from `celt/mfrngcod.h` and `celt/entcode.h`.
const EC_SYM_BITS: i32 = 8;
const EC_SYM_MAX: u32 = (1u32 << EC_SYM_BITS) - 1;
const EC_CODE_BITS: i32 = 32;
const EC_CODE_TOP: u32 = 1u32 << (EC_CODE_BITS - 1);
const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;
const EC_CODE_EXTRA: i32 = ((EC_CODE_BITS - 2) % EC_SYM_BITS) + 1;
const EC_UINT_BITS: i32 = 8;
const BITRES: i32 = 3;
const EC_WINDOW_SIZE: i32 = 32;

#[inline]
fn ec_ilog(v: u32) -> i32 {
    if v == 0 {
        0
    } else {
        32 - v.leading_zeros() as i32
    }
}

/// RFC 6716 §4.1 range decoder over a borrowed packet payload.
///
/// Range-coded symbols are read from the front of the buffer; raw bits
/// ([`dec_bits`](Self::dec_bits)) are read from the back, exactly mirroring
/// [`RangeEncoder::finish`](crate::opus_range::RangeEncoder::finish).
#[derive(Debug, Clone)]
pub struct RangeDecoder<'a> {
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

impl<'a> RangeDecoder<'a> {
    /// Initialise a decoder over `buf` (`ec_dec_init`).
    pub fn new(buf: &'a [u8]) -> Self {
        let storage = buf.len();
        let mut st = Self {
            buf,
            storage,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: EC_CODE_BITS + 1
                - ((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) * EC_SYM_BITS,
            offs: 0,
            rng: 1u32 << EC_CODE_EXTRA,
            val: 0,
            ext: 0,
            rem: 0,
            error: false,
        };
        st.rem = st.read_byte() as i32;
        st.val = st.rng - 1 - ((st.rem as u32) >> (EC_SYM_BITS - EC_CODE_EXTRA));
        st.normalize();
        st
    }

    /// `true` once the decoder has read past the end of the buffer or hit an
    /// out-of-range iCDF symbol. A conformant stream never sets this.
    pub fn is_error(&self) -> bool {
        self.error
    }

    /// Whole bits consumed so far (`ec_tell`).
    pub fn tell(&self) -> i32 {
        self.nbits_total - ec_ilog(self.rng)
    }

    /// Q3 fractional bits consumed so far (`ec_tell_frac`).
    pub fn tell_frac(&self) -> u32 {
        const CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
        let nbits = (self.nbits_total as u32) << BITRES;
        let mut l = ec_ilog(self.rng);
        let r = self.rng >> (l - 16);
        let mut b = ((r >> 12) - 8) as usize;
        b += usize::from(r > CORRECTION[b]);
        l = (l << 3) + b as i32;
        nbits - (l as u32)
    }

    /// Final range register value (`OPUS_GET_FINAL_RANGE`), for stream matching.
    pub fn final_range(&self) -> u32 {
        self.rng
    }

    #[inline]
    fn read_byte(&mut self) -> u8 {
        if self.offs < self.storage {
            let b = self.buf[self.offs];
            self.offs += 1;
            b
        } else {
            0
        }
    }

    #[inline]
    fn read_byte_from_end(&mut self) -> u8 {
        if self.end_offs < self.storage {
            self.end_offs += 1;
            self.buf[self.storage - self.end_offs]
        } else {
            0
        }
    }

    #[inline]
    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.nbits_total += EC_SYM_BITS;
            self.rng <<= EC_SYM_BITS;
            let mut sym = self.rem as u32;
            self.rem = self.read_byte() as i32;
            sym = (sym << EC_SYM_BITS | (self.rem as u32)) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            self.val = ((self.val << EC_SYM_BITS) + (EC_SYM_MAX & !sym)) & (EC_CODE_TOP - 1);
        }
    }

    /// `ec_decode`: return the cumulative frequency of the next symbol.
    ///
    /// Must be followed by [`update`](Self::update) with the `[fl, fh)` range
    /// containing the returned value.
    pub fn decode(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// `ec_dec_update`: commit the symbol range chosen after [`decode`](Self::decode).
    pub fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext.wrapping_mul(ft - fh);
        self.val = self.val.wrapping_sub(s);
        self.rng = if fl > 0 {
            self.ext.wrapping_mul(fh - fl)
        } else {
            self.rng.wrapping_sub(s)
        };
        self.normalize();
    }

    /// `ec_dec_bit_logp`: decode a bit whose "1" probability is `1/2^logp`.
    pub fn dec_bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = d < s;
        if !ret {
            self.val = d - s;
        }
        self.rng = if ret { s } else { r - s };
        self.normalize();
        ret
    }

    /// `ec_dec_icdf`: decode a symbol from a top-cumulative inverse CDF table.
    pub fn dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> i32 {
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

    /// `ec_dec_uint`: decode an unsigned integer uniformly distributed in `[0, ft)`.
    pub fn dec_uint(&mut self, ft_in: u32) -> u32 {
        if ft_in <= 1 {
            self.error = true;
            return 0;
        }
        let mut ftm1 = ft_in - 1;
        let mut ftb = ec_ilog(ftm1);
        if ftb > EC_UINT_BITS {
            ftb -= EC_UINT_BITS;
            let ft = (ftm1 >> ftb) + 1;
            let s = self.decode(ft);
            self.update(s, s + 1, ft);
            let t = (s << ftb) | self.dec_bits(ftb as u32);
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

    /// `ec_dec_bits`: read `bits` raw bits packed from the end of the packet.
    pub fn dec_bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if available < bits as i32 {
            loop {
                window |= (self.read_byte_from_end() as u32) << available;
                available += EC_SYM_BITS;
                if available > EC_WINDOW_SIZE - EC_SYM_BITS {
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::RangeDecoder;
    use crate::opus_range::RangeEncoder;

    /// Deterministic LCG so the round-trip corpus is reproducible.
    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0
        }
    }

    #[test]
    fn roundtrip_encode_update_symbols() {
        let mut rng = Lcg(0xC0FF_EE01);
        let ft = 97u32;
        let syms: Vec<u32> = (0..200).map(|_| rng.next() % ft).collect();
        let mut enc = RangeEncoder::new();
        for &s in &syms {
            enc.encode(s, s + 1, ft);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for (i, &s) in syms.iter().enumerate() {
            let fs = dec.decode(ft);
            dec.update(fs, fs + 1, ft);
            assert_eq!(fs, s, "symbol {i} mismatch");
        }
        assert!(!dec.is_error());
    }

    #[test]
    fn roundtrip_bit_logp_is_not_inverted() {
        // Regression pin: `enc_bit_logp(v)` must decode back as `v`, not `!v`.
        let mut rng = Lcg(0x1234_5678);
        let bits: Vec<(bool, u32)> = (0..300)
            .map(|_| {
                let r = rng.next();
                ((r & 1) == 0, 1 + (r >> 8) % 8)
            })
            .collect();
        let mut enc = RangeEncoder::new();
        for &(v, logp) in &bits {
            enc.enc_bit_logp(v, logp);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for (i, &(v, logp)) in bits.iter().enumerate() {
            let got = dec.dec_bit_logp(logp);
            assert_eq!(got, v, "bit {i} (logp {logp}) decoded as {got}, wrote {v}");
        }
    }

    #[test]
    fn roundtrip_icdf() {
        // Top-cumulative iCDF over 5 symbols with ftb = 8.
        const ICDF: [u8; 5] = [200, 120, 60, 20, 0];
        let mut rng = Lcg(0x55AA_1234);
        let syms: Vec<usize> = (0..250).map(|_| (rng.next() % 5) as usize).collect();
        let mut enc = RangeEncoder::new();
        for &s in &syms {
            enc.enc_icdf(s, &ICDF, 8);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for (i, &s) in syms.iter().enumerate() {
            assert_eq!(dec.dec_icdf(&ICDF, 8) as usize, s, "icdf symbol {i}");
        }
    }

    #[test]
    fn roundtrip_uint_and_raw_bits() {
        let mut rng = Lcg(0x0BAD_F00D);
        let uints: Vec<(u32, u32)> = (0..60)
            .map(|_| {
                let ft = 2 + rng.next() % 100_000;
                (rng.next() % ft, ft)
            })
            .collect();
        let raws: Vec<(u32, u32)> = (0..60)
            .map(|_| {
                let bits = 1 + rng.next() % 24;
                (rng.next() & ((1 << bits) - 1), bits)
            })
            .collect();
        let mut enc = RangeEncoder::new();
        for (i, &(v, ft)) in uints.iter().enumerate() {
            enc.enc_uint(v, ft);
            let (rv, rb) = raws[i];
            enc.enc_bits(rv, rb);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for (i, &(v, ft)) in uints.iter().enumerate() {
            assert_eq!(dec.dec_uint(ft), v, "uint {i}");
            let (rv, rb) = raws[i];
            assert_eq!(dec.dec_bits(rb), rv, "raw bits {i}");
        }
    }

    #[test]
    fn tell_matches_after_raw_bits() {
        // `enc_bits` must advance `nbits_total` exactly like `dec_bits` does, or
        // CELT's tell()-driven rate allocation drifts apart between the two
        // sides once fine-energy raw bits are written.
        let mut enc = RangeEncoder::new();
        let mut expected_tells = Vec::new();
        for i in 0..25u32 {
            enc.encode(i % 5, i % 5 + 1, 5);
            enc.enc_bits(i & 0x7, 3);
            expected_tells.push(enc.tell());
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for (i, want) in expected_tells.iter().enumerate() {
            let fs = dec.decode(5);
            dec.update(fs, fs + 1, 5);
            let _ = dec.dec_bits(3);
            assert_eq!(dec.tell(), *want, "tell diverged at step {i}");
        }
    }

    #[test]
    fn tell_tracks_between_encoder_and_decoder() {
        let mut enc = RangeEncoder::new();
        for i in 0..40u32 {
            enc.encode(i % 7, i % 7 + 1, 7);
        }
        let enc_tell = enc.tell();
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for _ in 0..40 {
            let fs = dec.decode(7);
            dec.update(fs, fs + 1, 7);
        }
        assert_eq!(
            dec.tell(),
            enc_tell,
            "encoder and decoder must agree on bits consumed"
        );
    }
}
