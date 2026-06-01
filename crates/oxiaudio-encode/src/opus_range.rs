//! Opus-style range (arithmetic) encoder.
//!
//! Produces a compact arithmetic-coded bitstream compatible with the private
//! `RangeDecoder` in `#[cfg(test)]`. The encoder/decoder pair here is
//! **self-consistent** but NOT bit-compatible with RFC 6716 decoders:
//!
//! - Raw bits use `encode_uint_small(v, 2)` per bit (interleaved with the range
//!   code), rather than the RFC §4.1.4 approach of packing raw bits backwards
//!   from the end of the packet.
//! - The priming protocol for the decoder uses 3-byte reads to match the 3-byte
//!   normalize flush.
//!
//! This is intentional: the encoder exists to exercise OGG container framing
//! and CELT quantization structure, not to produce decodable reference streams.

/// Range encoder.
///
/// Encodes a sequence of symbols into a compact arithmetic code. Call
/// [`Self::encode_icdf`] or [`Self::encode_bits_raw`] to add symbols, then
/// [`Self::finish`] to flush and return the bytes.
pub struct RangeEncoder {
    low: u32,
    rng: u32,
    output: Vec<u8>,
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeEncoder {
    /// Create a new, empty range encoder.
    pub fn new() -> Self {
        Self {
            low: 0,
            rng: 0x8000_0000u32,
            output: Vec::new(),
        }
    }

    /// Encode one symbol using an inverse-CDF table.
    ///
    /// `icdf[s]` = cumulative probability above symbol `s` in units of
    /// `1 / 2^ft_bits`. Symbol `s` selects the interval `[icdf[s+1], icdf[s])`.
    pub fn encode_icdf(&mut self, s: u32, icdf: &[u8], ft_bits: u8) {
        let fl = if (s as usize + 1) < icdf.len() {
            icdf[s as usize + 1] as u32
        } else {
            0
        };
        let fh = icdf[s as usize] as u32;
        let ft = 1u32 << ft_bits;
        let scale = self.rng >> ft_bits;
        self.low = self.low.wrapping_add(scale.wrapping_mul(fl));
        self.rng = if fh == ft {
            self.rng.wrapping_sub(scale.wrapping_mul(fl))
        } else {
            scale.wrapping_mul(fh - fl)
        };
        self.normalize();
    }

    /// Encode a uniform integer `val` in `[0, n)`.
    pub fn encode_uint(&mut self, val: u32, n: u32) {
        debug_assert!(val < n);
        if n > 256 {
            let nbits = u32::BITS - n.leading_zeros();
            let low_bits = nbits - 8;
            self.encode_uint(val >> low_bits, (n + 255) >> 8);
            self.encode_bits_raw(val & ((1 << low_bits) - 1), low_bits);
        } else {
            self.encode_uint_small(val, n);
        }
    }

    /// Encode `bits` raw bits from `val` (MSB first).
    pub fn encode_bits_raw(&mut self, val: u32, bits: u32) {
        for i in (0..bits).rev() {
            self.encode_uint_small((val >> i) & 1, 2);
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn encode_uint_small(&mut self, val: u32, n: u32) {
        debug_assert!(val < n, "val={val} n={n}");
        let scale = self.rng / n;
        self.low = self.low.wrapping_add(scale * val);
        self.rng = if val + 1 < n {
            scale
        } else {
            self.rng - scale * val
        };
        self.normalize();
    }

    /// Shift out bytes from the top of `low` whenever `rng` < 2^24.
    fn normalize(&mut self) {
        while self.rng < 0x0100_0000 {
            // Emit the top byte of `low`.
            self.output.push((self.low >> 24) as u8);
            self.low <<= 8;
            self.rng <<= 8;
        }
    }

    /// Flush remaining bits and return the encoded bytes.
    ///
    /// Pads the output with enough bytes so that the decoder can prime itself
    /// (it reads `PRIME_BYTES` ahead). The flush emits `self.low` shifted so the
    /// remaining range is covered.
    pub fn finish(mut self) -> Vec<u8> {
        // Final normalization: push remaining range bits.
        // After flush, the decoder reconstructs `low` from the stream.
        // We need to emit enough bytes that the decoder never reads past the end.
        // Emit 4 bytes of the final `low` (decoder primes with 4 bytes).
        for shift in [24u32, 16, 8, 0] {
            self.output.push((self.low >> shift) as u8);
        }
        self.output
    }
}

// ── Private round-trip decoder (test only) ────────────────────────────────────

/// Minimal range decoder paired with [`RangeEncoder`], used only in `#[cfg(test)]`.
///
/// This decoder implements the exact inverse of the simplified encoder above.
/// It is NOT an RFC-6716 decoder and cannot decode standard Opus streams.
#[cfg(test)]
struct RangeDecoder {
    low: u32,
    rng: u32,
    data: Vec<u8>,
    pos: usize,
}

#[cfg(test)]
impl RangeDecoder {
    /// Create a new decoder primed from the first 4 bytes of `data`.
    fn new(data: Vec<u8>) -> Self {
        let mut dec = Self {
            low: 0,
            rng: 0x8000_0000u32,
            data,
            pos: 0,
        };
        // Prime: read 4 bytes into `low` MSB-first to match the encoder's flush.
        for _ in 0..4 {
            dec.low = (dec.low << 8) | dec.read_byte() as u32;
        }
        // After priming, `low` holds the top 32 bits of the encoder's code word.
        // The decoder tracks a 32-bit window into the infinite-precision code.
        dec
    }

    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0xFF // padding if we read past the end
        }
    }

    /// Re-expand when `rng` has shrunk below 2^24 — mirrors encoder normalize.
    fn normalize(&mut self) {
        while self.rng < 0x0100_0000 {
            self.rng <<= 8;
            self.low = (self.low << 8) | self.read_byte() as u32;
        }
    }

    fn decode_uint_small(&mut self, n: u32) -> u32 {
        let scale = self.rng / n;
        // Determine which bucket `low` falls in.
        let idx = (self.low / scale).min(n - 1);
        // Remove the contribution of `idx` from `low`.
        self.low -= scale * idx;
        self.rng = if idx + 1 < n {
            scale
        } else {
            self.rng - scale * idx
        };
        self.normalize();
        idx
    }

    fn decode_bits_raw(&mut self, bits: u32) -> u32 {
        let mut val = 0u32;
        for _ in 0..bits {
            val = (val << 1) | self.decode_uint_small(2);
        }
        val
    }

    fn decode_icdf(&mut self, icdf: &[u8], ft_bits: u8) -> u32 {
        let ft = 1u32 << ft_bits;
        let scale = self.rng >> ft_bits;
        // What is the scaled "position" within the range?
        let scaled_pos = self.low / scale;
        // Find symbol s: icdf[s+1] <= scaled_pos < icdf[s]
        let mut s = 0u32;
        while (s as usize + 1) < icdf.len() && icdf[s as usize + 1] as u32 > scaled_pos {
            s += 1;
        }
        // Decode using the same formula as the encoder.
        let fl = if (s as usize + 1) < icdf.len() {
            icdf[s as usize + 1] as u32
        } else {
            0
        };
        let fh = icdf[s as usize] as u32;
        self.low -= scale * fl;
        self.rng = if fh == ft {
            self.rng - scale * fl
        } else {
            scale * (fh - fl)
        };
        self.normalize();
        s
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{RangeDecoder, RangeEncoder};

    // 2-symbol ICDF: P(0)=3/4, P(1)=1/4 in 2-bit (ft=4) space.
    // icdf[0]=4 (P(symbol > 0-1) = full range), icdf[1]=1 (P(symbol > 1) = 1/4).
    const ICDF_2SYM: &[u8] = &[4, 1];

    #[test]
    fn test_range_encoder_produces_non_empty_output() {
        let mut enc = RangeEncoder::new();
        enc.encode_icdf(0, ICDF_2SYM, 2);
        enc.encode_icdf(1, ICDF_2SYM, 2);
        let bytes = enc.finish();
        assert!(!bytes.is_empty(), "encoder output must not be empty");
    }

    #[test]
    fn test_range_encoder_empty_produces_output() {
        // Even an encoder with no symbols must flush without panicking.
        let enc = RangeEncoder::new();
        let bytes = enc.finish();
        assert!(
            !bytes.is_empty(),
            "empty encoder must produce bytes on finish"
        );
    }

    #[test]
    fn test_range_bits_raw_roundtrip() {
        let mut enc = RangeEncoder::new();
        enc.encode_bits_raw(0b101, 3);
        enc.encode_bits_raw(0b11, 2);
        enc.encode_bits_raw(0b0, 1);
        let bytes = enc.finish();

        let mut dec = RangeDecoder::new(bytes);
        let v0 = dec.decode_bits_raw(3);
        let v1 = dec.decode_bits_raw(2);
        let v2 = dec.decode_bits_raw(1);
        assert_eq!(v0, 0b101, "bits_raw round-trip: 3-bit value");
        assert_eq!(v1, 0b11, "bits_raw round-trip: 2-bit value");
        assert_eq!(v2, 0b0, "bits_raw round-trip: 1-bit value");
    }

    #[test]
    fn test_range_icdf_roundtrip() {
        let mut enc = RangeEncoder::new();
        let seq = [0u32, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1];
        for &s in &seq {
            enc.encode_icdf(s, ICDF_2SYM, 2);
        }
        let bytes = enc.finish();

        let mut dec = RangeDecoder::new(bytes);
        for (i, &expected) in seq.iter().enumerate() {
            let got = dec.decode_icdf(ICDF_2SYM, 2);
            assert_eq!(got, expected, "icdf round-trip failed at index {i}");
        }
    }

    #[test]
    fn test_range_bits_raw_various_widths() {
        let mut enc = RangeEncoder::new();
        enc.encode_bits_raw(0b1111, 4); // 15
        enc.encode_bits_raw(0b000, 3); // 0
        enc.encode_bits_raw(0b10, 2); // 2
        let bytes = enc.finish();

        let mut dec = RangeDecoder::new(bytes);
        assert_eq!(dec.decode_bits_raw(4), 0b1111);
        assert_eq!(dec.decode_bits_raw(3), 0b000);
        assert_eq!(dec.decode_bits_raw(2), 0b10);
    }

    #[test]
    fn test_range_mixed_icdf_and_bits_roundtrip() {
        let mut enc = RangeEncoder::new();
        enc.encode_icdf(1, ICDF_2SYM, 2);
        enc.encode_bits_raw(0b101, 3);
        enc.encode_icdf(0, ICDF_2SYM, 2);
        enc.encode_bits_raw(0b0, 1);
        let bytes = enc.finish();

        let mut dec = RangeDecoder::new(bytes);
        assert_eq!(dec.decode_icdf(ICDF_2SYM, 2), 1);
        assert_eq!(dec.decode_bits_raw(3), 0b101);
        assert_eq!(dec.decode_icdf(ICDF_2SYM, 2), 0);
        assert_eq!(dec.decode_bits_raw(1), 0b0);
    }
}
