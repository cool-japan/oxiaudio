//! Internal round-trip: decode the SILK payload with a minimal EcDec replica and
//! compare against the encoder's plan to localize any bitstream desync.
#![cfg(feature = "silk_debug")]
#![allow(clippy::needless_range_loop, clippy::manual_div_ceil)]

use oxiaudio_encode::opus_silk_encode::debug_plan;

// ── Minimal range decoder (port of opus-decoder `entropy.rs`) ────────────────

const EC_SYM_BITS: i32 = 8;
const EC_CODE_BITS: i32 = 32;
const EC_CODE_TOP: u32 = 1u32 << (EC_CODE_BITS - 1);
const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;
const EC_CODE_EXTRA: i32 = ((EC_CODE_BITS - 2) % EC_SYM_BITS) + 1;
const EC_SYM_MAX: u32 = (1u32 << EC_SYM_BITS) - 1;
const EC_CODE_TOP_MASK: u32 = EC_CODE_TOP - 1;

struct EcDec<'a> {
    buf: &'a [u8],
    offs: usize,
    rng: u32,
    val: u32,
    rem: i32,
}

impl<'a> EcDec<'a> {
    fn new(buf: &'a [u8]) -> Self {
        let mut st = Self {
            buf,
            offs: 0,
            rng: 1u32 << EC_CODE_EXTRA,
            val: 0,
            rem: 0,
        };
        st.rem = st.read_byte() as i32;
        st.val = st.rng - 1 - ((st.rem as u32) >> (EC_SYM_BITS - EC_CODE_EXTRA));
        st.normalize();
        st
    }
    fn read_byte(&mut self) -> u8 {
        if self.offs < self.buf.len() {
            let b = self.buf[self.offs];
            self.offs += 1;
            b
        } else {
            0
        }
    }
    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.rng <<= EC_SYM_BITS;
            let mut sym = self.rem as u32;
            self.rem = self.read_byte() as i32;
            sym = (sym << EC_SYM_BITS | (self.rem as u32)) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            self.val = ((self.val << EC_SYM_BITS) + (EC_SYM_MAX & !sym)) & EC_CODE_TOP_MASK;
        }
    }
    fn dec_bit_logp(&mut self, logp: u32) -> bool {
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
            s = r.wrapping_mul(icdf[ret as usize] as u32);
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

// ── SILK entropy tables needed to parse the payload ──────────────────────────

const TYPE_OFFSET_VAD_ICDF: [u8; 4] = [232, 158, 10, 0];
const GAIN_ICDF_1: [u8; 8] = [254, 237, 192, 132, 70, 23, 4, 0];
const UNIFORM8_ICDF: [u8; 8] = [224, 192, 160, 128, 96, 64, 32, 0];
const DELTA_GAIN_ICDF: [u8; 41] = [
    250, 245, 234, 203, 71, 50, 42, 38, 35, 33, 31, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
    17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
];
const NLSF_CB1_ICDF_NB_MB: [u8; 64] = [
    212, 178, 148, 129, 108, 96, 85, 82, 79, 77, 61, 59, 57, 56, 51, 49, 48, 45, 42, 41, 40, 38,
    36, 34, 31, 30, 21, 12, 10, 3, 1, 0, 255, 245, 244, 236, 233, 225, 217, 203, 190, 176, 175,
    161, 149, 136, 125, 114, 102, 91, 81, 71, 60, 52, 43, 35, 28, 20, 19, 18, 12, 11, 5, 0,
];
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
const NLSF_CB2_ICDF_NB_MB: [u8; 72] = [
    255, 254, 253, 238, 14, 3, 2, 1, 0, 255, 254, 252, 218, 35, 3, 2, 1, 0, 255, 254, 250, 208, 59,
    4, 2, 1, 0, 255, 254, 246, 194, 71, 10, 2, 1, 0, 255, 252, 236, 183, 82, 8, 2, 1, 0, 255, 252,
    235, 180, 90, 17, 2, 1, 0, 255, 248, 224, 171, 97, 30, 4, 1, 0, 255, 254, 236, 173, 95, 37, 7,
    1, 0,
];
const NLSF_EXT_ICDF: [u8; 7] = [100, 40, 16, 7, 3, 1, 0];
const NLSF_INTERP_FACTOR_ICDF: [u8; 5] = [243, 221, 192, 181, 0];
const UNIFORM4_ICDF: [u8; 4] = [192, 128, 64, 0];
const RATE_LEVELS_ICDF: [u8; 9] = [241, 190, 178, 132, 87, 74, 41, 14, 0];
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
const SHELL0: [u8; 152] = [
    128, 0, 214, 42, 0, 235, 128, 21, 0, 244, 184, 72, 11, 0, 248, 214, 128, 42, 7, 0, 248, 225,
    170, 80, 25, 5, 0, 251, 236, 198, 126, 54, 18, 3, 0, 250, 238, 211, 159, 82, 35, 15, 5, 0, 250,
    231, 203, 168, 128, 88, 53, 25, 6, 0, 252, 238, 216, 185, 148, 108, 71, 40, 18, 4, 0, 253, 243,
    225, 199, 166, 128, 90, 57, 31, 13, 3, 0, 254, 246, 233, 212, 183, 147, 109, 73, 44, 23, 10, 2,
    0, 255, 250, 240, 223, 198, 166, 128, 90, 58, 33, 16, 6, 1, 0, 255, 251, 244, 231, 210, 181,
    146, 110, 75, 46, 25, 12, 5, 1, 0, 255, 253, 248, 238, 221, 196, 164, 128, 92, 60, 35, 18, 8,
    3, 1, 0, 255, 253, 249, 242, 229, 208, 180, 146, 110, 76, 48, 27, 14, 7, 3, 1, 0,
];
const SHELL1: [u8; 152] = [
    129, 0, 207, 50, 0, 236, 129, 20, 0, 245, 185, 72, 10, 0, 249, 213, 129, 42, 6, 0, 250, 226,
    169, 87, 27, 4, 0, 251, 233, 194, 130, 62, 20, 4, 0, 250, 236, 207, 160, 99, 47, 17, 3, 0, 255,
    240, 217, 182, 131, 81, 41, 11, 1, 0, 255, 254, 233, 201, 159, 107, 61, 20, 2, 1, 0, 255, 249,
    233, 206, 170, 128, 86, 50, 23, 7, 1, 0, 255, 250, 238, 217, 186, 148, 108, 70, 39, 18, 6, 1,
    0, 255, 252, 243, 226, 200, 166, 128, 90, 56, 30, 13, 4, 1, 0, 255, 252, 245, 231, 209, 180,
    146, 110, 76, 47, 25, 11, 4, 1, 0, 255, 253, 248, 237, 219, 194, 163, 128, 93, 62, 37, 19, 8,
    3, 1, 0, 255, 254, 250, 241, 226, 205, 177, 145, 111, 79, 51, 30, 15, 6, 2, 1, 0,
];
const SHELL2: [u8; 152] = [
    129, 0, 203, 54, 0, 234, 129, 23, 0, 245, 184, 73, 10, 0, 250, 215, 129, 41, 5, 0, 252, 232,
    173, 86, 24, 3, 0, 253, 240, 200, 129, 56, 15, 2, 0, 253, 244, 217, 164, 94, 38, 10, 1, 0, 253,
    245, 226, 189, 132, 71, 27, 7, 1, 0, 253, 246, 231, 203, 159, 105, 56, 23, 6, 1, 0, 255, 248,
    235, 213, 179, 133, 85, 47, 19, 5, 1, 0, 255, 254, 243, 221, 194, 159, 117, 70, 37, 12, 2, 1,
    0, 255, 254, 248, 234, 208, 171, 128, 85, 48, 22, 8, 2, 1, 0, 255, 254, 250, 240, 220, 189,
    149, 107, 67, 36, 16, 6, 2, 1, 0, 255, 254, 251, 243, 227, 201, 166, 128, 90, 55, 29, 13, 5, 2,
    1, 0, 255, 254, 252, 246, 234, 213, 183, 147, 109, 73, 43, 22, 10, 4, 2, 1, 0,
];
const SHELL3: [u8; 152] = [
    130, 0, 200, 58, 0, 231, 130, 26, 0, 244, 184, 76, 12, 0, 249, 214, 130, 43, 6, 0, 252, 232,
    173, 87, 24, 3, 0, 253, 241, 203, 131, 56, 14, 2, 0, 254, 246, 221, 167, 94, 35, 8, 1, 0, 254,
    249, 232, 193, 130, 65, 23, 5, 1, 0, 255, 251, 239, 211, 162, 99, 45, 15, 4, 1, 0, 255, 251,
    243, 223, 186, 131, 74, 33, 11, 3, 1, 0, 255, 252, 245, 230, 202, 158, 105, 57, 24, 8, 2, 1, 0,
    255, 253, 247, 235, 214, 179, 132, 84, 44, 19, 7, 2, 1, 0, 255, 254, 250, 240, 223, 196, 159,
    112, 69, 36, 15, 6, 2, 1, 0, 255, 254, 253, 245, 231, 209, 176, 136, 93, 55, 27, 11, 3, 2, 1,
    0, 255, 254, 253, 252, 239, 221, 194, 158, 117, 76, 42, 18, 4, 3, 2, 1, 0,
];
const SHELL_OFFS: [u8; 17] = [
    0, 0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119, 135,
];
const SIGN_ICDF: [u8; 42] = [
    254, 49, 67, 77, 82, 93, 99, 198, 11, 18, 24, 31, 36, 45, 255, 46, 66, 78, 87, 94, 104, 208,
    14, 21, 32, 42, 51, 66, 255, 94, 104, 109, 112, 115, 118, 248, 53, 69, 80, 88, 95, 102,
];

fn dec_split(dec: &mut EcDec, p: i32, table: &[u8]) -> (i16, i16) {
    if p <= 0 {
        return (0, 0);
    }
    let offset = SHELL_OFFS[p as usize] as usize;
    let c1 = dec.dec_icdf(&table[offset..], 8) as i16;
    (c1, p as i16 - c1)
}

fn shell_decode(dec: &mut EcDec, pulses4: i32) -> [i16; 16] {
    let (p30, p31) = dec_split(dec, pulses4, &SHELL3);
    let (p20, p21) = dec_split(dec, p30 as i32, &SHELL2);
    let (p10, p11) = dec_split(dec, p20 as i32, &SHELL1);
    let (p00, p01) = dec_split(dec, p10 as i32, &SHELL0);
    let (p02, p03) = dec_split(dec, p11 as i32, &SHELL0);
    let (p12, p13) = dec_split(dec, p21 as i32, &SHELL1);
    let (p04, p05) = dec_split(dec, p12 as i32, &SHELL0);
    let (p06, p07) = dec_split(dec, p13 as i32, &SHELL0);
    let (p22, p23) = dec_split(dec, p31 as i32, &SHELL2);
    let (p14, p15) = dec_split(dec, p22 as i32, &SHELL1);
    let (p08, p09) = dec_split(dec, p14 as i32, &SHELL0);
    let (p10b, p11b) = dec_split(dec, p15 as i32, &SHELL0);
    let (p16, p17) = dec_split(dec, p23 as i32, &SHELL1);
    let (p12b, p13b) = dec_split(dec, p16 as i32, &SHELL0);
    let (p14b, p15b) = dec_split(dec, p17 as i32, &SHELL0);
    [
        p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10b, p11b, p12b, p13b, p14b, p15b,
    ]
}

#[test]
fn silk_payload_roundtrips() {
    let pcm: Vec<f32> = (0..960)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin() * 0.5)
        .collect();
    let plan = debug_plan(&pcm);
    let packet = oxiaudio_encode::encode_silk_frame_conformant(&pcm, 1);
    let payload = &packet[1..];
    let mut dec = EcDec::new(payload);

    // Header.
    let vad = dec.dec_bit_logp(1);
    let has_lbrr = dec.dec_bit_logp(1);
    assert!(vad, "vad flag");
    assert!(!has_lbrr, "no lbrr");

    // decode_indices.
    let ix = dec.dec_icdf(&TYPE_OFFSET_VAD_ICDF, 8) + 2;
    let signal_type = ix >> 1;
    let quant_offset = ix & 1;
    assert_eq!(signal_type, 1, "signal type unvoiced");
    assert_eq!(quant_offset, 0);

    let mut gain_index = (dec.dec_icdf(&GAIN_ICDF_1, 8) << 3) as i32;
    gain_index += dec.dec_icdf(&UNIFORM8_ICDF, 8);
    assert_eq!(gain_index, plan.gain_index, "gain index");
    for _ in 1..4 {
        let _ = dec.dec_icdf(&DELTA_GAIN_ICDF, 8);
    }

    let cb1 = dec.dec_icdf(&NLSF_CB1_ICDF_NB_MB, 8) as usize;
    assert_eq!(cb1, plan.cb1_index, "cb1 index");
    let mut stage2 = [0i8; 10];
    for i in 0..10 {
        let row = nlsf_ec_row(cb1, i);
        let mut s = dec.dec_icdf(&NLSF_CB2_ICDF_NB_MB[row..], 8);
        if s == 0 {
            s -= dec.dec_icdf(&NLSF_EXT_ICDF, 8);
        } else if s == 8 {
            s += dec.dec_icdf(&NLSF_EXT_ICDF, 8);
        }
        stage2[i] = (s - 4) as i8;
    }
    assert_eq!(stage2, plan.nlsf_stage2, "nlsf stage2");
    let _interp = dec.dec_icdf(&NLSF_INTERP_FACTOR_ICDF, 8);
    let seed = dec.dec_icdf(&UNIFORM4_ICDF, 8) as i8;
    assert_eq!(seed, plan.seed, "seed");

    // decode_pulses.
    let rate_level = dec.dec_icdf(&RATE_LEVELS_ICDF, 8) as usize;
    assert_eq!(rate_level, plan.rate_level, "rate level");
    let mut sums = [0i32; 10];
    for s in sums.iter_mut() {
        *s = dec.dec_icdf(&PULSES_PER_BLOCK_ICDF[rate_level], 8);
    }
    assert_eq!(sums, plan.block_sum, "block sums");

    let mut mag = [0i16; 160];
    for b in 0..10 {
        if sums[b] > 0 {
            let block = shell_decode(&mut dec, sums[b]);
            mag[b * 16..(b + 1) * 16].copy_from_slice(&block);
        }
    }
    assert_eq!(mag, plan.magnitude, "magnitudes");

    // Signs.
    let base = 7 * (quant_offset + (signal_type << 1)) as usize;
    let mut pulses = [0i16; 160];
    for b in 0..10 {
        if sums[b] <= 0 {
            continue;
        }
        let idx = (sums[b] & 0x1F).min(6) as usize;
        let icdf = [SIGN_ICDF[base + idx], 0];
        for i in 0..16 {
            let m = mag[b * 16 + i];
            if m > 0 {
                let sign = dec.dec_icdf(&icdf, 8);
                pulses[b * 16 + i] = if sign == 0 { -m } else { m };
            }
        }
    }
    assert_eq!(pulses, plan.pulses, "signed pulses");
}

fn nlsf_ec_row(cb1_index: usize, i: usize) -> usize {
    const NLSF_STAGE2_ROW: usize = 9;
    let sel_base = cb1_index * 10 / 2;
    let entry = NLSF_CB2_SELECT_NB_MB[sel_base + i / 2];
    if i % 2 == 0 {
        (((entry >> 1) & 7) as usize) * NLSF_STAGE2_ROW
    } else {
        (((entry >> 5) & 7) as usize) * NLSF_STAGE2_ROW
    }
}

/// Regression pin for the 0.2.1 range-coder fix: `enc_bit_logp(v)` must be
/// decoded by a standard RFC 6716 `dec_bit_logp` as `v`.
///
/// Before the fix the encoder emitted the logical complement while keeping the
/// stream byte-synchronized, so every CELT header flag (silence, intra,
/// transient, TF, skip) and every SILK VAD/LBRR flag was written inverted and
/// the defect was invisible to "does it decode without error?" tests.
#[test]
fn enc_bit_logp_roundtrips_verbatim() {
    use oxiaudio_encode::opus_range::RangeEncoder;
    let seq = [true, false, true, true, false, false, true];
    let mut enc = RangeEncoder::new();
    for &b in &seq {
        enc.enc_bit_logp(b, 1);
    }
    for _ in 0..4 {
        enc.enc_icdf(0, &UNIFORM4_ICDF, 8);
    }
    let bytes = enc.finish();
    let mut dec = EcDec::new(&bytes);
    let got: Vec<bool> = (0..seq.len()).map(|_| dec.dec_bit_logp(1)).collect();
    assert_eq!(got, seq.to_vec(), "enc_bit_logp(v) must decode as v");
}

#[test]
fn mini_decoder_icdf_roundtrips() {
    use oxiaudio_encode::opus_range::RangeEncoder;
    let syms = [0usize, 3, 1, 2, 0, 1, 3, 2, 1, 0];
    let mut enc = RangeEncoder::new();
    for &s in &syms {
        enc.enc_icdf(s, &UNIFORM4_ICDF, 8);
    }
    let bytes = enc.finish();
    let mut dec = EcDec::new(&bytes);
    let got: Vec<usize> = (0..syms.len())
        .map(|_| dec.dec_icdf(&UNIFORM4_ICDF, 8) as usize)
        .collect();
    assert_eq!(got, syms.to_vec(), "mini dec_icdf must round-trip");
}

#[test]
fn mixed_bit_and_icdf_sync() {
    use oxiaudio_encode::opus_range::RangeEncoder;
    let mut enc = RangeEncoder::new();
    enc.enc_bit_logp(true, 1);
    enc.enc_icdf(2, &UNIFORM4_ICDF, 8);
    enc.enc_bit_logp(false, 1);
    enc.enc_icdf(1, &UNIFORM4_ICDF, 8);
    enc.enc_bit_logp(true, 1);
    enc.enc_icdf(3, &UNIFORM4_ICDF, 8);
    let bytes = enc.finish();
    let mut dec = EcDec::new(&bytes);
    let b0 = dec.dec_bit_logp(1);
    let s0 = dec.dec_icdf(&UNIFORM4_ICDF, 8);
    let b1 = dec.dec_bit_logp(1);
    let s1 = dec.dec_icdf(&UNIFORM4_ICDF, 8);
    let b2 = dec.dec_bit_logp(1);
    let s2 = dec.dec_icdf(&UNIFORM4_ICDF, 8);
    // Both the icdf symbols and the raw bit values must survive interleaving.
    assert_eq!((s0, s1, s2), (2, 1, 3), "icdf symbols must stay in sync");
    assert_eq!(
        (b0, b1, b2),
        (true, false, true),
        "enc_bit_logp values must decode verbatim when interleaved with icdf"
    );
}

// ── Reference decoder NLSF→LPC (ported from opus-decoder) ────────────────────

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
const NLSF_DELTA_MIN: [i16; 11] = [250, 3, 6, 3, 3, 3, 4, 3, 3, 3, 461];
const QSTEP: i32 = 11_796;
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

fn rr(v: i32, s: usize) -> i32 {
    if s == 1 {
        (v >> 1) + (v & 1)
    } else {
        ((v >> (s - 1)) + 1) >> 1
    }
}
fn rr64(v: i64, s: usize) -> i32 {
    if s == 1 {
        ((v >> 1) + (v & 1)) as i32
    } else {
        (((v >> (s - 1)) + 1) >> 1) as i32
    }
}

fn ref_nlsf_decode(cb1: usize, idx: &[i8; 10]) -> [i16; 10] {
    let mut pred_q8 = [0u8; 10];
    let base = cb1 * 10 / 2;
    for i in (0..10).step_by(2) {
        let entry = NLSF_CB2_SELECT_NB_MB[base + i / 2];
        pred_q8[i] = NLSF_PRED_NB_MB_Q8[i + ((entry & 1) as usize) * 9];
        pred_q8[i + 1] = NLSF_PRED_NB_MB_Q8[i + (((entry >> 4) & 1) as usize) * 9 + 1];
    }
    let mut res_q10 = [0i16; 10];
    let mut out = 0i32;
    for i in (0..10).rev() {
        let pred = (out * pred_q8[i] as i32) >> 8;
        let mut v = (idx[i] as i32) << 10;
        if v > 0 {
            v -= 102;
        } else if v < 0 {
            v += 102;
        }
        out = pred + (((v as i64) * QSTEP as i64) >> 16) as i32;
        res_q10[i] = out as i16;
    }
    let cb_base = cb1 * 10;
    let mut nlsf = [0i16; 10];
    for i in 0..10 {
        let w = ((res_q10[i] as i32) << 14) / NLSF_CB1_WGHT_NB_MB_Q9[cb_base + i] as i32;
        let c = (NLSF_CB1_NB_MB_Q8[cb_base + i] as i32) << 7;
        nlsf[i] = (w + c).clamp(0, 32767) as i16;
    }
    ref_stabilize(&mut nlsf);
    nlsf
}

fn ref_stabilize(nlsf: &mut [i16; 10]) {
    let dm = &NLSF_DELTA_MIN;
    for _ in 0..20 {
        let mut md = nlsf[0] as i32 - dm[0] as i32;
        let mut split = 0usize;
        for i in 1..10 {
            let d = nlsf[i] as i32 - (nlsf[i - 1] as i32 + dm[i] as i32);
            if d < md {
                md = d;
                split = i;
            }
        }
        let tail = (1 << 15) - (nlsf[9] as i32 + dm[10] as i32);
        if tail < md {
            md = tail;
            split = 10;
        }
        if md >= 0 {
            return;
        }
        if split == 0 {
            nlsf[0] = dm[0];
        } else if split == 10 {
            nlsf[9] = ((1 << 15) - dm[10] as i32) as i16;
        } else {
            let mut minc = 0i32;
            for &d in &dm[..split] {
                minc += d as i32;
            }
            minc += (dm[split] as i32) >> 1;
            let mut maxc = 1 << 15;
            for &d in &dm[split + 1..=10] {
                maxc -= d as i32;
            }
            maxc -= (dm[split] as i32) >> 1;
            let center = rr(nlsf[split - 1] as i32 + nlsf[split] as i32, 1).clamp(minc, maxc);
            nlsf[split - 1] = (center - ((dm[split] as i32) >> 1)) as i16;
            nlsf[split] = (nlsf[split - 1] as i32 + dm[split] as i32) as i16;
        }
    }
    nlsf.sort_unstable();
    nlsf[0] = nlsf[0].max(dm[0]);
    for i in 1..10 {
        nlsf[i] = nlsf[i].max(nlsf[i - 1].saturating_add(dm[i]));
    }
    nlsf[9] = nlsf[9].min(((1 << 15) - dm[10] as i32) as i16);
    for i in (0..9).rev() {
        nlsf[i] = nlsf[i].min(nlsf[i + 1] - dm[i + 1]);
    }
}

fn ref_nlsf2a(nlsf: &[i16; 10]) -> [i16; 10] {
    const QA: usize = 16;
    let ord = [0usize, 9, 6, 3, 4, 5, 8, 1, 2, 7];
    let mut c = [0i32; 10];
    for k in 0..10 {
        let fi = (nlsf[k] as i32 >> 8) as usize;
        let ff = nlsf[k] as i32 - ((fi as i32) << 8);
        let cv = LSF_COS_TAB[fi] as i32;
        let d = LSF_COS_TAB[fi + 1] as i32 - cv;
        c[ord[k]] = rr((cv << 8) + d * ff, 20 - QA);
    }
    let dd = 5;
    let mut p = [0i32; 6];
    let mut q = [0i32; 6];
    find_poly(&mut p, &c, dd, QA);
    find_poly(&mut q, &c[1..], dd, QA);
    let mut a = [0i32; 10];
    for k in 0..dd {
        let pt = p[k + 1] + p[k];
        let qt = q[k + 1] - q[k];
        a[k] = -qt - pt;
        a[10 - k - 1] = qt - pt;
    }
    let mut lpc = [0i16; 10];
    fit(&mut lpc, &mut a, QA + 1);
    for i in 0..16 {
        if inv_gain(&lpc) != 0 {
            break;
        }
        bwe(&mut a, 65536 - (2 << i));
        for k in 0..10 {
            lpc[k] = rr(a[k], QA + 1 - 12) as i16;
        }
    }
    lpc
}

fn find_poly(out: &mut [i32; 6], c: &[i32], dd: usize, qa: usize) {
    out[0] = 1 << qa;
    out[1] = -c[0];
    for k in 1..dd {
        let ft = c[2 * k];
        out[k + 1] = (out[k - 1] << 1) - rr64((ft as i64) * out[k] as i64, qa);
        for n in (2..=k).rev() {
            out[n] += out[n - 2] - rr64((ft as i64) * out[n - 1] as i64, qa);
        }
        out[1] -= ft;
    }
}
fn fit(lpc: &mut [i16; 10], a: &mut [i32; 10], qin: usize) {
    let qout = 12;
    let mut peak = 0usize;
    for _ in 0..10 {
        let mut mx = 0i32;
        for (i, &v) in a.iter().enumerate() {
            if v.saturating_abs() > mx {
                mx = v.saturating_abs();
                peak = i;
            }
        }
        let sh = rr(mx, qin - qout);
        if sh <= i16::MAX as i32 {
            for k in 0..10 {
                lpc[k] = rr(a[k], qin - qout) as i16;
            }
            return;
        }
        let capped = sh.min(163838);
        let num = ((capped - i16::MAX as i32) as i64) << 14;
        let den = ((capped as i64) * (peak as i64 + 1)) >> 2;
        bwe(a, 65470 - (num / den) as i32);
    }
    for k in 0..10 {
        let r = rr(a[k], qin - qout).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        lpc[k] = r;
        a[k] = (r as i32) << (qin - qout);
    }
}
fn bwe(a: &mut [i32; 10], chirp: i32) {
    let mut ch = chirp;
    let cm = ch - 65536;
    for c in a.iter_mut().take(9) {
        *c = ((*c as i64 * ch as i64) >> 16) as i32;
        ch += rr(((ch as i64) * (cm as i64)) as i32, 16);
    }
    a[9] = ((a[9] as i64 * ch as i64) >> 16) as i32;
}
fn inv_gain(lpc: &[i16; 10]) -> i32 {
    let mut a = [0i32; 10];
    let mut dc = 0i32;
    for k in 0..10 {
        dc += lpc[k] as i32;
        a[k] = (lpc[k] as i32) << 12;
    }
    if dc >= 4096 {
        return 0;
    }
    let mut ig = 1 << 30;
    for k in (1..10).rev() {
        if !(-16773022..=16773022).contains(&a[k]) {
            return 0;
        }
        let rc = -(a[k] << 7);
        let m1 = (1 << 30) - smmul(rc, rc);
        ig = smmul(ig, m1) << 2;
        if ig < 107374 {
            return 0;
        }
        let mq = 32 - (m1.unsigned_abs().leading_zeros() as usize);
        let m2 = inv_var(m1, mq + 30);
        for n in 0..((k + 1) >> 1) {
            let t1 = a[n];
            let t2 = a[k - n - 1];
            let l = t1.saturating_sub(rr64((t2 as i64) * (rc as i64), 31));
            let r = t2.saturating_sub(rr64((t1 as i64) * (rc as i64), 31));
            a[n] = rr64((l as i64) * m2 as i64, mq);
            a[k - n - 1] = rr64((r as i64) * m2 as i64, mq);
        }
    }
    if !(-16773022..=16773022).contains(&a[0]) {
        return 0;
    }
    let rc = -(a[0] << 7);
    let m1 = (1 << 30) - smmul(rc, rc);
    ig = smmul(ig, m1) << 2;
    if ig < 107374 {
        0
    } else {
        ig
    }
}
fn smmul(a: i32, b: i32) -> i32 {
    (((a as i64) * (b as i64)) >> 32) as i32
}
fn abs_clz(v: i32) -> i32 {
    if v == i32::MIN {
        i32::MAX
    } else {
        v.abs()
    }
}
fn lsat(v: i32, s: usize) -> i32 {
    if s >= 31 {
        if v > 0 {
            i32::MAX
        } else if v < 0 {
            i32::MIN
        } else {
            0
        }
    } else {
        v.clamp(i32::MIN >> s, i32::MAX >> s).wrapping_shl(s as u32)
    }
}
fn swb(a: i32, b: i32) -> i32 {
    let b16 = b as i16 as i32;
    ((a >> 16) * b16).wrapping_add(((a & 0xFFFF) * b16) >> 16)
}
fn inv_var(v: i32, q: usize) -> i32 {
    let h = abs_clz(v).leading_zeros() as usize - 1;
    let bn = v << h;
    let bi = (i32::MAX >> 2) / (bn >> 16);
    let mut r = bi << 16;
    let e = ((1i32 << 29) - swb(bn, bi)) << 3;
    r = r.wrapping_add(((e as i64 * bi as i64) >> 16) as i32);
    let ls = 61isize - h as isize - q as isize;
    if ls <= 0 {
        lsat(r, (-ls) as usize)
    } else if ls < 32 {
        r >> ls
    } else {
        0
    }
}

#[test]
fn nlsf_and_lpc_match_decoder() {
    for freq in [200.0f32, 1000.0, 3000.0] {
        let pcm: Vec<f32> = (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let plan = debug_plan(&pcm);
        let ref_nlsf = ref_nlsf_decode(plan.cb1_index, &plan.nlsf_stage2);
        assert_eq!(ref_nlsf, plan.nlsf_q15, "nlsf_q15 mismatch at {freq}");
        let ref_a = ref_nlsf2a(&ref_nlsf);
        assert_eq!(ref_a, plan.a_q12, "a_q12 mismatch at {freq}");
    }
}
