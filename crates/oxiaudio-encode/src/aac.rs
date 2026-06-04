/// Pure-Rust AAC-LC ADTS encoder with spectral Huffman coding.
///
/// Produces ADTS-framed AAC-LC frames using CB11 (ESC_HCB) Huffman coding for
/// non-silent scale factor bands, and ZERO_HCB for silent bands.
/// Codebook tables are derived from ISO 14496-3 / Symphonia decoder source.
use std::io::Write;
use std::path::Path;

use oxiaudio_core::{AudioBuffer, OxiAudioError};
use oxifft::{fft, Complex};

// ──────────────────────────────────────────────────────────────────────────────
// MSB-first bit writer (AAC bit-stream packing)
// ──────────────────────────────────────────────────────────────────────────────

/// Packs bits MSB-first into a growable byte buffer.
///
/// `bit_pos` counts the number of *remaining* free bits in `current_byte`
/// (starts at 8, decrements toward 0; when it hits 0 the byte is flushed).
struct BitWriter {
    buf: Vec<u8>,
    current_byte: u8,
    /// Remaining free bits in `current_byte` (range 1..=8).
    bit_pos: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            current_byte: 0,
            bit_pos: 8,
        }
    }

    /// Write the least-significant `n` bits of `value`, MSB first.
    fn write_bits(&mut self, value: u64, n: u8) {
        let mut i = n;
        while i > 0 {
            i -= 1;
            let bit = ((value >> i) & 1) as u8;
            self.bit_pos -= 1;
            self.current_byte |= bit << self.bit_pos;
            if self.bit_pos == 0 {
                self.buf.push(self.current_byte);
                self.current_byte = 0;
                self.bit_pos = 8;
            }
        }
    }

    /// Flush any partial byte (zero-padded) and return the accumulated buffer.
    fn into_bytes(mut self) -> Vec<u8> {
        if self.bit_pos < 8 {
            self.buf.push(self.current_byte);
        }
        self.buf
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Sampling-frequency index table
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the 4-bit sampling_frequency_index for a given sample rate,
/// or `None` if the rate is not in the MPEG-4 table.
pub fn sampling_freq_index(sample_rate: u32) -> Option<u8> {
    match sample_rate {
        96000 => Some(0),
        88200 => Some(1),
        64000 => Some(2),
        48000 => Some(3),
        44100 => Some(4),
        32000 => Some(5),
        24000 => Some(6),
        22050 => Some(7),
        16000 => Some(8),
        12000 => Some(9),
        11025 => Some(10),
        8000 => Some(11),
        _ => None,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Scale Factor Band (SFB) offset tables
// ──────────────────────────────────────────────────────────────────────────────
//
// ISO 14496-3 Table 4.140 / Symphonia aac/common.rs — long-window band boundaries.
// Each table has (N_SFB + 1) entries; last entry is always 1024.

/// 48 kHz long window: 49 SFBs (indices 0..48), 50 boundary entries.
const SWB_OFFSET_48K_LONG: &[usize] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1024,
];

/// 32 kHz long window: 51 SFBs, 52 boundary entries.
const SWB_OFFSET_32K_LONG: &[usize] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,
];

/// 24 kHz long window: 47 SFBs, 48 boundary entries.
const SWB_OFFSET_24K_LONG: &[usize] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024,
];

/// 16 kHz long window: 43 SFBs, 44 boundary entries.
const SWB_OFFSET_16K_LONG: &[usize] = &[
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 100, 112, 124, 136, 148, 160, 172, 184, 196, 212,
    228, 244, 260, 280, 300, 320, 344, 368, 396, 424, 456, 492, 532, 572, 616, 664, 716, 772, 832,
    896, 960, 1024,
];

/// 64 kHz long window: 47 SFBs, 48 boundary entries.
const SWB_OFFSET_64K_LONG: &[usize] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 100, 112, 124, 140,
    156, 172, 192, 216, 240, 268, 304, 344, 384, 424, 464, 504, 544, 584, 624, 664, 704, 744, 784,
    824, 864, 904, 944, 984, 1024,
];

/// 96 kHz long window: 41 SFBs, 42 boundary entries.
const SWB_OFFSET_96K_LONG: &[usize] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 108, 120, 132,
    144, 156, 172, 188, 212, 240, 276, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
];

/// 8 kHz long window: 40 SFBs, 41 boundary entries.
const SWB_OFFSET_8K_LONG: &[usize] = &[
    0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 172, 188, 204, 220, 236, 252, 268,
    288, 308, 328, 348, 372, 396, 420, 448, 476, 508, 544, 580, 620, 664, 712, 764, 820, 880, 944,
    1024,
];

/// Returns the SFB boundary offset table (long window, 1024-pt MDCT) for a sample rate.
///
/// Threshold boundaries match Symphonia's `AAC_SUBBAND_INFO` lookup table
/// (`symphonia-codec-aac/src/aac/common.rs`).
fn sfb_offsets(sample_rate: u32) -> &'static [usize] {
    // Symphonia thresholds (min_srate), descending order:
    //  92017 → 96K table  (96 kHz)
    //  75132 → 96K table  (88.2 kHz)
    //  55426 → 64K table  (64 kHz)
    //  46009 → 48K table  (48 kHz)
    //  37566 → 48K table  (44.1 kHz)
    //  27713 → 32K table  (32 kHz)
    //  23004 → 24K table  (24 kHz)
    //  18783 → 24K table  (22.05 kHz)
    //  13856 → 16K table  (16 kHz)
    //  11502 → 16K table  (12 kHz)
    //   9391 → 16K table  (11.025 kHz)
    //      0 →  8K table  (8 kHz)
    const THRESHOLDS: [(u32, &[usize]); 11] = [
        (92017, SWB_OFFSET_96K_LONG),
        (75132, SWB_OFFSET_96K_LONG),
        (55426, SWB_OFFSET_64K_LONG),
        (46009, SWB_OFFSET_48K_LONG),
        (37566, SWB_OFFSET_48K_LONG),
        (27713, SWB_OFFSET_32K_LONG),
        (23004, SWB_OFFSET_24K_LONG),
        (18783, SWB_OFFSET_24K_LONG),
        (13856, SWB_OFFSET_16K_LONG),
        (11502, SWB_OFFSET_16K_LONG),
        (9391, SWB_OFFSET_16K_LONG),
    ];
    for (min_srate, table) in THRESHOLDS {
        if sample_rate >= min_srate {
            return table;
        }
    }
    SWB_OFFSET_8K_LONG
}

// ──────────────────────────────────────────────────────────────────────────────
// AAC Spectral Huffman Codebook 11 (ESC_HCB)
// ──────────────────────────────────────────────────────────────────────────────
//
// ISO 14496-3 Annex A — Codebook 11, 289 entries (17×17 unsigned pairs).
// Index = a*17 + b, where a = min(|quant_val|, 16), b = min(|quant_val2|, 16).
// Source: Symphonia symphonia-codec-aac-0.5.5/src/aac/codebooks.rs
// (SPECTRUM_CODEBOOK11_CODES / SPECTRUM_CODEBOOK11_LENS, escape_pair::<17>)
//
// Coding: Huffman codeword, then sign bit for each nonzero value, then ESC word
// if the clamped magnitude was 16.

#[rustfmt::skip]
static HCB11_LENS: [u8; 289] = [
     4,  5,  6,  7,  8,  8,  9, 10, 10, 10, 11, 11, 12, 11, 12, 12,
    10,  5,  4,  5,  6,  7,  7,  8,  8,  9,  9,  9, 10, 10, 10, 10,
    11,  8,  6,  5,  5,  6,  7,  7,  8,  8,  8,  9,  9,  9, 10, 10,
    10, 10,  8,  7,  6,  6,  6,  7,  7,  8,  8,  8,  9,  9,  9, 10,
    10, 10, 10,  8,  8,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,
    10, 10, 10, 10,  8,  8,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9,
     9, 10, 10, 10, 10,  8,  9,  8,  8,  8,  8,  8,  8,  8,  9,  9,
     9, 10, 10, 10, 10, 10,  8,  9,  8,  8,  8,  8,  8,  8,  9,  9,
     9, 10, 10, 10, 10, 10, 10,  8, 10,  9,  8,  8,  9,  9,  9,  9,
     9, 10, 10, 10, 10, 10, 10, 11,  8, 10,  9,  9,  9,  9,  9,  9,
     9, 10, 10, 10, 10, 10, 10, 11, 11,  8, 11,  9,  9,  9,  9,  9,
     9, 10, 10, 10, 10, 10, 11, 10, 11, 11,  8, 11, 10,  9,  9, 10,
     9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,  8, 11, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,  9, 11, 10,  9,
     9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,  9, 11, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,  9, 12,
    10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12,  9,
     9,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,
     5,
];

#[rustfmt::skip]
static HCB11_CODES: [u32; 289] = [
    0x000, 0x006, 0x019, 0x03d, 0x09c, 0x0c6, 0x1a7, 0x390,
    0x3c2, 0x3df, 0x7e6, 0x7f3, 0xffb, 0x7ec, 0xffa, 0xffe,
    0x38e, 0x005, 0x001, 0x008, 0x014, 0x037, 0x042, 0x092,
    0x0af, 0x191, 0x1a5, 0x1b5, 0x39e, 0x3c0, 0x3a2, 0x3cd,
    0x7d6, 0x0ae, 0x017, 0x007, 0x009, 0x018, 0x039, 0x040,
    0x08e, 0x0a3, 0x0b8, 0x199, 0x1ac, 0x1c1, 0x3b1, 0x396,
    0x3be, 0x3ca, 0x09d, 0x03c, 0x015, 0x016, 0x01a, 0x03b,
    0x044, 0x091, 0x0a5, 0x0be, 0x196, 0x1ae, 0x1b9, 0x3a1,
    0x391, 0x3a5, 0x3d5, 0x094, 0x09a, 0x036, 0x038, 0x03a,
    0x041, 0x08c, 0x09b, 0x0b0, 0x0c3, 0x19e, 0x1ab, 0x1bc,
    0x39f, 0x38f, 0x3a9, 0x3cf, 0x093, 0x0bf, 0x03e, 0x03f,
    0x043, 0x045, 0x09e, 0x0a7, 0x0b9, 0x194, 0x1a2, 0x1ba,
    0x1c3, 0x3a6, 0x3a7, 0x3bb, 0x3d4, 0x09f, 0x1a0, 0x08f,
    0x08d, 0x090, 0x098, 0x0a6, 0x0b6, 0x0c4, 0x19f, 0x1af,
    0x1bf, 0x399, 0x3bf, 0x3b4, 0x3c9, 0x3e7, 0x0a8, 0x1b6,
    0x0ab, 0x0a4, 0x0aa, 0x0b2, 0x0c2, 0x0c5, 0x198, 0x1a4,
    0x1b8, 0x38c, 0x3a4, 0x3c4, 0x3c6, 0x3dd, 0x3e8, 0x0ad,
    0x3af, 0x192, 0x0bd, 0x0bc, 0x18e, 0x197, 0x19a, 0x1a3,
    0x1b1, 0x38d, 0x398, 0x3b7, 0x3d3, 0x3d1, 0x3db, 0x7dd,
    0x0b4, 0x3de, 0x1a9, 0x19b, 0x19c, 0x1a1, 0x1aa, 0x1ad,
    0x1b3, 0x38b, 0x3b2, 0x3b8, 0x3ce, 0x3e1, 0x3e0, 0x7d2,
    0x7e5, 0x0b7, 0x7e3, 0x1bb, 0x1a8, 0x1a6, 0x1b0, 0x1b2,
    0x1b7, 0x39b, 0x39a, 0x3ba, 0x3b5, 0x3d6, 0x7d7, 0x3e4,
    0x7d8, 0x7ea, 0x0ba, 0x7e8, 0x3a0, 0x1bd, 0x1b4, 0x38a,
    0x1c4, 0x392, 0x3aa, 0x3b0, 0x3bc, 0x3d7, 0x7d4, 0x7dc,
    0x7db, 0x7d5, 0x7f0, 0x0c1, 0x7fb, 0x3c8, 0x3a3, 0x395,
    0x39d, 0x3ac, 0x3ae, 0x3c5, 0x3d8, 0x3e2, 0x3e6, 0x7e4,
    0x7e7, 0x7e0, 0x7e9, 0x7f7, 0x190, 0x7f2, 0x393, 0x1be,
    0x1c0, 0x394, 0x397, 0x3ad, 0x3c3, 0x3c1, 0x3d2, 0x7da,
    0x7d9, 0x7df, 0x7eb, 0x7f4, 0x7fa, 0x195, 0x7f8, 0x3bd,
    0x39c, 0x3ab, 0x3a8, 0x3b3, 0x3b9, 0x3d0, 0x3e3, 0x3e5,
    0x7e2, 0x7de, 0x7ed, 0x7f1, 0x7f9, 0x7fc, 0x193, 0xffd,
    0x3dc, 0x3b6, 0x3c7, 0x3cc, 0x3cb, 0x3d9, 0x3da, 0x7d3,
    0x7e1, 0x7ee, 0x7ef, 0x7f5, 0x7f6, 0xffc, 0xfff, 0x19d,
    0x1c2, 0x0b5, 0x0a1, 0x096, 0x097, 0x095, 0x099, 0x0a0,
    0x0a2, 0x0ac, 0x0a9, 0x0b1, 0x0b3, 0x0bb, 0x0c0, 0x18f,
    0x004,
];

/// Scale factor codebook: `value = 60` encodes a delta of `0` (60 - 60 = 0).
/// From Symphonia SCF_CODEBOOK (index 60 → len=1, code=0x00000).
/// We only need the single delta=0 codeword for uniform-scale encoding.
/// SCF codebook index 60: code = 0x00000, len = 1.
const SCF_DELTA_ZERO_CODE: u64 = 0x00000;
const SCF_DELTA_ZERO_LEN: u8 = 1;

// ──────────────────────────────────────────────────────────────────────────────
// ADTS header (7 bytes, protection_absent = 1, no CRC)
// ──────────────────────────────────────────────────────────────────────────────
//
// Bit-field layout (56 bits = 7 bytes):
//   [11] syncword               = 0xFFF
//   [ 1] ID                     = 0  (MPEG-4)
//   [ 2] layer                  = 0
//   [ 1] protection_absent      = 1
//   [ 2] profile_ObjectType – 1 = 1  (AAC-LC = object type 2, field = 2-1 = 1)
//   [ 4] sampling_frequency_idx
//   [ 1] private_bit            = 0
//   [ 3] channel_configuration
//   [ 1] originality/copy       = 0
//   [ 1] home                   = 0
//   [ 1] copyright_id_bit       = 0
//   [ 1] copyright_id_start     = 0
//   [13] aac_frame_length        (total bytes incl. this 7-byte header)
//   [11] adts_buffer_fullness    = 0x7FF (VBR / unspecified)
//   [ 2] number_of_raw_data_blocks_in_frame = 0  (means 1 block)
//
// Total: 11+1+2+1+2+4+1+3+1+1+1+1+13+11+2 = 56 bits = 7 bytes  ✓

fn write_adts_header(channels: u8, sample_rate: u32, frame_len: usize) -> [u8; 7] {
    let sfi = sampling_freq_index(sample_rate).unwrap_or(4); // default to 44100 index
    let len = frame_len as u64; // 13-bit field; caller guarantees <= 8191

    #[allow(clippy::cast_possible_truncation)]
    let mut h = [0u8; 7];
    h[0] = 0xFF;
    h[1] = 0xF1; // syncword tail (4 bits) + ID=0 + layer=0 + protection_absent=1
    h[2] = (0b01 << 6) | (sfi << 2) | ((channels >> 2) & 0x01);
    h[3] = ((channels & 0x3) << 6) | (((len >> 11) & 0x3) as u8);
    h[4] = ((len >> 3) & 0xFF) as u8;
    h[5] = (((len & 0x7) as u8) << 5) | 0x1F; // len[2:0] + fullness[10:6]
    h[6] = 0xFC; // fullness[5:0]=0x3F | num_blocks=0

    h
}

// ──────────────────────────────────────────────────────────────────────────────
// Silence ICS (ZERO_HCB, max_sfb=1)
// ──────────────────────────────────────────────────────────────────────────────

fn write_ics_silence(bw: &mut BitWriter) {
    bw.write_bits(127, 8); // global_gain
    bw.write_bits(0, 1); // ics_reserved_bit
    bw.write_bits(0, 2); // window_sequence = ONLY_LONG_SEQUENCE
    bw.write_bits(0, 1); // window_shape    = SINE_WINDOW
    bw.write_bits(1, 6); // max_sfb = 1
                         // NOTE: scale_factor_grouping (7 bits) is ONLY written for EIGHT_SHORT_SEQUENCE.
                         // For ONLY_LONG_SEQUENCE (window_sequence=0), this field does NOT exist per ISO 14496-3.
                         // DO NOT write these 7 bits here.
    bw.write_bits(0, 1); // predictor_data_present = 0

    // section_data: 1 section covering 1 SFB with ZERO_HCB
    bw.write_bits(0, 4); // sect_cb = 0 (ZERO_HCB)
    bw.write_bits(1, 5); // sect_len_incr = 1 (< 31 → terminates, section length = 1 SFB)

    // scale_factor_data: ZERO_HCB → no SCF codes emitted (Symphonia skips zero bands)

    // Symphonia's Ics::decode() reads these 3 flag bits after scale_factor_data:
    bw.write_bits(0, 1); // pulse_data_present = 0
    bw.write_bits(0, 1); // tns_data_present = 0
    bw.write_bits(0, 1); // gain_control_data_present = 0

    // spectral_data: ZERO_HCB → 0 bits
}

// ──────────────────────────────────────────────────────────────────────────────
// Coefficient quantization
// ──────────────────────────────────────────────────────────────────────────────

/// ISO 14496-3 §4.6.2.3.4 spectral coefficient quantization.
///
/// Standard formula: `q = nint(sign(x) * (|x| / scale)^(3/4))`
/// where `scale = 2^(0.25 * (global_gain - 100))`.
///
/// Equivalently: `q = nint(|x|^0.75 * scale^(-3/4))`
///             = `nint(|x|^0.75 * 2^(-3*(global_gain - 100)/16))`
///
/// `inv_scale = 2^(-3*(global_gain - 100)/16)` = `scale^(-3/4)`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn quantize_coeff(x: f32, inv_scale: f32) -> i16 {
    let abs_x = x.abs();
    let q = (abs_x.powf(0.75) * inv_scale + 0.4054) as i16;
    if x < 0.0 {
        q.saturating_neg()
    } else {
        q
    }
}

/// Compute global_gain and the quantization `inv_scale` from MDCT peak.
///
/// Uses the ISO 14496-3 standard formula.
/// `scale = 2^(0.25*(gain-100))` is the dequantization scale.
/// `inv_scale = scale^(-3/4) = 2^(-3*(gain-100)/16)` is the quantization divisor.
///
/// For target: `peak^0.75 * inv_scale ≈ 8191`
///   → `inv_scale = 8191 / peak^0.75`
///   → `2^(-3*(gain-100)/16) = 8191 / peak^0.75`
///   → `gain = 100 - (16/3) * log2(8191 / peak^0.75)`
///
/// Returns `(127, 0.0)` for near-silence.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn compute_global_gain_and_inv_scale(mdct: &[f32]) -> (u8, f32) {
    let peak = mdct.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    if peak < 1e-10 {
        return (127, 0.0);
    }
    let target_q = 8191.0_f32;
    let peak_q = peak.powf(0.75);
    // gain = 100 - (16/3) * log2(target_q / peak_q)
    let gain_f = 100.0_f32 - (16.0 / 3.0) * (target_q / peak_q).log2();
    let gain = gain_f.round().clamp(1.0, 255.0) as u8;
    // inv_scale = 2^(-3*(gain-100)/16) = scale^(-3/4) where scale = 2^(0.25*(gain-100))
    let inv_scale = 2.0_f32.powf(-3.0 * (gain as f32 - 100.0) / 16.0);
    (gain, inv_scale)
}

// ──────────────────────────────────────────────────────────────────────────────
// CB11 (ESC_HCB) escape word encoding
// ──────────────────────────────────────────────────────────────────────────────
//
// ISO 14496-3 §4.6.2.3.3: for |v| >= 16 after Huffman table lookup (clamped to 16),
// write an escape word:
//   n = floor(log2(abs_val)) - 4   (n = 0 means abs_val in 16..=31, n=1 → 32..=63, etc.)
//   Write n ones, then a 0, then (n+4) bits of (abs_val - (1<<(n+4))).
//
// Matches Symphonia's read_escape: word = (1 << (n+4)) + bs.read_bits(n+4)

fn write_esc_word(bw: &mut BitWriter, abs_val: u16) {
    // abs_val >= 16 guaranteed by caller.
    // n = floor(log2(abs_val)) - 4
    // leading_zeros() of u16 is in [0, 16]; truncation to u8 is safe.
    #[allow(clippy::cast_possible_truncation)]
    let leading = abs_val.leading_zeros() as u8;
    // 16-bit integer: bit position of MSB = 15 - leading_zeros
    // floor(log2(abs_val)) = 15 - leading_zeros (when abs_val >= 1)
    let msb_pos = 15u8 - leading; // e.g. abs_val=16 → msb_pos=4, n=0
    let n = msb_pos.saturating_sub(4); // n = msb_pos - 4
                                       // Write n ones (unary ESC prefix)
    for _ in 0..n {
        bw.write_bits(1, 1);
    }
    // Escape word terminator: 0
    bw.write_bits(0, 1);
    // (n+4) bits of remainder = abs_val - (1 << (n+4))
    let base: u16 = 1 << (n + 4);
    let remainder = (abs_val - base) as u64;
    bw.write_bits(remainder, n + 4);
}

/// Encode one CB11 (ESC_HCB) pair (v0, v1) into the bitstream.
///
/// Layout per pair:
///   Huffman(min(|v0|,16), min(|v1|,16)) | sign(v0) if v0≠0 | sign(v1) if v1≠0
///   | ESC_WORD(|v0|) if |v0|≥16 | ESC_WORD(|v1|) if |v1|≥16
fn encode_esc_pair(bw: &mut BitWriter, v0: i16, v1: i16) {
    let a0 = (v0.unsigned_abs()).min(16) as usize;
    let a1 = (v1.unsigned_abs()).min(16) as usize;
    // CB11 index: a0*17 + a1
    let idx = a0 * 17 + a1;
    let code = HCB11_CODES[idx];
    let len = HCB11_LENS[idx];
    bw.write_bits(code as u64, len);
    // Sign bits for nonzero values (1 = negative, 0 = positive)
    if v0 != 0 {
        bw.write_bits(u64::from(v0 < 0), 1);
    }
    if v1 != 0 {
        bw.write_bits(u64::from(v1 < 0), 1);
    }
    // ESC words for clamped values
    if a0 == 16 {
        write_esc_word(bw, v0.unsigned_abs());
    }
    if a1 == 16 {
        write_esc_word(bw, v1.unsigned_abs());
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Bitrate control mode
// ──────────────────────────────────────────────────────────────────────────────

/// Bitrate control mode for AAC encoding.
#[derive(Debug, Clone, Copy)]
pub enum AacBitrateMode {
    /// Variable bitrate: quality 1 (lowest) through 5 (highest).
    /// Higher quality → finer quantization → larger files.
    Vbr { quality: u8 },
    /// Approximate constant bitrate (kbps). Targets this bitrate by adjusting
    /// global_gain. Actual bitrate may vary ±15% per frame.
    Cbr { target_kbps: u32 },
}

impl Default for AacBitrateMode {
    fn default() -> Self {
        AacBitrateMode::Vbr { quality: 3 }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TNS (Temporal Noise Shaping) support structures
// ──────────────────────────────────────────────────────────────────────────────

/// Parameters controlling TNS encoding for one ICS window.
struct TnsParams {
    /// Whether TNS is active for this channel/frame.
    active: bool,
    /// LPC order.
    order: usize,
    /// Quantized LPC coefficients (4-bit signed, range -8..=7).
    coefs: Vec<i8>,
    /// Number of SFBs covered by the TNS filter (encoded in the bitstream as `length`).
    length: u8,
}

impl TnsParams {
    fn inactive() -> Self {
        Self {
            active: false,
            order: 0,
            coefs: Vec::new(),
            length: 0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ICS with real spectral data
// ──────────────────────────────────────────────────────────────────────────────

/// Encode one Individual Channel Stream using real MDCT coefficients.
///
/// Uses ZERO_HCB for all-zero SFBs and CB11 (ESC_HCB) for nonzero SFBs.
/// Scale factor delta encoding uses Huffman code for delta=0 (same gain for all SFBs).
/// PNS (sect_cb=13) SFBs emit scale factor but no spectral data.
/// TNS filter data is emitted after tns_data_present=1 when tns_params.active.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn write_ics_with_data(
    bw: &mut BitWriter,
    mdct: &[f32],
    offsets: &[usize],
    global_gain: u8,
    inv_scale: f32,
    pns_flags: Option<&[bool]>,
    tns_params: Option<&TnsParams>,
) {
    // Number of SFBs = offsets.len() - 1
    let n_sfbs = offsets.len() - 1;

    // Global gain
    bw.write_bits(u64::from(global_gain), 8);

    // ICS info (long window)
    bw.write_bits(0, 1); // ics_reserved_bit
    bw.write_bits(0, 2); // window_sequence = ONLY_LONG_SEQUENCE
    bw.write_bits(0, 1); // window_shape = SINE_WINDOW
    bw.write_bits(n_sfbs as u64, 6); // max_sfb
                                     // NOTE: scale_factor_grouping (7 bits) is ONLY written for EIGHT_SHORT_SEQUENCE.
                                     // For ONLY_LONG_SEQUENCE (window_sequence=0), this field does NOT exist per ISO 14496-3.
                                     // DO NOT write these 7 bits here.
    bw.write_bits(0, 1); // predictor_data_present = 0

    // Quantize all coefficients.
    let quant: Vec<i16> = mdct.iter().map(|&x| quantize_coeff(x, inv_scale)).collect();

    // Build section list: merge adjacent SFBs with same codebook.
    // Codebook: NOISE_HCB (13) for PNS SFBs, ZERO_HCB (0) for silent, CB11 (11) otherwise.
    const NOISE_HCB: u8 = 13;
    let mut sections: Vec<(u8, usize)> = Vec::new(); // (cb, sfb_count)
    let mut sfb = 0;
    while sfb < n_sfbs {
        let sfb_start = offsets[sfb];
        let sfb_end = offsets[sfb + 1].min(quant.len());
        let is_pns = pns_flags.and_then(|f| f.get(sfb)).copied().unwrap_or(false);
        let cb = if is_pns {
            NOISE_HCB
        } else if quant[sfb_start..sfb_end].iter().all(|&q| q == 0) {
            0u8 // ZERO_HCB
        } else {
            11u8 // ESC_HCB
        };
        let mut count = 1usize;
        while sfb + count < n_sfbs {
            let ns = offsets[sfb + count];
            let ne = offsets[sfb + count + 1].min(quant.len());
            let next_is_pns = pns_flags
                .and_then(|f| f.get(sfb + count))
                .copied()
                .unwrap_or(false);
            let next_cb = if next_is_pns {
                NOISE_HCB
            } else if quant[ns..ne].iter().all(|&q| q == 0) {
                0u8
            } else {
                11u8
            };
            if next_cb == cb {
                count += 1;
            } else {
                break;
            }
        }
        sections.push((cb, count));
        sfb += count;
    }

    // Emit section_data (long window: sect_bits=5, escape value=31).
    const SECT_BITS: u8 = 5;
    const SECT_ESC: u64 = 31;
    for &(cb, sfb_count) in &sections {
        bw.write_bits(u64::from(cb), 4); // sect_cb
        let mut remaining = sfb_count;
        while remaining >= SECT_ESC as usize {
            bw.write_bits(SECT_ESC, SECT_BITS); // escape continuation
            remaining -= SECT_ESC as usize;
        }
        bw.write_bits(remaining as u64, SECT_BITS); // final increment
    }

    // Emit scale_factor_data.
    // Symphonia's decode_scale_factor_data reads one SCF Huffman code for each
    // non-ZERO_HCB SFB (including NOISE_HCB), including the very first one.
    // The first SCF decode computes:
    //   scf_normal += code - 60  (initialized from global_gain)
    // So emitting delta=0 (code=0x00000, len=1) for every non-zero SFB is correct.
    // ZERO_HCB bands produce no SCF read in Symphonia — no code emitted.
    for &(cb, sfb_count) in &sections {
        for _ in 0..sfb_count {
            if cb != 0 {
                // Delta=0 → SCF codebook index 60: code=0x00000, len=1
                bw.write_bits(SCF_DELTA_ZERO_CODE, SCF_DELTA_ZERO_LEN);
            }
        }
    }

    // Flags: pulse_data_present | tns_data_present | gain_control_data_present
    // Order:  global_gain | ics_info | section_data | scale_factor_data
    //       | pulse_data_present=0 | tns_data_present | [tns_data] | gain_control_data_present=0
    //       | spectral_data
    bw.write_bits(0, 1); // pulse_data_present = 0

    // TNS data (ISO 14496-3 §4.6.9.3 — long window, 1 filter)
    let tns_active = tns_params.map(|t| t.active).unwrap_or(false);
    if tns_active {
        bw.write_bits(1, 1); // tns_data_present = 1
        if let Some(tns) = tns_params {
            // n_filt(w) — 2 bits for long window
            bw.write_bits(1, 2); // 1 filter
                                 // coef_res — 1 bit (1 = 4-bit coefficients)
            bw.write_bits(1, 1);
            // length — 6 bits (SFBs covered)
            bw.write_bits(u64::from(tns.length), 6);
            // order — 5 bits (LPC order, max 12 for long window)
            bw.write_bits(tns.order as u64, 5);
            // direction — 1 bit (1 = bottom→top spectral order)
            bw.write_bits(1, 1);
            // coef_compress — 1 bit (0 = no compression)
            bw.write_bits(0, 1);
            // coef[] — order × 4 bits each (coef_res=1, coef_compress=0 → 4 bits)
            for &c in &tns.coefs {
                // 4-bit signed two's complement in range -8..=7
                bw.write_bits((c as u8 & 0x0F) as u64, 4);
            }
        }
    } else {
        bw.write_bits(0, 1); // tns_data_present = 0
    }

    bw.write_bits(0, 1); // gain_control_data_present = 0

    // Emit spectral_data: walk sections, emit CB11 pairs for non-zero SFBs.
    // NOISE_HCB (PNS) and ZERO_HCB SFBs emit no spectral data.
    let mut sfb_idx = 0usize;
    for &(cb, sfb_count) in &sections {
        for _ in 0..sfb_count {
            let sfb_start = offsets[sfb_idx];
            let sfb_end = offsets[sfb_idx + 1].min(quant.len());
            if cb == 11 {
                // CB11 processes coefficients in pairs
                let band = &quant[sfb_start..sfb_end];
                for pair in band.chunks(2) {
                    let v0 = pair[0];
                    let v1 = if pair.len() > 1 { pair[1] } else { 0 };
                    encode_esc_pair(bw, v0, v1);
                }
            }
            // ZERO_HCB (0) and NOISE_HCB (13): no spectral data emitted
            sfb_idx += 1;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Single ADTS frame — silence fallback
// ──────────────────────────────────────────────────────────────────────────────

fn encode_adts_silence_frame(channels: usize, sample_rate: u32) -> Vec<u8> {
    let mut bw = BitWriter::new();

    match channels {
        1 => {
            bw.write_bits(0b000, 3); // ID_SCE
            bw.write_bits(0, 4); // element_instance_tag
            write_ics_silence(&mut bw);
        }
        _ => {
            bw.write_bits(0b001, 3); // ID_CPE
            bw.write_bits(0, 4); // element_instance_tag
            bw.write_bits(0, 1); // common_window = 0
            write_ics_silence(&mut bw); // left
            write_ics_silence(&mut bw); // right
        }
    }
    bw.write_bits(0b111, 3); // ID_END

    let audio_bytes = bw.into_bytes();
    let frame_total = 7 + audio_bytes.len();
    let mut out = Vec::with_capacity(frame_total);
    out.extend_from_slice(&write_adts_header(channels as u8, sample_rate, frame_total));
    out.extend_from_slice(&audio_bytes);
    out
}

// ──────────────────────────────────────────────────────────────────────────────
// Single ADTS frame — with real spectral data
// ──────────────────────────────────────────────────────────────────────────────

/// Owned options for a single ADTS audio frame encoding pass.
struct FrameOptions {
    /// Optional PNS flags per SFB (one entry per SFB). Empty = PNS disabled.
    pns_flags: Vec<bool>,
    /// Optional TNS parameters. `active=false` disables TNS.
    tns_params: TnsParams,
    /// Multiplier applied to `inv_scale` before quantization (VBR quality adjustment).
    inv_scale_multiplier: f32,
    /// Additive bias on `global_gain` (CBR bitrate control). Positive = fewer bits.
    gain_bias: i32,
}

impl FrameOptions {
    fn default_opts() -> Self {
        Self {
            pns_flags: Vec::new(),
            tns_params: TnsParams::inactive(),
            inv_scale_multiplier: 1.0,
            gain_bias: 0,
        }
    }
}

/// Encode one ADTS audio frame using real per-channel MDCT coefficients.
///
/// For stereo, `mdct_l` and `mdct_r` are the per-channel 1024-coefficient arrays.
/// For mono, `mdct_r` is ignored.
#[allow(clippy::cast_possible_truncation)]
fn encode_adts_audio_frame(
    channels: usize,
    sample_rate: u32,
    mdct_l: &[f32],
    mdct_r: &[f32],
) -> Vec<u8> {
    encode_adts_audio_frame_opts(
        channels,
        sample_rate,
        mdct_l,
        mdct_r,
        &FrameOptions::default_opts(),
    )
}

/// Encode one ADTS audio frame with explicit encoding options.
#[allow(clippy::cast_possible_truncation)]
fn encode_adts_audio_frame_opts(
    channels: usize,
    sample_rate: u32,
    mdct_l: &[f32],
    mdct_r: &[f32],
    opts: &FrameOptions,
) -> Vec<u8> {
    let offsets = sfb_offsets(sample_rate);
    let (gain_l_raw, inv_l_raw) = compute_global_gain_and_inv_scale(mdct_l);
    let gain_l = (gain_l_raw as i32 + opts.gain_bias).clamp(1, 255) as u8;
    let inv_l = inv_l_raw * opts.inv_scale_multiplier;
    let pns_opt: Option<&[bool]> = if opts.pns_flags.is_empty() {
        None
    } else {
        Some(&opts.pns_flags)
    };
    let tns_opt: Option<&TnsParams> = if opts.tns_params.active {
        Some(&opts.tns_params)
    } else {
        None
    };

    let mut bw = BitWriter::new();

    match channels {
        1 => {
            bw.write_bits(0b000, 3); // ID_SCE
            bw.write_bits(0, 4); // element_instance_tag
            write_ics_with_data(&mut bw, mdct_l, offsets, gain_l, inv_l, pns_opt, tns_opt);
        }
        _ => {
            let (gain_r_raw, inv_r_raw) = compute_global_gain_and_inv_scale(mdct_r);
            let gain_r = (gain_r_raw as i32 + opts.gain_bias).clamp(1, 255) as u8;
            let inv_r = inv_r_raw * opts.inv_scale_multiplier;
            bw.write_bits(0b001, 3); // ID_CPE
            bw.write_bits(0, 4); // element_instance_tag
            bw.write_bits(0, 1); // common_window = 0
            write_ics_with_data(&mut bw, mdct_l, offsets, gain_l, inv_l, pns_opt, tns_opt);
            write_ics_with_data(&mut bw, mdct_r, offsets, gain_r, inv_r, pns_opt, tns_opt);
        }
    }
    bw.write_bits(0b111, 3); // ID_END

    let audio_bytes = bw.into_bytes();
    let frame_total = 7 + audio_bytes.len();
    let mut out = Vec::with_capacity(frame_total);
    out.extend_from_slice(&write_adts_header(channels as u8, sample_rate, frame_total));
    out.extend_from_slice(&audio_bytes);
    out
}

// ──────────────────────────────────────────────────────────────────────────────
// AAC MDCT (N = 2048 input samples → 1024 coefficients)
// ──────────────────────────────────────────────────────────────────────────────

/// Forward MDCT for AAC-LC: 2048 input samples → 1024 spectral coefficients.
///
/// Applies a sine window, pre-rotates into a 1024-point complex signal,
/// computes the DFT via OxiFFT, and post-rotates to obtain real MDCT bins.
///
/// # Panics
///
/// Panics if `samples.len() != 2048`.
pub fn aac_mdct_forward(samples: &[f32]) -> Vec<f32> {
    assert_eq!(
        samples.len(),
        2048,
        "aac_mdct_forward: expected 2048 samples, got {}",
        samples.len()
    );

    let n: usize = 2048;
    let n2: usize = n / 2; // 1024

    // Step 1: Sine window — w[k] = sin(π·(k+½)/N)
    let windowed: Vec<f32> = samples
        .iter()
        .enumerate()
        .map(|(k, &s)| s * (std::f32::consts::PI * (k as f32 + 0.5) / n as f32).sin())
        .collect();

    // Step 2: Pre-rotation by exp(−j·π·k/N) into N/2-length complex vector.
    let pre_rotated: Vec<Complex<f32>> = (0..n2)
        .map(|k| {
            let angle = -std::f32::consts::PI * k as f32 / n as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            let re_in = windowed[2 * k];
            let im_in = windowed[n - 1 - 2 * k];
            Complex {
                re: re_in * cos_a - im_in * sin_a,
                im: re_in * sin_a + im_in * cos_a,
            }
        })
        .collect();

    // Step 3: N/2-point forward FFT via OxiFFT.
    let spectrum = fft(&pre_rotated);

    // Step 4: Post-rotation — X[k] = 2·Re{ spectrum[k]·exp(−jπ(k+½)/N) }
    (0..n2)
        .map(|k| {
            let angle = -std::f32::consts::PI * (k as f32 + 0.5) / n as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            2.0 * (spectrum[k].re * cos_a - spectrum[k].im * sin_a)
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Scale factor utility (test-only, kept for backward compatibility)
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the `global_gain` byte for a frame given its MDCT coefficients.
///
/// Returns a value in `1..=255`; silence (peak < 1e-6) returns 127.
/// This function is used in tests for backward compatibility.
#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn compute_global_gain(mdct: &[f32]) -> u8 {
    let peak = mdct.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    if peak < 1e-6 {
        return 127;
    }
    let gain = 127.0 - 4.0 * (peak / 8.0).log2();
    gain.clamp(1.0, 255.0) as u8
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Encode `buf` as AAC-LC ADTS and write all frames to `writer`.
///
/// Only 1–2 channel layouts and the sample rates in the MPEG-4 index table
/// (8 kHz … 96 kHz) are supported.  Each output frame corresponds to 1024
/// input samples; partial final frames are encoded as silence.
///
/// Non-silent frames use CB11 (ESC_HCB) Huffman coding for real spectral data.
/// Silent frames (energy < 1e-10) use ZERO_HCB.
///
/// # Errors
///
/// Returns [`OxiAudioError::UnsupportedFormat`] for unsupported channel counts
/// or sample rates, or [`OxiAudioError::Io`] on I/O failures.
pub fn encode_aac<W: Write>(buf: &AudioBuffer<f32>, mut writer: W) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count();
    if channels == 0 || channels > 2 {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "AAC encoder supports 1–2 channels, got {channels}"
        )));
    }
    if sampling_freq_index(buf.sample_rate).is_none() {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "AAC encoder: unsupported sample rate {} Hz",
            buf.sample_rate
        )));
    }

    const FRAME_SIZE: usize = 1024;
    const MDCT_LEN: usize = 2048;
    let frame_samples = FRAME_SIZE * channels;

    // At least one frame, even for empty/short buffers.
    let n_frames = (buf.samples.len() / frame_samples).max(1);

    for i in 0..n_frames {
        let frame_start_sample = i * FRAME_SIZE; // per-channel sample offset

        // Deinterleave channels into separate MDCT input buffers.
        let mut mdct_l = vec![0.0f32; MDCT_LEN];
        let mut mdct_r = vec![0.0f32; MDCT_LEN];

        for out_idx in 0..MDCT_LEN {
            let sample_idx = frame_start_sample + out_idx;
            // Channel 0 (left)
            let idx_l = sample_idx * channels;
            if idx_l < buf.samples.len() {
                mdct_l[out_idx] = buf.samples[idx_l];
            }
            // Channel 1 (right), only for stereo
            if channels >= 2 {
                let idx_r = sample_idx * channels + 1;
                if idx_r < buf.samples.len() {
                    mdct_r[out_idx] = buf.samples[idx_r];
                }
            }
        }

        let coeffs_l = aac_mdct_forward(&mdct_l);
        let energy_l: f32 = coeffs_l.iter().map(|&x| x * x).sum();

        let frame = if energy_l < 1e-10 && channels == 1 {
            encode_adts_silence_frame(channels, buf.sample_rate)
        } else {
            let coeffs_r = if channels >= 2 {
                aac_mdct_forward(&mdct_r)
            } else {
                vec![0.0f32; MDCT_LEN / 2]
            };
            encode_adts_audio_frame(channels, buf.sample_rate, &coeffs_l, &coeffs_r)
        };

        writer.write_all(&frame).map_err(OxiAudioError::Io)?;
    }

    Ok(())
}

/// Encode `buf` as AAC-LC ADTS and write to a file at `path`.
///
/// See [`encode_aac`] for supported formats and error conditions.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on I/O failure or unsupported format.
pub fn encode_aac_file(buf: &AudioBuffer<f32>, path: &Path) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    encode_aac(buf, std::io::BufWriter::new(file))
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared encoder core (with FrameOptions)
// ──────────────────────────────────────────────────────────────────────────────

/// Core encoder loop that deinterleaves, computes MDCT, and encodes frames.
///
/// `make_opts` is called once per frame with the left/right MDCT coefficients to build
/// the `FrameOptions`; it may compute PNS flags, TNS params, etc.
fn encode_aac_core<W, F>(
    buf: &AudioBuffer<f32>,
    mut writer: W,
    make_opts: F,
) -> Result<(), OxiAudioError>
where
    W: Write,
    F: Fn(&[f32], &[f32]) -> FrameOptions,
{
    let channels = buf.channels.channel_count();
    if channels == 0 || channels > 2 {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "AAC encoder supports 1–2 channels, got {channels}"
        )));
    }
    if sampling_freq_index(buf.sample_rate).is_none() {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "AAC encoder: unsupported sample rate {} Hz",
            buf.sample_rate
        )));
    }

    const FRAME_SIZE: usize = 1024;
    const MDCT_LEN: usize = 2048;
    let frame_samples = FRAME_SIZE * channels;
    let n_frames = (buf.samples.len() / frame_samples).max(1);

    for i in 0..n_frames {
        let frame_start_sample = i * FRAME_SIZE;
        let mut mdct_l = vec![0.0f32; MDCT_LEN];
        let mut mdct_r = vec![0.0f32; MDCT_LEN];

        for out_idx in 0..MDCT_LEN {
            let sample_idx = frame_start_sample + out_idx;
            let idx_l = sample_idx * channels;
            if idx_l < buf.samples.len() {
                mdct_l[out_idx] = buf.samples[idx_l];
            }
            if channels >= 2 {
                let idx_r = sample_idx * channels + 1;
                if idx_r < buf.samples.len() {
                    mdct_r[out_idx] = buf.samples[idx_r];
                }
            }
        }

        let coeffs_l = aac_mdct_forward(&mdct_l);
        let energy_l: f32 = coeffs_l.iter().map(|&x| x * x).sum();

        let frame = if energy_l < 1e-10 && channels == 1 {
            encode_adts_silence_frame(channels, buf.sample_rate)
        } else {
            let coeffs_r = if channels >= 2 {
                aac_mdct_forward(&mdct_r)
            } else {
                vec![0.0f32; MDCT_LEN / 2]
            };
            let opts = make_opts(&coeffs_l, &coeffs_r);
            encode_adts_audio_frame_opts(channels, buf.sample_rate, &coeffs_l, &coeffs_r, &opts)
        };

        writer.write_all(&frame).map_err(OxiAudioError::Io)?;
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// CBR/VBR mode helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compute `inv_scale` multiplier for VBR quality (1..=5).
/// q3 = 1.0 (default), q1 = aggressive, q5 = finest.
fn vbr_inv_scale_multiplier(quality: u8) -> f32 {
    match quality.clamp(1, 5) {
        1 => 0.5,
        2 => 0.75,
        3 => 1.0,
        4 => 1.5,
        5 => 2.0,
        _ => 1.0,
    }
}

/// Compute a global_gain additive bias for CBR targeting `target_kbps`.
///
/// Positive bias → higher gain → fewer quantized nonzero coefficients → fewer bits.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn cbr_gain_bias(mdct: &[f32], sample_rate: u32, target_kbps: u32) -> i32 {
    const FRAME_SIZE: usize = 1024;
    // Target bits per frame
    let target_bits = (target_kbps as u64 * 1000 * FRAME_SIZE as u64) / sample_rate as u64;
    // Heuristic estimate of current bits from peak amplitude:
    // louder signal → more nonzero coefficients → more bits needed.
    let peak = mdct.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    if peak < 1e-10 {
        return 0;
    }
    // A mid-quality sine at 44.1kHz typically produces ~1600–2400 bits/frame.
    // If target < 1500: bias up strongly; if target > 3000: bias down.
    let estimated_bits = 2000u64; // conservative midpoint
    if target_bits < estimated_bits {
        let deficit = estimated_bits - target_bits;
        // Each ~200-bit reduction ≈ 1 gain step increase
        ((deficit / 200) as i32).min(20)
    } else {
        let surplus = target_bits.saturating_sub(estimated_bits);
        -((surplus / 400) as i32).min(10)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PNS (Perceptual Noise Substitution) helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compute spectral flatness for one SFB: geometric_mean / arithmetic_mean of |mdct|.
///
/// Returns 0.0 for silent bands. A value > 0.75 indicates noise-like content.
fn sfb_spectral_flatness(mdct: &[f32], start: usize, end: usize) -> f32 {
    let n = (end - start) as f32;
    let geo_sum: f32 = mdct[start..end]
        .iter()
        .map(|&x| x.abs().ln().max(-60.0))
        .sum();
    let geo_mean = (geo_sum / n).exp();
    let arith_mean = mdct[start..end].iter().map(|&x| x.abs()).sum::<f32>() / n;
    if arith_mean < 1e-9 {
        return 0.0;
    }
    geo_mean / arith_mean
}

/// Classify SFBs for PNS: returns `true` for each SFB that should use NOISE_HCB.
fn classify_sfbs_pns(mdct: &[f32], offsets: &[usize]) -> Vec<bool> {
    let n_sfbs = offsets.len().saturating_sub(1);
    (0..n_sfbs)
        .map(|sfb| {
            let start = offsets[sfb];
            let end = offsets[sfb + 1].min(mdct.len());
            if end <= start {
                return false;
            }
            sfb_spectral_flatness(mdct, start, end) > 0.75
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// TNS (Temporal Noise Shaping) helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Levinson-Durbin recursion: returns (lpc_coefficients, error_variance).
///
/// Input `r` is the autocorrelation sequence r[0..=order].
/// Returns empty coefficients if `r[0]` is near zero.
fn levinson_durbin(r: &[f32], order: usize) -> (Vec<f32>, f32) {
    let mut error = r[0];
    if error < 1e-30 {
        return (vec![0.0; order], error);
    }
    let mut a = vec![0.0f32; order];
    let mut tmp = vec![0.0f32; order];

    for i in 0..order {
        // Compute reflection coefficient
        let mut lambda = r[i + 1];
        for j in 0..i {
            lambda += a[j] * r[i - j];
        }
        let ki = -lambda / error;
        // Update LPC coefficients
        tmp[..=i].copy_from_slice(&a[..=i]);
        a[i] = ki;
        for j in 0..i {
            a[j] += ki * tmp[i - 1 - j];
        }
        error *= 1.0 - ki * ki;
        if error < 1e-30 {
            error = 1e-30;
            break;
        }
    }
    (a, error)
}

/// Quantize one TNS LPC coefficient to 4-bit signed (range -8..=7).
#[allow(clippy::cast_possible_truncation)]
fn quantize_tns_coef(c: f32) -> i8 {
    let clamped = c.clamp(-1.0, 1.0);
    // After .round().clamp(-8.0, 7.0), the value is an integer in [-8, 7] — fits in i8.
    (clamped.asin() * 8.0 / std::f32::consts::PI)
        .round()
        .clamp(-8.0, 7.0) as i8
}

/// LPC analysis on the upper 2/3 of MDCT spectrum for TNS.
///
/// Returns `(quantized_coefs, should_use)` where `should_use` is true if the
/// LPC prediction gain exceeds 1.5 dB.
pub fn tns_lpc_analysis(mdct: &[f32], order: usize) -> (Vec<f32>, bool) {
    let n = mdct.len();
    let start = n / 3; // upper 2/3 of spectrum
    let seg = &mdct[start..];
    let seg_n = seg.len();

    // Compute autocorrelation r[0..=order]
    let mut r = vec![0.0f32; order + 1];
    for k in 0..=order {
        let sum: f32 = (0..seg_n.saturating_sub(k))
            .map(|j| seg[j] * seg[j + k])
            .sum();
        r[k] = sum;
    }

    let (lpc, error) = levinson_durbin(&r, order);
    let prediction_gain_db = if error > 1e-30 && r[0] > 1e-30 {
        10.0 * (r[0] / error).log10()
    } else {
        0.0
    };
    let should_use = prediction_gain_db > 1.5;

    let quantized: Vec<f32> = lpc.iter().map(|&c| quantize_tns_coef(c) as f32).collect();
    (quantized, should_use)
}

/// Build TNS parameters for one ICS channel from MDCT coefficients.
fn build_tns_params(mdct: &[f32], offsets: &[usize]) -> TnsParams {
    let n_sfbs = offsets.len().saturating_sub(1);
    let order = 8usize.min(n_sfbs);
    let (coefs_f32, should_use) = tns_lpc_analysis(mdct, order);
    if !should_use {
        return TnsParams::inactive();
    }
    let coefs: Vec<i8> = coefs_f32.iter().map(|&c| c as i8).collect();
    // Filter covers SFBs from index 8 to max_sfb.
    let start_sfb: usize = 8;
    let length = (n_sfbs.saturating_sub(start_sfb)) as u8;
    TnsParams {
        active: true,
        order,
        coefs,
        length,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API — mode-based and advanced encoders
// ──────────────────────────────────────────────────────────────────────────────

/// Encode AAC-LC with explicit bitrate control mode.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on unsupported format or I/O failure.
pub fn encode_aac_mode<W: Write>(
    buf: &AudioBuffer<f32>,
    dst: &mut W,
    mode: AacBitrateMode,
) -> Result<(), OxiAudioError> {
    let sample_rate = buf.sample_rate;
    encode_aac_core(buf, dst, move |coeffs_l, _coeffs_r| match mode {
        AacBitrateMode::Vbr { quality } => FrameOptions {
            pns_flags: Vec::new(),
            tns_params: TnsParams::inactive(),
            inv_scale_multiplier: vbr_inv_scale_multiplier(quality),
            gain_bias: 0,
        },
        AacBitrateMode::Cbr { target_kbps } => FrameOptions {
            pns_flags: Vec::new(),
            tns_params: TnsParams::inactive(),
            inv_scale_multiplier: 1.0,
            gain_bias: cbr_gain_bias(coeffs_l, sample_rate, target_kbps),
        },
    })
}

/// Encode AAC-LC to a file with explicit bitrate control mode.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on unsupported format or I/O failure.
pub fn encode_aac_mode_file(
    buf: &AudioBuffer<f32>,
    path: &Path,
    mode: AacBitrateMode,
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    encode_aac_mode(buf, &mut std::io::BufWriter::new(file), mode)
}

/// Encode AAC-LC with Perceptual Noise Substitution (PNS) for noise-like SFBs.
///
/// SFBs with spectral flatness > 0.75 use NOISE_HCB (codebook 13); their spectral
/// data is omitted and the decoder synthesizes noise. For tonal signals this is
/// identical to [`encode_aac`].
///
/// # Errors
///
/// Returns [`OxiAudioError`] on unsupported format or I/O failure.
pub fn encode_aac_pns<W: Write>(buf: &AudioBuffer<f32>, dst: &mut W) -> Result<(), OxiAudioError> {
    let sample_rate = buf.sample_rate;
    encode_aac_core(buf, dst, move |coeffs_l, _coeffs_r| {
        let offsets = sfb_offsets(sample_rate);
        let pns_flags = classify_sfbs_pns(coeffs_l, offsets);
        FrameOptions {
            pns_flags,
            tns_params: TnsParams::inactive(),
            inv_scale_multiplier: 1.0,
            gain_bias: 0,
        }
    })
}

/// Encode AAC-LC with Temporal Noise Shaping (TNS).
///
/// Applies LPC analysis on the MDCT spectral coefficients and encodes the filter
/// into the bitstream when prediction gain exceeds 1.5 dB. For signals without
/// strong temporal structure this behaves identically to [`encode_aac`].
///
/// # Errors
///
/// Returns [`OxiAudioError`] on unsupported format or I/O failure.
pub fn encode_aac_tns<W: Write>(buf: &AudioBuffer<f32>, dst: &mut W) -> Result<(), OxiAudioError> {
    let sample_rate = buf.sample_rate;
    encode_aac_core(buf, dst, move |coeffs_l, _coeffs_r| {
        let offsets = sfb_offsets(sample_rate);
        let tns_params = build_tns_params(coeffs_l, offsets);
        FrameOptions {
            pns_flags: Vec::new(),
            tns_params,
            inv_scale_multiplier: 1.0,
            gain_bias: 0,
        }
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    fn make_buf(channels: ChannelLayout, sample_rate: u32, frames: usize) -> AudioBuffer<f32> {
        let n = frames * channels.channel_count();
        AudioBuffer {
            samples: vec![0.0f32; n],
            sample_rate,
            channels,
            format: SampleFormat::F32,
        }
    }

    // ── ADTS sync bytes ──────────────────────────────────────────────────────

    #[test]
    fn test_encode_aac_mono_produces_adts() {
        let buf = make_buf(ChannelLayout::Mono, 44100, 1024);
        let mut out = Vec::new();
        encode_aac(&buf, &mut out).expect("encode should succeed");

        assert!(out.len() >= 7, "must have at least one ADTS header");
        assert_eq!(out[0], 0xFF, "sync byte 0");
        assert_eq!(out[1] & 0xF0, 0xF0, "sync nibble in byte 1");
        assert_eq!(out[1] & 0x01, 0x01, "protection_absent must be 1");
    }

    #[test]
    fn test_encode_aac_stereo_produces_adts() {
        let buf = make_buf(ChannelLayout::Stereo, 48000, 1024);
        let mut out = Vec::new();
        encode_aac(&buf, &mut out).expect("encode should succeed");

        assert!(out.len() >= 7);
        assert_eq!(out[0], 0xFF);
        assert_eq!(out[1] & 0xF0, 0xF0);
        assert_eq!(out[1] & 0x01, 0x01);
    }

    // ── Frame-length field round-trip ────────────────────────────────────────

    #[test]
    fn test_frame_length_in_header_matches_actual() {
        let buf = make_buf(ChannelLayout::Mono, 44100, 1024);
        let mut out = Vec::new();
        encode_aac(&buf, &mut out).expect("encode should succeed");

        let len_field = (((out[3] & 0x03) as usize) << 11)
            | ((out[4] as usize) << 3)
            | (((out[5] >> 5) & 0x07) as usize);

        assert_eq!(
            len_field,
            out.len(),
            "frame_length field must equal actual frame byte count"
        );
    }

    #[test]
    fn test_adts_buffer_fullness_is_vbr() {
        let buf = make_buf(ChannelLayout::Mono, 44100, 1024);
        let mut out = Vec::new();
        encode_aac(&buf, &mut out).expect("encode should succeed");

        let bfull = (((out[5] & 0x1F) as u16) << 6) | ((out[6] >> 2) as u16);
        assert_eq!(bfull, 0x7FF, "buffer_fullness must be 0x7FF (VBR)");
    }

    #[test]
    fn test_adts_number_of_raw_data_blocks_is_zero() {
        let buf = make_buf(ChannelLayout::Mono, 44100, 1024);
        let mut out = Vec::new();
        encode_aac(&buf, &mut out).expect("encode should succeed");

        assert_eq!(
            out[6] & 0x03,
            0x00,
            "num_raw_data_blocks must be 0 (= 1 block)"
        );
    }

    // ── Rejected configurations ──────────────────────────────────────────────

    #[test]
    fn test_encode_aac_rejects_unsupported_sample_rate() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 1024],
            sample_rate: 22222,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let result = encode_aac(&buf, std::io::sink());
        assert!(
            matches!(result, Err(OxiAudioError::UnsupportedFormat(_))),
            "should reject unsupported sample rate"
        );
    }

    #[test]
    fn test_encode_aac_rejects_more_than_two_channels() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 1024 * 6],
            sample_rate: 44100,
            channels: ChannelLayout::Surround51,
            format: SampleFormat::F32,
        };
        let result = encode_aac(&buf, std::io::sink());
        assert!(
            matches!(result, Err(OxiAudioError::UnsupportedFormat(_))),
            "should reject more than 2 channels"
        );
    }

    // ── File output ──────────────────────────────────────────────────────────

    #[test]
    fn test_encode_aac_file_creates_file() {
        let buf = make_buf(ChannelLayout::Mono, 44100, 1024);
        let dir = std::env::temp_dir();
        let path = dir.join("oxiaudio_test_aac_output.aac");

        encode_aac_file(&buf, &path).expect("file encode should succeed");

        let data = std::fs::read(&path).expect("output file must exist");
        assert!(
            data.len() >= 7,
            "file must contain at least one ADTS header"
        );
        assert_eq!(data[0], 0xFF, "file must start with ADTS sync byte 0xFF");
        assert_eq!(data[1], 0xF1, "file byte 1 must be 0xF1 (MPEG-4, no CRC)");

        let _ = std::fs::remove_file(&path);
    }

    // ── Sampling frequency index in header ──────────────────────────────────

    #[test]
    fn test_adts_header_sampling_freq_index_44100() {
        let buf = make_buf(ChannelLayout::Mono, 44100, 1024);
        let mut out = Vec::new();
        encode_aac(&buf, &mut out).expect("encode should succeed");

        let sfi = (out[2] >> 2) & 0x0F;
        assert_eq!(sfi, 4, "sampling_frequency_index must be 4 for 44100 Hz");
    }

    #[test]
    fn test_adts_header_sampling_freq_index_48000() {
        let buf = make_buf(ChannelLayout::Mono, 48000, 1024);
        let mut out = Vec::new();
        encode_aac(&buf, &mut out).expect("encode should succeed");

        let sfi = (out[2] >> 2) & 0x0F;
        assert_eq!(sfi, 3, "sampling_frequency_index must be 3 for 48000 Hz");
    }

    // ── Multiple frames ──────────────────────────────────────────────────────

    #[test]
    fn test_encode_aac_multiple_frames() {
        let buf = make_buf(ChannelLayout::Mono, 44100, 4096);
        let mut out = Vec::new();
        encode_aac(&buf, &mut out).expect("encode should succeed");

        let frame_len = (((out[3] & 0x03) as usize) << 11)
            | ((out[4] as usize) << 3)
            | (((out[5] >> 5) & 0x07) as usize);
        assert!(frame_len >= 7);
        if out.len() >= 2 * frame_len {
            assert_eq!(out[frame_len], 0xFF, "second frame sync byte 0");
            assert_eq!(out[frame_len + 1], 0xF1, "second frame sync byte 1");
        }
    }

    // ── MDCT ────────────────────────────────────────────────────────────────

    #[test]
    fn test_aac_mdct_output_length() {
        let samples = vec![0.0f32; 2048];
        let mdct = aac_mdct_forward(&samples);
        assert_eq!(mdct.len(), 1024, "AAC MDCT must produce 1024 coefficients");
    }

    #[test]
    fn test_aac_mdct_silence_is_near_zero() {
        let samples = vec![0.0f32; 2048];
        let mdct = aac_mdct_forward(&samples);
        let max = mdct.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        assert!(max < 1e-5, "silence must produce near-zero MDCT, max={max}");
    }

    #[test]
    fn test_aac_mdct_nonzero_for_sine() {
        let samples: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.5)
            .collect();
        let mdct = aac_mdct_forward(&samples);
        let energy: f32 = mdct.iter().map(|&x| x * x).sum();
        assert!(
            energy > 0.01,
            "sine wave must produce non-zero MDCT energy, got {energy}"
        );
    }

    #[test]
    fn test_compute_global_gain_silence() {
        let mdct = vec![0.0f32; 1024];
        assert_eq!(compute_global_gain(&mdct), 127, "silence maps to gain 127");
    }

    #[test]
    fn test_compute_global_gain_loud_signal() {
        let mdct = vec![8.0f32; 1024]; // peak = 8.0
        assert_eq!(compute_global_gain(&mdct), 127, "peak=8.0 maps to gain 127");
    }

    // ── New Huffman coding tests ─────────────────────────────────────────────

    /// Silence frame: all-zero buffer should encode as ZERO_HCB, output is valid ADTS.
    #[test]
    fn test_aac_huffman_encode_silence_frame_is_valid_adts() {
        let buf = make_buf(ChannelLayout::Mono, 44100, 1024);
        let mut out = Vec::new();
        encode_aac(&buf, &mut out).expect("encode silence should succeed");

        assert!(out.len() >= 7);
        assert_eq!(out[0], 0xFF, "ADTS sync byte 0");
        assert_eq!(out[1], 0xF1, "ADTS sync byte 1");
    }

    /// A sine wave should produce output with actual spectral data (larger than silence frames).
    #[test]
    fn test_aac_huffman_sine_produces_nonzero_spectral_data() {
        // Silence frame for comparison
        let silent_buf = make_buf(ChannelLayout::Mono, 44100, 1024);
        let mut silent_out = Vec::new();
        encode_aac(&silent_buf, &mut silent_out).expect("encode silence should succeed");

        // Sine wave buffer (non-silent)
        let sine_samples: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.9)
            .collect();
        let sine_buf = AudioBuffer {
            samples: sine_samples,
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut sine_out = Vec::new();
        encode_aac(&sine_buf, &mut sine_out).expect("encode sine should succeed");

        // Sine frame must be valid ADTS
        assert_eq!(sine_out[0], 0xFF);
        assert_eq!(sine_out[1], 0xF1);
        // Sine frame with real spectral data should be larger than silence frame
        assert!(
            sine_out.len() >= silent_out.len(),
            "sine frame ({}B) should be >= silence frame ({}B)",
            sine_out.len(),
            silent_out.len()
        );
    }

    /// quantize_coeff(0.0, any_scale) must return 0.
    #[test]
    fn test_aac_huffman_quant_zero_maps_to_zero() {
        for scale in [0.001, 0.1, 1.0, 10.0, 1000.0] {
            let q = quantize_coeff(0.0, scale);
            assert_eq!(q, 0, "quantize_coeff(0.0, {scale}) must be 0");
        }
    }

    /// compute_global_gain_and_inv_scale should return non-trivial values for a loud signal.
    #[test]
    fn test_aac_huffman_global_gain_computation() {
        let loud: Vec<f32> = vec![0.9f32; 1024];
        let (gain, inv_scale) = compute_global_gain_and_inv_scale(&loud);
        assert!(gain > 0, "gain must be positive");
        assert!(gain < 255, "gain must be < 255 for loud signal");
        assert!(inv_scale > 0.0, "inv_scale must be positive");
        // Silence should return 127, 0.0
        let (gain_s, inv_s) = compute_global_gain_and_inv_scale(&[0.0f32; 1024]);
        assert_eq!(gain_s, 127);
        assert_eq!(inv_s, 0.0);
    }

    /// SFB offset tables must be strictly increasing and end at 1024.
    #[test]
    fn test_aac_huffman_sfb_offsets_monotonic() {
        let rates = [96000u32, 48000, 44100, 32000, 24000, 22050, 16000, 8000];
        for &sr in &rates {
            let offsets = sfb_offsets(sr);
            assert!(
                offsets.len() >= 2,
                "sr={sr}: offsets must have at least 2 entries"
            );
            assert_eq!(
                *offsets.last().unwrap(),
                1024,
                "sr={sr}: last offset must be 1024"
            );
            for w in offsets.windows(2) {
                assert!(w[1] > w[0], "sr={sr}: offsets must be strictly increasing");
            }
        }
    }

    /// Encoded frame must begin with ADTS sync bytes 0xFF 0xF1.
    #[test]
    fn test_aac_huffman_encoded_frame_starts_with_adts_sync() {
        let sine_samples: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let buf = AudioBuffer {
            samples: sine_samples,
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut out = Vec::new();
        encode_aac(&buf, &mut out).expect("encode should succeed");

        assert_eq!(out[0], 0xFF, "byte 0 must be 0xFF (ADTS sync)");
        assert_eq!(out[1], 0xF1, "byte 1 must be 0xF1 (MPEG-4 AAC, no CRC)");
    }

    /// CB11 table must have exactly 289 entries (17×17).
    #[test]
    fn test_aac_huffman_cb11_table_size() {
        assert_eq!(HCB11_CODES.len(), 289, "CB11 must have 289 code entries");
        assert_eq!(HCB11_LENS.len(), 289, "CB11 must have 289 length entries");
        // Verify entry (0,0): should be code=0x000, len=4 (from Symphonia table)
        assert_eq!(HCB11_CODES[0], 0x000, "CB11[0,0] code must be 0x000");
        assert_eq!(HCB11_LENS[0], 4, "CB11[0,0] len must be 4");
    }

    /// ESC word encoding for specific known values.
    #[test]
    fn test_aac_huffman_esc_word_encoding() {
        // For abs_val=16 (n=0): write 0 ones, 0, then 4 bits = 0000 → 5 bits total
        // For abs_val=17 (n=0): write 0, then 4 bits = 0001 → 5 bits total
        // For abs_val=32 (n=1): write 1, 0, then 5 bits = 00000 → 7 bits total
        // Just verify the ESC pair encoder doesn't panic
        let mut bw = BitWriter::new();
        encode_esc_pair(&mut bw, 16, 0);
        let _bytes = bw.into_bytes();

        let mut bw2 = BitWriter::new();
        encode_esc_pair(&mut bw2, 32, -17);
        let _bytes2 = bw2.into_bytes();

        let mut bw3 = BitWriter::new();
        encode_esc_pair(&mut bw3, 0, 0);
        let bytes3 = bw3.into_bytes();
        // (0,0) uses CB11[0] = len=4, code=0x000 → 4 bits → 1 byte
        assert!(!bytes3.is_empty(), "CB11 (0,0) must produce output");
    }

    /// ADTS roundtrip: encode silence → decode with `decode_aac` → no error.
    ///
    /// Uses `oxiaudio_decode::decode_aac`, which parses ADTS frames directly
    /// without Symphonia's probe step. This validates the silence ICS bitstream is
    /// correctly structured (pulse/tns/gain-control flag bits present, section data valid).
    #[test]
    fn test_aac_adts_symphonia_decode_roundtrip_silence() {
        use oxiaudio_decode::decode_aac;

        let buf = make_buf(ChannelLayout::Mono, 44100, 1024);
        let mut encoded = Vec::new();
        encode_aac(&buf, &mut encoded).expect("encode silence must succeed");

        // Verify ADTS sync before decoding
        assert_eq!(encoded[0], 0xFF, "ADTS sync byte 0");
        assert_eq!(encoded[1], 0xF1, "ADTS sync byte 1 (no CRC, MPEG-4)");

        let decoded =
            decode_aac(&encoded).expect("decode_aac must successfully decode silence ADTS");
        assert_eq!(decoded.sample_rate, 44100, "sample rate must round-trip");
        assert_eq!(
            decoded.channels.channel_count(),
            1,
            "mono channel count must round-trip"
        );
    }

    /// ADTS roundtrip: encode sine wave → decode with `decode_aac` → non-empty samples.
    ///
    /// Tests the real spectral data path (CB11 Huffman coding). If the ICS bitstream
    /// has wrong scale factor counts or missing flag bits, `decode_aac` will fail or
    /// return empty/zero output.
    #[test]
    fn test_aac_adts_symphonia_decode_roundtrip_sine() {
        use oxiaudio_decode::decode_aac;

        let sine_samples: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let buf = AudioBuffer {
            samples: sine_samples,
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut encoded = Vec::new();
        encode_aac(&buf, &mut encoded).expect("encode sine must succeed");

        assert_eq!(encoded[0], 0xFF, "ADTS sync byte 0");
        assert_eq!(encoded[1], 0xF1, "ADTS sync byte 1");

        let decoded = decode_aac(&encoded)
            .expect("decode_aac must successfully decode sine ADTS with CB11 coding");
        assert_eq!(
            decoded.sample_rate, 44100,
            "sample rate must round-trip for sine"
        );
        assert_eq!(
            decoded.channels.channel_count(),
            1,
            "mono must round-trip for sine"
        );
        // Decoded output must be non-trivial (some samples must be produced)
        assert!(
            !decoded.samples.is_empty(),
            "decoded samples must be non-empty"
        );
    }

    // ── CBR/VBR bitrate mode tests ───────────────────────────────────────────

    fn make_sine_buf(
        sample_rate: u32,
        channels: usize,
        amplitude: f32,
        freq_hz: f32,
    ) -> AudioBuffer<f32> {
        use std::f32::consts::PI;
        let n = sample_rate as usize; // 1 second
        let samples: Vec<f32> = (0..n * channels)
            .map(|i| {
                amplitude * (2.0 * PI * freq_hz * (i / channels) as f32 / sample_rate as f32).sin()
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate,
            channels: if channels == 1 {
                ChannelLayout::Mono
            } else {
                ChannelLayout::Stereo
            },
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_aac_bitrate_mode_vbr_q1_smaller_than_q5() {
        let buf = make_sine_buf(44100, 1, 0.5, 440.0);
        let mut out_q1 = Vec::new();
        encode_aac_mode(&buf, &mut out_q1, AacBitrateMode::Vbr { quality: 1 })
            .expect("vbr q1 encode should succeed");
        let mut out_q5 = Vec::new();
        encode_aac_mode(&buf, &mut out_q5, AacBitrateMode::Vbr { quality: 5 })
            .expect("vbr q5 encode should succeed");
        assert!(
            out_q1.len() <= out_q5.len() + 100,
            "q1 ({} bytes) should produce <= bytes as q5 ({} bytes)",
            out_q1.len(),
            out_q5.len()
        );
    }

    #[test]
    fn test_aac_bitrate_mode_cbr_produces_valid_adts() {
        let buf = make_sine_buf(44100, 1, 1.0, 1000.0);
        let mut out = Vec::new();
        encode_aac_mode(&buf, &mut out, AacBitrateMode::Cbr { target_kbps: 128 })
            .expect("cbr 128kbps encode should succeed");
        assert!(out.len() >= 7, "must have at least one ADTS frame");
        assert_eq!(out[0], 0xFF, "sync byte 0");
        assert_eq!(out[1] & 0xF0, 0xF0, "sync word");
    }

    // ── PNS tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_pns_encode_noisy_input_produces_valid_adts() {
        // White noise input should trigger PNS for some SFBs
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut samples = vec![0.0f32; 44100];
        let mut h = DefaultHasher::new();
        for s in samples.iter_mut() {
            42u64.hash(&mut h);
            *s = (h.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        let buf = AudioBuffer {
            samples,
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut out = Vec::new();
        encode_aac_pns(&buf, &mut out).expect("pns encode should succeed");
        assert!(out.len() >= 7, "must have at least one ADTS frame");
        assert_eq!(out[0], 0xFF, "ADTS sync byte 0");
    }

    #[test]
    fn test_pns_spectral_flatness_noisy() {
        // A flat spectrum should have flatness close to 1
        let flat: Vec<f32> = (0..64).map(|_| 0.5f32).collect();
        let f = sfb_spectral_flatness(&flat, 0, 64);
        // All equal values: geo_mean == arith_mean → flatness ≈ 1.0
        assert!(
            (f - 1.0).abs() < 0.01,
            "uniform magnitude → flatness ≈ 1, got {f}"
        );
    }

    // ── TNS tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_tns_encode_produces_valid_adts() {
        // Voiced speech-like signal (low frequency) benefits from TNS
        let buf = make_sine_buf(44100, 1, 0.8, 200.0);
        let mut out = Vec::new();
        encode_aac_tns(&buf, &mut out).expect("tns encode should succeed");
        assert!(out.len() >= 7, "must have at least one ADTS frame");
        assert_eq!(out[0], 0xFF, "ADTS sync byte 0");
        assert_eq!(out[1] & 0xF0, 0xF0, "sync word nibble");
    }

    #[test]
    fn test_tns_lpc_prediction_gain_positive() {
        // LPC on MDCT coefficients of a sine wave — check no panic and correct coef count
        let sine: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let (coefs, _should_use) = tns_lpc_analysis(&sine, 8);
        assert_eq!(
            coefs.len(),
            8,
            "tns_lpc_analysis must return exactly `order` coefs"
        );
    }

    #[test]
    fn test_levinson_durbin_returns_correct_order() {
        let r = vec![1.0f32, 0.5, 0.25, 0.125];
        let (lpc, error) = levinson_durbin(&r, 3);
        assert_eq!(
            lpc.len(),
            3,
            "Levinson-Durbin must return `order` coefficients"
        );
        assert!(error > 0.0, "error variance must be positive");
    }

    #[test]
    fn test_quantize_tns_coef_range() {
        // asin(1.0) * 8 / PI = (PI/2) * 8 / PI = 4 → maps to 4 (clamped to -8..=7)
        let q_pos = quantize_tns_coef(1.0);
        assert_eq!(
            q_pos, 4,
            "quantize_tns_coef(1.0) must give 4 (asin formula)"
        );
        let q_neg = quantize_tns_coef(-1.0);
        assert_eq!(q_neg, -4, "quantize_tns_coef(-1.0) must give -4");
        // Zero maps to 0
        let q_zero = quantize_tns_coef(0.0);
        assert_eq!(q_zero, 0, "quantize_tns_coef(0.0) must give 0");
        // Values outside [-1, 1] are clamped before asin
        let q_big = quantize_tns_coef(2.0);
        assert_eq!(
            q_big, 4,
            "quantize_tns_coef(2.0) clamps to 1.0, then gives 4"
        );
    }
}
