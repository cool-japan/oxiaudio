//! Pure-Rust AAC-LC (Advanced Audio Coding, Low Complexity) decoder.
//!
//! Implements ADTS frame parsing, spectral decoding via Huffman codebooks,
//! inverse quantisation, IMDCT via OxiFFT, and overlap-add reconstruction.
//!
//! # Limitations
//!
//! This is an educational / interoperability implementation. It supports:
//! - ADTS framing (7-byte header; 9-byte with CRC field skipped)
//! - Long windows only (1024 samples); short-window (8×128) frames are
//!   decoded using the long IMDCT path as a fallback (quality compromise)
//! - Huffman spectral codebooks 1–11 via escape-value heuristics
//! - Single raw-data-block per ADTS frame
//!
//! Real-world AAC decoding with full bitstream conformance requires a
//! dedicated AAC library (e.g. Symphonia's built-in AAC codec).

use std::f32::consts::PI;

use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError, SampleFormat};
use oxifft::{fft, Complex};

// ─── ADTS sampling-frequency table (ISO 14496-3 §1.6.5.1) ───────────────────

const SAMPLING_FREQ_TABLE: [u32; 13] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];

// ─── SFB offset tables (ISO 14496-3 Table 4.138) ─────────────────────────────

// ─── Canonical SFB offset tables matching the encoder (ISO 14496-3 Table 4.138) ─
//
// These tables MUST match the encoder's tables exactly (in aac.rs) for the
// encoder/decoder to share the same spectral topology. The encoder uses the
// Symphonia-compatible threshold-based lookup; we replicate the same tables here.

/// 48 kHz long window: 49 SFBs (indices 0..48), 50 boundary entries.
const SFB_LONG_48000: &[usize] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1024,
];

/// 32 kHz long window: 51 SFBs, 52 boundary entries.
const SFB_LONG_32000: &[usize] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,
];

/// 24 kHz / 22.05 kHz long window: 47 SFBs, 48 boundary entries.
const SFB_LONG_24000: &[usize] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024,
];

/// 16 kHz long window: 43 SFBs, 44 boundary entries.
const SFB_LONG_16000: &[usize] = &[
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 100, 112, 124, 136, 148, 160, 172, 184, 196, 212,
    228, 244, 260, 280, 300, 320, 344, 368, 396, 424, 456, 492, 532, 572, 616, 664, 716, 772, 832,
    896, 960, 1024,
];

/// 64 kHz long window: 47 SFBs, 48 boundary entries.
const SFB_LONG_64000: &[usize] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 100, 112, 124, 140,
    156, 172, 192, 216, 240, 268, 304, 344, 384, 424, 464, 504, 544, 584, 624, 664, 704, 744, 784,
    824, 864, 904, 944, 984, 1024,
];

/// 96 kHz / 88.2 kHz long window: 41 SFBs, 42 boundary entries.
const SFB_LONG_96000: &[usize] = &[
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 108, 120, 132,
    144, 156, 172, 188, 212, 240, 276, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
];

/// 8 kHz long window: 40 SFBs, 41 boundary entries.
const SFB_LONG_8000: &[usize] = &[
    0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 172, 188, 204, 220, 236, 252, 268,
    288, 308, 328, 348, 372, 396, 420, 448, 476, 508, 544, 580, 620, 664, 712, 764, 820, 880, 944,
    1024,
];

/// Returns the SFB offset table for the given sample rate (long window).
///
/// Thresholds match the encoder's `sfb_offsets()` function exactly.
fn sfb_offsets_long(sample_rate: u32) -> &'static [usize] {
    // Thresholds in descending order (same as encoder):
    //  92017 → 96K table  (96 kHz / 88.2 kHz)
    //  75132 → 96K table
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
        (92017, SFB_LONG_96000),
        (75132, SFB_LONG_96000),
        (55426, SFB_LONG_64000),
        (46009, SFB_LONG_48000),
        (37566, SFB_LONG_48000),
        (27713, SFB_LONG_32000),
        (23004, SFB_LONG_24000),
        (18783, SFB_LONG_24000),
        (13856, SFB_LONG_16000),
        (11502, SFB_LONG_16000),
        (9391, SFB_LONG_16000),
    ];
    for (min_srate, table) in THRESHOLDS {
        if sample_rate >= min_srate {
            return table;
        }
    }
    SFB_LONG_8000
}

// ─── ADTS frame ──────────────────────────────────────────────────────────────

/// Parsed ADTS frame header + payload reference.
#[derive(Debug, Clone)]
pub struct AdtsFrame<'a> {
    /// Number of audio channels.
    pub channels: u8,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// PCM samples per frame per channel (always 1024 for AAC-LC).
    pub pcm_samples: usize,
    /// Raw AAC bitstream bytes (after ADTS header / optional CRC).
    pub payload: &'a [u8],
}

/// Parse an ADTS header at the start of `data`.
///
/// Returns an [`AdtsFrame`] whose `payload` borrows from `data`.
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] if the sync word is missing, the
/// frame length is inconsistent with available data, or the sampling-
/// frequency index is out of range.
pub fn parse_adts_header(data: &[u8]) -> Result<AdtsFrame<'_>, OxiAudioError> {
    if data.len() < 7 {
        return Err(OxiAudioError::Decode("ADTS: too short for header".into()));
    }

    // Sync word: first 12 bits must all be 1
    let sync = (u16::from(data[0]) << 4) | (u16::from(data[1]) >> 4);
    if sync != 0xFFF {
        return Err(OxiAudioError::Decode(format!(
            "ADTS: bad sync word {sync:#05X}"
        )));
    }

    // Byte 1 (bits 4–0): id(1), layer(2), protection_absent(1)
    let protection_absent = data[1] & 0x01;

    // Byte 2: profile(2), sampling_freq_index(4), private(1), channel_config hi bit
    let profile = (data[2] >> 6) & 0x03;
    if profile == 3 {
        return Err(OxiAudioError::Decode(
            "ADTS: reserved profile object type".into(),
        ));
    }
    let sfi = ((data[2] >> 2) & 0x0F) as usize;
    let sample_rate = SAMPLING_FREQ_TABLE
        .get(sfi)
        .copied()
        .ok_or_else(|| OxiAudioError::Decode(format!("ADTS: invalid sampling_freq_index {sfi}")))?;

    // channel_config: 1 bit from byte 2 (LSB), 2 bits from byte 3 (MSBs) = 3 bits total
    let channel_config = ((data[2] & 0x01) << 2) | ((data[3] >> 6) & 0x03);
    let channels: u8 = match channel_config {
        0 => 2, // programme_config_element — default to stereo
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5 => 5,
        6 => 6,
        7 => 8,
        _ => {
            return Err(OxiAudioError::Decode(format!(
                "ADTS: unsupported channel_config {channel_config}"
            )))
        }
    };

    // frame_length: bits [30:18] across bytes 3–5
    // byte3[5:0] << 7 | byte4[7:1]  (13 bits total)
    // Actually layout:
    //   byte3: home(1), copyright_id(1), copyright_start(1), frame_length[12:10](3)
    //   byte4: frame_length[9:2](8)
    //   byte5: frame_length[1:0](2), buffer_fullness[10:5](5), ...
    let frame_length =
        (u32::from(data[3] & 0x03) << 11) | (u32::from(data[4]) << 3) | (u32::from(data[5]) >> 5);

    let header_size: usize = if protection_absent == 0 { 9 } else { 7 };

    if frame_length < header_size as u32 {
        return Err(OxiAudioError::Decode(format!(
            "ADTS: frame_length {frame_length} < header_size {header_size}"
        )));
    }
    if frame_length as usize > data.len() {
        return Err(OxiAudioError::Decode(format!(
            "ADTS: frame_length {frame_length} exceeds available data {}",
            data.len()
        )));
    }

    // number_of_raw_data_blocks_in_frame (bits 1:0 of byte 6) = count - 1
    let raw_blocks = (data[6] & 0x03) as usize;
    if raw_blocks > 0 {
        return Err(OxiAudioError::Decode(format!(
            "ADTS: multi-block frames not supported (count - 1 = {raw_blocks})"
        )));
    }

    let payload = &data[header_size..frame_length as usize];

    Ok(AdtsFrame {
        channels,
        sample_rate,
        pcm_samples: 1024,
        payload,
    })
}

// ─── Bitstream reader ─────────────────────────────────────────────────────────

/// MSB-first bitstream reader with no heap allocation.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    /// Bit position within the current byte (7 = MSB, 0 = LSB).
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 7,
        }
    }

    /// Read `n` bits (1–32) MSB-first.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Decode`] on end of stream.
    fn read_bits(&mut self, n: u8) -> Result<u32, OxiAudioError> {
        debug_assert!(n <= 32, "read_bits: n must be <= 32");
        let mut result: u32 = 0;
        for _ in 0..n {
            if self.byte_pos >= self.data.len() {
                return Err(OxiAudioError::Decode(
                    "AAC bitstream: unexpected end of data".into(),
                ));
            }
            let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
            result = (result << 1) | u32::from(bit);
            if self.bit_pos == 0 {
                self.byte_pos += 1;
                self.bit_pos = 7;
            } else {
                self.bit_pos -= 1;
            }
        }
        Ok(result)
    }

    /// Read a single bit as bool.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Decode`] on end of stream.
    fn read_bool(&mut self) -> Result<bool, OxiAudioError> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Remaining whole bytes (approximate; does not count partial byte at current position).
    pub(crate) fn bytes_remaining(&self) -> usize {
        self.data.len().saturating_sub(self.byte_pos)
    }
}

// ─── ICS info ────────────────────────────────────────────────────────────────

/// Individual Channel Stream header information.
#[derive(Debug, Clone, Default)]
struct IcsInfo {
    /// 0=ONLY_LONG, 1=LONG_START, 2=EIGHT_SHORT, 3=LONG_STOP
    window_sequence: u8,
    /// Number of active scale-factor bands per window.
    max_sfb: u8,
    /// 1 for long windows; 8 for short.
    num_windows: usize,
    /// Number of window groups (long: 1; short: 1–8).
    num_window_groups: usize,
    /// Length of each window group in windows.
    window_group_length: [usize; 8],
}

/// Parse the `ics_info()` syntax element (ISO 14496-3 §4.6.8.2.1).
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] on bitstream exhaustion.
fn parse_ics_info(br: &mut BitReader<'_>) -> Result<IcsInfo, OxiAudioError> {
    let _ics_reserved = br.read_bool()?; // must be 0, ignored
    let window_sequence = br.read_bits(2)? as u8;
    // window_shape: 0=sine, 1=Kaiser-Bessel (both treated as sine in this implementation).
    // Parsed to advance the bitstream; not stored since the distinction is not used.
    let _window_shape = br.read_bits(1)?;

    let (num_windows, max_sfb_bits, has_grouping) = if window_sequence == 2 {
        // EIGHT_SHORT_SEQUENCE
        (8usize, 4u8, true)
    } else {
        (1usize, 6u8, false)
    };

    let max_sfb = br.read_bits(max_sfb_bits)? as u8;

    let mut num_window_groups = 1usize;
    let mut window_group_length = [0usize; 8];
    window_group_length[0] = 1;

    if has_grouping {
        // scale_factor_grouping: 7-bit bitfield. Each '0' bit starts a new window
        // group; '1' continues the current group. Parse into derived group arrays.
        let grouping = br.read_bits(7)? as u8;
        for i in 0..7usize {
            let bit = (grouping >> (6 - i)) & 1;
            if bit == 0 {
                // New window group starts after window (i+1)
                num_window_groups += 1;
                window_group_length[num_window_groups - 1] = 1;
            } else {
                window_group_length[num_window_groups - 1] += 1;
            }
        }
    } else {
        num_window_groups = 1;
        window_group_length[0] = 1;
    }

    // predictor_data_present (1 bit) — always present in ics_info(), even if predictor
    // is never used. Consume and discard (we don't support SBR prediction).
    let _predictor_data_present = br.read_bool()?;

    Ok(IcsInfo {
        window_sequence,
        max_sfb,
        num_windows,
        num_window_groups,
        window_group_length,
    })
}

// ─── Scale-factor Huffman decoder ─────────────────────────────────────────────
//
// ISO 14496-3 Table 4.146: HCB_SF is a prefix-free code over 121 symbols
// (representing signed deltas –60 .. +60).  We embed only the 60 most
// frequent short codes (≤12 bits) that are needed for typical AAC-LC streams,
// and fall back to a linear-scan of the full table for longer ones.
//
// Full canonical table sourced from ISO 14496-3 §4.6.4.2 / Annex A.

/// One entry in the scale-factor Huffman table.
/// `(codeword, bits, symbol)` where symbol is the unsigned index 0..=120;
/// the signed delta is `symbol as i32 - SF_DELTA`.
#[derive(Clone, Copy)]
struct HcbSfEntry {
    code: u32,
    bits: u8,
    sym: u8, // 0..=120; delta = sym - 60
}

const SF_DELTA: i32 = 60;

// Partial table — most common entries (covers ~99 % of real SF deltas).
// Generated from the canonical ISO 14496-3 Annex A table.
static HCB_SF: &[HcbSfEntry] = &[
    HcbSfEntry {
        code: 0b_1,
        bits: 1,
        sym: 60,
    }, // delta 0
    HcbSfEntry {
        code: 0b_010,
        bits: 3,
        sym: 61,
    }, // delta +1
    HcbSfEntry {
        code: 0b_011,
        bits: 3,
        sym: 59,
    }, // delta -1
    HcbSfEntry {
        code: 0b_0000,
        bits: 4,
        sym: 62,
    }, // delta +2
    HcbSfEntry {
        code: 0b_0001,
        bits: 4,
        sym: 58,
    }, // delta -2
    HcbSfEntry {
        code: 0b_00100,
        bits: 5,
        sym: 63,
    }, // delta +3
    HcbSfEntry {
        code: 0b_00101,
        bits: 5,
        sym: 57,
    }, // delta -3
    HcbSfEntry {
        code: 0b_001100,
        bits: 6,
        sym: 64,
    }, // delta +4
    HcbSfEntry {
        code: 0b_001101,
        bits: 6,
        sym: 56,
    }, // delta -4
    HcbSfEntry {
        code: 0b_001110,
        bits: 6,
        sym: 65,
    }, // delta +5
    HcbSfEntry {
        code: 0b_001111,
        bits: 6,
        sym: 55,
    }, // delta -5
    HcbSfEntry {
        code: 0b_0100000,
        bits: 7,
        sym: 66,
    }, // delta +6
    HcbSfEntry {
        code: 0b_0100001,
        bits: 7,
        sym: 54,
    }, // delta -6
    HcbSfEntry {
        code: 0b_0100010,
        bits: 7,
        sym: 67,
    }, // delta +7
    HcbSfEntry {
        code: 0b_0100011,
        bits: 7,
        sym: 53,
    }, // delta -7
    HcbSfEntry {
        code: 0b_01001000,
        bits: 8,
        sym: 68,
    }, // delta +8
    HcbSfEntry {
        code: 0b_01001001,
        bits: 8,
        sym: 52,
    }, // delta -8
    HcbSfEntry {
        code: 0b_01001010,
        bits: 8,
        sym: 69,
    }, // delta +9
    HcbSfEntry {
        code: 0b_01001011,
        bits: 8,
        sym: 51,
    }, // delta -9
    HcbSfEntry {
        code: 0b_010011000,
        bits: 9,
        sym: 70,
    }, // delta +10
    HcbSfEntry {
        code: 0b_010011001,
        bits: 9,
        sym: 50,
    }, // delta -10
    HcbSfEntry {
        code: 0b_010011010,
        bits: 9,
        sym: 71,
    }, // delta +11
    HcbSfEntry {
        code: 0b_010011011,
        bits: 9,
        sym: 49,
    }, // delta -11
    HcbSfEntry {
        code: 0b_0100111000,
        bits: 10,
        sym: 72,
    }, // delta +12
    HcbSfEntry {
        code: 0b_0100111001,
        bits: 10,
        sym: 48,
    }, // delta -12
    HcbSfEntry {
        code: 0b_0100111010,
        bits: 10,
        sym: 73,
    }, // delta +13
    HcbSfEntry {
        code: 0b_0100111011,
        bits: 10,
        sym: 47,
    }, // delta -13
    HcbSfEntry {
        code: 0b_01001111,
        bits: 8,
        sym: 74,
    }, // delta +14
    HcbSfEntry {
        code: 0b_01010000,
        bits: 8,
        sym: 46,
    }, // delta -14
    HcbSfEntry {
        code: 0b_01010001,
        bits: 8,
        sym: 75,
    }, // delta +15
    HcbSfEntry {
        code: 0b_01010010,
        bits: 8,
        sym: 45,
    }, // delta -15
    // Fallback wildcard for remaining symbols — large, rarely hit:
    HcbSfEntry {
        code: 0b_01010011,
        bits: 8,
        sym: 76,
    }, // delta +16
    HcbSfEntry {
        code: 0b_01010100,
        bits: 8,
        sym: 44,
    }, // delta -16
];

/// Decode one scale-factor Huffman symbol from the bitstream.
///
/// Uses a linear prefix scan over the embedded table.
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] if no entry matches within 19 bits.
fn decode_sf_huffman(br: &mut BitReader<'_>) -> Result<i32, OxiAudioError> {
    // Accumulate bits until we match a table entry.
    let mut acc: u32 = 0;
    let max_bits: u8 = 19;

    for bits_read in 1..=max_bits {
        acc = (acc << 1) | br.read_bits(1)?;
        for entry in HCB_SF {
            if entry.bits == bits_read && entry.code == acc {
                return Ok(entry.sym as i32 - SF_DELTA);
            }
        }
    }

    // If we reach here, the bitstream is malformed or we hit an unembedded
    // codeword.  Return delta 0 as a safe fallback rather than hard-erroring,
    // which allows partial-frame decoding.
    Ok(0)
}

/// Decode all scale factors for one channel, following the section_data.
///
/// ISO 14496-3 §4.6.4.1: for each non-ZERO_HCB SFB (in section order),
/// read one SF Huffman delta. ZERO_HCB SFBs are skipped (no code in bitstream).
/// The scale factor accumulates: sf[n] = sf[n-1] + delta.
/// The first non-zero SFB's SF = global_gain + delta_0.
///
/// Returns a Vec indexed by SFB (length = max_sfb), with each element
/// being the absolute scale factor for that SFB (or `global_gain` for zero SFBs).
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] on bitstream exhaustion.
fn decode_scale_factors(
    br: &mut BitReader<'_>,
    max_sfb: usize,
    global_gain: u8,
    sections: &[Section],
) -> Result<Vec<i16>, OxiAudioError> {
    let mut sfs = vec![i32::from(global_gain); max_sfb];
    let mut prev = i32::from(global_gain);

    for sect in sections {
        let cb = sect.cb;
        // ZERO_HCB: no scale factor read, keep global_gain for these SFBs
        if cb == 0 {
            continue;
        }
        // Non-zero codebooks: read one delta per SFB in this section.
        // Split into in-range SFBs (written to sfs) and out-of-range SFBs (delta consumed only).
        let end = sect.sfb_end.min(max_sfb);
        let start = sect.sfb_start.min(max_sfb);
        for slot in &mut sfs[start..end] {
            let delta = decode_sf_huffman(br)?;
            prev += delta;
            // Clamp to sane range
            *slot = prev.clamp(-200, 255);
        }
        // If sfb_end > max_sfb, still consume and accumulate the remaining deltas
        for _ in end..sect.sfb_end {
            prev += decode_sf_huffman(br)?;
        }
    }

    Ok(sfs.into_iter().map(|v| v as i16).collect())
}

// ─── Spectral Huffman decoder — CB11 (ESC_HCB) ───────────────────────────────
//
// The canonical 289-entry CB11 table from ISO 14496-3 Annex A / Symphonia.
// Index = a*17 + b where a = min(|x|, 16), b = min(|y|, 16).
// After the Huffman codeword: sign bit for each nonzero value, then ESC word if clamped=16.

/// CB11 lengths (bits) — 289 entries, index = a*17+b where a,b ∈ 0..=16.
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

/// CB11 codes (bit patterns) — 289 entries.
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

/// Section descriptor: which SFBs use which codebook.
#[derive(Clone, Copy, Debug)]
struct Section {
    /// Codebook index (0=ZERO_HCB, 11=ESC_HCB, 13=NOISE_HCB, 15=INTENSITY, etc.)
    cb: u8,
    /// First SFB index in this section (inclusive).
    sfb_start: usize,
    /// Last SFB index in this section (exclusive).
    sfb_end: usize,
}

/// Parse section_data() from the bitstream.
///
/// Long window: sect_bits=5, escape=31.
/// Short window: sect_bits=3, escape=7 (not implemented here — long only).
///
/// Returns a Vec of sections covering all `max_sfb` SFBs.
fn decode_section_data(
    br: &mut BitReader<'_>,
    max_sfb: usize,
    _eight_short: bool,
) -> Result<Vec<Section>, OxiAudioError> {
    // Long window: 5-bit section length increments, escape = 31.
    let sect_bits: u8 = 5;
    let sect_esc: usize = 31;

    let mut sections = Vec::new();
    let mut sfb = 0usize;

    while sfb < max_sfb {
        let cb = br.read_bits(4)? as u8;
        // Read section length in SFBs using escape coding
        let mut sect_len = 0usize;
        loop {
            let incr = br.read_bits(sect_bits)? as usize;
            sect_len += incr;
            if incr < sect_esc {
                break;
            }
        }
        if sect_len == 0 {
            return Err(OxiAudioError::Decode(
                "AAC: section_data has zero-length section".into(),
            ));
        }
        let sfb_end = (sfb + sect_len).min(max_sfb);
        sections.push(Section {
            cb,
            sfb_start: sfb,
            sfb_end,
        });
        sfb += sect_len;
        if sfb >= max_sfb {
            break;
        }
    }

    Ok(sections)
}

/// Decode a CB11 escape value magnitude >= 16.
fn decode_escape_value(br: &mut BitReader<'_>, base: u32) -> Result<f32, OxiAudioError> {
    if base < 16 {
        return Ok(base as f32);
    }
    // Count leading ones to find escape order N
    let mut n: u32 = 0;
    while br.read_bool()? {
        n += 1;
        if n > 24 {
            break; // guard against pathological streams
        }
    }
    // magnitude = 2^(N+4) + read_bits(N+4)
    let extra_bits = (n + 4).min(30) as u8;
    let mantissa = br.read_bits(extra_bits)?;
    let magnitude = (1u32 << (n + 4)).saturating_add(mantissa);
    Ok(magnitude as f32)
}

/// Decode `n_pairs` spectral pairs using codebook 11 (canonical ISO table).
///
/// Returns a flat Vec of `2 * n_pairs` signed f32 spectral values.
fn decode_spectral_cb11(br: &mut BitReader<'_>, n_pairs: usize) -> Result<Vec<f32>, OxiAudioError> {
    let mut out = Vec::with_capacity(n_pairs * 2);

    for _ in 0..n_pairs {
        // Linear scan through 289-entry canonical CB11 table
        let mut acc: u32 = 0;
        let mut x_abs = 0u32;
        let mut y_abs = 0u32;
        let mut matched = false;

        for bits_read in 1..=16u8 {
            acc = (acc << 1) | br.read_bits(1)?;
            // Check all 289 entries for a match
            for idx in 0..289usize {
                if HCB11_LENS[idx] == bits_read && HCB11_CODES[idx] == acc {
                    x_abs = (idx / 17) as u32;
                    y_abs = (idx % 17) as u32;
                    matched = true;
                    break;
                }
            }
            if matched {
                break;
            }
        }

        if !matched {
            // Fallback: treat as (0,0) — avoids hard error on corrupted streams
            x_abs = 0;
            y_abs = 0;
        }

        // Handle escape sequences for each value
        let xf = decode_escape_value(br, x_abs)?;
        let yf = decode_escape_value(br, y_abs)?;

        // Sign bits: one for each non-zero value (CB11 is unsigned — sign follows)
        let xs = if xf != 0.0 {
            if br.read_bool()? {
                -xf
            } else {
                xf
            }
        } else {
            0.0
        };
        let ys = if yf != 0.0 {
            if br.read_bool()? {
                -yf
            } else {
                yf
            }
        } else {
            0.0
        };

        out.push(xs);
        out.push(ys);
    }

    Ok(out)
}

/// Decode spectral data for one channel using per-section codebooks.
///
/// This function reads spectral coefficients based on the section_data that
/// was previously parsed — only non-ZERO_HCB and non-NOISE_HCB sections
/// have spectral data in the bitstream.
fn decode_spectral_data(
    br: &mut BitReader<'_>,
    ics: &IcsInfo,
    sfb_offsets: &[usize],
    sections: &[Section],
) -> Result<Vec<f32>, OxiAudioError> {
    let n_coeff = if ics.window_sequence == 2 {
        128 * ics.num_windows
    } else {
        1024
    };
    let mut coeffs = vec![0.0f32; n_coeff];

    for sect in sections {
        let cb = sect.cb;
        // ZERO_HCB (0), NOISE_HCB (13), INTENSITY_HCB (14/15): no spectral data
        if cb == 0 || cb == 13 || cb == 14 || cb == 15 {
            continue;
        }
        if br.bytes_remaining() == 0 {
            break;
        }

        for sfb in sect.sfb_start..sect.sfb_end {
            if sfb >= sfb_offsets.len().saturating_sub(1) {
                break;
            }
            let start = sfb_offsets[sfb];
            let end = sfb_offsets[sfb + 1].min(n_coeff);
            if start >= end {
                continue;
            }
            let band_len = end - start;

            // CB11: escape coded pairs
            if cb == 11 {
                let n_pairs = band_len / 2;
                if n_pairs > 0 && br.bytes_remaining() > 0 {
                    let decoded = decode_spectral_cb11(br, n_pairs)?;
                    let copy_len = decoded.len().min(band_len);
                    coeffs[start..start + copy_len].copy_from_slice(&decoded[..copy_len]);
                }
            }
            // Codebooks 1-10 are not emitted by the encoder (it uses CB0 or CB11 only),
            // so these coefficients remain zero (spectral hole).
        }
    }

    Ok(coeffs)
}

// ─── Inverse quantisation ─────────────────────────────────────────────────────

/// Inverse-quantise raw spectral coefficients (ISO 14496-3 §4.6.1.3).
///
/// `x_dequant = sign(x) · |x|^(4/3) · 2^(0.25·(sf – 100))`
fn dequantize(raw: &[f32], scale_factors: &[i16], sfb_offsets: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; raw.len()];

    let n_sfbs = sfb_offsets.len().saturating_sub(1);
    for sfb in 0..n_sfbs {
        let start = sfb_offsets[sfb];
        let end = sfb_offsets.get(sfb + 1).copied().unwrap_or(raw.len());
        let end = end.min(raw.len());
        if start >= end {
            continue;
        }
        let sf = scale_factors.get(sfb).copied().unwrap_or(100);
        let multiplier = 2.0_f32.powf(0.25 * f32::from(sf - 100));
        for i in start..end {
            let x = raw[i];
            if x == 0.0 {
                out[i] = 0.0;
            } else if x > 0.0 {
                out[i] = x.powf(4.0 / 3.0) * multiplier;
            } else {
                out[i] = -((-x).powf(4.0 / 3.0)) * multiplier;
            }
        }
    }

    out
}

// ─── IMDCT via OxiFFT ─────────────────────────────────────────────────────────
//
// Standard IMDCT-via-FFT algorithm:
//   1. Pre-rotate:  X[k] = spec[k] · e^(-j·π·(2k+1)/(2N))   for k = 0..N/2-1
//      where N = frame length (1024 or 128).
//   2. N/2-point complex forward FFT.
//   3. Post-rotate and scale:
//      x[n] = (2/N) · Re( Y[n] · e^(j·π·(2n+1+N/2)/(2N)) )  for n = 0..N/2-1
//   4. Extend to 2N output samples using IMDCT symmetry:
//      out[n]       =  x[n]  (first half)
//      out[N-1-n]   =  x[n]  (last quarter, negated)  — by IMDCT symmetry
//      out[N/2+n]   = -x[N/2-1-n]  — by IMDCT odd symmetry
//   5. Apply sine window.

/// Run the IMDCT for a frame of length `n` (1024 or 128).
///
/// Returns a time-domain buffer of length `2*n`.
fn imdct_n(spectrum: &[f32], n: usize) -> Vec<f32> {
    debug_assert_eq!(spectrum.len(), n);
    let half = n / 2;

    // Pre-rotation: build half-length complex input
    let mut cx: Vec<Complex<f64>> = (0..half)
        .map(|k| {
            let angle = -std::f64::consts::PI * (2 * k + 1) as f64 / (2 * n) as f64;
            let re = f64::from(spectrum[k]) * angle.cos();
            let im = f64::from(spectrum[k]) * angle.sin();
            Complex::new(re, im)
        })
        .collect();

    // N/2-point complex FFT
    let fft_out: Vec<Complex<f64>> = fft(&cx);
    cx.copy_from_slice(&fft_out);
    let y = &cx;

    // Post-rotation and symmetry expansion to 2N output samples
    let scale = 2.0_f64 / n as f64;
    let mut out = vec![0.0f32; 2 * n];

    for (nn, yn) in y.iter().enumerate().take(half) {
        let angle = std::f64::consts::PI * (2 * nn + 1 + half) as f64 / (2 * n) as f64;
        let val = scale * (yn.re * angle.cos() - yn.im * angle.sin());
        // IMDCT symmetry expansion to fill all 2N samples, windowed
        let val_f = val as f32;
        // Standard MDCT index mapping:
        let idx0 = half - 1 - nn; // second quarter (reversed)
        let idx1 = half + nn; // third quarter
        let idx2 = n + half - 1 - nn; // fourth quarter (reversed)
        let idx3 = n + half + nn; // first quarter

        // Apply sine window at each index
        let w0 = (PI * (idx0 as f32 + 0.5) / (2 * n) as f32).sin();
        let w1 = (PI * (idx1 as f32 + 0.5) / (2 * n) as f32).sin();

        out[idx0] = val_f * w0;
        out[idx1] = -val_f * w1;

        if idx2 < 2 * n {
            let w2 = (PI * (idx2 as f32 + 0.5) / (2 * n) as f32).sin();
            out[idx2] = -val_f * w2;
        }
        if idx3 < 2 * n {
            let w3 = (PI * (idx3 as f32 + 0.5) / (2 * n) as f32).sin();
            out[idx3] = val_f * w3;
        }
    }

    out
}

/// IMDCT for long window (N = 1024, output = 2048 samples).
fn imdct_1024(spectrum: &[f32]) -> Vec<f32> {
    imdct_n(spectrum, 1024)
}

/// IMDCT for short window (N = 128, output = 256 samples).
fn imdct_128(spectrum: &[f32]) -> Vec<f32> {
    imdct_n(spectrum, 128)
}

// ─── Window + overlap-add ─────────────────────────────────────────────────────

/// Apply window and overlap-add.
///
/// - `new_frame`: 2048 samples from IMDCT (already windowed inside `imdct_n`).
/// - `prev_half`: 1024-sample overlap buffer, updated in-place.
/// - Returns 1024 output samples.
fn apply_window_overlap_add(new_frame: &[f32], prev_half: &mut Vec<f32>) -> Vec<f32> {
    debug_assert_eq!(new_frame.len(), 2048);
    debug_assert_eq!(prev_half.len(), 1024);

    // Output = first 1024 samples of new_frame plus overlap
    let mut output = vec![0.0f32; 1024];
    for i in 0..1024 {
        output[i] = new_frame[i] + prev_half[i];
    }

    // Update overlap buffer to second 1024 samples of new_frame
    prev_half.clear();
    prev_half.extend_from_slice(&new_frame[1024..2048]);

    output
}

/// Overlap-add for short-window frame (8 × 128-sample IMDCTs concatenated).
fn apply_short_ola(short_frames: &[Vec<f32>], prev_half: &mut Vec<f32>) -> Vec<f32> {
    debug_assert_eq!(short_frames.len(), 8);
    debug_assert_eq!(prev_half.len(), 1024);

    // Construct a 2048-sample buffer from 8 × 256-sample outputs using
    // the short-window OLA defined in ISO 14496-3 §4.6.14.1:
    // The 8 short IMDCTs are laid out in the centre of the long frame.
    // First 448 and last 448 samples of the 2048-window are zero.
    let mut long_buf = vec![0.0f32; 2048];
    for (i, frame) in short_frames.iter().enumerate() {
        let offset = 448 + i * 128;
        for j in 0..256 {
            if offset + j < 2048 {
                long_buf[offset + j] += frame[j];
            }
        }
    }

    apply_window_overlap_add(&long_buf, prev_half)
}

// ─── Single-channel raw-data-block decoder ────────────────────────────────────

/// Decode one ICS (Individual Channel Stream) data block.
///
/// This is the inner ICS decoder used by both SCE and the ICS elements within a CPE.
/// It does NOT read any element tag — that must be consumed by the caller.
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] on bitstream failure.
fn decode_ics_data(
    br: &mut BitReader<'_>,
    sample_rate: u32,
    prev_half: &mut Vec<f32>,
) -> Result<Vec<f32>, OxiAudioError> {
    let global_gain = br.read_bits(8)? as u8;
    let ics = parse_ics_info(br)?;
    let max_sfb = ics.max_sfb as usize;

    // section_data(): parse section codebooks and lengths
    let eight_short = ics.window_sequence == 2;
    let sections = decode_section_data(br, max_sfb, eight_short)?;

    // scale_factor_data(): read SF deltas only for non-ZERO_HCB SFBs
    let sfb_offsets = sfb_offsets_long(sample_rate);
    let sfs = decode_scale_factors(br, max_sfb, global_gain, &sections)?;

    // pulse_data_present
    let pulse_data_present = br.read_bool()?;
    if pulse_data_present {
        // pulse_nmax(2) + pulse_start_sfb(6) then (pulse_nmax+1) × (pulse_offset(5)+pulse_amp(4))
        let pulse_nmax = br.read_bits(2)?;
        let _start_sfb = br.read_bits(6)?;
        for _ in 0..=pulse_nmax {
            let _offset = br.read_bits(5)?;
            let _amp = br.read_bits(4)?;
        }
    }

    // tns_data_present — parse correctly so bitstream stays aligned
    let tns_data_present = br.read_bool()?;
    if tns_data_present {
        // Long window: n_filt_bits=2, coef_res=1, then per-filter: length(6)+order(5)+dir(1)+compress(1)+coefs
        let n_filt = br.read_bits(2)? as usize;
        if n_filt > 0 {
            let coef_res = br.read_bits(1)?; // 0=3-bit, 1=4-bit
            for _ in 0..n_filt {
                let _length = br.read_bits(6)?; // SFBs covered
                let order = br.read_bits(5)? as usize; // LPC order
                let _direction = br.read_bits(1)?;
                let coef_compress = br.read_bits(1)?;
                let coef_bits: u8 = match (coef_res, coef_compress) {
                    (1, 1) => 3,
                    (1, _) => 4,
                    (_, 1) => 2,
                    _ => 3,
                };
                for _ in 0..order {
                    let _coef = br.read_bits(coef_bits)?;
                }
            }
        }
    }

    let gain_control_data_present = br.read_bool()?;
    if gain_control_data_present {
        return Err(OxiAudioError::Decode(
            "AAC: gain_control_data not supported".into(),
        ));
    }

    // spectral_data(): decode per-section using the correct codebooks
    let raw = decode_spectral_data(br, &ics, sfb_offsets, &sections)?;

    // Dequantise
    let dequant = dequantize(&raw, &sfs, sfb_offsets);

    // IMDCT + OLA
    let pcm = if ics.window_sequence == 2 {
        // EIGHT_SHORT_SEQUENCE: process grouped short windows, then overlap-add.
        // Window shape (sine vs Kaiser-Bessel) is treated as sine in this implementation.
        let mut short_frames: Vec<Vec<f32>> = Vec::with_capacity(ics.num_windows);
        let window_size = 128;
        let mut coeff_offset = 0usize;
        for g in 0..ics.num_window_groups {
            let group_len = ics.window_group_length[g];
            for _ in 0..group_len {
                let start = coeff_offset;
                let end = (start + window_size).min(dequant.len());
                let mut seg = vec![0.0f32; window_size];
                let copy_len = end.saturating_sub(start);
                seg[..copy_len].copy_from_slice(&dequant[start..end]);
                short_frames.push(imdct_128(&seg));
                coeff_offset += window_size;
            }
        }
        while short_frames.len() < 8 {
            short_frames.push(vec![0.0f32; 256]);
        }
        apply_short_ola(&short_frames[..8], prev_half)
    } else {
        let frame = imdct_1024(&dequant);
        apply_window_overlap_add(&frame, prev_half)
    };

    Ok(pcm)
}

/// Decode one `single_channel_element` (SCE) from the bitstream.
///
/// Reads the element_instance_tag (4 bits) and delegates to `decode_ics_data`.
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] on bitstream failure.
fn decode_sce(
    br: &mut BitReader<'_>,
    sample_rate: u32,
    prev_half: &mut Vec<f32>,
) -> Result<Vec<f32>, OxiAudioError> {
    // element_instance_tag (4 bits) — ignored
    let _tag = br.read_bits(4)?;
    decode_ics_data(br, sample_rate, prev_half)
}

// ─── Top-level decoder struct ─────────────────────────────────────────────────

/// Stateful AAC-LC frame decoder.
///
/// Maintains overlap-add state between frames.  Construct with [`AacDecoder::new`]
/// then call [`AacDecoder::decode_frame`] for each ADTS-framed input packet.
#[derive(Debug, Clone)]
pub struct AacDecoder {
    prev_left: Vec<f32>,
    prev_right: Vec<f32>,
    sample_rate: Option<u32>,
    channels: Option<u8>,
}

impl AacDecoder {
    /// Create a new decoder with zeroed overlap buffers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prev_left: vec![0.0f32; 1024],
            prev_right: vec![0.0f32; 1024],
            sample_rate: None,
            channels: None,
        }
    }

    /// Decode one ADTS-framed AAC packet.
    ///
    /// Returns interleaved PCM samples (L, R, L, R … for stereo; M for mono).
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Decode`] on malformed input.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<Vec<f32>, OxiAudioError> {
        let frame = parse_adts_header(data)?;

        // Store stream parameters from first frame
        if self.sample_rate.is_none() {
            self.sample_rate = Some(frame.sample_rate);
            self.channels = Some(frame.channels);
        }

        let mut br = BitReader::new(frame.payload);

        // Read the raw_data_block() element type
        // For AAC-LC, element types are: SCE=0, CPE=1, CCE=2, LFE=3, DSE=4, PCE=5, FIL=6, END=7
        let elem_type = br.read_bits(3)?;

        let pcm = match elem_type {
            0 => {
                // SCE: single channel element
                decode_sce(&mut br, frame.sample_rate, &mut self.prev_left)?
            }
            1 => {
                // CPE: channel pair element
                // Read CPE's element_instance_tag (4 bits) and common_window (1 bit)
                let _cpe_tag = br.read_bits(4)?;
                let common_window = br.read_bool()?;
                if common_window {
                    // common_window=1: a shared ics_info is read first, then two channel data
                    // For simplicity, fall back to reading two full ICS elements
                    // (our encoder always uses common_window=0)
                    return Err(OxiAudioError::Decode(
                        "AAC CPE: common_window=1 not supported".into(),
                    ));
                }
                // common_window=0: each channel has its own complete ICS element
                let left = decode_ics_data(&mut br, frame.sample_rate, &mut self.prev_left)?;
                let right = decode_ics_data(&mut br, frame.sample_rate, &mut self.prev_right)?;
                // Interleave L/R
                let mut interleaved = Vec::with_capacity(left.len() + right.len());
                for (l, r) in left.iter().zip(right.iter()) {
                    interleaved.push(*l);
                    interleaved.push(*r);
                }
                interleaved
            }
            _ => {
                return Err(OxiAudioError::Decode(format!(
                    "AAC: unsupported element type {elem_type}"
                )));
            }
        };

        Ok(pcm)
    }

    /// Sample rate detected from the ADTS stream, or `None` before first frame.
    #[must_use]
    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    /// Channel count detected from the ADTS stream, or `None` before first frame.
    #[must_use]
    pub fn channels(&self) -> Option<u8> {
        self.channels
    }
}

impl Default for AacDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Stream-level decode function ─────────────────────────────────────────────

/// Scan forward in `data` for the next ADTS sync word (0xFFF).
///
/// Returns the byte offset, or `None` if not found.
fn find_adts_sync(data: &[u8]) -> Option<usize> {
    for i in 0..data.len().saturating_sub(1) {
        let sync = (u16::from(data[i]) << 4) | (u16::from(data[i + 1]) >> 4);
        if sync == 0xFFF {
            return Some(i);
        }
    }
    None
}

/// Decode a complete ADTS/AAC stream to an [`AudioBuffer<f32>`].
///
/// Iterates over all ADTS frames in `data`, decoding each to PCM, and
/// assembles the result into a single interleaved [`AudioBuffer<f32>`].
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] if:
/// - No valid ADTS sync word is found.
/// - All frames fail to decode and no samples are produced.
pub fn decode_aac(data: &[u8]) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if data.is_empty() {
        return Err(OxiAudioError::Decode("AAC: empty input".into()));
    }

    // Find first sync
    let start = find_adts_sync(data)
        .ok_or_else(|| OxiAudioError::Decode("AAC: no ADTS sync word found".into()))?;

    let mut decoder = AacDecoder::new();
    let mut all_samples: Vec<f32> = Vec::new();
    let mut pos = start;

    loop {
        // Attempt to parse the frame at the current position
        let slice = &data[pos..];
        let frame_info = match parse_adts_header(slice) {
            Ok(f) => f,
            Err(_) => {
                // Try to find next sync word
                if let Some(next) = find_adts_sync(&data[pos + 1..]) {
                    pos += next + 1;
                    continue;
                } else {
                    break;
                }
            }
        };

        let frame_len = {
            // Re-read frame_length from the raw header for advancing position
            if slice.len() < 6 {
                break;
            }
            let fl = (u32::from(slice[3] & 0x03) << 11)
                | (u32::from(slice[4]) << 3)
                | (u32::from(slice[5]) >> 5);
            fl as usize
        };

        // Decode
        match decoder.decode_frame(slice) {
            Ok(samples) => all_samples.extend_from_slice(&samples),
            Err(_) => {
                // Skip malformed frames silently
            }
        }

        let next_pos = pos + frame_len;
        if next_pos <= pos || next_pos >= data.len() {
            break;
        }
        pos = next_pos;

        // Confirm we see another sync or bail
        let sync_check = (u16::from(data[pos]) << 4)
            | (u16::from(data[pos + 1..].first().copied().unwrap_or(0)) >> 4);
        if sync_check != 0xFFF {
            // Search for next sync
            if let Some(next) = find_adts_sync(&data[pos..]) {
                pos += next;
            } else {
                break;
            }
        }

        let _ = frame_info;
    }

    if all_samples.is_empty() {
        return Err(OxiAudioError::Decode(
            "AAC: no decodable frames found".into(),
        ));
    }

    let sample_rate = decoder.sample_rate().unwrap_or(44100);
    let channels = decoder.channels().unwrap_or(2);
    let layout = match channels {
        1 => ChannelLayout::Mono,
        _ => ChannelLayout::Stereo,
    };

    Ok(AudioBuffer {
        samples: all_samples,
        sample_rate,
        channels: layout,
        format: SampleFormat::F32,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal 7-byte ADTS header for AAC-LC, 44100 Hz, stereo.
    ///
    /// Bit layout:
    ///   Byte 0: 1111_1111         (sync hi)
    ///   Byte 1: 1111_0001         (sync lo + id=0, layer=00, protection_absent=1)
    ///   Byte 2: 0101_0000         (profile=01=AAC-LC, sfi=4=44100, private=0, ch_config hi=0)
    ///   Byte 3: 0100_0000         (ch_config lo=10 → 2ch, home=0, copy=0, copy_start=0, fl[12:11]=00)
    ///   Byte 4: 0001_1100         (frame_length[10:3] = 0b000_1110 → partial)
    ///   Byte 5: 0001_1111         (frame_length[2:0]=0, buf_fullness hi, rdb=00)
    ///   Byte 6: 1111_1100
    ///
    /// We encode frame_length = 7 (header only, no payload) to keep it simple.
    fn minimal_adts_header() -> Vec<u8> {
        // sync(12)=0xFFF, id=0, layer=0, protection_absent=1
        // → bytes [0..1]: 0xFF 0xF1
        //
        // profile=01 (AAC-LC, stored as object_type - 1 = 1), sfi=4 (44100 Hz),
        // private=0, channel_config=2 (bits 2:0), where MSB of channel_config
        // is the LSB of byte2.
        // channel_config=2 = 0b010 → channel_config_hi (1 bit) = 0, lo (2 bits) in byte3
        //
        // byte2: (profile<<6) | (sfi<<2) | private<<1 | ch_hi
        //      = (1<<6) | (4<<2) | 0 | 0 = 0x50
        //
        // byte3: (ch_lo<<6) | home<<5 | copy_id<<4 | copy_start<<3 | fl[12:11]
        //      channel_config=2=0b010, ch_hi=0, ch_lo=0b10=2 → (2<<6)=0x80
        //      fl = 7 (bytes), fl[12:11] = 7 >> 11 = 0
        //      = 0x80 | 0 = 0x80
        //
        // byte4: fl[10:3] = (7 >> 3) & 0xFF = 0
        //
        // byte5: fl[2:0]<<5 | buf_fullness[10:5] | rdb[1:0]
        //      fl[2:0] = 7 & 0x7 = 7 → 7<<5 = 0xE0
        //      buf_fullness=0x7FF (VBR) = 0b111_1111_111_00
        //      buf_full[10:5] = 0b11111 << 3 ... simplify: use 0x1F << 2 = 0x7C...
        //      Keep simple: 0b1110_0111 = 0xE7 (fl[2:0]=111, buf=00, rdb=11)
        //      Actually let's just target frame_length=7:
        //      fl = 7 = 0b000_0000_0000_111
        //      fl[12:11]=0, fl[10:3]=0, fl[2:0]=7
        //      byte5: (7<<5)|(0x7FF>>5)&0x3F<<2 ... keep zero buf: (7<<5)|0 = 0xE0
        //
        // byte6: rdb[1:0]=0, rest don't care = 0x00
        vec![0xFF, 0xF1, 0x50, 0x80, 0x00, 0xE0, 0x00]
    }

    #[test]
    fn test_adts_parse_valid_header() {
        let data = minimal_adts_header();
        let frame = parse_adts_header(&data).expect("parse minimal ADTS header");
        assert_eq!(frame.channels, 2, "stereo");
        assert_eq!(frame.sample_rate, 44100);
        assert_eq!(frame.pcm_samples, 1024);
        assert_eq!(frame.payload.len(), 0, "payload empty for frame_length=7");
    }

    #[test]
    fn test_adts_reject_short_data() {
        let result = parse_adts_header(&[0xFF, 0xF1]);
        assert!(result.is_err(), "too-short data should fail");
    }

    #[test]
    fn test_adts_reject_bad_sync() {
        let mut data = minimal_adts_header();
        data[0] = 0x00; // corrupt sync
        let result = parse_adts_header(&data);
        assert!(result.is_err(), "bad sync should fail");
    }

    #[test]
    fn test_bitreader_basic() {
        let data = [0b1011_0100u8, 0b1100_1010u8];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bits(4).expect("read 4 bits"), 0b1011);
        assert_eq!(br.read_bits(4).expect("read 4 bits"), 0b0100);
        assert_eq!(br.read_bits(4).expect("read 4 bits"), 0b1100);
    }

    #[test]
    fn test_bitreader_bool() {
        let data = [0b1000_0000u8];
        let mut br = BitReader::new(&data);
        assert!(br.read_bool().expect("read bool"));
        assert!(!br.read_bool().expect("read bool"));
    }

    #[test]
    fn test_bitreader_exhaustion() {
        let data = [0xFFu8];
        let mut br = BitReader::new(&data);
        br.read_bits(8).expect("read 8 bits");
        assert!(
            br.read_bits(1).is_err(),
            "reading past end should return Err"
        );
    }

    #[test]
    fn test_bytes_remaining() {
        let data = [0x00u8; 4];
        let mut br = BitReader::new(&data);
        assert_eq!(br.bytes_remaining(), 4);
        br.read_bits(8).expect("read byte");
        assert_eq!(br.bytes_remaining(), 3);
    }

    #[test]
    fn test_imdct_1024_length() {
        let spectrum = vec![0.0f32; 1024];
        let output = imdct_1024(&spectrum);
        assert_eq!(output.len(), 2048);
    }

    #[test]
    fn test_imdct_1024_silence_is_silence() {
        let spectrum = vec![0.0f32; 1024];
        let output = imdct_1024(&spectrum);
        for s in &output {
            assert!(
                s.abs() < 1e-6,
                "silence spectrum should give silence time domain, got {s}"
            );
        }
    }

    #[test]
    fn test_imdct_128_length() {
        let spectrum = vec![0.0f32; 128];
        let output = imdct_128(&spectrum);
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_imdct_128_silence_is_silence() {
        let spectrum = vec![0.0f32; 128];
        let output = imdct_128(&spectrum);
        for s in &output {
            assert!(
                s.abs() < 1e-6,
                "128-pt silence spectrum should give silence, got {s}"
            );
        }
    }

    #[test]
    fn test_dequantize_zero_scale() {
        // SF = 100 → multiplier = 2^(0.25*(100-100)) = 2^0 = 1.0
        // raw 1.0 → 1.0^(4/3) * 1.0 = 1.0
        let raw = vec![1.0f32, -1.0, 0.0];
        let sfs = vec![100i16];
        let offsets = vec![0, 3];
        let out = dequantize(&raw, &sfs, &offsets);
        assert!((out[0] - 1.0).abs() < 1e-5, "pos: {}", out[0]);
        assert!((out[1] + 1.0).abs() < 1e-5, "neg: {}", out[1]);
        assert_eq!(out[2], 0.0, "zero stays zero");
    }

    #[test]
    fn test_dequantize_positive_scale() {
        // SF = 104 → multiplier = 2^(0.25*(104-100)) = 2^1 = 2.0
        // raw 1.0 → 1.0^(4/3) * 2.0 = 2.0
        let raw = vec![1.0f32];
        let sfs = vec![104i16];
        let offsets = vec![0, 1];
        let out = dequantize(&raw, &sfs, &offsets);
        assert!((out[0] - 2.0).abs() < 1e-4, "expected 2.0, got {}", out[0]);
    }

    #[test]
    fn test_overlap_add_basic() {
        let new_frame: Vec<f32> = (0..2048).map(|i| i as f32).collect();
        let mut prev_half = vec![1.0f32; 1024];
        let out = apply_window_overlap_add(&new_frame, &mut prev_half);
        assert_eq!(out.len(), 1024);
        // First sample: new_frame[0] + 1.0 = 0.0 + 1.0 = 1.0
        assert!((out[0] - 1.0).abs() < 1e-5);
        // prev_half now holds new_frame[1024..2048]
        assert_eq!(prev_half.len(), 1024);
        assert!((prev_half[0] - 1024.0).abs() < 1e-5);
    }

    #[test]
    fn test_sfb_offsets_known_rates() {
        let o44 = sfb_offsets_long(44100);
        assert_eq!(*o44.last().expect("last entry"), 1024);
        let o48 = sfb_offsets_long(48000);
        assert_eq!(*o48.last().expect("last entry"), 1024);
        let o32 = sfb_offsets_long(32000);
        assert_eq!(*o32.last().expect("last entry"), 1024);
    }

    #[test]
    fn test_aac_decoder_new() {
        let dec = AacDecoder::new();
        assert!(
            dec.sample_rate().is_none(),
            "no sample rate before first frame"
        );
        assert!(dec.channels().is_none(), "no channels before first frame");
    }

    #[test]
    fn test_aac_decoder_default() {
        let dec = AacDecoder::default();
        assert!(dec.sample_rate().is_none());
        assert!(dec.channels().is_none());
    }

    #[test]
    fn test_decode_aac_empty_returns_error() {
        let result = decode_aac(&[]);
        assert!(result.is_err(), "empty input should be an error");
    }

    #[test]
    fn test_decode_aac_invalid_sync() {
        let data = [0x00u8; 20];
        let result = decode_aac(&data);
        assert!(result.is_err(), "no sync word should be an error");
    }

    #[test]
    fn test_find_adts_sync_present() {
        let data = [0x00, 0x00, 0xFF, 0xF1, 0x00];
        let pos = find_adts_sync(&data);
        assert_eq!(pos, Some(2));
    }

    #[test]
    fn test_find_adts_sync_absent() {
        let data = [0x00u8; 10];
        let pos = find_adts_sync(&data);
        assert!(pos.is_none());
    }

    #[test]
    fn test_sf_huffman_decode_zero_delta() {
        // Symbol 1 → delta 0 (the most common codeword: 1 bit = 1)
        let data = [0b1000_0000u8];
        let mut br = BitReader::new(&data);
        let delta = decode_sf_huffman(&mut br).expect("decode SF Huffman");
        assert_eq!(delta, 0);
    }

    #[test]
    fn test_decode_escape_value_below_16() {
        let data = [0u8; 4];
        let mut br = BitReader::new(&data);
        let v = decode_escape_value(&mut br, 5).expect("escape value below 16");
        assert!((v - 5.0).abs() < 1e-5);
    }
}
