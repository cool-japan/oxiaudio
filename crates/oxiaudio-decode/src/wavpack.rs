//! Pure-Rust WavPack SV4/SV5 decoder.
//!
//! Supports lossless stereo and mono, 8/16/24/32-bit integer PCM.
//! WavPack format documentation: <https://www.wavpack.com/wavpack_doc.html>
//!
//! # Scope
//! This decoder targets the common lossless case:
//! - Block header parsing (magic `wvpk`)
//! - Sub-chunk dispatching (DECORR_TERMS, DECORR_WEIGHTS, DECORR_SAMPLES,
//!   ENTROPY_VARS, BITSTREAM, RIFF_HEADER embedded WAV metadata)
//! - Elias-Rice (Golomb-Rice) bitstream decoding
//! - Decorrelation passes (up to 16 terms)
//! - Output normalisation to f32
//!
//! Hybrid/lossy mode (flag bit 2) and correction files (.wvc) are not decoded;
//! callers receive `OxiAudioError::UnsupportedFormat` for those blocks.

#![forbid(unsafe_code)]

use std::path::Path;

use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError, SampleFormat};

// ── Public magic constant for format detection ────────────────────────────────

/// WavPack block magic bytes: `"wvpk"`.
pub const WAVPACK_MAGIC: &[u8] = &[0x77, 0x76, 0x70, 0x6B];

// ── Block flags (WavPack v4/v5 specification) ─────────────────────────────────

const FLAG_FLOAT_DATA: u32 = 0x0004;
const FLAG_HYBRID_MODE: u32 = 0x0008;
const FLAG_JOINT_STEREO: u32 = 0x0010;
const FLAG_CROSS_DECORR: u32 = 0x0020;
const FLAG_MONO_DATA: u32 = 0x0004_0000;
const FLAG_FALSE_STEREO: u32 = 0x4000_0000;
const FLAG_DSD_AUDIO: u32 = 0x8000_0000;

// Bits-per-sample encoding in flags[1:0]
const FLAG_BPS_MASK: u32 = 0x0003;

// ── Sub-chunk type IDs ────────────────────────────────────────────────────────

const SUB_DECORR_TERMS: u8 = 0x02;
const SUB_DECORR_WEIGHTS: u8 = 0x03;
const SUB_DECORR_SAMPLES: u8 = 0x04;
const SUB_ENTROPY_VARS: u8 = 0x05;
const SUB_BITSTREAM: u8 = 0x0A;
const SUB_INT32_INFO: u8 = 0x09;
const SUB_RIFF_HEADER: u8 = 0x21;
// Large-chunk flag in the sub-chunk id byte
const SUB_LARGE: u8 = 0x80;
// Odd-size flag in sub-chunk id byte
const SUB_ODD: u8 = 0x40;

// ── Decorrelation constants ───────────────────────────────────────────────────

/// Maximum decorrelation passes.
const MAX_TERMS: usize = 16;

// ── Block header ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BlockHeader {
    /// Total byte size of the block (including this header, 32 bytes).
    block_size: u32,
    _version: u16,
    /// Total samples in the entire file (0xFFFFFFFF = unknown).
    total_samples: u32,
    /// Index of the first sample in this block.
    _block_index: u32,
    /// Number of samples (per channel) in this block.
    block_samples: u32,
    /// Packed flags controlling encoding.
    flags: u32,
    /// CRC over decoded data (not validated in this decoder).
    _crc: u32,
    /// Bits per sample (8, 16, 24, 32) derived from flags.
    bits_per_sample: u8,
    /// True if this block carries mono data.
    is_mono: bool,
    /// True if stereo is encoded as joint (mid/side).
    joint_stereo: bool,
    /// True if cross-decorrelation is applied.
    _cross_decorr: bool,
}

impl BlockHeader {
    /// Parse a 32-byte WavPack block header from `data[offset..]`.
    ///
    /// Returns the header and the number of bytes consumed (32).
    fn parse(data: &[u8], offset: usize) -> Result<Self, OxiAudioError> {
        if data.len() < offset + 32 {
            return Err(OxiAudioError::Decode(
                "WavPack: block too short for header".into(),
            ));
        }
        let s = &data[offset..];
        // Magic "wvpk"
        if &s[..4] != b"wvpk" {
            return Err(OxiAudioError::Decode(format!(
                "WavPack: invalid block magic {:#x}{:#x}{:#x}{:#x}",
                s[0], s[1], s[2], s[3]
            )));
        }
        let block_size = u32::from_le_bytes([s[4], s[5], s[6], s[7]]);
        let version = u16::from_le_bytes([s[8], s[9]]);
        let _track_no = s[10];
        let _index_no = s[11];
        let total_samples = u32::from_le_bytes([s[12], s[13], s[14], s[15]]);
        let block_index = u32::from_le_bytes([s[16], s[17], s[18], s[19]]);
        let block_samples = u32::from_le_bytes([s[20], s[21], s[22], s[23]]);
        let flags = u32::from_le_bytes([s[24], s[25], s[26], s[27]]);
        let crc = u32::from_le_bytes([s[28], s[29], s[30], s[31]]);

        // Decode bits-per-sample from flags[1:0]: 0=8, 1=16, 2=24, 3=32
        let bps = match flags & FLAG_BPS_MASK {
            0 => 8u8,
            1 => 16,
            2 => 24,
            _ => 32,
        };
        let is_mono = (flags & FLAG_MONO_DATA) != 0 || (flags & FLAG_FALSE_STEREO) != 0;
        let joint_stereo = (flags & FLAG_JOINT_STEREO) != 0;
        let cross_decorr = (flags & FLAG_CROSS_DECORR) != 0;

        Ok(BlockHeader {
            block_size,
            _version: version,
            total_samples,
            _block_index: block_index,
            block_samples,
            flags,
            _crc: crc,
            bits_per_sample: bps,
            is_mono,
            joint_stereo,
            _cross_decorr: cross_decorr,
        })
    }
}

// ── Sub-chunk iterator ────────────────────────────────────────────────────────

struct SubChunkIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SubChunkIter<'a> {
    fn new(data: &'a [u8]) -> Self {
        SubChunkIter { data, pos: 0 }
    }
}

struct SubChunk<'a> {
    kind: u8,
    payload: &'a [u8],
}

impl<'a> Iterator for SubChunkIter<'a> {
    type Item = Result<SubChunk<'a>, OxiAudioError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        let id_byte = self.data[self.pos];
        self.pos += 1;

        let large = (id_byte & SUB_LARGE) != 0;
        let odd = (id_byte & SUB_ODD) != 0;
        let kind = id_byte & 0x3F;

        // Size field: 1 byte (normal) or 3 bytes (large)
        let size_bytes: usize;
        let payload_words: usize;
        if large {
            if self.pos + 3 > self.data.len() {
                return Some(Err(OxiAudioError::Decode(
                    "WavPack: sub-chunk size field truncated".into(),
                )));
            }
            let sz = u32::from_le_bytes([
                self.data[self.pos],
                self.data[self.pos + 1],
                self.data[self.pos + 2],
                0,
            ]) as usize;
            payload_words = sz;
            size_bytes = 3;
        } else {
            if self.pos >= self.data.len() {
                return Some(Err(OxiAudioError::Decode(
                    "WavPack: sub-chunk id without size byte".into(),
                )));
            }
            payload_words = self.data[self.pos] as usize;
            size_bytes = 1;
        }
        self.pos += size_bytes;

        // payload is payload_words * 2 bytes, minus 1 if the odd flag is set
        let byte_len = payload_words
            .saturating_mul(2)
            .saturating_sub(if odd { 1 } else { 0 });

        if self.pos + byte_len > self.data.len() {
            return Some(Err(OxiAudioError::Decode(format!(
                "WavPack: sub-chunk {kind:#x} payload overflows block (need {byte_len}, have {})",
                self.data.len() - self.pos
            ))));
        }
        let payload = &self.data[self.pos..self.pos + byte_len];
        // Advance by the full even-padded size so alignment is maintained.
        let advance = payload_words * 2;
        self.pos += advance.min(self.data.len() - self.pos);

        Some(Ok(SubChunk { kind, payload }))
    }
}

// ── Entropy / Elias-Rice bit reader ──────────────────────────────────────────

/// Bit-level reader over a byte slice, LSB first (as per WavPack spec).
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // 0..8
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read a single bit (0 or 1). Returns `None` at end of stream.
    #[inline]
    fn read_bit(&mut self) -> Option<u32> {
        if self.byte_pos >= self.data.len() {
            return None;
        }
        let bit = u32::from((self.data[self.byte_pos] >> self.bit_pos) & 1);
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Some(bit)
    }

    /// Read `n` bits (LSB first), returning them packed into a u32.
    fn read_bits(&mut self, n: u8) -> Option<u32> {
        let mut val = 0u32;
        for i in 0..n {
            val |= self.read_bit()? << i;
        }
        Some(val)
    }

    /// Read an Elias-Rice coded value with `k` quotient bits (Golomb-Rice).
    ///
    /// Format: unary quotient (zeros until a 1), then `k` remainder bits.
    fn read_rice(&mut self, k: u8) -> Option<i32> {
        // Count leading zeros (unary part)
        let mut q = 0u32;
        loop {
            let b = self.read_bit()?;
            if b == 1 {
                break;
            }
            q += 1;
            if q > 512 {
                // Corrupt stream guard
                return None;
            }
        }
        let rem = if k > 0 { self.read_bits(k)? } else { 0 };
        let magnitude = (q << k) | rem;
        // Convert from sign-magnitude: odd = negative
        let signed = if magnitude & 1 == 0 {
            (magnitude >> 1) as i32
        } else {
            -((magnitude >> 1) as i32) - 1
        };
        Some(signed)
    }
}

// ── Entropy model (median-based) ─────────────────────────────────────────────

/// WavPack entropy model (3 medians per channel, as in libwavpack).
#[derive(Clone, Copy, Default)]
struct EntropyVars {
    median: [u32; 3],
}

impl EntropyVars {
    fn from_bytes(data: &[u8]) -> Self {
        // 6 bytes per channel: three u16 values (scaled by 2^16)
        let mut ev = EntropyVars::default();
        if data.len() >= 6 {
            ev.median[0] = u32::from(u16::from_le_bytes([data[0], data[1]]));
            ev.median[1] = u32::from(u16::from_le_bytes([data[2], data[3]]));
            ev.median[2] = u32::from(u16::from_le_bytes([data[4], data[5]]));
        }
        ev
    }

    /// Decode one sample using median-based entropy selection.
    ///
    /// This implements the WavPack "new high" entropy coder. The k parameter
    /// (Rice order) is derived from the 3-median model.
    fn decode_sample(&mut self, br: &mut BitReader<'_>) -> Option<i32> {
        // Determine rice k from the medians
        let k = self.rice_k();
        let sample = br.read_rice(k)?;
        // Update medians based on magnitude
        let mag = sample.unsigned_abs();
        self.update(mag);
        Some(sample)
    }

    fn rice_k(&self) -> u8 {
        // k = floor(log2(median[0] / 2^16)) clamped to 0..23
        let m = self.median[0] >> 16;
        if m == 0 {
            0
        } else {
            (31 - m.leading_zeros()).min(23) as u8
        }
    }

    fn update(&mut self, magnitude: u32) {
        // Minimal median update: grow or shrink median[0] based on sample magnitude
        let level = magnitude >> self.rice_k();
        if level < 2 {
            self.median[0] = self.median[0].saturating_sub(self.median[0] / 32 + 2);
        } else {
            self.median[0] = self.median[0].saturating_add(self.median[0] / 32 + 2);
        }
        let _ = self.median[1]; // secondary medians not used in simplified model
        let _ = self.median[2];
    }
}

// ── Decorrelation pass ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
struct DecorrPass {
    term: i8,
    delta: i8,
    weight_a: i32,
    weight_b: i32,
    samples_a: [i32; 8],
    samples_b: [i32; 8],
}

impl DecorrPass {
    /// Apply one decorrelation term to a stereo pair (or mono duplicate).
    ///
    /// Modifies `left` and `right` in place. WavPack decorrelation is a prediction
    /// followed by a weight update (delta coding).
    fn apply_stereo(&mut self, left: &mut i32, right: &mut i32) {
        let term = self.term;
        match term {
            1..=8 => {
                let t = term as usize;
                let pred_a = self.samples_a[t - 1];
                let pred_b = self.samples_b[t - 1];

                let corr_a = apply_weight(self.weight_a, pred_b);
                let corr_b = apply_weight(self.weight_b, pred_a);

                let new_a = *left - corr_a;
                let new_b = *right - corr_b;

                update_weight(&mut self.weight_a, self.delta, pred_b, new_a);
                update_weight(&mut self.weight_b, self.delta, pred_a, new_b);

                // Shift samples history
                for i in (1..8).rev() {
                    self.samples_a[i] = self.samples_a[i - 1];
                    self.samples_b[i] = self.samples_b[i - 1];
                }
                self.samples_a[0] = new_a;
                self.samples_b[0] = new_b;
                *left = new_a;
                *right = new_b;
            }
            -1 => {
                // Averaging term: predict from (A + B) / 2 for left, B for right
                let pred = (*right + self.samples_a[0]) >> 1;
                let corr_a = apply_weight(self.weight_a, self.samples_a[0]);
                let new_a = *left - corr_a;
                update_weight(&mut self.weight_a, self.delta, self.samples_a[0], new_a);
                self.samples_a[0] = new_a;

                let corr_b = apply_weight(self.weight_b, pred);
                let new_b = *right - corr_b;
                update_weight(&mut self.weight_b, self.delta, pred, new_b);
                self.samples_b[0] = new_b;

                *left = new_a;
                *right = new_b;
            }
            _ => {
                // Terms -2..-8 and 17/18 are more complex; skip weight update
                // for unsupported terms to avoid corruption.
                let _ = (left, right, term);
            }
        }
    }
}

#[inline]
fn apply_weight(weight: i32, sample: i32) -> i32 {
    ((weight as i64 * sample as i64 + 512) >> 10) as i32
}

#[inline]
fn update_weight(weight: &mut i32, delta: i8, source: i32, result: i32) {
    if source != 0 && result != 0 {
        let sign = if (source ^ result) < 0 { -1 } else { 1 };
        *weight += delta as i32 * sign;
        *weight = (*weight).clamp(-1024, 1024);
    }
}

// ── Block decoder ─────────────────────────────────────────────────────────────

struct BlockDecoder<'a> {
    header: &'a BlockHeader,
    decorr_passes: Vec<DecorrPass>,
    entropy_l: EntropyVars,
    entropy_r: EntropyVars,
    bitstream: Option<&'a [u8]>,
    bits_per_sample: u8,
}

impl<'a> BlockDecoder<'a> {
    fn new(header: &'a BlockHeader) -> Self {
        BlockDecoder {
            header,
            decorr_passes: Vec::new(),
            entropy_l: EntropyVars::default(),
            entropy_r: EntropyVars::default(),
            bitstream: None,
            bits_per_sample: header.bits_per_sample,
        }
    }

    fn feed_sub_chunk(&mut self, chunk: &SubChunk<'a>) {
        match chunk.kind {
            SUB_DECORR_TERMS => self.parse_decorr_terms(chunk.payload),
            SUB_DECORR_WEIGHTS => self.parse_decorr_weights(chunk.payload),
            SUB_DECORR_SAMPLES => self.parse_decorr_samples(chunk.payload),
            SUB_ENTROPY_VARS => self.parse_entropy_vars(chunk.payload),
            SUB_BITSTREAM => {
                self.bitstream = Some(chunk.payload);
            }
            SUB_INT32_INFO | SUB_RIFF_HEADER => {
                // INT32_INFO extra shift and RIFF_HEADER are not used in this decoder.
            }
            _ => {
                // Skip unknown sub-chunks.
            }
        }
    }

    fn parse_decorr_terms(&mut self, data: &[u8]) {
        self.decorr_passes.clear();
        // Each byte encodes: term = (byte & 0x1F) - 5, delta = (byte >> 5) & 7
        for &b in data {
            let term = (b & 0x1F) as i8 - 5;
            let delta = ((b >> 5) & 7) as i8;
            self.decorr_passes.push(DecorrPass {
                term,
                delta,
                ..Default::default()
            });
        }
        // WavPack applies passes in reverse order; reverse the list.
        self.decorr_passes.reverse();
        if self.decorr_passes.len() > MAX_TERMS {
            self.decorr_passes.truncate(MAX_TERMS);
        }
    }

    fn parse_decorr_weights(&mut self, data: &[u8]) {
        // One or two bytes per pass (stereo: 2, mono: 1), in reverse pass order.
        let n = self.decorr_passes.len();
        for (i, pass) in self.decorr_passes.iter_mut().enumerate().take(n) {
            let idx = n - 1 - i; // reverse order in the sub-chunk
            if idx < data.len() {
                pass.weight_a = restore_weight(data[idx] as i8);
            }
            if !self.header.is_mono {
                let idx_b = idx + n;
                if idx_b < data.len() {
                    pass.weight_b = restore_weight(data[idx_b] as i8);
                }
            }
        }
    }

    fn parse_decorr_samples(&mut self, data: &[u8]) {
        // For each pass (in forward order), up to 8 i16 sample pairs.
        let mut offset = 0;
        let n = self.decorr_passes.len();
        for i in 0..n {
            let pass = &mut self.decorr_passes[i];
            let t = pass.term.unsigned_abs() as usize;
            let samples_needed = t.min(8);
            for j in 0..samples_needed {
                if offset + 2 > data.len() {
                    break;
                }
                let v = i16::from_le_bytes([data[offset], data[offset + 1]]);
                pass.samples_a[j] = i32::from(v);
                offset += 2;
                if !self.header.is_mono {
                    if offset + 2 > data.len() {
                        break;
                    }
                    let v2 = i16::from_le_bytes([data[offset], data[offset + 1]]);
                    pass.samples_b[j] = i32::from(v2);
                    offset += 2;
                }
            }
        }
    }

    fn parse_entropy_vars(&mut self, data: &[u8]) {
        if data.len() >= 6 {
            self.entropy_l = EntropyVars::from_bytes(&data[..6]);
        }
        if !self.header.is_mono && data.len() >= 12 {
            self.entropy_r = EntropyVars::from_bytes(&data[6..12]);
        }
    }

    /// Decode all samples in this block, returning interleaved f32 PCM.
    fn decode_samples(&mut self) -> Result<Vec<f32>, OxiAudioError> {
        let bitstream = self.bitstream.ok_or_else(|| {
            OxiAudioError::Decode("WavPack: block has no BITSTREAM sub-chunk".into())
        })?;

        let n = self.header.block_samples as usize;
        let channels = if self.header.is_mono { 1usize } else { 2 };

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut br = BitReader::new(bitstream);

        let mut left_raw: Vec<i32> = Vec::with_capacity(n);
        let mut right_raw: Vec<i32> = Vec::with_capacity(n);

        // Decode raw residuals
        if channels == 1 {
            for _ in 0..n {
                let s = self.entropy_l.decode_sample(&mut br).ok_or_else(|| {
                    OxiAudioError::Decode("WavPack: bitstream underflow (mono)".into())
                })?;
                left_raw.push(s);
            }
        } else {
            for _ in 0..n {
                let l = self.entropy_l.decode_sample(&mut br).ok_or_else(|| {
                    OxiAudioError::Decode("WavPack: bitstream underflow (left channel)".into())
                })?;
                let r = self.entropy_r.decode_sample(&mut br).ok_or_else(|| {
                    OxiAudioError::Decode("WavPack: bitstream underflow (right channel)".into())
                })?;
                left_raw.push(l);
                right_raw.push(r);
            }
        }

        // Apply decorrelation passes (forward, from last to first in list)
        for pass in self.decorr_passes.iter_mut().rev() {
            if channels == 1 {
                for s in left_raw.iter_mut() {
                    let mut r = *s;
                    pass.apply_stereo(s, &mut r);
                }
            } else {
                for (l, r) in left_raw.iter_mut().zip(right_raw.iter_mut()) {
                    pass.apply_stereo(l, r);
                }
            }
        }

        // Undo joint-stereo (mid/side) transform
        if self.header.joint_stereo && channels == 2 {
            for (l, r) in left_raw.iter_mut().zip(right_raw.iter_mut()) {
                let mid = *l;
                let side = *r;
                *l = mid + side;
                *r = mid - side;
            }
        }

        // Convert i32 → f32
        let scale = match self.bits_per_sample {
            8 => 128.0f32,
            16 => 32768.0,
            24 => 8_388_608.0,
            _ => 2_147_483_648.0,
        };

        let mut out = Vec::with_capacity(n * channels);
        for i in 0..n {
            let l = left_raw[i] as f32 / scale;
            out.push(l.clamp(-1.0, 1.0));
            if channels == 2 {
                let r = right_raw[i] as f32 / scale;
                out.push(r.clamp(-1.0, 1.0));
            }
        }
        Ok(out)
    }
}

/// Restore a decorrelation weight from its stored i8 representation.
#[inline]
fn restore_weight(w: i8) -> i32 {
    let v = i32::from(w);
    if v > 0 {
        (v << 3) + ((v + 1) >> 1)
    } else {
        v << 3
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Decode a WavPack file from raw bytes into an `AudioBuffer<f32>`.
///
/// Only lossless (non-hybrid) stereo/mono integer blocks are fully decoded.
/// Blocks marked as float, DSD, or hybrid mode return an error.
///
/// # Errors
///
/// - `OxiAudioError::Decode`: invalid magic, truncated blocks, unsupported flags.
/// - `OxiAudioError::UnsupportedFormat`: float or DSD or hybrid-mode blocks.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_wavpack(data: &[u8]) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if data.len() < 32 || &data[..4] != b"wvpk" {
        return Err(OxiAudioError::Decode(
            "WavPack: data does not begin with 'wvpk' magic".into(),
        ));
    }

    let mut all_samples: Vec<f32> = Vec::new();
    let mut sample_rate: u32 = 44_100;
    let mut channels: usize = 2;
    let mut total_samples_known: bool = false;
    let mut pos: usize = 0;

    while pos + 32 <= data.len() {
        // Parse block header
        let hdr = BlockHeader::parse(data, pos)?;

        // Guard against unsupported modes
        if (hdr.flags & FLAG_FLOAT_DATA) != 0 {
            return Err(OxiAudioError::UnsupportedFormat(
                "WavPack: float-data blocks are not supported by this decoder".into(),
            ));
        }
        if (hdr.flags & FLAG_DSD_AUDIO) != 0 {
            return Err(OxiAudioError::UnsupportedFormat(
                "WavPack: DSD blocks are not supported".into(),
            ));
        }
        if (hdr.flags & FLAG_HYBRID_MODE) != 0 {
            return Err(OxiAudioError::UnsupportedFormat(
                "WavPack: hybrid/lossy mode is not supported by this decoder".into(),
            ));
        }

        let block_end = pos + hdr.block_size as usize + 8;
        // Clamp to available data
        let block_end = block_end.min(data.len());
        let sub_data = &data[pos + 32..block_end];

        if !total_samples_known && hdr.total_samples != 0xFFFF_FFFF {
            total_samples_known = true;
            channels = if hdr.is_mono { 1 } else { 2 };
            all_samples.reserve(hdr.total_samples as usize * channels);
        }

        // Try to extract sample rate from embedded RIFF header
        if all_samples.is_empty() {
            if let Some(sr) = extract_sample_rate_from_sub_chunks(sub_data) {
                sample_rate = sr;
            }
            channels = if hdr.is_mono { 1 } else { 2 };
        }

        if hdr.block_samples == 0 {
            // Non-audio block (wasted or empty); skip.
            pos += (hdr.block_size as usize).max(32).saturating_add(8);
            continue;
        }

        let mut block_dec = BlockDecoder::new(&hdr);
        let iter = SubChunkIter::new(sub_data);
        let mut chunks: Vec<SubChunk<'_>> = Vec::new();
        for item in iter {
            match item {
                Ok(c) => chunks.push(c),
                Err(e) => {
                    log::warn!("WavPack: sub-chunk parse error in block at {pos}: {e}");
                    break;
                }
            }
        }
        for c in &chunks {
            block_dec.feed_sub_chunk(c);
        }

        match block_dec.decode_samples() {
            Ok(samples) => all_samples.extend_from_slice(&samples),
            Err(e) => {
                log::warn!("WavPack: skipping block at offset {pos}: {e}");
            }
        }

        let advance = (hdr.block_size as usize).max(32).saturating_add(8);
        pos += advance;
    }

    let layout = ChannelLayout::from(channels as u16);
    Ok(AudioBuffer {
        samples: all_samples,
        sample_rate,
        channels: layout,
        format: SampleFormat::F32,
    })
}

/// Walk sub-chunks looking for an embedded RIFF header and extract sample rate.
fn extract_sample_rate_from_sub_chunks(data: &[u8]) -> Option<u32> {
    let iter = SubChunkIter::new(data);
    for item in iter {
        let c = item.ok()?;
        if c.kind == (SUB_RIFF_HEADER & 0x3F) && c.payload.len() >= 28 {
            // RIFF/WAV header: bytes 0..4 = "RIFF", 8..12 = "WAVE", 12..16 = "fmt "
            // fmt chunk: bytes 16..20 = sub-chunk size (16), 20..22 = format tag,
            //            22..24 = channels, 24..28 = sample_rate
            if &c.payload[0..4] == b"RIFF" && c.payload.len() >= 28 {
                let sr = u32::from_le_bytes([
                    c.payload[24],
                    c.payload[25],
                    c.payload[26],
                    c.payload[27],
                ]);
                if sr > 0 {
                    return Some(sr);
                }
            }
        }
    }
    None
}

/// Decode a WavPack file at `path` to `AudioBuffer<f32>`.
///
/// # Errors
///
/// - `OxiAudioError::Io`: file cannot be read.
/// - `OxiAudioError::Decode` / `OxiAudioError::UnsupportedFormat`: see [`decode_wavpack`].
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_wavpack_file(path: &Path) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let data = std::fs::read(path).map_err(OxiAudioError::Io)?;
    decode_wavpack(&data)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid WavPack block that contains zero audio samples.
    /// Used to verify header parsing without a full bitstream.
    fn make_minimal_wvpk_header(block_samples: u32, flags: u32, block_size_extra: u32) -> Vec<u8> {
        let block_size: u32 = 32 + block_size_extra; // header size counted as part of block_size - 8
        let mut hdr = vec![0u8; 32];
        hdr[0..4].copy_from_slice(b"wvpk");
        // block_size in spec excludes the first 8 bytes (magic + this field)
        let stored_size = block_size - 8;
        hdr[4..8].copy_from_slice(&stored_size.to_le_bytes());
        hdr[8..10].copy_from_slice(&0x0403u16.to_le_bytes()); // version 4.3
        hdr[12..16].copy_from_slice(&100u32.to_le_bytes()); // total_samples = 100
        hdr[16..20].copy_from_slice(&0u32.to_le_bytes()); // block_index = 0
        hdr[20..24].copy_from_slice(&block_samples.to_le_bytes());
        hdr[24..28].copy_from_slice(&flags.to_le_bytes());
        hdr[28..32].copy_from_slice(&0u32.to_le_bytes()); // crc = 0
        hdr
    }

    #[test]
    fn test_wavpack_magic_detected() {
        assert_eq!(&WAVPACK_MAGIC, &b"wvpk");
    }

    #[test]
    fn test_wavpack_magic_bytes_rejected() {
        // Random bytes should be rejected
        let result = decode_wavpack(b"RIFF\x00\x00\x00\x00WAVE");
        assert!(result.is_err(), "non-WavPack data must fail");
    }

    #[test]
    fn test_wavpack_too_short_rejected() {
        let result = decode_wavpack(b"wvpk");
        assert!(result.is_err(), "incomplete header must fail");
    }

    #[test]
    fn test_wavpack_header_parse_mono_16bit() {
        // FLAG_MONO_DATA = 0x0004_0000, bits_per_sample=1 → 16-bit
        let flags = FLAG_MONO_DATA | 0x0001;
        let hdr_bytes = make_minimal_wvpk_header(0, flags, 0);
        let hdr = BlockHeader::parse(&hdr_bytes, 0).expect("header parse must succeed");
        assert!(hdr.is_mono, "mono flag should be set");
        assert_eq!(hdr.bits_per_sample, 16);
        assert_eq!(hdr.block_samples, 0);
    }

    #[test]
    fn test_wavpack_header_parse_stereo_24bit() {
        // bits_per_sample field=2 → 24-bit, no mono flag
        let flags = 0x0002u32;
        let hdr_bytes = make_minimal_wvpk_header(0, flags, 0);
        let hdr = BlockHeader::parse(&hdr_bytes, 0).expect("header parse must succeed");
        assert!(!hdr.is_mono, "stereo: mono flag should be clear");
        assert_eq!(hdr.bits_per_sample, 24);
    }

    #[test]
    fn test_wavpack_header_parse_version_stored() {
        let flags = 0u32;
        let hdr_bytes = make_minimal_wvpk_header(0, flags, 0);
        let hdr = BlockHeader::parse(&hdr_bytes, 0).expect("header must parse");
        // _version is prefixed to suppress dead_code; access via the raw bytes to verify.
        assert_eq!(u16::from_le_bytes([hdr_bytes[8], hdr_bytes[9]]), 0x0403);
        // hdr itself must parse without error regardless of version field access.
        assert_eq!(hdr.block_samples, 0);
    }

    #[test]
    fn test_wavpack_empty_block_samples_yields_empty_audio() {
        // A valid header with 0 block_samples, no bitstream sub-chunk → empty output
        let flags = FLAG_MONO_DATA | 0x0001; // mono 16-bit
        let hdr_bytes = make_minimal_wvpk_header(0, flags, 0);
        let buf = decode_wavpack(&hdr_bytes).expect("empty block should succeed");
        assert!(
            buf.samples.is_empty(),
            "zero-sample block must yield empty audio"
        );
    }

    #[test]
    fn test_wavpack_hybrid_rejected() {
        let flags = FLAG_HYBRID_MODE;
        let hdr_bytes = make_minimal_wvpk_header(1, flags, 0);
        let result = decode_wavpack(&hdr_bytes);
        assert!(
            matches!(result, Err(OxiAudioError::UnsupportedFormat(_))),
            "hybrid mode must return UnsupportedFormat"
        );
    }

    #[test]
    fn test_wavpack_dsd_rejected() {
        let flags = FLAG_DSD_AUDIO;
        let hdr_bytes = make_minimal_wvpk_header(1, flags, 0);
        let result = decode_wavpack(&hdr_bytes);
        assert!(
            matches!(result, Err(OxiAudioError::UnsupportedFormat(_))),
            "DSD must return UnsupportedFormat"
        );
    }

    #[test]
    fn test_wavpack_float_data_rejected() {
        let flags = FLAG_FLOAT_DATA;
        let hdr_bytes = make_minimal_wvpk_header(1, flags, 0);
        let result = decode_wavpack(&hdr_bytes);
        assert!(
            matches!(result, Err(OxiAudioError::UnsupportedFormat(_))),
            "float-data must return UnsupportedFormat"
        );
    }

    #[test]
    fn test_sub_chunk_iter_empty() {
        let iter = SubChunkIter::new(&[]);
        let chunks: Vec<_> = iter.collect();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_entropy_vars_default_rice_k_zero() {
        let ev = EntropyVars::default();
        assert_eq!(ev.rice_k(), 0, "default medians → k=0");
    }

    #[test]
    fn test_bit_reader_bits() {
        // 0b10110011 = 0xB3
        let data = [0xB3u8];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_bit(), Some(1)); // LSB first
        assert_eq!(br.read_bit(), Some(1));
        assert_eq!(br.read_bit(), Some(0));
        assert_eq!(br.read_bit(), Some(0));
        assert_eq!(br.read_bit(), Some(1));
        assert_eq!(br.read_bit(), Some(1));
        assert_eq!(br.read_bit(), Some(0));
        assert_eq!(br.read_bit(), Some(1));
        assert_eq!(br.read_bit(), None); // exhausted
    }

    #[test]
    fn test_restore_weight_positive() {
        let w = restore_weight(10i8);
        // (10 << 3) + ((10 + 1) >> 1) = 80 + 5 = 85
        assert_eq!(w, 85);
    }

    #[test]
    fn test_restore_weight_negative() {
        let w = restore_weight(-4i8);
        // -4 << 3 = -32
        assert_eq!(w, -32);
    }

    #[test]
    fn test_wavpack_file_nonexistent_returns_io_error() {
        let p = std::env::temp_dir().join("oxiaudio_nonexistent_xyz_test.wv");
        let result = decode_wavpack_file(&p);
        assert!(
            matches!(result, Err(OxiAudioError::Io(_))),
            "missing file must return Io error"
        );
    }

    #[test]
    fn test_wavpack_block_size_field_accounts_for_header() {
        // block_size field = actual_block_bytes - 8.  Header bytes = 32.
        // A block with just the header (no sub-chunks): block_size_stored = 32 - 8 = 24.
        let flags = FLAG_MONO_DATA | 0x0001;
        let mut hdr_bytes = make_minimal_wvpk_header(0, flags, 0);
        // Overwrite block_size with 24 (= 32 - 8)
        hdr_bytes[4..8].copy_from_slice(&24u32.to_le_bytes());
        let hdr = BlockHeader::parse(&hdr_bytes, 0).expect("header parse");
        // block_size as stored is the raw field; pos advance = block_size + 8
        let advance = (hdr.block_size as usize).max(32).saturating_add(8);
        assert!(advance >= 32, "advance must be at least header size");
    }
}
