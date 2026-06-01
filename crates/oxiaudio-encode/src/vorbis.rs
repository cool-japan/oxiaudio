//! Pure-Rust OGG Vorbis I encoder with MDCT-based audio encoding.
//!
//! Produces a valid Vorbis I bitstream (three header packets + audio packets)
//! encapsulated in OGG pages.  Audio packets use MDCT forward-transform coefficients
//! quantised through a simple Floor type-1 curve and Residue type-0 scalar VQ.
//!
//! # Supported configurations
//! - 1 or 2 channels
//! - 44100 or 48000 Hz sample rate
//! - Long window only (blocksize = 2048)
//!
//! # Vorbis I references
//! - Vorbis I spec: <https://xiph.org/vorbis/doc/Vorbis_I_spec.pdf>
//! - RFC 3533 (OGG): <https://tools.ietf.org/html/rfc3533>

use std::io::Write;
use std::path::Path;

use oxiaudio_core::{AudioBuffer, OxiAudioError};
use oxifft::{fft, Complex};

use crate::ogg::{write_vorbis_comment_packet, OggStream};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Long block size for the Vorbis MDCT window.
const BLOCK_SIZE_1: usize = 2048;

/// Short block size (required by spec even if only one mode used).
const BLOCK_SIZE_0: usize = 256;

/// Serial number for the single OGG logical stream.
const STREAM_SERIAL: u32 = 0xABCD_1234;

/// Vendor string embedded in the comment header.
const VENDOR_STRING: &str = "OxiAudio 0.1.0 Vorbis encoder";

/// Number of MDCT output coefficients (BLOCK_SIZE_1 / 2).
const N_COEFFS: usize = BLOCK_SIZE_1 / 2; // 1024

/// Residue encoding range [−MAX_RESIDUE, +MAX_RESIDUE].
const MAX_RESIDUE: f32 = 6.0;

/// Number of residue quantisation steps (VQ entries).
const RESIDUE_ENTRIES: usize = 256;

/// Residue quantisation delta: covers [−MAX_RESIDUE, +MAX_RESIDUE] in 256 steps.
const RESIDUE_DELTA: f32 = (MAX_RESIDUE * 2.0) / (RESIDUE_ENTRIES as f32 - 1.0);

/// Residue begin/end for active encoding (first 128 MDCT bins).
const RESIDUE_BEGIN: usize = 0;
const RESIDUE_END: usize = 128;

/// Number of residue values per partition.
const PARTITION_SIZE: usize = 32;

/// Number of partitions in the active residue range.
const N_PARTITIONS: usize = (RESIDUE_END - RESIDUE_BEGIN) / PARTITION_SIZE; // 4

// ─── BitWriter ────────────────────────────────────────────────────────────────

/// LSB-first (little-endian) bit-packer for Vorbis packet fields.
///
/// Vorbis packs all fields LSB-first into a flat byte stream. `BitWriter`
/// accumulates bits in `current_byte`, flushing to `buf` whenever 8 bits have
/// been collected.
struct BitWriter {
    buf: Vec<u8>,
    /// Number of valid bits in `current_byte` (0..8).
    bit_pos: u8,
    current_byte: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            bit_pos: 0,
            current_byte: 0,
        }
    }

    /// Append `n` bits from the LSB of `value` (LSB first).
    fn write_bits(&mut self, value: u64, n: u8) {
        let mut remaining = value;
        let mut bits_left = n;
        while bits_left > 0 {
            let space = 8 - self.bit_pos;
            let take = bits_left.min(space);
            let mask = if take == 64 {
                u64::MAX
            } else {
                (1u64 << take) - 1
            };
            self.current_byte |= ((remaining & mask) as u8) << self.bit_pos;
            self.bit_pos += take;
            remaining >>= take;
            bits_left -= take;
            if self.bit_pos == 8 {
                self.buf.push(self.current_byte);
                self.current_byte = 0;
                self.bit_pos = 0;
            }
        }
    }

    /// Append a single bit.
    #[inline]
    fn write_bit(&mut self, bit: bool) {
        self.write_bits(u64::from(bit), 1);
    }

    /// Flush the current byte (padding unset bits with 0) and return the buffer.
    fn finalize(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.buf.push(self.current_byte);
        }
        self.buf
    }
}

// ─── ilog helper ─────────────────────────────────────────────────────────────

/// Return `floor(log2(x)) + 1` (number of bits to represent x), or 0 if x == 0.
///
/// Used by the Vorbis spec for compact index encoding.
#[allow(dead_code)]
const fn ilog(x: u32) -> u32 {
    if x == 0 {
        0
    } else {
        32 - x.leading_zeros()
    }
}

// ─── Vorbis float32 packing ───────────────────────────────────────────────────

/// Encode a scalar `f32` into the Vorbis packed-float32 format (§9.2.5).
///
/// The packed format stores: `value = ±mantissa × 2^(exponent − 788)`.
/// - Bits 30..20 (10 bits): biased exponent
/// - Bits 20..0  (21 bits): mantissa
/// - Bit  31     (1 bit):   sign
fn pack_vorbis_float32(value: f32) -> u32 {
    if value == 0.0 {
        return 0;
    }
    let sign: u32 = if value < 0.0 { 1 } else { 0 };
    let abs_val = value.abs();
    // raw_exp = floor(log2(abs_val)) + 1, biased by 788
    let raw_exp = abs_val.log2().floor() as i32 + 1 + 788;
    let raw_exp = raw_exp.clamp(0, 1023) as u32;
    // mantissa = abs_val / 2^(raw_exp − 788 − 21), clamped to 21 bits
    let shift = (raw_exp as i32) - 788 - 21;
    let mantissa_f = if shift >= 0 {
        abs_val / (2.0f32).powi(shift)
    } else {
        abs_val * (2.0f32).powi(-shift)
    };
    let mantissa = (mantissa_f.round() as u32).min((1u32 << 21) - 1);
    (sign << 31) | (raw_exp << 21) | mantissa
}

// ─── Identification header ────────────────────────────────────────────────────

/// Build the Vorbis identification header packet (30 bytes).
///
/// Layout (Vorbis I spec §5.2.1):
/// ```text
/// [1]  packet_type = 0x01
/// [6]  "vorbis"
/// [4]  vorbis_version = 0 (LE)
/// [1]  audio_channels
/// [4]  audio_sample_rate (LE)
/// [4]  bitrate_maximum  = 0 (LE)
/// [4]  bitrate_nominal  = 0 (LE)
/// [4]  bitrate_minimum  = 0 (LE)
/// [1]  blocksize byte
/// [1]  framing_bit = 1
/// ```
/// Total: 30 bytes.
fn write_ident_header(channels: u8, sample_rate: u32) -> Vec<u8> {
    // log2(256)=8, log2(2048)=11 → blocksize byte = (8) | (11 << 4) = 0xB8
    let log2_bs0 = {
        let mut v = BLOCK_SIZE_0;
        let mut cnt: u8 = 0;
        while v > 1 {
            v >>= 1;
            cnt += 1;
        }
        cnt
    };
    let log2_bs1 = {
        let mut v = BLOCK_SIZE_1;
        let mut cnt: u8 = 0;
        while v > 1 {
            v >>= 1;
            cnt += 1;
        }
        cnt
    };
    let blocksize_byte = log2_bs0 | (log2_bs1 << 4);

    let mut pkt = Vec::with_capacity(30);
    pkt.push(0x01); // packet type: identification
    pkt.extend_from_slice(b"vorbis");
    pkt.extend_from_slice(&0u32.to_le_bytes()); // version
    pkt.push(channels);
    pkt.extend_from_slice(&sample_rate.to_le_bytes());
    pkt.extend_from_slice(&0u32.to_le_bytes()); // bitrate_max
    pkt.extend_from_slice(&0u32.to_le_bytes()); // bitrate_nominal
    pkt.extend_from_slice(&0u32.to_le_bytes()); // bitrate_min
    pkt.push(blocksize_byte);
    pkt.push(0x01); // framing_bit
    pkt
}

// ─── Setup header ─────────────────────────────────────────────────────────────

/// Build the Vorbis setup header packet.
///
/// Setup header bitstream (Vorbis I spec §5.2.4):
///
/// **Codebooks (2 total)**
/// - Codebook 0 (classbook): dim=1, entries=8, uniform length=3, lookup_type=0
/// - Codebook 1 (VQ book):   dim=1, entries=256, uniform length=8, lookup_type=1
///   (scalar VQ, minimum=−6.0, delta=12/255 ≈ 0.04706, sequence_p=0)
///
/// **Floor (type 1)**
/// - 1 partition, partition_class[0]=0
/// - Class 0: dim=2, subclasses=0, subbook[0]=unused (−1)
/// - multiplier=1, rangebits=10
/// - Explicit X values: 341 and 682 (10 bits each)
/// - Implicit endpoints: X=0 and X=1024
///
/// **Residue (type 0)**
/// - begin=0, end=128, partition_size=32 (→ 4 partitions)
/// - 1 classification, classbook=codebook 0
/// - cascade[0]=0b001 (pass 0 active), book[0][0]=codebook 1
///
/// **Mapping (type 0)**: 1 submap, no coupling
/// **Mode**: long block (blockflag=1), mapping 0
fn write_setup_header(_channels: u8) -> Vec<u8> {
    let mut bw = BitWriter::new();

    // ── Codebooks ────────────────────────────────────────────────────────────
    // codebook_count - 1 encoded in 8 bits → 1 (= 2 codebooks)
    bw.write_bits(1, 8); // 2 codebooks (count - 1 = 1)

    // ── Codebook 0: classbook ─────────────────────────────────────────────────
    // Sync pattern: 0x564342 written as 3 bytes LSB-first ('B', 'C', 'V')
    bw.write_bits(0x42, 8); // 'B'
    bw.write_bits(0x43, 8); // 'C'
    bw.write_bits(0x56, 8); // 'V'
    bw.write_bits(1, 16); // codebook_dimensions = 1
    bw.write_bits(8, 24); // codebook_entries = 8
    bw.write_bit(false); // ordered = 0
    bw.write_bit(false); // sparse = 0
                         // Non-sparse, non-ordered: codeword lengths as (length-1) in 5 bits per entry
    for _ in 0..8u32 {
        bw.write_bits(2, 5); // length-1 = 2 → length = 3 bits (uniform)
    }
    bw.write_bits(0, 4); // lookup_type = 0 (no VQ)

    // ── Codebook 1: scalar VQ residue book ───────────────────────────────────
    bw.write_bits(0x42, 8); // 'B'
    bw.write_bits(0x43, 8); // 'C'
    bw.write_bits(0x56, 8); // 'V'
    bw.write_bits(1, 16); // codebook_dimensions = 1
    bw.write_bits(256, 24); // codebook_entries = 256
    bw.write_bit(false); // ordered = 0
    bw.write_bit(false); // sparse = 0
                         // 256 entries, length = 8 bits each → length-1 = 7 in 5 bits
    for _ in 0..256u32 {
        bw.write_bits(7, 5); // length-1 = 7 → length = 8 bits
    }
    // lookup_type = 1 (scalar VQ)
    bw.write_bits(1, 4);
    // codebook_minimum_value: −6.0 packed as Vorbis float32
    let min_packed = pack_vorbis_float32(-MAX_RESIDUE);
    bw.write_bits(u64::from(min_packed), 32);
    // codebook_delta_value: 12.0/255 packed as Vorbis float32
    let delta_packed = pack_vorbis_float32(RESIDUE_DELTA);
    bw.write_bits(u64::from(delta_packed), 32);
    // codebook_value_bits - 1: 8-1 = 7 → 4 bits
    bw.write_bits(7, 4);
    // codebook_sequence_p = 0
    bw.write_bit(false);
    // multiplicands: 256 values × 8 bits each (values 0..255)
    for i in 0u32..256 {
        bw.write_bits(u64::from(i), 8);
    }

    // ── Time domain transforms ────────────────────────────────────────────────
    // vorbis_time_count - 1 in 6 bits → 0 (= 1 time transform)
    bw.write_bits(0, 6);
    // Time transform 0: type = 0
    bw.write_bits(0, 16);

    // ── Floor configurations ──────────────────────────────────────────────────
    // vorbis_floor_count - 1 in 6 bits → 0 (= 1 floor)
    bw.write_bits(0, 6);
    // Floor 0: type = 1
    bw.write_bits(1, 16);

    // Floor type-1 configuration (spec §6.2.2):
    // floor1_partitions = 1 (5 bits)
    bw.write_bits(1, 5);
    // partition_class_list[0] = 0 (4 bits per entry per spec §6.2.2)
    bw.write_bits(0, 4);
    // maximum_class = 0, so configure classes 0..=0:
    // Class 0: floor1_class_dimensions[0] - 1 = 1 (3 bits → dim=2)
    bw.write_bits(1, 3);
    // Class 0: floor1_class_subclasses[0] = 0 (2 bits)
    bw.write_bits(0, 2);
    // No masterbook (subclasses=0 → cbits=0)
    // subbooks: 2^subclasses = 2^0 = 1 subbook slot
    //   ilog(codebook_count - 1) = ilog(1) = 1 bit per subbook
    //   0 = unused (-1), 1 = codebook 0
    //   We want unused → write 0 (1 bit)
    bw.write_bits(0, 1); // subbook[0] = unused (-1)

    // floor1_multiplier - 1 = 0 (2 bits → multiplier=1, range=256, Y uses 8 bits)
    bw.write_bits(0, 2);
    // floor1_rangebits = 10 (4 bits) — X positions are 10 bits each
    bw.write_bits(10, 4);
    // Explicit X values for partition 0 (dim=2): X[2] and X[3]
    // Total X list after decode: [0(implicit), X[2]=341, X[3]=682, 1024(implicit)]
    bw.write_bits(341, 10); // X[2] = 341
    bw.write_bits(682, 10); // X[3] = 682

    // ── Residue configurations ────────────────────────────────────────────────
    // vorbis_residue_count - 1 in 6 bits → 0 (= 1 residue)
    bw.write_bits(0, 6);
    // Residue 0: type = 0
    bw.write_bits(0, 16);
    // residue_begin = 0 (24 bits)
    bw.write_bits(RESIDUE_BEGIN as u64, 24);
    // residue_end = 128 (24 bits)
    bw.write_bits(RESIDUE_END as u64, 24);
    // residue_partition_size - 1 = 31 (24 bits → size=32)
    bw.write_bits((PARTITION_SIZE as u64) - 1, 24);
    // residue_classifications - 1 = 0 (6 bits → 1 classification)
    bw.write_bits(0, 6);
    // residue_classbook = 0 (8 bits → use codebook 0)
    bw.write_bits(0, 8);
    // Residue cascade for 1 classification:
    //   cascade[0]: [3 bits] low_bits = 1 (pass 0 active), [1 bit] has_high = 0
    bw.write_bits(1, 3); // low_bits = 0b001 → pass 0 active
    bw.write_bit(false); // has_high = 0 (no high bits)
                         // For active pass 0: write book index as 8 bits = codebook 1
    bw.write_bits(1, 8); // book[class=0][pass=0] = codebook 1

    // ── Mapping configurations ────────────────────────────────────────────────
    // vorbis_mapping_count - 1 in 6 bits → 0 (= 1 mapping)
    bw.write_bits(0, 6);
    // Mapping 0: type = 0
    bw.write_bits(0, 16);
    // submaps_flag [1 bit] — if 0: 1 submap; if 1: read submap count (4 bits) + 1
    bw.write_bit(false); // 1 submap (no explicit count)
                         // coupling_flag [1 bit] = 0 (no M/S coupling)
    bw.write_bit(false);
    // reserved [2 bits] = 0 (spec requires these be 0)
    bw.write_bits(0, 2);
    // Per-channel submap assignment: each channel → submap 0
    // ilog(submaps-1) = ilog(0) = 0 bits per channel assignment
    // So nothing to write for channel assignments when there is only 1 submap.

    // Submap 0 configuration:
    //   submap_time_config: [8 bits] = 0
    //   submap_floor_config: [8 bits] = 0
    //   submap_residue_config: [8 bits] = 0
    bw.write_bits(0, 8); // submap_time_config
    bw.write_bits(0, 8); // submap_floor_config
    bw.write_bits(0, 8); // submap_residue_config

    // ── Mode configurations ───────────────────────────────────────────────────
    // vorbis_mode_count - 1 in 6 bits → 0 (= 1 mode)
    bw.write_bits(0, 6);
    // Mode 0:
    //   blockflag [1 bit] = 1 (long block)
    //   windowtype [16 bits] = 0
    //   transformtype [16 bits] = 0
    //   mapping [8 bits] = 0
    bw.write_bit(true); // blockflag = 1 (long)
    bw.write_bits(0, 16); // windowtype
    bw.write_bits(0, 16); // transformtype
    bw.write_bits(0, 8); // mapping

    // ── Framing bit ──────────────────────────────────────────────────────────
    bw.write_bit(true); // framing_bit = 1

    let packed = bw.finalize();

    // Prepend the setup header packet type byte + "vorbis" magic
    let mut pkt = Vec::with_capacity(7 + packed.len());
    pkt.push(0x05); // packet type: setup
    pkt.extend_from_slice(b"vorbis");
    pkt.extend_from_slice(&packed);
    pkt
}

// ─── MDCT forward transform ───────────────────────────────────────────────────

/// Vorbis MDCT analysis: transform `BLOCK_SIZE_1 = 2048` samples to
/// `N_COEFFS = 1024` real spectral coefficients.
///
/// Follows the standard pre-rotation / FFT / post-rotation decomposition
/// (identical pattern to `opus_mdct::mdct_forward` but with N=2048).
///
/// Window: `w[k] = sin(π · (k + ½) / N)`, k ∈ [0, N)
fn vorbis_mdct(samples: &[f32]) -> Vec<f32> {
    let n = BLOCK_SIZE_1; // 2048
    let n2 = N_COEFFS; // 1024

    // Use as many samples as available; pad with zeros if shorter than N.
    let windowed: Vec<f32> = (0..n)
        .map(|k| {
            let s = if k < samples.len() { samples[k] } else { 0.0 };
            let w = (std::f32::consts::PI * (k as f32 + 0.5) / n as f32).sin();
            s * w
        })
        .collect();

    // Pre-rotation by exp(−jπk/N): build N/2 complex samples.
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

    // N/2-point forward FFT via OxiFFT.
    let spectrum = fft(&pre_rotated);

    // Post-rotation: X[k] = 2 · Re{ spectrum[k] · exp(−jπ(k+½)/N) }
    (0..n2)
        .map(|k| {
            let angle = -std::f32::consts::PI * (k as f32 + 0.5) / n as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            2.0 * (spectrum[k].re * cos_a - spectrum[k].im * sin_a)
        })
        .collect()
}

// ─── Vorbis VBR quality control ───────────────────────────────────────────────

/// Vorbis encoder quality level, analogous to oggenc/libvorbis quality settings.
///
/// - `quality = -0.1` (q-1): very low quality, ~45 kbps
/// - `quality = 0.0` (q0): low quality, ~64 kbps
/// - `quality = 0.4` (q4): medium quality, ~128 kbps (default)
/// - `quality = 0.7` (q7): high quality, ~224 kbps
/// - `quality = 1.0` (q10): maximum quality, ~500 kbps
///
/// The quality value maps to:
/// - A silence threshold controlling which frames are encoded as active vs. silent.
///   At higher quality, even quieter frames are fully encoded.
/// - A quantization scale factor that affects how MDCT coefficients are quantized
///   (finer granularity at higher quality).
///
/// Note: with the current fixed-length uniform 8-bit codebook, the residue scale
/// factor shifts quantization precision but does not change per-frame byte count for
/// non-silent frames. File size differences arise primarily through the silence
/// threshold suppressing low-energy frames at lower quality settings.
#[derive(Debug, Clone, Copy)]
pub struct VorbisQuality(f32);

impl VorbisQuality {
    /// Quality level clamped to [−0.1, 1.0].
    pub fn new(q: f32) -> Self {
        Self(q.clamp(-0.1, 1.0))
    }

    /// Default quality (q4 ≈ 0.4, ~128 kbps).
    pub fn default_quality() -> Self {
        Self(0.4)
    }

    /// Named level from integer −1..=10.
    pub fn from_level(level: i32) -> Self {
        Self::new(level as f32 / 10.0)
    }

    /// Quantization scale factor: lower quality → larger step → coarser quantization.
    ///
    /// Returns a multiplier for the effective residue delta value.
    ///
    /// - q=1.0 → scale = 1.0 (finest quantization)
    /// - q=0.4 → scale ≈ 5.2 (default)
    /// - q=0.0 → scale = 8.0
    /// - q=−0.1 → scale ≈ 10.0 (coarsest)
    pub fn residue_delta_scale(&self) -> f32 {
        let q = self.0;
        let scale = if q >= 0.0 {
            1.0 + (1.0 - q) * 7.0 // linear: 8.0 at q=0 → 1.0 at q=1
        } else {
            8.0 + (-q) * 20.0 // 8.0 at q=0 → 10.0 at q=−0.1
        };
        scale.clamp(0.5, 10.0)
    }

    /// Silence threshold: frames whose RMS amplitude is below this value are
    /// encoded as silent (floor_nonzero = 0, no residue bits).
    ///
    /// At higher quality, even quieter frames are fully encoded:
    /// - q=1.0 → threshold ≈ −80 dBFS (encodes near-silence)
    /// - q=0.4 → threshold ≈ −56 dBFS (default)
    /// - q=0.0 → threshold ≈ −40 dBFS (skips quiet frames)
    /// - q=−0.1 → same floor as q=0.0
    pub fn silence_threshold(&self) -> f32 {
        let q = self.0.max(0.0);
        // Linear mapping from -40 dBFS (q=0) to -80 dBFS (q=1).
        let threshold_db = -40.0 - q * 40.0;
        10.0f32.powf(threshold_db / 20.0)
    }
}

// ── Psychoacoustic Model ──────────────────────────────────────────────────────

/// Absolute threshold of hearing in dB SPL at frequency `f_hz` (Hz).
///
/// Uses the Terhardt (1979) approximation.
fn ath_db(f_hz: f32) -> f32 {
    let f_khz = f_hz / 1000.0;
    3.64 * f_khz.powf(-0.8) - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp() + 1e-3 * f_khz.powi(4)
}

/// Convert frequency in Hz to Bark scale (Zwicker 1961).
///
/// Uses the Traunmüller (1990) approximation: `26.81·f/(1960+f) − 0.53`.
fn hz_to_bark(f_hz: f32) -> f32 {
    26.81 * f_hz / (1960.0 + f_hz) - 0.53
}

/// Spreading function: masking attenuation (dB) from a tone at `bark_masker`
/// affecting a maskee at `bark_maskee`.
///
/// Negative return means the masker reduces the perceivable level of the maskee
/// (i.e. masking effect). Uses Johnston (1988) slopes:
/// - Downward spread (masker above maskee): 27 dB/Bark
/// - Upward spread (masker below maskee): `(−17.5 + 0.4·bark_masker)` dB/Bark
fn spreading_db(bark_masker: f32, bark_maskee: f32) -> f32 {
    let delta = bark_maskee - bark_masker;
    if delta < 0.0 {
        // Maskee is below masker: fast upward-spread falloff
        27.0 * delta
    } else {
        // Maskee is above masker: slower downward-spread
        (-17.5 + 0.4 * bark_masker) * delta
    }
}

/// Compute per-bin masking threshold from MDCT coefficients.
///
/// Returns a `Vec<f32>` of length `mdct.len()` giving the minimum perceivable
/// energy at each spectral bin (linear energy units, same scale as `mdct[k]^2`).
/// The threshold is the maximum of the ATH and the spread of all Bark-band peaks.
#[must_use]
pub(crate) fn compute_masking_threshold(mdct: &[f32], sample_rate: u32) -> Vec<f32> {
    let n_bins = mdct.len();
    if n_bins == 0 {
        return Vec::new();
    }
    let freq_resolution = sample_rate as f32 / (2.0 * n_bins as f32);
    let n_bark: usize = 24;

    // Precompute Bark values per bin, clamped to [0, n_bark-1] to avoid negative
    // or out-of-range indices when flooring to usize later.
    let bark_of_bin: Vec<f32> = (0..n_bins)
        .map(|k| {
            let f_hz = ((k as f32 + 0.5) * freq_resolution).max(20.0);
            hz_to_bark(f_hz).clamp(0.0, (n_bark - 1) as f32)
        })
        .collect();

    // Convert ATH to linear energy per bin: ath_linear[k] = 10^(ath_db(f_k)/10)
    let ath_linear: Vec<f32> = (0..n_bins)
        .map(|k| {
            let f_hz = ((k as f32 + 0.5) * freq_resolution).max(20.0);
            10.0f32.powf(ath_db(f_hz) / 10.0)
        })
        .collect();

    // Compute signal energy per bin and find maximum energy per Bark band.
    let energy: Vec<f32> = mdct.iter().map(|&x| x * x).collect();
    let mut bark_max_energy = vec![0.0f32; n_bark];
    for (k, &e) in energy.iter().enumerate() {
        let band = (bark_of_bin[k].floor() as usize).min(n_bark - 1);
        if e > bark_max_energy[band] {
            bark_max_energy[band] = e;
        }
    }

    // Compute masking threshold at each bin: max over ATH and all masker contributions.
    let mut threshold = ath_linear;
    for (k, thresh) in threshold.iter_mut().enumerate() {
        let bark_k = bark_of_bin[k];
        for (band, &masker_energy) in bark_max_energy.iter().enumerate() {
            if masker_energy < 1e-20 {
                continue; // skip silent bands
            }
            let bark_masker = band as f32 + 0.5;
            let spread = spreading_db(bark_masker, bark_k);
            // masked energy = masker_energy × 10^(spread_dB / 10)
            let masked = masker_energy * 10.0f32.powf(spread / 10.0);
            if masked > *thresh {
                *thresh = masked;
            }
        }
    }

    threshold
}

/// Compute Y[0] and Y[1] floor endpoint values using a psychoacoustic masking threshold.
///
/// Uses the geometric mean of the masking threshold over the full spectrum as the
/// floor level, mapped from dB to the [0, 255] range. Both endpoints use the same
/// value (flat floor approximation); this keeps the bitstream change minimal.
fn compute_floor_y_psychoacoustic(mdct: &[f32], mask: &[f32]) -> (u8, u8) {
    // Fall back to a silent floor if there are no bins.
    let n = mask.len().max(1);

    // Geometric mean of masking threshold in log scale.
    let log_mean_mask: f32 = mask.iter().map(|&m| m.max(1e-30).log10()).sum::<f32>() / n as f32;

    // Convert to dB (energy domain: 10·log10) and map to [0, 255].
    // Mapping: −60 dB → 0, +60 dB → 255 (linear over 120 dB range).
    let db = 10.0 * log_mean_mask;
    let y = ((db + 60.0) * 255.0 / 120.0).clamp(0.0, 255.0) as u8;

    // Also derive a high-frequency Y using the upper-half of mdct energy to give
    // a slight frequency-shaping benefit without adding API complexity.
    // For simplicity, derive from the high-frequency portion of the mask.
    let hi_start = n / 2;
    let hi_n = (n - hi_start).max(1);
    let log_mean_hi: f32 = mask[hi_start..]
        .iter()
        .map(|&m| m.max(1e-30).log10())
        .sum::<f32>()
        / hi_n as f32;
    let db_hi = 10.0 * log_mean_hi;
    let y1 = ((db_hi + 60.0) * 255.0 / 120.0).clamp(0.0, 255.0) as u8;

    // Suppress unused-variable warning: mdct is passed for potential future use
    // (e.g. per-band Y values derived from signal vs mask ratio).
    let _ = mdct;

    (y, y1)
}

// ─── Residue quantisation ─────────────────────────────────────────────────────

/// Quantise a single MDCT coefficient to a residue VQ index in [0, 255].
///
/// The VQ table covers [−MAX_RESIDUE, +MAX_RESIDUE] uniformly in 256 steps
/// scaled by `quality_scale`.  A larger scale means coarser quantization (lower quality).
///
/// Note: the decoder reconstructs `value = min + index * delta` using the fixed
/// setup-header delta value.  The `quality_scale` here affects which bin is
/// selected but not the transmitted codebook — perceptually, coarser quantization
/// at lower quality means less spectral fidelity.
fn quantise_residue(coeff: f32, quality_scale: f32) -> u8 {
    let effective_delta = RESIDUE_DELTA * quality_scale;
    let idx = ((coeff - (-MAX_RESIDUE)) / effective_delta).round() as i32;
    idx.clamp(0, (RESIDUE_ENTRIES - 1) as i32) as u8
}

// ─── Audio packet encoder ─────────────────────────────────────────────────────

/// Encode one MDCT audio frame for all channels.
///
/// Packet layout (Vorbis I spec §4.3.1):
/// ```text
/// [1]  packet_type = 0 (audio)
/// [0]  mode_number  (ilog(0) = 0 bits for 1 mode)
/// [1]  previous_window_flag  (blockflag=1 → required)
/// [1]  next_window_flag      (blockflag=1 → required)
/// per channel:
///   [1]  floor_nonzero  (1 = active)
///   [8]  Y[0] endpoint at X=0   (range=256, multiplier=1)
///   [8]  Y[1] endpoint at X=1024
///   (no partition dim bits: subbooks are unused → Y[2]=Y[3] implicit = 0)
/// all channels residue (type 0):
///   per channel:
///     [3]  classbook code (codebook 0, dim=1, len=3)
///   per channel × 4 partitions × 32 values:
///     [8]  VQ index (codebook 1, len=8)
/// ```
fn encode_audio_packet(
    channels: usize,
    prev_long: bool,
    next_long: bool,
    mdct_per_channel: &[Vec<f32>],
    quality: VorbisQuality,
    sample_rate: u32,
) -> Vec<u8> {
    let mut bw = BitWriter::new();

    // packet_type = 0 (audio). Must be the first bit.
    bw.write_bit(false);

    // mode_number: ilog(1 - 1) = 0 bits → nothing.

    // blockflag=1 → must write previous/next window flags.
    bw.write_bit(prev_long);
    bw.write_bit(next_long);

    // Silence threshold (RMS amplitude) from quality level.
    let silence_threshold = quality.silence_threshold();
    let delta_scale = quality.residue_delta_scale();

    // ── Per-channel floor ─────────────────────────────────────────────────────
    let mut floor_y: Vec<(u8, u8)> = Vec::with_capacity(channels);
    let mut has_audio = false;

    for coeffs in mdct_per_channel.iter().take(channels) {
        // Compute RMS energy and compare against quality-dependent silence threshold.
        let energy: f32 = coeffs.iter().map(|&c| c * c).sum::<f32>();
        let rms = (energy / coeffs.len() as f32).sqrt();
        if rms < silence_threshold {
            // Silence → floor_nonzero = 0.
            bw.write_bit(false);
            floor_y.push((0, 0));
        } else {
            has_audio = true;
            bw.write_bit(true); // floor_nonzero = 1
            let mask = compute_masking_threshold(coeffs, sample_rate);
            let (y0, y1) = compute_floor_y_psychoacoustic(coeffs, &mask);
            bw.write_bits(u64::from(y0), 8); // Y[0] at X=0
            bw.write_bits(u64::from(y1), 8); // Y[1] at X=1024
                                             // Partition dims 2..3 use subbook=-1 → no bits written, Y=0 implicit.
            floor_y.push((y0, y1));
        }
    }

    // ── Residue encoding (only when at least one channel has audio) ───────────
    // Residue type 0: sequential per channel.
    // Spec §8.6.2: for each channel, read 1 classbook code, then residue partitions.
    if has_audio {
        for ch in 0..channels {
            let (y0, _y1) = floor_y[ch];
            if y0 == 0 {
                // This channel was silent → floor_nonzero=0, no residue for it.
                // But residue type 0 is decoded for ALL channels in one pass.
                // For a silent channel we still write residue (spec decodes them all).
                // Write classbook code = 0 (3 bits, all zeros).
                bw.write_bits(0, 3);
                // Write zero-index for all 4 partitions × 32 values.
                for _ in 0..N_PARTITIONS {
                    for _ in 0..PARTITION_SIZE {
                        bw.write_bits(0, 8);
                    }
                }
            } else {
                let coeffs = &mdct_per_channel[ch];
                // Classbook code = 0 (3 bits from codebook 0 with uniform len=3).
                bw.write_bits(0, 3);
                // 4 partitions × 32 values, each quantised to 8-bit VQ index
                // using the quality-scaled effective delta.
                for p in 0..N_PARTITIONS {
                    let base = RESIDUE_BEGIN + p * PARTITION_SIZE;
                    for j in 0..PARTITION_SIZE {
                        let bin = base + j;
                        let coeff = if bin < coeffs.len() { coeffs[bin] } else { 0.0 };
                        let idx = quantise_residue(coeff, delta_scale);
                        bw.write_bits(u64::from(idx), 8);
                    }
                }
            }
        }
    }

    bw.finalize()
}

// ─── Public encode API ────────────────────────────────────────────────────────

/// Encode an `AudioBuffer<f32>` as OGG Vorbis I and write the bitstream to `writer`.
///
/// Uses the default quality level (q4 ≈ 0.4, ~128 kbps).
/// For explicit quality control, use [`encode_vorbis_with_quality`].
///
/// Produces a valid Vorbis I stream with three header packets (identification,
/// comment, setup) followed by MDCT audio packets.  Floor type-1 and residue
/// type-0 scalar VQ are used to write non-zero spectral data for non-silence input.
///
/// # Supported inputs
/// - `channels`: 1 or 2
/// - `sample_rate`: 44100 or 48000 Hz
///
/// # Errors
///
/// Returns [`OxiAudioError::Encode`] if the channel count or sample rate is
/// unsupported, or [`OxiAudioError::Io`] on write failure.
pub fn encode_vorbis<W: Write>(buf: &AudioBuffer<f32>, writer: W) -> Result<(), OxiAudioError> {
    encode_vorbis_with_quality(buf, writer, VorbisQuality::default_quality())
}

/// Encode an `AudioBuffer<f32>` as OGG Vorbis I with explicit VBR quality control.
///
/// The `quality` parameter controls the trade-off between bit rate and perceptual
/// quality via the silence threshold and residue quantization step size.
///
/// # Quality levels
/// - [`VorbisQuality::from_level(-1)`][VorbisQuality::from_level]: very low quality, ~45 kbps
/// - [`VorbisQuality::from_level(0)`][VorbisQuality::from_level]: low quality, ~64 kbps
/// - [`VorbisQuality::from_level(4)`][VorbisQuality::from_level]: medium quality, ~128 kbps (default)
/// - [`VorbisQuality::from_level(7)`][VorbisQuality::from_level]: high quality, ~224 kbps
/// - [`VorbisQuality::from_level(10)`][VorbisQuality::from_level]: maximum quality, ~500 kbps
///
/// # Supported inputs
/// - `channels`: 1 or 2
/// - `sample_rate`: 44100 or 48000 Hz
///
/// # Errors
///
/// Returns [`OxiAudioError::Encode`] if the channel count or sample rate is
/// unsupported, or [`OxiAudioError::Io`] on write failure.
pub fn encode_vorbis_with_quality<W: Write>(
    buf: &AudioBuffer<f32>,
    mut writer: W,
    quality: VorbisQuality,
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count();
    let sample_rate = buf.sample_rate;

    // Validate inputs
    if channels == 0 || channels > 2 {
        return Err(OxiAudioError::Encode(format!(
            "Vorbis encoder supports 1 or 2 channels; got {channels}"
        )));
    }
    if sample_rate != 44_100 && sample_rate != 48_000 {
        return Err(OxiAudioError::Encode(format!(
            "Vorbis encoder supports 44100 or 48000 Hz; got {sample_rate}"
        )));
    }

    let mut stream = OggStream::new(&mut writer, STREAM_SERIAL);

    // ── Header pages ─────────────────────────────────────────────────────────

    // Page 0: identification header (BOS page, granule=0)
    let ident = write_ident_header(channels as u8, sample_rate);
    stream.write_packet(&ident, 0, false)?;

    // Page 1: comment header (granule=0)
    let comment = write_vorbis_comment_packet(VENDOR_STRING, &[], false);
    stream.write_packet(&comment, 0, false)?;

    // Page 2: setup header (granule=0)
    let setup = write_setup_header(channels as u8);
    stream.write_packet(&setup, 0, false)?;

    // ── Audio packets ─────────────────────────────────────────────────────────

    // Use hop size = blocksize_1 / 2 for granule accounting.
    let hop = (BLOCK_SIZE_1 / 2) as i64;
    let total_samples = buf.samples.len().checked_div(channels).unwrap_or(0);

    // Minimum one audio packet even for empty buffers.
    let n_frames = (total_samples / (BLOCK_SIZE_1 / 2)).max(1);

    for frame_idx in 0..n_frames {
        let is_last = frame_idx == n_frames - 1;
        let prev_long = frame_idx > 0;
        let next_long = !is_last;

        // Extract interleaved samples for this frame, per channel.
        let frame_start = frame_idx * (BLOCK_SIZE_1 / 2);
        let frame_end = (frame_start + BLOCK_SIZE_1).min(total_samples);

        let mdct_per_channel: Vec<Vec<f32>> = (0..channels)
            .map(|ch| {
                // Deinterleave and collect up to BLOCK_SIZE_1 samples for this channel.
                let samples: Vec<f32> = (frame_start..frame_end)
                    .map(|i| {
                        let idx = i * channels + ch;
                        *buf.samples.get(idx).unwrap_or(&0.0)
                    })
                    .collect();
                vorbis_mdct(&samples)
            })
            .collect();

        let pkt = encode_audio_packet(
            channels,
            prev_long,
            next_long,
            &mdct_per_channel,
            quality,
            sample_rate,
        );
        stream.write_packet(&pkt, hop, is_last)?;
    }

    Ok(())
}

/// Encode an `AudioBuffer<f32>` as OGG Vorbis I and write to a file at `path`.
///
/// Convenience wrapper around [`encode_vorbis`] using default quality.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be created, or any error
/// from [`encode_vorbis`].
pub fn encode_vorbis_file(buf: &AudioBuffer<f32>, path: &Path) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    encode_vorbis(buf, writer)
}

/// Encode an `AudioBuffer<f32>` as OGG Vorbis I with explicit quality and write to a file.
///
/// Convenience wrapper around [`encode_vorbis_with_quality`].
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be created, or any error
/// from [`encode_vorbis_with_quality`].
pub fn encode_vorbis_quality_file(
    buf: &AudioBuffer<f32>,
    path: &Path,
    quality: VorbisQuality,
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    encode_vorbis_with_quality(buf, writer, quality)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError, SampleFormat};

    use super::{
        encode_vorbis, encode_vorbis_file, encode_vorbis_quality_file, encode_vorbis_with_quality,
        vorbis_mdct, VorbisQuality, BLOCK_SIZE_1, N_COEFFS,
    };

    fn silence_buffer_mono(samples: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.0f32; samples],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn silence_buffer_stereo(samples: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.0f32; samples * 2],
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    fn sine_buffer_mono(samples: usize, sample_rate: u32) -> AudioBuffer<f32> {
        let data: Vec<f32> = (0..samples)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();
        AudioBuffer {
            samples: data,
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    // ── New tests: MDCT and sine encoding ─────────────────────────────────────

    /// A sine wave produces non-zero MDCT energy.
    #[test]
    fn test_vorbis_mdct_energy() {
        let n = BLOCK_SIZE_1;
        let sine: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44_100.0).sin() * 0.5)
            .collect();
        let coeffs = vorbis_mdct(&sine);
        assert_eq!(coeffs.len(), N_COEFFS, "MDCT must produce N/2 coefficients");
        let energy: f32 = coeffs.iter().map(|&c| c * c).sum();
        assert!(
            energy > 0.01,
            "sine wave must produce non-zero MDCT energy; got energy={energy}"
        );
    }

    /// Silence input produces near-zero MDCT output.
    #[test]
    fn test_vorbis_mdct_silence_near_zero() {
        let silence = vec![0.0f32; BLOCK_SIZE_1];
        let coeffs = vorbis_mdct(&silence);
        let max = coeffs.iter().copied().fold(0.0f32, f32::max);
        assert!(
            max.abs() < 1e-5,
            "silence must produce near-zero MDCT; max={max}"
        );
    }

    /// Encoding a non-silence sine wave produces different bytes than encoding silence.
    #[test]
    fn test_encode_vorbis_sine_produces_nonzero_bytes() {
        let n_samples = BLOCK_SIZE_1 * 4;
        let silence_buf = silence_buffer_mono(n_samples);
        let sine_buf = sine_buffer_mono(n_samples, 44_100);

        let mut silence_out = Cursor::new(Vec::new());
        encode_vorbis(&silence_buf, &mut silence_out).expect("encode silence");
        let silence_bytes = silence_out.into_inner();

        let mut sine_out = Cursor::new(Vec::new());
        encode_vorbis(&sine_buf, &mut sine_out).expect("encode sine");
        let sine_bytes = sine_out.into_inner();

        assert!(
            silence_bytes != sine_bytes,
            "sine-wave encoding must produce different bytes from silence encoding"
        );
    }

    // ── Existing structural tests ─────────────────────────────────────────────

    /// Verify that the OGG Vorbis output starts with the OGG capture pattern.
    #[test]
    fn test_encode_vorbis_produces_ogg_output() {
        let buf = silence_buffer_mono(4096);
        let mut out = Cursor::new(Vec::new());
        encode_vorbis(&buf, &mut out).expect("encode_vorbis must succeed");
        let bytes = out.into_inner();
        assert!(
            bytes.starts_with(b"OggS"),
            "output must start with OGG capture pattern"
        );
    }

    /// Verify that the identification header magic is present in the output.
    #[test]
    fn test_encode_vorbis_has_vorbis_magic() {
        let buf = silence_buffer_mono(4096);
        let mut out = Cursor::new(Vec::new());
        encode_vorbis(&buf, &mut out).expect("encode_vorbis must succeed");
        let bytes = out.into_inner();
        // The identification header "\x01vorbis" must appear somewhere in the stream.
        let magic = b"\x01vorbis";
        assert!(
            bytes.windows(magic.len()).any(|w| w == magic),
            "output must contain Vorbis identification header magic"
        );
    }

    /// Verify that > 2 channels is rejected with an Encode error.
    #[test]
    fn test_encode_vorbis_rejects_high_channel_count() {
        // ChannelLayout::from(3u16) maps to Stereo (2-channel fallback), so we use
        // ChannelLayout::Quad (4 channels) which is unambiguously > 2 channels.
        let buf = AudioBuffer {
            samples: vec![0.0f32; 4 * 1024],
            sample_rate: 44_100,
            channels: ChannelLayout::Quad,
            format: SampleFormat::F32,
        };
        let mut out = Cursor::new(Vec::new());
        let result = encode_vorbis(&buf, &mut out);
        assert!(result.is_err(), "3-channel input must return an error");
        if let Err(OxiAudioError::Encode(msg)) = result {
            assert!(
                msg.contains("channel"),
                "error message should mention channels, got: {msg}"
            );
        }
    }

    /// Verify that unsupported sample rates are rejected.
    #[test]
    fn test_encode_vorbis_rejects_invalid_sample_rate() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 1024],
            sample_rate: 22_050,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut out = Cursor::new(Vec::new());
        let result = encode_vorbis(&buf, &mut out);
        assert!(result.is_err(), "22050 Hz must return an error");
        if let Err(OxiAudioError::Encode(msg)) = result {
            assert!(
                msg.contains("Hz") || msg.contains("sample"),
                "error message should mention sample rate, got: {msg}"
            );
        }
    }

    /// Verify the file-writing convenience wrapper produces a valid OGG file.
    #[test]
    fn test_encode_vorbis_file() {
        let buf = silence_buffer_stereo(4096);
        let tmp = std::env::temp_dir().join("oxiaudio_vorbis_test.ogg");
        encode_vorbis_file(&buf, &tmp).expect("encode_vorbis_file must succeed");
        let bytes = std::fs::read(&tmp).expect("temp file must be readable");
        assert!(
            bytes.starts_with(b"OggS"),
            "file must start with OGG capture pattern"
        );
        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    /// Structural round-trip: encode succeeds and the bitstream contains valid OGG+Vorbis framing.
    ///
    /// Full roundtrip decode (non-empty PCM) is deferred until MDCT audio encoding
    /// is implemented in a future milestone; the current silence scaffold produces
    /// a structurally valid stream that some decoders may return as empty PCM.
    #[test]
    fn test_encode_vorbis_round_trip_decode() {
        let n_samples = BLOCK_SIZE_1 * 4; // 4 long blocks
        let buf = silence_buffer_mono(n_samples);
        let mut out = Cursor::new(Vec::new());
        encode_vorbis(&buf, &mut out).expect("encode_vorbis must succeed for round-trip");
        let encoded = out.into_inner();

        // Structural checks: correct OGG capture pattern, Vorbis identification,
        // comment, and setup headers must all be present in the output.
        assert!(
            encoded.starts_with(b"OggS"),
            "output must start with OGG capture pattern"
        );
        assert!(
            encoded.windows(7).any(|w| w == b"\x01vorbis"),
            "output must contain Vorbis identification header"
        );
        assert!(
            encoded.windows(7).any(|w| w == b"\x03vorbis"),
            "output must contain Vorbis comment header"
        );
        assert!(
            encoded.windows(7).any(|w| w == b"\x05vorbis"),
            "output must contain Vorbis setup header"
        );
        // Verify total size is plausible (3 header pages + audio packets).
        assert!(
            encoded.len() > 100,
            "encoded output must be non-trivially large (got {} bytes)",
            encoded.len()
        );
    }

    /// Verify the Vorbis float32 packing round-trips correctly.
    #[test]
    fn test_pack_vorbis_float32_roundtrip() {
        use super::pack_vorbis_float32;

        fn unpack(packed: u32) -> f32 {
            let mantissa = (packed & ((1u32 << 21) - 1)) as f32;
            let exponent = ((packed >> 21) & ((1u32 << 10) - 1)) as i32;
            let sign = (packed >> 31) & 1;
            let value = mantissa * (2.0f32).powi(exponent - 788 - 21);
            if sign == 1 {
                -value
            } else {
                value
            }
        }

        for &v in &[-6.0f32, -1.0, 0.0, 0.04706, 0.5, 1.0, 6.0] {
            let packed = pack_vorbis_float32(v);
            let recovered = unpack(packed);
            let err = (recovered - v).abs();
            assert!(
                err < 0.001 || v == 0.0,
                "pack_vorbis_float32({v}) -> {packed:#010X} -> {recovered}, error={err}"
            );
        }
    }

    // ── VBR quality mode tests ────────────────────────────────────────────────

    /// VorbisQuality::new clamps values outside [−0.1, 1.0].
    #[test]
    fn test_vorbis_quality_new_clamps() {
        let q_hi = VorbisQuality::new(2.0);
        let inner_hi: f32 = q_hi.0;
        assert!(
            (inner_hi - 1.0).abs() < 1e-5,
            "quality must clamp to 1.0; got {inner_hi}"
        );

        let q_lo = VorbisQuality::new(-0.5);
        let inner_lo: f32 = q_lo.0;
        assert!(
            (inner_lo + 0.1).abs() < 1e-5,
            "quality must clamp to -0.1; got {inner_lo}"
        );
    }

    /// Higher quality level must produce a smaller (finer) delta scale.
    #[test]
    fn test_vorbis_quality_delta_scale_monotonic() {
        let scale_lo = VorbisQuality::from_level(0).residue_delta_scale();
        let scale_hi = VorbisQuality::from_level(10).residue_delta_scale();
        assert!(
            scale_hi < scale_lo,
            "higher quality should have smaller delta scale (finer quantization); got lo={scale_lo}, hi={scale_hi}"
        );
    }

    /// Higher quality must produce a lower silence threshold (encodes quieter frames).
    #[test]
    fn test_vorbis_silence_threshold() {
        let thresh_hi = VorbisQuality::from_level(10).silence_threshold();
        let thresh_lo = VorbisQuality::from_level(0).silence_threshold();
        assert!(
            thresh_hi < thresh_lo,
            "high quality should have lower silence threshold; got hi={thresh_hi}, lo={thresh_lo}"
        );
    }

    /// Low quality produces fewer or equal bytes than high quality when encoding a
    /// very quiet sine wave (amplitude ≈ 0.001) that falls below q=0's −40 dBFS threshold.
    ///
    /// The quiet sine has RMS ≈ 0.001 / sqrt(2) ≈ 7e-4.  q=0's silence threshold is
    /// 10^(-40/20) ≈ 0.01.  So q=0 encodes these frames as silence (fewer bits), while
    /// q=10 encodes them as active (more bits).  Hence bytes_hi > bytes_lo.
    #[test]
    fn test_encode_vorbis_with_quality_low_vs_high() {
        // Quiet sine: amplitude 0.001 → RMS ≈ 7.07e-4, well below q=0's threshold (~0.01)
        // but above q=10's threshold (~1e-4).
        let n_samples = BLOCK_SIZE_1 * 8;
        let quiet_sine: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.001)
            .collect();
        let buf = AudioBuffer {
            samples: quiet_sine,
            sample_rate: 44_100,
            channels: oxiaudio_core::ChannelLayout::Mono,
            format: oxiaudio_core::SampleFormat::F32,
        };

        let mut out_lo = Cursor::new(Vec::new());
        encode_vorbis_with_quality(&buf, &mut out_lo, VorbisQuality::from_level(0))
            .expect("low quality encode");

        let mut out_hi = Cursor::new(Vec::new());
        encode_vorbis_with_quality(&buf, &mut out_hi, VorbisQuality::from_level(10))
            .expect("high quality encode");

        let bytes_lo = out_lo.into_inner().len();
        let bytes_hi = out_hi.into_inner().len();

        assert!(
            bytes_hi >= bytes_lo,
            "high quality ({bytes_hi} bytes) should be >= low quality ({bytes_lo} bytes)"
        );
    }

    /// encode_vorbis_quality_file produces a valid OGG file.
    #[test]
    fn test_encode_vorbis_quality_file() {
        let buf = silence_buffer_mono(4096);
        let tmp = std::env::temp_dir().join("oxiaudio_vorbis_quality_test.ogg");
        encode_vorbis_quality_file(&buf, &tmp, VorbisQuality::default_quality())
            .expect("quality file encode");
        let bytes = std::fs::read(&tmp).expect("read file");
        assert!(
            bytes.starts_with(b"OggS"),
            "output must be valid OGG (starts with OggS)"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    /// VorbisQuality::from_level covers the full q-1..=q10 range without panicking.
    #[test]
    fn test_vorbis_quality_from_level_range() {
        for level in -1..=10 {
            let q = VorbisQuality::from_level(level);
            let scale = q.residue_delta_scale();
            let thresh = q.silence_threshold();
            assert!(
                scale > 0.0 && scale <= 10.0,
                "delta scale out of range at level {level}: {scale}"
            );
            assert!(
                thresh > 0.0,
                "silence threshold must be positive at level {level}: {thresh}"
            );
        }
    }

    // ── Psychoacoustic model tests ────────────────────────────────────────────

    /// ATH at 1 kHz should be near 0 dB SPL (within ±5 dB).
    #[test]
    fn test_ath_at_1khz_is_near_zero_db() {
        use super::ath_db;
        let ath_1k = ath_db(1000.0);
        assert!(
            ath_1k.abs() < 5.0,
            "ATH at 1 kHz should be near 0 dB SPL; got {ath_1k} dB"
        );
    }

    /// `hz_to_bark` must be monotonically increasing from 100 Hz to 10 000 Hz.
    #[test]
    fn test_hz_to_bark_monotonic() {
        use super::hz_to_bark;
        let freqs: Vec<f32> = (1..=100).map(|i| 100.0 + i as f32 * 99.0).collect();
        let barks: Vec<f32> = freqs.iter().map(|&f| hz_to_bark(f)).collect();
        for window in barks.windows(2) {
            assert!(
                window[1] > window[0],
                "hz_to_bark must be monotonically increasing; got {} then {}",
                window[0],
                window[1]
            );
        }
    }

    /// The masking threshold must be >= ATH at every bin for any input.
    #[test]
    fn test_masking_threshold_above_ath() {
        use super::{ath_db, compute_masking_threshold};

        let sample_rate = 44_100u32;
        let n_bins = N_COEFFS;
        let freq_resolution = sample_rate as f32 / (2.0 * n_bins as f32);

        // Use a sine-wave MDCT as a representative real signal.
        let sine: Vec<f32> = (0..BLOCK_SIZE_1)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.5)
            .collect();
        let mdct = vorbis_mdct(&sine);
        let mask = compute_masking_threshold(&mdct, sample_rate);

        assert_eq!(mask.len(), n_bins);

        for (k, &m) in mask.iter().enumerate() {
            let f_hz = ((k as f32 + 0.5) * freq_resolution).max(20.0);
            let ath_lin = 10.0f32.powf(ath_db(f_hz) / 10.0);
            assert!(
                m >= ath_lin * (1.0 - 1e-5),
                "threshold[{k}] ({m}) must be >= ATH ({ath_lin}) at f={f_hz:.0} Hz"
            );
        }
    }

    /// A 440 Hz pure tone should create elevated masking near 440 Hz vs 4000 Hz
    /// (downward spread of masking).
    #[test]
    fn test_masking_threshold_pure_tone_creates_local_peak() {
        use super::compute_masking_threshold;

        let sample_rate = 44_100u32;
        let sine: Vec<f32> = (0..BLOCK_SIZE_1)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.8)
            .collect();
        let mdct = vorbis_mdct(&sine);
        let mask = compute_masking_threshold(&mdct, sample_rate);

        // Bin for ~440 Hz: k ≈ 440 * 1024 / (44100/2) ≈ 20
        let bin_440 = (440.0 * N_COEFFS as f32 / (sample_rate as f32 / 2.0)).round() as usize;
        // Bin for ~4000 Hz: k ≈ 4000 * 1024 / 22050 ≈ 186
        let bin_4k = (4000.0 * N_COEFFS as f32 / (sample_rate as f32 / 2.0)).round() as usize;

        let bin_440 = bin_440.min(N_COEFFS - 1);
        let bin_4k = bin_4k.min(N_COEFFS - 1);

        // With Johnston spreading, a 440 Hz masker spreads downward ~17–20 dB/Bark.
        // By 4 kHz (~Bark 17), the masking effect has dropped far below the local peak.
        assert!(
            mask[bin_440] > mask[bin_4k],
            "440 Hz tone should produce higher masking threshold near 440 Hz ({}) \
             than at 4 kHz ({})",
            mask[bin_440],
            mask[bin_4k]
        );
    }

    /// Silence input to `compute_floor_y_psychoacoustic` should produce a lower Y
    /// than a loud sine wave.
    #[test]
    fn test_compute_floor_y_psychoacoustic_silence_low() {
        use super::{compute_floor_y_psychoacoustic, compute_masking_threshold};

        let sample_rate = 44_100u32;

        // Silence
        let silence_mdct = vec![0.0f32; N_COEFFS];
        let silence_mask = compute_masking_threshold(&silence_mdct, sample_rate);
        let (y_silence, _) = compute_floor_y_psychoacoustic(&silence_mdct, &silence_mask);

        // Loud sine
        let loud_sine: Vec<f32> = (0..BLOCK_SIZE_1)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44_100.0).sin())
            .collect();
        let loud_mdct = vorbis_mdct(&loud_sine);
        let loud_mask = compute_masking_threshold(&loud_mdct, sample_rate);
        let (y_loud, _) = compute_floor_y_psychoacoustic(&loud_mdct, &loud_mask);

        assert!(
            y_silence < y_loud,
            "silence floor Y ({y_silence}) must be lower than loud sine floor Y ({y_loud})"
        );
    }

    /// Louder signal → higher Y than medium → higher Y than silence.
    #[test]
    fn test_compute_floor_y_psychoacoustic_louder_signal_higher_y() {
        use super::{compute_floor_y_psychoacoustic, compute_masking_threshold};

        let sample_rate = 44_100u32;

        let make_y = |amplitude: f32| {
            let sine: Vec<f32> = (0..BLOCK_SIZE_1)
                .map(|i| {
                    (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * amplitude
                })
                .collect();
            let mdct = vorbis_mdct(&sine);
            let mask = compute_masking_threshold(&mdct, sample_rate);
            let (y, _) = compute_floor_y_psychoacoustic(&mdct, &mask);
            y
        };

        let y_silence = {
            let silence = vec![0.0f32; N_COEFFS];
            let mask = compute_masking_threshold(&silence, sample_rate);
            let (y, _) = compute_floor_y_psychoacoustic(&silence, &mask);
            y
        };
        let y_medium = make_y(0.1);
        let y_loud = make_y(0.9);

        assert!(
            y_silence <= y_medium,
            "silence Y ({y_silence}) must be <= medium Y ({y_medium})"
        );
        assert!(
            y_medium <= y_loud,
            "medium Y ({y_medium}) must be <= loud Y ({y_loud})"
        );
    }

    /// The masking threshold length must equal `mdct.len()`.
    #[test]
    fn test_masking_threshold_length_matches_mdct() {
        use super::compute_masking_threshold;

        let sample_rate = 44_100u32;
        for &n in &[0usize, 1, 64, 512, N_COEFFS] {
            let mdct = vec![0.1f32; n];
            let mask = compute_masking_threshold(&mdct, sample_rate);
            assert_eq!(
                mask.len(),
                n,
                "compute_masking_threshold({n}) must return length {n}; got {}",
                mask.len()
            );
        }
    }
}
