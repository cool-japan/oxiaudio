//! Pure-Rust OGG Vorbis I encoder with MDCT-based audio encoding.
//!
//! Produces a valid Vorbis I bitstream (three header packets + audio packets)
//! encapsulated in OGG pages.  Audio packets use MDCT forward-transform coefficients
//! quantised through a Floor type-1 curve and Residue type-0 scalar VQ.
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
const VENDOR_STRING: &str = concat!("OxiAudio ", env!("CARGO_PKG_VERSION"), " Vorbis encoder");

/// Number of MDCT output coefficients (BLOCK_SIZE_1 / 2).
const N_COEFFS: usize = BLOCK_SIZE_1 / 2; // 1024

/// Residue encoding range: min = -0.5, delta = 1/255.
const RESIDUE_MIN: f32 = -0.5;
const RESIDUE_DELTA: f32 = 1.0 / 255.0;

/// Number of residue quantisation steps (VQ entries).
const RESIDUE_ENTRIES: usize = 256;

/// Residue begin/end for active encoding (first 128 MDCT bins).
const RESIDUE_BEGIN: usize = 0;
const RESIDUE_END: usize = 128;

/// Number of residue values per partition.
const PARTITION_SIZE: usize = 32;

/// Number of partitions in the active residue range.
const N_PARTITIONS: usize = (RESIDUE_END - RESIDUE_BEGIN) / PARTITION_SIZE; // 4

/// Floor1 X-post positions (explicit; endpoints 0 and 1024 are implicit).
/// 3 partitions × 2 dims each = 6 explicit posts.
const FLOOR_X_POSTS: [u32; 6] = [32, 64, 128, 256, 512, 640];

/// Number of floor1 explicit X posts.
const N_FLOOR_POSTS: usize = FLOOR_X_POSTS.len(); // 6

/// Full floor X list: [0] + FLOOR_X_POSTS + [1024] (but note: decoder adds 1<<rangebits automatically)
/// The decoder inserts x=0 first and x=1<<rangebits=1024 second, then reads explicit posts.
/// So full_x = [0, 1024, 32, 64, 128, 256, 512, 640]
/// Total floor Y values = 2 + 6 = 8.
const N_FLOOR_Y: usize = 2 + N_FLOOR_POSTS; // 8

// ─── FLOOR1_INVERSE_DB_TABLE ──────────────────────────────────────────────────

/// Vorbis I floor1 inverse dB table (spec §10.1).
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
const FLOOR1_INVERSE_DB_TABLE: [f32; 256] = [
    1.0649863e-07, 1.1341951e-07, 1.2079015e-07, 1.2863978e-07,
    1.3699951e-07, 1.4590251e-07, 1.5538408e-07, 1.6548181e-07,
    1.7623575e-07, 1.8768855e-07, 1.9988561e-07, 2.1287530e-07,
    2.2670913e-07, 2.4144197e-07, 2.5713223e-07, 2.7384213e-07,
    2.9163793e-07, 3.1059021e-07, 3.3077411e-07, 3.5226968e-07,
    3.7516214e-07, 3.9954229e-07, 4.2550680e-07, 4.5315863e-07,
    4.8260743e-07, 5.1396998e-07, 5.4737065e-07, 5.8294187e-07,
    6.2082472e-07, 6.6116941e-07, 7.0413592e-07, 7.4989464e-07,
    7.9862701e-07, 8.5052630e-07, 9.0579828e-07, 9.6466216e-07,
    1.0273513e-06, 1.0941144e-06, 1.1652161e-06, 1.2409384e-06,
    1.3215816e-06, 1.4074654e-06, 1.4989305e-06, 1.5963394e-06,
    1.7000785e-06, 1.8105592e-06, 1.9282195e-06, 2.0535261e-06,
    2.1869758e-06, 2.3290978e-06, 2.4804557e-06, 2.6416497e-06,
    2.8133190e-06, 2.9961443e-06, 3.1908506e-06, 3.3982101e-06,
    3.6190449e-06, 3.8542308e-06, 4.1047004e-06, 4.3714470e-06,
    4.6555282e-06, 4.9580707e-06, 5.2802740e-06, 5.6234160e-06,
    5.9888572e-06, 6.3780469e-06, 6.7925283e-06, 7.2339451e-06,
    7.7040476e-06, 8.2047000e-06, 8.7378876e-06, 9.3057248e-06,
    9.9104632e-06, 1.0554501e-05, 1.1240392e-05, 1.1970856e-05,
    1.2748789e-05, 1.3577278e-05, 1.4459606e-05, 1.5399272e-05,
    1.6400004e-05, 1.7465768e-05, 1.8600792e-05, 1.9809576e-05,
    2.1096914e-05, 2.2467911e-05, 2.3928002e-05, 2.5482978e-05,
    2.7139006e-05, 2.8902651e-05, 3.0780908e-05, 3.2781225e-05,
    3.4911534e-05, 3.7180282e-05, 3.9596466e-05, 4.2169667e-05,
    4.4910090e-05, 4.7828601e-05, 5.0936773e-05, 5.4246931e-05,
    5.7772202e-05, 6.1526565e-05, 6.5524908e-05, 6.9783085e-05,
    7.4317983e-05, 7.9147585e-05, 8.4291040e-05, 8.9768747e-05,
    9.5602426e-05, 0.00010181521, 0.00010843174, 0.00011547824,
    0.00012298267, 0.00013097477, 0.00013948625, 0.00014855085,
    0.00015820453, 0.00016848555, 0.00017943469, 0.00019109536,
    0.00020351382, 0.00021673929, 0.00023082423, 0.00024582449,
    0.00026179955, 0.00027881276, 0.00029693158, 0.00031622787,
    0.00033677814, 0.00035866388, 0.00038197188, 0.00040679456,
    0.00043323036, 0.00046138411, 0.00049136745, 0.00052329927,
    0.00055730621, 0.00059352311, 0.00063209358, 0.00067317058,
    0.00071691700, 0.00076350630, 0.00081312324, 0.00086596457,
    0.00092223983, 0.00098217216, 0.0010459992,  0.0011139742,
    0.0011863665,  0.0012634633,  0.0013455702,  0.0014330129,
    0.0015261382,  0.0016253153,  0.0017309374,  0.0018434235,
    0.0019632195,  0.0020908006,  0.0022266726,  0.0023713743,
    0.0025254795,  0.0026895994,  0.0028643847,  0.0030505286,
    0.0032487691,  0.0034598925,  0.0036847358,  0.0039241906,
    0.0041792066,  0.0044507950,  0.0047400328,  0.0050480668,
    0.0053761186,  0.0057254891,  0.0060975636,  0.0064938176,
    0.0069158225,  0.0073652516,  0.0078438871,  0.0083536271,
    0.0088964928,  0.009474637,   0.010090352,   0.010746080,
    0.011444421,   0.012188144,   0.012980198,   0.013823725,
    0.014722068,   0.015678791,   0.016697687,   0.017782797,
    0.018938423,   0.020169149,   0.021479854,   0.022875735,
    0.024362330,   0.025945531,   0.027631618,   0.029427276,
    0.031339626,   0.033376252,   0.035545228,   0.037855157,
    0.040315199,   0.042935108,   0.045725273,   0.048696758,
    0.051861348,   0.055231591,   0.058820850,   0.062643361,
    0.066714279,   0.071049749,   0.075666962,   0.080584227,
    0.085821044,   0.091398179,   0.097337747,   0.10366330,
    0.11039993,    0.11757434,    0.12521498,    0.13335215,
    0.14201813,    0.15124727,    0.16107617,    0.17154380,
    0.18269168,    0.19456402,    0.20720788,    0.22067342,
    0.23501402,    0.25028656,    0.26655159,    0.28387361,
    0.30232132,    0.32196786,    0.34289114,    0.36517414,
    0.38890521,    0.41417847,    0.44109412,    0.46975890,
    0.50028648,    0.53279791,    0.56742212,    0.60429640,
    0.64356699,    0.68538959,    0.72993007,    0.77736504,
    0.82788260,    0.88168307,    0.9389798,     1.0,
];

// ─── BitWriter ────────────────────────────────────────────────────────────────

/// LSB-first (little-endian) bit-packer for Vorbis packet fields.
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

// ─── Vorbis float32 packing ───────────────────────────────────────────────────

/// Encode a scalar `f32` into the Vorbis packed-float32 format (§9.2.5).
fn pack_vorbis_float32(value: f32) -> u32 {
    if value == 0.0 {
        return 0;
    }
    let sign: u32 = if value < 0.0 { 1 } else { 0 };
    let abs_val = value.abs();
    let raw_exp = abs_val.log2().floor() as i32 + 1 + 788;
    let raw_exp = raw_exp.clamp(0, 1023) as u32;
    let shift = (raw_exp as i32) - 788 - 21;
    let mantissa_f = if shift >= 0 {
        abs_val / (2.0f32).powi(shift)
    } else {
        abs_val * (2.0f32).powi(-shift)
    };
    let mantissa = (mantissa_f.round() as u32).min((1u32 << 21) - 1);
    (sign << 31) | (raw_exp << 21) | mantissa
}

// ─── Canonical codewords ──────────────────────────────────────────────────────

/// Port of symphonia's `synthesize_codewords` algorithm.
///
/// Given a list of code lengths, assigns canonical Huffman codewords.
/// For a uniform-length book (all lengths equal), codeword[i] == i.
///
/// Returns a `Vec<u32>` parallel to `lens` (including zero-length entries
/// which get placeholder 0 — callers must not use these).
fn canonical_codewords(lens: &[u8]) -> Vec<u32> {
    let mut codewords = Vec::with_capacity(lens.len());
    let mut next_codeword = [0u32; 33];

    for &len in lens {
        if len == 0 {
            codewords.push(0u32);
            continue;
        }

        let codeword_len = usize::from(len);
        let codeword = next_codeword[codeword_len];

        // Update the next-codeword table by backtracking from N towards 1.
        for i in (1..=codeword_len).rev() {
            if next_codeword[i] & 1 == 1 {
                if i == 1 {
                    next_codeword[1] += 1;
                } else {
                    next_codeword[i] = next_codeword[i - 1] << 1;
                }
                break;
            }
            next_codeword[i] += 1;
        }

        // Propagate branch skipping downward.
        let branch = next_codeword[codeword_len];
        for (i, next) in next_codeword[codeword_len..].iter_mut().enumerate().skip(1) {
            if *next == codeword << i {
                *next = branch << i;
            } else {
                break;
            }
        }

        codewords.push(codeword);
    }

    codewords
}

// ─── Codebook definitions ─────────────────────────────────────────────────────
//
// Book 0: Floor1 class codebook: dim=1, entries=4, uniform len=2, lookup_type=0
// Book 1: Floor1 value codebook: dim=1, entries=256, uniform len=8, lookup_type=0
// Book 2: Residue classbook:     dim=1, entries=4,   uniform len=2, lookup_type=0
// Book 3: Residue VQ book:       dim=1, entries=256, uniform len=8, lookup_type=1

/// Write one codebook to the bit-writer.
///
/// # Parameters
/// * `bw`        — bit writer
/// * `dim`       — codebook dimensions
/// * `entries`   — number of entries
/// * `unif_len`  — uniform codeword length (all entries use this length)
/// * `lookup_type` — 0 = no VQ, 1 = scalar VQ
/// * `vq_min`    — VQ minimum value (only used if lookup_type=1)
/// * `vq_delta`  — VQ delta value (only used if lookup_type=1)
/// * `vq_value_bits` — VQ value bits (only used if lookup_type=1)
#[allow(clippy::too_many_arguments)]
fn write_codebook(
    bw: &mut BitWriter,
    dim: u16,
    entries: u32,
    unif_len: u8,
    lookup_type: u8,
    vq_min: f32,
    vq_delta: f32,
    vq_value_bits: u8,
) {
    // Sync word: 0x564342 written 24-bit LSB-first
    bw.write_bits(0x564342, 24);
    // dimensions (16 bits)
    bw.write_bits(u64::from(dim), 16);
    // entries (24 bits)
    bw.write_bits(u64::from(entries), 24);
    // ordered flag = 0
    bw.write_bit(false);
    // sparse flag = 0 (dense: all entries used)
    bw.write_bit(false);
    // For each entry: write length-1 as 5 bits
    for _ in 0..entries {
        bw.write_bits(u64::from(unif_len - 1), 5);
    }
    // lookup_type (4 bits)
    bw.write_bits(u64::from(lookup_type), 4);
    if lookup_type == 1 {
        // min_float (32 bits packed Vorbis float)
        bw.write_bits(u64::from(pack_vorbis_float32(vq_min)), 32);
        // delta_float (32 bits packed Vorbis float)
        bw.write_bits(u64::from(pack_vorbis_float32(vq_delta)), 32);
        // value_bits - 1 (4 bits)
        bw.write_bits(u64::from(vq_value_bits - 1), 4);
        // sequence_p = 0
        bw.write_bit(false);
        // multiplicands: entries values, each vq_value_bits wide
        for i in 0u32..entries {
            bw.write_bits(u64::from(i), vq_value_bits);
        }
    }
}

// ─── Identification header ────────────────────────────────────────────────────

/// Build the Vorbis identification header packet (30 bytes).
fn write_ident_header(channels: u8, sample_rate: u32) -> Vec<u8> {
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
/// Codebook set (4 books, indices 0-3):
///   Book 0: Floor1 class codebook  — dim=1, entries=4,   unif_len=2, lookup=0
///   Book 1: Floor1 value codebook  — dim=1, entries=256, unif_len=8, lookup=0
///   Book 2: Residue classbook      — dim=1, entries=4,   unif_len=2, lookup=0
///   Book 3: Residue VQ book        — dim=1, entries=256, unif_len=8, lookup=1
///
/// Floor1: 3 partitions, class 0 (dim=2, subclass_bits=1, mainbook=0),
///   subbook[0]=0(unused), subbook[1]=2(=book1+1), multiplier=1, rangebits=10.
///   X posts: 0(implicit), 1024(implicit), 32, 64, 128, 256, 512, 640.
///
/// Residue type 0: begin=0, end=128, partsize=32, 1 class, classbook=2,
///   cascade: low_bits=1 (pass 0 active), book[0][0]=4 (=book3+1).
///
/// Mapping type 0: no submaps flag, no coupling, reserved=0.
/// Mode: blockflag=1 (long), windowtype=0, transformtype=0, mapping=0.
fn write_setup_header(_channels: u8) -> Vec<u8> {
    let mut bw = BitWriter::new();

    // ── Codebooks ──────────────────────────────────────────────────────────────
    // codebook_count - 1 in 8 bits → 3 (= 4 codebooks)
    bw.write_bits(3, 8);

    // Book 0: Floor1 class codebook (dim=1, entries=4, unif_len=2, no VQ)
    write_codebook(&mut bw, 1, 4, 2, 0, 0.0, 0.0, 0);

    // Book 1: Floor1 value codebook (dim=1, entries=256, unif_len=8, no VQ)
    write_codebook(&mut bw, 1, 256, 8, 0, 0.0, 0.0, 0);

    // Book 2: Residue classbook (dim=1, entries=4, unif_len=2, no VQ)
    write_codebook(&mut bw, 1, 4, 2, 0, 0.0, 0.0, 0);

    // Book 3: Residue VQ book (dim=1, entries=256, unif_len=8, lookup_type=1)
    // min=-0.5, delta=1/255, value_bits=8, sequence_p=0
    write_codebook(&mut bw, 1, 256, 8, 1, RESIDUE_MIN, RESIDUE_DELTA, 8);

    // ── Time domain transforms ─────────────────────────────────────────────────
    // vorbis_time_count - 1 in 6 bits → 0 (= 1 time transform)
    bw.write_bits(0, 6);
    // Time transform 0: type = 0
    bw.write_bits(0, 16);

    // ── Floor configurations ───────────────────────────────────────────────────
    // vorbis_floor_count - 1 in 6 bits → 0 (= 1 floor)
    bw.write_bits(0, 6);
    // Floor 0: type = 1 (16 bits)
    bw.write_bits(1, 16);

    // Floor type-1 configuration (spec §6.2.2):
    // floor1_partitions = 3 (5 bits)
    bw.write_bits(3, 5);

    // partition_class_list[0..3] — all class 0 (4 bits each)
    bw.write_bits(0, 4); // partition 0 → class 0
    bw.write_bits(0, 4); // partition 1 → class 0
    bw.write_bits(0, 4); // partition 2 → class 0

    // max_class = 0 → configure class 0:
    //   dimensions - 1 (3 bits) → 1 → dim=2
    bw.write_bits(1, 3);
    //   subclass_bits (2 bits) = 1
    bw.write_bits(1, 2);
    //   Since subclass_bits > 0: mainbook (8 bits) = 0 (book 0)
    bw.write_bits(0, 8);
    // num_subclasses = 2^1 = 2 → write 2 subbook entries (8 bits each):
    //   subbook[0] = 0 (unused: value 0 means "no codebook used")
    bw.write_bits(0, 8);
    //   subbook[1] = 2 (= book_index + 1 = 1 + 1; decoder subtracts 1 → book 1)
    bw.write_bits(2, 8);

    // floor1_multiplier - 1 = 0 (2 bits → multiplier=1, range=256)
    bw.write_bits(0, 2);
    // floor1_rangebits = 10 (4 bits) — X positions are 10 bits each
    bw.write_bits(10, 4);

    // Explicit X values for all 3 partitions × 2 dims each = 6 posts (10 bits each)
    for &x in &FLOOR_X_POSTS {
        bw.write_bits(u64::from(x), 10);
    }

    // ── Residue configurations ─────────────────────────────────────────────────
    // vorbis_residue_count - 1 in 6 bits → 0 (= 1 residue)
    bw.write_bits(0, 6);
    // Residue 0: type = 0 (16 bits)
    bw.write_bits(0, 16);
    // residue_begin = 0 (24 bits)
    bw.write_bits(RESIDUE_BEGIN as u64, 24);
    // residue_end = 128 (24 bits)
    bw.write_bits(RESIDUE_END as u64, 24);
    // residue_partition_size - 1 = 31 (24 bits → size=32)
    bw.write_bits((PARTITION_SIZE as u64) - 1, 24);
    // residue_classifications - 1 = 0 (6 bits → 1 classification)
    bw.write_bits(0, 6);
    // residue_classbook = 2 (8 bits → use book 2)
    bw.write_bits(2, 8);
    // cascade for class 0: low_bits=1 (pass 0 active), has_high=0
    bw.write_bits(1, 3); // low_bits = 0b001 (pass 0 active)
    bw.write_bit(false); // has_high = 0
                         // For the one active pass (pass 0): write book index = 3 (8 bits, must be nonzero < max_codebook=4)
                         // Symphonia checks: book != 0 && book < max_codebook. Book 3 → write 3 (no +1 for residue books).
    bw.write_bits(3, 8);

    // ── Mapping configurations ─────────────────────────────────────────────────
    // vorbis_mapping_count - 1 in 6 bits → 0 (= 1 mapping)
    bw.write_bits(0, 6);
    // Mapping 0: type = 0 (16 bits)
    bw.write_bits(0, 16);
    // submaps_flag [1 bit] = 0 → 1 submap (no explicit count)
    bw.write_bit(false);
    // coupling_flag [1 bit] = 0 (no M/S coupling)
    bw.write_bit(false);
    // reserved [2 bits] = 0 (MUST be zero per spec)
    bw.write_bits(0, 2);
    // Per-channel submap assignment: none written when submaps=1 (ilog(0)=0 bits)

    // Submap 0 configuration:
    //   time_config  (8 bits) = 0
    //   floor_config (8 bits) = 0
    //   residue_cfg  (8 bits) = 0
    bw.write_bits(0, 8);
    bw.write_bits(0, 8);
    bw.write_bits(0, 8);

    // ── Mode configurations ────────────────────────────────────────────────────
    // vorbis_mode_count - 1 in 6 bits → 0 (= 1 mode)
    bw.write_bits(0, 6);
    // Mode 0:
    //   blockflag [1 bit] = 1 (long block)
    bw.write_bit(true);
    //   windowtype [16 bits] = 0
    bw.write_bits(0, 16);
    //   transformtype [16 bits] = 0
    bw.write_bits(0, 16);
    //   mapping [8 bits] = 0
    bw.write_bits(0, 8);

    // ── Framing bit ────────────────────────────────────────────────────────────
    bw.write_bit(true); // framing_bit = 1

    let packed = bw.finalize();

    // Prepend the setup header packet type byte + "vorbis" magic
    let mut pkt = Vec::with_capacity(7 + packed.len());
    pkt.push(0x05); // packet type: setup
    pkt.extend_from_slice(b"vorbis");
    pkt.extend_from_slice(&packed);
    pkt
}

// ─── MDCT forward transform (Vorbis window) ───────────────────────────────────

/// Vorbis MDCT analysis: transform `BLOCK_SIZE_1 = 2048` samples to
/// `N_COEFFS = 1024` real spectral coefficients.
///
/// Uses the exact Vorbis window:
///   w[k] = sin(π/2 · sin²(π/2 · (k+½)/N))
/// which is the `generate_win_curve` function from symphonia's window.rs.
fn vorbis_mdct(samples: &[f32]) -> Vec<f32> {
    let n = BLOCK_SIZE_1; // 2048
    let n2 = N_COEFFS; // 1024

    // Apply the Vorbis window (exact synthesis window).
    let windowed: Vec<f32> = (0..n)
        .map(|k| {
            let s = if k < samples.len() { samples[k] } else { 0.0 };
            // Vorbis window: sin(π/2 · sin²(π/2 · (k+½)/N))
            let frac = std::f32::consts::FRAC_PI_2 * (k as f32 + 0.5) / n as f32;
            let sin_frac = frac.sin();
            let w = (std::f32::consts::FRAC_PI_2 * sin_frac * sin_frac).sin();
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

// ─── Floor1 synthesis (encoder-side closed-loop) ──────────────────────────────

/// `render_point`: integer linear interpolation from symphonia floor.rs.
#[inline(always)]
fn render_point(x0: u32, y0: i32, x1: u32, y1: i32, x: u32) -> i32 {
    let dy = y1 - y0;
    let adx = x1 - x0;
    let err = dy.unsigned_abs() * (x - x0);
    let off = err / adx;
    if dy < 0 {
        y0 - off as i32
    } else {
        y0 + off as i32
    }
}

/// `render_line`: fill floor array from x0 to x1 using the Vorbis ramp algorithm.
fn render_line(x0: u32, y0: i32, x1: u32, y1: i32, n: usize, v: &mut [f32]) {
    if x0 as usize >= n {
        return;
    }

    let dy = y1 - y0;
    let adx = (x1 - x0) as i32;
    let base = dy / adx;
    let mut y = y0;
    let sy = if dy < 0 { base - 1 } else { base + 1 };
    let ady = dy.abs() - base.abs() * adx;

    let y_clamped = y.clamp(0, 255) as usize;
    v[x0 as usize] = FLOOR1_INVERSE_DB_TABLE[y_clamped];

    let mut err = 0i32;
    let x_begin = x0 as usize + 1;
    let x_end = (x1 as usize).min(n);

    if x_begin > x_end {
        return;
    }

    for vv in v[x_begin..x_end].iter_mut() {
        err += ady;
        y += if err >= adx {
            err -= adx;
            sy
        } else {
            base
        };
        let yc = y.clamp(0, 255) as usize;
        *vv = FLOOR1_INVERSE_DB_TABLE[yc];
    }
}

/// Synthesise the floor1 curve from the Y values that the decoder would reconstruct.
///
/// This is the encoder-side closed-loop floor synthesis, mirroring symphonia's
/// `Floor1::synthesis_step1` + `synthesis_step2`.
///
/// The full X list (as the decoder sees it) is:
///   index 0: x=0  (implicit)
///   index 1: x=1024 (implicit, = 1 << rangebits)
///   index 2..7: FLOOR_X_POSTS (explicit, read in partition order)
///
/// Returns a Vec<f32> of length N_COEFFS with the linear floor multiplier per bin.
fn synthesise_floor1_curve(floor_y: &[i32; N_FLOOR_Y]) -> Vec<f32> {
    // Full X list in decoder read order: [0, 1024, 32, 64, 128, 256, 512, 640]
    let full_x: [u32; N_FLOOR_Y] = [0, 1024, 32, 64, 128, 256, 512, 640];

    let n = N_COEFFS as u32; // 1024

    // Multiplier = 1, range = 256.
    // Step 1: unconditionally mark endpoints; compute step2_flag for explicit posts.
    let mut step2_flag = [false; N_FLOOR_Y];
    let mut final_y = [0i32; N_FLOOR_Y];
    let range = 256i32;

    step2_flag[0] = true;
    step2_flag[1] = true;
    final_y[0] = floor_y[0];
    final_y[1] = floor_y[1];

    // For posts 2..N_FLOOR_Y, find neighbours and compute final_y.
    for i in 2..N_FLOOR_Y {
        // Find low_neighbor and high_neighbor (positions in full_x before index i).
        let bound = full_x[i];
        let mut low_x = u32::MIN;
        let mut high_x = u32::MAX;
        let mut low_idx = 0usize;
        let mut high_idx = 1usize;
        for (j, &xv) in full_x[..i].iter().enumerate() {
            if xv > low_x && xv < bound {
                low_x = xv;
                low_idx = j;
            }
            if xv < high_x && xv > bound {
                high_x = xv;
                high_idx = j;
            }
        }

        let predicted = render_point(
            full_x[low_idx],
            final_y[low_idx],
            full_x[high_idx],
            final_y[high_idx],
            full_x[i],
        );

        let val = floor_y[i];
        let highroom = range - predicted;
        let lowroom = predicted;

        if val != 0 {
            let room = 2 * if highroom < lowroom {
                highroom
            } else {
                lowroom
            };

            step2_flag[low_idx] = true;
            step2_flag[high_idx] = true;
            step2_flag[i] = true;

            final_y[i] = if val >= room {
                if highroom > lowroom {
                    val - lowroom + predicted
                } else {
                    predicted - val + highroom - 1
                }
            } else {
                if val & 1 == 1 {
                    predicted - ((val + 1) / 2)
                } else {
                    predicted + (val / 2)
                }
            };
        } else {
            step2_flag[i] = false;
            final_y[i] = predicted;
        }
    }

    // Step 2: render lines in sort order.
    // Sort indices by X value.
    let mut sort_order: Vec<usize> = (0..N_FLOOR_Y).collect();
    sort_order.sort_by_key(|&i| full_x[i]);

    let mut floor = vec![0.0f32; N_COEFFS];

    let multiplier = 1i32;
    let ly0 = final_y[sort_order[0]] * multiplier;
    let ly0 = ly0.clamp(0, 255);

    let mut hx = 0u32;
    let mut hy = 0i32;
    let mut lx = 0u32;
    let mut ly = ly0;

    for &i in &sort_order[1..] {
        if step2_flag[i] {
            hy = (final_y[i] * multiplier).clamp(0, 255);
            hx = full_x[i];
            render_line(lx, ly, hx, hy, N_COEFFS, &mut floor);
            lx = hx;
            ly = hy;
        }
    }

    if hx < n {
        render_line(hx, hy, n, hy, N_COEFFS, &mut floor);
    }

    floor
}

// ─── Floor1 Y value computation ──────────────────────────────────────────────

/// Compute floor1 Y values for a given MDCT frame.
///
/// For each X post position, we compute the magnitude and find the best
/// matching Y value (0..255) such that FLOOR1_INVERSE_DB_TABLE[y] approximates
/// the signal magnitude at that frequency bin.
///
/// The floor encoding uses:
/// - Y[0] and Y[1]: explicit endpoints, written as 8-bit values (range=256, multiplier=1)
/// - Y[2..N_FLOOR_Y]: explicit posts, written via class+subbook scheme
///
/// Returns `[i32; N_FLOOR_Y]` with values in [0, 255].
fn compute_floor_y(mdct: &[f32]) -> [i32; N_FLOOR_Y] {
    // Build inverse lookup: for each possible magnitude, find the best y index.
    // FLOOR1_INVERSE_DB_TABLE is monotonically increasing (0..255 → small..large).
    // Given a magnitude m, find y such that FLOOR1_INVERSE_DB_TABLE[y] >= m.
    let magnitude_to_y = |mag: f32| -> i32 {
        if mag <= FLOOR1_INVERSE_DB_TABLE[0] {
            return 0;
        }
        if mag >= FLOOR1_INVERSE_DB_TABLE[255] {
            return 255;
        }
        // Binary search
        let mut lo = 0usize;
        let mut hi = 255usize;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if FLOOR1_INVERSE_DB_TABLE[mid] < mag {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // Return whichever is closer
        let dist_lo = (FLOOR1_INVERSE_DB_TABLE[lo] - mag).abs();
        let dist_hi = (FLOOR1_INVERSE_DB_TABLE[hi] - mag).abs();
        if dist_lo <= dist_hi {
            lo as i32
        } else {
            hi as i32
        }
    };

    // Full X list (decoder read order):
    let full_x: [u32; N_FLOOR_Y] = [0, 1024, 32, 64, 128, 256, 512, 640];

    let mut y = [0i32; N_FLOOR_Y];
    for (i, &x) in full_x.iter().enumerate() {
        // Use the MDCT bin at position x (or average around it for smoothness).
        let bin = (x as usize).min(N_COEFFS - 1);
        // Take the magnitude at this bin (absolute value of MDCT coefficient).
        let mag = mdct[bin].abs();
        y[i] = magnitude_to_y(mag);
    }
    y
}

// ─── Vorbis VBR quality control ───────────────────────────────────────────────

/// Vorbis encoder quality level.
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

    /// Silence threshold for this quality level.
    pub fn silence_threshold(&self) -> f32 {
        let q = self.0.max(0.0);
        let threshold_db = -40.0 - q * 40.0;
        10.0f32.powf(threshold_db / 20.0)
    }

    /// Quantization scale factor (residue_delta_scale).
    pub fn residue_delta_scale(&self) -> f32 {
        let q = self.0;
        let scale = if q >= 0.0 {
            1.0 + (1.0 - q) * 7.0
        } else {
            8.0 + (-q) * 20.0
        };
        scale.clamp(0.5, 10.0)
    }
}

// ── Psychoacoustic Model ──────────────────────────────────────────────────────

/// Absolute threshold of hearing in dB SPL at frequency `f_hz` (Hz).
#[cfg(test)]
fn ath_db(f_hz: f32) -> f32 {
    let f_khz = f_hz / 1000.0;
    3.64 * f_khz.powf(-0.8) - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp() + 1e-3 * f_khz.powi(4)
}

/// Convert frequency in Hz to Bark scale.
#[cfg(test)]
fn hz_to_bark(f_hz: f32) -> f32 {
    26.81 * f_hz / (1960.0 + f_hz) - 0.53
}

/// Spreading function: masking attenuation (dB).
#[cfg(test)]
fn spreading_db(bark_masker: f32, bark_maskee: f32) -> f32 {
    let delta = bark_maskee - bark_masker;
    if delta < 0.0 {
        27.0 * delta
    } else {
        (-17.5 + 0.4 * bark_masker) * delta
    }
}

/// Compute per-bin masking threshold from MDCT coefficients.
#[cfg(test)]
#[must_use]
pub(crate) fn compute_masking_threshold(mdct: &[f32], sample_rate: u32) -> Vec<f32> {
    let n_bins = mdct.len();
    if n_bins == 0 {
        return Vec::new();
    }
    let freq_resolution = sample_rate as f32 / (2.0 * n_bins as f32);
    let n_bark: usize = 24;

    let bark_of_bin: Vec<f32> = (0..n_bins)
        .map(|k| {
            let f_hz = ((k as f32 + 0.5) * freq_resolution).max(20.0);
            hz_to_bark(f_hz).clamp(0.0, (n_bark - 1) as f32)
        })
        .collect();

    let ath_linear: Vec<f32> = (0..n_bins)
        .map(|k| {
            let f_hz = ((k as f32 + 0.5) * freq_resolution).max(20.0);
            10.0f32.powf(ath_db(f_hz) / 10.0)
        })
        .collect();

    let energy: Vec<f32> = mdct.iter().map(|&x| x * x).collect();
    let mut bark_max_energy = vec![0.0f32; n_bark];
    for (k, &e) in energy.iter().enumerate() {
        let band = (bark_of_bin[k].floor() as usize).min(n_bark - 1);
        if e > bark_max_energy[band] {
            bark_max_energy[band] = e;
        }
    }

    let mut threshold = ath_linear;
    for (k, thresh) in threshold.iter_mut().enumerate() {
        let bark_k = bark_of_bin[k];
        for (band, &masker_energy) in bark_max_energy.iter().enumerate() {
            if masker_energy < 1e-20 {
                continue;
            }
            let bark_masker = band as f32 + 0.5;
            let spread = spreading_db(bark_masker, bark_k);
            let masked = masker_energy * 10.0f32.powf(spread / 10.0);
            if masked > *thresh {
                *thresh = masked;
            }
        }
    }

    threshold
}

/// Compute Y[0] and Y[1] floor endpoint values using psychoacoustic masking.
#[cfg(test)]
fn compute_floor_y_psychoacoustic(_mdct: &[f32], mask: &[f32]) -> (u8, u8) {
    let n = mask.len().max(1);
    let log_mean_mask: f32 = mask.iter().map(|&m| m.max(1e-30).log10()).sum::<f32>() / n as f32;
    let db = 10.0 * log_mean_mask;
    let y = ((db + 60.0) * 255.0 / 120.0).clamp(0.0, 255.0) as u8;
    let hi_start = n / 2;
    let hi_n = (n - hi_start).max(1);
    let log_mean_hi: f32 = mask[hi_start..]
        .iter()
        .map(|&m| m.max(1e-30).log10())
        .sum::<f32>()
        / hi_n as f32;
    let db_hi = 10.0 * log_mean_hi;
    let y1 = ((db_hi + 60.0) * 255.0 / 120.0).clamp(0.0, 255.0) as u8;
    (y, y1)
}

// ─── Audio packet encoder ─────────────────────────────────────────────────────

/// Encode one MDCT audio frame for all channels.
///
/// This implements the full Vorbis I audio packet encoding:
/// 1. Floor type-1 encoding with 8 X-posts and correct codebook field widths.
/// 2. Residue type-0 encoding using canonical codewords for both classbook and VQ.
/// 3. Closed-loop floor synthesis to compute accurate residuals.
fn encode_audio_packet(
    channels: usize,
    prev_long: bool,
    next_long: bool,
    mdct_per_channel: &[Vec<f32>],
    quality: VorbisQuality,
    _sample_rate: u32,
) -> Vec<u8> {
    let mut bw = BitWriter::new();

    // packet_type = 0 (audio). Must be the first bit.
    bw.write_bit(false);

    // mode_number: ilog(1 - 1) = ilog(0) = 0 bits → nothing to write.

    // blockflag=1 → must write previous/next window flags.
    bw.write_bit(prev_long);
    bw.write_bit(next_long);

    // Precompute canonical codewords for each book:
    // Book 0: uniform len=2, 4 entries → canonical = [0,1,2,3]
    let book0_lens: Vec<u8> = vec![2; 4];
    let book0_cw = canonical_codewords(&book0_lens); // [0b00, 0b01, 0b10, 0b11]

    // Book 1: uniform len=8, 256 entries → canonical[i] = i
    // (no need to precompute; for uniform 8-bit, codeword[i] = i)

    // Book 2: uniform len=2, 4 entries → canonical = [0,1,2,3]
    let book2_lens: Vec<u8> = vec![2; 4];
    let book2_cw = canonical_codewords(&book2_lens);

    // Book 3: uniform len=8, 256 entries → canonical[i] = i
    // (same as book 1 — codeword[i] = i, 8-bit)

    let silence_threshold = quality.silence_threshold();

    // ── Per-channel floor encoding ─────────────────────────────────────────────
    // For each channel:
    // - floor_nonzero [1 bit]
    // - If nonzero:
    //   - Y[0] [8 bits] (endpoint at X=0, range=256, multiplier=1 → 8-bit)
    //   - Y[1] [8 bits] (endpoint at X=1024)
    //   - For each partition (3 partitions × dim=2 posts each):
    //     - classword via book0 [2 bits]: encodes 2 subclass values
    //     - For each "active subclass" post: emit Y value via book1 [8 bits]

    let mut floor_y_per_ch: Vec<[i32; N_FLOOR_Y]> = Vec::with_capacity(channels);
    let mut ch_nonzero: Vec<bool> = Vec::with_capacity(channels);

    for coeffs in mdct_per_channel.iter().take(channels) {
        // RMS check for silence
        let energy: f32 = coeffs.iter().map(|&c| c * c).sum::<f32>();
        let rms = (energy / coeffs.len() as f32).sqrt();

        if rms < silence_threshold {
            bw.write_bit(false); // floor_nonzero = 0
            floor_y_per_ch.push([0i32; N_FLOOR_Y]);
            ch_nonzero.push(false);
        } else {
            bw.write_bit(true); // floor_nonzero = 1
            ch_nonzero.push(true);

            // Compute Y values from MDCT spectrum.
            let y = compute_floor_y(coeffs);
            floor_y_per_ch.push(y);

            // Write Y[0] and Y[1] (endpoints, 8 bits each).
            bw.write_bits(y[0] as u64, 8);
            bw.write_bits(y[1] as u64, 8);

            // Write 3 partitions × 1 classword + 2 explicit posts each.
            // Class 0 has dim=2, subclass_bits=1, mainbook=0.
            // Each classword from book0 encodes 2 subclass values packed as:
            //   cval = sub0 | (sub1 << 1)
            //   We always use subclass=1 (explicit coding) → sub0=1, sub1=1 → cval=3.
            //   But for floor1, the decoder reads:
            //     cval = read_scalar(mainbook=book0)
            //     for each dim post:
            //       subclass_idx = cval & csub (csub = (1<<1)-1 = 1)
            //       cval >>= cbits (cbits=1)
            //       if is_subbook_used (subclass_idx=1 → subbook[1]=1 → used):
            //         y = read_scalar(subbook[1]=book1)
            //       else (subclass_idx=0 → subbook[0]=unused):
            //         y = 0 (implicit)
            //
            // So: to emit both posts explicitly, set sub0=1 and sub1=1 → cval=3.
            // cval=3 → canonical codeword for book0[3] = book0_cw[3] (2 bits).
            // Then emit y[2] via book1 (8 bits), then y[3] via book1 (8 bits).
            //
            // Partition 0 → y[2], y[3]  (posts at X=32, X=64)
            // Partition 1 → y[4], y[5]  (posts at X=128, X=256)
            // Partition 2 → y[6], y[7]  (posts at X=512, X=640)

            for part in 0..3usize {
                let y0_idx = 2 + part * 2;
                let y1_idx = y0_idx + 1;

                // cval = sub0=1 | (sub1=1 << 1) = 3 → both posts explicitly coded
                let cval = 3u64;
                bw.write_bits(book0_cw[cval as usize] as u64, 2);

                // First post (subclass=1 → use book1): y[y0_idx] via book1 (8 bits)
                // For uniform 8-bit book, canonical codeword[v] = v.
                let v0 = (y[y0_idx] as u64) & 0xFF;
                bw.write_bits(v0, 8);

                // Second post (subclass=1 → use book1): y[y1_idx]
                let v1 = (y[y1_idx] as u64) & 0xFF;
                bw.write_bits(v1, 8);
            }
        }
    }

    // ── Residue encoding (type 0) ─────────────────────────────────────────────
    // For residue type 0, the spec reads all channels before moving to next partition.
    // Order for each pass/partition_batch:
    //   For each channel: read classbook (book2)
    //   For each channel × partition: if class[part] is used, read VQ (book3)
    //
    // We always use class=1 for all partitions, so all VQ books are active.
    // classbook (book2, dim=1) encodes 1 partition class per decode.
    // parts_per_classword = dim(book2) = 1.
    // n_partitions = 4 (128/32).
    //
    // For pass 0, for each partition_batch (step_by=1):
    //   For each channel: emit classword for class=1 via book2 (2 bits).
    //   For each channel × partition:
    //     If class is used: emit 32 VQ values via book3 (8 bits each).

    // Only emit residue when at least one channel is non-silent.
    let has_audio = ch_nonzero.iter().any(|&v| v);
    if has_audio {
        // Synthesise floor curves for closed-loop residual computation.
        let floor_curves: Vec<Vec<f32>> = (0..channels)
            .map(|ch| {
                if ch_nonzero[ch] {
                    synthesise_floor1_curve(&floor_y_per_ch[ch])
                } else {
                    vec![0.0f32; N_COEFFS]
                }
            })
            .collect();

        // Compute normalised residuals.
        // r[ch][k] = mdct[ch][k] / floor_curve[k]  (avoid div by zero)
        let residuals: Vec<Vec<f32>> = (0..channels)
            .map(|ch| {
                let coeffs = &mdct_per_channel[ch];
                let floor = &floor_curves[ch];
                (0..N_COEFFS)
                    .map(|k| {
                        let fv = floor[k].max(1e-8);
                        coeffs[k] / fv
                    })
                    .collect()
            })
            .collect();

        // Quantise residuals.
        let quantised: Vec<Vec<u8>> = residuals
            .iter()
            .map(|r| {
                r.iter()
                    .map(|&v| {
                        let idx = ((v - RESIDUE_MIN) / RESIDUE_DELTA).round() as i32;
                        idx.clamp(0, (RESIDUE_ENTRIES - 1) as i32) as u8
                    })
                    .collect()
            })
            .collect();

        // Residue type 0 encoding loop.
        // parts_per_classword = dim(classbook=book2) = 1.
        // For each partition batch (each partition, since ppc=1):
        //   For each channel: emit class via book2 (2 bits for class=1).
        //   For each channel: emit 32 VQ values via book3 (8 bits each).
        for part in 0..N_PARTITIONS {
            // Classword phase: for each channel, emit class=1 via book2.
            for (ch, &nonzero) in ch_nonzero.iter().take(channels).enumerate() {
                let _ = ch;
                if nonzero {
                    // class=1 → canonical codeword for book2[1] (2 bits)
                    bw.write_bits(book2_cw[1] as u64, 2);
                } else {
                    // Silent channel: emit class=0 → codeword for book2[0] (2 bits)
                    bw.write_bits(book2_cw[0] as u64, 2);
                }
            }

            // VQ phase: for each channel (if class=1 → used).
            let base = RESIDUE_BEGIN + part * PARTITION_SIZE;
            for (ch, ch_q) in quantised.iter().take(channels).enumerate() {
                let _ = ch;
                // Class 1 is always "used" (we set is_used bit for pass 0 in setup).
                // But if channel is silent, we still need to emit something;
                // the decoder will decode it regardless.
                for j in 0..PARTITION_SIZE {
                    let bin = base + j;
                    let idx = if bin < ch_q.len() { ch_q[bin] } else { 0 };
                    // For book3 (uniform 8-bit), canonical codeword[idx] = idx.
                    bw.write_bits(u64::from(idx), 8);
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

    let ident = write_ident_header(channels as u8, sample_rate);
    stream.write_packet(&ident, 0, false)?;

    let comment = write_vorbis_comment_packet(VENDOR_STRING, &[], false);
    stream.write_packet(&comment, 0, false)?;

    let setup = write_setup_header(channels as u8);
    stream.write_packet(&setup, 0, false)?;

    // ── Audio packets ─────────────────────────────────────────────────────────

    let hop = (BLOCK_SIZE_1 / 2) as i64;
    let total_samples = buf.samples.len().checked_div(channels).unwrap_or(0);

    // Minimum one audio packet even for empty buffers.
    let n_frames = (total_samples / (BLOCK_SIZE_1 / 2)).max(1);

    for frame_idx in 0..n_frames {
        let is_last = frame_idx == n_frames - 1;
        let prev_long = frame_idx > 0;
        let next_long = !is_last;

        let frame_start = frame_idx * (BLOCK_SIZE_1 / 2);
        let frame_end = (frame_start + BLOCK_SIZE_1).min(total_samples);

        let mdct_per_channel: Vec<Vec<f32>> = (0..channels)
            .map(|ch| {
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
        canonical_codewords, compute_masking_threshold, encode_vorbis, encode_vorbis_file,
        encode_vorbis_quality_file, encode_vorbis_with_quality, vorbis_mdct, VorbisQuality,
        BLOCK_SIZE_1, N_COEFFS,
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

    // ── canonical_codewords tests ─────────────────────────────────────────────

    /// For uniform lengths [2,2,2,2], canonical_codewords returns [0,1,2,3].
    #[test]
    fn test_canonical_codewords_uniform_4x2() {
        let lens = vec![2u8, 2, 2, 2];
        let cw = canonical_codewords(&lens);
        assert_eq!(
            cw,
            vec![0u32, 1, 2, 3],
            "uniform 2-bit book should yield 0,1,2,3"
        );
    }

    /// For uniform lengths [8; 256], canonical_codewords[i] == i.
    #[test]
    fn test_canonical_codewords_uniform_256x8() {
        let lens = vec![8u8; 256];
        let cw = canonical_codewords(&lens);
        for (i, &c) in cw.iter().enumerate() {
            assert_eq!(
                c, i as u32,
                "uniform 8-bit codeword[{i}] should be {i}, got {c}"
            );
        }
    }

    /// Symphonia's test vector: lengths [2,4,4,4,4,2,3,3] → [0,4,5,6,7,2,6,7].
    #[test]
    fn test_canonical_codewords_symphonia_vector() {
        let lens = vec![2u8, 4, 4, 4, 4, 2, 3, 3];
        let cw = canonical_codewords(&lens);
        assert_eq!(cw, vec![0u32, 0x4, 0x5, 0x6, 0x7, 0x2, 0x6, 0x7]);
    }

    // ── MDCT tests ────────────────────────────────────────────────────────────

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

    // ── Structural tests ──────────────────────────────────────────────────────

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
        let magic = b"\x01vorbis";
        assert!(
            bytes.windows(magic.len()).any(|w| w == magic),
            "output must contain Vorbis identification header magic"
        );
    }

    /// Verify that > 2 channels is rejected with an Encode error.
    #[test]
    fn test_encode_vorbis_rejects_high_channel_count() {
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
        let _ = std::fs::remove_file(&tmp);
    }

    /// Structural round-trip: encode succeeds and the bitstream contains valid OGG+Vorbis framing.
    #[test]
    fn test_encode_vorbis_round_trip_decode() {
        let n_samples = BLOCK_SIZE_1 * 4;
        let buf = silence_buffer_mono(n_samples);
        let mut out = Cursor::new(Vec::new());
        encode_vorbis(&buf, &mut out).expect("encode_vorbis must succeed for round-trip");
        let encoded = out.into_inner();

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

    /// Higher quality must produce a lower silence threshold.
    #[test]
    fn test_vorbis_silence_threshold() {
        let thresh_hi = VorbisQuality::from_level(10).silence_threshold();
        let thresh_lo = VorbisQuality::from_level(0).silence_threshold();
        assert!(
            thresh_hi < thresh_lo,
            "high quality should have lower silence threshold; got hi={thresh_hi}, lo={thresh_lo}"
        );
    }

    /// Low quality produces fewer or equal bytes than high quality for a very quiet sine.
    #[test]
    fn test_encode_vorbis_with_quality_low_vs_high() {
        let n_samples = BLOCK_SIZE_1 * 8;
        let quiet_sine: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.001)
            .collect();
        let buf = AudioBuffer {
            samples: quiet_sine,
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
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
        use super::ath_db;

        let sample_rate = 44_100u32;
        let n_bins = N_COEFFS;
        let freq_resolution = sample_rate as f32 / (2.0 * n_bins as f32);

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

    /// A 440 Hz pure tone should create elevated masking near 440 Hz vs 4000 Hz.
    #[test]
    fn test_masking_threshold_pure_tone_creates_local_peak() {
        let sample_rate = 44_100u32;
        let sine: Vec<f32> = (0..BLOCK_SIZE_1)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.8)
            .collect();
        let mdct = vorbis_mdct(&sine);
        let mask = compute_masking_threshold(&mdct, sample_rate);

        let bin_440 = (440.0 * N_COEFFS as f32 / (sample_rate as f32 / 2.0)).round() as usize;
        let bin_4k = (4000.0 * N_COEFFS as f32 / (sample_rate as f32 / 2.0)).round() as usize;
        let bin_440 = bin_440.min(N_COEFFS - 1);
        let bin_4k = bin_4k.min(N_COEFFS - 1);

        assert!(
            mask[bin_440] > mask[bin_4k],
            "440 Hz tone should produce higher masking threshold near 440 Hz ({}) than at 4 kHz ({})",
            mask[bin_440],
            mask[bin_4k]
        );
    }

    /// Silence input to `compute_floor_y_psychoacoustic` should produce a lower Y
    /// than a loud sine wave.
    #[test]
    fn test_compute_floor_y_psychoacoustic_silence_low() {
        use super::compute_floor_y_psychoacoustic;

        let sample_rate = 44_100u32;

        let silence_mdct = vec![0.0f32; N_COEFFS];
        let silence_mask = compute_masking_threshold(&silence_mdct, sample_rate);
        let (y_silence, _) = compute_floor_y_psychoacoustic(&silence_mdct, &silence_mask);

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
        use super::compute_floor_y_psychoacoustic;

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
