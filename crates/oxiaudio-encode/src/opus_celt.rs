//! CELT (Constrained Energy Lapped Transform) encoder for Opus.
//!
//! Implements band energy quantization and spectral coefficient encoding via
//! Pyramid Vector Quantization (PVQ), following RFC 6716 §4.3 structure.
//!
//! # Conformance note
//!
//! This default path (`encode_celt_frame` / `pvq_encode` /
//! `quantize_band_energy`) is structurally closer to RFC 6716 than the previous
//! 4-bit placeholder, but is **not fully conformant**: its PVQ shape selection
//! is a greedy pulse allocator and its band-energy quantization is a 4-bit
//! approximation, so a standard decoder will not accept it.
//!
//! The **conformant** path (`encode_celt_frame_conformant` /
//! `encode_celt_body_into`) is decoder-verified. Its PVQ shape selection now
//! uses the exact libopus rate-distortion search (`op_pvq_search`, maximising
//! `<x,y>²/<y,y>`) together with the matching forward `exp_rotation`, and the
//! resulting pulse vector is range-coded with the exact CWRS combinatorial
//! index (RFC 6716 §4.3.4.6). The range coder itself is the RFC 6716 §4.1
//! [`RangeEncoder`] (correct carry propagation and bit-reversal / raw-bit tail
//! packing) shared by both paths — it is *not* a private variant.
//!
//! As of oxiaudio 0.2.1 the conformant path also implements:
//!
//! * **Real split-band (`itheta`) coding** (RFC 6716 §4.3.4.3) — the angle is
//!   measured from the two half-bands' actual energies, quantised to `qn`
//!   levels and range-coded with the triangular distribution, and *both* halves
//!   are coded recursively. Previously `itheta` was pinned to 0, which zeroed
//!   the upper half of every split band.
//! * **Real fine-energy quantisation** (§4.3.2.1) — the coarse-energy residual
//!   is refined with the allocator's per-band fine bits and the final ±½-LSB
//!   pass, instead of writing zeros.
//! * **Lapped MDCT analysis with overlap history and CELT pre-emphasis**, so
//!   the decoder's overlap-add and de-emphasis are actually inverted. See
//!   `crate::opus_mdct`.
//!
//! Measured against the reference decoder, per-band energy is now reproduced
//! within ≈1 dB and the encoder's `final_range` matches the decoder's for every
//! frame size in `[MIN_CELT_FRAME_BYTES, MAX_CELT_FRAME_BYTES]` — the canonical
//! libopus conformance check. As of oxiaudio 0.2.1 that upper bound is the
//! RFC's own 1275-byte frame limit (510 kbps mono) rather than the 80 bytes
//! (≈32 kbps) the previous wave could prove; see [`MAX_CELT_FRAME_BYTES`] for
//! the root cause of the old ceiling. Remaining scope limits are mono-only
//! coding (stereo input is downmixed), no transient/TF adaptation and no
//! dynamic allocation.
//!
//! # Verification instruments
//!
//! [`encode_celt_frame_conformant_traced`] returns a [`CeltEncodeTrace`] whose
//! `stage_tells` mirror the field order and names
//! [`crate::opus_celt_verify::parse_celt_frame`] records, so a desynchronisation
//! can be bisected to a coding stage instead of only showing up as a mismatched
//! `final_range` at the end of the frame. The trace also reports whether the
//! range-coded (front) and raw-bit (back) halves physically fitted the packet —
//! a check `final_range` alone cannot make, because dropping a raw-bit byte
//! never touches the range register.
//!
//! # CELT Band Layout (RFC 6716 Table 1)
//!
//! For a 960-sample / 480-bin MDCT at 48 kHz, the band boundaries in MDCT bins are:
//!
//! ```text
//! Band  0:  bins [  0,   1)   ~    0 –  100 Hz
//! Band  1:  bins [  1,   2)   ~  100 –  200 Hz
//! ...
//! Band 20:  bins [ 78, 100)   ~ 7800 – 10000 Hz
//! (Remaining bins 100..480 are treated as trailing high-frequency content.)
//! ```

use crate::opus_celt_bands::{encode_band_with_splits, BandEncCtx, BandEncSink};
use crate::opus_celt_rate::{
    celt_compute_allocation_enc, celt_init_caps, SkipBitEncoder, BITRES, MAX_FINE_BITS,
};
use crate::opus_celt_tables::{EBAND_5MS, E_MEANS, E_PROB_MODEL, NUM_BANDS_CELT};
use crate::opus_mdct::mdct_forward;
use crate::opus_pvq;
use crate::opus_range::{ec_laplace_encode, RangeEncoder};

/// Number of CELT frequency bands (RFC 6716, Table 1).
pub const NUM_BANDS: usize = 21;

/// Band lower-bin boundaries (inclusive) in 480-bin MDCT space.
///
/// The 22nd entry is the exclusive upper bound of the last defined band.
/// These match RFC 6716 Table 5 for 480 MDCT coefficients at 48 kHz.
pub const BAND_BINS: [usize; NUM_BANDS + 1] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

// ── PVQ core helpers ──────────────────────────────────────────────────────────

/// Compute the L2 norm of a band's spectral coefficients.
fn band_norm(coeffs: &[f32]) -> f32 {
    coeffs.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Return a unit-norm copy of `coeffs`.
///
/// If the norm is negligible (< 1e-10), returns a zero vector — the PVQ
/// encoder will pile all pulses on index 0 in that case, which is the
/// correct degenerate behaviour for silent bands.
fn normalize_band(coeffs: &[f32]) -> Vec<f32> {
    let norm = band_norm(coeffs);
    if norm < 1e-10 {
        return vec![0.0; coeffs.len()];
    }
    coeffs.iter().map(|&x| x / norm).collect()
}

/// Quantize band energy to a 4-bit index (0..=15).
///
/// Converts `norm` to dB relative to `global_gain_db` and maps the
/// range [−60, 0] dB onto [0, 15].  Values above 0 dB (louder than global
/// gain) are clamped to 15; values below −60 dB clamp to 0.
fn quantize_band_energy(norm: f32, global_gain_db: f32) -> u8 {
    let db = if norm > 1e-20 {
        20.0 * norm.log10()
    } else {
        -120.0_f32
    };
    let relative_db = (db - global_gain_db).clamp(-60.0, 0.0);
    ((relative_db + 60.0) / 4.0) as u8
}

/// Greedy PVQ encoder — allocate `k_pulses` unit pulses across `normalized`.
///
/// The returned vector has the same length as `normalized` and contains signed
/// integers (±magnitude) whose L1 norm equals `k_pulses`.
///
/// The greedy strategy picks the dimension with the largest remaining absolute
/// value at each step, making it a single-pass O(N·K) algorithm.  It produces
/// a good — though not optimal — approximation of the target unit vector.
pub fn pvq_encode(normalized: &[f32], k_pulses: u32) -> Vec<i32> {
    let n = normalized.len();
    let mut y = vec![0i32; n];

    if k_pulses == 0 || n == 0 {
        return y;
    }

    // Working copy of absolute values used to guide greedy allocation.
    let mut mag: Vec<f32> = normalized.iter().map(|&x| x.abs()).collect();
    let step = 1.0 / k_pulses as f32;

    for _ in 0..k_pulses {
        // Find the dimension with the largest remaining magnitude.
        let best = mag
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        y[best] += 1;
        // Reduce the residual by one pulse-step so subsequent iterations
        // can reallocate to other dimensions if their magnitude is larger.
        mag[best] = (mag[best] - step).max(0.0);
    }

    // Apply original signs.
    for (i, yi) in y.iter_mut().enumerate() {
        if normalized[i] < 0.0 {
            *yi = -*yi;
        }
    }

    y
}

/// Compute the number of PVQ pulses for a band given the available bits.
///
/// For a band of `band_size` coefficients with `K` pulses the information
/// content is approximately `band_size · log2(K + 1)` bits.  Inverting gives
/// K ≈ 2^(bits / band_size) − 1.  We use the simpler linear approximation
/// `K = floor(bits / log2(band_size + 1))` which is well-behaved even for
/// very small bands (size = 1).
pub(crate) fn compute_k_pulses(band_size: usize, bits_available: u32) -> u32 {
    if band_size == 0 || bits_available == 0 {
        return 0;
    }
    let log2_n = (band_size as f32 + 1.0).log2().max(1.0);
    (bits_available as f32 / log2_n).floor() as u32
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Encode one CELT frame into the provided range encoder using PVQ band-shape coding.
///
/// # Arguments
///
/// * `pcm`      — Interleaved PCM samples for this frame.  For mono, length is
///   `FRAME_SIZE` (960); for stereo, length is `2 × FRAME_SIZE`.
/// * `channels` — Number of audio channels (1 or 2).
/// * `enc`      — Range encoder to write the coded symbols into.
///
/// # Implementation notes
///
/// Only the first channel is MDCT-analysed; for stereo the two channels are
/// mixed to mono for CELT analysis (a real encoder would use mid/side coding
/// per RFC 6716 §4.3.1).
///
/// Band energies are quantized to 4 bits (16 levels covering a 60 dB range
/// relative to the global gain).  Band shapes are encoded with a greedy PVQ
/// pulse allocator using a fixed budget of 8 bits per band.
pub fn encode_celt_frame(pcm: &[f32], channels: usize, enc: &mut RangeEncoder) {
    encode_celt_frame_inner(pcm, channels, 8, enc);
}

/// Standalone CELT frame encoder that manages its own range encoder.
///
/// Equivalent to constructing a fresh [`RangeEncoder`], calling
/// [`encode_celt_frame`] with the given PCM, and flushing.  The
/// `sample_rate` and `target_bitrate_kbps` parameters are used to compute
/// the per-band bit budget.
///
/// # Returns
///
/// The range-coded frame bytes (always non-empty because the encoder
/// emits at least its flush bytes).
pub fn encode_celt_frame_pvq(
    pcm: &[f32],
    channels: usize,
    sample_rate: u32,
    target_bitrate_kbps: u32,
) -> Vec<u8> {
    use crate::opus_mdct::FRAME_SIZE;
    // Bits per frame = kbps * 1000 * (frame_size / sample_rate)
    //               = kbps * frame_size / (sample_rate / 1000)
    // Use integer arithmetic in the order that avoids premature truncation:
    //   kbps * FRAME_SIZE * 1000 / sample_rate
    let bits_per_frame = target_bitrate_kbps
        .saturating_mul(FRAME_SIZE as u32)
        .saturating_mul(1000)
        / sample_rate.max(1);
    let bits_per_band = (bits_per_frame / NUM_BANDS as u32).max(1);

    let mut enc = RangeEncoder::new();
    encode_celt_frame_inner(pcm, channels, bits_per_band, &mut enc);
    enc.finish()
}

// ── RFC 6716 conformant encoder ──────────────────────────────────────────────

/// Encode one RFC 6716–conformant CELT-only mono 20 ms Opus frame.
///
/// Produces an Opus packet (TOC byte `0xF8` + range-coded frame data) that is
/// decodable by a standard Opus decoder.  The encoder performs:
///
/// 1. MDCT analysis of the input PCM.
/// 2. Per-band energy computation and Laplace-coded intra energy quantization.
/// 3. Neutral TF / spread / dynalloc / trim headers.
/// 4. Rate allocation (`clt_compute_allocation` port from libopus `rate.c`).
/// 5. Fine energy (zeros) written to the end stream.
/// 6. PVQ shapes (greedy CWRS) for each coded band.
///
/// # Attribution
///
/// Rate-allocation logic ported from libopus `celt/rate.c` and `celt/celt.c`
/// (Xiph.Org Foundation, BSD-3-Clause).
pub fn encode_celt_frame_conformant(pcm: &[f32], channels: usize) -> Vec<u8> {
    encode_celt_frame_conformant_with_history(&[], pcm, channels)
}

/// Encode one RFC 6716–conformant CELT-only mono 20 ms Opus frame **with MDCT
/// overlap history**.
///
/// CELT's MDCT is a *lapped* transform: the analysis block for frame `t` spans
/// the previous frame and the current frame, and the decoder reconstructs each
/// output frame by overlap-adding two consecutive inverse transforms. Encoding
/// every frame as if it were preceded by silence (which is what
/// [`encode_celt_frame_conformant`] does when called with no history) breaks
/// time-domain alias cancellation and leaves large audible aliasing in the
/// reconstruction.
///
/// Pass the previous 960-sample frame as `prev` (an empty slice for the first
/// frame of a stream) and the frame being coded as `cur`.
///
/// # Arguments
///
/// * `prev`     — previous 960-sample mono frame, or `&[]` at stream start.
/// * `cur`      — the 960-sample mono frame to encode (zero-padded if shorter).
/// * `channels` — accepted for API symmetry; the conformant CELT path is mono.
pub fn encode_celt_frame_conformant_with_history(
    prev: &[f32],
    cur: &[f32],
    channels: usize,
) -> Vec<u8> {
    let _ = channels;
    encode_celt_frame_conformant_sized(prev, cur, DEFAULT_CELT_FRAME_BYTES)
}

/// Default CELT payload size in bytes per 20 ms frame (≈ 25.6 kbps mono).
pub const DEFAULT_CELT_FRAME_BYTES: usize = 64;

/// Smallest CELT payload the conformant writer will emit (≈ 6.4 kbps).
pub const MIN_CELT_FRAME_BYTES: usize = 16;

/// Largest CELT payload this encoder will emit for a single 20 ms frame.
///
/// This is the RFC 6716 §3.2.1 **frame** limit: an Opus frame length field
/// represents at most `255·4 + 255 = 1275` bytes, so a code-0 (single-frame)
/// packet is at most `1 + 1275 = 1276` bytes including its TOC byte. At 20 ms
/// that is 510 kbps mono — the maximum rate Opus can express.
///
/// The cap is a *conformance* bound, not a taste bound: the encoder's
/// `final_range` register is checked against the reference decoder's over the
/// whole `[MIN_CELT_FRAME_BYTES, MAX_CELT_FRAME_BYTES]` range on a corpus that
/// includes full-scale white noise, near-silence, pure tones from 120 Hz to
/// 19 kHz, an impulse train and digital silence
/// (`tests/m_opus_celt_snr.rs::celt_final_range_matches_reference_across_sizes`).
///
/// Before oxiaudio 0.2.1 this constant was 80 (≈32 kbps) because frames above
/// ≈90 bytes desynchronised from the reference decoder for reasons that had not
/// been identified. The cause was
/// [`celt_bits2pulses`](crate::opus_celt_rate) clamping the *partition* scale
/// `lm` to 0 before indexing the pulse cache: a band that splits four times
/// (band 20 at LM = 3 goes 176 → 88 → 44 → 22 → 11 bins) reaches `lm = -1`,
/// which must select cache row 0, and only high-rate frames split that deeply.
pub const MAX_CELT_FRAME_BYTES: usize = 1275;

/// Convert a target bitrate to a CBR payload size for one 20 ms CELT frame.
///
/// `bytes = kbps · 1000 · 0.02 / 8 = kbps · 2.5`, clamped to
/// `[MIN_CELT_FRAME_BYTES, MAX_CELT_FRAME_BYTES]`. The TOC byte is *not*
/// included (it is added by the frame writer), so a 64 kbps request yields a
/// 160-byte payload and a 161-byte packet, and anything at or above 510 kbps
/// saturates at the RFC's 1275-byte frame limit.
pub fn celt_frame_bytes_for_bitrate(target_bitrate_kbps: u32) -> usize {
    let bytes = (target_bitrate_kbps as usize).saturating_mul(5) / 2;
    bytes.clamp(MIN_CELT_FRAME_BYTES, MAX_CELT_FRAME_BYTES)
}

/// Encode one conformant CELT-only mono 20 ms frame at an explicit payload size.
///
/// `target_bytes` is the CBR payload length in bytes, excluding the TOC byte;
/// it is clamped to `[MIN_CELT_FRAME_BYTES, MAX_CELT_FRAME_BYTES]`. Larger
/// frames give the rate allocator more pulses per band and therefore higher
/// reconstruction SNR — see `tests/m_opus_celt_snr.rs` for measured figures.
pub fn encode_celt_frame_conformant_sized(
    prev: &[f32],
    cur: &[f32],
    target_bytes: usize,
) -> Vec<u8> {
    encode_celt_frame_conformant_ranged(prev, cur, target_bytes).0
}

/// [`encode_celt_frame_conformant_sized`] plus the range coder's **final range**.
///
/// The final range register (`OPUS_GET_FINAL_RANGE`) is the canonical libopus
/// conformance check: an encoder and a decoder that consumed exactly the same
/// symbol sequence end with identical `rng`. Any desynchronisation — a missing
/// header flag, a wrong `itheta` layout, an off-by-one in the rate allocator —
/// changes it. `tests/m_opus_celt_conformance.rs` asserts it against the
/// reference decoder for a signal corpus at several frame sizes.
pub fn encode_celt_frame_conformant_ranged(
    prev: &[f32],
    cur: &[f32],
    target_bytes: usize,
) -> (Vec<u8>, u32) {
    // A near-full CBR frame can overflow its own packet by a byte or two when
    // the range-coded stream (front) and the raw-bit stream (back) meet; see
    // `RangeEncoder::finish_to_size_checked`. Shrinking the frame is always
    // safe because the decoder derives its bit budget from the emitted packet
    // length, so a smaller frame is simply a lower-rate — still exactly
    // conformant — encode of the same audio.
    let mut bytes = target_bytes.clamp(MIN_CELT_FRAME_BYTES, MAX_CELT_FRAME_BYTES);
    loop {
        let (packet, range, fits) = encode_celt_frame_try(prev, cur, bytes);
        if fits || bytes <= MIN_CELT_FRAME_BYTES {
            return (packet, range);
        }
        bytes -= 1;
    }
}

/// Encoder-side counterpart of
/// [`CeltFrameParse`](crate::opus_celt_verify::CeltFrameParse).
///
/// The verifier records the range-coder position (Q3) after every named stage
/// of a CELT frame so a desynchronisation can be bisected to a stage instead of
/// showing up only as a mismatched `final_range` at the very end. That is only
/// half an instrument: without the *encoder's* positions there is nothing to
/// compare the verifier's against.
///
/// [`encode_celt_frame_conformant_traced`] fills this in with the same stage
/// names, in the same order, so
/// `trace.stage_tells == parse_celt_frame(..).stage_tells` is a per-stage
/// equality check. It also records the PVQ statistics that bound the coder's
/// dynamic range (largest `K` reaching a leaf, and how many leaves saw `V(N,K)`
/// wrap `u32`).
#[derive(Debug, Clone, Default)]
pub struct CeltEncodeTrace {
    /// Range-coder position (Q3) after each named stage.
    pub stage_tells: Vec<(&'static str, u32)>,
    /// Number of PVQ leaves coded (recursive splits included).
    pub pvq_leaves: usize,
    /// Number of `itheta` split-angle symbols written.
    pub theta_symbols: usize,
    /// Largest pulse count `K` handed to a single PVQ leaf.
    pub max_leaf_pulses: u32,
    /// Number of leaves whose `V(N, K)` wrapped `u32`.
    ///
    /// The decoder has no wide path either, so a wrap costs reconstruction
    /// accuracy for that leaf but does **not** desynchronise the bitstream —
    /// both sides code the symbol over the same wrapped total.
    pub pvq_index_overflows: usize,
    /// Deepest (most negative) partition scale reached by the split recursion.
    ///
    /// Reaches `-1` on wide bands at high rates; that partition must index
    /// pulse-cache row 0.
    pub min_partition_lm: i32,
    /// `false` when the range-coded (front) and raw-bit (back) halves collided
    /// and bytes had to be dropped to hit the target size.
    pub fits: bool,
    /// Emitted payload length in bytes, excluding the TOC byte.
    pub payload_len: usize,
    /// Encoder `final_range` register (`OPUS_GET_FINAL_RANGE`).
    pub final_range: u32,
}

/// [`encode_celt_frame_conformant_ranged`] plus a full per-stage encoder trace.
///
/// `target_bytes` is used **verbatim**: it is not clamped to
/// `[MIN_CELT_FRAME_BYTES, MAX_CELT_FRAME_BYTES]` and no shrink-retry is
/// performed, so the caller sees exactly one encode attempt at exactly the
/// requested size. This is what the conformance sweep needs in order to prove
/// where the bit-exactness bound actually lies rather than measuring the
/// clamp.
///
/// Returns `(packet, trace)`; `packet[0]` is the TOC byte and `packet[1..]` is
/// the `target_bytes`-long payload.
pub fn encode_celt_frame_conformant_traced(
    prev: &[f32],
    cur: &[f32],
    target_bytes: usize,
) -> (Vec<u8>, CeltEncodeTrace) {
    let (packet, _range, _fits, trace) = encode_celt_frame_try_traced(prev, cur, target_bytes);
    (packet, trace)
}

/// One CELT encode attempt at an exact payload size.
///
/// Returns `(packet, final_range, fits)`; `fits == false` means the two halves
/// of the range-coder stream collided and the packet had to drop bytes.
///
/// `target_bytes` is used verbatim (no clamping) — the caller is responsible
/// for keeping it inside `[MIN_CELT_FRAME_BYTES, MAX_CELT_FRAME_BYTES]`.
fn encode_celt_frame_try(prev: &[f32], cur: &[f32], target_bytes: usize) -> (Vec<u8>, u32, bool) {
    let (packet, range, fits, _trace) = encode_celt_frame_try_traced(prev, cur, target_bytes);
    (packet, range, fits)
}

/// [`encode_celt_frame_try`] with the stage trace retained.
fn encode_celt_frame_try_traced(
    prev: &[f32],
    cur: &[f32],
    target_bytes: usize,
) -> (Vec<u8>, u32, bool, CeltEncodeTrace) {
    // Config 31 = CELT-only, Fullband (20 kHz), 20 ms = 960 samples @ 48 kHz.
    // TOC byte: (31 << 3) | stereo=0 | code=0 = 0xF8.
    const TOC: u8 = 0xF8;
    const LM: usize = 3;
    let active_bytes = target_bytes.max(1);
    let target_bits = active_bytes as i32 * 8;

    use crate::opus_mdct::{celt_analysis_spectrum, FRAME_SIZE};
    let mono: Vec<f32> = if cur.len() >= FRAME_SIZE {
        cur[..FRAME_SIZE].to_vec()
    } else {
        let mut v = cur.to_vec();
        v.resize(FRAME_SIZE, 0.0);
        v
    };
    let history: Vec<f32> = if prev.len() >= FRAME_SIZE {
        prev[prev.len() - FRAME_SIZE..].to_vec()
    } else {
        let mut v = vec![0.0f32; FRAME_SIZE - prev.len()];
        v.extend_from_slice(prev);
        v
    };
    let celt_spec = celt_analysis_spectrum(&history, &mono);

    let mut enc = RangeEncoder::new();
    let mut trace = CeltEncodeTrace::default();
    encode_celt_body_into(&celt_spec, 0, true, target_bits, LM, &mut enc, &mut trace);

    let final_range = enc.final_range();
    let (frame_bytes, fits) = enc.finish_to_size_checked(active_bytes);
    trace.final_range = final_range;
    trace.fits = fits;
    trace.payload_len = frame_bytes.len();
    let mut packet = Vec::with_capacity(1 + frame_bytes.len());
    packet.push(TOC);
    packet.extend_from_slice(&frame_bytes);
    (packet, final_range, fits, trace)
}

/// Encode the CELT high-band layer (bands `start_band`..21) into an existing
/// range encoder shared with a SILK layer.
///
/// Used by `encode_hybrid_frame_conformant` to write the CELT layer into the
/// same range-coder bitstream that the SILK WB silence layer already wrote into.
///
/// # RFC 6716 hybrid design
///
/// For hybrid mode (config 12–15) the decoder sets `start_band = 17` and reads
/// both the SILK and CELT layers from a single entropy coder. After SILK decodes,
/// `ec.tell()` is well above 1, so the decoder does **not** read a silence flag
/// for the CELT layer (the `tell == 1` branch is not taken). Post-filter is also
/// skipped because `start != 0`. This function mirrors those decoder expectations:
/// it omits the silence flag and post-filter flag, writing only transient, intra,
/// coarse energy, TF, spread, dynalloc, trim, allocation, fine energy and PVQ
/// shapes for bands `start_band`..21.
///
/// # Bit budget
///
/// `target_bytes` must be the length the **whole shared packet payload** will
/// have once [`RangeEncoder::finish_to_size_checked`] pads it, because the
/// decoder derives `total_bits = frame.len() * 8` from exactly that length and
/// both layers count bits from the start of the shared stream. Passing a
/// constant here while assembling the packet with a variable-length
/// [`RangeEncoder::finish`] — which is what this function did before oxiaudio
/// 0.2.1 — makes the decoder's `total_bits` differ from the encoder's whenever
/// the natural length is not exactly that constant, and the two sides then
/// disagree about the rate allocation. `encode_hybrid_frame_conformant` now
/// passes its CBR target and finishes to that exact size.
///
/// # CELT layer uses `start_band=17` and omits silence flag — decoder returns `Ok(960)`.
pub(crate) fn encode_celt_hybrid_layer_into(
    pcm: &[f32],
    target_bytes: usize,
    enc: &mut RangeEncoder,
    trace: &mut CeltEncodeTrace,
) {
    const LM: usize = 3;
    // First CELT band carried by a hybrid packet (8 kHz crossover).
    const START_BAND_HYBRID: usize = 17;

    use crate::opus_mdct::{celt_analysis_spectrum, FRAME_SIZE};
    let mono: Vec<f32> = if pcm.len() >= FRAME_SIZE {
        pcm[..FRAME_SIZE].to_vec()
    } else {
        let mut v = pcm.to_vec();
        v.resize(FRAME_SIZE, 0.0);
        v
    };
    // The hybrid entry point is stateless, so the lapped MDCT sees a zero
    // history here (documented on `encode_hybrid_frame_conformant`). Pre-emphasis
    // and the analysis gain are applied exactly as on the CELT-only path so the
    // two layers share one coefficient domain.
    let celt_spec = celt_analysis_spectrum(&[], &mono);

    encode_celt_body_into(
        &celt_spec,
        START_BAND_HYBRID,
        false, // no silence flag in hybrid mode
        (target_bytes as i32).saturating_mul(8),
        LM,
        enc,
        trace,
    );
}

/// Inner CELT encoder body: writes headers → energy → TF → spread → dynalloc →
/// trim → allocation → fine energy → PVQ shapes into `enc`.
///
/// This is the shared implementation used by both CELT-only and hybrid paths.
///
/// # Parameters
///
/// * `celt_spec`          — 960 CELT MDCT coefficients from `celt_mdct_960`.
/// * `start_band`         — First CELT band to encode: `0` for CELT-only, `17` for hybrid.
/// * `write_silence_flag` — When `true`, write a silence flag (logp=15) before postfilter.
///   Set to `false` for hybrid mode (decoder skips it when `tell > 1`).
/// * `target_bits`        — Range-coder budget (in bits). Both encoder and decoder compute
///   `total_bits = active_len * 8`; using the same value keeps allocation in sync.
/// * `lm`                 — Frame-size log-scale (3 for 20 ms / 960 samples at 48 kHz).
/// * `enc`                — Range encoder to write into (may already contain SILK bits).
/// * `trace`              — Per-stage bit positions and PVQ statistics; see
///   [`CeltEncodeTrace`].
fn encode_celt_body_into(
    celt_spec: &[f32],
    start_band: usize,
    write_silence_flag: bool,
    target_bits: i32,
    lm: usize,
    enc: &mut RangeEncoder,
    trace: &mut CeltEncodeTrace,
) {
    let target_bits_q = target_bits << BITRES;
    trace.min_partition_lm = lm as i32;

    // ── 2. Per-band energies (log2 scale) ─────────────────────────────────────
    let band_log2 = celt_band_log2_energies(celt_spec, lm);

    // ── 3. Write headers ──────────────────────────────────────────────────────

    // Silence flag (logp=15): only for CELT-only mode.
    // In hybrid mode the decoder does not read this bit because tell > 1 after SILK.
    if write_silence_flag {
        enc.enc_bit_logp(false, 15);
    }
    trace.stage_tells.push(("silence", enc.tell_frac()));

    // Post-filter flag (logp=1): only when start_band == 0 (pure CELT).
    // The hybrid decoder skips this because start != 0.
    if start_band == 0 && enc.tell() + 16 <= target_bits {
        enc.enc_bit_logp(false, 1);
    }
    trace.stage_tells.push(("postfilter", enc.tell_frac()));

    // Transient flag (logp=3): written when lm>0 && budget allows.
    if lm > 0 && enc.tell() + 3 <= target_bits {
        enc.enc_bit_logp(false, 3); // not transient
    }
    trace.stage_tells.push(("transient", enc.tell_frac()));

    // Intra energy flag (logp=3): write true (intra mode for first frame).
    if enc.tell() + 3 <= target_bits {
        enc.enc_bit_logp(true, 3);
    }
    trace.stage_tells.push(("intra", enc.tell_frac()));

    // ── 4. Coarse energy ──────────────────────────────────────────────────────
    // `error[i]` is the residual the fine-energy pass below refines.
    let mut error =
        celt_quant_coarse_energy_intra_ranged(&band_log2, start_band, lm, target_bits, enc);
    trace.stage_tells.push(("coarse", enc.tell_frac()));

    // ── 5. TF (neutral: all unchanged, non-transient) ────────────────────────
    celt_write_tf_neutral_ranged(start_band, lm, false, target_bits, enc);
    trace.stage_tells.push(("tf", enc.tell_frac()));

    // ── 6. Spread decision (SPREAD_NORMAL = symbol 2) ─────────────────────────
    if enc.tell() + 4 <= target_bits {
        enc.enc_icdf(2, &crate::opus_celt_tables::SPREAD_ICDF, 5);
    }
    trace.stage_tells.push(("spread", enc.tell_frac()));

    // ── 7. Dynamic allocation boosts (none) ───────────────────────────────────
    let cap = celt_init_caps(lm);
    let offsets = vec![0i32; NUM_BANDS_CELT];
    let dynalloc_logp = 6i32;
    let mut tell_q = enc.tell_frac() as i32;
    for &cap_val in cap[start_band..NUM_BANDS_CELT].iter() {
        if tell_q + (dynalloc_logp << BITRES) < target_bits_q && cap_val > 0 {
            enc.enc_bit_logp(false, dynalloc_logp as u32);
            tell_q = enc.tell_frac() as i32;
        }
    }
    trace.stage_tells.push(("dynalloc", enc.tell_frac()));

    // ── 8. Allocation trim (neutral = symbol 5) ───────────────────────────────
    tell_q = enc.tell_frac() as i32;
    let alloc_trim = if tell_q + (6 << BITRES) <= target_bits_q {
        enc.enc_icdf(5, &crate::opus_celt_tables::TRIM_ICDF, 7);
        tell_q = enc.tell_frac() as i32;
        5i32
    } else {
        5i32
    };
    trace.stage_tells.push(("trim", enc.tell_frac()));

    // ── 9. Rate allocation ────────────────────────────────────────────────────
    let avail_bits = (target_bits_q - tell_q - 1).max(0);
    let alloc = celt_compute_allocation_enc(
        &mut SkipBitEncoder(enc),
        avail_bits,
        &offsets,
        &cap,
        alloc_trim,
        lm,
        start_band,
    );
    trace.stage_tells.push(("alloc", enc.tell_frac()));

    // ── 10. Fine energy (raw bits at the END of the stream) ──────────────────
    // Mirrors `quant_fine_energy` in libopus `celt/quant_bands.c`. The decoder
    // adds `(q2 + ½)·2^-ebits − ½` to the coarse energy, so the encoder picks
    // the `q2` that best cancels the coarse residual and subtracts the applied
    // offset from `error[i]` for the finalise pass below.
    //
    // Before oxiaudio 0.2.1 these bits were written as zeros, which pinned
    // every band's energy correction to `−½ + 2^-(ebits+1)` log2 units — an
    // up to −0.5 log2 (≈ −3 dB) systematic error on every single band.
    for (i, err) in error
        .iter_mut()
        .enumerate()
        .take(NUM_BANDS_CELT)
        .skip(start_band)
    {
        let ebits = alloc.fine_quant[i];
        if ebits <= 0 {
            continue;
        }
        let frac = (1i32 << ebits) as f32;
        let q2 = (((*err + 0.5) * frac).floor() as i32).clamp(0, (1 << ebits) - 1);
        enc.enc_bits(q2 as u32, ebits as u32);
        let offset = (q2 as f32 + 0.5) / frac - 0.5;
        *err -= offset;
    }
    trace.stage_tells.push(("fine", enc.tell_frac()));

    // ── 11. PVQ band shapes ───────────────────────────────────────────────────
    // Mirror the decoder's `quant_all_bands_mono` balance-tracking loop exactly,
    // then hand each band to `encode_band_with_splits`, which reproduces
    // `quant_partition_mono`'s recursive split (real `itheta` angle coding).
    //
    // Mirrors `quant_all_bands_mono` / `quant_partition_mono` in libopus
    // `celt/bands.c` (BSD-3-Clause).
    let coded_bands = alloc.coded_bands;
    let mut balance = alloc.balance;
    // Scale factor from EBAND_5MS to CELT coefficient index space.
    // At LM=3: m = 1<<3 = 8.
    let celt_scale = 1usize << lm;
    for band in start_band..NUM_BANDS_CELT {
        let tell = enc.tell_frac() as i32;
        if band != start_band {
            balance -= tell;
        }
        let remaining_bits = target_bits_q - tell - 1;
        let b = if band < coded_bands {
            let curr_balance = balance / ((coded_bands - band).min(3) as i32);
            (alloc.pulses[band] + curr_balance)
                .clamp(0, 16_383)
                .min(remaining_bits + 1)
        } else {
            0
        };
        let celt_lo = (EBAND_5MS[band] as usize) * celt_scale;
        let width = (EBAND_5MS[band + 1] - EBAND_5MS[band]) as usize;
        let n0 = width << lm;
        let band_ctx = BandEncCtx { celt_spec, band };
        let mut sink = BandEncSink {
            enc,
            trace,
            remaining_bits,
        };
        encode_band_with_splits(&band_ctx, &mut sink, celt_lo, n0, lm as i32, b);
        balance += alloc.pulses[band] + tell;
    }
    trace.stage_tells.push(("bands", enc.tell_frac()));

    // ── 12. Finalise energy (extra ±½-LSB fine bits, END stream) ─────────────
    // Mirrors `quant_energy_finalise`: the decoder adds
    // `(q2 − ½)·2^-(fine_quant+1)`, so writing `1` when the residual is still
    // positive and `0` when it is negative halves the remaining error.
    let mut bits_left = target_bits - enc.tell();
    'outer: for prio in 0..2i32 {
        for (i, err) in error
            .iter_mut()
            .enumerate()
            .take(NUM_BANDS_CELT)
            .skip(start_band)
        {
            if bits_left < 1 {
                break 'outer;
            }
            if alloc.fine_quant[i] >= MAX_FINE_BITS || alloc.fine_priority[i] != prio {
                continue;
            }
            let q2 = u32::from(*err >= 0.0);
            enc.enc_bits(q2, 1);
            let offset = (q2 as f32 - 0.5) / ((1i64 << (alloc.fine_quant[i] + 1)) as f32);
            *err -= offset;
            bits_left -= 1;
        }
    }
    trace.stage_tells.push(("finalise", enc.tell_frac()));
}

// ── Conformant CELT helpers (ported from libopus, BSD-3-Clause) ──────────────
//
// © 2001–2011 Xiph.Org, Skype Limited, Octasic, Jean-Marc Valin,
// Timothy B. Terriberry, CSIRO, Gregory Maxwell, Mark Borgerding,
// Erik de Castro Lopo.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// · Redistributions of source code must retain the above copyright notice.
// · Redistributions in binary form must reproduce the above copyright notice.
// · Neither the name of the copyright holder nor the names of contributors
//   may be used to endorse or promote products derived from this software.

/// Compute per-band log2 energy from the CELT 960-coefficient MDCT spectrum.
///
/// For each CELT band `i` at frame-level `lm`, the coefficient range is
/// `[EBAND_5MS[i] * (1<<lm), EBAND_5MS[i+1] * (1<<lm))`.  At LM=3 this
/// gives 8-coefficient-wide bands matching the decoder's `denormalise_bands`.
///
/// The value returned is `log2(‖band‖₂)` — the **L2 norm**, matching libopus
/// `compute_band_energies` (`bandE[i] = sqrt(Σ X[j]²)`) and the decoder's
/// `denormalise_bands`, which multiplies a *unit-norm* band shape by
/// `2^(band_loge[i] + E_MEANS[i])`. Using an RMS (`sqrt(Σ/N)`) here instead
/// would attenuate every band by `sqrt(N_i)` — a per-band error of up to
/// 13× at LM=3 — because wider bands would be scaled down harder than
/// narrow ones.
fn celt_band_log2_energies(celt_spec: &[f32], lm: usize) -> Vec<f32> {
    let n_coeff = celt_spec.len();
    let m = 1usize << lm; // = 8 at LM=3
    (0..NUM_BANDS_CELT)
        .map(|i| {
            let lo = (EBAND_5MS[i] as usize) * m;
            let hi = ((EBAND_5MS[i + 1] as usize) * m).min(n_coeff);
            if lo >= n_coeff {
                return -9.0f32;
            }
            let energy: f32 = celt_spec[lo..hi].iter().map(|&x| x * x).sum();
            let norm = energy.sqrt();
            if norm > 1e-20 {
                norm.log2()
            } else {
                -9.0f32
            }
        })
        .collect()
}

/// Encode CELT coarse energies in intra (independent) mode via Laplace coding,
/// for bands `start_band`..`NUM_BANDS_CELT`.
///
/// For intra mode, `coef = 0` and `beta = BETA_INTRA`.  The encoder determines
/// `qi` such that the reconstructed energy approximates `band_log2[i] - E_MEANS[i]`.
///
/// When `start_band = 0` this encodes all 21 bands (CELT-only path).
/// When `start_band = 17` this encodes only bands 17–20 (hybrid CELT path).
///
/// # Returns
///
/// The per-band coarse residual `error[i] = target_i − qi_i` (log2 units, in
/// `[−0.5, 0.5]` whenever `qi` was not clamped), which the fine-energy pass
/// refines. Bands below `start_band` carry `0.0`.
///
/// Mirrors `quant_coarse_energy` in libopus `celt/quant_bands.c` (BSD-3-Clause).
fn celt_quant_coarse_energy_intra_ranged(
    band_log2: &[f32],
    start_band: usize,
    lm: usize,
    total_bits: i32,
    enc: &mut RangeEncoder,
) -> Vec<f32> {
    use crate::opus_celt_tables::{BETA_INTRA, SMALL_ENERGY_ICDF};
    let prob_model = &E_PROB_MODEL[lm][1];
    let mut prev = 0.0f32;
    let mut error = vec![0.0f32; NUM_BANDS_CELT];

    for (i, err) in error
        .iter_mut()
        .enumerate()
        .take(NUM_BANDS_CELT)
        .skip(start_band)
    {
        let tell = enc.tell();
        let target = band_log2[i] - E_MEANS[i.min(24)] - prev;
        // The decoder clamps its reconstruction to [-28, 28]; keep `qi` inside a
        // range the Laplace coder represents and mirror the clamp in `error`.
        let qi = (target.round() as i32).clamp(-28, 28);

        // `qi_coded` is what the decoder will reconstruct: the Laplace coder
        // clamps large magnitudes, and the low-budget fallbacks only represent
        // {-1, 0, +1}. Everything downstream (the `prev` chain and the
        // fine-energy residual) must track the *coded* value, not the request.
        let qi_coded = if total_bits - tell >= 15 {
            let pi = 2 * i.min(20);
            let fs = (prob_model[pi] as u32) << 7;
            let decay = (prob_model[pi + 1] as u32) << 6;
            ec_laplace_encode(enc, qi, fs, decay)
        } else if total_bits - tell >= 2 {
            let s = match qi.cmp(&0) {
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Less => 1,
                std::cmp::Ordering::Greater => 2,
            };
            enc.enc_icdf(s, &SMALL_ENERGY_ICDF, 2);
            qi.signum()
        } else if total_bits - tell >= 1 {
            enc.enc_bit_logp(qi < 0, 1);
            -i32::from(qi < 0)
        } else {
            // Out of budget: the decoder assumes qi = -1 without reading.
            -1
        };

        *err = target - qi_coded as f32;

        // Mirror decoder's prev accumulation: prev += qi * (1 - BETA_INTRA).
        prev += qi_coded as f32 * (1.0 - BETA_INTRA);
    }
    error
}

/// Write neutral TF decisions for bands `start_band`..`NUM_BANDS_CELT`.
///
/// Mirrors the decoder's `tf_decode` in `celt/celt.c` exactly: for
/// `is_transient = false`, `lm = 3`, writes `false` for each band where
/// `tell + logp <= budget`. `tf_select` is not written when both TF_SELECT
/// choices give the same result for the neutral (all-unchanged) case.
///
/// When `start_band = 0` this covers all 21 bands (CELT-only).
/// When `start_band = 17` this covers only bands 17–20 (hybrid).
fn celt_write_tf_neutral_ranged(
    start_band: usize,
    lm: usize,
    is_transient: bool,
    total_bits: i32,
    enc: &mut RangeEncoder,
) {
    use crate::opus_celt_tables::TF_SELECT_TABLE;
    let nb_bands = NUM_BANDS_CELT - start_band;
    let mut budget = total_bits;
    let mut tell = enc.tell();
    let mut logp = if is_transient { 2i32 } else { 4 };
    let tf_select_rsv = lm > 0 && tell + logp < budget;
    if tf_select_rsv {
        budget -= 1;
    }
    // tf_changed and curr stay 0 because we always write false (no TF change).
    let tf_changed = 0i32;
    for _i in 0..nb_bands {
        if tell + logp <= budget {
            // We want tf to stay 0 (no change), so write false.
            enc.enc_bit_logp(false, logp as u32);
            // bit_written = false → curr and tf_changed stay 0
            tell = enc.tell();
        }
        logp = if is_transient { 4 } else { 5 };
    }
    // Write tf_select only if both options differ.
    let idx0 = 4 * usize::from(is_transient) + tf_changed as usize;
    let idx1 = 4 * usize::from(is_transient) + 2 + tf_changed as usize;
    if tf_select_rsv
        && lm < TF_SELECT_TABLE.len()
        && idx0 < TF_SELECT_TABLE[lm].len()
        && idx1 < TF_SELECT_TABLE[lm].len()
        && TF_SELECT_TABLE[lm][idx0] != TF_SELECT_TABLE[lm][idx1]
    {
        enc.enc_bit_logp(false, 1); // tf_select = 0
    }
}

// ── Internal implementation ───────────────────────────────────────────────────

/// Core CELT frame encoder.
///
/// `bits_per_band` controls how many PVQ pulses are allocated per band.
fn encode_celt_frame_inner(
    pcm: &[f32],
    channels: usize,
    bits_per_band: u32,
    enc: &mut RangeEncoder,
) {
    // ── Step 1: Extract mono from interleaved PCM and run MDCT ────────────────
    let mono: Vec<f32> = if channels <= 1 {
        pcm.to_vec()
    } else {
        pcm.chunks_exact(channels)
            .map(|frame| frame[0] * 0.5 + frame[1] * 0.5)
            .collect()
    };

    // Ensure we have exactly FRAME_SIZE samples (pad with silence if short).
    use crate::opus_mdct::FRAME_SIZE;
    let mono_padded: Vec<f32> = if mono.len() >= FRAME_SIZE {
        mono[..FRAME_SIZE].to_vec()
    } else {
        let mut v = mono;
        v.resize(FRAME_SIZE, 0.0);
        v
    };

    let spectrum = mdct_forward(&mono_padded);

    // ── Step 2: Compute global gain (RMS of the full spectrum in dB) ──────────
    let rms: f32 = (spectrum.iter().map(|&x| x * x).sum::<f32>() / spectrum.len() as f32).sqrt();
    let global_gain_db = if rms > 1e-20 {
        20.0 * rms.log10()
    } else {
        -120.0_f32
    };

    // ── Step 3: Encode the 21 CELT bands ─────────────────────────────────────
    for band_idx in 0..NUM_BANDS {
        let lo = BAND_BINS[band_idx];
        let hi = BAND_BINS[band_idx + 1].min(spectrum.len());
        if lo >= spectrum.len() {
            break;
        }

        let band_coeffs = &spectrum[lo..hi];
        let norm = band_norm(band_coeffs);

        // 3a. Encode band energy (4-bit, 16 levels).
        let energy_q = quantize_band_energy(norm, global_gain_db);
        enc.encode_uint(energy_q as u32, 16);

        // 3b. Encode band shape via PVQ.
        let normalized = normalize_band(band_coeffs);
        let k = compute_k_pulses(band_coeffs.len(), bits_per_band);
        if k > 0 {
            let y = pvq_encode(&normalized, k);
            let _overflowed = opus_pvq::encode_pulses(enc, &y);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        band_norm, encode_celt_frame, encode_celt_frame_pvq, normalize_band, pvq_encode, BAND_BINS,
        NUM_BANDS,
    };
    use crate::opus_range::RangeEncoder;

    // ── Existing CELT frame tests (kept from the previous scaffold) ────────────

    fn silence_frame(channels: usize) -> Vec<f32> {
        vec![0.0f32; 960 * channels]
    }

    fn sine_frame(channels: usize) -> Vec<f32> {
        let n = 960 * channels;
        (0..n)
            .map(|i| {
                let sample_idx = i / channels;
                (2.0 * std::f32::consts::PI * 440.0 * sample_idx as f32 / 48_000.0).sin() * 0.5
            })
            .collect()
    }

    #[test]
    fn test_encode_celt_frame_silence_does_not_panic() {
        let pcm = silence_frame(1);
        let mut enc = RangeEncoder::new();
        encode_celt_frame(&pcm, 1, &mut enc);
        let bytes = enc.finish();
        assert!(
            !bytes.is_empty(),
            "CELT encoding must produce output even for silence"
        );
    }

    #[test]
    fn test_encode_celt_frame_stereo_does_not_panic() {
        let pcm = sine_frame(2);
        let mut enc = RangeEncoder::new();
        encode_celt_frame(&pcm, 2, &mut enc);
        let bytes = enc.finish();
        assert!(
            !bytes.is_empty(),
            "CELT stereo encoding must produce output"
        );
    }

    #[test]
    fn test_encode_celt_frame_sine_produces_output() {
        let pcm_silence = silence_frame(1);
        let mut enc_s = RangeEncoder::new();
        encode_celt_frame(&pcm_silence, 1, &mut enc_s);
        let silence_bytes = enc_s.finish().len();

        let pcm_sine = sine_frame(1);
        let mut enc_n = RangeEncoder::new();
        encode_celt_frame(&pcm_sine, 1, &mut enc_n);
        let sine_bytes = enc_n.finish().len();

        assert!(silence_bytes > 0 && sine_bytes > 0);
    }

    // ── Band constant sanity ────────────────────────────────────────────────────

    #[test]
    fn test_band_bins_count() {
        assert_eq!(
            BAND_BINS.len(),
            NUM_BANDS + 1,
            "BAND_BINS must have NUM_BANDS + 1 entries"
        );
    }

    #[test]
    fn test_band_bins_monotone() {
        for i in 0..NUM_BANDS {
            assert!(
                BAND_BINS[i] < BAND_BINS[i + 1],
                "BAND_BINS must be strictly increasing at index {i}"
            );
        }
    }

    // ── band_norm ───────────────────────────────────────────────────────────────

    #[test]
    fn test_band_norm_unit_vector() {
        let v = vec![0.6f32, 0.8];
        let norm = band_norm(&v);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "unit vector norm must be 1.0, got {norm}"
        );
    }

    #[test]
    fn test_band_norm_zero_vector() {
        let v = vec![0.0f32; 4];
        let norm = band_norm(&v);
        assert_eq!(norm, 0.0, "zero vector norm must be 0.0");
    }

    #[test]
    fn test_band_norm_single_element() {
        let v = vec![3.0f32];
        let norm = band_norm(&v);
        assert!((norm - 3.0).abs() < 1e-6, "single-element norm, got {norm}");
    }

    // ── normalize_band ──────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_band_unit_output() {
        let v = vec![3.0f32, 4.0];
        let n = normalize_band(&v);
        let out_norm = band_norm(&n);
        assert!(
            (out_norm - 1.0).abs() < 1e-5,
            "normalized vector must have unit norm, got {out_norm}"
        );
    }

    #[test]
    fn test_normalize_band_zero_input() {
        let v = vec![0.0f32; 4];
        let n = normalize_band(&v);
        assert!(
            n.iter().all(|&x| x == 0.0),
            "zero input normalizes to zeros"
        );
    }

    // ── pvq_encode ──────────────────────────────────────────────────────────────

    #[test]
    fn test_pvq_encode_output_length() {
        let normalized = vec![0.5f32, -0.5, 0.5, -0.5];
        let y = pvq_encode(&normalized, 4);
        assert_eq!(y.len(), 4, "PVQ output must match input length");
    }

    #[test]
    fn test_pvq_encode_sums_to_k() {
        let normalized = vec![0.6f32, 0.4, 0.0, 0.0];
        let y = pvq_encode(&normalized, 5);
        let l1: i32 = y.iter().map(|&x| x.abs()).sum();
        assert_eq!(l1, 5, "PVQ L1 norm must equal K pulses, got {l1}");
    }

    #[test]
    fn test_pvq_encode_silent_input() {
        let normalized = vec![0.0f32; 8];
        let y = pvq_encode(&normalized, 0);
        assert!(
            y.iter().all(|&x| x == 0),
            "zero pulses must produce all-zero output"
        );
    }

    #[test]
    fn test_pvq_encode_zero_k_is_all_zeros() {
        let normalized = vec![0.5f32, 0.5, 0.0, 0.0];
        let y = pvq_encode(&normalized, 0);
        assert!(
            y.iter().all(|&x| x == 0),
            "k=0 must produce all-zero output"
        );
    }

    #[test]
    fn test_pvq_encode_single_pulse_goes_to_largest() {
        // With k=1, the single pulse must land on the largest-magnitude dimension.
        let normalized = vec![0.1f32, 0.9, 0.3, 0.0];
        let y = pvq_encode(&normalized, 1);
        let l1: i32 = y.iter().map(|&x| x.abs()).sum();
        assert_eq!(l1, 1, "single pulse: L1 must be 1");
        // The pulse should land on index 1 (largest magnitude = 0.9).
        assert_eq!(
            y[1].abs(),
            1,
            "single pulse should land on highest-magnitude dimension"
        );
    }

    #[test]
    fn test_pvq_encode_sign_preservation() {
        // Negative component should get negative pulse.
        let normalized = vec![-0.9f32, 0.1, 0.0, 0.0];
        let y = pvq_encode(&normalized, 1);
        assert!(y[0] < 0, "pulse on negative component must be negative");
    }

    // ── encode_celt_frame_pvq ───────────────────────────────────────────────────

    #[test]
    fn test_celt_frame_with_pvq_nonempty() {
        let pcm: Vec<f32> = (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let encoded = encode_celt_frame_pvq(&pcm, 1, 48_000, 64);
        assert!(
            !encoded.is_empty(),
            "PVQ-encoded CELT frame must not be empty"
        );
    }

    #[test]
    fn test_celt_frame_pvq_differs_from_silence() {
        let sine: Vec<f32> = (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let silence = vec![0.0f32; 960];
        let enc_sine = encode_celt_frame_pvq(&sine, 1, 48_000, 64);
        let enc_silence = encode_celt_frame_pvq(&silence, 1, 48_000, 64);
        // Non-silence input should produce a different bitstream than silence
        // (the band energies differ even if the PVQ shapes are degenerate).
        assert_ne!(
            enc_sine, enc_silence,
            "sine and silence should produce different CELT frames"
        );
    }

    #[test]
    fn test_celt_frame_pvq_two_freqs_differ() {
        // Two non-silent sine waves at different frequencies should produce different
        // bitstreams — this exercises both the energy quantization AND the PVQ shape
        // path (the energy peaks land in different CELT bands).
        let sine_440: Vec<f32> = (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let sine_8k: Vec<f32> = (0..960)
            .map(|i| (2.0 * std::f32::consts::PI * 8_000.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();
        let enc_440 = encode_celt_frame_pvq(&sine_440, 1, 48_000, 64);
        let enc_8k = encode_celt_frame_pvq(&sine_8k, 1, 48_000, 64);
        assert_ne!(
            enc_440, enc_8k,
            "440 Hz and 8 kHz sines must produce different CELT bitstreams"
        );
    }

    #[test]
    fn test_celt_frame_pvq_bits_per_band_nonzero() {
        // Verify the bit-budget formula gives meaningful K values (K > 0) for typical
        // bands with the standard 64 kbps / 48 kHz setting.
        // bits_per_frame = 64 * 960 * 1000 / 48000 = 1280, bits_per_band = 1280/21 = 60.
        // N=1: K = floor(60 / log2(2)) = 60.
        // N=6: K = floor(60 / log2(7)) ≈ 21.
        // Both are non-zero, meaning PVQ shape is emitted for all bands.
        use super::compute_k_pulses;
        let bits_per_frame: u32 = 64_u32.saturating_mul(960).saturating_mul(1000) / 48_000;
        let bits_per_band = (bits_per_frame / 21).max(1);
        // All bands must get K ≥ 1 at 64 kbps so that shape information is emitted.
        for band_size in [1usize, 2, 4, 6, 8, 12, 18, 22] {
            let k = compute_k_pulses(band_size, bits_per_band);
            assert!(
                k >= 1,
                "band_size={band_size}: K must be ≥ 1 at 64 kbps, got K={k}"
            );
        }
    }

    #[test]
    fn test_celt_frame_pvq_stereo_nonempty() {
        let pcm: Vec<f32> = (0..1920)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * (i / 2) as f32 / 48_000.0).sin() * 0.3)
            .collect();
        let encoded = encode_celt_frame_pvq(&pcm, 2, 48_000, 128);
        assert!(
            !encoded.is_empty(),
            "stereo PVQ CELT frame must not be empty"
        );
    }
}
