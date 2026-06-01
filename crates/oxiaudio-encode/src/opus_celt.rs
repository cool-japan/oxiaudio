//! CELT (Constrained Energy Lapped Transform) encoder for Opus.
//!
//! Implements band energy quantization and spectral coefficient encoding via
//! Pyramid Vector Quantization (PVQ), following RFC 6716 §4.3 structure.
//!
//! # Conformance note
//!
//! This encoder is structurally closer to RFC 6716 than the previous 4-bit
//! placeholder, but is **not fully conformant**: PVQ is done with a greedy
//! pulse allocator (not the exact combinatorial search in RFC 6716 §4.3.4.6),
//! and the range coder is our private self-consistent variant rather than the
//! RFC bit-reversal packing. The resulting bitstream will NOT be decoded
//! by a standard Opus decoder.
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

use crate::opus_mdct::mdct_forward;
use crate::opus_range::RangeEncoder;

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

/// Encode a PVQ band shape vector into the range encoder.
///
/// Each non-zero coefficient contributes its unsigned magnitude (in [0, K])
/// followed by a sign bit (0 = positive, 1 = negative).  Coefficients
/// where `|y[k]| == 0` do not emit a sign bit.
fn encode_pvq_shape(enc: &mut RangeEncoder, y: &[i32], k_pulses: u32) {
    if k_pulses == 0 {
        return;
    }
    let alphabet = k_pulses + 1; // magnitudes live in [0, k_pulses]
    for &yi in y {
        let mag = yi.unsigned_abs();
        enc.encode_uint(mag, alphabet);
        if mag > 0 {
            let sign = u32::from(yi < 0);
            enc.encode_uint(sign, 2);
        }
    }
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
            encode_pvq_shape(enc, &y, k);
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
