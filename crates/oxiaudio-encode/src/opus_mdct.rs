//! MDCT analysis for the CELT Opus encoder.
//!
//! Implements a 960-point Modified Discrete Cosine Transform (MDCT) for 20 ms
//! frames at 48 kHz, producing N/2 = 480 spectral coefficients. OxiFFT's
//! free-function `fft` is used for the underlying DFT of the N/2 = 480-point
//! pre-rotated signal.
//!
//! The MDCT algorithm follows the standard pre-rotation / FFT / post-rotation
//! decomposition described in Malvar, "Signal Processing with Lapped Transforms"
//! (Artech House, 1992), §3.2.

use oxifft::{fft, Complex};

/// Number of PCM samples per CELT frame (20 ms at 48 kHz).
pub const FRAME_SIZE: usize = 960;

/// MDCT analysis: transform `FRAME_SIZE` PCM samples to `FRAME_SIZE / 2` spectral coefficients.
///
/// Applies a sine window, pre-rotates the signal, computes a `FRAME_SIZE/2`-point complex FFT
/// via OxiFFT, then post-rotates to obtain real MDCT coefficients.
///
/// `samples` need not be exactly [`FRAME_SIZE`] long: shorter input is zero-padded
/// and longer input is truncated (a single `Vec::resize`), so this function is
/// infallible and panic-free for any input length. This mirrors the
/// tolerant-length convention already used by the conformant CELT path
/// (`encode_celt_frame_conformant` in `opus_celt.rs`), and matters because this
/// is a `pub fn` on a `pub mod` reachable by any external caller with no
/// length guarantee on their buffer.
pub fn mdct_forward(samples: &[f32]) -> Vec<f32> {
    let owned;
    let samples: &[f32] = if samples.len() == FRAME_SIZE {
        samples
    } else {
        let mut v = samples.to_vec();
        v.resize(FRAME_SIZE, 0.0);
        owned = v;
        &owned
    };

    let n = FRAME_SIZE;
    let n2 = n / 2; // 480

    // ── Step 1: Sine window ───────────────────────────────────────────────────
    //
    // w[k] = sin(π · (k + ½) / N), k ∈ [0, N)
    let windowed: Vec<f32> = samples
        .iter()
        .enumerate()
        .map(|(k, &s)| {
            let w = (std::f32::consts::PI * (k as f32 + 0.5) / n as f32).sin();
            s * w
        })
        .collect();

    // ── Step 2: Pre-rotation by exp(−j·π·k/N) ────────────────────────────────
    //
    // For each k ∈ [0, N/2) build a complex sample:
    //   re_in[k] = windowed[2k]
    //   im_in[k] = windowed[N − 1 − 2k]
    // then multiply by exp(−jπk/N).
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

    // ── Step 3: N/2-point forward FFT (OxiFFT free function) ─────────────────
    let spectrum = fft(&pre_rotated);

    // ── Step 4: Post-rotation and extraction ─────────────────────────────────
    //
    // X[k] = 2 · Re{ spectrum[k] · exp(−jπ(k + ½)/N) }
    (0..n2)
        .map(|k| {
            let angle = -std::f32::consts::PI * (k as f32 + 0.5) / n as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            2.0 * (spectrum[k].re * cos_a - spectrum[k].im * sin_a)
        })
        .collect()
}

/// CELT Vorbis overlap window — the rising half of the 120-sample transition.
///
/// Values mirror `WINDOW_120` in `modes.rs` of the reference `opus-decoder`
/// crate (BSD-3-Clause; © Xiph.Org Foundation et al.).
pub const CELT_WINDOW_120: [f32; 120] = [
    6.7286966e-05,
    0.00060551348,
    0.001_681_597,
    0.0032947962,
    0.0054439943,
    0.008_127_692,
    0.011344001,
    0.015090633,
    0.019364886,
    0.024163635,
    0.029483315,
    0.035319905,
    0.041_668_91,
    0.048_525_35,
    0.055883718,
    0.063737999,
    0.072_081_62,
    0.080_907_43,
    0.090_207_7,
    0.099_974_11,
    0.11019769,
    0.12086883,
    0.13197729,
    0.14351214,
    0.15546177,
    0.167_813_9,
    0.180_555_5,
    0.193_672_9,
    0.20715171,
    0.22097682,
    0.23513243,
    0.24960208,
    0.264_368_6,
    0.27941419,
    0.294_720_4,
    0.310_268_2,
    0.32603788,
    0.342_009_3,
    0.35816177,
    0.37447407,
    0.39092462,
    0.40749142,
    0.42415215,
    0.44088423,
    0.45766484,
    0.47447104,
    0.49127978,
    0.50806798,
    0.52481261,
    0.541_490_8,
    0.558_079_7,
    0.574_557,
    0.590_900_5,
    0.607_088_4,
    0.623_099_5,
    0.63891306,
    0.65450896,
    0.66986776,
    0.684_970_8,
    0.699_800_1,
    0.714_338_7,
    0.728_570_5,
    0.74248043,
    0.756_054_2,
    0.76927895,
    0.782_142_6,
    0.794_634_3,
    0.80674445,
    0.818_464_6,
    0.829_787_3,
    0.840_706_7,
    0.851_217_8,
    0.861_317,
    0.87100183,
    0.88027111,
    0.889_124_8,
    0.897_564,
    0.90559094,
    0.913_209,
    0.920_422_7,
    0.927_237_4,
    0.93365955,
    0.93969656,
    0.945_356_7,
    0.950_649_1,
    0.955_583_5,
    0.960_170_7,
    0.964_421_7,
    0.968_348_5,
    0.97196334,
    0.97527906,
    0.97830883,
    0.98106616,
    0.983_564_8,
    0.985_818_7,
    0.987_841_9,
    0.989_648_6,
    0.991_252_7,
    0.992_668_5,
    0.993_909_7,
    0.99499004,
    0.995_923,
    0.996_721_6,
    0.99739874,
    0.99796667,
    0.998_437_3,
    0.998_822,
    0.99913147,
    0.99937606,
    0.99956527,
    0.999_708,
    0.999_812_5,
    0.99988613,
    0.999_935_6,
    0.999_967,
    0.99998518,
    0.999_994_6,
    0.99999859,
    0.999_999_8,
    1.0000000,
];

/// CELT MDCT geometry: number of spectral bins produced per 20 ms frame.
pub const CELT_MDCT_BINS: usize = 960;

/// CELT MDCT geometry: length of the lapped analysis block (`2 · CELT_MDCT_BINS`).
const CELT_MDCT_BLOCK: usize = 2 * CELT_MDCT_BINS;

/// CELT low-overlap transition length in samples (RFC 6716 §4.3.7).
const CELT_OVERLAP: usize = 120;

/// Analysis-window value at block position `m` of the 1920-sample lapped block.
///
/// The CELT low-overlap window is 1 across the central `2N − 2·(N/2 − overlap/2)`
/// samples and tapers over `overlap` samples centred on `N/2` and `3N/2`
/// (RFC 6716 §4.3.7 / libopus `celt/modes.c`):
///
/// ```text
/// m ∈ [0,   420) → 0                       (zero-padded left tail)
/// m ∈ [420, 540) → CELT_WINDOW_120[m − 420] (rising)
/// m ∈ [540, 1380) → 1                       (flat)
/// m ∈ [1380, 1500) → CELT_WINDOW_120[1499 − m] (falling)
/// m ∈ [1500, 1920) → 0                      (zero-padded right tail)
/// ```
fn celt_lapped_window(m: usize) -> f32 {
    const FLAT_LO: usize = CELT_MDCT_BINS / 2 - CELT_OVERLAP / 2; // 420
    const RISE_HI: usize = FLAT_LO + CELT_OVERLAP; // 540
    const FALL_LO: usize = 3 * CELT_MDCT_BINS / 2 - CELT_OVERLAP / 2; // 1380
    const FALL_HI: usize = FALL_LO + CELT_OVERLAP; // 1500
    if !(FLAT_LO..FALL_HI).contains(&m) {
        0.0
    } else if m < RISE_HI {
        CELT_WINDOW_120[m - FLAT_LO]
    } else if m < FALL_LO {
        1.0
    } else {
        CELT_WINDOW_120[FALL_HI - 1 - m]
    }
}

/// Naive reference MDCT used to validate [`celt_mdct_960_overlap`]'s fast path.
///
/// Evaluates the defining sum directly:
/// `X[k] = Σ_m w[m]·x[m]·cos(π/N·(m + ½ + N/2)·(k + ½))`.
#[cfg(test)]
fn celt_mdct_960_overlap_naive(prev: &[f32], cur: &[f32]) -> Vec<f32> {
    const N: usize = CELT_MDCT_BINS;
    let block: Vec<f32> = (0..CELT_MDCT_BLOCK)
        .map(|m| {
            let s = if m < N {
                prev.get(m).copied().unwrap_or(0.0)
            } else {
                cur.get(m - N).copied().unwrap_or(0.0)
            };
            s * celt_lapped_window(m)
        })
        .collect();
    (0..N)
        .map(|k| {
            let mut acc = 0.0f64;
            for (m, &v) in block.iter().enumerate() {
                if v == 0.0 {
                    continue;
                }
                let phase = std::f64::consts::PI / N as f64
                    * (m as f64 + 0.5 + N as f64 / 2.0)
                    * (k as f64 + 0.5);
                acc += v as f64 * phase.cos();
            }
            acc as f32
        })
        .collect()
}

/// CELT pre-emphasis coefficient (libopus `mode->preemph[0]` at 48 kHz).
///
/// The encoder applies `p[n] = x[n] − COEF·x[n−1]`; the decoder's mandatory
/// de-emphasis (`y[n] = p[n] + COEF·y[n−1]`) is its exact inverse. Skipping
/// pre-emphasis on the analysis side leaves the decoder's low-pass de-emphasis
/// uncompensated, which tilts the reconstruction by ~11× from 100 Hz to
/// 18 kHz — audible as a heavily muffled decode even when every coded field is
/// bit-exact.
pub const CELT_PREEMPH_COEF: f32 = 0.850_006_1;

/// CELT internal signal scale (`CELT_SIG_SCALE` in libopus).
///
/// libopus runs CELT in a ±32768 signal domain and the reference decoder
/// divides its `i16` output by 32768 to produce floats.
const CELT_SIG_SCALE: f32 = 32768.0;

/// Analysis gain that puts [`celt_analysis_spectrum`]'s output in the same
/// domain as the reference decoder's `denormalise_bands` input.
///
/// Two factors combine:
/// * `CELT_SIG_SCALE` — the ±32768 internal domain described above.
/// * `2 / CELT_MDCT_BINS` — the MDCT normalisation. [`celt_mdct_960_overlap`]
///   evaluates the unnormalised defining sum, while the decoder's
///   `clt_mdct_backward` is likewise unnormalised, so the whole `2/N` of the
///   analysis/synthesis pair has to live on the analysis side.
///
/// Verified end-to-end by `tests/m_opus_celt_snr.rs`, which asserts the decoded
/// level matches the input within ±2 dB across tones and noise.
const CELT_ANALYSIS_GAIN: f32 = CELT_SIG_SCALE * 2.0 / CELT_MDCT_BINS as f32;

/// Pre-emphasise and scale one lapped 1920-sample analysis block.
///
/// Returns `CELT_ANALYSIS_GAIN · (x[m] − COEF·x[m−1])` over the block
/// `[prev | cur]`. `x[-1]` is taken as 0; that start-up transient sits at block
/// position 0, where [`celt_lapped_window`] is exactly zero, so it never
/// reaches the transform.
fn celt_preemphasised_block(prev: &[f32], cur: &[f32]) -> Vec<f32> {
    let n = CELT_MDCT_BINS;
    let sample = |m: usize| -> f32 {
        if m < n {
            prev.get(m).copied().unwrap_or(0.0)
        } else {
            cur.get(m - n).copied().unwrap_or(0.0)
        }
    };
    (0..CELT_MDCT_BLOCK)
        .map(|m| {
            let prev_sample = if m == 0 { 0.0 } else { sample(m - 1) };
            CELT_ANALYSIS_GAIN * (sample(m) - CELT_PREEMPH_COEF * prev_sample)
        })
        .collect()
}

/// Full CELT analysis front-end: pre-emphasis, gain and lapped MDCT.
///
/// This is what the conformant CELT encoder feeds to band-energy computation
/// and PVQ shape coding. `prev` is the previous 960-sample frame (empty at
/// stream start) and `cur` is the frame being coded.
pub fn celt_analysis_spectrum(prev: &[f32], cur: &[f32]) -> Vec<f32> {
    let block = celt_preemphasised_block(prev, cur);
    celt_mdct_960_block(&block)
}

/// Forward CELT MDCT analysis of one 20 ms frame with the previous frame as
/// overlap history.
///
/// This is the analysis transform whose synthesis counterpart is the reference
/// decoder's `clt_mdct_backward` + overlap-add, i.e. the pair satisfies
/// time-domain alias cancellation (TDAC). `prev` is the previous 960-sample
/// frame (all zeros for the first frame); `cur` is the frame being coded. Both
/// are zero-padded / truncated to 960 samples.
///
/// # Algorithm
///
/// 1. Window the 1920-sample lapped block `[prev | cur]` with
///    `celt_lapped_window`.
/// 2. Fold the four quarters `[a | b | c | d]` into `u = [−c_R − d | a − b_R]`
///    (the standard MDCT → DCT-IV reduction).
/// 3. Evaluate the size-960 DCT-IV with a 480-point complex FFT using the
///    `+1/8` twiddle convention that matches the decoder's trig table.
///
/// # Returns
///
/// 960 MDCT bins (25 Hz apart at 48 kHz) in the coefficient space consumed by
/// `denormalise_bands` in the reference decoder.
pub fn celt_mdct_960_overlap(prev: &[f32], cur: &[f32]) -> Vec<f32> {
    let raw: Vec<f32> = (0..CELT_MDCT_BLOCK)
        .map(|m| {
            if m < CELT_MDCT_BINS {
                prev.get(m).copied().unwrap_or(0.0)
            } else {
                cur.get(m - CELT_MDCT_BINS).copied().unwrap_or(0.0)
            }
        })
        .collect();
    celt_mdct_960_block(&raw)
}

/// Lapped MDCT of an already-assembled 1920-sample analysis block.
///
/// Applies `celt_lapped_window`, folds the four quarters into the size-960
/// DCT-IV input and evaluates it with a 480-point complex FFT.
pub fn celt_mdct_960_block(raw: &[f32]) -> Vec<f32> {
    const N: usize = CELT_MDCT_BINS; // 960 output bins
    const HALF: usize = N / 2; // 480
    const L: usize = N / 2; // FFT length

    // ── 1. Window the lapped 1920-sample block ────────────────────────────────
    let block: Vec<f32> = (0..CELT_MDCT_BLOCK)
        .map(|m| raw.get(m).copied().unwrap_or(0.0) * celt_lapped_window(m))
        .collect();

    // ── 2. Fold [a|b|c|d] → u = [−c_R − d | a − b_R] ─────────────────────────
    let mut u = vec![0.0f32; N];
    for n in 0..HALF {
        // c[HALF-1-n] = block[N + HALF - 1 - n]; d[n] = block[N + HALF + n]
        u[n] = -block[N + HALF - 1 - n] - block[N + HALF + n];
        // a[n] = block[n]; b[HALF-1-n] = block[HALF + HALF - 1 - n]
        u[HALF + n] = block[n] - block[N - 1 - n];
    }

    // ── 3. DCT-IV of size N via an L-point complex FFT ───────────────────────
    let pre: Vec<Complex<f32>> = (0..L)
        .map(|r| {
            let a = u[2 * r];
            let b = u[N - 1 - 2 * r];
            let angle = -std::f32::consts::PI * (r as f32 + 0.125) / N as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            Complex {
                re: a * cos_a - b * sin_a,
                im: a * sin_a + b * cos_a,
            }
        })
        .collect();
    let spectrum = fft(&pre);

    let mut out = vec![0.0f32; N];
    for k in 0..L {
        let angle = -std::f32::consts::PI * (k as f32 + 0.125) / N as f32;
        let (sin_a, cos_a) = angle.sin_cos();
        let re = spectrum[k].re * cos_a - spectrum[k].im * sin_a;
        let im = spectrum[k].re * sin_a + spectrum[k].im * cos_a;
        out[2 * k] = re;
        out[N - 1 - 2 * k] = -im;
    }
    out
}

/// Forward CELT MDCT analysis for a single 960-sample first frame.
///
/// Produces the 960 spectral coefficients in the **CELT 1920-point MDCT**
/// space used by `quant_all_bands_mono` / `denormalise_bands` in the
/// reference decoder.  The overlap buffer is assumed zero (first-frame /
/// intra mode).
///
/// # Analysis window
///
/// ```text
/// w[m] = CELT_WINDOW_120[m]           m ∈ [0,   120)   rising
/// w[m] = 1.0                           m ∈ [120,  840)   flat
/// w[m] = CELT_WINDOW_120[959 − m]     m ∈ [840,  960)   falling
/// ```
///
/// # MDCT formula
///
/// With `a[m] = pcm[m] · w[m]` and `N = 960`:
/// ```text
/// X[k] = Σ_{m=0}^{959} a[m] · cos( π · (k + ½) · (m + 1440.5) / N )
/// ```
/// This is the right-half (current frame) contribution of a 1920-point
/// MDCT whose left half (previous frame) is zero.
pub fn celt_mdct_960(pcm: &[f32]) -> Vec<f32> {
    const N: usize = 960;
    let len = pcm.len().min(N);

    // Build windowed signal a[m] = pcm[m] · w[m].
    let a: Vec<f32> = (0..N)
        .map(|m| {
            let s = if m < len { pcm[m] } else { 0.0 };
            let w: f32 = if m < 120 {
                CELT_WINDOW_120[m]
            } else if m >= 840 {
                CELT_WINDOW_120[959 - m]
            } else {
                1.0
            };
            s * w
        })
        .collect();

    // Naive O(N²) MDCT.  For N=960 this is ~0.9 M multiply-adds — fast enough
    // for offline encoding and all tests.
    let n_f = N as f32;
    (0..N)
        .map(|k| {
            let scale = std::f32::consts::PI * (k as f32 + 0.5) / n_f;
            a.iter()
                .enumerate()
                .map(|(m, &am)| am * (scale * (m as f32 + 1440.5)).cos())
                .sum::<f32>()
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        celt_mdct_960_overlap, celt_mdct_960_overlap_naive, mdct_forward, CELT_MDCT_BINS,
        FRAME_SIZE,
    };

    /// The FFT-accelerated lapped MDCT must agree with the defining sum.
    #[test]
    fn test_celt_mdct_overlap_matches_naive_reference() {
        // Deterministic pseudo-random content in both the history and the frame.
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1 << 24) as f32) - 0.5
        };
        let prev: Vec<f32> = (0..CELT_MDCT_BINS).map(|_| next()).collect();
        let cur: Vec<f32> = (0..CELT_MDCT_BINS).map(|_| next()).collect();

        let fast = celt_mdct_960_overlap(&prev, &cur);
        let slow = celt_mdct_960_overlap_naive(&prev, &cur);
        assert_eq!(fast.len(), CELT_MDCT_BINS);
        assert_eq!(slow.len(), CELT_MDCT_BINS);

        let scale = slow.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        for k in 0..CELT_MDCT_BINS {
            let err = (fast[k] - slow[k]).abs();
            assert!(
                err <= 2e-3 * scale.max(1.0),
                "bin {k}: fast={} slow={} (err {err:.6}, scale {scale:.3})",
                fast[k],
                slow[k]
            );
        }
    }

    /// Generate a sine wave at `freq_hz` Hz into a `FRAME_SIZE`-sample buffer.
    fn sine_frame(freq_hz: f32) -> Vec<f32> {
        (0..FRAME_SIZE)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / 48_000.0).sin() * 0.5)
            .collect()
    }

    #[test]
    fn test_mdct_forward_output_length() {
        let samples = sine_frame(440.0);
        let spec = mdct_forward(&samples);
        assert_eq!(
            spec.len(),
            FRAME_SIZE / 2,
            "mdct_forward must return N/2 coefficients"
        );
    }

    #[test]
    fn test_mdct_forward_silence_is_near_zero() {
        let samples = vec![0.0f32; FRAME_SIZE];
        let spec = mdct_forward(&samples);
        let max = spec.iter().copied().fold(0.0f32, f32::max);
        assert!(
            max.abs() < 1e-5,
            "silence input must produce near-zero MDCT output, max={max}"
        );
    }

    #[test]
    fn test_mdct_forward_non_zero_for_sine() {
        let samples = sine_frame(1000.0);
        let spec = mdct_forward(&samples);
        // At least some coefficients must be non-trivial.
        let energy: f32 = spec.iter().map(|&x| x * x).sum();
        assert!(
            energy > 0.01,
            "sine wave must produce non-zero MDCT energy, got {energy}"
        );
    }

    #[test]
    fn test_mdct_forward_energy_conservation() {
        // Energy in the spectral domain should be proportional to time-domain energy.
        // Parseval: sum(X[k]^2) ≈ N/2 · sum(x[k]^2) for a sine-windowed MDCT.
        let samples = sine_frame(440.0);
        let time_energy: f32 = samples.iter().map(|&x| x * x).sum();
        let spec = mdct_forward(&samples);
        let spec_energy: f32 = spec.iter().map(|&x| x * x).sum();
        // Allow a wide tolerance because the window + normalization factor are approximate.
        // The ratio should be in a reasonable range, not off by orders of magnitude.
        let ratio = spec_energy / time_energy.max(1e-12);
        assert!(
            ratio > 1.0 && ratio < 2000.0,
            "MDCT energy ratio {ratio:.1} is out of expected range [1, 2000]"
        );
    }

    /// Regression test: `mdct_forward` is `pub` on a `pub mod`, so any external
    /// caller can pass a slice of the wrong length. It must return the normal
    /// `FRAME_SIZE / 2`-length spectrum (zero-padded internally) instead of
    /// panicking (the old behavior was a release-active `assert_eq!`).
    #[test]
    fn test_mdct_forward_short_input_does_not_panic() {
        let samples = vec![0.5f32; 3]; // far shorter than FRAME_SIZE
        let spec = mdct_forward(&samples);
        assert_eq!(
            spec.len(),
            FRAME_SIZE / 2,
            "short input must still produce a full-length spectrum"
        );
        assert!(
            spec.iter().all(|x| x.is_finite()),
            "short-input spectrum must be finite"
        );
    }

    /// Regression test: empty input (the extreme short case) must not panic.
    #[test]
    fn test_mdct_forward_empty_input_does_not_panic() {
        let spec = mdct_forward(&[]);
        assert_eq!(spec.len(), FRAME_SIZE / 2);
        assert!(spec.iter().all(|x| x.is_finite()));
    }

    /// Regression test: longer-than-`FRAME_SIZE` input must be truncated, not
    /// panic.
    #[test]
    fn test_mdct_forward_long_input_does_not_panic() {
        let samples = sine_frame(440.0)
            .into_iter()
            .chain(sine_frame(880.0))
            .collect::<Vec<f32>>(); // 2 * FRAME_SIZE samples
        let spec = mdct_forward(&samples);
        assert_eq!(
            spec.len(),
            FRAME_SIZE / 2,
            "over-long input must be truncated to FRAME_SIZE before transform"
        );
        assert!(spec.iter().all(|x| x.is_finite()));
    }
}
