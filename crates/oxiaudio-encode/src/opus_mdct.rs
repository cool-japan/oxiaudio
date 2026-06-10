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
/// # Panics
///
/// Panics in debug mode if `samples.len() != FRAME_SIZE`.
pub fn mdct_forward(samples: &[f32]) -> Vec<f32> {
    assert_eq!(
        samples.len(),
        FRAME_SIZE,
        "mdct_forward: expected {FRAME_SIZE} samples, got {}",
        samples.len()
    );

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
    use super::{mdct_forward, FRAME_SIZE};

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
}
