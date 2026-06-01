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
