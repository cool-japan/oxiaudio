//! FIR filter design: windowed-sinc lowpass/highpass/Hilbert, and the Bessel I0 helper
//! used by the Kaiser window.

use std::f64::consts::PI;

use oxiaudio_core::{AudioBuffer, AudioFilter, OxiAudioError};

/// A finite impulse response (FIR) filter with explicit coefficients.
///
/// Processing is direct convolution; state is the most recent `taps - 1` input
/// samples (per channel), reset to zero on each [`process`](FirFilter::process)
/// call (block mode).
#[derive(Debug, Clone)]
pub struct FirFilter {
    /// Filter taps (impulse response).
    pub coefficients: Vec<f32>,
}

impl FirFilter {
    /// Create a FIR filter from explicit taps.
    pub fn new(coefficients: Vec<f32>) -> Self {
        Self { coefficients }
    }

    /// Design a windowed-sinc **lowpass** FIR with `num_taps` taps.
    ///
    /// `cutoff` is the -6 dB cutoff in Hz. `window` shapes the sinc to trade
    /// transition width against side-lobe rejection.
    pub fn design_lowpass(
        num_taps: usize,
        cutoff: f32,
        sample_rate: u32,
        window: FirWindow,
    ) -> Self {
        let n = num_taps.max(1);
        let fc = f64::from(cutoff) / f64::from(sample_rate); // normalized (cycles/sample)
        let m = (n - 1) as f64;
        let mut taps = Vec::with_capacity(n);
        for i in 0..n {
            let x = i as f64 - m / 2.0;
            // Ideal sinc lowpass.
            let sinc = if x.abs() < 1e-9 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * x).sin() / (PI * x)
            };
            taps.push((sinc * window.value(i, n)) as f32);
        }
        // Normalize for unity DC gain.
        let sum: f32 = taps.iter().sum();
        if sum.abs() > 1e-12 {
            for t in &mut taps {
                *t /= sum;
            }
        }
        Self::new(taps)
    }

    /// Design a windowed-sinc **highpass** FIR via spectral inversion of a lowpass.
    pub fn design_highpass(
        num_taps: usize,
        cutoff: f32,
        sample_rate: u32,
        window: FirWindow,
    ) -> Self {
        let mut lp = Self::design_lowpass(num_taps, cutoff, sample_rate, window);
        let mid = lp.coefficients.len() / 2;
        for (i, t) in lp.coefficients.iter_mut().enumerate() {
            *t = -*t;
            if i == mid {
                *t += 1.0;
            }
        }
        lp
    }

    /// Design a **Hilbert transformer** FIR with `num_taps` taps.
    ///
    /// The Hilbert transformer shifts the phase of all positive-frequency components
    /// by −90° (and negative-frequency components by +90°), producing an analytic
    /// signal when paired with the original signal.
    ///
    /// The ideal impulse response is:
    /// ```text
    /// h[n] = 0              for (n − M) even  (including center)
    /// h[n] = 2 / (π(n − M)) for (n − M) odd
    /// ```
    /// where `M = (num_taps − 1) / 2` is the center index.  A Hamming window is
    /// applied to taper the truncated response and reduce Gibbs ringing.
    ///
    /// # Notes
    /// - For a causal, linear-phase Hilbert transformer `num_taps` should be **odd**
    ///   so that the center tap is zero and symmetry is exact.
    /// - The filter introduces a group-delay of `M` samples.
    pub fn design_hilbert(num_taps: usize) -> Self {
        // Hilbert FIR requires odd length for exact linear phase (Type III);
        // increment even inputs to the nearest odd value.
        let n = {
            let n0 = num_taps.max(1);
            if n0 % 2 == 0 {
                n0 + 1
            } else {
                n0
            }
        };
        let m = (n - 1) as f64 / 2.0;
        let mut taps = Vec::with_capacity(n);
        for i in 0..n {
            let x = i as f64 - m;
            // h[n]: zero for even (n-M), including center (x≈0); 2/(π(n-M)) for odd.
            let h = if x.abs() < 1e-9 {
                // Center tap: Hilbert transformer has h=0 at the center.
                0.0
            } else {
                let xi = x.round() as i64;
                if xi.abs() % 2 == 0 {
                    0.0
                } else {
                    2.0 / (PI * x)
                }
            };
            // Hamming window for smooth tapering.
            let w = if n > 1 {
                0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            } else {
                1.0
            };
            taps.push((h * w) as f32);
        }
        Self::new(taps)
    }

    /// Apply the FIR filter to each channel of an interleaved buffer (block mode).
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let ch = buf.channels.channel_count();
        let frames = buf.samples.len() / ch.max(1);
        let taps = &self.coefficients;
        let mut out = vec![0.0f32; buf.samples.len()];
        for c in 0..ch {
            for i in 0..frames {
                let mut acc = 0.0f32;
                for (j, &coeff) in taps.iter().enumerate() {
                    if i >= j {
                        acc += coeff * buf.samples[(i - j) * ch + c];
                    }
                }
                out[i * ch + c] = acc;
            }
        }
        AudioBuffer {
            samples: out,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }
}

impl AudioFilter for FirFilter {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

/// Window functions for FIR design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FirWindow {
    /// Rectangular (no taper).
    Rectangular,
    /// Hamming window.
    Hamming,
    /// Hann (raised cosine) window.
    Hann,
    /// Blackman window.
    Blackman,
    /// Kaiser window with adjustable shape parameter `beta`.
    Kaiser { beta: f32 },
}

impl FirWindow {
    pub(crate) fn value(self, i: usize, n: usize) -> f64 {
        if n <= 1 {
            return 1.0;
        }
        let m = (n - 1) as f64;
        let x = i as f64 / m;
        match self {
            FirWindow::Rectangular => 1.0,
            FirWindow::Hamming => 0.54 - 0.46 * (2.0 * PI * x).cos(),
            FirWindow::Hann => 0.5 - 0.5 * (2.0 * PI * x).cos(),
            FirWindow::Blackman => 0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos(),
            FirWindow::Kaiser { beta } => {
                let beta = f64::from(beta);
                let r = 2.0 * (i as f64) / m - 1.0;
                bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / bessel_i0(beta)
            }
        }
    }
}

/// Zeroth-order modified Bessel function of the first kind (series expansion).
/// Used internally for the Kaiser window.
pub(crate) fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half_x = x / 2.0;
    for k in 1..50 {
        term *= (half_x / k as f64).powi(2);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    use std::f32::consts::PI as PI_F32;

    fn sine(freq: f32, sr: u32, n: usize) -> AudioBuffer<f32> {
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / sr as f32).sin())
            .collect();
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn rms(s: &[f32], skip: usize) -> f32 {
        let sl = &s[skip..];
        (sl.iter().map(|x| x * x).sum::<f32>() / sl.len() as f32).sqrt()
    }

    #[test]
    fn fir_lowpass_attenuates_high_freq() {
        let sr = 48_000u32;
        let fir = FirFilter::design_lowpass(101, 1_000.0, sr, FirWindow::Hamming);
        let high = fir.process(&sine(10_000.0, sr, sr as usize));
        let high_db = 20.0
            * (rms(&high.samples, 200) / rms(&sine(10_000.0, sr, sr as usize).samples, 200))
                .log10();
        assert!(
            high_db < -30.0,
            "FIR LP should attenuate 10kHz, got {high_db:.1}dB"
        );
    }

    #[test]
    fn fir_dc_gain_unity() {
        let fir = FirFilter::design_lowpass(64, 2_000.0, 48_000, FirWindow::Blackman);
        let sum: f32 = fir.coefficients.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "FIR DC gain should be ~1, got {sum}"
        );
    }

    #[test]
    fn fir_kaiser_window_finite() {
        let fir = FirFilter::design_lowpass(64, 2_000.0, 48_000, FirWindow::Kaiser { beta: 6.0 });
        assert!(fir.coefficients.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn bessel_i0_known_values() {
        // I0(0) = 1, I0(1) ~= 1.2661.
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-9);
        assert!((bessel_i0(1.0) - 1.266_065_88).abs() < 1e-4);
    }

    #[test]
    fn fir_hilbert_center_tap_is_zero() {
        // The center tap of a Hilbert transformer is always 0 by definition.
        let fir = FirFilter::design_hilbert(63);
        let mid = 63 / 2;
        assert!(
            fir.coefficients[mid].abs() < 1e-6,
            "center tap must be zero, got {}",
            fir.coefficients[mid]
        );
    }

    #[test]
    fn fir_hilbert_antisymmetric() {
        // The Hilbert transformer has odd symmetry: h[M-k] = -h[M+k].
        let n = 63usize;
        let fir = FirFilter::design_hilbert(n);
        let mid = n / 2;
        for k in 1..=mid {
            let lo = fir.coefficients[mid - k];
            let hi = fir.coefficients[mid + k];
            assert!(
                (lo + hi).abs() < 1e-6,
                "antisymmetry violated at offset {k}: h[{0}]={lo}, h[{1}]={hi}",
                mid - k,
                mid + k
            );
        }
    }

    #[test]
    fn fir_hilbert_odd_taps_nonzero() {
        // All odd-offset taps (from center) should be non-zero for a well-formed Hilbert filter.
        let n = 63usize;
        let fir = FirFilter::design_hilbert(n);
        let mid = n / 2;
        for k in (1..=mid).step_by(2) {
            let tap = fir.coefficients[mid + k];
            assert!(
                tap.abs() > 1e-4,
                "odd-offset tap at +{k} (offset from center) should be nonzero, got {tap}"
            );
        }
    }

    #[test]
    fn test_fir_hilbert_odd_length() {
        // design_hilbert(32) must return length 33 (even → odd forced)
        let fir = FirFilter::design_hilbert(32);
        assert_eq!(
            fir.coefficients.len(),
            33,
            "design_hilbert(32) should return 33 taps (forced odd), got {}",
            fir.coefficients.len()
        );
    }

    #[test]
    fn test_fir_hilbert_center_zero() {
        let fir = FirFilter::design_hilbert(63);
        let mid = 63 / 2; // already odd, so center is at index 31
        assert!(
            fir.coefficients[mid].abs() < 1e-6,
            "center coefficient must be 0.0, got {}",
            fir.coefficients[mid]
        );
    }

    #[test]
    fn test_fir_hilbert_even_indices_zero() {
        let n = 63usize;
        let fir = FirFilter::design_hilbert(n);
        let mid = n / 2;
        // All even-offset-from-center coefficients must be zero
        for k in (2..=mid).step_by(2) {
            let lo = fir.coefficients[mid - k];
            let hi = fir.coefficients[mid + k];
            assert!(
                lo.abs() < 1e-6,
                "even-offset coeff at mid-{k} should be 0, got {lo}"
            );
            assert!(
                hi.abs() < 1e-6,
                "even-offset coeff at mid+{k} should be 0, got {hi}"
            );
        }
    }

    #[test]
    fn fir_hilbert_90_degree_phase_shift() {
        // Processing a cosine through a Hilbert transformer should approximate a sine
        // (delayed by the filter's group delay M = (N-1)/2 samples).
        let sr = 48_000u32;
        let n = 511usize;
        let fir = FirFilter::design_hilbert(n);
        let delay = n / 2; // group delay in samples
        let freq = 1_000.0f32;
        // Build cosine and sine buffers of length 2*sr samples.
        let len = sr as usize * 2;
        let cos_samples: Vec<f32> = (0..len)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / sr as f32).cos())
            .collect();
        let cos_buf = AudioBuffer {
            samples: cos_samples.clone(),
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let sin_ref: Vec<f32> = (0..len)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / sr as f32).sin())
            .collect();
        let hilbert_out = fir.process(&cos_buf);
        // Compare the Hilbert output (≈ sin) to the delayed reference sine.
        // Skip early samples (transient) and account for the group delay.
        let start = delay + sr as usize / 4; // skip first 0.25s + delay
        let end = len - delay;
        let mut max_err = 0.0f32;
        for i in start..end {
            // Hilbert output at sample i corresponds to input at sample i-delay (already baked in).
            let err = (hilbert_out.samples[i] - sin_ref[i - delay]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 0.05,
            "Hilbert 90° phase shift error too large: {max_err:.4} (expect < 0.05)"
        );
    }
}
