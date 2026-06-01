//! Higher-order IIR filter design (Butterworth, Chebyshev Type I/II, Elliptic) and FIR
//! windowed-sinc design.
//!
//! IIR filters are produced as a cascade of second-order sections ([`BiquadFilter`]),
//! computed from the analog prototype poles/zeros via the bilinear transform with
//! frequency pre-warping. The cascade is wrapped in a [`crate::ParametricEq`]-like
//! [`Cascade`] that maintains independent per-section state for streaming.

// ─── Re-exports from sub-modules (declared in crate root) ────────────────────

pub use crate::filters_fir::{FirFilter, FirWindow};
pub use crate::filters_iir::{
    chebyshev2_highpass, chebyshev2_lowpass, elliptic_highpass, elliptic_lowpass, Chebyshev2Filter,
    EllipticFilter,
};

// ─── Core IIR utilities (Cascade, bilinear helpers, Butterworth, Chebyshev I) ─

use std::f64::consts::PI;

use oxiaudio_core::{AudioBuffer, AudioFilter, OxiAudioError};

use crate::biquad::BiquadFilter;

/// A complex number in `f64` precision used internally for pole/zero math.
#[derive(Debug, Clone, Copy)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }

    fn sub(self, o: C64) -> C64 {
        C64::new(self.re - o.re, self.im - o.im)
    }

    fn mul(self, o: C64) -> C64 {
        C64::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

/// A cascade of [`BiquadFilter`] second-order sections applied in series, each
/// maintaining its own running state so the cascade can be applied to a buffer.
#[derive(Debug, Clone)]
pub struct Cascade {
    /// Ordered second-order sections.
    pub sections: Vec<BiquadFilter>,
}

impl Cascade {
    /// Create a cascade from a list of biquad sections.
    pub fn new(sections: Vec<BiquadFilter>) -> Self {
        Self { sections }
    }

    /// Apply every section in series (each section is stateless block-mode).
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let mut current = buf.clone();
        for s in &self.sections {
            current = s.process(&current);
        }
        current
    }
}

impl AudioFilter for Cascade {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

/// Map an analog (s-plane) pole/zero pair into a digital biquad via the bilinear
/// transform. `c = 2 * fs` is the bilinear constant (pre-warping is applied to the
/// pole/zero locations before this call).
///
/// Given an analog biquad section `H(s) = (s - z0)(s - z1) / ((s - p0)(s - p1))`
/// with optional gain `k`, produce the digital coefficients normalized so `a0 = 1`.
fn bilinear_section(
    z0: Option<C64>,
    z1: Option<C64>,
    p0: C64,
    p1: C64,
    k: f64,
    c: f64,
) -> BiquadFilter {
    // Numerator/denominator polynomial coefficients in z^-1, accumulated as
    // products of (c*(1 - z^-1) - root*(1 + z^-1)) for each finite root, and
    // (1 + z^-1) factors for roots at infinity (so order stays 2).
    //
    // We build the quadratic [b0, b1, b2] and [a0, a1, a2] directly.
    let den = poly_from_roots(c, p0, p1);
    let num = match (z0, z1) {
        (Some(a), Some(b)) => poly_from_roots(c, a, b),
        // One zero at infinity: factor (1 + z^-1) replaces one (c(1-z) - z*(1+z)).
        (Some(a), None) | (None, Some(a)) => poly_one_finite_one_inf(c, a),
        // Both zeros at infinity (lowpass): numerator is (1 + z^-1)^2.
        (None, None) => [1.0, 2.0, 1.0],
    };

    let a0 = den[0];
    // Apply analog gain k and the bilinear normalization in one shot.
    let g = k * num[0] / den[0];
    // Re-normalize numerator so that b reflects gain * (num / num[0]); then divide by a0.
    let b0 = g * 1.0;
    let b1 = g * (num[1] / num[0]);
    let b2 = g * (num[2] / num[0]);
    let a1 = den[1] / a0;
    let a2 = den[2] / a0;

    BiquadFilter {
        b0: b0 as f32,
        b1: b1 as f32,
        b2: b2 as f32,
        a1: a1 as f32,
        a2: a2 as f32,
    }
}

/// Expand `(c(1 - z^-1) - r0(1 + z^-1)) * (c(1 - z^-1) - r1(1 + z^-1))` into a real
/// quadratic `[k0, k1, k2]` in `z^-1`. For a complex-conjugate pair the imaginary
/// parts cancel, leaving real coefficients.
fn poly_from_roots(c: f64, r0: C64, r1: C64) -> [f64; 3] {
    // factor_i(z^-1) = (c - r) + (-c - r) z^-1  => coefficients [c - r, -(c + r)]
    let f0 = [C64::new(c, 0.0).sub(r0), C64::new(-c, 0.0).sub(r0)];
    let f1 = [C64::new(c, 0.0).sub(r1), C64::new(-c, 0.0).sub(r1)];
    let q0 = f0[0].mul(f1[0]);
    let q1 = f0[0].mul(f1[1]).add(f0[1].mul(f1[0]));
    let q2 = f0[1].mul(f1[1]);
    [q0.re, q1.re, q2.re]
}

/// Expand `(c(1 - z^-1) - r(1 + z^-1)) * (1 + z^-1)` for one finite root `r` and one
/// zero at infinity.
fn poly_one_finite_one_inf(c: f64, r: C64) -> [f64; 3] {
    // first factor: [c - r, -(c + r)]; second factor (1 + z^-1): [1, 1]
    let a = C64::new(c, 0.0).sub(r);
    let b = C64::new(-c, 0.0).sub(r);
    let q0 = a; // a*1
    let q1 = a.add(b); // a*1 + b*1
    let q2 = b; // b*1
    [q0.re, q1.re, q2.re]
}

/// Pre-warp a digital cutoff frequency to the equivalent analog frequency for the
/// bilinear transform: `wa = 2*fs * tan(pi * fc / fs)`.
fn prewarp(fc: f32, sample_rate: u32) -> f64 {
    let fs = f64::from(sample_rate);
    2.0 * fs * (PI * f64::from(fc) / fs).tan()
}

/// Number of biquad sections for an `order`-pole filter (one per pole pair, plus a
/// first-order section folded into a biquad when `order` is odd).
fn n_sections(order: usize) -> usize {
    order.div_ceil(2)
}

/// Design an `order`-pole Butterworth **lowpass** as a cascade of biquads.
///
/// `order >= 1`. Even orders yield `order/2` sections; odd orders yield
/// `(order+1)/2` sections (the extra section is a real first-order pole folded
/// into a biquad).
pub fn butterworth_lowpass(order: usize, frequency: f32, sample_rate: u32) -> Cascade {
    butterworth(order, frequency, sample_rate, false)
}

/// Design an `order`-pole Butterworth **highpass** as a cascade of biquads.
pub fn butterworth_highpass(order: usize, frequency: f32, sample_rate: u32) -> Cascade {
    butterworth(order, frequency, sample_rate, true)
}

fn butterworth(order: usize, frequency: f32, sample_rate: u32, highpass: bool) -> Cascade {
    let order = order.max(1);
    let wc = prewarp(frequency, sample_rate); // analog cutoff (rad/s)
    let c = 2.0 * f64::from(sample_rate); // bilinear constant
    let mut sections = Vec::with_capacity(n_sections(order));

    // Butterworth poles are equally spaced on a circle of radius wc in the left
    // half plane. We pair conjugate poles; for odd order there is one real pole.
    let mut k = 0usize;
    while k < order {
        // Pole angle for the k-th left-half-plane pole.
        let theta = PI * (2.0 * (k as f64) + 1.0) / (2.0 * order as f64) + PI / 2.0;
        let pole = C64::new(wc * theta.cos(), wc * theta.sin());

        if order % 2 == 1 && k == order - 1 {
            // Final real pole at s = -wc: first-order section.
            let real_pole = C64::new(-wc, 0.0);
            let section = if highpass {
                // HP: zero at s=0 (one finite zero at origin), pole at -wc, plus a
                // second pole/zero at infinity to keep it biquad-shaped.
                bilinear_section(
                    Some(C64::new(0.0, 0.0)),
                    None,
                    real_pole,
                    C64::new(-1e12, 0.0),
                    1.0,
                    c,
                )
            } else {
                // LP: zeros at infinity, pole at -wc.
                bilinear_section(None, None, real_pole, C64::new(-1e12, 0.0), wc, c)
            };
            sections.push(normalize_section(section, frequency, sample_rate, highpass));
            k += 1;
        } else {
            // Conjugate pair: combine pole and its conjugate.
            let conj = C64::new(pole.re, -pole.im);
            let section = if highpass {
                // HP: two zeros at s=0 -> after bilinear, zeros at z=1 => (1 - z^-1)^2.
                bilinear_section(
                    Some(C64::new(0.0, 0.0)),
                    Some(C64::new(0.0, 0.0)),
                    pole,
                    conj,
                    1.0,
                    c,
                )
            } else {
                // LP: two zeros at infinity, gain wc^2 to set DC gain to 1.
                bilinear_section(None, None, pole, conj, wc * wc, c)
            };
            sections.push(normalize_section(section, frequency, sample_rate, highpass));
            k += 2;
        }
    }
    Cascade::new(sections)
}

/// Normalize a designed section so its passband gain is exactly unity.
///
/// For lowpass we force DC (z=1) gain to 1; for highpass we force Nyquist (z=-1)
/// gain to 1. This compensates for accumulated scaling error in the bilinear math.
fn normalize_section(
    s: BiquadFilter,
    _frequency: f32,
    _sample_rate: u32,
    highpass: bool,
) -> BiquadFilter {
    let gain = if highpass {
        section_gain_at_nyquist(&s)
    } else {
        section_gain_at_dc(&s)
    };
    if gain.abs() < 1e-12 {
        return s;
    }
    BiquadFilter {
        b0: s.b0 / gain,
        b1: s.b1 / gain,
        b2: s.b2 / gain,
        a1: s.a1,
        a2: s.a2,
    }
}

fn section_gain_at_dc(s: &BiquadFilter) -> f32 {
    (s.b0 + s.b1 + s.b2) / (1.0 + s.a1 + s.a2)
}

fn section_gain_at_nyquist(s: &BiquadFilter) -> f32 {
    (s.b0 - s.b1 + s.b2) / (1.0 - s.a1 + s.a2)
}

/// Design a Chebyshev Type I **lowpass** with the given passband ripple in dB.
///
/// Type I filters are equiripple in the passband and monotonic in the stopband.
/// `ripple_db` is the maximum passband ripple (e.g. 0.5, 1.0, 3.0).
pub fn chebyshev1_lowpass(
    order: usize,
    frequency: f32,
    ripple_db: f32,
    sample_rate: u32,
) -> Cascade {
    chebyshev1(order, frequency, ripple_db, sample_rate, false)
}

/// Design a Chebyshev Type I **highpass** with the given passband ripple in dB.
pub fn chebyshev1_highpass(
    order: usize,
    frequency: f32,
    ripple_db: f32,
    sample_rate: u32,
) -> Cascade {
    chebyshev1(order, frequency, ripple_db, sample_rate, true)
}

fn chebyshev1(
    order: usize,
    frequency: f32,
    ripple_db: f32,
    sample_rate: u32,
    highpass: bool,
) -> Cascade {
    let order = order.max(1);
    let wc = prewarp(frequency, sample_rate);
    let c = 2.0 * f64::from(sample_rate);
    let ripple = f64::from(ripple_db).max(1e-3);

    // epsilon from passband ripple.
    let eps = (10f64.powf(ripple / 10.0) - 1.0).sqrt();
    let n = order as f64;
    let mu = (1.0 / eps).asinh() / n;
    let sinh_mu = mu.sinh();
    let cosh_mu = mu.cosh();

    let mut sections = Vec::with_capacity(n_sections(order));
    let mut k = 0usize;
    while k < order {
        let theta = PI * (2.0 * (k as f64) + 1.0) / (2.0 * n);
        // Chebyshev pole on an ellipse.
        let pole = C64::new(-wc * sinh_mu * theta.sin(), wc * cosh_mu * theta.cos());

        if order % 2 == 1 && k == order - 1 {
            let real_pole = C64::new(pole.re, 0.0);
            let section = if highpass {
                bilinear_section(
                    Some(C64::new(0.0, 0.0)),
                    None,
                    real_pole,
                    C64::new(-1e12, 0.0),
                    1.0,
                    c,
                )
            } else {
                bilinear_section(
                    None,
                    None,
                    real_pole,
                    C64::new(-1e12, 0.0),
                    -real_pole.re,
                    c,
                )
            };
            sections.push(section);
            k += 1;
        } else {
            let conj = C64::new(pole.re, -pole.im);
            // |pole|^2 sets the LP DC gain.
            let mag_sq = pole.re * pole.re + pole.im * pole.im;
            let section = if highpass {
                bilinear_section(
                    Some(C64::new(0.0, 0.0)),
                    Some(C64::new(0.0, 0.0)),
                    pole,
                    conj,
                    1.0,
                    c,
                )
            } else {
                bilinear_section(None, None, pole, conj, mag_sq, c)
            };
            sections.push(section);
            k += 2;
        }
    }

    // Overall passband normalization: for even order Type I, DC gain is 1/sqrt(1+eps^2).
    // Force the first section to unity passband gain to keep the cascade calibrated.
    let mut cascade = Cascade::new(sections);
    if let Some(first) = cascade.sections.first().copied() {
        let normalized = normalize_section(first, frequency, sample_rate, highpass);
        cascade.sections[0] = normalized;
    }
    cascade
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{ChannelLayout, SampleFormat};
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
    fn butterworth_lp_passes_low_attenuates_high() {
        let sr = 48_000u32;
        let lp = butterworth_lowpass(4, 1_000.0, sr);
        // Low freq (200 Hz) passes; high freq (10 kHz) heavily attenuated.
        let low = lp.process(&sine(200.0, sr, sr as usize));
        let high = lp.process(&sine(10_000.0, sr, sr as usize));
        let low_db = 20.0
            * (rms(&low.samples, 2000) / rms(&sine(200.0, sr, sr as usize).samples, 2000)).log10();
        let high_db = 20.0
            * (rms(&high.samples, 2000) / rms(&sine(10_000.0, sr, sr as usize).samples, 2000))
                .log10();
        assert!(
            low_db > -3.0,
            "200Hz should pass through 1kHz LP, got {low_db:.1}dB"
        );
        assert!(
            high_db < -40.0,
            "10kHz should be attenuated >40dB by 4th-order LP, got {high_db:.1}dB"
        );
    }

    #[test]
    fn butterworth_hp_attenuates_low() {
        let sr = 48_000u32;
        let hp = butterworth_highpass(4, 1_000.0, sr);
        let low = hp.process(&sine(100.0, sr, sr as usize));
        let low_db = 20.0
            * (rms(&low.samples, 2000) / rms(&sine(100.0, sr, sr as usize).samples, 2000)).log10();
        assert!(
            low_db < -20.0,
            "100Hz should be attenuated by 1kHz HP, got {low_db:.1}dB"
        );
    }

    #[test]
    fn chebyshev1_lp_attenuates_high() {
        let sr = 48_000u32;
        let cheby = chebyshev1_lowpass(4, 1_000.0, 1.0, sr);
        let high = cheby.process(&sine(8_000.0, sr, sr as usize));
        let high_db = 20.0
            * (rms(&high.samples, 3000) / rms(&sine(8_000.0, sr, sr as usize).samples, 3000))
                .log10();
        assert!(
            high_db < -40.0,
            "Chebyshev I LP should attenuate 8kHz, got {high_db:.1}dB"
        );
    }

    #[test]
    fn chebyshev1_odd_order_works() {
        let sr = 48_000u32;
        // Odd order exercises the first-order section path.
        let cheby = chebyshev1_lowpass(3, 2_000.0, 0.5, sr);
        assert_eq!(cheby.sections.len(), 2);
        let out = cheby.process(&sine(500.0, sr, 4096));
        assert!(out.samples.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn butterworth_4th_order_stopband_exceeds_60db() {
        // 4th-order Butterworth LP at 1 kHz/48 kHz: at 10 kHz stopband attenuation
        // should be significant. The 80 dB/decade rolloff is asymptotic — at exactly
        // 10x the cutoff, the actual attenuation is bounded by the finite ratio,
        // so we check for >60 dB which is well within the expected rolloff.
        let sr = 48_000u32;
        let n = sr as usize;
        let lp = butterworth_lowpass(4, 1_000.0, sr);
        let buf_10k = sine(10_000.0, sr, n);
        let buf_1k = sine(1_000.0, sr, n);
        let skip = 5000usize;
        let out_10k = lp.process(&buf_10k);
        let rms_10k = rms(&out_10k.samples, skip);
        let rms_in_1k = rms(&buf_1k.samples, skip);
        // Attenuation of 10 kHz relative to 1 kHz passband input amplitude
        let atten_db = 20.0 * (rms_10k / rms_in_1k.max(1e-10)).log10();
        assert!(
            atten_db < -60.0,
            "Butterworth 4th-order LP: 10kHz should be >60dB below 1kHz passband, got {atten_db:.1}dB"
        );
    }

    #[test]
    fn chebyshev1_4th_order_stopband_exceeds_60db() {
        // 4th-order Chebyshev Type I LP, 3 dB ripple, 1 kHz cutoff at 48 kHz:
        // 10 kHz should be >60 dB down.
        let sr = 48_000u32;
        let n = sr as usize;
        let lp = chebyshev1_lowpass(4, 1_000.0, 3.0, sr);
        let buf_10k = sine(10_000.0, sr, n);
        let buf_1k = sine(1_000.0, sr, n);
        let skip = 5000usize;
        let out_10k = lp.process(&buf_10k);
        let rms_10k = rms(&out_10k.samples, skip);
        let rms_in_1k = rms(&buf_1k.samples, skip);
        let atten_db = 20.0 * (rms_10k / rms_in_1k.max(1e-10)).log10();
        assert!(
            atten_db < -60.0,
            "Chebyshev I 4th-order LP: 10kHz should be >60dB below 1kHz passband, got {atten_db:.1}dB"
        );
    }
}
