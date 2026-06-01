//! Higher-order IIR filter design: Chebyshev Type II (inverse Chebyshev) and
//! Elliptic (Cauer) filters as cascades of biquad second-order sections.

use std::f64::consts::PI;

use oxiaudio_core::{AudioBuffer, AudioFilter, OxiAudioError};

use crate::biquad::BiquadFilter;

// ─── Chebyshev Type II (Inverse Chebyshev) ───────────────────────────────────

/// Chebyshev Type II (inverse Chebyshev) filter as a cascade of biquad SOS stages.
///
/// Type II filters have maximally flat passband and equiripple stopband.
#[derive(Debug, Clone)]
pub struct Chebyshev2Filter {
    /// Second-order sections forming the cascade.
    pub stages: Vec<BiquadFilter>,
    /// Filter order.
    pub order: usize,
}

impl Chebyshev2Filter {
    /// Apply each biquad stage in series to the buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let mut result = buf.clone();
        for stage in &self.stages {
            result = stage.process(&result);
        }
        result
    }
}

impl AudioFilter for Chebyshev2Filter {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

/// Design a Chebyshev Type II (inverse Chebyshev) **lowpass** filter.
///
/// Chebyshev Type II normalizes the prototype stopband to Ω = 1, so
/// `cutoff_hz` is the **stopband edge** — the frequency at which the gain
/// first drops to −`stopband_db` dB.  The passband is maximally flat below
/// this frequency.
///
/// - `order`: filter order (1–8).
/// - `cutoff_hz`: **stopband** edge frequency in Hz.
/// - `stopband_db`: stopband attenuation in dB (e.g. 40.0, 60.0, 80.0).
/// - `sample_rate`: sample rate in Hz.
///
/// Returns the filter as a cascade of biquad SOS stages.
#[must_use = "returns the designed Chebyshev II lowpass filter"]
pub fn chebyshev2_lowpass(
    order: usize,
    cutoff_hz: f32,
    stopband_db: f32,
    sample_rate: u32,
) -> Result<Chebyshev2Filter, OxiAudioError> {
    if order == 0 || order > 8 {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "Chebyshev II order must be 1–8, got {order}"
        )));
    }
    let fs = sample_rate as f64;
    // Pre-warped stopband edge (rad/s).  For Chebyshev II the "cutoff" parameter
    // specifies the stopband edge — the frequency at which attenuation first reaches
    // stopband_db.  The prototype normalises the stopband to Ω = 1.
    let omega_c = 2.0 * fs * (PI * cutoff_hz as f64 / fs).tan();
    // Epsilon from stopband attenuation: ε_s = sqrt(10^(A/10) − 1)
    let eps_s = (10.0_f64.powf(stopband_db as f64 / 10.0) - 1.0).sqrt();
    // α = asinh(ε_s) / N  — determines how far off-axis the Chebyshev I poles sit;
    // using eps_s (not 1/eps_s) so that the poles are well off the imaginary axis for
    // high-attenuation designs.
    let alpha = eps_s.asinh() / order as f64;

    let mut stages: Vec<BiquadFilter> = Vec::new();
    let n_pairs = order / 2;
    let c = 2.0 * fs; // bilinear constant

    for k in 1..=n_pairs {
        let theta_k = PI * (2 * k - 1) as f64 / (2 * order) as f64;
        // Chebyshev I prototype poles on the ellipse
        let sigma_k = -alpha.sinh() * theta_k.sin();
        let omega_k = alpha.cosh() * theta_k.cos();
        let denom = sigma_k * sigma_k + omega_k * omega_k;
        // Chebyshev II poles: invert the prototype (s → 1/s, then conjugate back)
        let p_re = sigma_k / denom;
        let p_im = -omega_k / denom;
        // Chebyshev II zeros at ±j/cos(θ_k) on the prototype, scaled by ω_c
        let z_im = 1.0 / theta_k.cos();

        // Scale all frequencies by pre-warped cutoff
        let sp_re = p_re * omega_c;
        let sp_im = p_im * omega_c;
        let sz_im = z_im * omega_c;

        // Bilinear transform for conjugate pole pair + conjugate zero pair.
        // Normalized analog: H(s) = k_dc·(s²+sz_im²) / (s²−2·sp_re·s+pole_mag_sq)
        // where k_dc = pole_mag_sq/zero_mag_sq ensures unity DC gain (H(0)=1).
        // Substitute s = c(z−1)/(z+1):
        let pole_mag_sq = sp_re * sp_re + sp_im * sp_im;
        let zero_mag_sq = sz_im * sz_im;
        // DC normalization gain per SOS section
        let k_dc = pole_mag_sq / zero_mag_sq.max(1e-30);

        // Numerator coefficients (with k_dc applied): k_dc*(c²+z²), k_dc*(2z²−2c²), k_dc*(c²+z²)
        let nb0 = k_dc * (c * c + zero_mag_sq);
        let nb1 = k_dc * (2.0 * zero_mag_sq - 2.0 * c * c);
        let nb2 = k_dc * (c * c + zero_mag_sq);
        // Denominator coefficients
        let da0 = c * c - 2.0 * sp_re * c + pole_mag_sq;
        let da1 = 2.0 * pole_mag_sq - 2.0 * c * c;
        let da2 = c * c + 2.0 * sp_re * c + pole_mag_sq;

        let b0 = (nb0 / da0) as f32;
        let b1 = (nb1 / da0) as f32;
        let b2 = (nb2 / da0) as f32;
        let a1 = (da1 / da0) as f32;
        let a2 = (da2 / da0) as f32;

        stages.push(BiquadFilter { b0, b1, b2, a1, a2 });
    }

    // Odd order: one real pole from the Chebyshev I prototype at s = −sinh(α).
    // After inversion (s → 1/s) the Chebyshev II real pole is at s = −1/sinh(α).
    // Scale by ω_c: sp_abs = ω_c / sinh(α).
    if order % 2 == 1 {
        let sp_abs = omega_c / alpha.sinh().max(1e-30);
        // Bilinear: H(s) = sp_abs/(s+sp_abs) → H(z) = sp_abs·(1+z⁻¹)/((c+sp_abs)+(sp_abs−c)z⁻¹)
        let da0_1 = c + sp_abs;
        let b0_1 = (sp_abs / da0_1) as f32;
        let b1_1 = (sp_abs / da0_1) as f32;
        let a1_1 = ((sp_abs - c) / da0_1) as f32;
        stages.push(BiquadFilter {
            b0: b0_1,
            b1: b1_1,
            b2: 0.0,
            a1: a1_1,
            a2: 0.0,
        });
    }

    Ok(Chebyshev2Filter { stages, order })
}

/// Design a Chebyshev Type II (inverse Chebyshev) **highpass** filter.
///
/// Applies the LP→HP frequency transformation on the analog prototype before
/// the bilinear transform, so that:
/// - LP poles at p_LP → HP poles at ω_c² / p_LP (complex division)
/// - LP zeros at ±j·z_im → HP zeros at ±j·ω_c / z_im
///
/// Chebyshev Type II normalizes the prototype stopband edge to Ω=1, so
/// `cutoff_hz` is the **stopband edge** — the frequency below which the
/// attenuation reaches `stopband_db` dB.  Passband frequencies (above the
/// stopband edge) exhibit maximally-flat response.
///
/// - `order`: filter order (1–8).
/// - `cutoff_hz`: stopband edge frequency in Hz.
/// - `stopband_db`: stopband attenuation in dB.
/// - `sample_rate`: sample rate in Hz.
#[must_use = "returns the designed Chebyshev II highpass filter"]
pub fn chebyshev2_highpass(
    order: usize,
    cutoff_hz: f32,
    stopband_db: f32,
    sample_rate: u32,
) -> Result<Chebyshev2Filter, OxiAudioError> {
    if order == 0 || order > 8 {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "Chebyshev II order must be 1–8, got {order}"
        )));
    }
    let fs = sample_rate as f64;
    // For Chebyshev II HP, cutoff_hz is the stopband edge (below which attenuation
    // reaches stopband_db).  Pre-warp it to the analog domain.
    let omega_c = 2.0 * fs * (PI * cutoff_hz as f64 / fs).tan();
    let eps_s = (10.0_f64.powf(stopband_db as f64 / 10.0) - 1.0).sqrt();
    let alpha = eps_s.asinh() / order as f64;

    let mut stages: Vec<BiquadFilter> = Vec::new();
    let n_pairs = order / 2;
    let c = 2.0 * fs;

    for k in 1..=n_pairs {
        let theta_k = PI * (2 * k - 1) as f64 / (2 * order) as f64;
        // Chebyshev I prototype poles on the ellipse
        let sigma_k = -alpha.sinh() * theta_k.sin();
        let omega_k = alpha.cosh() * theta_k.cos();

        // LP→HP pole transform applied to the Chebyshev II LP prototype:
        // p_lp_proto = (σ_k − j·ω_k) / (σ_k²+ω_k²); p_hp = ω_c / p_lp_proto.
        // After simplification: sp_re = ω_c·σ_k, sp_im = ω_c·ω_k.
        let sp_re = omega_c * sigma_k;
        let sp_im = omega_c * omega_k;

        // LP zeros at ±j/cos(θ_k); LP→HP: z_hp_im = ω_c / z_lp_im = ω_c·cos(θ_k)
        let sz_im = omega_c * theta_k.cos();

        // Bilinear transform for conjugate HP pole pair + conjugate zero pair
        let pole_mag_sq = sp_re * sp_re + sp_im * sp_im;
        let zero_mag_sq = sz_im * sz_im;

        let nb0 = c * c + zero_mag_sq;
        let nb1 = 2.0 * zero_mag_sq - 2.0 * c * c;
        let nb2 = c * c + zero_mag_sq;
        let da0 = c * c - 2.0 * sp_re * c + pole_mag_sq;
        let da1 = 2.0 * pole_mag_sq - 2.0 * c * c;
        let da2 = c * c + 2.0 * sp_re * c + pole_mag_sq;

        let b0 = (nb0 / da0) as f32;
        let b1 = (nb1 / da0) as f32;
        let b2 = (nb2 / da0) as f32;
        let a1 = (da1 / da0) as f32;
        let a2 = (da2 / da0) as f32;

        stages.push(BiquadFilter { b0, b1, b2, a1, a2 });
    }

    // Odd order: Cheby II LP prototype real pole at s = −1/sinh(α) (normalized).
    // LP→HP transform: p_hp = ω_c / |p_lp| = ω_c * sinh(α).
    if order % 2 == 1 {
        let sp_hp = omega_c * alpha.sinh().max(1e-30);
        // Bilinear of a real HP first-order section: zeros at DC (z=1), pole at sp_hp
        // H(s) = s / (s + sp_hp); bilinear: H(z) = c(1−z⁻¹)/((c+sp_hp)+(sp_hp−c)z⁻¹)
        let da0_1 = c + sp_hp;
        let b0_1 = (c / da0_1) as f32;
        let b1_1 = -(c / da0_1) as f32;
        let a1_1 = ((sp_hp - c) / da0_1) as f32;
        stages.push(BiquadFilter {
            b0: b0_1,
            b1: b1_1,
            b2: 0.0,
            a1: a1_1,
            a2: 0.0,
        });
    }

    Ok(Chebyshev2Filter { stages, order })
}

// ─── Elliptic (Cauer) Filter ──────────────────────────────────────────────────

/// Elliptic (Cauer) filter as a cascade of biquad SOS stages.
///
/// Elliptic filters are equiripple in both the passband and stopband, giving
/// the sharpest rolloff for a given order.
#[derive(Debug, Clone)]
pub struct EllipticFilter {
    /// Second-order sections forming the cascade.
    pub stages: Vec<BiquadFilter>,
    /// Filter order.
    pub order: usize,
}

impl EllipticFilter {
    /// Apply each biquad stage in series to the buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let mut result = buf.clone();
        for stage in &self.stages {
            result = stage.process(&result);
        }
        result
    }
}

impl AudioFilter for EllipticFilter {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

/// Arithmetic-geometric mean (used for complete elliptic integral K).
fn agm_f64(mut a: f64, mut b: f64) -> f64 {
    for _ in 0..60 {
        let a_new = (a + b) * 0.5;
        let b_new = (a * b).sqrt();
        if (a_new - b_new).abs() < 1e-14 * a_new.abs().max(1e-300) {
            return a_new;
        }
        a = a_new;
        b = b_new;
    }
    a
}

/// Complete elliptic integral of the first kind K(k).
fn complete_k(k: f64) -> f64 {
    let kp = (1.0 - k * k).sqrt().max(0.0);
    std::f64::consts::FRAC_PI_2 / agm_f64(1.0, kp)
}

/// Complete elliptic integral K'(k) = K(sqrt(1−k²)).
fn complete_kp(k: f64) -> f64 {
    complete_k((1.0 - k * k).sqrt().max(0.0))
}

/// Jacobi elliptic sn(u, k) using the **descending** Landen transform.
///
/// Iteratively reduces k toward 0, where sn(u, 0) = sin(u), then
/// unwinds the substitution via the ascending recurrence.
fn jacobi_sn(u: f64, k: f64) -> f64 {
    if k < 1e-12 {
        return u.sin();
    }
    if 1.0 - k < 1e-12 {
        return u.tanh();
    }
    // Descending Landen: k_new = (1 − k')/(1 + k') < k, scaling u by 1/(1+k_new)
    let mut ks: Vec<f64> = Vec::with_capacity(25);
    let mut us: Vec<f64> = Vec::with_capacity(26);
    us.push(u);
    let mut kn = k;
    loop {
        let kp = (1.0 - kn * kn).sqrt().max(0.0);
        let k_new = (1.0 - kp) / (1.0 + kp);
        ks.push(kn);
        us.push(*us.last().unwrap_or(&u) / (1.0 + k_new));
        kn = k_new;
        if kn < 1e-12 || ks.len() >= 50 {
            break;
        }
    }
    // At the end kn ≈ 0: sn(u_N, 0) = sin(u_N)
    let mut result = us.last().copied().unwrap_or(0.0).sin();
    // Ascend back using: sn(u_{n-1}, k_{n-1}) = (1+k_n)*sn(u_n,k_n)/(1+k_n*sn²)
    for i in (0..ks.len()).rev() {
        let k_next = if i + 1 < ks.len() { ks[i + 1] } else { kn };
        result = (1.0 + k_next) * result / (1.0 + k_next * result * result);
    }
    result
}

/// Jacobi cn(u, k) = sqrt(1 − sn²(u, k)).
#[inline]
fn jacobi_cn(u: f64, k: f64) -> f64 {
    (1.0 - jacobi_sn(u, k).powi(2)).max(0.0).sqrt()
}

/// Jacobi dn(u, k) = sqrt(1 − k²·sn²(u, k)).
#[inline]
fn jacobi_dn(u: f64, k: f64) -> f64 {
    (1.0 - k * k * jacobi_sn(u, k).powi(2)).max(0.0).sqrt()
}

/// Find the selectivity modulus k for an elliptic filter such that
/// K(k)/K'(k) = n · K(k1)/K'(k1).
///
/// The condition comes from the degree equation that ensures the elliptic
/// filter meets both passband ripple and stopband attenuation simultaneously.
fn find_elliptic_k(n: usize, k1: f64) -> f64 {
    let target = n as f64 * complete_k(k1) / complete_kp(k1);
    let mut lo = 1e-12_f64;
    let mut hi = 1.0 - 1e-12;
    for _ in 0..100 {
        let mid = (lo + hi) * 0.5;
        if complete_k(mid) / complete_kp(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}

/// Compute the v0 parameter for the elliptic prototype:
/// solve `sc(v0_raw, kp1) = 1/eps_p` where `kp1 = sqrt(1 − k1²)`,
/// then `v0 = K(k) · v0_raw / (n · K(k1))`.
///
/// Since k1 is very small in practice, `sc(u, kp1≈1) ≈ sinh(u)`,
/// so `v0_raw ≈ asinh(1/eps_p)`.  For generality we use bisection on sc.
fn elliptic_v0(eps_p: f64, k1: f64, big_k: f64, n: usize) -> f64 {
    let kp1 = (1.0 - k1 * k1).sqrt().max(0.0);
    let target_sc = 1.0 / eps_p; // sc(v0_raw, kp1) = target_sc
    let k1_big = complete_k(k1);
    let kp1_big = complete_k(kp1); // = K'(k1)
                                   // sc = sn/cn; as u → K(kp1), sc → ∞, so solution exists in (0, K(kp1))
    let mut lo = 0.0_f64;
    let mut hi = kp1_big * (1.0 - 1e-10);
    for _ in 0..100 {
        let mid = (lo + hi) * 0.5;
        let sn = jacobi_sn(mid, kp1);
        let cn = jacobi_cn(mid, kp1);
        let sc = sn / cn.max(1e-300);
        if sc < target_sc {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let v0_raw = (lo + hi) * 0.5;
    big_k * v0_raw / (n as f64 * k1_big)
}

/// Compute one biquad SOS from an elliptic LP prototype pole pair + zero pair,
/// with DC gain normalization, then apply bilinear transform.
fn elliptic_lp_sos(sp_re: f64, sp_im: f64, sz_im: f64, c: f64) -> BiquadFilter {
    let pole_mag_sq = sp_re * sp_re + sp_im * sp_im;
    let zero_mag_sq = sz_im * sz_im;
    // DC gain normalization factor: k_dc = pole_mag_sq / zero_mag_sq
    let k_dc = pole_mag_sq / zero_mag_sq.max(1e-30);
    let nb0 = k_dc * (c * c + zero_mag_sq);
    let nb1 = k_dc * (2.0 * zero_mag_sq - 2.0 * c * c);
    let nb2 = nb0;
    let da0 = c * c - 2.0 * sp_re * c + pole_mag_sq;
    let da1 = 2.0 * pole_mag_sq - 2.0 * c * c;
    let da2 = c * c + 2.0 * sp_re * c + pole_mag_sq;
    BiquadFilter {
        b0: (nb0 / da0) as f32,
        b1: (nb1 / da0) as f32,
        b2: (nb2 / da0) as f32,
        a1: (da1 / da0) as f32,
        a2: (da2 / da0) as f32,
    }
}

/// Design an elliptic (Cauer) **lowpass** filter.
///
/// Uses AGM-based complete elliptic integrals, descending Landen Jacobi sn,
/// and the Darlington pole formula to construct the normalized LP prototype,
/// then maps it to the digital domain via the bilinear transform.
///
/// - `order`: filter order (1–8).
/// - `cutoff_hz`: passband edge frequency in Hz (where gain first drops to −Rp dB).
/// - `passband_ripple_db`: maximum passband ripple in dB (e.g. 1.0).
/// - `stopband_attenuation_db`: minimum stopband attenuation in dB (e.g. 60.0).
/// - `sample_rate`: sample rate in Hz.
#[must_use = "returns the designed elliptic lowpass filter"]
pub fn elliptic_lowpass(
    order: usize,
    cutoff_hz: f32,
    passband_ripple_db: f32,
    stopband_attenuation_db: f32,
    sample_rate: u32,
) -> Result<EllipticFilter, OxiAudioError> {
    if order == 0 || order > 8 {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "Elliptic filter order must be 1–8, got {order}"
        )));
    }
    let fs = sample_rate as f64;
    let omega_c = 2.0 * fs * (PI * cutoff_hz as f64 / fs).tan();
    let c = 2.0 * fs;

    let eps_p = (10.0_f64.powf(passband_ripple_db as f64 / 10.0) - 1.0).sqrt();
    let eps_s = (10.0_f64.powf(stopband_attenuation_db as f64 / 10.0) - 1.0).sqrt();
    let k1 = (eps_p / eps_s).clamp(1e-15, 1.0 - 1e-12);

    // Selectivity modulus k: K(k)/K'(k) = n·K(k1)/K'(k1)
    let k = find_elliptic_k(order, k1);
    let big_k = complete_k(k);
    let kp = (1.0 - k * k).sqrt().max(0.0); // complementary modulus

    // v0 parameter for pole imaginary shift
    let v0 = elliptic_v0(eps_p, k1, big_k, order);

    // Jacobi functions at v0 (with complementary modulus kp)
    let sv = jacobi_sn(v0, kp);
    let cv = jacobi_cn(v0, kp);
    let dv = jacobi_dn(v0, kp);

    let mut stages: Vec<BiquadFilter> = Vec::new();
    let n_pairs = order / 2;

    for i in 1..=n_pairs {
        let u_i = (2 * i - 1) as f64 * big_k / order as f64;
        let su = jacobi_sn(u_i, k);
        let cu = jacobi_cn(u_i, k);
        let du = jacobi_dn(u_i, k);

        // Prototype zero imaginary part: z_im = 1/(k·sn(u_i, k))
        let z_im_proto = 1.0 / (k * su.abs().max(1e-30));

        // Darlington pole formula (from scipy/Lutovac):
        // p = -(cu·du·sv·cv + j·su·dv) / (1 − (du·sv)²)
        let d_sq = (du * sv).powi(2);
        let denom_p = (1.0 - d_sq).max(1e-30);
        let p_re_proto = -(cu * du * sv * cv) / denom_p;
        let p_im_proto = -(su * dv) / denom_p; // negative imaginary → stable

        // Scale to actual frequency and apply bilinear
        let sp_re = p_re_proto * omega_c;
        let sp_im = p_im_proto.abs() * omega_c; // take abs for symmetric pair
        let sz_im = z_im_proto * omega_c;

        stages.push(elliptic_lp_sos(sp_re, sp_im, sz_im, c));
    }

    // Odd order: one real prototype pole at s = −sn(v0, kp)/cn(v0, kp) = −sc(v0, kp)
    // = −1/eps_p (approximately), scaled by omega_c.
    if order % 2 == 1 {
        // The real LP prototype pole: s = -sc(v0_raw, kp1) ≈ -1/eps_p (normalized)
        // After scaling by omega_c: sp_abs = omega_c / eps_p (approx, works for small k1)
        let sp_abs = omega_c * sv / cv.max(1e-30);
        let da0_1 = c + sp_abs;
        stages.push(BiquadFilter {
            b0: (sp_abs / da0_1) as f32,
            b1: (sp_abs / da0_1) as f32,
            b2: 0.0,
            a1: ((sp_abs - c) / da0_1) as f32,
            a2: 0.0,
        });
    }

    Ok(EllipticFilter { stages, order })
}

/// Design an elliptic (Cauer) **highpass** filter.
///
/// Applies the LP→HP frequency transformation after computing the normalized
/// elliptic LP prototype poles and zeros.
///
/// - `order`: filter order (1–8).
/// - `cutoff_hz`: passband edge in Hz.
/// - `passband_ripple_db`: maximum passband ripple in dB.
/// - `stopband_attenuation_db`: minimum stopband attenuation in dB.
/// - `sample_rate`: sample rate in Hz.
#[must_use = "returns the designed elliptic highpass filter"]
pub fn elliptic_highpass(
    order: usize,
    cutoff_hz: f32,
    passband_ripple_db: f32,
    stopband_attenuation_db: f32,
    sample_rate: u32,
) -> Result<EllipticFilter, OxiAudioError> {
    if order == 0 || order > 8 {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "Elliptic filter order must be 1–8, got {order}"
        )));
    }
    let fs = sample_rate as f64;
    let omega_c = 2.0 * fs * (PI * cutoff_hz as f64 / fs).tan();
    let c = 2.0 * fs;

    let eps_p = (10.0_f64.powf(passband_ripple_db as f64 / 10.0) - 1.0).sqrt();
    let eps_s = (10.0_f64.powf(stopband_attenuation_db as f64 / 10.0) - 1.0).sqrt();
    let k1 = (eps_p / eps_s).clamp(1e-15, 1.0 - 1e-12);

    let k = find_elliptic_k(order, k1);
    let big_k = complete_k(k);
    let kp = (1.0 - k * k).sqrt().max(0.0);
    let v0 = elliptic_v0(eps_p, k1, big_k, order);

    let sv = jacobi_sn(v0, kp);
    let cv = jacobi_cn(v0, kp);
    let dv = jacobi_dn(v0, kp);

    let mut stages: Vec<BiquadFilter> = Vec::new();
    let n_pairs = order / 2;

    for i in 1..=n_pairs {
        let u_i = (2 * i - 1) as f64 * big_k / order as f64;
        let su = jacobi_sn(u_i, k);
        let cu = jacobi_cn(u_i, k);
        let du = jacobi_dn(u_i, k);

        let z_im_proto = 1.0 / (k * su.abs().max(1e-30));

        let d_sq = (du * sv).powi(2);
        let denom_p = (1.0 - d_sq).max(1e-30);
        let p_re_proto = -(cu * du * sv * cv) / denom_p;
        let p_im_proto = -(su * dv) / denom_p;

        // LP→HP transform: p_hp = omega_c / p_lp_proto
        // Using: omega_c / (p_re + j*p_im) = omega_c*(p_re - j*p_im)/(p_re^2+p_im^2)
        let p_mag_sq = (p_re_proto * p_re_proto + p_im_proto * p_im_proto).max(1e-60);
        let sp_re = omega_c * p_re_proto / p_mag_sq;
        let sp_im = omega_c * p_im_proto.abs() / p_mag_sq;

        // LP→HP zero: z_hp = omega_c / z_lp → imaginary: omega_c / z_im_proto
        let sz_im = omega_c / z_im_proto;

        // HP SOS: unity gain at Nyquist (no k_dc factor for HP)
        let pole_mag_sq = sp_re * sp_re + sp_im * sp_im;
        let zero_mag_sq = sz_im * sz_im;
        let nb0 = c * c + zero_mag_sq;
        let nb1 = 2.0 * zero_mag_sq - 2.0 * c * c;
        let nb2 = nb0;
        let da0 = c * c - 2.0 * sp_re * c + pole_mag_sq;
        let da1 = 2.0 * pole_mag_sq - 2.0 * c * c;
        let da2 = c * c + 2.0 * sp_re * c + pole_mag_sq;
        if da0.abs() < 1e-30 {
            continue;
        }
        stages.push(BiquadFilter {
            b0: (nb0 / da0) as f32,
            b1: (nb1 / da0) as f32,
            b2: (nb2 / da0) as f32,
            a1: (da1 / da0) as f32,
            a2: (da2 / da0) as f32,
        });
    }

    // Odd order: HP real pole
    if order % 2 == 1 {
        let sp_abs_lp = omega_c * sv / cv.max(1e-30); // LP pole magnitude
        let sp_hp = omega_c * omega_c / sp_abs_lp; // LP→HP: p_hp = omega_c / p_lp
        let da0_1 = c + sp_hp;
        stages.push(BiquadFilter {
            b0: (c / da0_1) as f32,
            b1: -(c / da0_1) as f32,
            b2: 0.0,
            a1: ((sp_hp - c) / da0_1) as f32,
            a2: 0.0,
        });
    }

    Ok(EllipticFilter { stages, order })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    use std::f32::consts::PI as PI_F32;

    fn sine_buf_mono(freq: f32, sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI_F32 * freq * i as f32 / sample_rate as f32).sin() * 0.5)
            .collect();
        AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn rms_amp(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn test_chebyshev2_lowpass_attenuates_stopband() {
        let sr = 48_000u32;
        let filt =
            chebyshev2_lowpass(4, 1000.0, 60.0, sr).expect("chebyshev2_lowpass should succeed");
        // 5kHz sine — well above 1kHz cutoff — should be strongly attenuated
        let buf = sine_buf_mono(5000.0, sr, 0.5);
        let out = filt.process(&buf);
        // skip transient
        let skip = (sr as f32 * 0.1) as usize;
        let out_rms = rms_amp(&out.samples[skip..]);
        assert!(
            out_rms < 1e-3,
            "5kHz through 1kHz Cheby II LP should be < 1e-3, got {out_rms:.6}"
        );
    }

    #[test]
    fn test_chebyshev2_lowpass_passes_passband() {
        let sr = 48_000u32;
        let filt =
            chebyshev2_lowpass(4, 5000.0, 60.0, sr).expect("chebyshev2_lowpass should succeed");
        // 500Hz sine — well below 5kHz cutoff — should pass
        let buf = sine_buf_mono(500.0, sr, 0.5);
        let out = filt.process(&buf);
        let skip = (sr as f32 * 0.05) as usize;
        let out_rms = rms_amp(&out.samples[skip..]);
        assert!(
            out_rms > 0.3,
            "500Hz through 5kHz Cheby II LP should pass (>0.3), got {out_rms:.4}"
        );
    }

    #[test]
    fn test_chebyshev2_highpass_attenuates_stopband() {
        let sr = 48_000u32;
        let filt =
            chebyshev2_highpass(4, 5000.0, 60.0, sr).expect("chebyshev2_highpass should succeed");
        // 500Hz sine — well below 5kHz cutoff — should be attenuated
        let buf = sine_buf_mono(500.0, sr, 0.5);
        let out = filt.process(&buf);
        let skip = (sr as f32 * 0.1) as usize;
        let out_rms = rms_amp(&out.samples[skip..]);
        assert!(
            out_rms < 1e-2,
            "500Hz through 5kHz Cheby II HP should be < 1e-2, got {out_rms:.6}"
        );
    }

    #[test]
    fn test_chebyshev2_highpass_passes_passband() {
        let sr = 48_000u32;
        let filt =
            chebyshev2_highpass(4, 1000.0, 60.0, sr).expect("chebyshev2_highpass should succeed");
        // 10kHz sine — well above 1kHz cutoff — should pass
        let buf = sine_buf_mono(10_000.0, sr, 0.5);
        let out = filt.process(&buf);
        let skip = (sr as f32 * 0.05) as usize;
        let out_rms = rms_amp(&out.samples[skip..]);
        assert!(
            out_rms > 0.3,
            "10kHz through 1kHz Cheby II HP should pass (>0.3), got {out_rms:.4}"
        );
    }

    #[test]
    fn test_elliptic_lowpass_sharp_rolloff() {
        let sr = 48_000u32;
        let filt =
            elliptic_lowpass(4, 1000.0, 1.0, 60.0, sr).expect("elliptic_lowpass should succeed");
        // 2kHz sine — one octave above cutoff — should be attenuated
        let buf = sine_buf_mono(2000.0, sr, 0.5);
        let out = filt.process(&buf);
        let skip = (sr as f32 * 0.1) as usize;
        let out_rms = rms_amp(&out.samples[skip..]);
        assert!(
            out_rms < 1e-2,
            "2kHz through 1kHz elliptic LP should be < 1e-2 (sharp rolloff), got {out_rms:.6}"
        );
    }

    #[test]
    fn test_elliptic_highpass_attenuates_stopband() {
        // 4th-order elliptic HP with passband edge at 5 kHz (1 dB ripple, 60 dB stopband).
        // A 500 Hz tone is well within the stopband — should be strongly attenuated.
        let sr = 44_100u32;
        let filt =
            elliptic_highpass(4, 5000.0, 1.0, 60.0, sr).expect("elliptic_highpass should succeed");
        let buf = sine_buf_mono(500.0, sr, 0.5);
        let out = filt.process(&buf);
        let skip = (sr as f32 * 0.1) as usize;
        let out_rms = rms_amp(&out.samples[skip..]);
        assert!(
            out_rms < 1e-2,
            "500 Hz through 5 kHz elliptic HP should be < 1e-2 (sharp stopband), got {out_rms:.6}"
        );
    }
}
