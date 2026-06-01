//! Opus SILK mode encoder (RFC 6716 §5) — LP analysis + range-coded bitstream.
//!
//! SILK (Skype Internet Low-bitrate Codec) is the voice codec used by Opus
//! for narrowband and wideband speech at low bitrates. This module provides
//! the LP analysis pipeline:
//!
//! 1. Autocorrelation with Gaussian lag window
//! 2. Levinson-Durbin recursion → LPC coefficients
//! 3. LPC → NLSF (Normalized Line Spectral Frequencies) via sum/difference
//!    Chebyshev polynomials
//! 4. Pitch estimation (normalized cross-correlation over min..max lag)
//! 5. 5-tap LTP gain estimation
//! 6. Range-coded bitstream packing via [`crate::opus_range::RangeEncoder`]
//!
//! # Frame structure
//!
//! SILK operates at 8/12/16/24 kHz internal sample rates. Each SILK frame is
//! 20 ms (160/240/320/480 samples at 8/12/16/24 kHz respectively).

use crate::opus_range::RangeEncoder;
use oxiaudio_core::OxiAudioError;

// ── Bandwidth parameters ───────────────────────────────────────────────────────

/// SILK internal sample rate selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilkBandwidth {
    /// 8 kHz narrowband (NB) — 160 samples per 20 ms frame.
    Narrowband,
    /// 12 kHz mediumband (MB) — 240 samples per 20 ms frame.
    Mediumband,
    /// 16 kHz wideband (WB) — 320 samples per 20 ms frame.
    Wideband,
    /// 24 kHz superwideband (SWB) — 480 samples per 20 ms frame.
    Superwideband,
}

impl SilkBandwidth {
    /// Internal sample rate in Hz.
    pub fn sample_rate_hz(self) -> u32 {
        match self {
            Self::Narrowband => 8_000,
            Self::Mediumband => 12_000,
            Self::Wideband => 16_000,
            Self::Superwideband => 24_000,
        }
    }

    /// Samples per 20 ms frame at this bandwidth.
    pub fn frame_size(self) -> usize {
        (self.sample_rate_hz() as usize * 20) / 1000
    }
}

/// LPC predictor order per bandwidth.
fn lpc_order(bw: SilkBandwidth) -> usize {
    match bw {
        SilkBandwidth::Narrowband => 10,
        SilkBandwidth::Mediumband => 12,
        SilkBandwidth::Wideband => 16,
        SilkBandwidth::Superwideband => 16,
    }
}

/// Minimum pitch lag (samples) per bandwidth.
fn pitch_min_lag(bw: SilkBandwidth) -> usize {
    match bw {
        SilkBandwidth::Narrowband => 12,
        SilkBandwidth::Mediumband => 18,
        SilkBandwidth::Wideband => 24,
        SilkBandwidth::Superwideband => 24,
    }
}

/// Maximum pitch lag (samples) — 288 is the SILK spec ceiling.
fn pitch_max_lag(_bw: SilkBandwidth) -> usize {
    288
}

// ── Frame type ────────────────────────────────────────────────────────────────

/// Linear prediction coefficients (LPC) for one SILK frame.
///
/// LPC captures the short-term spectral envelope using an autoregressive
/// model of order `lpc_order` (typically 16 for WB, 10 for NB).
#[derive(Debug, Clone)]
pub struct SilkLpcFrame {
    /// LPC coefficients as NLSFs (Normalized Line Spectral Frequencies),
    /// sorted ascending in (0, 1). Length equals `lpc_order(bw)`.
    pub nlsf: Vec<f32>,
    /// Short-term prediction residual after LPC synthesis filter removal.
    pub residual: Vec<f32>,
    /// Noise-shaped LPC residual (residual after perceptual pre-emphasis).
    pub shaped_residual: Vec<f32>,
    /// Noise shaping filter coefficients (same length as LPC order).
    pub ns_filter: Vec<f32>,
    /// Long-term prediction (LTP) pitch period in samples.
    pub pitch_lag: u16,
    /// LTP gains (5 taps centered on pitch_lag).
    pub ltp_gains: [f32; 5],
    /// Signal gain (logarithmic, 6 dB steps, stored as u8).
    pub gain_log2: u8,
}

impl SilkLpcFrame {
    /// Create a zero-signal (silence) SILK frame for the given LPC order.
    ///
    /// `order` is typically obtained via `lpc_order(bw)`:
    /// - Narrowband: 10, Mediumband: 12, Wideband/SWB: 16.
    pub fn silence(order: usize) -> Self {
        // Give evenly-spaced NLSFs so silence frames are valid (monotonic, in range).
        let nlsf = (1..=order).map(|i| i as f32 / (order + 1) as f32).collect();
        Self {
            nlsf,
            residual: Vec::new(),
            shaped_residual: Vec::new(),
            ns_filter: Vec::new(),
            pitch_lag: 100,
            ltp_gains: [0.0; 5],
            gain_log2: 0,
        }
    }
}

// ── Autocorrelation ───────────────────────────────────────────────────────────

/// Compute biased autocorrelation `r[k] = sum_{i=k}^{N-1} x[i]*x[i-k]` for k=0..=order.
fn autocorrelate(pcm: &[f32], order: usize) -> Vec<f64> {
    let n = pcm.len();
    (0..=order)
        .map(|k| {
            (k..n)
                .map(|i| (pcm[i] as f64) * (pcm[i - k] as f64))
                .sum::<f64>()
        })
        .collect()
}

/// Apply a Gaussian lag-window to the autocorrelation sequence (r[0] unchanged).
///
/// `w[k] = exp(-0.5 * (0.003 * k)^2)` — smoothly reduces high-lag correlations
/// to improve LP stability.
fn apply_lag_window(r: &mut [f64]) {
    for (k, v) in r.iter_mut().enumerate().skip(1) {
        let w = (-0.5 * (0.003 * k as f64).powi(2)).exp();
        *v *= w;
    }
}

// ── Levinson-Durbin ───────────────────────────────────────────────────────────

/// Levinson-Durbin recursion → LPC predictor coefficients of length `order`.
///
/// Returns `a[1..=order]` (the negated reflection coefficients) as f32.
fn levinson_durbin(r: &[f64]) -> Vec<f32> {
    let order = r.len() - 1;
    if r[0].abs() < 1e-20 {
        return vec![0.0f32; order];
    }
    let mut a = vec![0.0f64; order + 1];
    let mut a_prev = vec![0.0f64; order + 1];
    let mut err = r[0];

    for i in 1..=order {
        let mut lambda = r[i];
        for j in 1..i {
            lambda -= a[j] * r[i - j];
        }
        let ki = -lambda / err;

        // Copy forward so we can update in-place referencing the old values.
        a_prev[..=i].copy_from_slice(&a[..=i]);
        a[i] = ki;
        for j in 1..i {
            a[j] = a_prev[j] + ki * a_prev[i - j];
        }

        err *= 1.0 - ki * ki;
        if err <= 0.0 {
            break;
        }
    }

    a[1..=order].iter().map(|&x| x as f32).collect()
}

// ── LPC → NLSF ───────────────────────────────────────────────────────────────

/// Convert LPC coefficients to Normalized Line Spectral Frequencies (NLSFs).
///
/// Uses the standard sum-and-difference polynomial approach:
/// - P(z) = A(z) + z^{-(p+1)} A(z^{-1})  (symmetric)
/// - Q(z) = A(z) - z^{-(p+1)} A(z^{-1})  (antisymmetric)
///
/// The roots of P and Q on the unit circle alternate and give the NLSFs in (0, 1).
fn lpc_to_nlsf(lpc: &[f32]) -> Vec<f32> {
    let p = lpc.len();

    // Build coefficients for P (symmetric) and Q (antisymmetric) of length p+2.
    let mut pa = vec![0.0f64; p + 2];
    let mut qa = vec![0.0f64; p + 2];
    pa[0] = 1.0;
    qa[0] = 1.0;
    for (i, &c) in lpc.iter().enumerate() {
        let cv = c as f64;
        pa[i + 1] += cv;
        pa[p - i] += cv; // symmetric partner
        qa[i + 1] += cv;
        qa[p - i] -= cv; // antisymmetric partner
    }
    pa[p + 1] += 1.0;
    qa[p + 1] -= 1.0;

    // Number of real roots in (0, π) expected for each polynomial.
    let n_roots_p = p / 2 + 1;
    let n_roots_q = p / 2 + (p % 2);

    let mut roots_p = find_roots_chebyshev(&pa, n_roots_p);
    let mut roots_q = find_roots_chebyshev(&qa, n_roots_q);

    // Combine, sort, and take the p best roots.
    roots_p.append(&mut roots_q);
    roots_p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    roots_p.truncate(p);

    // Clamp to valid range and enforce strict monotonicity with minimum spacing.
    enforce_nlsf_validity(&mut roots_p);
    roots_p
}

/// Find up to `n_roots` roots of the Chebyshev polynomial form of `poly` in θ ∈ (0, π).
///
/// Evaluates `eval_cheby(poly, cos(θ))` on a fine grid and uses bisection on sign changes.
fn find_roots_chebyshev(poly: &[f64], n_roots: usize) -> Vec<f32> {
    const N_STEPS: usize = 2000;
    let mut roots = Vec::with_capacity(n_roots);
    let mut prev = eval_cheby(poly, 1.0); // θ=0 → cos(0)=1

    for i in 1..=N_STEPS {
        let theta = std::f64::consts::PI * i as f64 / N_STEPS as f64;
        let x = theta.cos();
        let curr = eval_cheby(poly, x);

        if prev * curr < 0.0 {
            // Bisect to locate the sign change.
            let theta_lo = std::f64::consts::PI * (i - 1) as f64 / N_STEPS as f64;
            let theta_hi = theta;
            let mut lo = theta_lo;
            let mut hi = theta_hi;

            for _ in 0..30 {
                let mid = (lo + hi) / 2.0;
                let y = eval_cheby(poly, mid.cos());
                if eval_cheby(poly, lo.cos()) * y < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }

            let root_theta = (lo + hi) / 2.0;
            roots.push((root_theta / std::f64::consts::PI) as f32);
            if roots.len() >= n_roots {
                break;
            }
        }
        prev = curr;
    }

    // Pad with evenly spaced values if we didn't find enough roots.
    // Place padded values in the gaps left by existing roots.
    while roots.len() < n_roots {
        let t = (roots.len() + 1) as f32 / (n_roots + 1) as f32;
        roots.push(t);
    }

    roots
}

/// Evaluate the polynomial `poly[0] + poly[1]*x + poly[2]*x^2 + ...` at `x`.
fn eval_cheby(poly: &[f64], x: f64) -> f64 {
    poly.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Enforce valid NLSF constraints: values in (0.001, 0.999) and strictly increasing
/// with minimum spacing of 0.002.
///
/// Two-pass algorithm:
/// 1. Forward pass: clamp and push each NLSF right of its predecessor.
/// 2. Backward pass: if the last value exceeds the ceiling, push back from the top.
fn enforce_nlsf_validity(nlsf: &mut [f32]) {
    if nlsf.is_empty() {
        return;
    }
    let n = nlsf.len();
    const MIN_VAL: f32 = 0.001;
    const MAX_VAL: f32 = 0.999;
    const MIN_SPACING: f32 = 0.002;

    // Forward pass: ensure each value is strictly above its predecessor.
    nlsf[0] = nlsf[0].clamp(MIN_VAL, MAX_VAL);
    for i in 1..n {
        let floor = (nlsf[i - 1] + MIN_SPACING).min(MAX_VAL);
        nlsf[i] = nlsf[i].clamp(floor, MAX_VAL);
    }

    // Backward pass: if the last value is at the ceiling, work backwards.
    if nlsf[n - 1] >= MAX_VAL - f32::EPSILON {
        nlsf[n - 1] = MAX_VAL;
        for i in (0..n - 1).rev() {
            let ceil = (nlsf[i + 1] - MIN_SPACING).max(MIN_VAL);
            if nlsf[i] > ceil {
                nlsf[i] = ceil;
            }
        }
    }

    // Final clamp pass for safety.
    for v in nlsf.iter_mut() {
        *v = v.clamp(MIN_VAL, MAX_VAL);
    }
}

// ── Pitch estimation ──────────────────────────────────────────────────────────

/// Estimate pitch lag by normalized cross-correlation search over `min_lag..=max_lag`.
fn estimate_pitch(pcm: &[f32], min_lag: usize, max_lag: usize) -> u16 {
    let n = pcm.len();
    let max_lag = max_lag.min(n / 2);
    if max_lag < min_lag {
        return min_lag as u16;
    }

    let mut best_lag = min_lag;
    let mut best_corr = f32::NEG_INFINITY;

    for lag in min_lag..=max_lag {
        let mut numer = 0.0f32;
        let mut denom_a = 0.0f32;
        let mut denom_b = 0.0f32;
        for i in lag..n {
            numer += pcm[i] * pcm[i - lag];
            denom_a += pcm[i] * pcm[i];
            denom_b += pcm[i - lag] * pcm[i - lag];
        }
        let denom = (denom_a * denom_b).sqrt().max(1e-10);
        let corr = numer / denom;
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    best_lag as u16
}

// ── LTP gains ─────────────────────────────────────────────────────────────────

/// Compute 5-tap LTP predictor gains centered on `pitch_lag`.
fn compute_ltp_gains(pcm: &[f32], pitch_lag: u16) -> [f32; 5] {
    let lag = pitch_lag as usize;
    let n = pcm.len();
    let mut gains = [0.0f32; 5];
    let energy: f32 = pcm.iter().map(|&x| x * x).sum::<f32>().max(1e-10);

    for (tap, gain) in gains.iter_mut().enumerate() {
        let offset = tap as isize - 2;
        let shifted = lag as isize + offset;
        if shifted > 0 {
            let sl = shifted as usize;
            if sl < n {
                let cross: f32 = (sl..n).map(|i| pcm[i] * pcm[i - sl]).sum();
                *gain = (cross / energy).clamp(-1.0, 1.0);
            }
        }
    }

    gains
}

// ── LPC residual ──────────────────────────────────────────────────────────────

/// Compute the LPC prediction residual: `e[n] = x[n] - sum_k a[k] * x[n-k-1]`.
fn compute_residual(pcm: &[f32], lpc: &[f32]) -> Vec<f32> {
    pcm.iter()
        .enumerate()
        .map(|(n, &s)| {
            let pred: f32 = lpc
                .iter()
                .enumerate()
                .filter(|&(k, _)| n > k)
                .map(|(k, &a)| -a * pcm[n - k - 1])
                .sum();
            s - pred
        })
        .collect()
}

// ── Noise shaping ─────────────────────────────────────────────────────────────

/// Compute noise shaping filter coefficients from LPC coefficients.
///
/// The noise shaping filter is `A(z)` with bandwidth expanded by a factor < 1.0,
/// which widens the LPC filter's formants slightly. This concentrates the
/// quantization noise into frequency regions where the ear is less sensitive.
///
/// `bw_expansion`: bandwidth expansion factor, typically 0.85 for SILK.
pub fn compute_noise_shaping_filter(lpc: &[f32], bw_expansion: f32) -> Vec<f32> {
    lpc.iter()
        .enumerate()
        .map(|(k, &a)| a * bw_expansion.powi((k as i32) + 1))
        .collect()
}

/// Apply noise shaping to a residual signal.
///
/// The noise shaping filter is applied as an IIR filter to the LPC residual,
/// spreading quantization noise into less perceptible frequency regions.
///
/// # Arguments
/// * `residual` — LPC residual signal (from `SilkLpcFrame::residual`)
/// * `ns_filter` — noise shaping filter coefficients (from `compute_noise_shaping_filter`)
pub fn apply_noise_shaping(residual: &[f32], ns_filter: &[f32]) -> Vec<f32> {
    let p = ns_filter.len();
    let mut shaped = Vec::with_capacity(residual.len());
    let mut state = vec![0.0f32; p];

    for &r in residual {
        // IIR prediction: pred = sum_k(ns_filter[k] * state[k])
        let pred: f32 = ns_filter
            .iter()
            .zip(state.iter())
            .map(|(&a, &s)| -a * s)
            .sum();
        let out = r - pred;
        shaped.push(out);
        // Update state (shift register)
        if p > 0 {
            state.rotate_right(1);
            state[0] = out;
        }
    }
    shaped
}

// ── LBRR (Low Bitrate Redundancy) ────────────────────────────────────────────

/// LBRR (Low Bitrate Redundancy) mode flag.
///
/// When enabled, a compressed version of the previous frame is embedded
/// in the current frame for packet loss recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LbrrMode {
    /// No redundancy (default for high-bitrate scenarios).
    #[default]
    Disabled,
    /// Include redundant previous-frame data.
    Enabled,
}

/// Encode a SILK frame with optional LBRR header.
///
/// The LBRR flag is the first byte of the SILK payload (simplified from RFC 6716 §3.2.2).
/// When `lbrr == Enabled` and `prev_frame` is `Some`, the previous frame is embedded
/// before the current frame for packet-loss recovery.
///
/// # Packet layout
/// ```text
/// [lbrr_flag: u8] [prev_len: u16 BE]? [prev_frame_bytes]? [current_frame_bytes]
/// ```
pub fn encode_silk_frame_with_lbrr(
    frame: &SilkLpcFrame,
    bw: SilkBandwidth,
    lbrr: LbrrMode,
    prev_frame: Option<&SilkLpcFrame>,
) -> Vec<u8> {
    let mut out = Vec::new();

    // LBRR flag (1 byte): 0 = no LBRR, 1 = LBRR active.
    let lbrr_active = lbrr == LbrrMode::Enabled && prev_frame.is_some();
    out.push(u8::from(lbrr_active));

    if lbrr_active {
        if let Some(prev) = prev_frame {
            // Encode previous frame at the same quality as the current frame.
            let prev_bytes = encode_silk_frame(prev, bw);
            // LBRR frame length (2 bytes, big-endian) followed by the frame data.
            let len = prev_bytes.len() as u16;
            out.extend_from_slice(&len.to_be_bytes());
            out.extend_from_slice(&prev_bytes);
        }
    }

    // Current frame.
    let frame_bytes = encode_silk_frame(frame, bw);
    out.extend_from_slice(&frame_bytes);
    out
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Analyze a PCM frame and extract SILK LP parameters via Levinson-Durbin.
///
/// # Algorithm
///
/// 1. Biased autocorrelation with Gaussian lag window.
/// 2. Levinson-Durbin → LPC coefficients.
/// 3. LPC → NLSF via sum/difference Chebyshev polynomial root finding.
/// 4. Pitch estimation via normalized cross-correlation.
/// 5. 5-tap LTP gain estimation.
/// 6. Signal gain quantized to 6 dB steps (u8).
pub fn analyze_silk_frame(pcm: &[f32], bw: SilkBandwidth) -> SilkLpcFrame {
    let order = lpc_order(bw);

    let mut r = autocorrelate(pcm, order);
    apply_lag_window(&mut r);
    let lpc = levinson_durbin(&r);

    let nlsf = lpc_to_nlsf(&lpc);

    let pitch_lag = estimate_pitch(pcm, pitch_min_lag(bw), pitch_max_lag(bw));
    let ltp_gains = compute_ltp_gains(pcm, pitch_lag);

    let energy: f32 = pcm.iter().map(|&x| x * x).sum::<f32>() / pcm.len().max(1) as f32;
    let gain_log2 = if energy > 1e-10 {
        (energy.sqrt().log2() * 16.0 + 127.0).clamp(0.0, 255.0) as u8
    } else {
        0
    };

    let residual = compute_residual(pcm, &lpc);

    let ns_filter = compute_noise_shaping_filter(&lpc, 0.85);
    let shaped_residual = apply_noise_shaping(&residual, &ns_filter);

    SilkLpcFrame {
        nlsf,
        residual,
        shaped_residual,
        ns_filter,
        pitch_lag,
        ltp_gains,
        gain_log2,
    }
}

/// Encode a [`SilkLpcFrame`] into a compact bitstream using the Opus range coder.
///
/// Encoding layout (all uniform):
/// - `gain_log2` — uniform 256
/// - `pitch_lag` — 9 raw bits, offset by `min_lag(bw)` (covers 0..=288)
/// - Each NLSF — uniform 256 (scaled from \[0,1\] to \[0,255\])
/// - Each LTP gain (5 taps) — uniform 256 (shifted from \[-1,1\] to \[0,255\])
pub fn encode_silk_frame(frame: &SilkLpcFrame, bw: SilkBandwidth) -> Vec<u8> {
    let mut enc = RangeEncoder::new();

    // Gain (8-bit uniform).
    enc.encode_uint(frame.gain_log2 as u32, 256);

    // Pitch lag: offset from min_lag, encoded as 9 raw bits (covers 0..=288).
    // We do NOT use encode_uint(val, 277) because the range coder's encode_uint
    // splits val >> (nbits-8) and asserts val < n_high, which fails for values
    // in (256, 277) with the current RangeEncoder implementation.
    // Raw bits are always safe and give equivalent entropy.
    let min_lag = pitch_min_lag(bw) as u32;
    let max_lag = pitch_max_lag(bw) as u32;
    let lag_val = (frame.pitch_lag as u32)
        .saturating_sub(min_lag)
        .min(max_lag - min_lag);
    // 9 bits covers 0..=288
    enc.encode_bits_raw(lag_val, 9);

    // NLSFs (each quantized to 8-bit uniform).
    for &n in &frame.nlsf {
        let quantized = (n.clamp(0.0, 1.0) * 255.0).round() as u32;
        enc.encode_uint(quantized.min(255), 256);
    }

    // LTP gains (5 taps, shifted from [-1,1] to [0,255]).
    for &g in &frame.ltp_gains {
        let quantized = ((g.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u32;
        enc.encode_uint(quantized.min(255), 256);
    }

    enc.finish()
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors specific to SILK mode encoding.
#[derive(Debug)]
pub enum SilkError {
    /// PCM buffer length does not match expected frame size.
    FrameSizeMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for SilkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FrameSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "SILK frame size mismatch: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl From<SilkError> for OxiAudioError {
    fn from(e: SilkError) -> Self {
        OxiAudioError::UnsupportedFormat(e.to_string())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_pcm(freq: f32, n: usize, rate: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / rate).sin() * 0.5)
            .collect()
    }

    // ── Original structural tests (updated for new behavior) ──────────────────

    #[test]
    fn test_silk_bandwidth_frame_sizes() {
        assert_eq!(SilkBandwidth::Narrowband.frame_size(), 160);
        assert_eq!(SilkBandwidth::Mediumband.frame_size(), 240);
        assert_eq!(SilkBandwidth::Wideband.frame_size(), 320);
        assert_eq!(SilkBandwidth::Superwideband.frame_size(), 480);
    }

    #[test]
    fn test_silk_bandwidth_sample_rates() {
        assert_eq!(SilkBandwidth::Narrowband.sample_rate_hz(), 8_000);
        assert_eq!(SilkBandwidth::Wideband.sample_rate_hz(), 16_000);
    }

    #[test]
    fn test_analyze_silk_frame_returns_silence() {
        let pcm = vec![0.0f32; 320]; // Wideband frame
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        assert_eq!(frame.nlsf.len(), 16);
        assert_eq!(frame.residual.len(), 320);
    }

    /// Updated: encode now produces non-empty bytes (range coder flush always
    /// emits at least 4 bytes).
    #[test]
    fn test_encode_silk_frame_silence_no_panic() {
        let frame = SilkLpcFrame::silence(10); // Narrowband order = 10
        let bytes = encode_silk_frame(&frame, SilkBandwidth::Narrowband);
        // After LP implementation, the range coder always produces bytes.
        assert!(!bytes.is_empty(), "encoded frame must produce bytes");
    }

    #[test]
    fn test_silk_lpc_frame_silence_dimensions() {
        let nb = SilkLpcFrame::silence(10); // Narrowband order
        assert_eq!(nb.nlsf.len(), 10);
        let wb = SilkLpcFrame::silence(16); // Wideband order
        assert_eq!(wb.nlsf.len(), 16);
    }

    // ── LP analysis tests ─────────────────────────────────────────────────────

    #[test]
    fn test_silk_analyze_nlsf_count() {
        let pcm = sine_pcm(440.0, 320, 16000.0);
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        assert_eq!(frame.nlsf.len(), 16, "WB needs 16 NLSFs");
    }

    #[test]
    fn test_silk_analyze_nlsf_range() {
        let pcm = sine_pcm(440.0, 320, 16000.0);
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        for &n in &frame.nlsf {
            assert!(n > 0.0 && n < 1.0, "NLSF must be in (0, 1), got {n}");
        }
    }

    #[test]
    fn test_silk_analyze_nlsf_monotonic() {
        let pcm = sine_pcm(200.0, 320, 16000.0);
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        for i in 1..frame.nlsf.len() {
            assert!(
                frame.nlsf[i] > frame.nlsf[i - 1],
                "NLSFs must be monotonic: [{}]={} <= [{}]={}",
                i - 1,
                frame.nlsf[i - 1],
                i,
                frame.nlsf[i]
            );
        }
    }

    #[test]
    fn test_silk_analyze_nlsf_monotonic_sine_1khz() {
        let pcm = sine_pcm(1000.0, 320, 16000.0);
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        for i in 1..frame.nlsf.len() {
            assert!(
                frame.nlsf[i] > frame.nlsf[i - 1],
                "NLSFs must be strictly monotonic at 1 kHz: [{i}]={}",
                frame.nlsf[i]
            );
        }
    }

    #[test]
    fn test_silk_analyze_pitch_in_range() {
        // 100 Hz sine at 16 kHz → pitch lag ≈ 160 samples
        let pcm = sine_pcm(100.0, 320, 16000.0);
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        assert!(
            frame.pitch_lag >= 12 && frame.pitch_lag <= 288,
            "pitch_lag out of range: {}",
            frame.pitch_lag
        );
    }

    #[test]
    fn test_silk_analyze_silence_gain_zero() {
        let pcm = vec![0.0f32; 320];
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        assert_eq!(frame.gain_log2, 0, "silence should give gain_log2=0");
    }

    #[test]
    fn test_silk_encode_frame_nonempty() {
        let pcm = sine_pcm(440.0, 320, 16000.0);
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        let bytes = encode_silk_frame(&frame, SilkBandwidth::Wideband);
        assert!(!bytes.is_empty(), "encoded frame must be non-empty");
    }

    #[test]
    fn test_silk_analyze_residual_length() {
        let pcm = sine_pcm(440.0, 320, 16000.0);
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        assert_eq!(
            frame.residual.len(),
            pcm.len(),
            "residual must be same length as input"
        );
    }

    #[test]
    fn test_silk_analyze_narrowband() {
        let pcm = sine_pcm(300.0, 160, 8000.0);
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Narrowband);
        assert_eq!(frame.nlsf.len(), 10);
        assert_eq!(frame.residual.len(), 160);
        for &n in &frame.nlsf {
            assert!(n > 0.0 && n < 1.0, "NB NLSF out of range: {n}");
        }
    }

    #[test]
    fn test_silk_analyze_superwideband() {
        let pcm = sine_pcm(440.0, 480, 24000.0);
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Superwideband);
        assert_eq!(frame.nlsf.len(), 16);
        assert_eq!(frame.residual.len(), 480);
    }

    #[test]
    fn test_silk_silence_frame_valid_nlsf() {
        // Silence frames should have valid (monotonic, in-range) NLSFs.
        let frame = SilkLpcFrame::silence(16); // Wideband order = 16
        assert_eq!(frame.nlsf.len(), 16);
        for &n in &frame.nlsf {
            assert!(n > 0.0 && n < 1.0, "silence NLSF out of range: {n}");
        }
        for i in 1..frame.nlsf.len() {
            assert!(
                frame.nlsf[i] > frame.nlsf[i - 1],
                "silence NLSFs not monotonic at index {i}"
            );
        }
    }

    #[test]
    fn test_silk_encode_narrowband_nonempty() {
        let pcm = sine_pcm(300.0, 160, 8000.0);
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Narrowband);
        let bytes = encode_silk_frame(&frame, SilkBandwidth::Narrowband);
        assert!(!bytes.is_empty(), "NB encoded frame must be non-empty");
    }

    #[test]
    fn test_levinson_durbin_zero_input() {
        // Zero autocorrelation should produce all-zero predictor.
        let r = vec![0.0f64; 11];
        let lpc = levinson_durbin(&r);
        assert_eq!(lpc.len(), 10);
        for c in lpc {
            assert_eq!(c, 0.0f32);
        }
    }

    #[test]
    fn test_enforce_nlsf_validity_forces_monotone() {
        let mut v = vec![0.5f32; 16]; // all equal — should be spread out
        enforce_nlsf_validity(&mut v);
        for i in 1..v.len() {
            assert!(
                v[i] > v[i - 1],
                "enforce_nlsf_validity failed at index {i}: {} vs {}",
                v[i - 1],
                v[i]
            );
        }
    }

    // ── Noise shaping tests ───────────────────────────────────────────────────

    #[test]
    fn test_noise_shaping_filter_length() {
        let lpc = vec![0.5f32, -0.3, 0.1];
        let ns = compute_noise_shaping_filter(&lpc, 0.85);
        assert_eq!(
            ns.len(),
            lpc.len(),
            "NS filter must have same length as LPC"
        );
    }

    #[test]
    fn test_noise_shaping_filter_decays() {
        let lpc = vec![0.5f32, -0.3, 0.1];
        let ns = compute_noise_shaping_filter(&lpc, 0.85);
        // Bandwidth expansion makes higher-order coefficients smaller in magnitude.
        assert!(
            ns[1].abs() < lpc[1].abs() * 0.9 + 1e-6,
            "Higher-order NS coefficients should be smaller due to BW expansion"
        );
    }

    #[test]
    fn test_apply_noise_shaping_output_length() {
        let residual = vec![0.1f32; 100];
        let ns_filter = vec![0.3f32, -0.1];
        let shaped = apply_noise_shaping(&residual, &ns_filter);
        assert_eq!(
            shaped.len(),
            residual.len(),
            "shaped residual must match input length"
        );
    }

    #[test]
    fn test_apply_noise_shaping_silence_stays_near_zero() {
        let residual = vec![0.0f32; 100];
        let ns_filter = vec![0.3f32, -0.1, 0.05];
        let shaped = apply_noise_shaping(&residual, &ns_filter);
        let max = shaped.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        assert!(
            max < 1e-6,
            "zero residual should produce near-zero shaped output, max={max}"
        );
    }

    #[test]
    fn test_analyze_silk_frame_has_shaped_residual() {
        let pcm: Vec<f32> = (0..320)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        assert_eq!(
            frame.shaped_residual.len(),
            pcm.len(),
            "shaped_residual must have same length as input PCM"
        );
        assert_eq!(
            frame.ns_filter.len(),
            16, // WB LPC order = 16
            "ns_filter must have LPC order = 16 for Wideband"
        );
    }

    // ── LBRR tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_lbrr_disabled_matches_base_encode() {
        let pcm: Vec<f32> = (0..320)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        let base = encode_silk_frame(&frame, SilkBandwidth::Wideband);
        let lbrr_off =
            encode_silk_frame_with_lbrr(&frame, SilkBandwidth::Wideband, LbrrMode::Disabled, None);
        // LBRR disabled: lbrr_off = [0x00] + base bytes
        assert_eq!(
            lbrr_off.len(),
            base.len() + 1,
            "LBRR-disabled packet should be 1 byte longer than base"
        );
        assert_eq!(lbrr_off[0], 0, "LBRR flag byte must be 0 when disabled");
    }

    #[test]
    fn test_lbrr_enabled_embeds_prev_frame() {
        let pcm: Vec<f32> = (0..320)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let frame = analyze_silk_frame(&pcm, SilkBandwidth::Wideband);
        let prev = SilkLpcFrame::silence(16); // previous frame (WB order = 16)
        let encoded = encode_silk_frame_with_lbrr(
            &frame,
            SilkBandwidth::Wideband,
            LbrrMode::Enabled,
            Some(&prev),
        );
        assert_eq!(encoded[0], 1, "LBRR flag byte must be 1 when enabled");
        assert!(
            encoded.len() > 10,
            "LBRR-enabled packet must contain prev frame data"
        );
    }
}
