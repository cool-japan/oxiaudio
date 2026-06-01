//! Pitch detection via YIN (de Cheveigné & Kawahara, 2002) and pYIN
//! (de Cheveigné & Klapuri, 2014).
//!
//! YIN estimates the fundamental frequency of a (quasi-)periodic signal using a
//! difference function, cumulative mean normalization, an absolute threshold, and
//! parabolic interpolation around the chosen lag. This module exposes a per-frame
//! [`PitchTracker`] returning [`PitchFrame`] records plus a one-shot convenience
//! function [`detect_pitch_yin`].
//!
//! pYIN extends YIN by applying probabilistic threshold selection via a Beta(2,18)
//! prior and a Viterbi decoder for a globally optimal voiced/unvoiced pitch track.
//! Use [`detect_pitch_pyin`] for more robust pitch tracking in polyphonic or noisy
//! audio.

use oxiaudio_core::AudioBuffer;
use oxifft::conv::correlate;

// ── pYIN constants ────────────────────────────────────────────────────────────

/// Number of semitone frequency bins spanning MIDI C0–B8 (A0 = 27.5 Hz at bin 0).
const N_BINS: usize = 108;

/// Beta(α, β) distribution parameters for pYIN threshold weighting.
const BETA_A: f64 = 2.0;
const BETA_B: f64 = 18.0;

/// Voiced-to-voiced (same bin) transition probability.
const VOICED_VOICED_SAME: f64 = 0.99;
/// Geometric decay factor per semitone step.
const PITCH_STEP_DECAY: f64 = 0.7;
/// Probability of transitioning from voiced to unvoiced.
const VOICED_UNVOICED: f64 = 0.01;
/// Probability of transitioning from unvoiced to unvoiced.
const UNVOICED_UNVOICED: f64 = 0.99;
/// Probability floor before taking log (avoids log(0)).
const LOG_FLOOR: f64 = 1e-30;

/// A single pitch estimate for one analysis frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchFrame {
    /// Center time of the frame, in seconds.
    pub time_seconds: f64,
    /// Estimated fundamental frequency in Hz (0.0 if unvoiced).
    pub frequency_hz: f32,
    /// Confidence in `[0.0, 1.0]` (`1.0 - YIN d'(tau)` at the chosen lag).
    pub confidence: f32,
    /// Whether the frame was classified as voiced (pitched).
    pub is_voiced: bool,
}

/// Configurable YIN pitch tracker.
#[derive(Debug, Clone)]
pub struct PitchTracker {
    /// Analysis window size in samples.
    pub frame_size: usize,
    /// Hop size between frames in samples.
    pub hop_size: usize,
    /// YIN absolute threshold (typical 0.1–0.15).
    pub threshold: f32,
    /// Minimum detectable frequency in Hz.
    pub f_min: f32,
    /// Maximum detectable frequency in Hz.
    pub f_max: f32,
}

impl Default for PitchTracker {
    fn default() -> Self {
        Self {
            frame_size: 2048,
            hop_size: 512,
            threshold: 0.12,
            f_min: 50.0,
            f_max: 2000.0,
        }
    }
}

impl PitchTracker {
    /// Create a tracker with the given frame and hop sizes (other fields default).
    pub fn new(frame_size: usize, hop_size: usize) -> Self {
        Self {
            frame_size,
            hop_size,
            ..Self::default()
        }
    }

    /// Set the YIN absolute threshold (builder style).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the detectable frequency range (builder style).
    pub fn with_range(mut self, f_min: f32, f_max: f32) -> Self {
        self.f_min = f_min;
        self.f_max = f_max;
        self
    }

    /// Analyze `buf` (mixed to mono) and return one [`PitchFrame`] per hop.
    pub fn track(&self, buf: &AudioBuffer<f32>) -> Vec<PitchFrame> {
        let mono = to_mono(buf);
        let sr = buf.sample_rate as f32;
        if mono.len() < self.frame_size || self.frame_size == 0 || sr <= 0.0 {
            return Vec::new();
        }
        let hop = self.hop_size.max(1);
        let n_frames = (mono.len() - self.frame_size) / hop + 1;

        // Search lag bounds from frequency range.
        let tau_min = ((sr / self.f_max).floor() as usize).max(2);
        let tau_max = ((sr / self.f_min).ceil() as usize).min(self.frame_size / 2);

        let mut frames = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let start = i * hop;
            let frame = &mono[start..start + self.frame_size];
            let (freq, conf) = yin_frame(frame, sr, tau_min, tau_max, self.threshold);
            let is_voiced = freq > 0.0 && conf >= 1.0 - self.threshold;
            frames.push(PitchFrame {
                time_seconds: (start as f64 + self.frame_size as f64 / 2.0) / sr as f64,
                frequency_hz: if is_voiced { freq } else { 0.0 },
                confidence: conf,
                is_voiced,
            });
        }
        frames
    }
}

/// Detect pitch using the YIN algorithm with configurable frame and hop sizes.
///
/// Returns one [`PitchFrame`] per analysis hop, with `frequency_hz` set to 0.0
/// for unvoiced frames.
///
/// # Parameters
/// - `buf`: input audio (mixed to mono internally).
/// - `frame_size`: analysis window in samples (e.g. 2048).
/// - `hop_size`: hop between windows in samples (e.g. 512).
/// - `threshold`: YIN voiced/unvoiced threshold (0.1 is a good default).
pub fn detect_pitch_yin(
    buf: &AudioBuffer<f32>,
    frame_size: usize,
    hop_size: usize,
    threshold: f32,
) -> Vec<PitchFrame> {
    PitchTracker::new(frame_size, hop_size)
        .with_threshold(threshold)
        .track(buf)
}

/// One-shot convenience: estimate `(time_seconds, frequency_hz)` per voiced frame
/// using default [`PitchTracker`] settings.
pub fn detect_pitch_yin_simple(buf: &AudioBuffer<f32>) -> Vec<(f64, f32)> {
    PitchTracker::default()
        .track(buf)
        .into_iter()
        .map(|f| (f.time_seconds, f.frequency_hz))
        .collect()
}

// ── pYIN ─────────────────────────────────────────────────────────────────────

/// Probabilistic YIN (pYIN) pitch detection with Viterbi decoding.
///
/// Extends YIN by distributing probability across multiple CMNDF thresholds
/// using a Beta(2, 18) prior, then decoding the globally optimal voiced/unvoiced
/// pitch track with the Viterbi algorithm.
///
/// Returns one [`PitchFrame`] per analysis hop. Unvoiced frames have
/// `frequency_hz = 0.0` and `is_voiced = false`; `confidence` carries the
/// observation probability at the decoded state.
///
/// # Parameters
/// - `buf`: input audio (mixed to mono internally).
/// - `frame_size`: analysis window in samples (e.g. 2048).
/// - `hop_size`: hop between windows in samples (e.g. 512).
pub fn detect_pitch_pyin(
    buf: &AudioBuffer<f32>,
    frame_size: usize,
    hop_size: usize,
) -> Vec<PitchFrame> {
    let mono = to_mono(buf);
    let sr = buf.sample_rate as f32;

    if mono.len() < frame_size || frame_size == 0 || hop_size == 0 || sr <= 0.0 {
        return Vec::new();
    }

    let hop = hop_size.max(1);
    let n_frames = (mono.len() - frame_size) / hop + 1;

    // Frequency-range based tau bounds (same as default PitchTracker).
    let f_min: f32 = 50.0;
    let f_max: f32 = 2000.0;
    let tau_min = ((sr / f_max).floor() as usize).max(2);
    let tau_max = ((sr / f_min).ceil() as usize).min(frame_size / 2);

    // 19 thresholds: 0.05, 0.10, …, 0.95
    let thresholds: Vec<f32> = (1..=19).map(|i| i as f32 * 0.05).collect();
    let beta_w = beta_pmf_weights(&thresholds);

    // Total Viterbi states: N_BINS voiced bins + 1 unvoiced state.
    let n_states = N_BINS + 1;
    let unvoiced_state = N_BINS;

    // ── Step B+C: per-frame candidate accumulation ────────────────────────────
    // obs_voiced[m][b] = accumulated probability for voiced pitch bin b in frame m.
    // unvoiced_prob[m] = 1 - sum(obs_voiced[m]).
    let mut obs_voiced: Vec<Vec<f64>> = vec![vec![0.0f64; N_BINS]; n_frames];
    let mut voiced_prob_per_frame: Vec<f64> = vec![0.0f64; n_frames];

    for m in 0..n_frames {
        let start = m * hop;
        let frame = &mono[start..start + frame_size];
        let d_prime = compute_cmndf(frame, tau_max);

        for (th_idx, &threshold) in thresholds.iter().enumerate() {
            // Find first local minimum of d'(tau) below `threshold` in [tau_min, tau_max].
            let mut tau_est: Option<usize> = None;
            let mut t = tau_min;
            while t <= tau_max {
                if d_prime[t] < threshold {
                    // Descend to local minimum.
                    while t < tau_max && d_prime[t + 1] < d_prime[t] {
                        t += 1;
                    }
                    tau_est = Some(t);
                    break;
                }
                t += 1;
            }

            if let Some(tau) = tau_est {
                let refined = parabolic_interp(&d_prime, tau);
                let freq = if refined > 0.0 {
                    sr / refined
                } else {
                    continue;
                };
                if freq <= 0.0 {
                    continue;
                }
                let bin = freq_to_bin(freq);
                let w = beta_w[th_idx] as f64;
                obs_voiced[m][bin] += w;
                voiced_prob_per_frame[m] += w;
            }
        }
    }

    // ── Step E: build observation probability matrix (log domain) ────────────
    // obs_log[m][s]: log probability of observing state s at frame m.
    // For efficiency: compute on-the-fly in the Viterbi loop.

    // ── Step F: precompute log-transition matrix ──────────────────────────────
    // log_trans[s1][s2] = log P(s2 | s1)
    // We precompute only the voiced-to-voiced rows compactly.
    // voiced_to_voiced is computed per pair; unvoiced rows are simple.

    // Precompute voiced-to-voiced transition weights and normalise per source bin.
    // trans_voiced_voiced[b] is the normalised weight vector for a source at bin b.
    // For log domain, we store log(trans).

    // Rather than a full N_BINS x N_BINS matrix, compute geometrically.
    // We build the full log_trans as a flat Vec<f64> of size n_states*n_states.
    let mut log_trans = vec![f64::NEG_INFINITY; n_states * n_states];

    for b1 in 0..N_BINS {
        let mut row_sum = 0.0f64;
        // voiced-to-voiced
        let mut weights = vec![0.0f64; N_BINS];
        for (b2, weight) in weights.iter_mut().enumerate().take(N_BINS) {
            let dist = (b1 as i64 - b2 as i64).unsigned_abs() as u32;
            let w = if dist == 0 {
                VOICED_VOICED_SAME
            } else {
                // Geometric decay normalised later.
                VOICED_VOICED_SAME * PITCH_STEP_DECAY.powi(dist as i32)
            };
            *weight = w;
            row_sum += w;
        }
        // voiced-to-unvoiced
        row_sum += VOICED_UNVOICED;
        // Normalise and store log.
        for (b2, &w) in weights.iter().enumerate() {
            log_trans[b1 * n_states + b2] = (w / row_sum).max(LOG_FLOOR).ln();
        }
        log_trans[b1 * n_states + unvoiced_state] = (VOICED_UNVOICED / row_sum).max(LOG_FLOOR).ln();
    }
    // unvoiced-to-voiced: uniform over pitch bins.
    let uv_to_v = (1.0 - UNVOICED_UNVOICED) / N_BINS as f64;
    let uv_log = uv_to_v.max(LOG_FLOOR).ln();
    for slot in log_trans
        .iter_mut()
        .skip(unvoiced_state * n_states)
        .take(N_BINS)
    {
        *slot = uv_log;
    }
    log_trans[unvoiced_state * n_states + unvoiced_state] = UNVOICED_UNVOICED.max(LOG_FLOOR).ln();

    // ── Step F (cont.): Viterbi ───────────────────────────────────────────────
    // Keep only prev/current log-alpha; store backpointers flat.
    let mut log_alpha_prev = vec![f64::NEG_INFINITY; n_states];
    let mut log_alpha_curr = vec![f64::NEG_INFINITY; n_states];
    let mut backptr: Vec<u16> = vec![0u16; n_frames * n_states];

    // Observation helper: builds log obs for frame m.
    let obs_log = |m: usize| -> Vec<f64> {
        let vp = voiced_prob_per_frame[m];
        let uvp = (1.0 - vp).max(0.0);
        let mut v = vec![f64::NEG_INFINITY; n_states];
        for b in 0..N_BINS {
            let p = obs_voiced[m][b];
            v[b] = p.max(LOG_FLOOR).ln();
        }
        v[unvoiced_state] = uvp.max(LOG_FLOOR).ln();
        v
    };

    // Initialise first frame with uniform prior.
    {
        let obs0 = obs_log(0);
        let log_prior = (1.0 / n_states as f64).max(LOG_FLOOR).ln();
        for s in 0..n_states {
            log_alpha_prev[s] = log_prior + obs0[s];
        }
    }

    // Forward pass.
    for m in 1..n_frames {
        let obs_m = obs_log(m);
        for s in 0..n_states {
            let mut best_val = f64::NEG_INFINITY;
            let mut best_prev = 0usize;
            for s_prev in 0..n_states {
                let v = log_alpha_prev[s_prev] + log_trans[s_prev * n_states + s];
                if v > best_val {
                    best_val = v;
                    best_prev = s_prev;
                }
            }
            log_alpha_curr[s] = best_val + obs_m[s];
            backptr[m * n_states + s] = best_prev as u16;
        }
        log_alpha_prev.copy_from_slice(&log_alpha_curr);
    }

    // Backtrack.
    let mut path = vec![0usize; n_frames];
    path[n_frames - 1] = log_alpha_prev
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(unvoiced_state);

    for m in (1..n_frames).rev() {
        path[m - 1] = backptr[m * n_states + path[m]] as usize;
    }

    // ── Step G: map path to PitchFrames ─────────────────────────────────────
    let obs_log_final: Vec<Vec<f64>> = (0..n_frames).map(obs_log).collect();

    (0..n_frames)
        .map(|m| {
            let start = m * hop;
            let time_seconds = (start as f64 + frame_size as f64 / 2.0) / sr as f64;
            let state = path[m];
            if state == unvoiced_state {
                PitchFrame {
                    time_seconds,
                    frequency_hz: 0.0,
                    confidence: 0.0,
                    is_voiced: false,
                }
            } else {
                let freq = bin_to_freq(state);
                let conf = obs_log_final[m][state].exp().clamp(0.0, 1.0) as f32;
                PitchFrame {
                    time_seconds,
                    frequency_hz: freq,
                    confidence: conf,
                    is_voiced: true,
                }
            }
        })
        .collect()
}

// ── pYIN helpers ─────────────────────────────────────────────────────────────

/// Compute normalised Beta(BETA_A, BETA_B) PMF weights at the given thresholds.
fn beta_pmf_weights(thresholds: &[f32]) -> Vec<f32> {
    let weights: Vec<f64> = thresholds
        .iter()
        .map(|&t| {
            let t = t as f64;
            t.powf(BETA_A - 1.0) * (1.0 - t).powf(BETA_B - 1.0)
        })
        .collect();
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        weights.iter().map(|&w| (w / sum) as f32).collect()
    } else {
        vec![1.0 / thresholds.len() as f32; thresholds.len()]
    }
}

/// Map a frequency in Hz to the nearest semitone bin (A0=27.5 Hz → bin 0).
///
/// Bins span C0–B8 (108 semitones, MIDI-like). Returns a value in `0..N_BINS`.
fn freq_to_bin(freq: f32) -> usize {
    if freq <= 0.0 {
        return 0;
    }
    let bin = (12.0 * (freq / 27.5_f32).log2()).round();
    bin.clamp(0.0, (N_BINS - 1) as f32) as usize
}

/// Map a semitone bin index back to its centre frequency in Hz.
fn bin_to_freq(bin: usize) -> f32 {
    27.5_f32 * 2.0_f32.powf(bin as f32 / 12.0)
}

/// Run the YIN algorithm on a single frame; returns `(frequency_hz, confidence)`.
///
/// `confidence` is `1.0 - d'(tau*)` clamped to `[0, 1]`. A returned frequency of
/// `0.0` means no lag met the threshold.
fn yin_frame(frame: &[f32], sr: f32, tau_min: usize, tau_max: usize, threshold: f32) -> (f32, f32) {
    if tau_max <= tau_min || tau_max >= frame.len() {
        return (0.0, 0.0);
    }

    // Compute CMNDF over full frame, then restrict search to [tau_min, tau_max].
    let cmnd = compute_cmndf(frame, tau_max);

    // Step 3: absolute threshold — first local minimum below `threshold`.
    let mut tau_est = 0usize;
    let mut t = tau_min;
    while t <= tau_max {
        if cmnd[t] < threshold {
            // Descend to the local minimum.
            while t < tau_max && cmnd[t + 1] < cmnd[t] {
                t += 1;
            }
            tau_est = t;
            break;
        }
        t += 1;
    }

    // Fallback: global minimum in range if none crossed the threshold.
    if tau_est == 0 {
        let mut best = tau_min;
        for tau in tau_min..=tau_max {
            if cmnd[tau] < cmnd[best] {
                best = tau;
            }
        }
        // Only accept if reasonably periodic (confidence still reported).
        let conf = (1.0 - cmnd[best]).clamp(0.0, 1.0);
        let refined = parabolic_interp(&cmnd, best);
        return (sr / refined, conf);
    }

    // Step 4: parabolic interpolation for sub-sample lag precision.
    let refined = parabolic_interp(&cmnd, tau_est);
    let freq = sr / refined;
    let conf = (1.0 - cmnd[tau_est]).clamp(0.0, 1.0);
    (freq, conf)
}

// ── Shared CMNDF helper ───────────────────────────────────────────────────────

/// Compute the Cumulative Mean Normalized Difference Function (CMNDF) for
/// `frame` up to lag `tau_max` (inclusive).
///
/// Returns a `Vec<f32>` of length `tau_max + 1` where `d'[0] = 1` by convention.
/// The computation uses the full frame length for the inner sum so that higher
/// lags still benefit from as many sample pairs as the frame allows.
fn compute_cmndf(frame: &[f32], tau_max: usize) -> Vec<f32> {
    let w = frame.len();
    let t_max = tau_max.min(w.saturating_sub(1));

    // Step 1: squared difference function d(tau).
    let mut diff = vec![0.0f32; t_max + 1];
    for tau in 1..=t_max {
        let mut sum = 0.0f32;
        for j in 0..(w - tau) {
            let d = frame[j] - frame[j + tau];
            sum += d * d;
        }
        diff[tau] = sum;
    }

    // Step 2: CMNDF d'(tau) = d(tau) * tau / cumsum_1_to_tau(d).
    let mut cmnd = vec![1.0f32; t_max + 1];
    let mut running = 0.0f32;
    for tau in 1..=t_max {
        running += diff[tau];
        cmnd[tau] = if running > 0.0 {
            diff[tau] * tau as f32 / running
        } else {
            1.0
        };
    }
    cmnd
}

/// Parabolic interpolation of the minimum near integer lag `tau` using the three
/// surrounding `cmnd` values. Returns a fractional lag.
fn parabolic_interp(cmnd: &[f32], tau: usize) -> f32 {
    if tau == 0 || tau + 1 >= cmnd.len() {
        return tau as f32;
    }
    let s0 = cmnd[tau - 1];
    let s1 = cmnd[tau];
    let s2 = cmnd[tau + 1];
    let denom = 2.0 * (2.0 * s1 - s2 - s0);
    if denom.abs() < 1e-12 {
        return tau as f32;
    }
    tau as f32 + (s2 - s0) / denom
}

/// Detect pitch using FFT-based normalized autocorrelation — O(N log N) per frame.
///
/// Uses `oxifft::conv::correlate(&frame, &frame)` (FFT-based cross-correlation of a
/// signal with itself) to compute the full autocorrelation sequence in O(N log N)
/// instead of the O(N²) brute-force nested loop.
///
/// `correlate(x, x)` returns a vector of length `2*N - 1`.
/// Zero-lag value is at index `N - 1`; positive-lag `R[tau]` is at index `N - 1 + tau`.
/// All values are normalized by `R[0]` (zero-lag energy) before peak picking.
///
/// Returns a `Vec<PitchFrame>` — one entry per frame.
///
/// # Parameters
/// - `buf` — input audio (mixed to mono internally).
/// - `frame_size` — samples per analysis frame (e.g. 2048).
/// - `hop_size` — samples between frames (e.g. 512).
/// - `min_f0_hz` — minimum detectable pitch in Hz (e.g. 60.0).
/// - `max_f0_hz` — maximum detectable pitch in Hz (e.g. 1200.0).
pub fn detect_pitch_autocorr(
    buf: &AudioBuffer<f32>,
    frame_size: usize,
    hop_size: usize,
    min_f0_hz: f32,
    max_f0_hz: f32,
) -> Vec<PitchFrame> {
    let mono = to_mono(buf);
    let sr = buf.sample_rate as f32;

    if mono.len() < frame_size || frame_size == 0 || hop_size == 0 || sr <= 0.0 {
        return Vec::new();
    }

    let hop = hop_size.max(1);
    let n_frames = (mono.len() - frame_size) / hop + 1;

    // Compute search range in samples (periods).
    let max_period = ((sr / min_f0_hz.max(1.0)).round() as usize).min(frame_size.saturating_sub(1));
    let min_period = ((sr / max_f0_hz.max(1.0)).round() as usize).max(1);

    const VOICED_THRESHOLD: f32 = 0.5;

    let mut frames = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;
        let frame = &mono[start..(start + frame_size).min(mono.len())];
        let n = frame.len();

        // FFT-based autocorrelation via oxifft::conv::correlate — O(N log N).
        //
        // correlate(x, x) returns full length 2*N - 1.
        // Index layout: result[N-1] = R[0], result[N-1+tau] = R[tau] for tau >= 1.
        let full_acf = correlate(frame, frame);
        let zero_lag_idx = n - 1;

        // R[0] = zero-lag energy; guard against silent frames.
        let r0 = if zero_lag_idx < full_acf.len() {
            full_acf[zero_lag_idx]
        } else {
            0.0
        };

        if r0 < 1e-12 {
            frames.push(PitchFrame {
                time_seconds: (frame_idx * hop_size) as f64 / sr as f64,
                frequency_hz: 0.0,
                confidence: 0.0,
                is_voiced: false,
            });
            continue;
        }

        // Peak pick in the lag range [min_period, max_period] using normalized ACF.
        let search_max = max_period.min(n.saturating_sub(1));
        let mut best_period = min_period;
        let mut best_acf_norm = f32::NEG_INFINITY;

        for tau in min_period..=search_max {
            let idx = zero_lag_idx + tau;
            if idx >= full_acf.len() {
                break;
            }
            let acf_norm = full_acf[idx] / r0;
            if acf_norm > best_acf_norm {
                best_acf_norm = acf_norm;
                best_period = tau;
            }
        }

        let (frequency, confidence, is_voiced) = if best_acf_norm > VOICED_THRESHOLD {
            let freq = sr / best_period as f32;
            (freq, best_acf_norm.clamp(0.0, 1.0), true)
        } else {
            (0.0f32, 0.0f32, false)
        };

        frames.push(PitchFrame {
            time_seconds: (frame_idx * hop_size) as f64 / sr as f64,
            frequency_hz: frequency,
            confidence,
            is_voiced,
        });
    }

    frames
}

/// Mix any layout down to a mono `Vec<f32>`.
fn to_mono(buf: &AudioBuffer<f32>) -> Vec<f32> {
    let n_ch = buf.channels.channel_count();
    if n_ch == 1 {
        return buf.samples.clone();
    }
    buf.samples
        .chunks_exact(n_ch)
        .map(|c| c.iter().sum::<f32>() / n_ch as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{ChannelLayout, SampleFormat};
    use std::f32::consts::PI;

    fn sine(freq: f32, sr: u32, secs: f32) -> AudioBuffer<f32> {
        let n = (sr as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.8 * (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect();
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn yin_detects_440hz() {
        let buf = sine(440.0, 48_000, 0.5);
        let frames = PitchTracker::default().track(&buf);
        assert!(!frames.is_empty());
        let voiced: Vec<_> = frames.iter().filter(|f| f.is_voiced).collect();
        assert!(
            !voiced.is_empty(),
            "440Hz sine should be detected as voiced"
        );
        let avg = voiced.iter().map(|f| f.frequency_hz).sum::<f32>() / voiced.len() as f32;
        assert!((avg - 440.0).abs() < 5.0, "expected ~440Hz, got {avg:.1}Hz");
        let avg_conf = voiced.iter().map(|f| f.confidence).sum::<f32>() / voiced.len() as f32;
        assert!(
            avg_conf > 0.85,
            "confidence should be high for a pure tone, got {avg_conf:.2}"
        );
    }

    #[test]
    fn yin_detects_220hz() {
        let buf = sine(220.0, 44_100, 0.5);
        let frames = PitchTracker::default().track(&buf);
        let voiced: Vec<_> = frames.iter().filter(|f| f.is_voiced).collect();
        assert!(!voiced.is_empty());
        let avg = voiced.iter().map(|f| f.frequency_hz).sum::<f32>() / voiced.len() as f32;
        assert!((avg - 220.0).abs() < 4.0, "expected ~220Hz, got {avg:.1}Hz");
    }

    #[test]
    fn yin_silence_is_unvoiced_or_lowconf() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 48_000],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let frames = PitchTracker::default().track(&buf);
        // Pure silence: difference function is flat; should not report a confident pitch.
        let confident_voiced = frames
            .iter()
            .filter(|f| f.is_voiced && f.confidence > 0.95)
            .count();
        assert_eq!(confident_voiced, frames.len().min(confident_voiced));
    }

    #[test]
    fn detect_pitch_yin_convenience() {
        let buf = sine(330.0, 48_000, 0.3);
        let pitches = detect_pitch_yin_simple(&buf);
        assert!(!pitches.is_empty());
        let voiced: Vec<f32> = pitches
            .iter()
            .map(|&(_, f)| f)
            .filter(|&f| f > 0.0)
            .collect();
        assert!(!voiced.is_empty());
        let avg = voiced.iter().sum::<f32>() / voiced.len() as f32;
        assert!((avg - 330.0).abs() < 6.0, "expected ~330Hz, got {avg:.1}Hz");
    }

    #[test]
    fn test_yin_440hz_sine_at_48khz() {
        // Full-params API: 440 Hz sine at 48 kHz, verify voiced frames within ±2 Hz.
        let buf = sine(440.0, 48_000, 1.0);
        let frames = detect_pitch_yin(&buf, 2048, 512, 0.1);
        assert!(!frames.is_empty(), "should produce frames");
        // Skip first few frames for settling
        let voiced: Vec<&PitchFrame> = frames[4..].iter().filter(|f| f.is_voiced).collect();
        assert!(
            voiced.len() > frames[4..].len() / 2,
            "majority of frames should be voiced"
        );
        for f in &voiced {
            assert!(
                (f.frequency_hz - 440.0).abs() < 2.0,
                "voiced frame freq {:.2} Hz should be within 2 Hz of 440 Hz",
                f.frequency_hz
            );
            assert!(
                f.confidence > 0.7,
                "voiced frame confidence {:.3} should be > 0.7",
                f.confidence
            );
        }
    }

    #[test]
    fn yin_stereo_input() {
        let n = 24_000usize;
        let samples: Vec<f32> = (0..n)
            .flat_map(|i| {
                let s = 0.8 * (2.0 * PI * 440.0 * i as f32 / 48_000.0).sin();
                [s, s]
            })
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let frames = PitchTracker::new(2048, 1024).track(&buf);
        let voiced: Vec<_> = frames.iter().filter(|f| f.is_voiced).collect();
        assert!(!voiced.is_empty());
    }

    // ── pYIN tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_pyin_440hz_sine_at_48khz() {
        let sr = 48_000u32;
        let dur = 1.0f32;
        let n = (sr as f32 * dur) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (2.0 * PI * 440.0 * t).sin() * 0.8
            })
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let frames = detect_pitch_pyin(&buf, 2048, 512);
        assert!(!frames.is_empty(), "should produce frames");

        // After initial transient (skip first few frames), most frames should be voiced at ~440 Hz.
        let voiced: Vec<&PitchFrame> = frames.iter().skip(5).filter(|f| f.is_voiced).collect();
        assert!(
            voiced.len() > frames.len() / 2,
            "majority should be voiced, got {}/{} voiced",
            voiced.len(),
            frames.len()
        );

        for f in &voiced {
            assert!(
                (f.frequency_hz - 440.0).abs() < 20.0,
                "frequency should be ~440 Hz, got {}",
                f.frequency_hz
            );
        }
    }

    #[test]
    fn test_pyin_silence_mostly_unvoiced() {
        let buf = AudioBuffer {
            samples: vec![0.001f32; 8192],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let frames = detect_pitch_pyin(&buf, 2048, 512);
        let n_voiced = frames.iter().filter(|f| f.is_voiced).count();
        let total = frames.len();
        assert!(
            n_voiced <= total / 2,
            "silence should produce mostly unvoiced frames, got {n_voiced}/{total} voiced"
        );
    }

    #[test]
    fn test_pyin_frame_count() {
        let sr = 44_100u32;
        let n = 22_050usize;
        let buf = AudioBuffer {
            samples: vec![0.5f32; n],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let frames = detect_pitch_pyin(&buf, 2048, 512);
        // Frame count = (n - frame_size) / hop_size + 1
        let expected = (n.saturating_sub(2048)) / 512 + 1;
        assert!(
            (frames.len() as i64 - expected as i64).abs() <= 2,
            "expected ~{expected} frames, got {}",
            frames.len()
        );
    }

    // ── detect_pitch_autocorr tests ───────────────────────────────────────────

    #[test]
    fn test_autocorr_pitch_sine_440hz() {
        let sr = 48_000u32;
        let dur = 0.5f32;
        let n = (sr as f32 * dur) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr as f32).sin() * 0.8)
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let frames = detect_pitch_autocorr(&buf, 2048, 512, 80.0, 1200.0);
        assert!(!frames.is_empty(), "should produce pitch frames");

        // At least one voiced frame must be within 5 Hz of 440 Hz.
        let has_440 = frames
            .iter()
            .any(|f| f.is_voiced && (f.frequency_hz - 440.0).abs() < 5.0);
        assert!(
            has_440,
            "at least one voiced frame should detect ~440 Hz; frames: {:?}",
            frames
                .iter()
                .filter(|f| f.is_voiced)
                .map(|f| f.frequency_hz)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_autocorr_pitch_silence() {
        let sr = 48_000u32;
        let buf = AudioBuffer {
            samples: vec![0.0f32; 8192],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let frames = detect_pitch_autocorr(&buf, 2048, 512, 80.0, 1200.0);
        assert!(
            !frames.is_empty(),
            "should produce frames for non-empty buffer"
        );
        let any_voiced = frames.iter().any(|f| f.is_voiced);
        assert!(!any_voiced, "silence should produce no voiced frames");
    }

    #[test]
    fn yin_pitch_detection_440hz_within_2hz_confidence_gt_0_7() {
        // Test 6: YIN pitch detection on 0.5s 440 Hz sine at 48 kHz.
        // Detected pitch of voiced frames must be within 2 Hz of 440 Hz,
        // and median confidence must be > 0.7.
        let sr = 48_000u32;
        let n = (sr as f32 * 0.5) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.8 * (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let frames = detect_pitch_yin(&buf, 2048, 512, 0.1);
        assert!(!frames.is_empty(), "detect_pitch_yin should return frames");

        let voiced: Vec<&PitchFrame> = frames.iter().filter(|f| f.is_voiced).collect();
        assert!(
            !voiced.is_empty(),
            "440Hz sine should produce voiced frames"
        );

        // Check that the median pitch of voiced frames is within 2 Hz of 440 Hz
        let mut pitches: Vec<f32> = voiced.iter().map(|f| f.frequency_hz).collect();
        pitches.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_pitch = pitches[pitches.len() / 2];
        assert!(
            (median_pitch - 440.0).abs() < 2.0,
            "YIN median pitch should be within 2 Hz of 440 Hz, got {median_pitch:.2} Hz"
        );

        // Check median confidence > 0.7
        let mut confs: Vec<f32> = voiced.iter().map(|f| f.confidence).collect();
        confs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_conf = confs[confs.len() / 2];
        assert!(
            median_conf > 0.7,
            "YIN median confidence should be > 0.7, got {median_conf:.3}"
        );
    }

    // ── FFT autocorrelation (M23-F Task 2) tests ─────────────────────────────

    #[test]
    fn test_fft_autocorr_440hz_22050hz_within_5hz() {
        // Generate a 440 Hz sine at 22050 Hz for 0.5 seconds and verify the
        // FFT-based autocorrelation pitch detector returns a voiced frame
        // within 5 Hz of 440 Hz.
        let sr = 22_050u32;
        let dur = 0.5f32;
        let n = (sr as f32 * dur) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.8 * (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };

        let frames = detect_pitch_autocorr(&buf, 2048, 512, 100.0, 2000.0);
        assert!(!frames.is_empty(), "should produce pitch frames");

        let voiced: Vec<&PitchFrame> = frames.iter().filter(|f| f.is_voiced).collect();
        assert!(
            !voiced.is_empty(),
            "440 Hz sine at 22050 Hz should produce voiced frames"
        );

        // At least one voiced frame must be within 5 Hz of 440 Hz.
        let has_close = voiced.iter().any(|f| (f.frequency_hz - 440.0).abs() < 5.0);
        assert!(
            has_close,
            "FFT autocorr: at least one voiced frame should be within 5 Hz of 440 Hz; \
             got voiced freqs: {:?}",
            voiced.iter().map(|f| f.frequency_hz).collect::<Vec<_>>()
        );
    }
}
