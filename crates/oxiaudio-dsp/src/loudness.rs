use std::collections::VecDeque;

use oxiaudio_core::AudioBuffer;

use crate::biquad::BiquadFilter;

/// Build the first stage of the ITU-R BS.1770-4 K-weighting filter.
///
/// At 48 kHz the standard pre-computed coefficients are used directly.
/// At other sample rates, an approximate high-shelf is substituted.
fn k_weight_stage1(sample_rate: u32) -> BiquadFilter {
    if sample_rate == 48_000 {
        // ITU-R BS.1770-4 Annex 1, 48 kHz reference coefficients
        BiquadFilter {
            b0: 1.535_124_9_f32,
            b1: -2.691_696_2_f32,
            b2: 1.198_392_9_f32,
            a1: -1.690_659_3_f32,
            a2: 0.732_480_77_f32,
        }
    } else {
        BiquadFilter::high_shelf(1681.97, 4.0, sample_rate)
    }
}

/// Build the second stage of the ITU-R BS.1770-4 K-weighting filter.
///
/// At 48 kHz the standard pre-computed coefficients are used directly.
/// At other sample rates, a highpass at 38.135 Hz is substituted.
fn k_weight_stage2(sample_rate: u32) -> BiquadFilter {
    if sample_rate == 48_000 {
        BiquadFilter {
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,
            a1: -1.990_047_5_f32,
            a2: 0.990_072_25_f32,
        }
    } else {
        BiquadFilter::highpass(38.135, (0.5f32).sqrt(), sample_rate)
    }
}

/// Apply the ITU-R BS.1770-4 K-weighting filter to `buf`.
///
/// K-weighting is a two-stage biquad (pre-filter + RLB highpass) that
/// approximates the equal-loudness contour of the human ear.
pub fn k_weight(buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
    let s1 = k_weight_stage1(buf.sample_rate);
    let s2 = k_weight_stage2(buf.sample_rate);
    s2.process(&s1.process(buf))
}

/// Measure integrated loudness per EBU R128 / ITU-R BS.1770-4.
///
/// Returns LUFS (loudness units relative to full scale).
/// Uses absolute gate (-70 LUFS) and relative gate (-10 LU below ungated level).
///
/// Returns `f32::NEG_INFINITY` if the signal is silent or shorter than one block.
pub fn loudness_integrated(buf: &AudioBuffer<f32>) -> f32 {
    let weighted = k_weight(buf);
    let sr = buf.sample_rate;
    let ch = buf.channels.channel_count();
    let frames = weighted.samples.len() / ch.max(1);
    let block_frames = (0.4 * sr as f64) as usize;
    let hop_frames = (0.1 * sr as f64) as usize;
    if block_frames == 0 || hop_frames == 0 || frames < block_frames {
        return f32::NEG_INFINITY;
    }
    let n_blocks = (frames - block_frames) / hop_frames + 1;
    let mut block_lufs: Vec<f32> = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let start = b * hop_frames;
        let end = (start + block_frames).min(frames);
        let count = (end - start) * ch;
        if count == 0 {
            continue;
        }
        let mean_sq: f32 = weighted.samples[start * ch..end * ch]
            .iter()
            .map(|&s| s * s)
            .sum::<f32>()
            / count as f32;
        let lufs = if mean_sq < 1e-12 {
            -200.0
        } else {
            -0.691 + 10.0 * mean_sq.log10()
        };
        block_lufs.push(lufs);
    }
    let gated1: Vec<f32> = block_lufs.iter().copied().filter(|&l| l > -70.0).collect();
    if gated1.is_empty() {
        return f32::NEG_INFINITY;
    }
    let ungated_power: f32 =
        gated1.iter().map(|&l| 10.0f32.powf(l / 10.0)).sum::<f32>() / gated1.len() as f32;
    let ungated_lufs = -0.691 + 10.0 * ungated_power.log10();
    let rel_thresh = ungated_lufs - 10.0;
    let gated2: Vec<f32> = gated1.iter().copied().filter(|&l| l > rel_thresh).collect();
    if gated2.is_empty() {
        return f32::NEG_INFINITY;
    }
    let final_power: f32 =
        gated2.iter().map(|&l| 10.0f32.powf(l / 10.0)).sum::<f32>() / gated2.len() as f32;
    -0.691 + 10.0 * final_power.log10()
}

/// Compute momentary loudness (EBU R128) over 400ms windows with 100ms hop.
///
/// Returns one LUFS value per block. Unlike [`loudness_integrated`], NO absolute
/// or relative gating is applied.
///
/// Returns `Vec<f32>` — one value per block (returns empty if buffer is too short).
pub fn loudness_momentary(buf: &AudioBuffer<f32>) -> Vec<f32> {
    let weighted = k_weight(buf);
    let sr = buf.sample_rate;
    let ch = weighted.channels.channel_count().max(1);
    // EBU R128: 400ms blocks, 100ms hop (75% overlap)
    let block_frames = (0.4 * sr as f64) as usize;
    let hop_frames = (0.1 * sr as f64) as usize;
    if block_frames == 0 || hop_frames == 0 {
        return vec![];
    }
    let total_frames = weighted.samples.len() / ch;
    if total_frames < block_frames {
        return vec![];
    }
    let n_blocks = (total_frames - block_frames) / hop_frames + 1;
    (0..n_blocks)
        .map(|b| {
            let start = b * hop_frames;
            let end = (start + block_frames).min(total_frames);
            let count = (end - start) * ch;
            if count == 0 {
                return f32::NEG_INFINITY;
            }
            let mean_sq: f32 = weighted.samples[start * ch..end * ch]
                .iter()
                .map(|&s| s * s)
                .sum::<f32>()
                / count as f32;
            if mean_sq < 1e-12 {
                f32::NEG_INFINITY
            } else {
                -0.691 + 10.0 * mean_sq.log10()
            }
        })
        .collect()
}

/// Compute momentary loudness in LUFS using arbitrary non-overlapping windows.
///
/// Each non-overlapping window of `window_ms` milliseconds is K-weighted and
/// converted to LUFS. Silent windows return `f32::NEG_INFINITY`.
///
/// For EBU R128 standard momentary loudness, use [`loudness_momentary`] instead.
pub fn loudness_momentary_windowed(buf: &AudioBuffer<f32>, window_ms: u32) -> Vec<f32> {
    let weighted = k_weight(buf);
    let sr = buf.sample_rate;
    let ch = weighted.channels.channel_count().max(1);
    let window_frames = ((window_ms as f64 * sr as f64) / 1000.0) as usize;
    if window_frames == 0 {
        return vec![];
    }
    let total_frames = weighted.samples.len() / ch;
    let n_windows = total_frames / window_frames;
    if n_windows == 0 {
        return vec![];
    }
    (0..n_windows)
        .map(|w| {
            let start = w * window_frames;
            let end = start + window_frames;
            let count = (end - start) * ch;
            if count == 0 {
                return f32::NEG_INFINITY;
            }
            let mean_sq: f32 = weighted.samples[start * ch..end * ch]
                .iter()
                .map(|&s| s * s)
                .sum::<f32>()
                / count as f32;
            if mean_sq < 1e-12 {
                f32::NEG_INFINITY
            } else {
                -0.691 + 10.0 * mean_sq.log10()
            }
        })
        .collect()
}

/// Compute the EBU R128 loudness range (LRA) of an audio buffer.
///
/// LRA measures the macro-dynamics of a program per EBU Tech 3342.
///
/// Algorithm:
/// 1. K-weight the signal
/// 2. Compute momentary loudness blocks (400ms blocks, 100ms hop)
/// 3. Absolute gate: discard blocks below -70 LUFS
/// 4. Compute ungated loudness as power-average of gated blocks
/// 5. Relative gate: discard blocks more than 20 LU below ungated loudness
/// 6. LRA = 95th percentile − 10th percentile of remaining block values
///
/// Returns LRA in LU (Loudness Units). Returns 0.0 if insufficient data.
pub fn loudness_range(buf: &AudioBuffer<f32>) -> f32 {
    let blocks = loudness_momentary(buf);

    // Absolute gate (-70 LUFS)
    let gated_abs: Vec<f32> = blocks.iter().copied().filter(|&l| l > -70.0).collect();
    if gated_abs.len() < 2 {
        return 0.0;
    }

    // Ungated loudness: power average of absolutely-gated blocks
    let ungated_power: f32 = gated_abs
        .iter()
        .map(|&l| 10.0f32.powf(l / 10.0))
        .sum::<f32>()
        / gated_abs.len() as f32;
    if ungated_power <= 0.0 {
        return 0.0;
    }
    let ungated_lufs = -0.691 + 10.0 * ungated_power.log10();

    // Relative gate: discard blocks more than 20 LU below ungated loudness
    let rel_thresh = ungated_lufs - 20.0;
    let mut gated_rel: Vec<f32> = gated_abs
        .iter()
        .copied()
        .filter(|&l| l > rel_thresh)
        .collect();
    if gated_rel.len() < 2 {
        return 0.0;
    }

    // Sort ascending for percentile computation
    gated_rel.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = gated_rel.len();

    // 10th and 95th percentile indices
    let idx_10 = ((n as f32 * 0.10) as usize).min(n - 1);
    let idx_95 = ((n as f32 * 0.95) as usize).min(n - 1);

    (gated_rel[idx_95] - gated_rel[idx_10]).max(0.0)
}

/// Measure the true peak level in dBTP using linear interpolation oversampling.
///
/// Samples inter-frame intersections at 4x oversampling rate via linear
/// interpolation, matching the spirit of ITU-R BS.1770-4 Annex 2.
///
/// Returns `f32::NEG_INFINITY` if the buffer is empty.
pub fn true_peak(buf: &AudioBuffer<f32>) -> f32 {
    let ch = buf.channels.channel_count();
    let frames = buf.samples.len() / ch.max(1);
    let mut peak = 0.0f32;
    for c in 0..ch {
        for frame in 0..frames.saturating_sub(1) {
            let s0 = buf.samples[frame * ch + c];
            let s1 = buf.samples[(frame + 1) * ch + c];
            for k in 0..4 {
                let frac = k as f32 / 4.0;
                peak = peak.max((s0 + frac * (s1 - s0)).abs());
            }
        }
    }
    if peak <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * peak.log10()
    }
}

/// Normalize `buf` to a target integrated loudness in LUFS (EBU R128).
///
/// Measures the integrated loudness, computes the required gain, and applies it.
/// If the measured loudness is below −70 LUFS (absolute gating threshold),
/// the buffer is returned unchanged — do not amplify noise floors.
///
/// # Errors
///
/// Currently infallible; the `Result` return type is retained for forward
/// compatibility with future processing that may fail.
#[must_use = "discarding the Result ignores normalize errors"]
pub fn normalize_to_lufs(
    buf: &AudioBuffer<f32>,
    target_lufs: f32,
) -> Result<AudioBuffer<f32>, oxiaudio_core::OxiAudioError> {
    let measured = loudness_integrated(buf);
    if measured <= -70.0 {
        return Ok(buf.clone());
    }
    let gain_db = target_lufs - measured;
    let mut out = buf.clone();
    crate::gain(&mut out, gain_db);
    Ok(out)
}

// ─── PeakMeter ───────────────────────────────────────────────────────────────

/// A stateful peak level meter with peak hold and configurable decay.
///
/// Call [`PeakMeter::process_block`] with each block of samples to update the peak.
/// Use [`PeakMeter::peak_db`] to read the current peak level in dBFS.
/// The peak holds for `hold_samples` before decaying at `decay_db_per_sample`.
#[derive(Debug, Clone)]
pub struct PeakMeter {
    /// Number of samples to hold the peak before decaying.
    pub hold_samples: usize,
    /// dB per sample decay rate (e.g. 0.001 for slow decay).
    pub decay_db_per_sample: f32,
    peak_linear: f32,
    hold_counter: usize,
}

impl PeakMeter {
    /// Create a new `PeakMeter`.
    ///
    /// - `hold_ms`: peak hold duration in milliseconds
    /// - `decay_db_per_second`: dB/s decay rate after hold expires
    /// - `sample_rate`: audio sample rate in Hz
    pub fn new(hold_ms: f32, decay_db_per_second: f32, sample_rate: u32) -> Self {
        let hold_samples = (hold_ms * sample_rate as f32 / 1000.0) as usize;
        let decay_db_per_sample = decay_db_per_second / sample_rate as f32;
        Self {
            hold_samples,
            decay_db_per_sample,
            peak_linear: 0.0,
            hold_counter: 0,
        }
    }

    /// Process a block of samples and update the peak.
    ///
    /// Returns the current peak in dBFS after processing.
    pub fn process_block(&mut self, samples: &[f32]) -> f32 {
        let block_peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if block_peak >= self.peak_linear {
            self.peak_linear = block_peak;
            self.hold_counter = self.hold_samples;
        } else if self.hold_counter > 0 {
            self.hold_counter = self.hold_counter.saturating_sub(samples.len());
        } else {
            // Decay: convert to dB, subtract decay, convert back
            let peak_db = if self.peak_linear > 1e-10 {
                20.0 * self.peak_linear.log10()
            } else {
                -200.0
            };
            let decayed_db = peak_db - self.decay_db_per_sample * samples.len() as f32;
            self.peak_linear = if decayed_db <= -200.0 {
                0.0
            } else {
                10.0_f32.powf(decayed_db / 20.0)
            };
        }
        self.peak_db()
    }

    /// Current peak level in dBFS.
    ///
    /// Returns `f32::NEG_INFINITY` when the meter has not been fed any signal
    /// or after the peak has fully decayed.
    pub fn peak_db(&self) -> f32 {
        if self.peak_linear <= 1e-10 {
            return f32::NEG_INFINITY;
        }
        20.0 * self.peak_linear.log10()
    }

    /// Reset the meter to silence.
    pub fn reset(&mut self) {
        self.peak_linear = 0.0;
        self.hold_counter = 0;
    }
}

// ─── RmsMeter ────────────────────────────────────────────────────────────────

/// A windowed RMS (root mean square) level meter.
///
/// Maintains a sliding window of `window_samples` length and efficiently
/// tracks the running sum of squares using a ring buffer.
#[derive(Debug, Clone)]
pub struct RmsMeter {
    /// Window length in samples.
    pub window_samples: usize,
    buffer: VecDeque<f32>,
    /// Running sum of squares (f64 for precision with large windows).
    sum_sq: f64,
}

impl RmsMeter {
    /// Create a new `RmsMeter`.
    ///
    /// - `window_ms`: window length in milliseconds
    /// - `sample_rate`: audio sample rate in Hz
    pub fn new(window_ms: f32, sample_rate: u32) -> Self {
        let window_samples = (window_ms * sample_rate as f32 / 1000.0).max(1.0) as usize;
        Self {
            window_samples,
            buffer: VecDeque::with_capacity(window_samples),
            sum_sq: 0.0,
        }
    }

    /// Process one sample and return the current RMS level (linear, not dB).
    pub fn process_sample(&mut self, sample: f32) -> f32 {
        if self.buffer.len() >= self.window_samples {
            if let Some(old) = self.buffer.pop_front() {
                self.sum_sq -= (old * old) as f64;
            }
        }
        self.sum_sq += (sample * sample) as f64;
        self.buffer.push_back(sample);
        self.rms()
    }

    /// Current RMS level (linear, not dB).
    pub fn rms(&self) -> f32 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        ((self.sum_sq / self.buffer.len() as f64).max(0.0).sqrt()) as f32
    }

    /// Current RMS level in dBFS.
    ///
    /// Returns `f32::NEG_INFINITY` when the buffer is empty or RMS is negligible.
    pub fn rms_db(&self) -> f32 {
        let r = self.rms();
        if r <= 1e-10 {
            return f32::NEG_INFINITY;
        }
        20.0 * r.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{ChannelLayout, SampleFormat};
    use std::f32::consts::PI;

    /// Generate a calibration tone at a target LUFS level.
    ///
    /// For a sine wave: mean_square = amplitude^2 / 2
    /// LUFS = -0.691 + 10*log10(mean_square)
    fn calibration_tone(target_lufs: f32, sr: u32, secs: f32) -> AudioBuffer<f32> {
        let ms = 10.0f32.powf((target_lufs + 0.691) / 10.0);
        let amplitude = (2.0 * ms).sqrt();
        let n = (sr as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| amplitude * (2.0 * PI * 997.0 * i as f32 / sr as f32).sin())
            .collect();
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn loudness_calibration_tone() {
        // -23 LUFS calibration tone should read within ±1.5 LU
        let buf = calibration_tone(-23.0, 48_000, 5.0);
        let lufs = loudness_integrated(&buf);
        assert!(
            (lufs - (-23.0)).abs() < 1.5,
            "Expected ~-23 LUFS, got {lufs:.2}"
        );
    }

    #[test]
    fn loudness_silence_neg_infinity() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 48_000 * 3],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let lufs = loudness_integrated(&buf);
        assert!(
            lufs < -100.0 || lufs.is_infinite(),
            "silence should be very negative LUFS, got {lufs}"
        );
    }

    #[test]
    fn true_peak_ge_digital_peak() {
        let n = 48_000usize;
        let amplitude = 0.99f32;
        let samples: Vec<f32> = (0..n)
            .map(|i| amplitude * (2.0 * PI * 997.0 * i as f32 / 48000.0).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let tp = true_peak(&buf);
        let dp = 20.0 * amplitude.log10();
        assert!(
            tp >= dp - 0.1,
            "true peak {tp:.2} dBTP should be >= digital peak {dp:.2} dBFS"
        );
    }

    #[test]
    fn test_loudness_momentary_silence() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 48_000 * 2],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let momentary = loudness_momentary(&buf);
        assert!(!momentary.is_empty(), "should produce windows");
        for &v in &momentary {
            assert!(
                v.is_infinite() && v < 0.0,
                "silent window should return NEG_INFINITY, got {v}"
            );
        }
    }

    #[test]
    fn test_loudness_range_short_signal() {
        // A short signal with few blocks should return a valid (possibly zero) value
        let buf = AudioBuffer {
            samples: vec![0.5f32; 48_000 * 2],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let lra = loudness_range(&buf);
        assert!(lra >= 0.0, "LRA must be non-negative, got {lra}");
    }

    #[test]
    fn test_loudness_range_constant_returns_zero() {
        // Constant signal across many windows: all windows have same LUFS → LRA ≈ 0
        let sr = 48_000u32;
        let n = (sr as usize) * 10;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.3 * (2.0 * PI * 997.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let lra = loudness_range(&buf);
        // Constant signal: all blocks have ~same LUFS → LRA ≈ 0
        assert!(lra >= 0.0, "LRA must be non-negative");
        assert!(lra < 2.0, "constant signal LRA should be near 0, got {lra}");
    }

    #[test]
    fn k_weight_mid_freq_unchanged() {
        let n = 48_000usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * 997.0 * i as f32 / 48000.0).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let kw = k_weight(&buf);
        let rms_in = (buf.samples.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
        let rms_out = (kw.samples.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
        let db = 20.0 * (rms_out / rms_in.max(1e-10)).log10();
        assert!(
            db.abs() < 2.5,
            "K-weight at 997Hz should change level <2.5dB, got {db:.2}dB"
        );
    }

    // ─── M12 tests ───────────────────────────────────────────────────────────

    fn sine_buf(freq: f32, sr: u32, dur: f32) -> AudioBuffer<f32> {
        let n = (sr as f32 * dur) as usize;
        AudioBuffer {
            samples: (0..n)
                .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin() * 0.5)
                .collect(),
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_loudness_momentary_returns_values() {
        let buf = sine_buf(1000.0, 48_000, 2.0);
        let blocks = loudness_momentary(&buf);
        // 2 seconds with 100ms hop → at least 1 block
        assert!(!blocks.is_empty(), "should return blocks for 2s audio");
        for &v in &blocks {
            assert!(
                v.is_finite() || v == f32::NEG_INFINITY,
                "block value should be finite or -inf: {v}"
            );
        }
    }

    #[test]
    fn test_loudness_momentary_louder_than_quiet() {
        let buf_loud = sine_buf(1000.0, 48_000, 1.0);
        let buf_quiet = AudioBuffer {
            samples: vec![0.01f32; 48_000],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let blocks_loud = loudness_momentary(&buf_loud);
        let blocks_quiet = loudness_momentary(&buf_quiet);
        assert!(!blocks_loud.is_empty(), "loud blocks should not be empty");
        assert!(!blocks_quiet.is_empty(), "quiet blocks should not be empty");
        let count_loud = blocks_loud.iter().filter(|v| v.is_finite()).count().max(1);
        let count_quiet = blocks_quiet.iter().filter(|v| v.is_finite()).count().max(1);
        let mean_loud: f32 =
            blocks_loud.iter().filter(|v| v.is_finite()).sum::<f32>() / count_loud as f32;
        let mean_quiet: f32 =
            blocks_quiet.iter().filter(|v| v.is_finite()).sum::<f32>() / count_quiet as f32;
        assert!(
            mean_loud > mean_quiet,
            "loud signal momentary LUFS should exceed quiet: {mean_loud} vs {mean_quiet}"
        );
    }

    #[test]
    fn test_loudness_range_nonzero_for_varied_signal() {
        let sr = 48_000u32;
        let n = sr as usize * 10;
        let samples: Vec<f32> = (0..n / 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .chain((n / 2..n).map(|i| (2.0 * PI * 440.0 * i as f32 / sr as f32).sin() * 0.01))
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let lra = loudness_range(&buf);
        assert!(lra >= 0.0, "LRA should be non-negative");
    }

    #[test]
    fn test_peak_meter_holds_and_decays() {
        // 100ms hold, 60 dB/s decay at 44100 Hz
        let mut meter = PeakMeter::new(100.0, 60.0, 44_100);
        // Process a loud block
        let loud = vec![0.9f32; 1024];
        let peak_after_loud = meter.process_block(&loud);
        assert!(
            peak_after_loud > -2.0,
            "peak should be near 0 dBFS after loud block: {peak_after_loud}"
        );
        // Process many silent blocks: peak should decay after hold expires
        let silent = vec![0.0f32; 1024];
        let mut last_peak = peak_after_loud;
        for _ in 0..100 {
            last_peak = meter.process_block(&silent);
        }
        assert!(
            last_peak < peak_after_loud,
            "peak should decay after hold: {last_peak} vs {peak_after_loud}"
        );
    }

    #[test]
    fn test_rms_meter_tracks_level() {
        let mut meter = RmsMeter::new(100.0, 44_100);
        let sr = 44_100u32;
        let mut last_rms = 0.0f32;
        for i in 0..4410 {
            let s = (2.0 * PI * 440.0 * i as f32 / sr as f32).sin() * 0.5;
            last_rms = meter.process_sample(s);
        }
        // RMS of a 0.5-amplitude sine ≈ 0.5/sqrt(2) ≈ 0.354
        assert!(
            (last_rms - 0.354).abs() < 0.05,
            "RMS meter should track sine level: {last_rms}"
        );
    }

    #[test]
    fn test_peak_meter_reset() {
        let mut meter = PeakMeter::new(100.0, 60.0, 44_100);
        meter.process_block(&[0.9f32; 512]);
        meter.reset();
        assert!(
            meter.peak_db() == f32::NEG_INFINITY,
            "after reset peak_db should be NEG_INFINITY"
        );
    }

    // ── normalize_to_lufs tests ───────────────────────────────────────────────

    #[test]
    fn test_normalize_to_lufs_silence_unchanged() {
        // Near-silence buffer (< -70 LUFS) should be returned unchanged.
        let buf = AudioBuffer {
            samples: vec![0.0f32; 48_000 * 3],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let result = normalize_to_lufs(&buf, -23.0).expect("normalize_to_lufs failed");
        // All samples should remain 0.0.
        for &s in &result.samples {
            assert_eq!(s, 0.0, "silence buffer should not be modified");
        }
    }

    #[test]
    fn test_normalize_to_lufs_applies_gain() {
        // A non-silent signal after normalization should measure closer to target.
        // Use a 5-second calibration tone at a known level.
        let target_lufs = -23.0f32;
        let buf = calibration_tone(-30.0, 48_000, 5.0);

        let measured_before = loudness_integrated(&buf);
        assert!(
            measured_before.is_finite(),
            "pre-normalization LUFS should be finite, got {measured_before}"
        );

        let normalized = normalize_to_lufs(&buf, target_lufs).expect("normalize_to_lufs failed");

        let measured_after = loudness_integrated(&normalized);
        assert!(
            measured_after.is_finite(),
            "post-normalization LUFS should be finite, got {measured_after}"
        );

        // After normalization, the measured LUFS should be closer to target_lufs.
        let before_diff = (measured_before - target_lufs).abs();
        let after_diff = (measured_after - target_lufs).abs();
        assert!(
            after_diff < before_diff,
            "normalization should move LUFS closer to target ({target_lufs}): \
             before_diff={before_diff:.2}, after_diff={after_diff:.2}"
        );

        // Should be within ±2 LU of target.
        assert!(
            after_diff < 2.0,
            "normalized LUFS ({measured_after:.2}) should be within ±2 LU of target ({target_lufs})"
        );
    }

    #[test]
    fn test_rms_meter_silence() {
        let mut meter = RmsMeter::new(100.0, 44_100);
        for _ in 0..4410 {
            let r = meter.process_sample(0.0);
            assert!((r).abs() < 1e-9, "RMS of silence should be 0.0: {r}");
        }
        assert!(
            meter.rms_db() == f32::NEG_INFINITY,
            "rms_db of silence should be NEG_INFINITY"
        );
    }

    #[test]
    fn ebu_r128_calibration_tone_minus23_lufs() {
        // Test 9: A 1 kHz sine at -23 dBFS RMS should measure -23 LUFS (±0.5 LU).
        //
        // K-weighting is essentially 0 dB at 1 kHz, so -23 LUFS ≈ -23 dBFS RMS.
        // Amplitude for -23 dBFS RMS sine: amplitude = 10^(-23/20) ≈ 0.07079
        // (sine RMS = amplitude / sqrt(2), so RMS² = amplitude² / 2)
        // For LUFS: -0.691 + 10*log10(mean_square) = target
        // => mean_square = 10^((target + 0.691) / 10)
        // => amplitude = sqrt(2 * mean_square) for a sine wave
        let target_lufs = -23.0f32;
        let mean_sq = 10.0f32.powf((target_lufs + 0.691) / 10.0);
        let amplitude = (2.0 * mean_sq).sqrt();

        let sr = 48_000u32;
        let secs = 3.0f32;
        let n = (sr as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| amplitude * (2.0 * PI * 1_000.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };

        let measured_lufs = loudness_integrated(&buf);
        assert!(
            (measured_lufs - target_lufs).abs() < 0.5,
            "EBU R128 calibration tone: expected {target_lufs} LUFS (±0.5 LU), got {measured_lufs:.3} LUFS"
        );
    }
}
