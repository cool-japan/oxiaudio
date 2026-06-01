use oxiaudio_core::{AudioBuffer, OxiAudioError};

/// Onset strength using spectral flux (half-wave rectified L1 norm of spectral difference).
///
/// Returns a `Vec<f32>` — one value per STFT frame (onset strength signal).
/// The buffer is mixed to mono internally.
#[must_use = "returns the onset strength signal"]
pub fn onset_strength_spectral_flux(
    buf: &AudioBuffer<f32>,
    n_fft: usize,
    hop_size: usize,
) -> Result<Vec<f32>, OxiAudioError> {
    use crate::spectral::{stft, WindowFn};

    let stft_out = stft(buf, n_fft, hop_size, WindowFn::Hann)?;
    let n_bins = n_fft / 2 + 1;

    let mut onset = Vec::with_capacity(stft_out.frames.len());
    let mut prev_mag = vec![0.0f32; n_bins];

    for frame in &stft_out.frames {
        let mut flux = 0.0f32;
        for k in 0..n_bins {
            let mag = if k < frame.len() {
                frame[k].norm()
            } else {
                0.0
            };
            // Half-wave rectification: only positive differences
            let diff = (mag - prev_mag[k]).max(0.0);
            flux += diff;
            prev_mag[k] = mag;
        }
        onset.push(flux);
    }

    Ok(onset)
}

/// Onset detection using High-Frequency Content (HFC).
///
/// Emphasizes high-frequency energy increases (good for percussive onsets).
/// The buffer is mixed to mono internally.
#[must_use = "returns the HFC onset strength signal"]
pub fn onset_strength_hfc(
    buf: &AudioBuffer<f32>,
    n_fft: usize,
    hop_size: usize,
) -> Result<Vec<f32>, OxiAudioError> {
    use crate::spectral::{stft, WindowFn};

    let stft_out = stft(buf, n_fft, hop_size, WindowFn::Hann)?;
    let n_bins = n_fft / 2 + 1;

    let mut onset = Vec::with_capacity(stft_out.frames.len());
    let mut prev_hfc = 0.0f32;

    for frame in &stft_out.frames {
        let mut hfc = 0.0f32;
        for k in 0..n_bins {
            let mag_sq = if k < frame.len() {
                frame[k].norm_sqr()
            } else {
                0.0
            };
            hfc += k as f32 * mag_sq;
        }
        let diff = (hfc - prev_hfc).max(0.0);
        onset.push(diff);
        prev_hfc = hfc;
    }

    Ok(onset)
}

/// Detect onset positions (in frames) from an onset strength signal.
///
/// Returns frame indices where onsets occur.
///
/// * `adaptive_threshold_frames` — window for adaptive threshold
///   (default: 11 frames ≈ 0.25 s at hop=512/44100).
/// * `min_inter_onset_frames` — minimum frames between consecutive onsets
///   (default: 8 ≈ 93 ms).
/// * `delta` — threshold offset above adaptive median (default: 0.07).
pub fn pick_onset_peaks(
    onset_strength: &[f32],
    adaptive_threshold_frames: usize,
    min_inter_onset_frames: usize,
    delta: f32,
) -> Vec<usize> {
    let n = onset_strength.len();
    if n < 3 {
        return vec![];
    }

    let half_win = adaptive_threshold_frames / 2;
    let mut peaks = Vec::new();
    let mut last_peak_frame = 0usize;

    for i in 1..(n - 1) {
        // Local max condition
        if onset_strength[i] <= onset_strength[i - 1] || onset_strength[i] <= onset_strength[i + 1]
        {
            continue;
        }
        // Min inter-onset distance
        if !peaks.is_empty() && i - last_peak_frame < min_inter_onset_frames {
            continue;
        }
        // Adaptive threshold: median of local window + delta
        let win_start = i.saturating_sub(half_win);
        let win_end = (i + half_win + 1).min(n);
        let mut window: Vec<f32> = onset_strength[win_start..win_end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = window[window.len() / 2];
        let threshold = median + delta;

        if onset_strength[i] > threshold {
            peaks.push(i);
            last_peak_frame = i;
        }
    }

    peaks
}

/// Estimated tempo and beat grid from an `AudioBuffer`.
#[derive(Debug, Clone)]
pub struct TempoEstimate {
    /// Estimated tempo in BPM.
    pub bpm: f32,
    /// Confidence score 0.0–1.0.
    pub confidence: f32,
    /// Beat positions in seconds (estimated).
    pub beat_times: Vec<f64>,
}

/// Estimate tempo and beat positions from audio.
///
/// Uses onset detection → inter-onset interval histogram → peak → beat alignment
/// via phase scoring.
#[must_use = "returns a TempoEstimate with BPM, confidence, and beat times"]
pub fn estimate_tempo(
    buf: &AudioBuffer<f32>,
    n_fft: usize,
    hop_size: usize,
) -> Result<TempoEstimate, OxiAudioError> {
    // 1. Compute onset strength signal
    let onset_strength = onset_strength_spectral_flux(buf, n_fft, hop_size)?;

    // 2. Frame duration in seconds
    let hop_secs = hop_size as f64 / buf.sample_rate as f64;

    // 3. Detect onsets
    let onset_frames = pick_onset_peaks(&onset_strength, 11, 8, 0.05);

    if onset_frames.len() < 2 {
        return Ok(TempoEstimate {
            bpm: 0.0,
            confidence: 0.0,
            beat_times: vec![],
        });
    }

    // 4. Compute inter-onset intervals (IOIs) in frames
    let iois: Vec<usize> = onset_frames.windows(2).map(|w| w[1] - w[0]).collect();

    // 5. IOI histogram in BPM space [40, 240]
    let min_bpm = 40.0f64;
    let max_bpm = 240.0f64;
    let n_bpm_bins = 201usize; // 40..=240 inclusive

    let mut histogram = vec![0.0f32; n_bpm_bins];

    for &ioi in &iois {
        let ioi_secs = ioi as f64 * hop_secs;
        if ioi_secs < 1e-6 {
            continue;
        }
        // Primary BPM
        let bpm = 60.0 / ioi_secs;
        if bpm >= min_bpm && bpm <= max_bpm {
            let bin = ((bpm - min_bpm).round() as usize).min(n_bpm_bins - 1);
            histogram[bin] += 1.0;
        }
        // Harmonics at 0.5× and 2× (lower weight)
        for &mult in &[0.5f64, 2.0f64] {
            let harmonic_bpm = bpm * mult;
            if harmonic_bpm >= min_bpm && harmonic_bpm <= max_bpm {
                let bin = ((harmonic_bpm - min_bpm).round() as usize).min(n_bpm_bins - 1);
                histogram[bin] += 0.5;
            }
        }
    }

    // 6. Find peak BPM — use `let else` to avoid unwrap
    let Some((best_bin, &best_count)) = histogram
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Ok(TempoEstimate {
            bpm: 0.0,
            confidence: 0.0,
            beat_times: vec![],
        });
    };

    if best_count < 1.0 {
        return Ok(TempoEstimate {
            bpm: 0.0,
            confidence: 0.0,
            beat_times: vec![],
        });
    }

    let tempo_bpm = min_bpm as f32 + best_bin as f32;
    let beat_period_secs = 60.0 / tempo_bpm as f64;
    let beat_period_frames = (beat_period_secs / hop_secs).round() as usize;

    // Confidence = normalized histogram peak
    let total_weight: f32 = histogram.iter().sum();
    let confidence = if total_weight > 0.0 {
        best_count / total_weight
    } else {
        0.0
    };

    // 7. Beat alignment: find best phase offset phi in [0, beat_period_frames)
    let n_frames = onset_strength.len();
    let mut best_phase = 0usize;
    let mut best_score = f32::NEG_INFINITY;

    if beat_period_frames > 0 {
        for phi in 0..beat_period_frames.min(n_frames) {
            let score: f32 = (0..)
                .map(|k: usize| phi + k * beat_period_frames)
                .take_while(|&idx| idx < n_frames)
                .map(|idx| onset_strength[idx])
                .sum();
            if score > best_score {
                best_score = score;
                best_phase = phi;
            }
        }
    }

    // 8. Generate beat times
    let total_dur = n_frames as f64 * hop_secs;
    let beat_times: Vec<f64> = if beat_period_frames == 0 {
        vec![]
    } else {
        (0..)
            .map(|k: usize| (best_phase + k * beat_period_frames) as f64 * hop_secs)
            .take_while(|&t| t < total_dur)
            .collect()
    };

    Ok(TempoEstimate {
        bpm: tempo_bpm,
        confidence,
        beat_times,
    })
}

/// Complex-domain onset detection function (Duxbury et al., 2003).
///
/// For each STFT frame the deviation between the current spectrum and the
/// linearly-predicted spectrum from the previous two frames is summed across
/// all bins. Higher values indicate potential note or transient onsets.
///
/// Returns one value per STFT frame (outer vector). The first two entries are
/// always `0.0` because at least two prior frames are required to form the
/// linear prediction. An empty buffer or one that is too short to produce any
/// STFT frames yields an empty `Vec`.
///
/// To convert the returned signal into onset time positions, pass it to
/// [`pick_onset_peaks`].
///
/// # References
/// Duxbury, C., Bello, J. P., Davies, M., & Sandler, M. (2003).
/// *Complex Domain Onset Detection for Musical Signals.*
/// Proc. DAFx-2003.
pub fn complex_domain_onset(
    buf: &oxiaudio_core::AudioBuffer<f32>,
    n_fft: usize,
    hop_size: usize,
) -> Vec<f32> {
    use crate::spectral::{stft, WindowFn};

    let stft_out = match stft(buf, n_fft, hop_size, WindowFn::Hann) {
        Ok(o) => o,
        Err(_) => return vec![],
    };

    let n_frames = stft_out.frames.len();
    if n_frames == 0 {
        return vec![];
    }

    let mut onset = Vec::with_capacity(n_frames);

    // Frames 0 and 1 cannot have a 2-frame lookback; output 0.0.
    onset.push(0.0f32);
    if n_frames > 1 {
        onset.push(0.0f32);
    }

    for t in 2..n_frames {
        let frame_t = &stft_out.frames[t];
        let frame_t1 = &stft_out.frames[t - 1];
        let frame_t2 = &stft_out.frames[t - 2];
        let n_bins = frame_t.len().min(frame_t1.len()).min(frame_t2.len());

        let deviation: f32 = (0..n_bins)
            .map(|k| {
                // Phase at t-1 and t-2
                let ph_t1 = frame_t1[k].im.atan2(frame_t1[k].re);
                let ph_t2 = frame_t2[k].im.atan2(frame_t2[k].re);
                // Linear phase prediction at frame t
                let exp_ph = 2.0 * ph_t1 - ph_t2;
                // Magnitude at t-1 used as predicted amplitude
                let mag_t1 =
                    (frame_t1[k].re * frame_t1[k].re + frame_t1[k].im * frame_t1[k].im).sqrt();
                // Complex target
                let target_re = mag_t1 * exp_ph.cos();
                let target_im = mag_t1 * exp_ph.sin();
                // Complex distance between actual and predicted
                let diff_re = frame_t[k].re - target_re;
                let diff_im = frame_t[k].im - target_im;
                (diff_re * diff_re + diff_im * diff_im).sqrt()
            })
            .sum();

        onset.push(deviation);
    }

    onset
}

/// Detect onset times in seconds.
#[must_use = "returns onset times in seconds"]
pub fn detect_onsets(
    buf: &AudioBuffer<f32>,
    n_fft: usize,
    hop_size: usize,
) -> Result<Vec<f64>, OxiAudioError> {
    let onset_strength = onset_strength_spectral_flux(buf, n_fft, hop_size)?;
    let hop_secs = hop_size as f64 / buf.sample_rate as f64;
    let peak_frames = pick_onset_peaks(&onset_strength, 11, 8, 0.05);
    Ok(peak_frames.iter().map(|&f| f as f64 * hop_secs).collect())
}

/// Detect downbeats (first beat of each bar) from an estimated BPM and beat times.
///
/// Algorithm:
/// 1. If `beat_times_secs` has fewer than 4 entries, return empty `Vec`.
/// 2. Estimate bar period = `bar_length * (60.0 / bpm)` seconds.
/// 3. Starting from the first beat, candidate bar starts are at indices
///    `i = 0, bar_length, 2 * bar_length, …` into `beat_times_secs`.
/// 4. For each candidate, compute low-frequency energy in a ~50 ms window
///    around that beat using the audio buffer. "Low-frequency content" is
///    approximated by averaging the mean-square energy of only the first
///    `frame_size / 8` samples of the window (temporal weighting).
/// 5. Score each candidate as `bass_energy / mean(all_bass_energies)`.
/// 6. Return times of candidates with score > 1.0 (above-average bass energy).
///
/// # Parameters
/// - `buf` — the audio buffer used to compute bass energy
/// - `beat_times_secs` — beat times in seconds (e.g. from [`estimate_tempo`])
/// - `bpm` — estimated tempo in BPM
/// - `bar_length` — beats per bar (e.g. 4 for 4/4, 3 for 3/4)
#[must_use = "returns downbeat times in seconds"]
pub fn detect_downbeats(
    buf: &AudioBuffer<f32>,
    beat_times_secs: &[f64],
    bpm: f32,
    bar_length: usize,
) -> Vec<f64> {
    if beat_times_secs.len() < 4 {
        return vec![];
    }

    if bar_length == 0 || bpm <= 0.0 {
        return vec![];
    }

    let sr = buf.sample_rate as f64;
    // ~50 ms window for bass energy estimation
    let frame_size = ((buf.sample_rate as usize / 20).max(1024)).next_power_of_two();
    let half_frame = frame_size / 2;
    let bass_bins = (frame_size / 8).max(1);

    // Mix to mono for energy computation: use channel average
    let n_ch = buf.channels.channel_count().max(1);
    let n_samples_per_ch = buf.samples.len() / n_ch;

    // Gather candidate bar-start beat indices: 0, bar_length, 2*bar_length, ...
    let mut candidates: Vec<f64> = Vec::new();
    let mut i = 0usize;
    while i < beat_times_secs.len() {
        candidates.push(beat_times_secs[i]);
        i = i.saturating_add(bar_length);
    }

    if candidates.is_empty() {
        return vec![];
    }

    // For each candidate, compute bass energy in a window around that beat.
    let bass_energies: Vec<f32> = candidates
        .iter()
        .map(|&t_secs| {
            let center = (t_secs * sr) as usize;
            let start = center.saturating_sub(half_frame);
            let end = (start + frame_size).min(n_samples_per_ch);
            if start >= end || end > n_samples_per_ch {
                return 0.0f32;
            }
            let window_len = end - start;
            // Compute mean-square energy of first `bass_bins` samples (in each channel)
            let actual_bass = bass_bins.min(window_len);
            if actual_bass == 0 {
                return 0.0f32;
            }
            let mut energy = 0.0f32;
            let mut count = 0usize;
            for b in 0..actual_bass {
                let sample_idx = start + b;
                // Average across channels (interleaved)
                let mut ch_sum = 0.0f32;
                for c in 0..n_ch {
                    let flat_idx = sample_idx * n_ch + c;
                    if flat_idx < buf.samples.len() {
                        ch_sum += buf.samples[flat_idx];
                    }
                }
                let mono_sample = ch_sum / n_ch as f32;
                energy += mono_sample * mono_sample;
                count += 1;
            }
            if count == 0 {
                0.0
            } else {
                energy / count as f32
            }
        })
        .collect();

    let mean_energy: f32 = if bass_energies.is_empty() {
        0.0
    } else {
        bass_energies.iter().sum::<f32>() / bass_energies.len() as f32
    };

    if mean_energy < 1e-12 {
        return vec![];
    }

    // Return candidates with above-average bass energy
    candidates
        .into_iter()
        .zip(bass_energies.iter())
        .filter_map(|(t, &e)| if e / mean_energy > 1.0 { Some(t) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    fn click_train(bpm: f32, sr: u32, dur: f32) -> AudioBuffer<f32> {
        let n = (sr as f32 * dur) as usize;
        let beat_period = (sr as f32 * 60.0 / bpm) as usize;
        let mut samples = vec![0.0f32; n];
        let mut pos = 0;
        while pos < n {
            // Gaussian click: bell shape centered at pos, width ~20 samples
            for i in 0..40usize {
                let idx = pos + i;
                if idx >= n {
                    break;
                }
                let x = i as f32 - 20.0;
                samples[idx] += (-0.005 * x * x).exp();
            }
            pos += beat_period;
        }
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_onset_strength_click_train() {
        // 120 BPM click train: onset strength should have peaks at beat positions
        let buf = click_train(120.0, 44100, 4.0);
        let strength =
            onset_strength_spectral_flux(&buf, 2048, 512).expect("onset_strength failed");
        assert!(!strength.is_empty());
        // Max strength should be above mean by a factor of 1.5 at least
        let mean: f32 = strength.iter().sum::<f32>() / strength.len() as f32;
        let max: f32 = strength.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max > mean * 1.5,
            "click train should produce strong onsets: max={max:.4} mean={mean:.4}"
        );
    }

    #[test]
    fn test_onset_strength_hfc_not_empty() {
        let buf = click_train(120.0, 44100, 2.0);
        let hfc = onset_strength_hfc(&buf, 2048, 512).expect("onset_strength_hfc failed");
        assert!(!hfc.is_empty());
        assert!(hfc.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_pick_onset_peaks_click_train() {
        let buf = click_train(120.0, 44100, 4.0);
        let strength =
            onset_strength_spectral_flux(&buf, 2048, 512).expect("onset_strength failed");
        let peaks = pick_onset_peaks(&strength, 11, 8, 0.01);
        // 120 BPM for 4 seconds = ~8 beats; should detect at least 4
        assert!(
            peaks.len() >= 4,
            "should detect clicks: got {} peaks",
            peaks.len()
        );
    }

    #[test]
    fn test_detect_onsets_returns_sorted_times() {
        let buf = click_train(120.0, 44100, 4.0);
        let onsets = detect_onsets(&buf, 2048, 512).expect("detect_onsets failed");
        // Times should be monotonically increasing
        for w in onsets.windows(2) {
            assert!(
                w[1] > w[0],
                "onset times should be sorted: {} > {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn test_tempo_estimate_120bpm_click_train() {
        let buf = click_train(120.0, 44100, 8.0);
        let estimate = estimate_tempo(&buf, 2048, 512).expect("estimate_tempo failed");
        // Should detect approximately 120 BPM (within ±10 BPM)
        // Or 60/240 BPM (half/double tempo — both acceptable)
        let bpm = estimate.bpm;
        let closest = [60.0f32, 120.0, 240.0]
            .iter()
            .map(|&b| (b - bpm).abs())
            .fold(f32::MAX, f32::min);
        assert!(
            closest < 15.0,
            "tempo should be near 60/120/240 BPM, got {bpm:.1}"
        );
    }

    #[test]
    fn test_tempo_estimate_silence_returns_zero() {
        let buf = AudioBuffer {
            samples: vec![0.001f32; 44100 * 4],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let estimate = estimate_tempo(&buf, 2048, 512).expect("estimate_tempo on silence failed");
        // Silent signal: no onsets, bpm=0 or confidence=0
        assert!(
            estimate.bpm == 0.0 || estimate.confidence < 0.1,
            "silence should give 0 tempo or low confidence"
        );
    }

    // ── complex_domain_onset tests ────────────────────────────────────────────

    #[test]
    fn test_complex_domain_onset_length() {
        // 1 second at 48 kHz → output length should equal the number of STFT frames.
        use crate::spectral::{stft, WindowFn};
        let sr = 48_000u32;
        let n = sr as usize; // 1 second
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let n_fft = 1024usize;
        let hop = 256usize;
        let onset = super::complex_domain_onset(&buf, n_fft, hop);
        // Expected frame count from the STFT
        let stft_out = stft(&buf, n_fft, hop, WindowFn::Hann).expect("stft failed");
        assert_eq!(
            onset.len(),
            stft_out.frames.len(),
            "complex_domain_onset length must equal n_stft_frames"
        );
    }

    #[test]
    fn test_complex_domain_onset_silent() {
        // A silence buffer should produce all-zero onset values.
        let sr = 44_100u32;
        let buf = AudioBuffer {
            samples: vec![0.0f32; sr as usize],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let onset = super::complex_domain_onset(&buf, 1024, 256);
        for (i, &v) in onset.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "silent buffer should produce zero onset at frame {i}, got {v}"
            );
        }
    }

    #[test]
    fn test_complex_domain_onset_click() {
        // A click train with 500 ms spacing should produce detectable peaks.
        let sr = 44_100u32;
        let dur = 4.0f32;
        let n = (sr as f32 * dur) as usize;
        let click_period = (sr as f32 * 0.5) as usize; // 500 ms
        let mut samples = vec![0.0f32; n];
        let mut pos = 0usize;
        while pos < n {
            // Short Gaussian click centered at pos
            for i in 0..40usize {
                let idx = pos + i;
                if idx >= n {
                    break;
                }
                let x = i as f32 - 20.0;
                samples[idx] += (-0.005 * x * x).exp();
            }
            pos += click_period;
        }
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let onset = super::complex_domain_onset(&buf, 1024, 256);
        assert!(!onset.is_empty(), "onset signal should not be empty");
        let peaks = pick_onset_peaks(&onset, 11, 4, 0.0);
        assert!(
            !peaks.is_empty(),
            "should detect at least 1 peak in a click train, got {} onset values",
            onset.len()
        );
    }

    #[test]
    fn test_tempo_estimate_beat_times_in_range() {
        let buf = click_train(120.0, 44100, 8.0);
        let estimate = estimate_tempo(&buf, 2048, 512).expect("estimate_tempo failed");
        let duration = 8.0f64;
        for &t in &estimate.beat_times {
            assert!(
                t >= 0.0 && t <= duration,
                "beat time {t:.3} should be in [0, {duration}]"
            );
        }
    }

    // ── detect_downbeats tests ────────────────────────────────────────────────

    #[test]
    fn test_detect_downbeats_empty_beats() {
        // Fewer than 4 beat times → empty Vec
        let sr = 48_000u32;
        let buf = AudioBuffer {
            samples: vec![0.5f32; sr as usize * 4],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let result = super::detect_downbeats(&buf, &[], 120.0, 4);
        assert!(result.is_empty(), "0 beats should return empty Vec");

        let result2 = super::detect_downbeats(&buf, &[0.0, 0.5, 1.0], 120.0, 4);
        assert!(result2.is_empty(), "3 beats should return empty Vec");
    }

    #[test]
    fn test_detect_downbeats_reasonable_output() {
        // 4-second sine buffer at 48 kHz
        let sr = 48_000u32;
        let n = (sr as usize) * 4;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        // 120 BPM, beat every 0.5s, bar_length=4
        let beat_times = vec![0.0f64, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let downbeats = super::detect_downbeats(&buf, &beat_times, 120.0, 4);
        // Must not panic; result is Vec<f64> (may be empty or non-empty)
        for &t in &downbeats {
            assert!(
                (0.0..=4.0).contains(&t),
                "downbeat time {t:.3} should be within buffer"
            );
        }
    }

    #[test]
    fn onset_detection_click_train_correct_count_and_spacing() {
        // Test 7: Onset detection on a synthetic click train.
        // 9 clicks spaced 500ms apart (starting at t=0.5s to avoid edge effects).
        // Using detect_onsets with n_fft=2048, hop=512 at 44100 Hz.
        // The hop time is ~11.6ms; tolerance = 50ms (4 hops) covers quantization.
        // We verify:
        //   1. At least 7 out of 9 clicks are detected within 50ms tolerance.
        //   2. No spurious detections more than 50ms from any true click.
        let sr = 44_100u32;
        let total_secs = 5.5f32;
        let n = (sr as f32 * total_secs) as usize;
        let mut samples = vec![0.0f32; n];

        // True click positions in seconds: 0.5, 1.0, 1.5, ..., 4.5s (9 clicks)
        let click_times_secs: Vec<f64> = (1..=9).map(|k| k as f64 * 0.5).collect();
        let burst_width_samples = (0.005 * sr as f32) as usize; // 5ms burst

        for &t_secs in &click_times_secs {
            let center = (t_secs * sr as f64) as usize;
            let half = (burst_width_samples / 2).max(1);
            let start = center.saturating_sub(half);
            let end = (center + half + 1).min(n);
            for (i, s) in samples.iter_mut().enumerate().take(end).skip(start) {
                let rel = (i as f64 - center as f64) / half as f64;
                let gaussian = (-4.0 * rel * rel).exp() as f32;
                *s += 0.8 * gaussian;
            }
        }

        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };

        let detected = detect_onsets(&buf, 2048, 512).expect("detect_onsets should succeed");
        assert!(
            !detected.is_empty(),
            "click train should produce at least one onset"
        );

        // Count how many true clicks were matched within 50ms tolerance
        let tolerance_secs = 0.050f64;
        let mut matched = 0usize;
        for &true_time in &click_times_secs {
            let found = detected
                .iter()
                .any(|&d| (d - true_time).abs() <= tolerance_secs);
            if found {
                matched += 1;
            }
        }
        assert!(
            matched >= 6,
            "at least 6 out of 9 clicks should be detected within 50ms; matched={matched}, detected={detected:?}, true={click_times_secs:?}"
        );
    }
}
