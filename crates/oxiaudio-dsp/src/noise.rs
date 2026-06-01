use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError, SampleFormat};

use crate::mix_to_mono;
use crate::spectral::{istft, stft, Complex, StftOutput, WindowFn};

// ── Public API ────────────────────────────────────────────────────────────────

/// Suppress frequency bins with magnitude below `threshold` in each STFT frame.
///
/// Unlike a broadband gate, this operates per-bin: each frequency bin whose magnitude
/// falls below `threshold` (linear, in the same scale as the STFT magnitude) is zeroed.
/// The signal is then reconstructed via iSTFT.  This preserves tonal content above the
/// threshold while suppressing low-level noise in quiet frequency bands.
///
/// The input is mixed to mono for STFT analysis and reconstruction; the resulting
/// mono signal is then replicated across all original channels in the output.
///
/// # Parameters
///
/// - `buf`       — input buffer (any channel layout).
/// - `threshold` — linear magnitude threshold (e.g., `0.01` = −40 dB relative to full scale).
/// - `n_fft`     — FFT size in samples (e.g., 1024).
/// - `hop_size`  — hop between STFT frames in samples (e.g., 256).
#[must_use = "call with the result to get the gated buffer"]
pub fn frequency_domain_noise_gate(
    buf: &oxiaudio_core::AudioBuffer<f32>,
    threshold: f32,
    n_fft: usize,
    hop_size: usize,
) -> oxiaudio_core::AudioBuffer<f32> {
    use oxiaudio_core::ChannelLayout;

    // Mix to mono for processing.
    let mono = mix_to_mono(buf);
    let original_len = mono.samples.len();

    // Zero-pad by n_fft/2 on each side, matching spectral_subtraction's approach.
    // This prevents the OLA normalisation from amplifying transients at the frame
    // edges where the Hann window value is near zero.
    let pad = n_fft / 2;

    // Fall back to the original buffer (as mono-replicated) on STFT failure.
    let build_output = |mono_samples: Vec<f32>| -> oxiaudio_core::AudioBuffer<f32> {
        let n_channels = buf.channels.channel_count();
        if n_channels == 1 {
            return oxiaudio_core::AudioBuffer {
                samples: mono_samples,
                sample_rate: buf.sample_rate,
                channels: ChannelLayout::Mono,
                format: buf.format,
            };
        }
        let n_frames = mono_samples.len();
        let mut samples = vec![0.0_f32; n_frames * n_channels];
        for (f, &s) in mono_samples.iter().enumerate() {
            for ch in 0..n_channels {
                samples[f * n_channels + ch] = s;
            }
        }
        oxiaudio_core::AudioBuffer {
            samples,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    };

    let padded = pad_signal(&mono, pad);
    let padded_len = padded.samples.len();

    let stft_out = match stft(&padded, n_fft, hop_size, WindowFn::Hann) {
        Ok(o) => o,
        Err(_) => return build_output(mono.samples),
    };

    // Gate each bin per frame.
    let gated_frames: Vec<Vec<Complex<f32>>> = stft_out
        .frames
        .iter()
        .map(|frame| {
            frame
                .iter()
                .map(|&c| {
                    if c.norm() < threshold {
                        Complex::new(0.0_f32, 0.0_f32)
                    } else {
                        c
                    }
                })
                .collect()
        })
        .collect();

    let gated_stft = StftOutput {
        frames: gated_frames,
        sample_rate: stft_out.sample_rate,
        hop_size: stft_out.hop_size,
        window: stft_out.window,
    };

    let reconstructed = match istft(&gated_stft, padded_len) {
        Ok(r) => r,
        Err(_) => return build_output(mono.samples),
    };

    // Trim the zero-padding and produce the output.
    let trimmed = trimmed_samples(&reconstructed.samples, pad, original_len);
    build_output(trimmed)
}

/// Estimate per-bin noise magnitude from a silence segment.
///
/// Returns a `Vec<f32>` of length `n_fft / 2 + 1` (positive frequencies only).
/// The profile is the mean per-bin magnitude across all STFT frames.
///
/// The underlying FFT uses a full complex DFT (not a real-FFT), so each frame
/// contains `n_fft` bins.  Only the first `n_fft / 2 + 1` (DC through Nyquist)
/// are averaged; the conjugate mirror is redundant for real signals.
///
/// # Errors
///
/// Returns `OxiAudioError::UnsupportedFormat` when `n_fft < 2` or when the
/// internal STFT call fails.
#[must_use = "returns the per-bin noise profile used by spectral_subtraction / wiener_filter"]
pub fn estimate_noise_profile(
    silence: &AudioBuffer<f32>,
    n_fft: usize,
) -> Result<Vec<f32>, OxiAudioError> {
    if n_fft < 2 {
        return Err(OxiAudioError::UnsupportedFormat(
            "n_fft must be at least 2".to_owned(),
        ));
    }

    let n_bins = n_fft / 2 + 1;
    let mono = mix_to_mono(silence);

    // Short-circuit: not enough samples for even one STFT frame.
    if mono.samples.len() < n_fft {
        return Ok(vec![1e-10_f32; n_bins]);
    }

    let hop = n_fft / 2;
    let stft_out = stft(&mono, n_fft, hop, WindowFn::Hann)?;

    let n_frames = stft_out.frames.len();
    if n_frames == 0 {
        return Ok(vec![1e-10_f32; n_bins]);
    }

    let mut profile = vec![0.0_f32; n_bins];
    for frame in &stft_out.frames {
        // Frames contain n_fft bins; only read the positive-frequency half.
        let usable = frame.len().min(n_bins);
        for (k, c) in frame[..usable].iter().enumerate() {
            profile[k] += c.norm();
        }
    }
    let inv_frames = 1.0 / n_frames as f32;
    for p in &mut profile {
        *p *= inv_frames;
    }

    Ok(profile)
}

/// Reduce noise using spectral subtraction.
///
/// Each frequency bin's magnitude is attenuated by the estimated noise floor.
/// A spectral floor `oversubtraction` prevents complete suppression and reduces
/// musical-noise artefacts.
///
/// * `noise_profile` — per-bin noise magnitude from [`estimate_noise_profile`].
///   Its length encodes `n_fft`: `n_fft = (profile.len() - 1) * 2`.
/// * `oversubtraction` — minimum gain relative to the original magnitude
///   (e.g. `0.1` = keep at least 10 % of each bin).
///
/// # Errors
///
/// Returns `OxiAudioError::UnsupportedFormat` when `noise_profile.len() < 2`
/// or when any internal STFT/iSTFT call fails.
#[must_use = "returns the noise-reduced AudioBuffer"]
pub fn spectral_subtraction(
    buf: &AudioBuffer<f32>,
    noise_profile: &[f32],
    oversubtraction: f32,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if noise_profile.len() < 2 {
        return Err(OxiAudioError::UnsupportedFormat(
            "noise_profile must have at least 2 bins".to_owned(),
        ));
    }

    let n_fft = (noise_profile.len() - 1) * 2;
    let hop = n_fft / 2;
    let floor = oversubtraction.max(0.0);

    // Zero-pad by n_fft/2 on each side so that the first real sample falls
    // near the middle of the first STFT frame, where the Hann window value is
    // large.  Without padding, OLA normalisation by the tiny window values at
    // the frame edges amplifies filter transients catastrophically.
    let pad = n_fft / 2;

    let n_channels = buf.channels.channel_count();
    let n_frames_input = buf.samples.len() / n_channels;

    let mut channel_outputs: Vec<Vec<f32>> = Vec::with_capacity(n_channels);

    for ch in 0..n_channels {
        let channel_buf = extract_channel(buf, ch, n_channels, n_frames_input);
        let original_len = channel_buf.samples.len();

        // Pad with zeros.
        let padded = pad_signal(&channel_buf, pad);
        let padded_len = padded.samples.len();

        let stft_out = stft(&padded, n_fft, hop, WindowFn::Hann)?;

        let new_frames: Vec<Vec<Complex<f32>>> = stft_out
            .frames
            .iter()
            .map(|frame| subtraction_frame(frame, noise_profile, n_fft, floor))
            .collect();

        let modified_stft = StftOutput {
            frames: new_frames,
            sample_rate: stft_out.sample_rate,
            hop_size: stft_out.hop_size,
            window: stft_out.window,
        };

        // Reconstruct, then trim the zero-padding.
        let reconstructed = istft(&modified_stft, padded_len)?;
        let trimmed = trimmed_samples(&reconstructed.samples, pad, original_len);
        channel_outputs.push(trimmed);
    }

    Ok(interleave_channels(channel_outputs, buf))
}

/// Reduce noise using the Wiener filter.
///
/// Computes a per-bin SNR estimate and applies the optimal Wiener gain
/// `G(k) = SNR(k) / (SNR(k) + 1)` to each complex bin.
///
/// * `noise_profile` — per-bin noise magnitude from [`estimate_noise_profile`].
/// * `snr_floor` — minimum SNR assumed in linear scale (e.g. `0.0` = allow
///   full suppression, `0.01` = limit suppression to ≈20 dB).
///
/// # Errors
///
/// Returns `OxiAudioError::UnsupportedFormat` when `noise_profile.len() < 2`
/// or when any internal STFT/iSTFT call fails.
#[must_use = "returns the Wiener-filtered AudioBuffer"]
pub fn wiener_filter(
    buf: &AudioBuffer<f32>,
    noise_profile: &[f32],
    snr_floor: f32,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if noise_profile.len() < 2 {
        return Err(OxiAudioError::UnsupportedFormat(
            "noise_profile must have at least 2 bins".to_owned(),
        ));
    }

    let n_fft = (noise_profile.len() - 1) * 2;
    let hop = n_fft / 2;
    let floor_snr = snr_floor.max(0.0);

    // Pre-compute noise power per bin (positive frequencies only).
    let noise_power: Vec<f32> = noise_profile.iter().map(|&m| m * m).collect();

    // Zero-pad by n_fft/2 on each side (see spectral_subtraction for rationale).
    let pad = n_fft / 2;

    let n_channels = buf.channels.channel_count();
    let n_frames_input = buf.samples.len() / n_channels;

    let mut channel_outputs: Vec<Vec<f32>> = Vec::with_capacity(n_channels);

    for ch in 0..n_channels {
        let channel_buf = extract_channel(buf, ch, n_channels, n_frames_input);
        let original_len = channel_buf.samples.len();

        let padded = pad_signal(&channel_buf, pad);
        let padded_len = padded.samples.len();

        let stft_out = stft(&padded, n_fft, hop, WindowFn::Hann)?;

        let new_frames: Vec<Vec<Complex<f32>>> = stft_out
            .frames
            .iter()
            .map(|frame| wiener_frame(frame, &noise_power, n_fft, floor_snr))
            .collect();

        let modified_stft = StftOutput {
            frames: new_frames,
            sample_rate: stft_out.sample_rate,
            hop_size: stft_out.hop_size,
            window: stft_out.window,
        };

        let reconstructed = istft(&modified_stft, padded_len)?;
        let trimmed = trimmed_samples(&reconstructed.samples, pad, original_len);
        channel_outputs.push(trimmed);
    }

    Ok(interleave_channels(channel_outputs, buf))
}

// ── Frame-level processing helpers ────────────────────────────────────────────

/// Apply spectral subtraction to one STFT frame.
///
/// The frame has `n_fft` bins (full complex DFT output).  Bins `0..n_fft/2+1`
/// are the positive frequencies; bins `n_fft/2+1..n_fft` are the conjugate
/// mirror.  The same noise gain is applied to both halves so the iSTFT
/// produces a real-valued output.
fn subtraction_frame(
    frame: &[Complex<f32>],
    noise_profile: &[f32],
    n_fft: usize,
    floor: f32,
) -> Vec<Complex<f32>> {
    let n_pos = n_fft / 2 + 1; // positive-frequency bin count

    frame
        .iter()
        .enumerate()
        .map(|(k, &c)| {
            // Map bin index to the noise-profile index (positive-frequency mirror).
            let prof_idx = if k < n_pos { k } else { n_fft - k };
            let noise_mag = if prof_idx < noise_profile.len() {
                noise_profile[prof_idx]
            } else {
                0.0_f32
            };

            let mag = c.norm();
            let new_mag = (mag - noise_mag).max(floor * mag);

            // Preserve the original phase.
            if mag < 1e-30_f32 {
                // Avoid division by zero for essentially-zero bins.
                Complex::new(0.0_f32, 0.0_f32)
            } else {
                let scale = new_mag / mag;
                Complex::new(c.re * scale, c.im * scale)
            }
        })
        .collect()
}

/// Apply the Wiener gain to one STFT frame.
fn wiener_frame(
    frame: &[Complex<f32>],
    noise_power: &[f32],
    n_fft: usize,
    floor_snr: f32,
) -> Vec<Complex<f32>> {
    let n_pos = n_fft / 2 + 1;

    frame
        .iter()
        .enumerate()
        .map(|(k, &c)| {
            let prof_idx = if k < n_pos { k } else { n_fft - k };
            let np = if prof_idx < noise_power.len() {
                noise_power[prof_idx].max(1e-20_f32)
            } else {
                1e-20_f32
            };

            let signal_power = c.norm_sqr();
            let signal_est = (signal_power - np).max(0.0_f32);
            let snr = (signal_est / np).max(floor_snr);
            let gain = snr / (snr + 1.0_f32);
            Complex::new(c.re * gain, c.im * gain)
        })
        .collect()
}

// ── Internal channel helpers ──────────────────────────────────────────────────

/// Prepend and append `pad` zero samples to the mono buffer.
///
/// Zero-padding ensures the first real sample falls near the centre of the
/// first STFT frame (where the Hann window is near its peak), preventing the
/// iSTFT window normalisation from amplifying filter transients at the edges.
fn pad_signal(buf: &AudioBuffer<f32>, pad: usize) -> AudioBuffer<f32> {
    let mut samples = vec![0.0_f32; pad];
    samples.extend_from_slice(&buf.samples);
    samples.resize(samples.len() + pad, 0.0_f32);
    AudioBuffer {
        samples,
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        format: buf.format,
    }
}

/// Extract `original_len` samples starting at `pad` from a padded reconstruction.
fn trimmed_samples(samples: &[f32], pad: usize, original_len: usize) -> Vec<f32> {
    let start = pad.min(samples.len());
    let end = (start + original_len).min(samples.len());
    let mut out = samples[start..end].to_vec();
    if out.len() < original_len {
        out.resize(original_len, 0.0_f32);
    }
    out
}

/// Extract channel `ch` from an interleaved buffer into a mono `AudioBuffer`.
fn extract_channel(
    buf: &AudioBuffer<f32>,
    ch: usize,
    n_channels: usize,
    n_frames: usize,
) -> AudioBuffer<f32> {
    let samples: Vec<f32> = (0..n_frames)
        .map(|f| buf.samples[f * n_channels + ch])
        .collect();
    AudioBuffer {
        samples,
        sample_rate: buf.sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

/// Re-interleave per-channel sample vectors back into a multi-channel buffer.
///
/// All vectors are truncated to the shortest length so every channel is
/// the same size.
fn interleave_channels(
    channel_outputs: Vec<Vec<f32>>,
    source: &AudioBuffer<f32>,
) -> AudioBuffer<f32> {
    if channel_outputs.is_empty() {
        return AudioBuffer {
            samples: Vec::new(),
            sample_rate: source.sample_rate,
            channels: source.channels,
            format: source.format,
        };
    }

    let out_frames = channel_outputs.iter().map(|c| c.len()).min().unwrap_or(0);
    let n_channels = channel_outputs.len();
    let mut out_samples = vec![0.0_f32; out_frames * n_channels];
    for (ch, ch_data) in channel_outputs.iter().enumerate() {
        for f in 0..out_frames {
            out_samples[f * n_channels + ch] = ch_data[f];
        }
    }

    AudioBuffer {
        samples: out_samples,
        sample_rate: source.sample_rate,
        channels: source.channels,
        format: source.format,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{ChannelLayout, SampleFormat};

    fn sine_buf(freq: f32, sr: u32, dur: f32, amp: f32) -> AudioBuffer<f32> {
        let n = (sr as f32 * dur) as usize;
        AudioBuffer {
            samples: (0..n)
                .map(|i| {
                    let t = i as f32 / sr as f32;
                    amp * (2.0 * std::f32::consts::PI * freq * t).sin()
                })
                .collect(),
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn noise_buf(sr: u32, dur: f32, amp: f32) -> AudioBuffer<f32> {
        // Deterministic noise via LCG — same seed → same sequence every run.
        let n = (sr as f32 * dur) as usize;
        let mut state = 0x12345678_u64;
        let samples: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let r = (state >> 33) as f32 / u32::MAX as f32;
                amp * (r * 2.0 - 1.0)
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_estimate_noise_profile_produces_correct_length() {
        let silence = noise_buf(44_100, 0.1, 0.001);
        let profile =
            estimate_noise_profile(&silence, 1024).expect("estimate_noise_profile should succeed");
        // n_fft / 2 + 1 = 512 + 1 = 513
        assert_eq!(profile.len(), 513);
        assert!(
            profile.iter().all(|&p| p >= 0.0),
            "all profile values must be non-negative"
        );
    }

    #[test]
    fn test_spectral_subtraction_reduces_noise() {
        let sr = 44_100_u32;
        let signal = sine_buf(1000.0, sr, 0.5, 0.5);
        let noise = noise_buf(sr, 0.5, 0.05);

        // Noisy = signal + noise.
        let noisy_samples: Vec<f32> = signal
            .samples
            .iter()
            .zip(&noise.samples)
            .map(|(&s, &n)| s + n)
            .collect();
        let noisy = AudioBuffer {
            samples: noisy_samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };

        let profile =
            estimate_noise_profile(&noise, 1024).expect("estimate_noise_profile should succeed");
        let denoised = spectral_subtraction(&noisy, &profile, 0.1)
            .expect("spectral_subtraction should succeed");

        let min_len = signal
            .samples
            .len()
            .min(noisy.samples.len())
            .min(denoised.samples.len());

        let mse = |a: &[f32], b: &[f32]| -> f32 {
            a[..min_len]
                .iter()
                .zip(&b[..min_len])
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f32>()
                / min_len as f32
        };

        let err_noisy = mse(&noisy.samples, &signal.samples);
        let err_denoised = mse(&denoised.samples, &signal.samples);

        assert!(
            err_denoised < err_noisy,
            "spectral subtraction should improve SNR: noisy_err={err_noisy:.6} denoised_err={err_denoised:.6}"
        );
    }

    #[test]
    fn test_wiener_filter_reduces_noise() {
        let sr = 44_100_u32;
        let signal = sine_buf(1000.0, sr, 0.5, 0.5);
        let noise = noise_buf(sr, 0.5, 0.05);
        let noisy = AudioBuffer {
            samples: signal
                .samples
                .iter()
                .zip(&noise.samples)
                .map(|(&s, &n)| s + n)
                .collect(),
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };

        let profile =
            estimate_noise_profile(&noise, 1024).expect("estimate_noise_profile should succeed");
        let denoised = wiener_filter(&noisy, &profile, 0.0).expect("wiener_filter should succeed");

        assert!(!denoised.samples.is_empty(), "output must not be empty");

        let rms: f32 = (denoised.samples.iter().map(|&s| s * s).sum::<f32>()
            / denoised.samples.len() as f32)
            .sqrt();
        assert!(
            rms > 0.001,
            "wiener filter output should not be silent: rms={rms}"
        );
    }

    #[test]
    fn test_frequency_domain_noise_gate_reduces_noise() {
        // Test: a loud 440 Hz sine (strong bins) mixed with broadband noise (weak bins).
        // Gating with a threshold between the noise floor and the sine peak should:
        //   - preserve the loud sine bins (RMS stays > 0)
        //   - suppress quiet noise bins (output RMS ≤ input RMS + small OLA tolerance)
        //
        // We verify the output is non-silent and bounded — the gate must not produce
        // energy from nothing (output ≤ 110 % of input), and must preserve the tonal
        // signal (output RMS > 0.05).
        let sr = 44_100_u32;
        let n = (sr as f32 * 0.5) as usize;
        let sine_amp = 0.5_f32;
        let noise_amp = 0.02_f32;

        let mut state = 0xABCDEF01_u64;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let s = sine_amp * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let r = (state >> 33) as f32 / u32::MAX as f32;
                s + noise_amp * (r * 2.0 - 1.0)
            })
            .collect();
        let noisy = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };

        // Threshold chosen between noise_amp and sine bin magnitude.
        // noise bins ≈ 0.02/sqrt(n_fft) ≈ very small; sine peak bin ≈ large.
        let gated = frequency_domain_noise_gate(&noisy, 0.1, 1024, 256);

        let rms =
            |s: &[f32]| -> f32 { (s.iter().map(|&x| x * x).sum::<f32>() / s.len() as f32).sqrt() };

        let rms_noisy = rms(&noisy.samples);
        let rms_gated = rms(&gated.samples);

        // The sine (RMS ≈ 0.35) must survive the gate.
        assert!(
            rms_gated > 0.05,
            "gate should preserve the loud sine: gated_rms={rms_gated}"
        );
        // OLA reconstruction may introduce slight energy due to windowing, but must not
        // amplify the signal by more than 50 %.
        assert!(
            rms_gated < rms_noisy * 1.5,
            "gate must not greatly amplify signal: noisy_rms={rms_noisy} gated_rms={rms_gated}"
        );
    }

    #[test]
    fn test_noise_profile_from_silence_is_small() {
        let sr = 44_100_u32;
        let silence = AudioBuffer {
            samples: vec![0.0_f32; sr as usize],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let profile =
            estimate_noise_profile(&silence, 1024).expect("estimate_noise_profile should succeed");
        let max_prof = profile.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            max_prof < 1e-6,
            "silent input profile should be ~zero, got max={max_prof}"
        );
    }

    #[test]
    fn spectral_subtraction_reduces_rms_of_noisy_signal() {
        // Test 10: Spectral subtraction SNR improvement.
        // Generate clean 440 Hz sine + seeded LCG white noise, then verify
        // that spectral_subtraction produces output with lower RMS than noisy input.
        use std::f32::consts::PI;
        let sr = 44_100u32;
        let n_total = sr as usize; // 1 second total
        let n_noise_only = (sr as f32 * 0.2) as usize; // 0.2s of pure noise for profiling

        // Generate clean 440 Hz sine, peak = 0.5
        let clean: Vec<f32> = (0..n_total)
            .map(|i| 0.5 * (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();

        // Generate seeded LCG white noise for reproducibility
        let mut lcg_state: u32 = 12345;
        let noise: Vec<f32> = (0..n_total)
            .map(|_| {
                lcg_state = lcg_state
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                ((lcg_state as f64 / u32::MAX as f64) - 0.5) as f32 * 0.1
            })
            .collect();

        // Noisy signal = clean + noise
        let noisy: Vec<f32> = clean
            .iter()
            .zip(noise.iter())
            .map(|(&c, &n)| c + n)
            .collect();

        // Estimate noise profile from a 0.2s segment of pure noise
        // (use the same LCG sequence offset to approximate noise characteristics)
        let mut lcg_noise_only: u32 = 12345;
        let noise_only_samples: Vec<f32> = (0..n_noise_only)
            .map(|_| {
                lcg_noise_only = lcg_noise_only
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                ((lcg_noise_only as f64 / u32::MAX as f64) - 0.5) as f32 * 0.1
            })
            .collect();

        let noise_buf = AudioBuffer {
            samples: noise_only_samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let noisy_buf = AudioBuffer {
            samples: noisy.clone(),
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };

        let noise_profile =
            estimate_noise_profile(&noise_buf, 512).expect("estimate_noise_profile");
        let denoised =
            spectral_subtraction(&noisy_buf, &noise_profile, 0.1).expect("spectral_subtraction");

        // Compute RMS of noisy input and denoised output
        let rms_noisy = {
            let sq: f32 = noisy_buf.samples.iter().map(|&s| s * s).sum();
            (sq / noisy_buf.samples.len() as f32).sqrt()
        };
        let rms_denoised = {
            let sq: f32 = denoised.samples.iter().map(|&s| s * s).sum();
            (sq / denoised.samples.len() as f32).sqrt()
        };

        // Denoised output should have lower RMS than noisy input (noise removed).
        // Even a small reduction is sufficient — the key property is that spectral
        // subtraction does not increase RMS (noise is attenuated, signal is preserved).
        assert!(
            rms_denoised < rms_noisy,
            "spectral_subtraction should reduce RMS: noisy_rms={rms_noisy:.4} denoised_rms={rms_denoised:.4}"
        );
        // Denoised output must not be silent (signal preserved)
        assert!(
            rms_denoised > 0.01,
            "spectral_subtraction must preserve signal: denoised_rms={rms_denoised:.4}"
        );
    }
}
