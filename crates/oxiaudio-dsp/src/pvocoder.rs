//! Phase vocoder: high-quality time-stretching and pitch shifting.
//!
//! The phase vocoder (PV) operates in the STFT domain, propagating phase
//! coherently across frames so that time-stretched synthesis is artifact-free.
//!
//! # Algorithm overview
//!
//! 1. **Analysis**: Compute STFT of the input with analysis hop `hop_a`.
//! 2. **Phase propagation**: For each frame/bin, compute the instantaneous
//!    frequency from the phase difference, accounting for the expected advance,
//!    then accumulate a synthesis phase at the synthesis hop rate `hop_s`.
//! 3. **Synthesis**: Build a new spectrogram using magnitude from analysis but
//!    phase from the accumulator, then reconstruct via ISTFT with `hop_s`.

use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError, SampleFormat};

use crate::spectral::{istft, stft, Complex, StftOutput, WindowFn};
use crate::{resample, split_channels};

/// Wrap a phase value into [-π, π].
#[inline]
fn wrap_to_pi(x: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    x - two_pi * (x / two_pi).round()
}

/// Time-stretch `buf` by `ratio` (>1 = slower, <1 = faster) without changing pitch.
///
/// Internally performs phase-vocoder analysis at `hop_a` and synthesis at
/// `hop_s = (hop_a as f64 * ratio).round() as usize`.
///
/// # Errors
///
/// Returns `OxiAudioError` if the STFT or ISTFT operations fail.
#[must_use = "returns the time-stretched AudioBuffer"]
pub fn time_stretch(
    buf: &AudioBuffer<f32>,
    ratio: f64,
    n_fft: usize,
    hop_a: usize,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if buf.samples.is_empty() {
        return Ok(AudioBuffer {
            samples: Vec::new(),
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        });
    }

    let n_channels = buf.channels.channel_count();

    if n_channels == 1 {
        time_stretch_mono(buf, ratio, n_fft, hop_a)
    } else {
        // Split, process each channel, re-interleave.
        let channels = split_channels(buf);
        let processed: Result<Vec<AudioBuffer<f32>>, OxiAudioError> = channels
            .iter()
            .map(|ch| time_stretch_mono(ch, ratio, n_fft, hop_a))
            .collect();
        let processed = processed?;
        interleave_channels(&processed, buf.sample_rate, buf.channels, buf.format)
    }
}

/// Phase-vocoder pitch shift: time-stretch by `1/ratio` then resample back to
/// the original length, which shifts pitch by `semitones` semitones.
///
/// Positive `semitones` = pitch up, negative = pitch down.
///
/// # Errors
///
/// Returns `OxiAudioError` if the underlying DSP operations fail.
#[must_use = "returns the pitch-shifted AudioBuffer"]
pub fn pitch_shift_pv(
    buf: &AudioBuffer<f32>,
    semitones: f32,
    n_fft: usize,
    hop_a: usize,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if semitones == 0.0 {
        return Ok(AudioBuffer {
            samples: buf.samples.clone(),
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        });
    }

    let ratio = 2.0_f64.powf(f64::from(semitones) / 12.0);
    // Time-stretch by the reciprocal: output is 1/ratio × original length.
    let stretched = time_stretch(buf, 1.0 / ratio, n_fft, hop_a)?;

    // Resample the stretched signal back to original sample-rate (scaled).
    // Resampling at a rate ratio equal to `ratio` will compress the signal
    // back to approximately the original length while shifting pitch up/down.
    let target_rate = (buf.sample_rate as f64 * ratio).round() as u32;
    let target_rate = target_rate.max(1);

    let mut resampled = resample(&stretched, target_rate)?;
    // Re-label as the original sample rate (the resampling already adjusted
    // the time axis; relabelling does not change the samples).
    resampled.sample_rate = buf.sample_rate;

    Ok(AudioBuffer {
        samples: resampled.samples,
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        format: SampleFormat::F32,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Core phase-vocoder time-stretch for a mono `AudioBuffer<f32>`.
fn time_stretch_mono(
    buf: &AudioBuffer<f32>,
    ratio: f64,
    n_fft: usize,
    hop_a: usize,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let hop_s = ((hop_a as f64) * ratio).round() as usize;
    let hop_s = hop_s.max(1);

    // Compute STFT with analysis hop.
    let stft_out = stft(buf, n_fft, hop_a, WindowFn::Hann)?;
    let frames = &stft_out.frames;

    if frames.is_empty() {
        let out_len = (buf.samples.len() as f64 * ratio).round() as usize;
        return Ok(AudioBuffer {
            samples: vec![0.0_f32; out_len],
            sample_rate: buf.sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        });
    }

    let n_bins = frames[0].len(); // typically n_fft/2 + 1

    let two_pi = std::f64::consts::TAU;
    // Expected phase advance per bin per analysis hop.
    // For bin k: expected_advance[k] = 2π * k * hop_a / n_fft
    let expected_advances: Vec<f64> = (0..n_bins)
        .map(|k| two_pi * k as f64 * hop_a as f64 / n_fft as f64)
        .collect();

    // Phase accumulators for synthesis.
    let mut phi_accum: Vec<f64> = frames[0]
        .iter()
        .map(|c| {
            let re = f64::from(c.re);
            let im = f64::from(c.im);
            im.atan2(re)
        })
        .collect();

    // Previous analysis phases (initialised to first frame).
    let mut prev_phase: Vec<f64> = phi_accum.clone();

    // Build the synthesis frame list.
    let mut new_frames: Vec<Vec<Complex<f32>>> = Vec::with_capacity(frames.len());

    // First frame: use unmodified phase from input.
    let first_frame: Vec<Complex<f32>> = frames[0]
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let mag = f64::from(c.norm());
            let ph = phi_accum[k];
            Complex::new((mag * ph.cos()) as f32, (mag * ph.sin()) as f32)
        })
        .collect();
    new_frames.push(first_frame);

    // Subsequent frames: propagate phase.
    for frame in frames.iter().skip(1) {
        let mut synth_frame = Vec::with_capacity(n_bins);

        for (k, c) in frame.iter().enumerate() {
            let re = f64::from(c.re);
            let im = f64::from(c.im);
            let cur_phase = im.atan2(re);

            // Phase difference from previous frame.
            let delta = cur_phase - prev_phase[k];

            // Deviation from expected advance.
            let deviation = wrap_to_pi(delta - expected_advances[k]);

            // Instantaneous frequency (as phase advance per analysis sample).
            let inst_freq = expected_advances[k] + deviation;

            // Accumulate synthesis phase using the synthesis hop.
            phi_accum[k] += inst_freq * hop_s as f64 / hop_a as f64;

            prev_phase[k] = cur_phase;

            let mag = f64::from(c.norm());
            let ph = phi_accum[k];
            synth_frame.push(Complex::new(
                (mag * ph.cos()) as f32,
                (mag * ph.sin()) as f32,
            ));
        }

        new_frames.push(synth_frame);
    }

    // Build a synthetic StftOutput with synthesis hop for reconstruction.
    let synth_stft = StftOutput {
        frames: new_frames,
        sample_rate: stft_out.sample_rate,
        hop_size: hop_s,
        window: WindowFn::Hann,
    };

    // Expected output length.
    let orig_len = buf.samples.len();
    let out_len = (orig_len as f64 * ratio).round() as usize;

    let mut result = istft(&synth_stft, out_len)?;
    result.channels = ChannelLayout::Mono;
    Ok(result)
}

/// Interleave per-channel mono buffers back into a multi-channel buffer.
fn interleave_channels(
    channels: &[AudioBuffer<f32>],
    sample_rate: u32,
    layout: ChannelLayout,
    format: SampleFormat,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if channels.is_empty() {
        return Ok(AudioBuffer {
            samples: Vec::new(),
            sample_rate,
            channels: layout,
            format,
        });
    }

    let n_ch = channels.len();
    // Use the minimum length so all slices are valid.
    let n_frames = channels.iter().map(|c| c.samples.len()).min().unwrap_or(0);

    let mut out = Vec::with_capacity(n_frames * n_ch);
    for f in 0..n_frames {
        for ch in channels {
            out.push(ch.samples[f]);
        }
    }

    Ok(AudioBuffer {
        samples: out,
        sample_rate,
        channels: layout,
        format,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{ChannelLayout, SampleFormat};

    fn sine_buf(freq: f32, sr: u32, dur: f32) -> AudioBuffer<f32> {
        let n = (sr as f32 * dur) as usize;
        AudioBuffer {
            samples: (0..n)
                .map(|i| {
                    let t = i as f32 / sr as f32;
                    (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
                })
                .collect(),
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_time_stretch_ratio_1_approx_identity() {
        let buf = sine_buf(440.0, 44100, 0.5);
        let out = time_stretch(&buf, 1.0, 2048, 512).unwrap();
        // Length should be approximately original
        let diff = (out.samples.len() as i64 - buf.samples.len() as i64).abs();
        assert!(diff < 1000, "length diff too large: {diff}");
    }

    #[test]
    fn test_time_stretch_double_length() {
        let buf = sine_buf(440.0, 44100, 0.5);
        let out = time_stretch(&buf, 2.0, 2048, 512).unwrap();
        // Should be roughly 2x longer
        let expected = buf.samples.len() * 2;
        let actual = out.samples.len();
        let tolerance = expected / 5;
        assert!(
            (actual as i64 - expected as i64).abs() < tolerance as i64,
            "expected ~{expected} samples, got {actual}"
        );
    }

    #[test]
    fn test_pitch_shift_pv_up_12_semitones() {
        // Shifting up one octave: output should have same length, approx same energy
        let buf = sine_buf(440.0, 44100, 0.5);
        let out = pitch_shift_pv(&buf, 12.0, 2048, 512).unwrap();
        assert_eq!(out.sample_rate, buf.sample_rate);
        // Output not silent
        let rms: f32 =
            (out.samples.iter().map(|&s| s * s).sum::<f32>() / out.samples.len() as f32).sqrt();
        assert!(rms > 0.01, "output was silent");
    }

    #[test]
    fn test_pitch_shift_pv_zero_semitones() {
        // 0 semitones: output should be approximately the input
        let buf = sine_buf(440.0, 44100, 0.5);
        let out = pitch_shift_pv(&buf, 0.0, 2048, 512).unwrap();
        assert!(!out.samples.is_empty());
    }

    #[test]
    fn test_time_stretch_doubles_length() {
        // A 0.5-second sine at 44100 Hz, stretched 2x, should be ~1 second
        let sr = 44_100u32;
        let n = (sr as f32 * 0.5) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let stretched = time_stretch(&buf, 2.0, 2048, 512).expect("time_stretch 2x");
        // Output should be approximately 2x as long
        let expected_len = (buf.samples.len() as f32 * 2.0) as usize;
        let actual_len = stretched.samples.len();
        let ratio = actual_len as f32 / expected_len as f32;
        assert!(
            ratio > 0.9 && ratio < 1.1,
            "stretched length {actual_len} should be ~{expected_len}"
        );
    }

    #[test]
    fn test_time_stretch_half_length() {
        let sr = 44_100u32;
        let n = (sr as f32 * 1.0) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let compressed = time_stretch(&buf, 0.5, 2048, 512).expect("time_stretch 0.5x");
        let expected_len = buf.samples.len() / 2;
        let actual_len = compressed.samples.len();
        let ratio = actual_len as f32 / expected_len as f32;
        assert!(
            ratio > 0.85 && ratio < 1.15,
            "compressed length {actual_len} should be ~{expected_len}"
        );
    }

    #[test]
    fn test_time_stretch_preserves_sample_rate() {
        let sr = 48_000u32;
        let n = 4096usize;
        let buf = AudioBuffer {
            samples: vec![0.1f32; n],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let out = time_stretch(&buf, 1.5, 1024, 256).expect("time_stretch");
        assert_eq!(out.sample_rate, sr);
    }

    #[test]
    fn test_pitch_shift_pv_preserves_length() {
        let sr = 44_100u32;
        let n = (sr as f32 * 0.5) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let shifted = pitch_shift_pv(&buf, 2.0, 2048, 512).expect("pitch_shift_pv +2 semitones");
        // Output length should equal input length (resampled back)
        assert_eq!(
            shifted.samples.len(),
            buf.samples.len(),
            "pitch_shift_pv should preserve frame count"
        );
    }
}
