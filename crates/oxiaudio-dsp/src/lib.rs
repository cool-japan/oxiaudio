#![forbid(unsafe_code)]

#[cfg(feature = "dasp")]
pub mod dasp_adapter;

pub mod biquad;
pub mod chain;
pub mod dither;
pub mod dynamics;
pub mod effects;
pub mod filters;
pub mod filters_fir;
pub mod filters_iir;
pub mod loudness;
pub mod noise;
pub mod pitch;
pub mod pvocoder;
pub mod rhythm;
pub mod segment;
pub mod spectral;
pub mod stereo;

pub use biquad::{BiquadFilter, ParametricEq};
pub use chain::DspChain;
pub use dither::{apply_noise_shaped_dither, apply_tpdf_dither};
pub use dynamics::{
    BandSettings, Compressor, DeEsser, Expander, Limiter, MultibandCompressor, NoiseGate,
};
pub use effects::{
    ChannelVocoder, Chorus, ConvolutionReverb, DelayLine, EarlyReflections, Flanger, Freeverb,
    PartitionedConvolutionReverb, Phaser, Tremolo, Vibrato,
};
pub use filters::{
    butterworth_highpass, butterworth_lowpass, chebyshev1_highpass, chebyshev1_lowpass,
    chebyshev2_highpass, chebyshev2_lowpass, elliptic_highpass, elliptic_lowpass, Cascade,
    Chebyshev2Filter, EllipticFilter, FirFilter, FirWindow,
};
pub use loudness::{
    k_weight, loudness_integrated, loudness_momentary, loudness_momentary_windowed, loudness_range,
    normalize_to_lufs, true_peak, PeakMeter, RmsMeter,
};
pub use noise::{
    estimate_noise_profile, frequency_domain_noise_gate, spectral_subtraction, wiener_filter,
};
pub use pitch::{
    detect_pitch_autocorr, detect_pitch_pyin, detect_pitch_yin, detect_pitch_yin_simple,
    PitchFrame, PitchTracker,
};
pub use pvocoder::{pitch_shift_pv, time_stretch};
pub use rhythm::{
    complex_domain_onset, detect_downbeats, detect_onsets, estimate_tempo, onset_strength_hfc,
    onset_strength_spectral_flux, pick_onset_peaks, TempoEstimate,
};
pub use segment::silence_split;
pub use spectral::{
    chromagram, chromagram_normalized, harmonic_ratio, istft, melspectrogram, mfcc, pitch_shift,
    short_time_energy, spectral_bandwidth, spectral_centroid, spectral_contrast,
    spectral_crest_factor, spectral_entropy, spectral_flatness, spectral_flux, spectral_rolloff,
    stft, stft_multichannel, tonnetz, zero_crossing_rate, Complex, StftOutput, WindowFn,
};
pub use stereo::{ms_decode, ms_encode};

use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError, SampleFormat};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::audioadapter_buffers::owned::InterleavedOwned;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction as RubatoWindow,
};

/// Resample `buf` to `target_rate` using a high-quality sinc interpolation resampler.
///
/// SIMD acceleration (SSE2/AVX/NEON) is chosen automatically at runtime.
#[must_use = "returns the resampled AudioBuffer at target_rate"]
pub fn resample(
    buf: &AudioBuffer<f32>,
    target_rate: u32,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if buf.sample_rate == target_rate {
        return Ok(AudioBuffer {
            samples: buf.samples.clone(),
            sample_rate: target_rate,
            channels: buf.channels,
            format: buf.format,
        });
    }

    let n_channels = buf.channels.channel_count();

    let in_rate = buf.sample_rate as f64;
    let out_rate = target_rate as f64;

    // Number of frames (samples per channel)
    let n_frames = buf.samples.len() / n_channels;

    if n_frames == 0 {
        return Ok(AudioBuffer {
            samples: Vec::new(),
            sample_rate: target_rate,
            channels: buf.channels,
            format: buf.format,
        });
    }

    // chunk_size must be at least sinc_len to avoid resampler construction error.
    // Use n_frames so we do a single one-shot resample pass.
    let sinc_len = 256usize;
    let chunk_size = n_frames.max(sinc_len * 2);

    let params = SincInterpolationParameters {
        sinc_len,
        f_cutoff: 0.95,
        oversampling_factor: 128,
        interpolation: SincInterpolationType::Cubic,
        window: RubatoWindow::Blackman,
    };

    let ratio = out_rate / in_rate;
    let mut resampler = Async::<f32>::new_sinc(
        ratio,
        1.0, // fixed ratio — no dynamic rate change needed
        &params,
        chunk_size,
        n_channels,
        FixedAsync::Input,
    )
    .map_err(|e| OxiAudioError::UnsupportedFormat(e.to_string()))?;

    // Build interleaved input adapter
    let input_adapter = InterleavedSlice::new(&buf.samples, n_channels, n_frames)
        .map_err(|e| OxiAudioError::UnsupportedFormat(e.to_string()))?;

    // Calculate required output length
    let out_frames = resampler.process_all_needed_output_len(n_frames);

    // Allocate output buffer
    let mut output_buf = InterleavedOwned::new(0.0f32, n_channels, out_frames);

    // Resample all frames at once
    let (_in_consumed, out_written) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_buf, n_frames, None)
        .map_err(|e| OxiAudioError::UnsupportedFormat(e.to_string()))?;

    // Extract the output samples, trimmed to actual written frames
    let mut out_samples = output_buf.take_data();
    out_samples.truncate(out_written * n_channels);

    Ok(AudioBuffer {
        samples: out_samples,
        sample_rate: target_rate,
        channels: buf.channels,
        format: SampleFormat::F32,
    })
}

/// Apply a gain in decibels to all samples in the buffer.
///
/// Positive dB values amplify; negative dB values attenuate.
/// 0 dB = unity gain; +6 dB ≈ 2× amplitude; -6 dB ≈ 0.5× amplitude.
///
/// Uses `iter_mut().for_each()` to hint LLVM auto-vectorization (SSE/AVX/NEON).
pub fn gain(buf: &mut AudioBuffer<f32>, db: f32) {
    let linear = 10_f32.powf(db / 20.0);
    buf.samples.iter_mut().for_each(|s| *s *= linear);
}

/// Apply a linear gain factor in-place without any dB conversion.
///
/// Prefer this over [`gain`] in hot loops to avoid the `10^(db/20)` transcendental.
/// LLVM auto-vectorizes this loop on x86_64 (SSE2/AVX) and aarch64 (NEON).
pub fn gain_inplace(buf: &mut AudioBuffer<f32>, factor: f32) {
    buf.samples.iter_mut().for_each(|s| *s *= factor);
}

/// Average all channels into a single mono channel.
pub fn mix_to_mono(buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
    let n_channels = buf.channels.channel_count();
    if n_channels == 1 {
        return AudioBuffer {
            samples: buf.samples.clone(),
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        };
    }
    let n_frames = buf.samples.len() / n_channels;
    let mut mono = Vec::with_capacity(n_frames);
    for frame in 0..n_frames {
        let sum: f32 = (0..n_channels)
            .map(|c| buf.samples[frame * n_channels + c])
            .sum();
        mono.push(sum / n_channels as f32);
    }
    AudioBuffer {
        samples: mono,
        sample_rate: buf.sample_rate,
        channels: ChannelLayout::Mono,
        format: buf.format,
    }
}

/// De-interleave into one `AudioBuffer<f32>` per channel (all Mono).
pub fn split_channels(buf: &AudioBuffer<f32>) -> Vec<AudioBuffer<f32>> {
    let n_channels = buf.channels.channel_count();
    let n_frames = buf.samples.len() / n_channels;
    (0..n_channels)
        .map(|c| {
            let samples: Vec<f32> = (0..n_frames)
                .map(|f| buf.samples[f * n_channels + c])
                .collect();
            AudioBuffer {
                samples,
                sample_rate: buf.sample_rate,
                channels: ChannelLayout::Mono,
                format: buf.format,
            }
        })
        .collect()
}

/// Peak-normalize in-place to `target_db` dBFS. Silent buffers are unchanged.
///
/// Uses `iter_mut().for_each()` to hint LLVM auto-vectorization (SSE/AVX/NEON).
pub fn normalize(buf: &mut AudioBuffer<f32>, target_db: f32) {
    let max_abs = buf.samples.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
    if max_abs == 0.0 {
        return;
    }
    let scale = 10f32.powf(target_db / 20.0) / max_abs;
    buf.samples.iter_mut().for_each(|s| *s *= scale);
}

/// Peak-normalize in-place to a linear `target_peak` amplitude (0.0–1.0).
///
/// Silent buffers (peak ≤ 1e-10) are left unchanged.
/// Prefer this over [`normalize`] in hot paths to avoid dB conversion.
/// LLVM auto-vectorizes the inner loops on x86_64 (SSE2/AVX) and aarch64 (NEON).
pub fn normalize_inplace(buf: &mut AudioBuffer<f32>, target_peak: f32) {
    let peak = buf.samples.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
    if peak <= 1e-10 {
        return;
    }
    let scale = target_peak / peak;
    buf.samples.iter_mut().for_each(|s| *s *= scale);
}

/// Remove leading and trailing silent frames. A frame is silent when ALL channel
/// samples are below `threshold_db` dBFS in absolute value.
pub fn trim_silence(buf: &AudioBuffer<f32>, threshold_db: f32) -> AudioBuffer<f32> {
    let n_channels = buf.channels.channel_count();
    let thr = 10f32.powf(threshold_db / 20.0);
    let n_frames = buf.samples.len() / n_channels;

    let frame_is_silent = |frame: usize| -> bool {
        (0..n_channels).all(|c| buf.samples[frame * n_channels + c].abs() < thr)
    };

    let start = (0..n_frames)
        .find(|&f| !frame_is_silent(f))
        .unwrap_or(n_frames);
    let end = (0..n_frames)
        .rev()
        .find(|&f| !frame_is_silent(f))
        .map(|f| f + 1)
        .unwrap_or(0);

    if start >= end {
        return AudioBuffer {
            samples: vec![],
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        };
    }

    let samples = buf.samples[start * n_channels..end * n_channels].to_vec();
    AudioBuffer {
        samples,
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        format: buf.format,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a stereo sine-wave `AudioBuffer<f32>` (interleaved L/R).
    fn stereo_sine_buf(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
        let n_frames = (sample_rate as f32 * duration_secs) as usize;
        let mut samples = Vec::with_capacity(n_frames * 2);
        for i in 0..n_frames {
            let t = i as f32 / sample_rate as f32;
            let s = (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.5;
            samples.push(s); // left
            samples.push(s); // right (identical)
        }
        AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_resample_sinc_downsample() {
        // 48k → 44100 stereo: expected output frames ≈ 44100 (1s * 44100)
        let buf = stereo_sine_buf(440.0, 48_000, 1.0);
        let out = resample(&buf, 44_100).expect("sinc resample failed");
        assert_eq!(out.sample_rate, 44_100);
        assert_eq!(out.channels, ChannelLayout::Stereo);
        let expected_frames = 44_100usize;
        let actual_frames = out.samples.len() / 2;
        let tolerance = 1_500usize; // resampler intro/flush delay
        assert!(
            actual_frames.abs_diff(expected_frames) <= tolerance,
            "expected ~{expected_frames} frames, got {actual_frames}"
        );
    }

    #[test]
    fn test_mix_to_mono_cancellation() {
        // L=1.0, R=-1.0 → average = 0.0
        let buf = AudioBuffer {
            samples: vec![1.0f32, -1.0, 1.0, -1.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let mono = mix_to_mono(&buf);
        assert_eq!(mono.channels, ChannelLayout::Mono);
        for &s in &mono.samples {
            assert!(s.abs() < 1e-6, "expected 0.0, got {s}");
        }
    }

    #[test]
    fn test_split_channels_stereo() {
        let buf = AudioBuffer {
            samples: vec![1.0f32, -1.0, 0.5, -0.5],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let channels = split_channels(&buf);
        assert_eq!(channels.len(), 2);
        assert_eq!(channels[0].samples, vec![1.0f32, 0.5]); // left
        assert_eq!(channels[1].samples, vec![-1.0f32, -0.5]); // right
        assert_eq!(channels[0].channels, ChannelLayout::Mono);
        assert_eq!(channels[1].channels, ChannelLayout::Mono);
    }

    #[test]
    fn test_normalize_peak() {
        let mut buf = AudioBuffer {
            samples: vec![0.5f32, -0.25, 0.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        normalize(&mut buf, 0.0); // target 0 dBFS
                                  // peak was 0.5, scale = 1.0 / 0.5 = 2.0
        assert!(
            (buf.samples[0] - 1.0).abs() < 1e-6,
            "expected 1.0, got {}",
            buf.samples[0]
        );
        assert!(
            (buf.samples[1] - (-0.5)).abs() < 1e-6,
            "expected -0.5, got {}",
            buf.samples[1]
        );
        assert!(buf.samples[2].abs() < 1e-6);
    }

    #[test]
    fn test_normalize_silent_no_panic() {
        let mut buf = AudioBuffer {
            samples: vec![0.0f32; 100],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        normalize(&mut buf, 0.0); // must not panic
        assert!(buf.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_trim_silence() {
        // stereo: [0,0, 0,0, 1,2, 3,4, 0,0] — 5 frames, silent at start and end
        let buf = AudioBuffer {
            samples: vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let trimmed = trim_silence(&buf, -60.0); // -60 dBFS ≈ 0.001 threshold
                                                 // Should keep frames 2 and 3: [1.0, 2.0, 3.0, 4.0]
        assert_eq!(trimmed.samples, vec![1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(trimmed.channels, ChannelLayout::Stereo);
    }
}
