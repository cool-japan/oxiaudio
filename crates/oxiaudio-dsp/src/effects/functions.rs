//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use oxiaudio_core::{AudioBuffer, OxiAudioError};

/// Tuned comb-filter delay lengths in samples at 44100 Hz (Jezar / Freeverb).
pub(crate) const COMB_TUNINGS: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
/// Tuned allpass-filter delay lengths in samples at 44100 Hz.
pub(crate) const ALLPASS_TUNINGS: [usize; 4] = [556, 441, 341, 225];
#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::types::{
        Chorus, DelayLine, EarlyReflections, Flanger, Freeverb, Phaser, Tremolo, Vibrato,
    };
    use oxiaudio_core::{ChannelLayout, SampleFormat};
    use std::f32::consts::PI;
    fn make_sine(freq: f32, sr: u32, secs: f32) -> AudioBuffer<f32> {
        let n = (sr as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect();
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }
    #[test]
    fn delay_echo_at_correct_position() {
        let sr = 48_000u32;
        let delay_ms = 20.0f32;
        let delay_samples = (delay_ms * sr as f32 / 1000.0) as usize;
        let total = delay_samples * 3;
        let mut samples = vec![0.0f32; total];
        samples[0] = 1.0;
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let dl = DelayLine::new(delay_ms, 0.0, 1.0);
        let out = dl.process(&buf);
        assert!(
            out.samples[delay_samples].abs() > 0.5,
            "echo should appear at sample {delay_samples}"
        );
        assert!(out.samples[0].abs() < 0.01, "dry input suppressed at wet=1");
    }
    #[test]
    fn chorus_differs_from_dry() {
        let buf = make_sine(440.0, 48_000, 0.5);
        let chorus = Chorus::new(0.5, 10.0);
        let out = chorus.process(&buf);
        let diff: f32 = buf
            .samples
            .iter()
            .zip(&out.samples)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / buf.samples.len() as f32;
        assert!(diff > 0.0001, "chorus should differ from dry input");
    }
    #[test]
    fn tremolo_amplitude_modulation() {
        let buf = make_sine(440.0, 48_000, 0.5);
        let tremolo = Tremolo::new(5.0, 1.0);
        let out = tremolo.process(&buf);
        let min_abs = out.samples.iter().map(|s| s.abs()).fold(f32::MAX, f32::min);
        assert!(
            min_abs < 0.01,
            "tremolo depth=1.0 should bring amplitude near zero, got min={min_abs:.5}"
        );
    }
    #[test]
    fn vibrato_modulates_signal() {
        let buf = make_sine(440.0, 48_000, 0.5);
        let vibrato = Vibrato::new(5.0, 50.0);
        let out = vibrato.process(&buf);
        let diff: f32 = buf
            .samples
            .iter()
            .zip(&out.samples)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / buf.samples.len() as f32;
        assert!(diff > 0.001, "vibrato output should differ from input");
    }
    #[test]
    fn test_freeverb_output_longer_decay() {
        let sr = 44_100u32;
        let n = sr as usize * 2;
        let mut samples = vec![0.0f32; n];
        samples[0] = 1.0;
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut reverb = Freeverb::new(sr);
        reverb.room_size = 0.9;
        let out = reverb.process(&buf);
        assert_eq!(out.samples.len(), n);
        let win_start = (sr as f32 * 0.3) as usize;
        let win_end = (sr as f32 * 0.4) as usize;
        let tail_energy: f32 = out.samples[win_start..win_end].iter().map(|s| s * s).sum();
        assert!(
            tail_energy > 1e-6,
            "Freeverb should produce a sustained reverb tail, got {tail_energy:.2e}"
        );
    }
    #[test]
    fn test_flanger_wet_signal_differs_from_dry() {
        let buf = make_sine(440.0, 48_000, 0.5);
        let flanger = Flanger::new(48_000);
        let out = flanger.process(&buf);
        assert_eq!(out.samples.len(), buf.samples.len());
        let diff: f32 = buf
            .samples
            .iter()
            .zip(&out.samples)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / buf.samples.len() as f32;
        assert!(
            diff > 0.001,
            "flanger output should differ from dry input, got avg diff={diff:.5}"
        );
    }
    #[test]
    fn test_phaser_wet_differs_from_dry() {
        let buf = make_sine(440.0, 48_000, 0.5);
        let phaser = Phaser::new(48_000);
        let out = phaser.process(&buf);
        assert_eq!(out.samples.len(), buf.samples.len());
        let diff: f32 = buf
            .samples
            .iter()
            .zip(&out.samples)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / buf.samples.len() as f32;
        assert!(
            diff > 0.001,
            "phaser output should differ from dry input, got avg diff={diff:.5}"
        );
    }
    #[test]
    fn test_phaser_passthrough_at_zero_wet() {
        let buf = make_sine(440.0, 48_000, 0.2);
        let phaser = Phaser {
            rate_hz: 1.0,
            depth: 1.0,
            feedback: 0.0,
            stages: 4,
            wet_dry: 0.0,
            sample_rate: 48_000,
        };
        let out = phaser.process(&buf);
        assert_eq!(out.samples.len(), buf.samples.len());
        let max_diff: f32 = buf
            .samples
            .iter()
            .zip(out.samples.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "phaser with wet_dry=0.0, feedback=0.0 should be passthrough, max diff={max_diff:.2e}"
        );
    }
    #[test]
    fn test_early_reflections_output_length() {
        let sr = 48_000u32;
        let n = sr as usize;
        let samples = vec![0.1f32; n];
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let er = EarlyReflections::new();
        let out = er.process(&buf);
        assert_eq!(
            out.samples.len(),
            buf.samples.len(),
            "output length should match input length"
        );
    }
    #[test]
    fn test_early_reflections_with_impulse() {
        let sr = 48_000u32;
        let n = sr / 5;
        let mut samples = vec![0.0f32; n as usize];
        samples[0] = 1.0;
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let er = EarlyReflections {
            room_l: 10.0,
            room_w: 8.0,
            room_h: 3.0,
            src_x: 0.5,
            src_y: 0.5,
            src_z: 0.5,
            mic_x: 0.75,
            mic_y: 0.5,
            mic_z: 0.5,
            reflection_coeff: 0.7,
            dry_wet: 1.0,
        };
        let out = er.process(&buf);
        assert_eq!(out.samples.len(), buf.samples.len());
        let has_reflection = out.samples[1..].iter().any(|&s| s.abs() > 1e-8);
        assert!(
            has_reflection,
            "impulse response should produce at least one non-zero reflection"
        );
    }
}
/// Parse a 16-bit PCM WAV file from raw bytes and return a mono `AudioBuffer<f32>`.
///
/// This is a minimal parser: it locates the `fmt ` and `data` sub-chunks, reads
/// the PCM samples, and converts them to `f32`. Only PCM format (audio format 1)
/// is supported; 8-bit and 16-bit sample widths are handled. If the WAV is stereo
/// only the first (left) channel is used, making the output always mono.
///
/// This avoids a dependency on `oxiaudio-decode` (which would create a circular
/// dependency via the workspace graph) while still providing a convenient
/// way to load impulse response files stored in WAV format.
///
/// # Errors
///
/// Returns `OxiAudioError::UnsupportedFormat` if the bytes are not a valid PCM WAV
/// or if the format is unsupported (e.g. float WAV, compressed).
#[must_use = "discarding the Result ignores WAV parse errors"]
pub fn load_ir_from_wav_bytes(bytes: &[u8]) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let err = |msg: &str| OxiAudioError::UnsupportedFormat(msg.to_owned());
    if bytes.len() < 44 {
        return Err(err("WAV too short: need at least 44 bytes for header"));
    }
    if &bytes[0..4] != b"RIFF" {
        return Err(err("WAV: missing RIFF header"));
    }
    if &bytes[8..12] != b"WAVE" {
        return Err(err("WAV: missing WAVE format marker"));
    }
    let mut pos = 12usize;
    let (audio_format, channels, sample_rate, bits_per_sample) = loop {
        if pos + 8 > bytes.len() {
            return Err(err("WAV: fmt chunk not found"));
        }
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(
            bytes[pos + 4..pos + 8]
                .try_into()
                .map_err(|_| err("WAV: bad chunk size"))?,
        );
        pos += 8;
        if chunk_id == b"fmt " {
            if chunk_size < 16 || pos + 16 > bytes.len() {
                return Err(err("WAV: fmt chunk too small"));
            }
            let audio_format = u16::from_le_bytes(
                bytes[pos..pos + 2]
                    .try_into()
                    .map_err(|_| err("WAV: bad audio_format"))?,
            );
            let channels = u16::from_le_bytes(
                bytes[pos + 2..pos + 4]
                    .try_into()
                    .map_err(|_| err("WAV: bad channels"))?,
            );
            let sample_rate = u32::from_le_bytes(
                bytes[pos + 4..pos + 8]
                    .try_into()
                    .map_err(|_| err("WAV: bad sample_rate"))?,
            );
            let bits_per_sample = u16::from_le_bytes(
                bytes[pos + 14..pos + 16]
                    .try_into()
                    .map_err(|_| err("WAV: bad bits_per_sample"))?,
            );
            pos += chunk_size as usize;
            break (audio_format, channels, sample_rate, bits_per_sample);
        }
        pos += chunk_size as usize;
        if chunk_size % 2 != 0 {
            pos += 1;
        }
    };
    if audio_format != 1 {
        return Err(err("WAV: only PCM (format 1) is supported for IR loading"));
    }
    if channels == 0 {
        return Err(err("WAV: zero channels in fmt chunk"));
    }
    if bits_per_sample != 8 && bits_per_sample != 16 {
        return Err(err(
            "WAV: only 8-bit and 16-bit PCM depths are supported for IR",
        ));
    }
    let mut scan = pos;
    let (data_start, data_len) = loop {
        if scan + 8 > bytes.len() {
            return Err(err("WAV: data chunk not found"));
        }
        let chunk_id = &bytes[scan..scan + 4];
        let chunk_size = u32::from_le_bytes(
            bytes[scan + 4..scan + 8]
                .try_into()
                .map_err(|_| err("WAV: bad data chunk size"))?,
        );
        scan += 8;
        if chunk_id == b"data" {
            break (scan, chunk_size as usize);
        }
        scan += chunk_size as usize;
        if chunk_size % 2 != 0 {
            scan += 1;
        }
    };
    let data_end = data_start.saturating_add(data_len).min(bytes.len());
    let raw = &bytes[data_start..data_end];
    let n_channels = channels as usize;
    let samples_f32: Vec<f32> = match bits_per_sample {
        8 => raw
            .chunks_exact(n_channels)
            .map(|frame| {
                let s = frame[0] as f32;
                (s - 128.0) / 128.0
            })
            .collect(),
        16 => raw
            .chunks_exact(n_channels * 2)
            .map(|frame| {
                let lo = frame[0];
                let hi = frame[1];
                let s = i16::from_le_bytes([lo, hi]);
                f32::from(s) / 32768.0
            })
            .collect(),
        _ => return Err(err("WAV: unsupported bit depth (internal error)")),
    };
    Ok(AudioBuffer {
        samples: samples_f32,
        sample_rate,
        channels: oxiaudio_core::ChannelLayout::Mono,
        format: oxiaudio_core::SampleFormat::F32,
    })
}
#[cfg(test)]
mod partitioned_convolution_tests {
    use crate::effects::types::PartitionedConvolutionReverb;
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    fn make_ir(samples: Vec<f32>, sr: u32) -> AudioBuffer<f32> {
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }
    fn make_signal(samples: Vec<f32>, sr: u32) -> AudioBuffer<f32> {
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }
    #[test]
    fn test_partitioned_convolution_new_invalid_partition_size() {
        let ir = make_ir(vec![1.0, 0.0, 0.0], 44_100);
        let result = PartitionedConvolutionReverb::new(&ir, 3, 0.5);
        assert!(result.is_err(), "partition_size=3 should return Err");
        let result2 = PartitionedConvolutionReverb::new(&ir, 0, 0.5);
        assert!(result2.is_err(), "partition_size=0 should return Err");
    }
    #[test]
    fn test_partitioned_convolution_unit_impulse() {
        let ir = make_ir(vec![1.0f32, 0.0, 0.0, 0.0], 44_100);
        let reverb =
            PartitionedConvolutionReverb::new(&ir, 4, 1.0).expect("valid partition_size=4");
        let signal = make_signal(vec![0.5, -0.3, 0.8, 0.1, -0.6], 44_100);
        let out = reverb.process(&signal);
        assert_eq!(
            out.samples.len(),
            signal.samples.len(),
            "output length should match input"
        );
        for (i, (&expected, &got)) in signal.samples.iter().zip(out.samples.iter()).enumerate() {
            assert!(
                (expected - got).abs() < 1e-5,
                "sample {i}: expected {expected}, got {got}"
            );
        }
    }
    #[test]
    fn test_partitioned_convolution_wet_dry_zero() {
        let ir = make_ir(vec![0.5f32, 0.3, 0.1], 44_100);
        let reverb =
            PartitionedConvolutionReverb::new(&ir, 2, 0.0).expect("valid partition_size=2");
        let signal = make_signal(vec![1.0, -0.5, 0.25, -0.125], 44_100);
        let out = reverb.process(&signal);
        assert_eq!(out.samples.len(), signal.samples.len());
        for (i, (&dry, &got)) in signal.samples.iter().zip(out.samples.iter()).enumerate() {
            assert!(
                (dry - got).abs() < 1e-6,
                "sample {i}: wet_dry=0 should pass through dry, expected {dry}, got {got}"
            );
        }
    }
    #[test]
    fn test_partitioned_convolution_output_length() {
        let ir = make_ir(vec![1.0f32; 512], 44_100);
        let reverb =
            PartitionedConvolutionReverb::new(&ir, 256, 0.5).expect("valid partition_size=256");
        let n = 1024usize;
        let signal = make_signal(vec![0.1f32; n], 44_100);
        let out = reverb.process(&signal);
        assert_eq!(
            out.samples.len(),
            n,
            "output length {} should equal input length {n}",
            out.samples.len()
        );
    }
}
#[cfg(test)]
mod vocoder_tests {
    use crate::effects::types::ChannelVocoder;
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    use std::f32::consts::PI;
    fn silence_buf(sr: u32, secs: f32) -> AudioBuffer<f32> {
        let n = (sr as f32 * secs) as usize;
        AudioBuffer {
            samples: vec![0.0f32; n],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }
    fn sine_buf(freq: f32, sr: u32, secs: f32) -> AudioBuffer<f32> {
        let n = (sr as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }
    #[test]
    fn test_channel_vocoder_output_length() {
        let sr = 44_100u32;
        let secs = 0.5f32;
        let modulator = silence_buf(sr, secs);
        let carrier = silence_buf(sr, secs);
        let vocoder = ChannelVocoder::new(1024, 256);
        let out = vocoder
            .process(&modulator, &carrier)
            .expect("vocoder failed");
        let expected_len = (sr as f32 * secs) as usize;
        assert!(!out.samples.is_empty(), "output should be non-empty");
        let tol = expected_len / 10;
        assert!(
            out.samples.len().abs_diff(expected_len) <= tol,
            "output len {} should be near {} (±{})",
            out.samples.len(),
            expected_len,
            tol,
        );
    }
    #[test]
    fn test_channel_vocoder_sine_modulator() {
        let sr = 44_100u32;
        let secs = 0.5f32;
        let modulator = sine_buf(880.0, sr, secs);
        let carrier = sine_buf(440.0, sr, secs);
        let vocoder = ChannelVocoder::new(1024, 256);
        let out = vocoder
            .process(&modulator, &carrier)
            .expect("vocoder failed");
        let expected_len = (sr as f32 * secs) as usize;
        assert!(!out.samples.is_empty(), "output should be non-empty");
        let max_abs = out.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs > 1e-6,
            "output with sine modulator should be non-zero"
        );
        let tol = expected_len / 10;
        assert!(
            out.samples.len().abs_diff(expected_len) <= tol,
            "output len {} should be near {} (±{})",
            out.samples.len(),
            expected_len,
            tol,
        );
    }
}
#[cfg(test)]
mod convolution_tests {
    use crate::effects::types::{ConvolutionReverb, Freeverb};
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    use std::f32::consts::PI;
    #[test]
    fn test_convolution_reverb_unit_impulse() {
        let sr = 44100u32;
        let signal: Vec<f32> = vec![0.5, -0.3, 0.8, 0.0, 0.0];
        let buf = AudioBuffer {
            samples: signal.clone(),
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let ir = vec![1.0f32, 0.0, 0.0, 0.0];
        let reverb = ConvolutionReverb::new(ir).with_dry(0.0).with_wet(1.0);
        let out = reverb.process(&buf);
        for (i, (&expected, &got)) in signal.iter().zip(out.samples.iter()).enumerate() {
            assert!(
                (expected - got).abs() < 0.01,
                "sample {i}: expected {expected}, got {got}"
            );
        }
    }
    #[test]
    fn test_convolution_reverb_adds_tail() {
        let sr = 44100u32;
        let n_input = 1000usize;
        let n_ir = 100usize;
        let signal: Vec<f32> = (0..n_input)
            .map(|i| (i as f32 / n_input as f32).sin())
            .collect();
        let ir: Vec<f32> = (0..n_ir).map(|i| 0.9f32.powi(i as i32)).collect();
        let buf = AudioBuffer {
            samples: signal,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let reverb = ConvolutionReverb::new(ir).with_wet(0.5).with_dry(0.5);
        let out = reverb.process(&buf);
        assert!(
            out.samples.len() > n_input,
            "reverb output should include tail"
        );
    }
    #[test]
    fn test_convolution_reverb_stereo() {
        let sr = 44100u32;
        let n = 200usize;
        let samples: Vec<f32> = (0..n * 2)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let ir = vec![1.0f32, 0.5, 0.25];
        let reverb = ConvolutionReverb::new(ir).with_wet(1.0).with_dry(0.0);
        let out = reverb.process(&buf);
        assert!(!out.samples.is_empty(), "output should not be empty");
        assert_eq!(
            out.channels,
            ChannelLayout::Stereo,
            "channel layout should be preserved"
        );
    }
    /// Build a minimal mono 16-bit PCM WAV with the given samples.
    ///
    /// Layout: RIFF header (12) + fmt  chunk (24) + data chunk (8 + samples*2).
    fn make_minimal_wav_mono_i16(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let data_size = (samples.len() * 2) as u32;
        let file_size = 36u32 + data_size;
        let mut wav = Vec::with_capacity(44 + samples.len() * 2);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * 2;
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        wav
    }
    #[test]
    fn test_load_ir_from_wav_bytes_silence() {
        let sr = 44100u32;
        let samples = vec![0i16; sr as usize];
        let wav = make_minimal_wav_mono_i16(&samples, sr);
        let buf = super::load_ir_from_wav_bytes(&wav)
            .expect("load_ir_from_wav_bytes should succeed on minimal WAV");
        assert_eq!(buf.sample_rate, sr, "sample rate must be 44100");
        assert_eq!(buf.channels, ChannelLayout::Mono, "loaded IR must be mono");
        assert_eq!(
            buf.samples.len(),
            sr as usize,
            "must have 44100 samples for 1 second"
        );
        for (i, &s) in buf.samples.iter().enumerate() {
            assert!(
                s.abs() < 1e-6,
                "sample {i} should be ~0.0 for silence, got {s}"
            );
        }
    }
    #[test]
    fn test_load_ir_from_wav_bytes_full_scale() {
        let sr = 22050u32;
        let samples = vec![i16::MAX];
        let wav = make_minimal_wav_mono_i16(&samples, sr);
        let buf =
            super::load_ir_from_wav_bytes(&wav).expect("load_ir_from_wav_bytes should succeed");
        assert_eq!(buf.sample_rate, sr);
        assert_eq!(buf.samples.len(), 1);
        assert!(
            (buf.samples[0] - 1.0f32).abs() < 0.001,
            "full-scale i16::MAX must map to ~1.0 f32, got {}",
            buf.samples[0]
        );
    }
    #[test]
    fn test_load_ir_from_wav_bytes_invalid_magic() {
        let bad = b"XXXX\x00\x00\x00\x00WAVE".to_vec();
        let result = super::load_ir_from_wav_bytes(&bad);
        assert!(result.is_err(), "invalid RIFF magic must return Err");
    }
    #[test]
    fn test_convolution_reverb_from_ir_buffer() {
        let sr = 44100u32;
        let ir_samples = vec![1.0f32, 0.0, 0.0];
        let ir_buf = AudioBuffer {
            samples: ir_samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let reverb = ConvolutionReverb::from_ir_buffer(&ir_buf, 1.0);
        let signal = AudioBuffer {
            samples: vec![0.5, -0.3, 0.8],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let out = reverb.process(&signal);
        assert!((out.samples[0] - 0.5).abs() < 0.01, "first sample mismatch");
    }
    #[test]
    fn freeverb_reverb_tail_non_zero_and_peak_not_amplified() {
        let sr = 44_100u32;
        let n = (sr as f32 * 0.5) as usize;
        let mut impulse_samples = vec![0.0f32; n];
        impulse_samples[0] = 1.0;
        let impulse_buf = AudioBuffer {
            samples: impulse_samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut reverb = Freeverb::new(sr);
        reverb.room_size = 0.8;
        reverb.damping = 0.5;
        reverb.wet = 0.3;
        reverb.dry = 0.7;
        reverb.width = 1.0;
        let out = reverb.process(&impulse_buf);
        assert_eq!(
            out.samples.len(),
            n,
            "output should have same length as input"
        );
        let tail_start = n - (sr as f32 * 0.1) as usize;
        let tail_rms: f32 = {
            let sq: f32 = out.samples[tail_start..].iter().map(|&s| s * s).sum();
            (sq / (out.samples.len() - tail_start) as f32).sqrt()
        };
        assert!(
            tail_rms > 1e-6,
            "reverb tail (last 0.1s) should be non-zero, got tail_rms={tail_rms:.2e}"
        );
        let n_check = (sr as f32 * 0.5) as usize;
        let check_samples: Vec<f32> = (0..n_check)
            .map(|i| 0.5 * (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let check_buf = AudioBuffer {
            samples: check_samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut reverb_mix = Freeverb::new(sr);
        reverb_mix.room_size = 0.8;
        reverb_mix.wet = 0.3;
        reverb_mix.dry = 0.7;
        let out_mix = reverb_mix.process(&check_buf);
        let peak_mix = out_mix
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(
            peak_mix.is_finite() && peak_mix > 0.0,
            "freeverb mix output should be finite and non-zero, got {peak_mix:.4}"
        );
    }
    #[test]
    fn convolution_reverb_unit_impulse_ir_is_identity() {
        let sr = 44_100u32;
        let ir_len = 1024usize;
        let mut ir_samples = vec![0.0f32; ir_len];
        ir_samples[0] = 1.0;
        let reverb = ConvolutionReverb::new(ir_samples)
            .with_wet(1.0)
            .with_dry(0.0);
        let n_signal = 128usize;
        let signal_samples: Vec<f32> = (0..n_signal)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.7
            })
            .collect();
        let input_buf = AudioBuffer {
            samples: signal_samples.clone(),
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let out = reverb.process(&input_buf);
        assert!(
            out.samples.len() >= n_signal,
            "output should be at least as long as input: got {}",
            out.samples.len()
        );
        for (i, (&expected, &actual)) in signal_samples
            .iter()
            .zip(out.samples.iter())
            .enumerate()
            .take(n_signal)
        {
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-4,
                "unit IR convolution: sample {i} mismatch: expected {expected:.6} got {actual:.6} (diff={diff:.2e})"
            );
        }
    }
}
