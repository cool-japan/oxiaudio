//! Facade DSP tests: verify facade wrappers produce identical results to direct dsp calls.
//! Also includes pipeline integration tests: decode-like synthesis → DSP → encode roundtrips.

#[cfg(feature = "pure")]
mod facade_dsp_tests {
    use oxiaudio::{
        decode_file, dsp, encode_flac_to_vec, encode_wav, encode_wav_to_vec, AudioBuffer,
        ChannelLayout, SampleFormat,
    };
    use std::f32::consts::PI;

    fn make_test_buf(amplitude: f32) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![amplitude; 100],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    /// Verify that `dsp::gain` via facade produces the same result as `oxiaudio_dsp::gain` directly.
    #[test]
    fn test_facade_dsp_gain_matches_direct() {
        let buf = AudioBuffer {
            samples: vec![0.5f32; 100],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut via_facade = buf.clone();
        dsp::gain(&mut via_facade, 6.0);
        let mut via_direct = buf.clone();
        oxiaudio_dsp::gain(&mut via_direct, 6.0);
        let max_diff = via_facade
            .samples
            .iter()
            .zip(via_direct.samples.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-7, "gain mismatch: {max_diff}");
    }

    /// Verify that normalizing to -1.0 dBFS produces a peak at -1 dBFS (within 0.01 dB).
    #[test]
    fn test_facade_normalize_then_peak_is_target_db() {
        let mut buf = AudioBuffer {
            samples: vec![0.1f32, -0.3, 0.2, -0.15],
            sample_rate: 44100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        dsp::normalize(&mut buf, -1.0); // target -1 dBFS
        let peak = buf.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let peak_db = 20.0 * peak.log10();
        assert!(
            (peak_db - (-1.0)).abs() < 0.01,
            "peak_db = {peak_db:.4}, expected -1.0 dBFS"
        );
    }

    /// Verify gain at 0 dB is unity.
    #[test]
    fn test_facade_gain_zero_db_is_unity() {
        let buf = make_test_buf(0.5);
        let mut out = buf.clone();
        dsp::gain(&mut out, 0.0);
        for (a, b) in buf.samples.iter().zip(out.samples.iter()) {
            assert!((a - b).abs() < 1e-7, "0 dB gain changed sample values");
        }
    }

    /// Verify mix_to_mono averages channels correctly.
    #[test]
    fn test_facade_mix_to_mono_averages() {
        let buf = AudioBuffer {
            samples: vec![0.4f32, 0.8, 0.4, 0.8], // L=0.4, R=0.8 for 2 frames
            sample_rate: 44100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let mono = dsp::mix_to_mono(&buf);
        assert_eq!(mono.channels, ChannelLayout::Mono);
        assert_eq!(mono.frame_count(), 2);
        // Each frame average: (0.4 + 0.8) / 2 = 0.6
        assert!(
            (mono.samples[0] - 0.6).abs() < 1e-6,
            "mono[0]={}",
            mono.samples[0]
        );
        assert!(
            (mono.samples[1] - 0.6).abs() < 1e-6,
            "mono[1]={}",
            mono.samples[1]
        );
    }

    /// Test convert WAV→FLAC→WAV roundtrip preserves sample values within tolerance.
    #[test]
    fn test_convert_wav_to_flac_and_back_preserves_samples() {
        use std::f32::consts::PI;
        let sample_rate = 44100u32;
        let n_frames = 4410usize; // 0.1s
        let samples: Vec<f32> = (0..n_frames)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5)
            .collect();
        let original = AudioBuffer {
            samples: samples.clone(),
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };

        let dir = std::env::temp_dir();
        let wav_path = dir.join("oxiaudio_m23_facade_roundtrip.wav");
        let flac_path = dir.join("oxiaudio_m23_facade_roundtrip.flac");
        let wav2_path = dir.join("oxiaudio_m23_facade_roundtrip2.wav");

        // Encode to WAV
        encode_wav(&original, &wav_path).expect("encode_wav failed");

        // Convert WAV → FLAC
        oxiaudio::convert(&wav_path, &flac_path).expect("convert wav→flac failed");

        // Convert FLAC → WAV
        oxiaudio::convert(&flac_path, &wav2_path).expect("convert flac→wav failed");

        // Decode back
        let decoded = decode_file(&wav2_path).expect("decode roundtrip WAV failed");

        // Cleanup
        let _ = std::fs::remove_file(&wav_path);
        let _ = std::fs::remove_file(&flac_path);
        let _ = std::fs::remove_file(&wav2_path);

        // Verify sample count is the same
        assert_eq!(
            decoded.samples.len(),
            original.samples.len(),
            "sample count changed: orig={} decoded={}",
            original.samples.len(),
            decoded.samples.len()
        );

        // FLAC is lossless; max abs diff should be very small (quantization only from i16 FLAC)
        let max_diff = original
            .samples
            .iter()
            .zip(decoded.samples.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 2e-4,
            "max abs diff too large after WAV→FLAC→WAV roundtrip: {max_diff}"
        );
    }

    // ─── Pipeline helpers ────────────────────────────────────────────────────

    fn mono_sine(freq: f32, sr: u32, frames: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: (0..frames)
                .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin() * 0.5)
                .collect(),
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn stereo_sine(freq: f32, sr: u32, frames: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: (0..frames * 2)
                .map(|i| {
                    let frame = i / 2;
                    (2.0 * PI * freq * frame as f32 / sr as f32).sin() * 0.3
                })
                .collect(),
            sample_rate: sr,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    // ─── Test: resample → encode WAV ────────────────────────────────────────

    /// Synthetic 44100 Hz mono buffer resampled to 48000 Hz, then encoded to WAV.
    /// Verifies sample rate change, non-empty output, and valid RIFF header.
    #[test]
    fn test_resample_encode_wav_pipeline() {
        let buf = mono_sine(440.0, 44100, 44100); // 1 second at 44100 Hz

        let resampled = dsp::resample(&buf, 48000).expect("resample to 48000 should succeed");
        assert_eq!(resampled.sample_rate, 48000, "resampled rate must be 48000");
        assert!(
            !resampled.samples.is_empty(),
            "resampled buffer must not be empty"
        );

        let wav_bytes = encode_wav_to_vec(&resampled).expect("encode_wav_to_vec should succeed");
        assert!(wav_bytes.len() >= 44, "WAV output too short");
        assert_eq!(&wav_bytes[..4], b"RIFF", "output must start with RIFF");
    }

    // ─── Test: pitch shift → encode WAV ─────────────────────────────────────

    /// Synthetic stereo buffer pitch-shifted up 2 semitones, then encoded to WAV.
    /// Verifies non-empty shifted output and valid RIFF header.
    #[test]
    fn test_pitch_shift_encode_wav_pipeline() {
        let buf = stereo_sine(440.0, 44100, 44100 * 2); // 2 seconds stereo

        let shifted = dsp::pitch_shift(&buf, 2.0).expect("pitch_shift by 2 semitones");
        assert!(
            !shifted.samples.is_empty(),
            "shifted buffer must not be empty"
        );

        let wav_bytes = encode_wav_to_vec(&shifted).expect("encode_wav_to_vec of shifted");
        assert!(wav_bytes.len() >= 44, "WAV output too short");
        assert_eq!(
            &wav_bytes[..4],
            b"RIFF",
            "pitch-shifted WAV must have RIFF header"
        );
    }

    // ─── Test: biquad filter → encode FLAC ──────────────────────────────────

    /// Synthetic stereo buffer lowpass-filtered at 4000 Hz, then encoded to FLAC.
    /// Verifies sample count is preserved and valid fLaC marker in output.
    #[test]
    fn test_biquad_filter_encode_flac_pipeline() {
        let sr = 48000u32;
        let buf = stereo_sine(1000.0, sr, sr as usize); // 1 second stereo at 48000 Hz

        let filter = dsp::BiquadFilter::lowpass(4000.0, 0.707, sr);
        let filtered = filter.process(&buf);
        assert_eq!(
            filtered.samples.len(),
            buf.samples.len(),
            "filter must preserve sample count"
        );

        let flac_bytes = encode_flac_to_vec(&filtered).expect("encode_flac_to_vec should succeed");
        assert!(flac_bytes.len() >= 4, "FLAC output too short");
        assert_eq!(
            &flac_bytes[..4],
            b"fLaC",
            "FLAC output must start with fLaC marker"
        );
    }
}
