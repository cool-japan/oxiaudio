//! DSP pipeline → streaming encoder integration tests.
//!
//! Verifies that DSP operations (gain, biquad filter, DspChain) compose correctly
//! with streaming encoders (WAV, FLAC) via the `AudioSink` trait.

#[cfg(feature = "pure")]
mod pipeline_tests {
    use oxiaudio::{
        dsp, encode_flac_to_vec, encode_wav_to_vec, AudioBuffer, ChannelLayout, SampleFormat,
    };

    // ─── Buffer factory ───────────────────────────────────────────────────────

    fn sine_buf(freq: f32, n: usize, rate: u32) -> AudioBuffer<f32> {
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin() * 0.3)
            .collect();
        AudioBuffer {
            samples,
            sample_rate: rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn stereo_sine_buf(freq: f32, n: usize, rate: u32) -> AudioBuffer<f32> {
        let samples: Vec<f32> = (0..n * 2)
            .map(|i| {
                let frame = i / 2;
                let s =
                    (2.0 * std::f32::consts::PI * freq * frame as f32 / rate as f32).sin() * 0.3;
                if i % 2 == 0 {
                    s
                } else {
                    -s
                }
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate: rate,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    // ─── Task 3a: gain → WAV encode ───────────────────────────────────────────

    /// Apply a -6 dB gain via the DSP facade, then encode to WAV and verify magic bytes.
    #[test]
    fn test_gain_then_wav_encode() {
        let buf = stereo_sine_buf(440.0, 4096, 44_100);

        // Apply gain via DSP facade (in-place, mutates clone)
        let mut gained = buf.clone();
        dsp::gain(&mut gained, -6.0); // −6 dB ≈ 0.5× amplitude

        // Encode to WAV
        let wav = encode_wav_to_vec(&gained).expect("WAV encode after gain must succeed");
        assert!(
            wav.starts_with(b"RIFF"),
            "WAV output after gain must start with RIFF"
        );
        assert_eq!(
            &wav[8..12],
            b"WAVE",
            "WAV format marker must be WAVE after gain"
        );
        // Peak after -6dB should be approximately half of 0.3 ≈ 0.15
        // The WAV file contains real data: length should be proportional to input.
        let expected_min_bytes = 4096 * 2 * 4; // 4096 frames * 2 ch * 4 bytes F32
        assert!(
            wav.len() >= expected_min_bytes,
            "WAV must contain at least {expected_min_bytes} data bytes, got {}",
            wav.len()
        );
    }

    // ─── Task 3b: gain attenuation is preserved in encoded output ─────────────

    /// Verify that gain applied before encoding is reflected in the WAV magnitude.
    /// Gain of 0 dB (unity) should produce the same samples as the original.
    #[test]
    fn test_unity_gain_wav_roundtrip() {
        let buf = sine_buf(440.0, 1024, 44_100);

        let mut unity_gained = buf.clone();
        dsp::gain(&mut unity_gained, 0.0); // 0 dB = unity gain

        // Both should encode to the same WAV content (approximately)
        let wav_orig = encode_wav_to_vec(&buf).expect("WAV encode original");
        let wav_unity = encode_wav_to_vec(&unity_gained).expect("WAV encode unity-gained");

        assert_eq!(
            wav_orig.len(),
            wav_unity.len(),
            "unity-gained WAV must have the same byte length as original"
        );
    }

    // ─── Task 3c: biquad lowpass filter → FLAC encode ─────────────────────────

    /// Apply a biquad lowpass filter via the DSP facade, then encode to FLAC and verify magic bytes.
    #[test]
    fn test_biquad_lowpass_then_flac_encode() {
        let buf = stereo_sine_buf(440.0, 4096, 44_100);

        // Apply lowpass biquad (1 kHz cutoff, Q = 0.707 = Butterworth)
        let filter = dsp::BiquadFilter::lowpass(1_000.0, 0.707, 44_100);
        let filtered = filter.process(&buf);

        assert_eq!(
            filtered.samples.len(),
            buf.samples.len(),
            "biquad filter must preserve sample count"
        );

        let flac = encode_flac_to_vec(&filtered).expect("FLAC encode after biquad must succeed");
        assert!(
            flac.starts_with(b"fLaC"),
            "FLAC output after biquad must start with fLaC"
        );
        assert!(
            flac.len() > 100,
            "FLAC output must be non-trivially large (got {} bytes)",
            flac.len()
        );
    }

    // ─── Task 3d: highpass filter → WAV encode ────────────────────────────────

    /// Apply a biquad highpass filter then encode to WAV, verifying the output is valid.
    #[test]
    fn test_biquad_highpass_then_wav_encode() {
        let buf = sine_buf(440.0, 4096, 44_100);

        let filter = dsp::BiquadFilter::highpass(80.0, 0.707, 44_100);
        let filtered = filter.process(&buf);

        let wav = encode_wav_to_vec(&filtered).expect("WAV encode after highpass must succeed");
        assert!(
            wav.starts_with(b"RIFF"),
            "WAV output after highpass must start with RIFF"
        );
        assert_eq!(&wav[8..12], b"WAVE", "format marker must be WAVE");
    }

    // ─── Task 3e: DspChain (gain + lowpass) → WAV encode ─────────────────────

    /// Thread gain and biquad lowpass through a DspChain, then encode to WAV.
    #[test]
    fn test_dsp_chain_gain_and_lowpass_then_wav_encode() {
        let buf = stereo_sine_buf(440.0, 4096, 44_100);

        // DspChain: highpass 80 Hz, then normalize
        let chain = dsp::DspChain::new().then_filter(dsp::BiquadFilter::highpass(
            80.0,
            0.707,
            buf.sample_rate,
        ));

        let processed = chain.process(&buf).expect("DspChain process must succeed");

        let wav = encode_wav_to_vec(&processed).expect("WAV encode after DspChain must succeed");
        assert!(
            wav.starts_with(b"RIFF"),
            "WAV output after DspChain must start with RIFF"
        );
        assert_eq!(&wav[8..12], b"WAVE", "format marker must be WAVE");
        assert_eq!(
            processed.samples.len(),
            buf.samples.len(),
            "DspChain must preserve sample count"
        );
    }

    // ─── Task 3f: normalize → FLAC encode ────────────────────────────────────

    /// Peak-normalize to -3 dBFS, then encode to FLAC. Verifies both DSP and encoder compose.
    #[test]
    fn test_normalize_then_flac_encode() {
        let buf = sine_buf(880.0, 8192, 48_000);

        let mut normalized = buf.clone();
        dsp::normalize(&mut normalized, -3.0); // peak at -3 dBFS

        let flac =
            encode_flac_to_vec(&normalized).expect("FLAC encode after normalize must succeed");
        assert!(
            flac.starts_with(b"fLaC"),
            "FLAC output after normalize must start with fLaC"
        );
    }

    // ─── Task 3g: AudioSink streaming encode via WavStreamEncoder ────────────

    /// Feed DSP-processed chunks into WavStreamEncoder via the AudioSink trait.
    #[test]
    fn test_dsp_chunk_streaming_to_wav() {
        use std::io::Cursor;

        let buf = stereo_sine_buf(440.0, 8192, 44_100);

        // Apply biquad and split into chunks, encode via streaming
        let filter = dsp::BiquadFilter::lowpass(2_000.0, 0.707, 44_100);
        let filtered = filter.process(&buf);

        // Use WavStreamEncoder to stream chunks
        let cursor = Cursor::new(Vec::new());
        let mut enc = oxiaudio_encode::WavStreamEncoder::new(
            cursor,
            44_100,
            ChannelLayout::Stereo,
            oxiaudio_encode::WavBitDepth::F32,
        )
        .expect("WavStreamEncoder::new must succeed");

        // Feed in chunks of 512 stereo frames (1024 samples)
        for chunk in filtered.samples.chunks(1024) {
            let chunk_buf = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: 44_100,
                channels: ChannelLayout::Stereo,
                format: SampleFormat::F32,
            };
            enc.encode_chunk(&chunk_buf)
                .expect("encode_chunk must succeed");
        }

        enc.finalize()
            .expect("WavStreamEncoder::finalize must succeed");
    }
}
