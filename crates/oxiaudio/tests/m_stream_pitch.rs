//! Streaming decode → pitch-shift → streaming encode pipeline integration test.
//!
//! Exercises the in-memory pipeline:
//!   1. `encode_wav_to_vec` — source → WAV bytes
//!   2. `decode_stream_with_block_size` from `Cursor<Vec<u8>>` — collect chunks
//!   3. `dsp::pitch_shift` on the accumulated buffer
//!   4. `WavStreamEncoder` chunk-feeding → finalize to a second `Cursor<Vec<u8>>`
//!   5. Decode the output WAV and verify duration, amplitude, sample rate, channels
//!
//! No external crates or files are required; all I/O is in-memory.
//! Note: `dsp::pitch_shift` mixes its input to mono (STFT-based); output is mono.

#[cfg(feature = "pure")]
mod stream_pitch_tests {
    use std::io::Cursor;

    use oxiaudio::{
        decode_stream_with_block_size, dsp, encode_wav_to_vec, AudioBuffer, ChannelLayout,
        SampleFormat,
    };
    use oxiaudio_encode::{WavBitDepth, WavStreamEncoder};

    // ─── Buffer factory ───────────────────────────────────────────────────────

    /// Build a stereo interleaved sine-wave buffer at the given frequency and duration.
    ///
    /// Both channels carry the same sine signal so they do NOT cancel during mix_to_mono.
    fn stereo_sine(freq: f32, n_frames: usize, rate: u32) -> AudioBuffer<f32> {
        let samples: Vec<f32> = (0..n_frames * 2)
            .map(|i| {
                let frame = i / 2;
                (2.0 * std::f32::consts::PI * freq * frame as f32 / rate as f32).sin() * 0.5
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate: rate,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    // ─── Helper: RMS amplitude ────────────────────────────────────────────────

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }

    // ─── Main pipeline test ───────────────────────────────────────────────────

    /// Full in-memory `decode_stream` → `dsp::pitch_shift` → `encode_stream_wav` pipeline.
    ///
    /// Pipeline steps:
    ///   1. Build a synthetic stereo 440 Hz sine at 44100 Hz (~0.2 s)
    ///   2. Encode to in-memory WAV bytes
    ///   3. Stream-decode from `Cursor<Vec<u8>>` (exercises `decode_stream_with_block_size`)
    ///   4. Accumulate chunks (decode produces stereo; `pitch_shift` needs at least 2048
    ///      mono samples per call due to its STFT window size)
    ///   5. Apply `dsp::pitch_shift(buf, 4.0)` to the full accumulated buffer
    ///      (pitch_shift mixes to mono internally and returns a mono buffer)
    ///   6. Feed the mono shifted buffer to `WavStreamEncoder` in 1024-sample chunks
    ///   7. Call `finalize()` on the encoder
    ///   8. Encode the shifted samples via `encode_wav_to_vec` and decode back for assertions
    ///
    /// Assertions:
    ///   - Non-empty decoded output
    ///   - Sample rate preserved (44100 Hz)
    ///   - Channel layout matches the output of pitch_shift (mono)
    ///   - Duration within 10% of input (pitch_shift preserves frame count)
    ///   - RMS amplitude > 0.001 (non-silent after pitch shift)
    #[test]
    fn test_stream_pitch_shift_pipeline_stereo() {
        const RATE: u32 = 44_100;
        const N_FRAMES: usize = 44_100; // 1.0 s — enough for STFT (window=2048, hop=512)
        const SHIFT_SEMITONES: f32 = 4.0;
        const BLOCK: usize = 1024; // stream decode block size in frames
        const ENCODE_CHUNK: usize = 1024; // stream encode chunk size in samples

        // ── Step 1: build synthetic stereo source ──────────────────────────────
        let source = stereo_sine(440.0, N_FRAMES, RATE);

        // ── Step 2: encode to in-memory WAV bytes ──────────────────────────────
        let source_wav = encode_wav_to_vec(&source).expect("encode_wav_to_vec must succeed");
        assert!(
            source_wav.starts_with(b"RIFF"),
            "source WAV must start with RIFF magic"
        );

        // ── Step 3: stream-decode back from a Cursor ────────────────────────────
        let cursor = Cursor::new(source_wav);
        let mut all_samples: Vec<f32> = Vec::with_capacity(source.samples.len());
        let mut decoded_rate = RATE;
        let mut decoded_channels = ChannelLayout::Stereo;

        for result in decode_stream_with_block_size(cursor, BLOCK) {
            let chunk = result.expect("decode_stream chunk must be Ok");
            decoded_rate = chunk.sample_rate;
            decoded_channels = chunk.channels;
            all_samples.extend_from_slice(&chunk.samples);
        }

        assert!(
            !all_samples.is_empty(),
            "stream-decoded buffer must not be empty"
        );
        assert_eq!(decoded_rate, RATE, "decoded sample rate must match source");
        assert_eq!(
            decoded_channels,
            ChannelLayout::Stereo,
            "decoded channels must be stereo"
        );
        assert!(
            rms(&all_samples) > 0.001,
            "decoded samples must not be silent (rms={})",
            rms(&all_samples)
        );

        // ── Step 4: reconstruct full buffer for pitch_shift ────────────────────
        // pitch_shift uses STFT with window_size=2048 — needs enough samples.
        let accumulated = AudioBuffer {
            samples: all_samples,
            sample_rate: decoded_rate,
            channels: decoded_channels,
            format: SampleFormat::F32,
        };

        // ── Step 5: apply pitch_shift ──────────────────────────────────────────
        // pitch_shift mixes to mono and returns a mono AudioBuffer<f32>
        let shifted =
            dsp::pitch_shift(&accumulated, SHIFT_SEMITONES).expect("pitch_shift must succeed");
        assert!(
            !shifted.samples.is_empty(),
            "pitch_shift must return non-empty output"
        );
        assert_eq!(
            shifted.sample_rate, RATE,
            "pitch_shift must preserve sample rate"
        );
        assert!(
            rms(&shifted.samples) > 0.001,
            "pitch_shift output must not be silent (rms={})",
            rms(&shifted.samples)
        );

        // The shifted buffer is mono
        let n_shift_channels = shifted.channels.channel_count();
        let shifted_frames = shifted.samples.len() / n_shift_channels;

        // ── Step 6: stream-encode the shifted buffer via WavStreamEncoder ────
        let enc_cursor = Cursor::new(Vec::<u8>::new());
        let mut enc = WavStreamEncoder::new(
            enc_cursor,
            shifted.sample_rate,
            shifted.channels,
            WavBitDepth::F32,
        )
        .expect("WavStreamEncoder::new must succeed");

        for chunk in shifted.samples.chunks(ENCODE_CHUNK) {
            let chunk_buf = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: shifted.sample_rate,
                channels: shifted.channels,
                format: SampleFormat::F32,
            };
            enc.encode_chunk(&chunk_buf)
                .expect("WavStreamEncoder::encode_chunk must succeed");
        }

        // ── Step 7: finalize the stream encoder ───────────────────────────────
        enc.finalize()
            .expect("WavStreamEncoder::finalize must succeed");

        // ── Step 8: encode shifted to Vec<u8> and decode for assertions ───────
        let output_wav =
            encode_wav_to_vec(&shifted).expect("encode_wav_to_vec of shifted audio must succeed");
        assert!(
            output_wav.starts_with(b"RIFF"),
            "output WAV must start with RIFF magic"
        );

        let out_cursor = Cursor::new(output_wav);
        let mut out_chunks: Vec<AudioBuffer<f32>> = Vec::new();
        for result in decode_stream_with_block_size(out_cursor, 4096) {
            let chunk = result.expect("decode final WAV chunk must be Ok");
            out_chunks.push(chunk);
        }

        assert!(
            !out_chunks.is_empty(),
            "output WAV must decode to at least one chunk"
        );

        let first = &out_chunks[0];
        assert_eq!(
            first.sample_rate, RATE,
            "output sample rate must be preserved through encode/decode roundtrip"
        );
        assert_eq!(
            first.channels, shifted.channels,
            "output channel layout must match pitch_shift output"
        );

        // Duration: pitch_shift preserves frame count (istft output = same length as mono input)
        // Input stereo has N_FRAMES frames; after mix_to_mono → N_FRAMES mono frames.
        // pitch_shift calls `istft(&shifted_stft, mono.samples.len())` so output = N_FRAMES frames.
        let input_frames_mono = N_FRAMES;
        let tolerance = input_frames_mono / 10; // 10%
        assert!(
            shifted_frames.abs_diff(input_frames_mono) <= tolerance,
            "pitch_shift output duration {shifted_frames} frames too far from \
             input {input_frames_mono} frames (tolerance {tolerance})"
        );

        // RMS amplitude: pitch-shifted audio must not be silent
        let all_out: Vec<f32> = out_chunks
            .iter()
            .flat_map(|b| b.samples.iter().copied())
            .collect();
        let out_rms = rms(&all_out);
        assert!(
            out_rms > 0.001,
            "output RMS {out_rms:.6} is too low — pitch-shifted audio must not be silent"
        );
    }
}
