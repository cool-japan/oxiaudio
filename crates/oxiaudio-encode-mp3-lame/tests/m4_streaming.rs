//! M4 streaming integration test: chunk-at-a-time MP3 encode via `LameMp3StreamEncoder`.
//!
//! Encodes a 10-second synthetic stereo sine wave in 4096-frame chunks and verifies
//! that the resulting MP3 is decodable by Symphonia with the correct sample rate.

#[cfg(feature = "mp3-encode-lame")]
mod tests {
    use oxiaudio_core::{AudioBuffer, AudioDecoder, ChannelLayout, SampleFormat};
    use oxiaudio_decode::SymphoniaDecoder;
    use oxiaudio_encode_mp3_lame::lame::{LameMp3Encoder, LameMp3StreamEncoder};
    use std::io::Cursor;

    fn sine_buf(secs: f32) -> AudioBuffer<f32> {
        let sr = 44_100u32;
        let n = (sr as f32 * secs) as usize;
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let s = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
            samples.push(s);
            samples.push(s);
        }
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_mp3_stream_10s_valid() {
        let full_buf = sine_buf(10.0);
        let encoder_config = LameMp3Encoder::default();

        // Encode in 4096-frame (stereo → 8192 sample) chunks.
        let mut out = Cursor::new(Vec::new());
        let mut stream_enc = LameMp3StreamEncoder::new(
            &mut out,
            &encoder_config,
            full_buf.sample_rate,
            full_buf.channels,
        )
        .expect("LameMp3StreamEncoder::new failed");

        let chunk_samples = 4096 * 2; // stereo: 2 samples per frame
        for chunk in full_buf.samples.chunks(chunk_samples) {
            let buf = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: full_buf.sample_rate,
                channels: full_buf.channels,
                format: full_buf.format,
            };
            stream_enc.encode_chunk(&buf).expect("encode_chunk failed");
        }
        stream_enc.finalize().expect("finalize failed");

        // Verify the output is a valid, decodable MP3.
        out.set_position(0);
        let decoded = SymphoniaDecoder
            .decode(out)
            .expect("decode of streamed MP3 failed");
        assert!(!decoded.samples.is_empty(), "decoded MP3 has no samples");
        assert_eq!(
            decoded.sample_rate, full_buf.sample_rate,
            "sample rate round-trip failed"
        );
    }

    #[test]
    fn test_mp3_stream_sample_rate_mismatch_returns_err() {
        let buf = sine_buf(1.0);
        let config = LameMp3Encoder::default();
        let mut out = Cursor::new(Vec::new());

        let mut stream_enc =
            LameMp3StreamEncoder::new(&mut out, &config, buf.sample_rate, buf.channels)
                .expect("new failed");

        // Pass a chunk with wrong sample rate.
        let bad_buf = AudioBuffer {
            samples: buf.samples[..1024].to_vec(),
            sample_rate: 22_050, // wrong
            channels: buf.channels,
            format: buf.format,
        };
        assert!(
            stream_enc.encode_chunk(&bad_buf).is_err(),
            "expected Err on sample rate mismatch"
        );
    }

    #[test]
    fn test_mp3_stream_channel_mismatch_returns_err() {
        let buf = sine_buf(1.0);
        let config = LameMp3Encoder::default();
        let mut out = Cursor::new(Vec::new());

        let mut stream_enc =
            LameMp3StreamEncoder::new(&mut out, &config, buf.sample_rate, buf.channels)
                .expect("new failed");

        // Pass a mono chunk when stream was opened as stereo.
        let bad_buf = AudioBuffer {
            samples: buf.samples[..1024].to_vec(),
            sample_rate: buf.sample_rate,
            channels: ChannelLayout::Mono, // wrong
            format: buf.format,
        };
        assert!(
            stream_enc.encode_chunk(&bad_buf).is_err(),
            "expected Err on channel layout mismatch"
        );
    }
}
