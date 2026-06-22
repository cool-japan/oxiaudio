//! Extracted from oxiaudio-encode/src/flac_streaming.rs to break the encode↔decode dev-dependency cycle.
//!
//! This streaming-FLAC round-trip test encodes with `oxiaudio_encode::FlacStreamingEncoder` and
//! decodes with `oxiaudio_decode::SymphoniaDecoder`, so it lives in the integration-tests crate
//! that dev-depends on both.

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use oxiaudio_encode::FlacStreamingEncoder;

fn sine_buf_stereo(sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
    let n_frames = (sample_rate as f32 * duration_secs) as usize;
    let samples: Vec<f32> = (0..n_frames)
        .flat_map(|i| {
            let s =
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.4;
            [s, -s]
        })
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

#[test]
fn test_flac_streaming_encoder_decode_roundtrip() {
    use oxiaudio_core::AudioDecoder;
    use oxiaudio_decode::SymphoniaDecoder;

    let sample_rate = 44_100u32;
    let buf = sine_buf_stereo(sample_rate, 2.0);
    let expected_frames = (sample_rate as f32 * 2.0) as usize;

    // Write to a temp file for decode.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let path = std::env::temp_dir().join(format!("oxiaudio_flac_streaming_{ts}.flac"));

    {
        let file = std::fs::File::create(&path).expect("create temp flac");
        let bw = std::io::BufWriter::new(file);
        let mut enc = FlacStreamingEncoder::new(bw, sample_rate, ChannelLayout::Stereo, 5)
            .expect("FlacStreamingEncoder::new");

        // Feed in 4096-frame chunks.
        let n_ch = 2usize;
        for chunk in buf.samples.chunks(4096 * n_ch) {
            let chunk_buf = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: buf.sample_rate,
                channels: buf.channels,
                format: buf.format,
            };
            enc.encode_chunk(&chunk_buf).expect("encode_chunk");
        }
        enc.finalize().expect("finalize");
    }

    // Decode and verify.
    let file = std::fs::File::open(&path).expect("open temp flac");
    let reader = std::io::BufReader::new(file);
    let decoded = SymphoniaDecoder.decode(reader).expect("decode FLAC");

    let _ = std::fs::remove_file(&path);

    // Verify sample count.
    assert_eq!(
        decoded.samples.len(),
        expected_frames * 2, // stereo
        "decoded sample count must match: expected {} got {}",
        expected_frames * 2,
        decoded.samples.len()
    );

    // Verify non-silence: at least some samples should be non-zero.
    let max_abs = decoded
        .samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs > 0.01,
        "decoded audio must be non-silent (max_abs={max_abs})"
    );
}
