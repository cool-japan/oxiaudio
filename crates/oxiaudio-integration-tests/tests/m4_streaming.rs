use std::io::BufWriter;

use oxiaudio_core::{
    AudioBuffer, AudioDecoder, AudioEncoder, AudioSink, ChannelLayout, SampleFormat,
};
use oxiaudio_decode::SymphoniaDecoder;
use oxiaudio_encode::{FlacStreamEncoder, WavBitDepth, WavEncoder, WavStreamEncoder};

/// Generate a mono sine wave buffer.
fn sine_buffer_mono(freq: f32, sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
    let n_frames = (sample_rate as f32 * duration_secs) as usize;
    let mut samples = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let s = (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin() * 0.5;
        samples.push(s);
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

/// Encode `buf` as WAV via `WavStreamEncoder` in `chunk_samples`-sample chunks,
/// writing to a temp file and returning the raw bytes.
fn encode_wav_stream_to_vec(buf: &AudioBuffer<f32>, chunk_samples: usize) -> Vec<u8> {
    let tmp_path = std::env::temp_dir().join(format!(
        "oxiaudio_wav_stream_{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));

    {
        let file = std::fs::File::create(&tmp_path).expect("create temp wav file");
        let mut enc = WavStreamEncoder::new(
            BufWriter::new(file),
            buf.sample_rate,
            buf.channels,
            WavBitDepth::F32,
        )
        .expect("WavStreamEncoder::new");

        let mut frames_expected = 0u64;
        let n_ch = buf.channels.channel_count();

        for chunk in buf.samples.chunks(chunk_samples) {
            let chunk_buf = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: buf.sample_rate,
                channels: buf.channels,
                format: buf.format,
            };
            frames_expected += (chunk.len() / n_ch) as u64;
            enc.encode_chunk(&chunk_buf).expect("encode_chunk");
        }

        // Verify frames_written accessor is consistent
        assert_eq!(
            enc.frames_written(),
            frames_expected,
            "frames_written must match total frames encoded"
        );

        enc.finalize().expect("finalize");
    }

    let bytes = std::fs::read(&tmp_path).expect("read temp wav file");
    let _ = std::fs::remove_file(&tmp_path);
    bytes
}

/// `test_wav_stream_60s` (using 5 seconds to keep the test fast):
///
/// Encodes a 5-second sine wave via `WavStreamEncoder` in 4096-frame chunks and
/// also encodes the *same* buffer with `WavEncoder`. Both must produce byte-identical output.
#[test]
fn test_wav_stream_60s() {
    // 5 seconds at 44100 Hz, 440 Hz sine, mono — multiple chunks guaranteed
    let buf = sine_buffer_mono(440.0, 44_100, 5.0);

    // Encode with the batch encoder using a temp file (same as streaming path)
    let batch_path = std::env::temp_dir().join(format!(
        "oxiaudio_wav_batch_{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));
    {
        let file = std::fs::File::create(&batch_path).expect("create temp batch wav file");
        let mut batch_enc = WavEncoder::default();
        batch_enc
            .encode(&buf, BufWriter::new(file))
            .expect("WavEncoder::encode must succeed");
    }
    let batch_bytes = std::fs::read(&batch_path).expect("read batch wav");
    let _ = std::fs::remove_file(&batch_path);

    // Encode with the streaming encoder in 4096-frame chunks
    let stream_bytes = encode_wav_stream_to_vec(&buf, 4096);

    assert_eq!(
        batch_bytes, stream_bytes,
        "WavStreamEncoder output must be byte-identical to WavEncoder output"
    );
}

/// Encode `buf` as FLAC via `FlacStreamEncoder` in `chunk_samples`-sample chunks,
/// then decode the result with `SymphoniaDecoder`.
fn encode_flac_stream_and_decode(buf: &AudioBuffer<f32>, chunk_samples: usize) -> AudioBuffer<f32> {
    let tmp_path = std::env::temp_dir().join(format!(
        "oxiaudio_flac_stream_{}.flac",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));

    {
        let file = std::fs::File::create(&tmp_path).expect("create temp flac file");
        let mut enc =
            FlacStreamEncoder::new(BufWriter::new(file), buf.sample_rate, buf.channels, 5);

        for chunk in buf.samples.chunks(chunk_samples) {
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

    let file = std::fs::File::open(&tmp_path).expect("open temp flac file");
    let decoded = SymphoniaDecoder
        .decode(file)
        .expect("SymphoniaDecoder::decode");
    let _ = std::fs::remove_file(&tmp_path);
    decoded
}

/// `test_flac_stream_roundtrip`:
///
/// Encodes a 1-second 440 Hz sine wave via `FlacStreamEncoder` in 512-frame chunks,
/// then decodes with `SymphoniaDecoder` and verifies sample count and values within `1e-4`.
#[test]
fn test_flac_stream_roundtrip() {
    let sample_rate = 44_100u32;
    let duration_secs = 1.0f32;
    let freq = 440.0f32;
    let buf = sine_buffer_mono(freq, sample_rate, duration_secs);

    let expected_frames = (sample_rate as f32 * duration_secs) as usize;
    assert_eq!(buf.samples.len(), expected_frames);

    let decoded = encode_flac_stream_and_decode(&buf, 512);

    assert_eq!(
        decoded.sample_rate, sample_rate,
        "decoded sample rate must match"
    );
    assert_eq!(
        decoded.samples.len(),
        buf.samples.len(),
        "decoded sample count must match original"
    );

    let tolerance = 1e-4_f32;
    for (i, (&orig, &dec)) in buf.samples.iter().zip(decoded.samples.iter()).enumerate() {
        let diff = (orig - dec).abs();
        assert!(
            diff <= tolerance,
            "sample[{i}]: original={orig}, decoded={dec}, diff={diff} > tolerance={tolerance}"
        );
    }
}

/// `test_wav_stream_audio_sink_trait`:
///
/// Verifies that `WavStreamEncoder` implements `AudioSink` and that `write_chunk`
/// correctly delegates to `encode_chunk`.
#[test]
fn test_wav_stream_audio_sink_trait() {
    let buf = sine_buffer_mono(220.0, 48_000, 0.1);
    let chunk_samples = 256;

    let tmp_path = std::env::temp_dir().join(format!(
        "oxiaudio_wav_sink_{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));

    {
        let file = std::fs::File::create(&tmp_path).expect("create temp file");
        let mut enc = WavStreamEncoder::new(
            BufWriter::new(file),
            48_000,
            ChannelLayout::Mono,
            WavBitDepth::F32,
        )
        .expect("new");

        for chunk in buf.samples.chunks(chunk_samples) {
            let chunk_buf = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: 48_000,
                channels: ChannelLayout::Mono,
                format: SampleFormat::F32,
            };
            // Call via AudioSink trait method
            enc.write_chunk(&chunk_buf)
                .expect("write_chunk via AudioSink");
        }
        enc.finalize().expect("finalize");
    }

    let bytes = std::fs::read(&tmp_path).expect("read wav");
    let _ = std::fs::remove_file(&tmp_path);
    // A valid WAV with some samples should be larger than the 44-byte header
    assert!(
        bytes.len() > 44,
        "encoded WAV must be larger than the header"
    );
    // WAV must start with RIFF marker
    assert_eq!(&bytes[..4], b"RIFF", "WAV must start with RIFF marker");
}

/// `test_flac_stream_audio_sink_trait`:
///
/// Verifies that `FlacStreamEncoder` implements `AudioSink` and that `write_chunk`
/// correctly delegates to `encode_chunk`.
#[test]
fn test_flac_stream_audio_sink_trait() {
    let buf = sine_buffer_mono(220.0, 44_100, 0.1);
    let chunk_samples = 256;

    let tmp_path = std::env::temp_dir().join(format!(
        "oxiaudio_flac_sink_{}.flac",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));

    {
        let file = std::fs::File::create(&tmp_path).expect("create temp file");
        let mut enc = FlacStreamEncoder::new(BufWriter::new(file), 44_100, ChannelLayout::Mono, 5);

        for chunk in buf.samples.chunks(chunk_samples) {
            let chunk_buf = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: 44_100,
                channels: ChannelLayout::Mono,
                format: SampleFormat::F32,
            };
            // Call via AudioSink trait method
            enc.write_chunk(&chunk_buf)
                .expect("write_chunk via AudioSink");
        }
        enc.finalize().expect("finalize");
    }

    let bytes = std::fs::read(&tmp_path).expect("read flac");
    let _ = std::fs::remove_file(&tmp_path);
    // A valid FLAC must start with the fLaC marker
    assert!(bytes.len() > 4, "encoded FLAC must have content");
    assert_eq!(&bytes[..4], b"fLaC", "FLAC must start with fLaC marker");
}
