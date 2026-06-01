use std::io::Cursor;

use oxiaudio_core::{AudioDecoder, AudioSource, ChannelLayout};
use oxiaudio_decode::{StreamingDecoder, SymphoniaDecoder};

/// Build an in-memory stereo WAV at 48 kHz with `n_frames` frames of silence.
/// The WAV is written to a temp file (so `StreamingDecoder` can seek it) and also
/// returned as raw bytes for use with `SymphoniaDecoder`.
fn make_stereo_wav_bytes(n_frames: usize) -> Vec<u8> {
    let sample_rate = 48_000u32;
    let channels = 2u16;
    let mut buf = Vec::new();
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::new(Cursor::new(&mut buf), spec).expect("WavWriter::new should succeed");
    // interleaved stereo: 2 samples per frame
    for _ in 0..(n_frames * channels as usize) {
        writer.write_sample(0.0f32).expect("write_sample");
    }
    writer.finalize().expect("finalize");
    buf
}

/// Write bytes to a temp file and return a seekable `std::fs::File`.
fn write_temp_wav(bytes: &[u8]) -> std::fs::File {
    use std::io::Write;
    let mut path = std::env::temp_dir();
    path.push(format!(
        "oxiaudio_m3_test_{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    let mut f = std::fs::File::create(&path).expect("create temp file");
    f.write_all(bytes).expect("write temp wav");
    drop(f);
    std::fs::OpenOptions::new()
        .read(true)
        .write(false)
        .open(&path)
        .expect("open temp file for reading")
}

/// Concatenate all chunks from a `StreamingDecoder` into a single interleaved sample vec.
fn collect_all_streaming(decoder: StreamingDecoder) -> Vec<f32> {
    let mut all = Vec::new();
    for chunk in decoder {
        let buf = chunk.expect("chunk should decode without error");
        all.extend_from_slice(&buf.samples);
    }
    all
}

#[test]
fn test_stream_chunks_concat_equals_full_decode() {
    // Use a large buffer so we have multiple chunks: 2 seconds at 48k stereo = 96_000 frames.
    let n_frames = 96_000usize;
    let wav_bytes = make_stereo_wav_bytes(n_frames);

    // --- Full decode via SymphoniaDecoder ---
    let mut full_dec = SymphoniaDecoder;
    let full_buf = full_dec
        .decode(Cursor::new(wav_bytes.clone()))
        .expect("full decode should succeed");
    assert_eq!(full_buf.channels, ChannelLayout::Stereo);
    assert_eq!(full_buf.sample_rate, 48_000);

    // --- Streaming decode ---
    let file = write_temp_wav(&wav_bytes);
    let streaming_dec =
        StreamingDecoder::new(file, 4096).expect("StreamingDecoder::new should succeed");
    let streamed_samples = collect_all_streaming(streaming_dec);

    // Total sample counts must match.
    assert_eq!(
        streamed_samples.len(),
        full_buf.samples.len(),
        "streamed total samples ({}) != full decode samples ({})",
        streamed_samples.len(),
        full_buf.samples.len()
    );

    // Each sample must be within 1e-6 of the full decode result.
    for (i, (a, b)) in full_buf
        .samples
        .iter()
        .zip(streamed_samples.iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-6,
            "sample mismatch at index {i}: full={a}, streamed={b}"
        );
    }
}

#[test]
fn test_seek_mid_point() {
    let n_frames = 48_000usize; // 1 second at 48k stereo
    let wav_bytes = make_stereo_wav_bytes(n_frames);

    let file = write_temp_wav(&wav_bytes);
    let mut dec = StreamingDecoder::new(file, 4096).expect("StreamingDecoder::new should succeed");

    // Read the first two chunks and record the first chunk's samples.
    let first_chunk = dec
        .read_chunk()
        .expect("read_chunk 1 should succeed")
        .expect("first chunk should be Some");
    let first_samples = first_chunk.samples.clone();

    // Read a second chunk (just to advance the state).
    let _second = dec.read_chunk().expect("read_chunk 2 should succeed");

    // Seek back to the beginning (frame 0).
    dec.seek(0).expect("seek to frame 0 should succeed");

    // Read the first chunk again; it must match the pre-seek first chunk.
    let after_seek_first = dec
        .read_chunk()
        .expect("post-seek read_chunk should succeed")
        .expect("post-seek first chunk should be Some");

    assert_eq!(
        after_seek_first.samples.len(),
        first_samples.len(),
        "chunk length changed after seek"
    );
    for (i, (a, b)) in first_samples
        .iter()
        .zip(after_seek_first.samples.iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-6,
            "post-seek mismatch at sample {i}: before={a}, after={b}"
        );
    }
}

#[test]
fn test_audio_source_trait_usable() {
    // Verify that StreamingDecoder can be used through the AudioSource trait object.
    let n_frames = 4096usize;
    let wav_bytes = make_stereo_wav_bytes(n_frames);
    let file = write_temp_wav(&wav_bytes);
    let dec = StreamingDecoder::new(file, 1024).expect("StreamingDecoder::new should succeed");

    // Box it as an AudioSource trait object to confirm object-safety holds.
    let mut src: Box<dyn AudioSource> = Box::new(dec);
    let chunk = src.read_chunk().expect("read_chunk through trait object");
    assert!(chunk.is_some(), "expected at least one chunk");
    let buf = chunk.unwrap();
    assert_eq!(buf.channels, ChannelLayout::Stereo);
    assert_eq!(buf.sample_rate, 48_000);
}
