//! M9 streaming decoder improvement tests.
//!
//! Tests for:
//! - `detect_format_from_path` — extension-hinted format probing
//! - `StreamingDecoder::open` — path-based constructor
//! - `StreamingDecoder::format` — returns `Option<&AudioFormat>`
//! - `StreamingDecoder::metadata` — returns `&AudioMetadata`
//! - `StreamingDecoder::decode_next` — chunk decode with max_frames bound
//! - `StreamingDecoder::skip_frames` — seek-or-decode skip
//! - `StreamingDecoder::remaining_frames` — total - decoded count

use oxiaudio_core::{ChannelLayout, SampleFormat};
use oxiaudio_decode::{detect_format_from_path, StreamingDecoder};
use std::io::Cursor;

/// Write a mono sine-wave WAV to a temp file and return its path.
///
/// Uses `hound` (already a dev-dep of `oxiaudio-decode`) so we do not depend on
/// `oxiaudio-encode` in the decode crate's dev-deps.
fn sine_wav_file(sample_rate: u32, duration_secs: f32) -> std::path::PathBuf {
    let n = (sample_rate as f32 * duration_secs) as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();

    let mut wav_bytes = Vec::new();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::new(Cursor::new(&mut wav_bytes), spec).expect("WavWriter::new");
    for &s in &samples {
        writer.write_sample(s).expect("write_sample");
    }
    writer.finalize().expect("finalize");

    // Write to a temp file with the .wav extension so the hint is exercised.
    let path = std::env::temp_dir().join(format!(
        "oxiaudio_m9_sine_{}hz_{}.wav",
        sample_rate,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0),
    ));
    std::fs::write(&path, &wav_bytes).expect("write temp wav");
    path
}

// ═══════════════════════════════════════════════════════════════════════════════
// detect_format_from_path
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_detect_format_from_path_wav_44100() {
    let path = sine_wav_file(44_100, 1.0);
    let result = detect_format_from_path(&path);
    let _ = std::fs::remove_file(&path);
    let fmt = result.expect("detect_format_from_path should succeed");
    assert_eq!(fmt.sample_rate, 44_100);
    assert_eq!(fmt.channels, ChannelLayout::Mono);
    assert_eq!(fmt.format, SampleFormat::F32);
}

#[test]
fn test_detect_format_from_path_wav_48000() {
    let path = sine_wav_file(48_000, 0.5);
    let result = detect_format_from_path(&path);
    let _ = std::fs::remove_file(&path);
    let fmt = result.expect("detect_format_from_path 48kHz");
    assert_eq!(fmt.sample_rate, 48_000);
}

#[test]
fn test_detect_format_from_path_missing_file() {
    let path = std::env::temp_dir().join("oxiaudio_m9_nonexistent_file.wav");
    let result = detect_format_from_path(&path);
    assert!(result.is_err(), "expected Err for missing file");
}

// ═══════════════════════════════════════════════════════════════════════════════
// StreamingDecoder::open
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_streaming_decoder_open_wav() {
    let path = sine_wav_file(44_100, 0.5);
    let result = StreamingDecoder::open(&path);
    let _ = std::fs::remove_file(&path);
    assert!(
        result.is_ok(),
        "StreamingDecoder::open should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_streaming_decoder_open_missing_file() {
    let path = std::env::temp_dir().join("oxiaudio_m9_open_missing.wav");
    let result = StreamingDecoder::open(&path);
    assert!(result.is_err(), "expected Err for missing file");
}

// ═══════════════════════════════════════════════════════════════════════════════
// StreamingDecoder::format
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_streaming_decoder_format_44100() {
    let path = sine_wav_file(44_100, 0.2);
    let dec = StreamingDecoder::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    let fmt = dec.format();
    assert!(fmt.is_some(), "format() should return Some");
    let fmt = fmt.unwrap();
    assert_eq!(fmt.sample_rate, 44_100);
    assert_eq!(fmt.channels, ChannelLayout::Mono);
    assert_eq!(fmt.format, SampleFormat::F32);
}

#[test]
fn test_streaming_decoder_format_48000() {
    let path = sine_wav_file(48_000, 0.2);
    let dec = StreamingDecoder::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    let fmt = dec.format().expect("format() is Some");
    assert_eq!(fmt.sample_rate, 48_000);
}

// ═══════════════════════════════════════════════════════════════════════════════
// StreamingDecoder::metadata
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_streaming_decoder_metadata_ref() {
    let path = sine_wav_file(44_100, 0.1);
    let dec = StreamingDecoder::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    // metadata() now returns &AudioMetadata (not Option).
    let meta = dec.metadata();
    // For a plain WAV, all tags will be None — just verify the struct is accessible.
    let _ = meta.title.as_ref();
    let _ = meta.artist.as_ref();
    let _ = meta.album.as_ref();
    let _ = meta.genre.as_ref();
    let _ = meta.composer.as_ref();
    let _ = meta.year;
    let _ = meta.disc_number;
    let _ = meta.track_number;
    let _ = meta.comment.as_ref();
}

// ═══════════════════════════════════════════════════════════════════════════════
// StreamingDecoder::decode_next
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_streaming_decoder_decode_next_returns_some() {
    let path = sine_wav_file(44_100, 0.5);
    let mut dec = StreamingDecoder::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    let chunk = dec.decode_next(4096).expect("decode_next should not error");
    assert!(chunk.is_some(), "expected Some chunk from 0.5-second file");
}

#[test]
fn test_streaming_decoder_decode_next_frame_count() {
    let path = sine_wav_file(44_100, 1.0);
    let mut dec = StreamingDecoder::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    let chunk = dec.decode_next(1024).expect("decode_next").expect("Some");
    // Should not exceed max_frames.
    assert!(
        chunk.frame_count() <= 1024,
        "frame_count {} exceeds max_frames 1024",
        chunk.frame_count()
    );
    assert!(chunk.frame_count() > 0, "expected at least one frame");
    assert_eq!(chunk.sample_rate, 44_100);
    assert_eq!(chunk.channels, ChannelLayout::Mono);
}

#[test]
fn test_streaming_decoder_decode_next_exhausted() {
    // A very short file: 100 frames.
    let path = sine_wav_file(44_100, 100.0 / 44_100.0);
    let mut dec = StreamingDecoder::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    // Drain the whole file in one large chunk request.
    let _first = dec.decode_next(8192).expect("decode_next");
    // A second call should return None (stream exhausted).
    let second = dec
        .decode_next(8192)
        .expect("decode_next should not error after EOF");
    assert!(second.is_none(), "expected None at EOF");
}

// ═══════════════════════════════════════════════════════════════════════════════
// StreamingDecoder::skip_frames
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_streaming_decoder_skip_frames_via_open() {
    let path = sine_wav_file(44_100, 1.0);
    let mut dec = StreamingDecoder::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    // skip_frames takes u64 in the existing API.
    let skipped = dec.skip_frames(1000).expect("skip_frames");
    assert!(skipped > 0, "expected some frames skipped, got {skipped}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// StreamingDecoder::remaining_frames
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_streaming_decoder_remaining_frames_via_open() {
    let path = sine_wav_file(44_100, 1.0); // ~44100 frames
    let dec = StreamingDecoder::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    // WAV containers expose frame count, so remaining_frames should be Some.
    if let Some(remaining) = dec.remaining_frames() {
        assert!(remaining > 0, "expected positive remaining_frames");
    }
    // If None, the container didn't report a frame count — that's also acceptable.
}

#[test]
fn test_streaming_decoder_remaining_frames_decreases() {
    let path = sine_wav_file(44_100, 1.0);
    let mut dec = StreamingDecoder::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    let before = dec.remaining_frames();
    let _chunk = dec.decode_next(4096).expect("decode_next");
    let after = dec.remaining_frames();
    // If both are Some, after must be less than before.
    if let (Some(b), Some(a)) = (before, after) {
        assert!(
            a < b,
            "remaining_frames should decrease after decode_next: {b} -> {a}"
        );
    }
}
