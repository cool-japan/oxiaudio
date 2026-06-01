//! M20 integration tests: full pipeline decode→DSP→encode roundtrips.

use oxiaudio::{
    decode_file, dsp, encode_flac, encode_flac_with_picture_file, encode_flac_with_seektable_file,
    encode_wav, encode_wav_with_cues_file, transcode_batch, CuePoint, FlacPicture, TranscodeStream,
};
use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

fn sine_buf(freq_hz: f32, seconds: f32, sample_rate: u32) -> AudioBuffer<f32> {
    let frames = (seconds * sample_rate as f32) as usize;
    let samples: Vec<f32> = (0..frames)
        .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

fn stereo_sine_buf(freq_hz: f32, seconds: f32, sample_rate: u32) -> AudioBuffer<f32> {
    let frames = (seconds * sample_rate as f32) as usize;
    let samples: Vec<f32> = (0..frames * 2)
        .map(|i| {
            let frame = i / 2;
            (2.0 * std::f32::consts::PI * freq_hz * frame as f32 / sample_rate as f32).sin() * 0.5
        })
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

/// Test 1: WAV encode + decode roundtrip (mono).
#[test]
#[cfg(feature = "pure")]
fn test_wav_roundtrip_mono() {
    let path = temp_path("m20_wav_roundtrip_mono.wav");
    let orig = sine_buf(440.0, 1.0, 44100);
    encode_wav(&orig, &path).expect("encode WAV");
    let decoded = decode_file(&path).expect("decode WAV");
    let _ = std::fs::remove_file(&path);

    assert_eq!(decoded.sample_rate, 44100);
    assert_eq!(decoded.channels, ChannelLayout::Mono);

    // F32 WAV roundtrip — sample count must match exactly.
    let expected_samples = 44100usize;
    assert!(
        (decoded.samples.len() as i64 - expected_samples as i64).abs() < 100,
        "sample count mismatch: {} vs {}",
        decoded.samples.len(),
        expected_samples
    );
}

/// Test 2: FLAC encode + decode roundtrip (stereo).
#[test]
#[cfg(feature = "pure")]
fn test_flac_roundtrip_stereo() {
    let path = temp_path("m20_flac_roundtrip_stereo.flac");
    let orig = stereo_sine_buf(880.0, 2.0, 48000);
    encode_flac(&orig, &path).expect("encode FLAC");
    let decoded = decode_file(&path).expect("decode FLAC");
    let _ = std::fs::remove_file(&path);

    assert_eq!(decoded.sample_rate, 48000);
    assert_eq!(decoded.channels, ChannelLayout::Stereo);
    assert!(
        !decoded.samples.is_empty(),
        "decoded buffer must not be empty"
    );
}

/// Test 3: FLAC with seektable.
#[test]
#[cfg(feature = "pure")]
fn test_flac_with_seektable() {
    let path = temp_path("m20_flac_seektable.flac");
    let buf = sine_buf(440.0, 3.0, 44100);
    // Third argument is compression_level (0–8).
    encode_flac_with_seektable_file(&buf, &path, 5).expect("encode FLAC with seektable");

    let decoded = decode_file(&path).expect("decode FLAC with seektable");
    assert_eq!(decoded.sample_rate, 44100);

    // Verify fLaC magic bytes.
    let bytes = std::fs::read(&path).expect("read file");
    let _ = std::fs::remove_file(&path);
    assert_eq!(&bytes[0..4], b"fLaC", "expected fLaC magic");
}

/// Test 4: FLAC with embedded album art (1×1 PNG).
#[test]
#[cfg(feature = "pure")]
fn test_flac_with_picture() {
    let path = temp_path("m20_flac_picture.flac");
    let buf = sine_buf(440.0, 1.0, 44100);

    // Minimal valid 1×1 RGB PNG.
    let png_data: Vec<u8> = vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
        0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8,
        0xCF, 0xC0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC, 0x33, 0x00, 0x00, 0x00,
        0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    let picture = FlacPicture::front_cover_png(png_data);
    encode_flac_with_picture_file(&buf, &path, &picture).expect("encode FLAC with picture");

    let bytes = std::fs::read(&path).expect("read file");
    let _ = std::fs::remove_file(&path);

    assert_eq!(&bytes[0..4], b"fLaC", "expected fLaC magic");
    assert!(
        bytes.len() > 200,
        "FLAC with picture must exceed 200 bytes, got {}",
        bytes.len()
    );
}

/// Test 5: WAV with embedded cue points.
#[test]
#[cfg(feature = "pure")]
fn test_wav_with_cue_points() {
    let path = temp_path("m20_wav_cues.wav");
    let buf = sine_buf(440.0, 5.0, 44100);
    let cues = vec![
        CuePoint::with_label(1, 44100, "verse"),
        CuePoint::with_label(2, 88200, "chorus"),
        CuePoint::new(3, 132300),
    ];
    encode_wav_with_cues_file(&buf, &path, &cues).expect("encode WAV with cues");

    let bytes = std::fs::read(&path).expect("read WAV file");
    let _ = std::fs::remove_file(&path);

    assert_eq!(&bytes[0..4], b"RIFF", "expected RIFF header");

    // Verify cue  chunk marker is present in the WAV file.
    let cue_marker: &[u8] = b"cue ";
    let has_cue = bytes.windows(4).any(|w| w == cue_marker);
    assert!(has_cue, "cue  chunk not found in WAV");
}

/// Test 6: DSP normalize pipeline then encode.
#[test]
#[cfg(feature = "pure")]
fn test_dsp_normalize_then_encode() {
    let path = temp_path("m20_normalized.wav");
    let mut buf = sine_buf(440.0, 1.0, 44100);

    // Scale to half amplitude first so normalize has work to do.
    for s in &mut buf.samples {
        *s *= 0.5;
    }

    dsp::normalize(&mut buf, -1.0); // normalize to -1 dBFS
    encode_wav(&buf, &path).expect("encode normalized WAV");

    let decoded = decode_file(&path).expect("decode normalized WAV");
    let _ = std::fs::remove_file(&path);

    // Peak after -1 dBFS normalize should be close to 0.891 linear (= 10^(-1/20)).
    let peak = decoded
        .samples
        .iter()
        .fold(0.0f32, |acc, &s| acc.max(s.abs()));
    assert!(
        peak > 0.85,
        "peak too low after normalize to -1 dBFS: {peak:.4}"
    );
}

/// Test 7: TranscodeStream WAV → FLAC.
#[test]
#[cfg(feature = "pure")]
fn test_transcode_stream_wav_to_flac() {
    let src = temp_path("m20_transcode_src.wav");
    let dst = temp_path("m20_transcode_dst.flac");

    let buf = stereo_sine_buf(440.0, 2.0, 44100);
    encode_wav(&buf, &src).expect("create source WAV");

    TranscodeStream::new(&src, &dst)
        .expect("create TranscodeStream")
        .run()
        .expect("transcode WAV→FLAC");

    let bytes = std::fs::read(&dst).expect("read FLAC");
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);

    assert_eq!(
        &bytes[0..4],
        b"fLaC",
        "expected fLaC magic in TranscodeStream output"
    );
}

/// Test 8: transcode_batch converts multiple WAV files to FLAC in parallel.
#[test]
#[cfg(feature = "pure")]
fn test_transcode_batch() {
    let out_dir = std::env::temp_dir();
    let src1 = temp_path("m20_batch_src1.wav");
    let src2 = temp_path("m20_batch_src2.wav");

    let buf = sine_buf(440.0, 0.5, 44100);
    encode_wav(&buf, &src1).expect("create src1");
    encode_wav(&buf, &src2).expect("create src2");

    let inputs: Vec<&std::path::Path> = vec![src1.as_path(), src2.as_path()];
    let results = transcode_batch(&inputs, &out_dir, "flac");

    let _ = std::fs::remove_file(&src1);
    let _ = std::fs::remove_file(&src2);

    assert_eq!(results.len(), 2, "should return one result per input");

    for (i, r) in results.into_iter().enumerate() {
        let out_path = r.unwrap_or_else(|e| panic!("transcode_batch item {i} failed: {e}"));
        assert!(
            out_path.exists(),
            "output file not created: {}",
            out_path.display()
        );
        let magic = std::fs::read(&out_path).expect("read batch output");
        assert_eq!(&magic[0..4], b"fLaC", "batch item {i} is not valid FLAC");
        let _ = std::fs::remove_file(&out_path);
    }
}
