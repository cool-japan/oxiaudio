//! Extracted from `oxiaudio-decode/src/wav_cue.rs` to break the decode→encode
//! dev-dependency cycle. These WAV-cue roundtrip tests synthesize input via
//! `oxiaudio_encode::encode_wav_with_cues` and verify `oxiaudio_decode`'s parser.

use std::io::Cursor;

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use oxiaudio_decode::{parse_wav_cues, parse_wav_cues_reader, WavCuePoint};
use oxiaudio_encode::{encode_wav_with_cues, CuePoint};

// ── helpers ───────────────────────────────────────────────────────────────

fn make_audio_buffer(frames: usize) -> AudioBuffer<f32> {
    AudioBuffer {
        samples: vec![0.0f32; frames],
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

fn build_wav_bytes_no_cues(frames: usize) -> Vec<u8> {
    let buf = make_audio_buffer(frames);
    let mut cursor = Cursor::new(Vec::new());
    encode_wav_with_cues(&buf, &mut cursor, &[]).expect("encode_wav_with_cues");
    cursor.into_inner()
}

fn build_wav_bytes_with_cues(frames: usize, cues: &[CuePoint]) -> Vec<u8> {
    let buf = make_audio_buffer(frames);
    let mut cursor = Cursor::new(Vec::new());
    encode_wav_with_cues(&buf, &mut cursor, cues).expect("encode_wav_with_cues");
    cursor.into_inner()
}

// ── tests ─────────────────────────────────────────────────────────────────

/// Encoding a WAV without cues and parsing it should yield an empty Vec.
#[test]
fn test_parse_wav_cues_no_cues() {
    let bytes = build_wav_bytes_no_cues(4096);
    let mut cursor = Cursor::new(bytes);
    let result = parse_wav_cues_reader(&mut cursor).expect("parse_wav_cues_reader");
    assert!(
        result.is_empty(),
        "expected empty cue list for a WAV without a cue chunk, got {:?}",
        result
    );
}

/// Write a WAV with cues via oxiaudio-encode, parse it back, verify positions and labels.
#[test]
fn test_parse_wav_cues_roundtrip() {
    let cues_in = vec![
        CuePoint::with_label(1, 100, "Intro"),
        CuePoint::new(2, 500),
        CuePoint::with_label(3, 1000, "Verse"),
    ];
    let bytes = build_wav_bytes_with_cues(4096, &cues_in);
    let mut cursor = Cursor::new(bytes);
    let parsed = parse_wav_cues_reader(&mut cursor).expect("parse_wav_cues_reader");

    assert_eq!(
        parsed.len(),
        cues_in.len(),
        "number of parsed cue points must match encoded count"
    );

    for (expected, got) in cues_in.iter().zip(parsed.iter()) {
        assert_eq!(
            got.id, expected.id,
            "cue id mismatch: expected {}, got {}",
            expected.id, got.id
        );
        assert_eq!(
            got.position, expected.position,
            "cue position mismatch for id={}: expected {}, got {}",
            expected.id, expected.position, got.position
        );
        assert_eq!(
            got.label.as_deref(),
            expected.label.as_deref(),
            "cue label mismatch for id={}: expected {:?}, got {:?}",
            expected.id,
            expected.label,
            got.label
        );
    }
}

/// File-path API: write a WAV to a temp file and parse via `parse_wav_cues`.
#[test]
fn test_parse_wav_cues_file_path() {
    use std::io::Write;

    let cues_in = vec![CuePoint::with_label(7, 44_100, "LoopStart")];
    let bytes = build_wav_bytes_with_cues(88_200, &cues_in);

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("oxiaudio_decode_wav_cue_{ts}.wav"));
    {
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(&bytes).expect("write wav bytes");
    }

    let parsed = parse_wav_cues(&path).expect("parse_wav_cues");
    let _ = std::fs::remove_file(&path);

    assert_eq!(parsed.len(), 1, "expected exactly 1 cue point");
    assert_eq!(
        parsed[0],
        WavCuePoint {
            id: 7,
            position: 44_100,
            label: Some("LoopStart".to_string()),
        }
    );
}
