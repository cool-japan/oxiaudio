#![cfg(feature = "mp3-encode-lame")]

use oxiaudio_core::{AudioBuffer, AudioEncoder, ChannelLayout, SampleFormat};
use oxiaudio_encode_mp3_lame::lame::{LameMode, LameMp3Encoder, Mp3Tags};
use std::io::Cursor;

/// Generate a 1-second stereo 440 Hz sine wave at the given sample rate.
fn sine_stereo_1s(sample_rate: u32) -> AudioBuffer<f32> {
    let n_frames = sample_rate as usize;
    let mut samples = Vec::with_capacity(n_frames * 2);
    for i in 0..n_frames {
        let s = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5;
        samples.push(s);
        samples.push(s);
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

/// VBR V0 (best quality) should produce more bytes than VBR V9 (worst quality)
/// when encoding the same audio signal.
#[test]
fn test_vbr_quality_size_ordering() {
    let buf = sine_stereo_1s(44_100);

    let mut enc_v0 = LameMp3Encoder {
        bitrate: 0, // ignored for VBR
        mode: LameMode::Vbr { quality: 0 },
        id3_tags: None,
    };
    let mut out_v0 = Cursor::new(Vec::new());
    enc_v0
        .encode(&buf, &mut out_v0)
        .expect("VBR V0 encode should succeed");
    let v0_size = out_v0.into_inner().len();

    let mut enc_v9 = LameMp3Encoder {
        bitrate: 0,
        mode: LameMode::Vbr { quality: 9 },
        id3_tags: None,
    };
    let mut out_v9 = Cursor::new(Vec::new());
    enc_v9
        .encode(&buf, &mut out_v9)
        .expect("VBR V9 encode should succeed");
    let v9_size = out_v9.into_inner().len();

    assert!(
        v0_size > v9_size,
        "VBR V0 (best quality) must produce more bytes than VBR V9 (worst quality): \
         V0={v0_size} bytes, V9={v9_size} bytes"
    );
}

/// VBR quality out of range (10) must return an error, not panic.
#[test]
fn test_vbr_quality_out_of_range() {
    let buf = sine_stereo_1s(44_100);
    let mut enc = LameMp3Encoder {
        bitrate: 0,
        mode: LameMode::Vbr { quality: 10 },
        id3_tags: None,
    };
    let mut out = Cursor::new(Vec::new());
    let result = enc.encode(&buf, &mut out);
    assert!(result.is_err(), "quality 10 should be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("10"),
        "error should mention quality 10, got: {msg}"
    );
}

/// Encoding with ID3 tags must produce output whose first 3 bytes are "ID3"
/// and which contains the TRCK frame somewhere in the tag region.
#[test]
fn test_id3_tags_present() {
    let buf = sine_stereo_1s(44_100);
    let tags = Mp3Tags {
        title: Some("Test".into()),
        track_number: Some(3),
        year: Some(2024),
        ..Default::default()
    };
    let mut enc = LameMp3Encoder {
        bitrate: 128,
        mode: LameMode::JointStereo,
        id3_tags: Some(tags),
    };
    let mut out = Cursor::new(Vec::new());
    enc.encode(&buf, &mut out)
        .expect("encode with tags should succeed");
    let data = out.into_inner();

    // The output must start with the ID3v2 magic.
    assert!(
        data.starts_with(b"ID3"),
        "output must start with ID3 magic, got: {:?}",
        &data[..3.min(data.len())]
    );

    // TRCK frame must be present in the tag region.
    let trck_pos = data.windows(4).position(|w| w == b"TRCK");
    assert!(
        trck_pos.is_some(),
        "TRCK frame must be present in the ID3 tag"
    );

    // TIT2 (title) must also be present.
    let tit2_pos = data.windows(4).position(|w| w == b"TIT2");
    assert!(
        tit2_pos.is_some(),
        "TIT2 frame must be present in the ID3 tag"
    );
}

/// Encoding without ID3 tags must NOT start with "ID3".
#[test]
fn test_no_tags_no_id3_header() {
    let buf = sine_stereo_1s(44_100);
    let mut enc = LameMp3Encoder::default();
    let mut out = Cursor::new(Vec::new());
    enc.encode(&buf, &mut out)
        .expect("encode without tags should succeed");
    let data = out.into_inner();
    assert!(
        !data.starts_with(b"ID3"),
        "output without tags must not start with ID3 magic"
    );
    // MP3 sync word: first byte 0xFF, second byte 0xE0–0xFF (layer bits set).
    assert!(
        data.first().copied() == Some(0xFF),
        "bare MP3 output should start with sync byte 0xFF"
    );
}

/// All five tag fields round-trip into the serialised bytes.
#[test]
fn test_all_id3_fields_present() {
    let buf = sine_stereo_1s(44_100);
    let tags = Mp3Tags {
        title: Some("Song Title".into()),
        artist: Some("Artist Name".into()),
        album: Some("Album Name".into()),
        track_number: Some(7),
        year: Some(1999),
        ..Default::default()
    };
    let mut enc = LameMp3Encoder {
        bitrate: 128,
        mode: LameMode::JointStereo,
        id3_tags: Some(tags),
    };
    let mut out = Cursor::new(Vec::new());
    enc.encode(&buf, &mut out)
        .expect("encode with all tags should succeed");
    let data = out.into_inner();

    for (label, frame_id) in &[
        ("TIT2", b"TIT2"),
        ("TPE1", b"TPE1"),
        ("TALB", b"TALB"),
        ("TRCK", b"TRCK"),
        ("TDRC", b"TDRC"),
    ] {
        assert!(
            data.windows(4).any(|w| w == frame_id.as_slice()),
            "{label} frame must be present in the ID3 tag"
        );
    }
}
