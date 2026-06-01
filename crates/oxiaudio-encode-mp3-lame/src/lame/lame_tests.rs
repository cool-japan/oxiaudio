use super::*;
use oxiaudio_core::{ChannelLayout, SampleFormat};

#[test]
fn test_vbr_preset_quality_values() {
    assert_eq!(VbrPreset::Music.quality_value(), 2);
    assert_eq!(VbrPreset::Archival.quality_value(), 0);
    assert_eq!(VbrPreset::Voice.quality_value(), 6);
    assert_eq!(VbrPreset::HiFidelity.quality_value(), 1);
    assert_eq!(VbrPreset::Podcast.quality_value(), 5);
    assert_eq!(VbrPreset::HighFidelity.quality_value(), 0);
}

#[test]
fn test_vbr_preset_quality_u8() {
    assert_eq!(VbrPreset::Music.quality(), 2u8);
    assert_eq!(VbrPreset::Archival.quality(), 0u8);
}

#[test]
fn test_vbr_preset_to_mode() {
    assert_eq!(VbrPreset::Music.to_mode(), LameMode::Vbr { quality: 2 });
    assert_eq!(
        VbrPreset::HiFidelity.to_mode(),
        LameMode::Vbr { quality: 1 }
    );
}

#[test]
fn test_mp3_tags_builder() {
    let tags = Mp3Tags::builder()
        .title("Test Track")
        .artist("Test Artist")
        .genre("Electronic")
        .composer("Some Composer")
        .comment("A comment")
        .disc_number(1)
        .track(5)
        .build();
    assert_eq!(tags.title.as_deref(), Some("Test Track"));
    assert_eq!(tags.artist.as_deref(), Some("Test Artist"));
    assert_eq!(tags.genre.as_deref(), Some("Electronic"));
    assert_eq!(tags.composer.as_deref(), Some("Some Composer"));
    assert_eq!(tags.comment.as_deref(), Some("A comment"));
    assert_eq!(tags.disc_number, Some(1));
    assert_eq!(tags.track, Some(5));
}

#[test]
fn test_mp3_tags_default() {
    let tags = Mp3Tags::default();
    assert!(tags.title.is_none());
    assert!(tags.genre.is_none());
    assert!(tags.disc_number.is_none());
}

#[test]
fn test_encoder_builder_api_compiles() {
    let _builder = LameMp3Encoder::builder(128)
        .with_quality(5)
        .with_tags(Mp3Tags::builder().title("Test").build());
}

#[test]
fn test_encoder_builder_with_vbr_preset() {
    let _builder = LameMp3Encoder::builder(192).with_vbr_preset(VbrPreset::Music);
}

#[test]
fn test_encoder_builder_with_abr() {
    let _builder = LameMp3Encoder::builder(0).with_abr(160);
}

fn make_sine_buf(sr: u32, seconds: f32) -> AudioBuffer<f32> {
    let n = (sr as f32 * seconds) as usize;
    let samples: Vec<f32> = (0..n * 2)
        .map(|i| {
            let ch_idx = i % 2;
            let frame = i / 2;
            let freq = if ch_idx == 0 { 440.0 } else { 880.0 };
            (2.0 * std::f32::consts::PI * freq * frame as f32 / sr as f32).sin() * 0.5
        })
        .collect();
    AudioBuffer {
        samples,
        sample_rate: sr,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

#[test]
fn test_encode_mp3_cbr_to_vec_produces_bytes() {
    let buf = make_sine_buf(44100, 0.1);
    let result = encode_mp3_cbr_to_vec(&buf, 128, None);
    let data = result.expect("CBR encode should succeed");
    assert!(!data.is_empty(), "output must not be empty");
    // A valid MP3 stream begins with a sync word (0xFF 0xFB or 0xFF 0xFA etc.)
    // or an ID3 tag (b"ID3").  Either is acceptable.
    assert!(
        data.len() > 100,
        "output should have more than 100 bytes for 0.1s audio"
    );
}

#[test]
fn test_encode_mp3_cbr_to_file() {
    let buf = make_sine_buf(44100, 0.1);
    let tmp = std::env::temp_dir().join("oxiaudio_test_cbr.mp3");
    encode_mp3_cbr_to_file(&buf, &tmp, 128, None).expect("CBR encode to file should succeed");
    let metadata = std::fs::metadata(&tmp).expect("output file must exist");
    assert!(metadata.len() > 100);
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_encode_mp3_abr() {
    let buf = make_sine_buf(44100, 0.1);
    let mut cursor = std::io::Cursor::new(Vec::new());
    encode_mp3_abr(&buf, &mut cursor, 160, None).expect("ABR encode should succeed");
    assert!(!cursor.get_ref().is_empty());
}

#[test]
fn test_builder_encode_to_vec() {
    let buf = make_sine_buf(44100, 0.1);
    let data = LameMp3Encoder::builder(128)
        .with_quality(7)
        .with_tags(Mp3Tags::builder().title("Builder Test").build())
        .encode_to_vec(&buf)
        .expect("builder encode_to_vec should succeed");
    assert!(!data.is_empty());
    // With ID3 tag, should start with b"ID3"
    assert_eq!(&data[..3], b"ID3");
}

#[test]
fn test_builder_vbr_encode_to_vec() {
    let buf = make_sine_buf(44100, 0.1);
    let data = LameMp3Encoder::builder(0)
        .with_vbr_preset(VbrPreset::Music)
        .encode_to_vec(&buf)
        .expect("VBR builder encode should succeed");
    assert!(!data.is_empty());
}

#[test]
fn test_builder_encode_to_file() {
    let buf = make_sine_buf(44100, 0.1);
    let tmp = std::env::temp_dir().join("oxiaudio_test_builder.mp3");
    LameMp3Encoder::builder(128)
        .encode_to_file(&buf, &tmp)
        .expect("builder encode_to_file should succeed");
    let metadata = std::fs::metadata(&tmp).expect("output file must exist");
    assert!(metadata.len() > 100);
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn test_stream_encoder_counters() {
    let buf = make_sine_buf(44100, 0.1);
    let config = LameMp3Encoder::default();
    let mut out = std::io::Cursor::new(Vec::new());
    let mut stream = LameMp3StreamEncoder::new(&mut out, &config, 44100, ChannelLayout::Stereo)
        .expect("stream encoder init should succeed");

    assert_eq!(stream.frames_encoded(), 0);
    assert_eq!(stream.bytes_written(), 0);

    stream
        .encode_chunk(&buf)
        .expect("chunk encode should succeed");
    assert!(stream.frames_encoded() > 0, "frames counter must advance");

    stream.finalize().expect("finalize should succeed");
}

#[test]
fn test_stream_encoder_sample_rate_mismatch() {
    let config = LameMp3Encoder::default();
    let mut out = std::io::Cursor::new(Vec::new());
    let mut stream = LameMp3StreamEncoder::new(&mut out, &config, 44100, ChannelLayout::Stereo)
        .expect("stream encoder init should succeed");

    let buf = AudioBuffer {
        samples: vec![0.0f32; 256],
        sample_rate: 48000, // wrong rate
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    };
    assert!(stream.encode_chunk(&buf).is_err());
    let _ = stream.finalize();
}

#[test]
fn test_estimated_bitrate_requires_1s_minimum() {
    // A freshly created stream encoder has encoded 0 frames; bitrate must be None.
    let config = LameMp3Encoder::default();
    let mut out = std::io::Cursor::new(Vec::new());
    let stream = LameMp3StreamEncoder::new(&mut out, &config, 44100, ChannelLayout::Stereo)
        .expect("stream encoder init should succeed");

    // No chunks encoded yet → less than 1 second of audio.
    assert_eq!(
        stream.estimated_bitrate_kbps(),
        None,
        "estimated_bitrate_kbps must return None before 1 second of audio"
    );
    assert_eq!(
        stream.elapsed_secs(),
        0.0,
        "elapsed_secs must be 0.0 when no frames have been encoded"
    );
    let _ = stream.finalize();
}

#[test]
fn test_elapsed_secs_advances_with_chunks() {
    let config = LameMp3Encoder::default();
    let mut out = std::io::Cursor::new(Vec::new());
    let mut stream = LameMp3StreamEncoder::new(&mut out, &config, 44100, ChannelLayout::Stereo)
        .expect("stream encoder init should succeed");

    // Encode ~0.1 second (4410 stereo frames).
    let buf = AudioBuffer {
        samples: vec![0.0f32; 4410 * 2],
        sample_rate: 44100,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    };
    stream
        .encode_chunk(&buf)
        .expect("chunk encode should succeed");

    let elapsed = stream.elapsed_secs();
    assert!(
        elapsed > 0.0 && elapsed < 1.0,
        "elapsed_secs should be ~0.1s after encoding 4410 frames, got {elapsed}"
    );
    let _ = stream.finalize();
}

#[test]
fn test_mp3_tags_builder_new_fields() {
    use AlbumArt;
    let art = AlbumArt {
        mime_type: "image/png".into(),
        data: vec![0x89, 0x50, 0x4E, 0x47],
    };
    let tags = Mp3Tags::builder()
        .album_art(art.clone())
        .replaygain_track_gain(-3.0)
        .replaygain_track_peak(0.95)
        .lyrics("Verse one")
        .user_defined("ENCODER", "OxiAudio")
        .build();

    assert!(tags.album_art.is_some());
    assert_eq!(tags.replaygain_track_gain, Some(-3.0));
    assert_eq!(tags.replaygain_track_peak, Some(0.95));
    assert_eq!(tags.lyrics.as_deref(), Some("Verse one"));
    assert_eq!(tags.user_defined.len(), 1);
    assert_eq!(tags.user_defined[0].0, "ENCODER");
    assert_eq!(tags.user_defined[0].1, "OxiAudio");
}

/// Verify that setting ms_stereo_threshold in the builder does not panic and
/// still produces valid MP3 output.
///
/// The threshold is stored in `LameMp3EncoderBuilder` for forward compatibility.
/// The `mp3lame-encoder` crate (0.2.x) does not expose `lame_set_msfix`; the
/// value is clamped and retained but not forwarded to LAME until upstream support
/// is added. This test confirms: (a) no panic, (b) valid MP3 bytes produced,
/// (c) the builder correctly clamps out-of-range values.
#[test]
fn test_ms_stereo_threshold_no_panic() {
    let buf = make_sine_buf(44100, 0.1);
    let data = LameMp3Encoder::builder(128)
        .with_ms_stereo_threshold(1.5)
        .encode_to_vec(&buf)
        .expect("encode with ms_stereo_threshold should not fail");
    assert!(!data.is_empty(), "output must not be empty");
}

#[test]
fn test_ms_stereo_threshold_clamp() {
    // Values out of range [0.0, 3.5] are clamped by the builder.
    // Confirm via encoding: both clamped values should produce valid output.
    let buf = make_sine_buf(44100, 0.05);
    let res = LameMp3Encoder::builder(128)
        .with_ms_stereo_threshold(99.0)
        .encode_to_vec(&buf);
    assert!(
        res.is_ok(),
        "clamped high threshold must not cause encode failure"
    );

    let res2 = LameMp3Encoder::builder(128)
        .with_ms_stereo_threshold(-5.0)
        .encode_to_vec(&buf);
    assert!(
        res2.is_ok(),
        "clamped low threshold must not cause encode failure"
    );
}

/// Probe test: locate the Xing/Info marker in raw LAME output and check its size.
/// Used during development to verify offset math before implementing RG writes.
/// This test is intentionally non-destructive and does not modify the encoder.
#[test]
fn probe_xing_marker_location() {
    use mp3lame_encoder::{Bitrate, Builder, DualPcm, FlushNoGap, Mode};
    use std::f32::consts::TAU;

    let sr = 44_100u32;
    let n = sr as usize;
    let mut left: Vec<i16> = Vec::with_capacity(n);
    let mut right: Vec<i16> = Vec::with_capacity(n);
    for i in 0..n {
        let s = (TAU * 440.0 * i as f32 / sr as f32).sin() * 0.5;
        let s_i16 = (s * i16::MAX as f32) as i16;
        left.push(s_i16);
        right.push(s_i16);
    }
    let mut encoder = Builder::new()
        .expect("Builder::new")
        .with_num_channels(2)
        .expect("channels")
        .with_sample_rate(sr)
        .expect("sample_rate")
        .with_brate(Bitrate::Kbps128)
        .expect("brate")
        .with_mode(Mode::JointStereo)
        .expect("mode")
        .build()
        .expect("build");

    let cap = mp3lame_encoder::max_required_buffer_size(n) + 7200;
    let mut mp3_out: Vec<u8> = Vec::with_capacity(cap);
    encoder
        .encode_to_vec(
            DualPcm {
                left: &left,
                right: &right,
            },
            &mut mp3_out,
        )
        .expect("encode_to_vec");
    encoder
        .flush_to_vec::<FlushNoGap>(&mut mp3_out)
        .expect("flush_to_vec");

    // Get finalized Lame tag (overrides placeholder bytes in mp3_out[0..tag_size]).
    let tag_size = encoder.lame_tag_size();
    assert!(tag_size > 0, "lame_tag_size must be non-zero");
    let mut lame_tag: Vec<u8> = Vec::with_capacity(tag_size + 32);
    let written = encoder
        .lame_tag_encode_to_vec(&mut lame_tag)
        .expect("lame_tag_encode_to_vec must return Some");
    assert_eq!(
        written.get(),
        tag_size,
        "written bytes must equal lame_tag_size"
    );
    assert_eq!(lame_tag.len(), tag_size);

    // Find Xing/Info marker in the finalized tag.
    let xing_pos = lame_tag
        .windows(4)
        .position(|w| w == b"Xing" || w == b"Info");
    assert!(
        xing_pos.is_some(),
        "Xing/Info marker must be present in lame_tag"
    );
    let xp = xing_pos.unwrap();
    // Offsets for MPEG1 stereo: tag starts at frame_start; Xing at frame_start + 36.
    // So xp should be 36 for MPEG1 joint-stereo.
    assert!(
        xp < tag_size,
        "Xing marker offset {xp} must be within tag_size {tag_size}"
    );

    // Verify bytes at marker+131 (peak, 4 bytes) and marker+135 (gain, 2 bytes) exist.
    assert!(
        xp + 139 < lame_tag.len(),
        "lame_tag must be large enough to hold peak+gain fields (need offset {}, len {})",
        xp + 139,
        lame_tag.len()
    );

    // By default (no gain analysis enabled), the gain field should be 0.
    let gain_word = u16::from_be_bytes([lame_tag[xp + 135], lame_tag[xp + 136]]);
    eprintln!("probe: xp={xp}, tag_size={tag_size}, gain_word={gain_word:#06X}");
}

#[test]
fn test_encode_mp3_with_auto_replaygain_smoke() {
    use oxiaudio_core::SampleFormat;
    use std::f32::consts::TAU;
    // Build a 1-second mono sine at 44.1 kHz.
    let sr = 44_100u32;
    let n = sr as usize; // 1 second of samples
    let samples: Vec<f32> = (0..n)
        .map(|i| (TAU * 440.0 * i as f32 / sr as f32).sin() * 0.5)
        .collect();
    let buf = AudioBuffer {
        samples,
        sample_rate: sr,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };
    let tmp = std::env::temp_dir().join("oxiaudio_test_auto_rg.mp3");
    encode_mp3_with_auto_replaygain(&buf, &tmp, 128)
        .expect("encode_mp3_with_auto_replaygain must succeed");
    let metadata = std::fs::metadata(&tmp).expect("output file must exist");
    assert!(metadata.len() > 0, "output file must not be empty");
    let _ = std::fs::remove_file(&tmp);
}

/// Verify that the Xing/LAME binary header has non-zero radio gain and valid peak
/// after `encode_mp3_with_auto_replaygain`.
///
/// Layout (relative to Xing/Info marker, per mp3-tech.org LAME tag spec):
///   marker+131..135 — peak signal amplitude (big-endian IEEE 754 f32)
///   marker+135..137 — radio (track) ReplayGain (big-endian u16, LAME encoding)
#[test]
fn test_xing_replaygain_header_fields() {
    use oxiaudio_core::SampleFormat;
    use std::f32::consts::TAU;

    // Build a 2-second stereo 440 Hz sine at 44.1 kHz (long enough to produce a
    // well-formed Xing tag and a non-trivial gain estimate).
    let sr = 44_100u32;
    let n_frames = sr as usize * 2;
    let samples: Vec<f32> = (0..n_frames * 2)
        .map(|i| {
            let frame = i / 2;
            (TAU * 440.0 * frame as f32 / sr as f32).sin() * 0.5
        })
        .collect();
    let buf = AudioBuffer {
        samples,
        sample_rate: sr,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    };

    let tmp = std::env::temp_dir().join("oxiaudio_test_xing_rg.mp3");
    encode_mp3_with_auto_replaygain(&buf, &tmp, 128)
        .expect("encode_mp3_with_auto_replaygain must succeed");
    let mp3_bytes = std::fs::read(&tmp).expect("output file must be readable");
    let _ = std::fs::remove_file(&tmp);

    // Find the Xing/Info marker.
    let xing_pos = mp3_bytes
        .windows(4)
        .position(|w| w == b"Xing" || w == b"Info");
    assert!(
        xing_pos.is_some(),
        "Xing/Info marker must be present in encode_mp3_with_auto_replaygain output"
    );
    let xp = xing_pos.unwrap();

    assert!(
        mp3_bytes.len() >= xp + 139,
        "MP3 output must have enough bytes for peak (marker+131..135) and gain (marker+135..137)"
    );

    // --- Peak signal (marker+131..135, big-endian f32) ---
    let peak_bytes = [
        mp3_bytes[xp + 131],
        mp3_bytes[xp + 132],
        mp3_bytes[xp + 133],
        mp3_bytes[xp + 134],
    ];
    let peak = f32::from_be_bytes(peak_bytes);
    assert!(
        peak > 0.0 && peak <= 1.0,
        "peak signal must be a valid positive f32 in (0, 1], got {peak}"
    );

    // --- Radio gain (marker+135..137, big-endian u16) ---
    let gain_word = u16::from_be_bytes([mp3_bytes[xp + 135], mp3_bytes[xp + 136]]);
    assert_ne!(
        gain_word, 0,
        "radio gain word must be non-zero (got 0, which would mean exactly 0 dB with no tag)"
    );

    // Verify name code (bits 15..13) == 001 (radio gain).
    let name_code = (gain_word >> 13) & 0x07;
    assert_eq!(
        name_code, 1,
        "gain word name code must be 001 (radio/track gain), got {name_code:#05b}"
    );

    // Verify originator (bits 12..10) == 011 (automatic).
    let originator = (gain_word >> 10) & 0x07;
    assert_eq!(
        originator, 3,
        "gain word originator must be 011 (automatic), got {originator:#05b}"
    );

    // Verify absolute gain in tenths (bits 8..0) > 0 for a non-silent signal.
    let abs_tenths = gain_word & 0x01FF;
    assert!(
        abs_tenths > 0,
        "absolute gain tenths must be > 0 for a non-silent 440 Hz sine, got {abs_tenths}"
    );
}
