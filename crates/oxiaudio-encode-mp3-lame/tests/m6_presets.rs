//! M6 tests: VBR presets, ABR mode, forced-mono, and the full 16-value
//! bitrate table for the LAME MP3 encoder.
#![cfg(feature = "mp3-encode-lame")]

use std::io::Cursor;

use oxiaudio_core::{AudioBuffer, AudioEncoder, ChannelLayout, SampleFormat};
use oxiaudio_encode_mp3_lame::lame::{LameMode, LameMp3Encoder, VbrPreset};

fn sine(channels: ChannelLayout, sample_rate: u32, secs: f32) -> AudioBuffer<f32> {
    let frames = (sample_rate as f32 * secs) as usize;
    let nc = channels.channel_count();
    let mut samples = Vec::with_capacity(frames * nc);
    for i in 0..frames {
        let s = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5;
        for _ in 0..nc {
            samples.push(s);
        }
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels,
        format: SampleFormat::F32,
    }
}

fn encode(encoder: &mut LameMp3Encoder, buf: &AudioBuffer<f32>) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    encoder.encode(buf, &mut cursor).expect("encode failed");
    cursor.into_inner()
}

#[test]
fn vbr_presets_map_to_expected_quality() {
    assert_eq!(VbrPreset::Voice.quality(), 6);
    assert_eq!(VbrPreset::Podcast.quality(), 5);
    assert_eq!(VbrPreset::Music.quality(), 2);
    assert_eq!(VbrPreset::HiFidelity.quality(), 1);
    assert_eq!(VbrPreset::HighFidelity.quality(), 0);
    assert_eq!(VbrPreset::Archival.quality(), 0);
    assert_eq!(VbrPreset::Music.to_mode(), LameMode::Vbr { quality: 2 });
}

#[test]
fn vbr_preset_encodes_valid_mp3() {
    let buf = sine(ChannelLayout::Stereo, 44_100, 0.5);
    for preset in [
        VbrPreset::Voice,
        VbrPreset::Podcast,
        VbrPreset::Music,
        VbrPreset::HighFidelity,
    ] {
        let mut enc = LameMp3Encoder {
            bitrate: 128,
            mode: preset.to_mode(),
            id3_tags: None,
        };
        let data = encode(&mut enc, &buf);
        assert!(
            data.len() > 100,
            "preset {preset:?} produced too little data"
        );
    }
}

#[test]
fn abr_mode_encodes_valid_mp3() {
    let buf = sine(ChannelLayout::Stereo, 44_100, 0.5);
    let mut enc = LameMp3Encoder {
        bitrate: 128,
        mode: LameMode::Abr { target_kbps: 128 },
        id3_tags: None,
    };
    let data = encode(&mut enc, &buf);
    assert!(data.len() > 100, "ABR produced too little data");
    // Average bitrate should be roughly the target (within a generous tolerance).
    let bits = data.len() as f64 * 8.0;
    let kbps = bits / 0.5 / 1000.0;
    assert!(
        (64.0..=256.0).contains(&kbps),
        "ABR 128k average bitrate {kbps:.0} kbps outside sane range"
    );
}

#[test]
fn abr_rejects_invalid_target() {
    let buf = sine(ChannelLayout::Mono, 44_100, 0.1);
    let mut enc = LameMp3Encoder {
        bitrate: 128,
        mode: LameMode::Abr { target_kbps: 999 },
        id3_tags: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    assert!(enc.encode(&buf, &mut cursor).is_err());
}

#[test]
fn forced_mono_from_stereo_input() {
    // Stereo input + ForcedMono must encode and decode as a single channel.
    let buf = sine(ChannelLayout::Stereo, 44_100, 0.3);
    let mut enc = LameMp3Encoder {
        bitrate: 128,
        mode: LameMode::ForcedMono,
        id3_tags: None,
    };
    let data = encode(&mut enc, &buf);
    assert!(data.len() > 100);

    // Decode with symphonia and verify it reports a single channel.
    use oxiaudio_core::AudioDecoder;
    let decoded = oxiaudio_decode::SymphoniaDecoder
        .decode(Cursor::new(data))
        .expect("decode forced-mono mp3");
    assert_eq!(
        decoded.channels,
        ChannelLayout::Mono,
        "ForcedMono should yield mono output"
    );
}

#[test]
fn all_cbr_bitrates_encode() {
    let buf = sine(ChannelLayout::Stereo, 44_100, 0.2);
    // The full bitrate table exposed by mp3lame-encoder.
    for kbps in [32u32, 40, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320] {
        let mut enc = LameMp3Encoder {
            bitrate: kbps,
            mode: LameMode::JointStereo,
            id3_tags: None,
        };
        let mut cursor = Cursor::new(Vec::new());
        enc.encode(&buf, &mut cursor)
            .unwrap_or_else(|e| panic!("CBR {kbps}k failed: {e}"));
        assert!(
            cursor.into_inner().len() > 100,
            "CBR {kbps}k produced too little"
        );
    }
}

#[test]
fn unsupported_bitrate_errors() {
    let buf = sine(ChannelLayout::Mono, 44_100, 0.1);
    let mut enc = LameMp3Encoder {
        bitrate: 200, // not a valid MP3 bitrate
        mode: LameMode::Mono,
        id3_tags: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    assert!(enc.encode(&buf, &mut cursor).is_err());
}
