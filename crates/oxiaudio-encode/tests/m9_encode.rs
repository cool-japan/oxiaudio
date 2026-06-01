use std::io::BufWriter;

use oxiaudio_core::{
    AudioBuffer, AudioEncoder, AudioMetadata, AudioSink, ChannelLayout, SampleFormat,
};
use oxiaudio_encode::{AiffBitDepth, AiffStreamEncoder};

fn sine_mono(sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
    let n = (sample_rate as f32 * duration_secs) as usize;
    let samples = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

#[test]
fn test_aiff_stream_roundtrip_i16() {
    let buf = sine_mono(44_100, 0.5);
    let path = std::env::temp_dir().join("oxiaudio_m9_aiff_stream.aiff");
    {
        let file = std::fs::File::create(&path).expect("create");
        let mut enc = AiffStreamEncoder::new(
            BufWriter::new(file),
            44_100,
            ChannelLayout::Mono,
            AiffBitDepth::I16,
        )
        .expect("new");
        for chunk in buf.samples.chunks(512) {
            let cb = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: 44_100,
                channels: ChannelLayout::Mono,
                format: SampleFormat::F32,
            };
            enc.encode_chunk(&cb).expect("encode_chunk");
        }
        enc.finalize().expect("finalize");
    }
    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert!(bytes.len() > 54, "file too small: {}", bytes.len());
    assert_eq!(&bytes[..4], b"FORM");
    assert_eq!(&bytes[8..12], b"AIFF");
}

#[test]
fn test_aiff_stream_audio_sink_trait() {
    let buf = sine_mono(44_100, 0.2);
    let path = std::env::temp_dir().join("oxiaudio_m9_aiff_sink.aiff");
    {
        let file = std::fs::File::create(&path).expect("create");
        let mut enc: AiffStreamEncoder<_> = AiffStreamEncoder::new(
            BufWriter::new(file),
            44_100,
            ChannelLayout::Mono,
            AiffBitDepth::I16,
        )
        .expect("new");
        // Use AudioSink trait
        for chunk in buf.samples.chunks(256) {
            let cb = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: 44_100,
                channels: ChannelLayout::Mono,
                format: SampleFormat::F32,
            };
            enc.write_chunk(&cb).expect("write_chunk via AudioSink");
        }
        enc.finalize().expect("finalize");
    }
    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(&bytes[..4], b"FORM");
}

#[test]
fn test_aiff_stream_frames_written() {
    let buf = sine_mono(44_100, 1.0);
    let path = std::env::temp_dir().join("oxiaudio_m9_aiff_frames.aiff");
    {
        let file = std::fs::File::create(&path).expect("create");
        let mut enc = AiffStreamEncoder::new(
            BufWriter::new(file),
            44_100,
            ChannelLayout::Mono,
            AiffBitDepth::I16,
        )
        .expect("new");
        enc.encode_chunk(&buf).expect("encode_chunk");
        assert_eq!(enc.frames_written(), 44_100);
        enc.finalize().expect("finalize");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_wav_info_list_chunk() {
    use oxiaudio_encode::WavEncoder;
    let buf = sine_mono(44_100, 0.1);
    let meta = AudioMetadata {
        title: Some("Test Title".to_string()),
        artist: Some("Test Artist".to_string()),
        ..Default::default()
    };
    let path = std::env::temp_dir().join("oxiaudio_m9_wav_meta.wav");
    {
        let file = std::fs::File::create(&path).expect("create");
        let enc = WavEncoder::default();
        enc.encode_with_metadata(&buf, BufWriter::new(file), &meta)
            .expect("encode_with_metadata");
    }
    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    // Find "LIST" in the bytes
    let has_list = bytes.windows(4).any(|w| w == b"LIST");
    assert!(has_list, "Expected LIST chunk in WAV output");
    let has_inam = bytes.windows(4).any(|w| w == b"INAM");
    assert!(has_inam, "Expected INAM chunk in WAV output");
}

#[test]
fn test_wav_extensible_quad() {
    use oxiaudio_encode::WavEncoder;
    // Create a 4-channel buffer
    let n = 4410;
    let samples = vec![0.0f32; n * 4];
    let buf = AudioBuffer {
        samples,
        sample_rate: 44_100,
        channels: ChannelLayout::Quad,
        format: SampleFormat::F32,
    };
    let path = std::env::temp_dir().join("oxiaudio_m9_wav_quad.wav");
    {
        let file = std::fs::File::create(&path).expect("create");
        let mut enc = WavEncoder::default();
        enc.encode(&buf, BufWriter::new(file)).expect("encode quad");
    }
    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    // fmt chunk format tag is at bytes 8..10 (after "RIFF", size, "WAVE", "fmt ", fmt_size)
    // Layout: [0..4]=RIFF, [4..8]=size, [8..12]=WAVE, [12..16]="fmt ", [16..20]=fmt_size, [20..22]=format_tag
    if bytes.len() > 22 {
        let format_tag = u16::from_le_bytes([bytes[20], bytes[21]]);
        assert_eq!(
            format_tag, 0xFFFE,
            "Expected WAVE_FORMAT_EXTENSIBLE (0xFFFE) for Quad, got 0x{format_tag:04X}"
        );
    }
}

#[test]
fn test_aiff_stream_i24_bit_depth() {
    let buf = sine_mono(44_100, 0.1);
    let path = std::env::temp_dir().join("oxiaudio_m9_aiff_i24.aiff");
    {
        let file = std::fs::File::create(&path).expect("create");
        let mut enc = AiffStreamEncoder::new(
            BufWriter::new(file),
            44_100,
            ChannelLayout::Mono,
            AiffBitDepth::I24,
        )
        .expect("new");
        enc.encode_chunk(&buf).expect("encode_chunk");
        enc.finalize().expect("finalize");
    }
    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(&bytes[..4], b"FORM");
    // AIFF COMM chunk layout (all big-endian):
    // FORM(4) + FORM_size(4) + AIFF(4)          = 12 bytes
    // COMM_tag(4) + COMM_size(4)                 = 8 bytes → total 20
    // numChannels(2) + numSampleFrames(4)         = 6 bytes → total 26
    // sampleSize(2) is at bytes[26..28]
    if bytes.len() > 28 {
        let bits = u16::from_be_bytes([bytes[26], bytes[27]]);
        assert_eq!(
            bits, 24,
            "Expected 24-bit sampleSize in COMM chunk, got {bits}"
        );
    }
}

#[test]
fn test_aiff_with_metadata_name_auth() {
    use oxiaudio_encode::encode_aiff_with_metadata;
    let buf = sine_mono(44_100, 0.05);
    let meta = AudioMetadata {
        title: Some("My Track".to_string()),
        artist: Some("OxiAudio".to_string()),
        comment: Some("M9 test".to_string()),
        ..Default::default()
    };
    let path = std::env::temp_dir().join("oxiaudio_m9_aiff_meta.aiff");
    {
        let file = std::fs::File::create(&path).expect("create");
        encode_aiff_with_metadata(&buf, BufWriter::new(file), AiffBitDepth::I16, &meta)
            .expect("encode_aiff_with_metadata");
    }
    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(&bytes[..4], b"FORM");
    assert!(
        bytes.windows(4).any(|w| w == b"NAME"),
        "Expected NAME chunk"
    );
    assert!(
        bytes.windows(4).any(|w| w == b"AUTH"),
        "Expected AUTH chunk"
    );
    assert!(
        bytes.windows(4).any(|w| w == b"ANNO"),
        "Expected ANNO chunk"
    );
}

#[test]
fn test_wav_info_list_all_fields() {
    use oxiaudio_encode::WavEncoder;
    let buf = sine_mono(44_100, 0.05);
    let meta = AudioMetadata {
        title: Some("Album Track".to_string()),
        artist: Some("Artist Name".to_string()),
        album: Some("Best Of".to_string()),
        genre: Some("Electronic".to_string()),
        comment: Some("Encoded by OxiAudio".to_string()),
        year: Some(2024),
        composer: Some("Test Composer".to_string()),
        ..Default::default()
    };
    let path = std::env::temp_dir().join("oxiaudio_m9_wav_all_meta.wav");
    {
        let file = std::fs::File::create(&path).expect("create");
        let enc = WavEncoder::default();
        enc.encode_with_metadata(&buf, BufWriter::new(file), &meta)
            .expect("encode_with_metadata");
    }
    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert!(bytes.windows(4).any(|w| w == b"INAM"), "Expected INAM");
    assert!(bytes.windows(4).any(|w| w == b"IART"), "Expected IART");
    assert!(bytes.windows(4).any(|w| w == b"IPRD"), "Expected IPRD");
    assert!(bytes.windows(4).any(|w| w == b"IGNR"), "Expected IGNR");
    assert!(bytes.windows(4).any(|w| w == b"ICMT"), "Expected ICMT");
    assert!(
        bytes.windows(4).any(|w| w == b"ICRD"),
        "Expected ICRD (year)"
    );
    assert!(
        bytes.windows(4).any(|w| w == b"IMUS"),
        "Expected IMUS (composer)"
    );
}

#[test]
fn test_wav_extensible_surround51() {
    use oxiaudio_encode::WavEncoder;
    let n = 1000;
    let samples = vec![0.0f32; n * 6];
    let buf = AudioBuffer {
        samples,
        sample_rate: 48_000,
        channels: ChannelLayout::Surround51,
        format: SampleFormat::F32,
    };
    let path = std::env::temp_dir().join("oxiaudio_m9_wav_51.wav");
    {
        let file = std::fs::File::create(&path).expect("create");
        let mut enc = WavEncoder::default();
        enc.encode(&buf, BufWriter::new(file)).expect("encode 5.1");
    }
    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    if bytes.len() > 22 {
        let format_tag = u16::from_le_bytes([bytes[20], bytes[21]]);
        assert_eq!(
            format_tag, 0xFFFE,
            "Expected WAVE_FORMAT_EXTENSIBLE for 5.1, got 0x{format_tag:04X}"
        );
    }
}
