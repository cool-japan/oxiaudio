use oxiaudio_core::{AudioBuffer, AudioMetadata, ChannelLayout, SampleFormat};
use oxiaudio_encode::{encode_wav_with_config, WavBitDepth, WavEncodeConfig};
use std::io::Cursor;

fn sine_mono(sr: u32, n: usize) -> AudioBuffer<f32> {
    let samples = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
        .collect();
    AudioBuffer {
        samples,
        sample_rate: sr,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

#[test]
fn test_wav_config_default_f32() {
    let buf = sine_mono(44_100, 4410);
    let config = WavEncodeConfig::default();
    let mut cursor = Cursor::new(Vec::new());
    encode_wav_with_config(&buf, &mut cursor, &config).expect("encode f32");
    let bytes = cursor.into_inner();
    assert_eq!(&bytes[..4], b"RIFF");
}

#[test]
fn test_wav_config_i16() {
    let buf = sine_mono(44_100, 4410);
    let config = WavEncodeConfig {
        bit_depth: WavBitDepth::I16,
        metadata: None,
    };
    let mut cursor = Cursor::new(Vec::new());
    encode_wav_with_config(&buf, &mut cursor, &config).expect("encode i16");
    let bytes = cursor.into_inner();
    assert_eq!(&bytes[..4], b"RIFF");
    // fmt chunk format tag at bytes[20..22] should be 1 (PCM) for I16
    let fmt_tag = u16::from_le_bytes([bytes[20], bytes[21]]);
    assert!(
        fmt_tag == 1 || fmt_tag == 0xFFFE,
        "I16 should be PCM (1) or extensible (0xFFFE), got {fmt_tag:#06X}"
    );
}

#[test]
fn test_wav_config_with_metadata() {
    let buf = sine_mono(44_100, 4410);
    let meta = AudioMetadata {
        title: Some("M12 Test".to_string()),
        ..Default::default()
    };
    let config = WavEncodeConfig {
        bit_depth: WavBitDepth::F32,
        metadata: Some(meta),
    };
    let mut cursor = Cursor::new(Vec::new());
    encode_wav_with_config(&buf, &mut cursor, &config).expect("encode with metadata");
    let bytes = cursor.into_inner();
    let has_list = bytes.windows(4).any(|w| w == b"LIST");
    assert!(has_list, "WAV with metadata config should embed LIST chunk");
    let has_inam = bytes.windows(4).any(|w| w == b"INAM");
    assert!(has_inam, "should have INAM tag");
}

#[test]
fn test_wav_config_roundtrip_i16() {
    use oxiaudio_core::AudioDecoder;
    use oxiaudio_decode::SymphoniaDecoder;
    use std::io::BufWriter;

    let buf = sine_mono(44_100, 4410);
    let path = std::env::temp_dir().join("oxiaudio_m12_wav_i16_rt.wav");
    {
        let file = std::fs::File::create(&path).expect("create");
        let config = WavEncodeConfig {
            bit_depth: WavBitDepth::I16,
            metadata: None,
        };
        encode_wav_with_config(&buf, BufWriter::new(file), &config).expect("encode");
    }
    let file = std::fs::File::open(&path).expect("open");
    let decoded = SymphoniaDecoder.decode(file).expect("decode");
    let _ = std::fs::remove_file(&path);
    assert_eq!(decoded.sample_rate, 44_100);
    assert_eq!(decoded.samples.len(), buf.samples.len());
    // 16-bit quantization tolerance
    for (o, d) in buf.samples.iter().zip(decoded.samples.iter()) {
        assert!(
            (o - d).abs() < 1e-3,
            "I16 roundtrip tolerance exceeded: orig={o} decoded={d}"
        );
    }
}
