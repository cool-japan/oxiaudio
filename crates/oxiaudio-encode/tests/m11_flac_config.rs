use oxiaudio_core::{AudioBuffer, AudioDecoder, ChannelLayout, SampleFormat};
use oxiaudio_decode::SymphoniaDecoder;
use oxiaudio_encode::{encode_flac_with_config, FlacBitDepth, FlacConfig};

fn sine_mono(n: usize) -> AudioBuffer<f32> {
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.5)
        .collect();
    AudioBuffer {
        samples,
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

#[test]
fn test_encode_flac_i16_config() {
    let buf = sine_mono(4410);

    let path = std::env::temp_dir().join("oxiaudio_m11_flac_i16.flac");
    {
        let file = std::fs::File::create(&path).expect("create");
        let config = FlacConfig {
            compression: 5,
            bit_depth: FlacBitDepth::I16,
        };
        encode_flac_with_config(&buf, &mut std::io::BufWriter::new(file), &config)
            .expect("encode i16");
    }

    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(&bytes[..4], b"fLaC", "should be valid FLAC");
}

#[test]
fn test_encode_flac_i24_config() {
    let n = 4410;
    let buf = sine_mono(n);

    let path = std::env::temp_dir().join("oxiaudio_m11_flac_i24.flac");
    {
        let file = std::fs::File::create(&path).expect("create");
        let config = FlacConfig {
            compression: 3,
            bit_depth: FlacBitDepth::I24,
        };
        encode_flac_with_config(&buf, &mut std::io::BufWriter::new(file), &config)
            .expect("encode i24");
    }

    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(&bytes[..4], b"fLaC", "should be valid FLAC");
    // A FLAC file should be smaller than raw f32 PCM
    assert!(
        bytes.len() < n * 4,
        "FLAC should compress better than raw f32"
    );
}

#[test]
fn test_flac_config_default() {
    let cfg = FlacConfig::default();
    assert_eq!(cfg.compression, 5);
    assert_eq!(cfg.bit_depth, FlacBitDepth::I16);
    assert_eq!(cfg.bit_depth.bits(), 16);
}

#[test]
fn test_flac_bitdepth_bits() {
    assert_eq!(FlacBitDepth::I16.bits(), 16);
    assert_eq!(FlacBitDepth::I24.bits(), 24);
}

#[test]
fn test_encode_flac_config_roundtrip() {
    let n = 44_100;
    let buf = sine_mono(n);
    let samples = buf.samples.clone();

    let path = std::env::temp_dir().join("oxiaudio_m11_flac_roundtrip.flac");
    {
        let file = std::fs::File::create(&path).expect("create");
        let config = FlacConfig {
            compression: 5,
            bit_depth: FlacBitDepth::I16,
        };
        encode_flac_with_config(&buf, &mut std::io::BufWriter::new(file), &config).expect("encode");
    }

    let file = std::fs::File::open(&path).expect("open");
    let decoded = SymphoniaDecoder
        .decode(std::io::BufReader::new(file))
        .expect("decode");
    let _ = std::fs::remove_file(&path);

    assert_eq!(decoded.sample_rate, 44_100);
    assert_eq!(decoded.samples.len(), samples.len());
    // 16-bit quantization: tolerance ~1/32767 ≈ 3e-5; use 1e-3 to be safe
    for (i, (&o, &d)) in samples.iter().zip(decoded.samples.iter()).enumerate() {
        assert!((o - d).abs() < 1e-3, "sample[{i}]: orig={o} decoded={d}");
    }
}

#[test]
fn test_encode_flac_cursor() {
    use std::io::Cursor;

    let buf = sine_mono(4096);
    let mut out = Cursor::new(Vec::new());
    let config = FlacConfig::default();
    encode_flac_with_config(&buf, &mut out, &config).expect("encode to cursor");
    let bytes = out.into_inner();
    assert_eq!(&bytes[..4], b"fLaC", "cursor output should be valid FLAC");
}
