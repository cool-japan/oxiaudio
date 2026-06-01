//! M11 facade integration tests: FlacConfig / FlacBitDepth re-exports and
//! encode_flac_with_config facade function.

use oxiaudio::{AudioBuffer, ChannelLayout, SampleFormat};

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
#[cfg(feature = "pure")]
fn test_facade_flac_config_default() {
    let cfg = oxiaudio::FlacConfig::default();
    assert_eq!(cfg.compression, 5);
    assert_eq!(cfg.bit_depth, oxiaudio::FlacBitDepth::I16);
}

#[test]
#[cfg(feature = "pure")]
fn test_facade_flac_bit_depth_reexport() {
    let _i16 = oxiaudio::FlacBitDepth::I16;
    let _i24 = oxiaudio::FlacBitDepth::I24;
    assert_ne!(
        format!("{:?}", oxiaudio::FlacBitDepth::I16),
        format!("{:?}", oxiaudio::FlacBitDepth::I24)
    );
}

#[test]
#[cfg(feature = "pure")]
fn test_facade_encode_flac_with_config_i16() {
    let buf = sine_mono(4410);
    let path = std::env::temp_dir().join("oxiaudio_m11_facade_flac_i16.flac");
    let config = oxiaudio::FlacConfig {
        compression: 5,
        bit_depth: oxiaudio::FlacBitDepth::I16,
    };
    oxiaudio::encode_flac_with_config(&buf, &path, &config)
        .expect("facade encode_flac_with_config i16");
    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(&bytes[..4], b"fLaC");
}

#[test]
#[cfg(feature = "pure")]
fn test_facade_encode_flac_with_config_i24() {
    let buf = sine_mono(4410);
    let path = std::env::temp_dir().join("oxiaudio_m11_facade_flac_i24.flac");
    let config = oxiaudio::FlacConfig {
        compression: 3,
        bit_depth: oxiaudio::FlacBitDepth::I24,
    };
    oxiaudio::encode_flac_with_config(&buf, &path, &config)
        .expect("facade encode_flac_with_config i24");
    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(&bytes[..4], b"fLaC");
}

#[test]
#[cfg(feature = "pure")]
fn test_facade_encode_flac_with_config_default() {
    let buf = sine_mono(4410);
    let path = std::env::temp_dir().join("oxiaudio_m11_facade_flac.flac");
    let config = oxiaudio::FlacConfig::default();
    oxiaudio::encode_flac_with_config(&buf, &path, &config)
        .expect("facade encode_flac_with_config default");
    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);
    assert_eq!(&bytes[..4], b"fLaC");
}
