//! Integration tests for `encode_to_file` convenience methods.
//! Previously in lib.rs `encode_to_file_tests` inline module; moved to keep lib.rs under 2000 lines.

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use oxiaudio_encode::{FlacEncoder, WavEncoder};

fn mono_sine(sr: u32, n: usize) -> AudioBuffer<f32> {
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

fn tmp_path(name: &str, ext: &str) -> std::path::PathBuf {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    std::env::temp_dir().join(format!("oxiaudio_etf_{name}_{ts}.{ext}"))
}

#[test]
fn test_wav_encoder_encode_to_file() {
    let path = tmp_path("wav_enc_to_file", "wav");
    let buf = mono_sine(44_100, 4096);
    let mut encoder = WavEncoder::default();
    encoder
        .encode_to_file(&buf, &path)
        .expect("encode_to_file failed");
    assert!(path.exists(), "WAV file must exist after encode_to_file");
    let bytes = std::fs::read(&path).expect("read WAV file");
    assert_eq!(&bytes[..4], b"RIFF", "WAV file must start with RIFF magic");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_flac_encoder_encode_to_file() {
    let path = tmp_path("flac_enc_to_file", "flac");
    let buf = mono_sine(44_100, 4096);
    let mut encoder = FlacEncoder::default();
    encoder
        .encode_to_file(&buf, &path)
        .expect("encode_to_file failed");
    assert!(path.exists(), "FLAC file must exist after encode_to_file");
    let bytes = std::fs::read(&path).expect("read FLAC file");
    assert_eq!(&bytes[..4], b"fLaC", "FLAC file must start with fLaC magic");
    let _ = std::fs::remove_file(&path);
}
