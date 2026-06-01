use oxiaudio::{AudioBuffer, ChannelLayout, SampleFormat};
use std::path::PathBuf;

fn sine_buffer(sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
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

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

#[test]
#[cfg(feature = "pure")]
fn test_dsp_eq_flat_response() {
    let buf = sine_buffer(44_100, 0.1);
    // gain_db=0.0 = flat EQ = passthrough
    let result = oxiaudio::dsp::eq(&buf, &[(1000.0, 0.0, 1.414)]).unwrap();
    assert_eq!(result.sample_rate, buf.sample_rate);
    let diff: f32 = buf
        .samples
        .iter()
        .zip(result.samples.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff < 1e-4,
        "flat EQ should be near-passthrough, max diff = {diff}"
    );
}

#[test]
#[cfg(feature = "pure")]
fn test_dsp_compressor_reduces_loud() {
    let n = 44_100;
    let samples = vec![0.9f32; n]; // loud signal
    let buf = AudioBuffer {
        samples,
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };
    let compressed = oxiaudio::dsp::compressor(&buf, -6.0, 4.0, 5.0, 50.0).unwrap();
    let rms_in: f32 = (buf.samples.iter().map(|&s| s * s).sum::<f32>() / n as f32).sqrt();
    let rms_out: f32 = (compressed.samples.iter().map(|&s| s * s).sum::<f32>()
        / compressed.samples.len() as f32)
        .sqrt();
    assert!(
        rms_out < rms_in,
        "compressor should reduce loud signal RMS: in={rms_in}, out={rms_out}"
    );
}

#[test]
#[cfg(feature = "pure")]
fn test_convert_with_dsp_roundtrip() {
    let buf = sine_buffer(44_100, 0.5);
    let input_path = temp_path("oxiaudio_m9_convert_in.wav");
    let output_path = temp_path("oxiaudio_m9_convert_out.wav");

    oxiaudio::encode_wav(&buf, &input_path).expect("encode input");
    oxiaudio::convert_with_dsp(&input_path, &output_path, |b| b).expect("convert_with_dsp");

    let _ = std::fs::remove_file(&input_path);
    assert!(output_path.exists(), "output file should exist");
    let decoded = oxiaudio::decode_file(&output_path).expect("decode output");
    let _ = std::fs::remove_file(&output_path);
    assert_eq!(decoded.sample_rate, buf.sample_rate);
}

#[test]
#[cfg(feature = "pure")]
fn test_write_metadata_embeds_tags() {
    use oxiaudio::AudioMetadata;

    let buf = sine_buffer(44_100, 0.1);
    let path = temp_path("oxiaudio_m9_write_meta.wav");

    // Write initial WAV
    oxiaudio::encode_wav(&buf, &path).expect("encode_wav");

    // Embed metadata via write_metadata
    let meta = AudioMetadata {
        title: Some("Facade Title".to_string()),
        artist: Some("Facade Artist".to_string()),
        year: Some(2026),
        composer: Some("Facade Composer".to_string()),
        ..Default::default()
    };
    oxiaudio::write_metadata(&path, &meta).expect("write_metadata");

    let bytes = std::fs::read(&path).expect("read");
    let _ = std::fs::remove_file(&path);

    assert_eq!(&bytes[..4], b"RIFF", "Expected RIFF header");
    assert!(bytes.windows(4).any(|w| w == b"INAM"), "Expected INAM tag");
    assert!(bytes.windows(4).any(|w| w == b"IART"), "Expected IART tag");
    assert!(
        bytes.windows(4).any(|w| w == b"ICRD"),
        "Expected ICRD tag (year)"
    );
    assert!(
        bytes.windows(4).any(|w| w == b"IMUS"),
        "Expected IMUS tag (composer)"
    );
}
