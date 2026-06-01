//! M10 facade integration tests: gate, delay, chorus DSP wrappers;
//! detect_format_from_path, AiffStreamEncoder, AiffBitDepth re-exports.

use oxiaudio::{AudioBuffer, ChannelLayout, SampleFormat};
use std::path::PathBuf;

fn sine_mono(sr: u32, dur: f32) -> AudioBuffer<f32> {
    let n = (sr as f32 * dur) as usize;
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

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

#[test]
#[cfg(feature = "pure")]
fn test_dsp_gate_attenuates_quiet() {
    // A quiet signal should be gated (attenuated)
    let samples = vec![0.001f32; 44_100]; // very quiet
    let buf = AudioBuffer {
        samples,
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };
    let gated = oxiaudio::dsp::gate(&buf, -20.0, 5.0, 50.0, 100.0).unwrap();
    let rms_in: f32 =
        (buf.samples.iter().map(|&s| s * s).sum::<f32>() / buf.samples.len() as f32).sqrt();
    let rms_out: f32 =
        (gated.samples.iter().map(|&s| s * s).sum::<f32>() / gated.samples.len() as f32).sqrt();
    assert!(
        rms_out < rms_in,
        "gate should attenuate quiet signal: in={rms_in}, out={rms_out}"
    );
}

#[test]
#[cfg(feature = "pure")]
fn test_dsp_delay_changes_signal() {
    let buf = sine_mono(44_100, 0.5);
    let delayed = oxiaudio::dsp::delay(&buf, 100.0, 0.3, 0.5).unwrap();
    assert_eq!(delayed.samples.len(), buf.samples.len());
    let diff: f32 = buf
        .samples
        .iter()
        .zip(delayed.samples.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.01, "delay should modify the signal");
}

#[test]
#[cfg(feature = "pure")]
fn test_dsp_chorus_changes_signal() {
    let buf = sine_mono(44_100, 0.5);
    let chorused = oxiaudio::dsp::chorus(&buf, 0.5, 5.0, 2, 0.5).unwrap();
    assert_eq!(chorused.samples.len(), buf.samples.len());
    let diff: f32 = buf
        .samples
        .iter()
        .zip(chorused.samples.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.01, "chorus should modify the signal");
}

#[test]
#[cfg(feature = "pure")]
fn test_detect_format_from_path_wav() {
    let buf = sine_mono(44_100, 0.1);
    let path = temp("oxiaudio_m10_facade_fmt.wav");
    oxiaudio::encode_wav(&buf, &path).expect("encode_wav");
    let fmt = oxiaudio::detect_format_from_path(&path).expect("detect_format_from_path");
    let _ = std::fs::remove_file(&path);
    assert_eq!(fmt.sample_rate, 44_100);
}

#[test]
#[cfg(feature = "pure")]
fn test_aiff_bit_depth_reexport() {
    // AiffBitDepth should be accessible through the facade.
    let _depth = oxiaudio::AiffBitDepth::I16;
    let _depth2 = oxiaudio::AiffBitDepth::I24;
    // Both variants should be distinguishable.
    assert_ne!(
        format!("{:?}", oxiaudio::AiffBitDepth::I16),
        format!("{:?}", oxiaudio::AiffBitDepth::I24)
    );
}

#[test]
#[cfg(feature = "pure")]
fn test_aiff_stream_encoder_reexport() {
    use oxiaudio::{AiffBitDepth, AiffStreamEncoder};
    let buf = sine_mono(44_100, 0.1);
    let path = temp("oxiaudio_m10_aiff_stream.aiff");
    let file = std::fs::File::create(&path).expect("create aiff file");
    let writer = std::io::BufWriter::new(file);
    let mut enc = AiffStreamEncoder::new(writer, buf.sample_rate, buf.channels, AiffBitDepth::I16)
        .expect("create AiffStreamEncoder");
    enc.encode_chunk(&buf).expect("encode_chunk");
    enc.finalize().expect("finalize");
    let _ = std::fs::remove_file(&path);
}
