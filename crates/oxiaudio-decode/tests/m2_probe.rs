use std::io::Cursor;

use oxiaudio_core::{AudioDecoder, ChannelLayout};
use oxiaudio_decode::{detect_format, SymphoniaDecoder};

fn make_wav_bytes(sample_rate: u32, channels: u16, samples: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::new(Cursor::new(&mut buf), spec).unwrap();
    for &s in samples {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
    buf
}

#[test]
fn test_detect_format_sample_rate() {
    // 0.1s at 44100 Hz stereo (2 ch)
    let samples: Vec<f32> = vec![0.0f32; 4410];
    let wav_bytes = make_wav_bytes(44_100, 2, &samples);
    let cursor = Cursor::new(wav_bytes);
    let fmt = detect_format(cursor).expect("detect_format should succeed");
    assert_eq!(fmt.sample_rate, 44_100);
    assert_eq!(fmt.channels, ChannelLayout::Stereo);
}

#[test]
fn test_decode_silent_wav_all_zeros() {
    // 0.1s mono silence at 44100 Hz
    let n = 4410;
    let samples: Vec<f32> = vec![0.0f32; n];
    let wav_bytes = make_wav_bytes(44_100, 1, &samples);
    let cursor = Cursor::new(wav_bytes);
    let mut dec = SymphoniaDecoder;
    let buf = dec.decode(cursor).expect("decode should succeed");
    assert!(!buf.samples.is_empty(), "should have samples");
    for &s in &buf.samples {
        assert!(!s.is_nan(), "NaN in decoded samples");
        assert!(s.abs() < 1e-6, "expected zero sample, got {s}");
    }
}
