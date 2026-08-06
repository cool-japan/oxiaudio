//! M3 round-trip tests for 16-bit and 24-bit WAV encoding.
//!
//! Encode a sine wave with the new `WavBitDepth` variants, decode back via
//! `SymphoniaDecoder`, and assert that every sample is within the expected
//! quantization tolerance.

use std::io::{Cursor, Write as IoWrite};

use oxiaudio_core::{AudioBuffer, AudioDecoder, AudioEncoder, ChannelLayout, SampleFormat};
use oxiaudio_decode::SymphoniaDecoder;
use oxiaudio_encode::{WavBitDepth, WavEncoder};

/// Generate a 0.1-second mono sine wave at 440 Hz / 44.1 kHz.
fn sine_buffer_mono() -> AudioBuffer<f32> {
    let sample_rate = 44_100u32;
    let n_frames = (sample_rate as f32 * 0.1) as usize;
    let mut samples = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let s = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5;
        samples.push(s);
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

/// Encode `buf` with the given `bit_depth` into a temp WAV file, then decode
/// it back with `SymphoniaDecoder` and return the decoded buffer.
///
/// Symphonia requires a seekable source with a known `byte_len`, so we write
/// to a temp file rather than using an in-memory `Cursor`.
fn encode_and_decode_wav(buf: &AudioBuffer<f32>, bit_depth: WavBitDepth) -> AudioBuffer<f32> {
    // Encode into memory first.
    let mut enc = WavEncoder { bit_depth };
    let mut wav_bytes = Cursor::new(Vec::new());
    enc.encode(buf, &mut wav_bytes)
        .expect("WAV encode should succeed");
    let data = wav_bytes.into_inner();

    // Write to a temp file so Symphonia can probe byte_len.
    let tmp_path = std::env::temp_dir().join(format!(
        "oxiaudio_m3_{}_{}.wav",
        match bit_depth {
            WavBitDepth::F32 => "f32",
            WavBitDepth::I16 => "i16",
            WavBitDepth::I24 => "i24",
            WavBitDepth::I32 => "i32",
            WavBitDepth::U8 => "u8",
        },
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));

    {
        let mut f = std::fs::File::create(&tmp_path).expect("create temp file");
        f.write_all(&data).expect("write temp file");
    }

    let f = std::fs::File::open(&tmp_path).expect("open temp file");
    let mut dec = SymphoniaDecoder;
    let decoded = dec
        .decode(std::io::BufReader::new(f))
        .expect("WAV decode should succeed");

    let _ = std::fs::remove_file(&tmp_path);

    decoded
}

/// 16-bit integer PCM round-trip.
///
/// Encode with `WavBitDepth::I16` (scale by `i16::MAX = 32767`).
/// Symphonia normalises by `32768`, so worst-case total error per sample is
/// just under `2 / 32767`.  We allow `2.0 / 32767.0` tolerance here.
#[test]
fn test_wav_i16_roundtrip() {
    let original = sine_buffer_mono();
    let decoded = encode_and_decode_wav(&original, WavBitDepth::I16);

    assert_eq!(
        decoded.sample_rate, original.sample_rate,
        "sample_rate mismatch"
    );
    assert!(!decoded.samples.is_empty(), "decoded buffer is empty");

    let min_len = original.samples.len().min(decoded.samples.len());
    // Allow slightly more than 1/32767 to account for encode (÷32767) + decode (÷32768) rounding.
    let tolerance = 2.0_f32 / 32767.0_f32;
    for (idx, (&orig, &dec)) in original.samples[..min_len]
        .iter()
        .zip(decoded.samples[..min_len].iter())
        .enumerate()
    {
        assert!(
            (orig - dec).abs() <= tolerance,
            "sample[{idx}] i16 mismatch: orig={orig:.8} dec={dec:.8} diff={:.2e} tol={tolerance:.2e}",
            (orig - dec).abs()
        );
    }
}

/// 24-bit integer PCM round-trip.
///
/// Encode with `WavBitDepth::I24` (scale by `8_388_607 = 2^23 − 1`).
/// Symphonia normalises 24-bit integers by `2^23 = 8_388_608`, so there is
/// a small asymmetry between encode and decode scales that adds up to roughly
/// `1 / 8_388_607` in the worst case on top of quantization error.
/// We allow `2.0 / 8_388_607.0` to account for both sources of error.
#[test]
fn test_wav_i24_roundtrip() {
    let original = sine_buffer_mono();
    let decoded = encode_and_decode_wav(&original, WavBitDepth::I24);

    assert_eq!(
        decoded.sample_rate, original.sample_rate,
        "sample_rate mismatch"
    );
    assert!(!decoded.samples.is_empty(), "decoded buffer is empty");

    let min_len = original.samples.len().min(decoded.samples.len());
    // Allow 2 LSBs of tolerance: one for truncation, one for encode/decode scale mismatch.
    let tolerance = 2.0_f32 / 8_388_607.0_f32;
    for (idx, (&orig, &dec)) in original.samples[..min_len]
        .iter()
        .zip(decoded.samples[..min_len].iter())
        .enumerate()
    {
        assert!(
            (orig - dec).abs() <= tolerance,
            "sample[{idx}] i24 mismatch: orig={orig:.8} dec={dec:.8} diff={:.2e} tol={tolerance:.2e}",
            (orig - dec).abs()
        );
    }
}
