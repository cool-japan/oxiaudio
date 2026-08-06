//! M6 feature tests: AIFF writer, WAV U8 PCM, FLAC compression level,
//! TPDF dithering, encode_to_vec, StreamEncoder trait, EncoderConfig builder.

use std::io::{BufWriter, Cursor};

use oxiaudio_core::{AudioBuffer, AudioDecoder, ChannelLayout, SampleFormat};
use oxiaudio_decode::SymphoniaDecoder;
use oxiaudio_encode::{
    apply_tpdf_dither, encode_flac_to_vec, encode_flac_with_level, encode_wav_to_vec, write_aiff,
    write_aiff_file, EncoderConfig, FlacStreamEncoder, StreamEncoder, WavBitDepth, WavEncoder,
    WavStreamEncoder,
};

// ─── helpers ──────────────────────────────────────────────────────────────────

fn sine_buffer(freq: f32, sample_rate: u32, channels: u16, duration_secs: f32) -> AudioBuffer<f32> {
    let n_frames = (sample_rate as f32 * duration_secs) as usize;
    let n_ch = channels as usize;
    let mut samples = Vec::with_capacity(n_frames * n_ch);
    for i in 0..n_frames {
        let s = (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin() * 0.5;
        for _ in 0..n_ch {
            samples.push(s);
        }
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: if channels == 1 {
            ChannelLayout::Mono
        } else {
            ChannelLayout::Stereo
        },
        format: SampleFormat::F32,
    }
}

fn mono_sine() -> AudioBuffer<f32> {
    sine_buffer(440.0, 44_100, 1, 0.1)
}

// ─── 1. AIFF writer ───────────────────────────────────────────────────────────

#[test]
fn test_write_aiff_roundtrip() {
    // Encode to a temp .aiff file, verify FORM/AIFF/COMM/SSND magic bytes.
    let buf = sine_buffer(440.0, 44_100, 2, 1.0); // 1s stereo
    let path = std::env::temp_dir().join(format!(
        "oxiaudio_m6_aiff_{}.aiff",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));

    write_aiff_file(&buf, &path).expect("write_aiff_file must succeed");

    let data = std::fs::read(&path).expect("read aiff file");
    let _ = std::fs::remove_file(&path);

    // FORM header at offset 0
    assert_eq!(&data[0..4], b"FORM", "FORM magic missing");
    // AIFF type at offset 8
    assert_eq!(&data[8..12], b"AIFF", "AIFF type missing");
    // COMM chunk somewhere after offset 12
    assert_eq!(&data[12..16], b"COMM", "COMM chunk missing");
    // SSND chunk should follow COMM (COMM payload = 18 bytes, so at offset 12+8+18 = 38)
    assert_eq!(&data[38..42], b"SSND", "SSND chunk missing");

    // File must be larger than the header
    assert!(data.len() > 50, "AIFF file too small: {} bytes", data.len());
}

#[test]
fn test_write_aiff_to_cursor() {
    let buf = mono_sine();
    let mut cursor = Cursor::new(Vec::new());
    write_aiff(&buf, &mut cursor).expect("write_aiff must succeed");
    let data = cursor.into_inner();

    assert_eq!(&data[0..4], b"FORM");
    assert_eq!(&data[8..12], b"AIFF");
    assert!(data.len() > 44);
}

#[test]
fn test_write_aiff_silent_buffer() {
    // Zero-sample buffer should produce a valid (empty) AIFF
    let buf = AudioBuffer {
        samples: vec![0.0f32; 441], // 10ms of silence at 44.1kHz mono
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };
    let mut cursor = Cursor::new(Vec::new());
    write_aiff(&buf, &mut cursor).expect("write_aiff silent buffer must succeed");
    let data = cursor.into_inner();
    assert_eq!(&data[0..4], b"FORM");
    assert_eq!(&data[8..12], b"AIFF");
}

// ─── 2. WAV U8 PCM ────────────────────────────────────────────────────────────

#[test]
fn test_encode_wav_u8() {
    // Encode with U8, decode back, check non-empty and within expected range.
    let buf = mono_sine();
    let mut cursor = Cursor::new(Vec::new());
    let mut enc = WavEncoder {
        bit_depth: WavBitDepth::U8,
    };
    use oxiaudio_core::AudioEncoder;
    enc.encode(&buf, &mut cursor)
        .expect("WAV U8 encode must succeed");
    let data = cursor.into_inner();

    // Must start with RIFF marker
    assert_eq!(&data[0..4], b"RIFF", "WAV U8 must start with RIFF");

    // Write to temp file so Symphonia can probe it
    let path = std::env::temp_dir().join(format!(
        "oxiaudio_m6_u8_{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));
    std::fs::write(&path, &data).expect("write temp u8 wav");

    let f = std::fs::File::open(&path).expect("open temp u8 wav");
    let decoded = SymphoniaDecoder
        .decode(std::io::BufReader::new(f))
        .expect("decode U8 WAV must succeed");
    let _ = std::fs::remove_file(&path);

    assert!(
        !decoded.samples.is_empty(),
        "decoded U8 WAV must have samples"
    );
    // U8 has ~8-bit quantization, so samples should be roughly in [-1, 1]
    for &s in &decoded.samples {
        assert!(s.abs() <= 1.1, "decoded U8 sample out of range: {s}");
    }
}

#[test]
fn test_wav_u8_silence_is_128() {
    // Silence (0.0) should encode to u8 128 — the WAV 8-bit offset-binary midpoint.
    let buf = AudioBuffer {
        samples: vec![0.0f32; 4],
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };
    let mut cursor = Cursor::new(Vec::new());
    let mut enc = WavEncoder {
        bit_depth: WavBitDepth::U8,
    };
    use oxiaudio_core::AudioEncoder;
    enc.encode(&buf, &mut cursor).expect("encode silence as u8");
    let data = cursor.into_inner();

    // WAV header is 44 bytes for simple PCM.
    // data[44..] should be all 128 (0x80) for silence.
    assert!(data.len() > 44, "WAV header too short");
    for &byte in &data[44..] {
        assert_eq!(byte, 128, "silence byte should be 128 (0x80), got {byte}");
    }
}

// ─── 3. FLAC configurable compression level ───────────────────────────────────

#[test]
fn test_encode_flac_with_level_roundtrip() {
    let buf = mono_sine();
    for level in [0u8, 3, 5, 8] {
        let mut cursor = Cursor::new(Vec::new());
        encode_flac_with_level(&buf, &mut cursor, level)
            .unwrap_or_else(|e| panic!("encode_flac_with_level level={level} failed: {e}"));
        let data = cursor.into_inner();
        assert_eq!(&data[0..4], b"fLaC", "level={level}: not a FLAC stream");
        assert!(data.len() > 42, "level={level}: FLAC output too small");
    }
}

#[test]
fn test_flac_configurable_bit_depth_roundtrip() {
    use oxiaudio_core::AudioEncoder;
    use oxiaudio_encode::FlacEncoder;

    let buf = mono_sine();
    // flacenc 0.5.1 supports up to 24-bit PCM.
    for bits in [16u8, 20, 24] {
        let mut cursor = Cursor::new(Vec::new());
        FlacEncoder::new(5)
            .with_bits_per_sample(bits)
            .encode(&buf, &mut cursor)
            .unwrap_or_else(|e| panic!("FLAC bits={bits} encode failed: {e}"));
        let data = cursor.into_inner();
        assert_eq!(&data[0..4], b"fLaC", "bits={bits}: not a FLAC stream");

        // Decode back and verify samples land near the original within the
        // quantization tolerance for the chosen depth.
        let decoded = SymphoniaDecoder
            .decode(Cursor::new(data))
            .unwrap_or_else(|e| panic!("FLAC bits={bits} decode failed: {e}"));
        assert_eq!(
            decoded.samples.len(),
            buf.samples.len(),
            "bits={bits}: length mismatch"
        );
        let tol = 4.0 / (1u64 << (bits - 1)) as f32; // a few LSBs
        for (o, d) in buf.samples.iter().zip(decoded.samples.iter()) {
            assert!(
                (o - d).abs() < tol + 1e-4,
                "bits={bits}: sample drift {o} vs {d}"
            );
        }
    }
}

#[test]
fn test_flac_bit_depth_clamping() {
    use oxiaudio_encode::FlacEncoder;
    // Out-of-range / odd values snap to the nearest supported FLAC depth.
    // flacenc caps at 24-bit, so requests above 24 clamp down to 24.
    assert_eq!(
        FlacEncoder::new(5).with_bits_per_sample(8).bits_per_sample,
        16
    );
    assert_eq!(
        FlacEncoder::new(5).with_bits_per_sample(19).bits_per_sample,
        20
    );
    assert_eq!(
        FlacEncoder::new(5).with_bits_per_sample(23).bits_per_sample,
        24
    );
    assert_eq!(
        FlacEncoder::new(5).with_bits_per_sample(64).bits_per_sample,
        24
    );
}

// ─── 4. TPDF dithering ────────────────────────────────────────────────────────

#[test]
fn test_tpdf_dither_no_overflow() {
    let mut samples: Vec<f32> = (0..1024)
        .map(|i| (i as f32 / 512.0 - 1.0).clamp(-1.0, 1.0))
        .collect();
    apply_tpdf_dither(&mut samples, 16);

    // All dithered samples must stay within a small band around [-1, 1].
    for (i, &s) in samples.iter().enumerate() {
        assert!(s.abs() <= 1.1, "dithered sample[{i}] out of range: {s}");
    }
}

#[test]
fn test_tpdf_dither_adds_noise() {
    // The mean absolute deviation after dithering must be non-zero.
    let mut samples = vec![0.5f32; 512];
    let original = samples.clone();
    apply_tpdf_dither(&mut samples, 16);

    let deviation: f32 = samples
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / 512.0;

    assert!(
        deviation > 1e-8,
        "TPDF dither must change samples; mean deviation = {deviation}"
    );
}

// ─── 5. encode_to_vec ─────────────────────────────────────────────────────────

#[test]
fn test_encode_to_vec_wav() {
    let buf = mono_sine();
    let v = encode_wav_to_vec(&buf).expect("encode_wav_to_vec must succeed");
    assert_eq!(&v[0..4], b"RIFF", "encode_wav_to_vec: missing RIFF header");
    assert!(v.len() > 44, "encode_wav_to_vec: output too small");
}

#[test]
fn test_encode_to_vec_flac() {
    let buf = mono_sine();
    let v = encode_flac_to_vec(&buf).expect("encode_flac_to_vec must succeed");
    assert_eq!(&v[0..4], b"fLaC", "encode_flac_to_vec: missing fLaC marker");
    assert!(v.len() > 42, "encode_flac_to_vec: output too small");
}

// ─── 6. StreamEncoder trait ───────────────────────────────────────────────────

#[test]
fn test_stream_encoder_wav_via_trait() {
    // Verify WavStreamEncoder implements StreamEncoder (object-safe, boxed finalize).
    let buf = mono_sine();
    let n_frames = buf.samples.len() as u64;

    let path = std::env::temp_dir().join(format!(
        "oxiaudio_m6_stream_wav_{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));

    {
        let file = std::fs::File::create(&path).expect("create temp wav");
        // Use 3 chunks
        let chunk_size = buf.samples.len() / 3;
        let mut enc: Box<dyn StreamEncoder> = Box::new(
            WavStreamEncoder::new(
                BufWriter::new(file),
                buf.sample_rate,
                buf.channels,
                WavBitDepth::F32,
            )
            .expect("WavStreamEncoder::new"),
        );

        for chunk in buf.samples.chunks(chunk_size) {
            let chunk_buf = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: buf.sample_rate,
                channels: buf.channels,
                format: buf.format,
            };
            enc.write_chunk(&chunk_buf).expect("write_chunk");
        }
        enc.finalize().expect("finalize");
    }

    let bytes = std::fs::read(&path).expect("read wav");
    let _ = std::fs::remove_file(&path);

    assert_eq!(&bytes[0..4], b"RIFF", "StreamEncoder WAV: missing RIFF");
    assert!(
        bytes.len() > 44 + n_frames as usize * 4,
        "StreamEncoder WAV: too few bytes"
    );
}

#[test]
fn test_stream_encoder_flac_via_trait() {
    // Verify FlacStreamEncoder implements StreamEncoder (object-safe).
    let buf = mono_sine();

    let path = std::env::temp_dir().join(format!(
        "oxiaudio_m6_stream_flac_{}.flac",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));

    {
        let file = std::fs::File::create(&path).expect("create temp flac");
        let mut enc: Box<dyn StreamEncoder> = Box::new(FlacStreamEncoder::new(
            BufWriter::new(file),
            buf.sample_rate,
            buf.channels,
            5,
        ));

        for chunk in buf.samples.chunks(256) {
            let chunk_buf = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: buf.sample_rate,
                channels: buf.channels,
                format: buf.format,
            };
            enc.write_chunk(&chunk_buf).expect("write_chunk");
        }
        enc.finalize().expect("finalize");
    }

    let bytes = std::fs::read(&path).expect("read flac");
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        &bytes[0..4],
        b"fLaC",
        "StreamEncoder FLAC: missing fLaC marker"
    );
}

// ─── 7. EncoderConfig builder ────────────────────────────────────────────────

#[test]
fn test_encoder_config_builder_wav() {
    let buf = mono_sine();

    let mut out = Cursor::new(Vec::new());
    EncoderConfig::new(44_100, 1)
        .with_bit_depth(WavBitDepth::I16)
        .with_dither(true)
        .encode_wav(&buf, &mut out)
        .expect("EncoderConfig::encode_wav must succeed");

    let v = out.into_inner();
    assert_eq!(&v[0..4], b"RIFF", "EncoderConfig WAV: missing RIFF");
    assert!(v.len() > 44, "EncoderConfig WAV: output too small");
}

#[test]
fn test_encoder_config_builder_flac() {
    let buf = mono_sine();

    let mut out = Cursor::new(Vec::new());
    EncoderConfig::new(44_100, 1)
        .with_flac_compression(3)
        .encode_flac(&buf, &mut out)
        .expect("EncoderConfig::encode_flac must succeed");

    let v = out.into_inner();
    assert_eq!(&v[0..4], b"fLaC", "EncoderConfig FLAC: missing fLaC marker");
    assert!(v.len() > 42, "EncoderConfig FLAC: output too small");
}

#[test]
fn test_encoder_config_normalize() {
    // With a very quiet signal, normalization should scale it up.
    let quiet: Vec<f32> = (0..1024)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.01)
        .collect();
    let buf = AudioBuffer {
        samples: quiet.clone(),
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };

    let mut out = Cursor::new(Vec::new());
    EncoderConfig::new(44_100, 1)
        .with_normalize(true)
        .with_bit_depth(WavBitDepth::F32)
        .encode_wav(&buf, &mut out)
        .expect("normalize + encode must succeed");

    let v = out.into_inner();
    assert_eq!(&v[0..4], b"RIFF");
    // Decoded peak should be ~1.0, not 0.01
    let path = std::env::temp_dir().join(format!(
        "oxiaudio_m6_norm_{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));
    std::fs::write(&path, &v).expect("write normalized wav");
    let f = std::fs::File::open(&path).expect("open normalized wav");
    let decoded = SymphoniaDecoder
        .decode(std::io::BufReader::new(f))
        .expect("decode normalized wav");
    let _ = std::fs::remove_file(&path);

    let peak = decoded
        .samples
        .iter()
        .fold(0.0f32, |acc, &s| acc.max(s.abs()));
    assert!(
        peak > 0.5,
        "normalized peak should be close to 1.0, got {peak}"
    );
}

#[test]
fn test_encoder_config_u8_no_dither() {
    let buf = mono_sine();
    let mut out = Cursor::new(Vec::new());
    EncoderConfig::new(44_100, 1)
        .with_bit_depth(WavBitDepth::U8)
        .encode_wav(&buf, &mut out)
        .expect("EncoderConfig U8 WAV must succeed");

    let v = out.into_inner();
    assert_eq!(&v[0..4], b"RIFF");
    assert!(v.len() > 44);
}
