use criterion::{criterion_group, criterion_main, Criterion};
use oxiaudio_core::{AudioBuffer, AudioDecoder, ChannelLayout, SampleFormat};
use oxiaudio_decode::{detect_format, SymphoniaDecoder};
use std::io::{BufWriter, Cursor};

/// Generate a synthetic stereo 48 kHz 32-bit float WAV in memory using hound.
///
/// Formats benchmarked: WAV 32-bit float, FLAC level 5, streaming WAV.
/// MP3/Vorbis/OGG are not benchmarked here because oxiaudio-encode has no pure-Rust
/// MP3 or Vorbis encoder (the LAME encoder is an FFI feature and is excluded from defaults).
fn make_test_wav(duration_secs: f32) -> Vec<u8> {
    let sample_rate = 48_000u32;
    let n_frames = (sample_rate as f32 * duration_secs) as usize;
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut buf = Vec::new();
    {
        let mut writer = hound::WavWriter::new(BufWriter::new(Cursor::new(&mut buf)), spec)
            .expect("WavWriter::new failed");
        for i in 0..n_frames {
            let t = i as f32 / sample_rate as f32;
            let s = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
            writer.write_sample(s).expect("write_sample L");
            writer.write_sample(s).expect("write_sample R");
        }
        writer.finalize().expect("finalize");
    }
    buf
}

/// Generate a synthetic stereo 48 kHz AudioBuffer for FLAC encoding.
fn make_test_audio_buffer(duration_secs: f32) -> AudioBuffer<f32> {
    let sample_rate = 48_000u32;
    let n_frames = (sample_rate as f32 * duration_secs) as usize;
    let mut samples = Vec::with_capacity(n_frames * 2);
    for i in 0..n_frames {
        let t = i as f32 / sample_rate as f32;
        let s = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        samples.push(s); // L
        samples.push(s); // R
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

/// Encode a FLAC file in memory using oxiaudio-encode at compression level 5.
fn make_test_flac(duration_secs: f32) -> Vec<u8> {
    let buf = make_test_audio_buffer(duration_secs);
    let mut out = Vec::new();
    {
        let mut cursor = Cursor::new(&mut out);
        oxiaudio_encode::encode_flac(&buf, &mut cursor).expect("encode_flac failed");
    }
    out
}

fn bench_decode_wav_10s(c: &mut Criterion) {
    let wav_bytes = make_test_wav(10.0);
    c.bench_function("decode_wav_10s_stereo_48k", |b| {
        b.iter(|| {
            let cursor = Cursor::new(wav_bytes.clone());
            let mut decoder = SymphoniaDecoder;
            decoder.decode(cursor).expect("decode failed");
        });
    });
}

fn bench_decode_flac_10s(c: &mut Criterion) {
    let flac_bytes = make_test_flac(10.0);
    c.bench_function("decode_flac_10s_stereo_48k_level5", |b| {
        b.iter(|| {
            let cursor = Cursor::new(flac_bytes.clone());
            let mut decoder = SymphoniaDecoder;
            decoder.decode(cursor).expect("decode flac failed");
        });
    });
}

fn bench_detect_format(c: &mut Criterion) {
    let wav_bytes = make_test_wav(0.1); // small file for probe latency
    c.bench_function("detect_format_wav", |b| {
        b.iter(|| {
            let cursor = Cursor::new(wav_bytes.clone());
            detect_format(cursor).expect("detect_format failed");
        });
    });
}

fn bench_decode_wav_streaming(c: &mut Criterion) {
    let wav_bytes = make_test_wav(10.0);
    c.bench_function("decode_wav_streaming_4096frames", |b| {
        b.iter(|| {
            let cursor = Cursor::new(wav_bytes.clone());
            let mut dec = oxiaudio_decode::StreamingDecoder::new(cursor, 4096)
                .expect("StreamingDecoder::new failed");
            while dec.next_block().expect("next_block error").is_some() {}
        });
    });
}

criterion_group!(
    benches,
    bench_decode_wav_10s,
    bench_decode_flac_10s,
    bench_detect_format,
    bench_decode_wav_streaming,
);
criterion_main!(benches);
