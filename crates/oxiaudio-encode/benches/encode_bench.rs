/// Criterion benchmarks for `oxiaudio-encode`.
///
/// Covers WAV encoding at F32/I16/I24 bit depths and FLAC encoding at
/// compression levels 0, 5, and 8 using a 10-second 48 kHz stereo sine wave.
use std::io::Cursor;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxiaudio_core::AudioEncoder;
use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use oxiaudio_encode::{
    encode_flac_with_level, FlacStreamEncoder, WavBitDepth, WavEncoder, WavStreamEncoder,
};
use std::hint::black_box;

/// Synthesise a stereo sine wave of the requested duration.
fn make_sine_stereo(sample_rate: u32, n_frames: usize) -> AudioBuffer<f32> {
    let samples: Vec<f32> = (0..n_frames)
        .flat_map(|i| {
            let s = (2.0_f32 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
                * 0.5;
            [s, -s]
        })
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

// ─── WAV encoding benchmarks ──────────────────────────────────────────────────

fn bench_wav_formats(c: &mut Criterion) {
    let buf = make_sine_stereo(48_000, 48_000 * 10); // 10 s at 48 kHz stereo

    let mut group = c.benchmark_group("encode_wav");

    // F32
    group.bench_with_input(BenchmarkId::new("wav", "F32"), &buf, |b, buf| {
        b.iter(|| {
            let mut out = Cursor::new(Vec::new());
            WavEncoder {
                bit_depth: WavBitDepth::F32,
            }
            .encode(black_box(buf), &mut out)
            .expect("encode WAV F32");
        });
    });

    // I16
    group.bench_with_input(BenchmarkId::new("wav", "I16"), &buf, |b, buf| {
        b.iter(|| {
            let mut out = Cursor::new(Vec::new());
            WavEncoder {
                bit_depth: WavBitDepth::I16,
            }
            .encode(black_box(buf), &mut out)
            .expect("encode WAV I16");
        });
    });

    // I24
    group.bench_with_input(BenchmarkId::new("wav", "I24"), &buf, |b, buf| {
        b.iter(|| {
            let mut out = Cursor::new(Vec::new());
            WavEncoder {
                bit_depth: WavBitDepth::I24,
            }
            .encode(black_box(buf), &mut out)
            .expect("encode WAV I24");
        });
    });

    group.finish();
}

// ─── FLAC encoding benchmarks ─────────────────────────────────────────────────

fn bench_flac_levels(c: &mut Criterion) {
    let buf = make_sine_stereo(48_000, 48_000 * 10); // 10 s at 48 kHz stereo

    let mut group = c.benchmark_group("encode_flac");

    for level in [0u8, 5, 8] {
        group.bench_with_input(BenchmarkId::new("flac_level", level), &buf, |b, buf| {
            b.iter(|| {
                let mut out = Cursor::new(Vec::new());
                encode_flac_with_level(black_box(buf), &mut out, level).expect("encode FLAC");
            });
        });
    }

    group.finish();
}

// ─── Streaming WAV encoding benchmark ────────────────────────────────────────

fn bench_wav_streaming(c: &mut Criterion) {
    let buf = make_sine_stereo(48_000, 48_000 * 2); // 2 s — streaming overhead

    c.bench_function("wav_stream_i16_2s", |b| {
        b.iter(|| {
            let out = Cursor::new(Vec::new());
            let mut enc =
                WavStreamEncoder::new(out, buf.sample_rate, buf.channels, WavBitDepth::I16)
                    .expect("WavStreamEncoder::new");
            // Feed in 4096-frame chunks
            let n_ch = buf.channels.channel_count();
            for chunk in buf.samples.chunks(4096 * n_ch) {
                let chunk_buf = AudioBuffer {
                    samples: chunk.to_vec(),
                    sample_rate: buf.sample_rate,
                    channels: buf.channels,
                    format: buf.format,
                };
                enc.encode_chunk(black_box(&chunk_buf))
                    .expect("encode_chunk");
            }
            enc.finalize().expect("finalize");
        });
    });
}

// ─── Streaming FLAC encoding benchmark ───────────────────────────────────────

fn bench_flac_streaming(c: &mut Criterion) {
    let buf = make_sine_stereo(48_000, 48_000 * 2); // 2 s

    c.bench_function("flac_stream_l5_2s", |b| {
        b.iter(|| {
            let out = Cursor::new(Vec::new());
            let mut enc = FlacStreamEncoder::new(out, buf.sample_rate, buf.channels, 5);
            let n_ch = buf.channels.channel_count();
            for chunk in buf.samples.chunks(4096 * n_ch) {
                let chunk_buf = AudioBuffer {
                    samples: chunk.to_vec(),
                    sample_rate: buf.sample_rate,
                    channels: buf.channels,
                    format: buf.format,
                };
                enc.encode_chunk(black_box(&chunk_buf))
                    .expect("encode_chunk");
            }
            enc.finalize().expect("finalize");
        });
    });
}

criterion_group!(
    benches,
    bench_wav_formats,
    bench_flac_levels,
    bench_wav_streaming,
    bench_flac_streaming,
);
criterion_main!(benches);
