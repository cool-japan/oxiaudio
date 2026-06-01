use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use std::hint::black_box;

fn make_f32_buf(frames: usize) -> AudioBuffer<f32> {
    AudioBuffer {
        samples: (0..frames * 2)
            .map(|i| (i as f32) / frames as f32)
            .collect(),
        sample_rate: 44100,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

fn bench_format_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_conversion");
    for size in [4410usize, 44100, 441000] {
        let buf = make_f32_buf(size);
        group.bench_with_input(BenchmarkId::new("f32_to_i16", size), &buf, |b, buf| {
            b.iter(|| {
                let out: AudioBuffer<i16> = black_box(buf).into();
                black_box(out)
            });
        });
        group.bench_with_input(BenchmarkId::new("f32_to_i32", size), &buf, |b, buf| {
            b.iter(|| {
                let out: AudioBuffer<i32> = black_box(buf).into();
                black_box(out)
            });
        });
    }
    group.finish();
}

fn bench_audio_buffer_append(c: &mut Criterion) {
    let make_buf = |n: usize| -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.5f32; n * 2],
            sample_rate: 48000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    };

    let mut group = c.benchmark_group("audio_buffer_append");
    for frame_count in [100usize, 4096, 65536] {
        group.throughput(criterion::Throughput::Elements((frame_count * 2) as u64));
        group.bench_with_input(
            BenchmarkId::new("append_stereo", frame_count),
            &frame_count,
            |b, &n| {
                b.iter(|| {
                    let mut dst = make_buf(n);
                    let src = make_buf(n);
                    dst.append(&src).expect("append should succeed");
                    black_box(dst)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_format_conversion, bench_audio_buffer_append);
criterion_main!(benches);
