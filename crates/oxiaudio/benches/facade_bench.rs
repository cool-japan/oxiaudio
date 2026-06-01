use criterion::{criterion_group, criterion_main, Criterion};
use oxiaudio::{AudioBuffer, ChannelLayout, SampleFormat};
use std::f32::consts::PI;

fn make_stereo_buf(sample_rate: u32, seconds: f32) -> AudioBuffer<f32> {
    let n_frames = (sample_rate as f32 * seconds) as usize;
    let mut samples = Vec::with_capacity(n_frames * 2);
    for i in 0..n_frames {
        let t = i as f32 / sample_rate as f32;
        let s = (2.0 * PI * 440.0 * t).sin() * 0.5;
        samples.push(s);
        samples.push(s);
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

fn bench_decode_wav_10s(c: &mut Criterion) {
    let buf = make_stereo_buf(48_000, 10.0);
    let tmp = std::env::temp_dir().join("oxiaudio_facade_bench_decode.wav");
    oxiaudio::encode_wav(&buf, &tmp).expect("encode_wav");
    c.bench_function("decode_wav_10s_48k_stereo", |b| {
        b.iter(|| oxiaudio::decode_file(&tmp).expect("decode_file"));
    });
    let _ = std::fs::remove_file(&tmp);
}

fn bench_encode_wav_10s(c: &mut Criterion) {
    let buf = make_stereo_buf(48_000, 10.0);
    let tmp = std::env::temp_dir().join("oxiaudio_facade_bench_encode.wav");
    c.bench_function("encode_wav_10s_48k_stereo", |b| {
        b.iter(|| oxiaudio::encode_wav(&buf, &tmp).expect("encode_wav"));
    });
    let _ = std::fs::remove_file(&tmp);
}

fn bench_resample(c: &mut Criterion) {
    let buf = make_stereo_buf(48_000, 5.0);
    c.bench_function("resample_48k_to_44100_5s_stereo", |b| {
        b.iter(|| oxiaudio::dsp::resample(&buf, 44_100).expect("resample"));
    });
}

fn bench_gain(c: &mut Criterion) {
    let buf = make_stereo_buf(48_000, 10.0);
    c.bench_function("gain_6db_10s_stereo", |b| {
        b.iter(|| {
            let mut tmp = buf.clone();
            oxiaudio::dsp::gain(&mut tmp, 6.0);
        });
    });
}

fn bench_stft(c: &mut Criterion) {
    let buf = make_stereo_buf(48_000, 2.0);
    c.bench_function("stft_2048_512_hann_2s", |b| {
        b.iter(|| {
            oxiaudio::dsp::spectral::stft(&buf, 2048, 512, oxiaudio::dsp::spectral::WindowFn::Hann)
                .expect("stft")
        });
    });
}

criterion_group!(
    benches,
    bench_decode_wav_10s,
    bench_encode_wav_10s,
    bench_resample,
    bench_gain,
    bench_stft
);
criterion_main!(benches);
