use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use oxiaudio_dsp::{
    detect_pitch_yin, gain, gain_inplace, loudness_integrated, normalize, normalize_inplace,
    resample, stft, BiquadFilter, Compressor, ConvolutionReverb, WindowFn,
};

fn make_mono_buf(n_samples: usize) -> AudioBuffer<f32> {
    AudioBuffer {
        samples: vec![0.5f32; n_samples],
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

fn make_stereo_buf(seconds: f32, sample_rate: u32) -> AudioBuffer<f32> {
    let frames = (seconds * sample_rate as f32) as usize;
    let samples: Vec<f32> = (0..frames * 2)
        .map(|i| {
            (2.0 * std::f32::consts::PI * 440.0 * i as f32 / (sample_rate as f32 * 2.0)).sin() * 0.5
        })
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

fn bench_biquad_lowpass(c: &mut Criterion) {
    let buf = make_stereo_buf(10.0, 48_000);
    let filter = BiquadFilter::lowpass(1_000.0, 0.707, 48_000);

    c.bench_function("biquad_lowpass_10s_stereo_48k", |b| {
        b.iter(|| {
            let _ = filter.process(&buf);
        });
    });
}

fn bench_biquad_multichannel(c: &mut Criterion) {
    let buf = make_stereo_buf(10.0, 48_000);
    let filter = BiquadFilter::lowpass(1_000.0, 0.707, 48_000);

    c.bench_function("biquad_multichannel_10s_stereo_48k", |b| {
        b.iter(|| {
            let _ = filter.process_multichannel(&buf);
        });
    });
}

fn bench_compressor(c: &mut Criterion) {
    let buf = make_stereo_buf(10.0, 48_000);
    let comp = Compressor::new(-20.0, 4.0, 10.0, 100.0);

    c.bench_function("compressor_10s_stereo_48k", |b| {
        b.iter(|| {
            let _ = comp.process(&buf);
        });
    });
}

fn bench_stft(c: &mut Criterion) {
    let frames = 44_100 * 5; // 5 seconds mono
    let samples: Vec<f32> = (0..frames)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin())
        .collect();
    let buf = AudioBuffer {
        samples,
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };

    c.bench_function("stft_5s_mono_44100_n2048", |b| {
        b.iter(|| {
            let _ = stft(&buf, 2048, 512, WindowFn::Hann);
        });
    });
}

fn bench_convolution_reverb(c: &mut Criterion) {
    // 2s IR at 48 kHz (sine-derived to avoid trivial cases)
    let ir_len = 48_000usize * 2;
    let ir: Vec<f32> = (0..ir_len)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            let decay = (-3.0 * t).exp();
            decay * (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.5
        })
        .collect();
    let reverb = ConvolutionReverb::new(ir).with_wet(0.3).with_dry(0.7);

    // 10s stereo 48 kHz input
    let input = make_stereo_buf(10.0, 48_000);

    c.bench_function("convolution_reverb_2s_ir_10s_stereo_48k", |b| {
        b.iter(|| {
            let _ = reverb.process(&input);
        });
    });
}

fn bench_yin_pitch(c: &mut Criterion) {
    // 10s mono 48 kHz 440 Hz sine
    let frames = 48_000usize * 10;
    let samples: Vec<f32> = (0..frames)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.8)
        .collect();
    let buf = AudioBuffer {
        samples,
        sample_rate: 48_000,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };

    c.bench_function("yin_pitch_10s_mono_48k", |b| {
        b.iter(|| {
            let _ = detect_pitch_yin(&buf, 2048, 512, 0.1);
        });
    });
}

fn bench_ebu_r128(c: &mut Criterion) {
    // 60s stereo 48 kHz
    let frames = 48_000usize * 60;
    let samples: Vec<f32> = (0..(frames * 2))
        .map(|i| {
            let t = i as f32 / (48_000.0 * 2.0);
            (2.0 * std::f32::consts::PI * 1_000.0 * t).sin() * 0.07
        })
        .collect();
    let buf = AudioBuffer {
        samples,
        sample_rate: 48_000,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    };

    c.bench_function("ebu_r128_loudness_60s_stereo_48k", |b| {
        b.iter(|| {
            let _ = loudness_integrated(&buf);
        });
    });
}

fn bench_gain(c: &mut Criterion) {
    let mut group = c.benchmark_group("gain");
    for n in [4_096usize, 65_536, 524_288] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("gain_db", n), &n, |b, &n| {
            let mut buf = make_mono_buf(n);
            b.iter(|| gain(&mut buf, -6.0));
        });
        group.bench_with_input(BenchmarkId::new("gain_inplace", n), &n, |b, &n| {
            let mut buf = make_mono_buf(n);
            b.iter(|| gain_inplace(&mut buf, 0.5));
        });
    }
    group.finish();
}

fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");
    for n in [4_096usize, 65_536, 524_288] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("normalize_db", n), &n, |b, &n| {
            let mut buf = make_mono_buf(n);
            b.iter(|| normalize(&mut buf, 0.0));
        });
        group.bench_with_input(BenchmarkId::new("normalize_inplace", n), &n, |b, &n| {
            let mut buf = make_mono_buf(n);
            b.iter(|| normalize_inplace(&mut buf, 1.0));
        });
    }
    group.finish();
}

fn bench_resample(c: &mut Criterion) {
    // 1 second stereo 48 kHz → 44.1 kHz (representative downsample)
    let buf = make_stereo_buf(1.0, 48_000);
    c.bench_function("resample_sinc_48k_to_44100_1s_stereo", |b| {
        b.iter(|| resample(&buf, 44_100).expect("resample failed"));
    });
}

criterion_group!(
    benches,
    bench_biquad_lowpass,
    bench_biquad_multichannel,
    bench_compressor,
    bench_stft,
    bench_convolution_reverb,
    bench_yin_pitch,
    bench_ebu_r128,
    bench_gain,
    bench_normalize,
    bench_resample
);
criterion_main!(benches);
