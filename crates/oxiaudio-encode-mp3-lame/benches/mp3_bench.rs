//! Criterion benchmarks for `oxiaudio-encode-mp3-lame`.
//!
//! Benchmarks:
//! - CBR 128 kbps stereo 10 s (baseline)
//! - CBR 320 kbps stereo 10 s (high bitrate)
//! - VBR V2 (~190 kbps) stereo 10 s (quality preset)
//!
//! Run with:
//! ```text
//! cargo bench -p oxiaudio-encode-mp3-lame --features mp3-encode-lame
//! ```

#[cfg(feature = "mp3-encode-lame")]
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(feature = "mp3-encode-lame")]
use oxiaudio_core::{AudioBuffer, AudioEncoder, ChannelLayout, SampleFormat};
#[cfg(feature = "mp3-encode-lame")]
use oxiaudio_encode_mp3_lame::lame::{LameMode, LameMp3Encoder};
#[cfg(feature = "mp3-encode-lame")]
use std::io::Cursor;

#[cfg(feature = "mp3-encode-lame")]
fn make_sine_stereo_44k(secs: f32) -> AudioBuffer<f32> {
    let sr = 44_100u32;
    let n = (sr as f32 * secs) as usize;
    let mut samples = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let s = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        samples.push(s);
        samples.push(s);
    }
    AudioBuffer {
        samples,
        sample_rate: sr,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

/// Benchmark CBR 128 kbps — 10 s stereo 44.1 kHz.
#[cfg(feature = "mp3-encode-lame")]
fn bench_cbr_128(c: &mut Criterion) {
    let buf = make_sine_stereo_44k(10.0);
    c.bench_function("encode_mp3_cbr_128kbps_10s_stereo_44k", |b| {
        b.iter(|| {
            let mut out = Cursor::new(Vec::new());
            let mut enc = LameMp3Encoder {
                bitrate: 128,
                mode: LameMode::JointStereo,
                id3_tags: None,
            };
            enc.encode(&buf, &mut out).expect("encode failed");
            std::hint::black_box(out.into_inner());
        });
    });
}

/// Benchmark CBR 320 kbps — 10 s stereo 44.1 kHz.
#[cfg(feature = "mp3-encode-lame")]
fn bench_cbr_320(c: &mut Criterion) {
    let buf = make_sine_stereo_44k(10.0);
    c.bench_function("encode_mp3_cbr_320kbps_10s_stereo_44k", |b| {
        b.iter(|| {
            let mut out = Cursor::new(Vec::new());
            let mut enc = LameMp3Encoder {
                bitrate: 320,
                mode: LameMode::JointStereo,
                id3_tags: None,
            };
            enc.encode(&buf, &mut out).expect("encode failed");
            std::hint::black_box(out.into_inner());
        });
    });
}

/// Benchmark VBR V2 (~190 kbps) — 10 s stereo 44.1 kHz.
#[cfg(feature = "mp3-encode-lame")]
fn bench_vbr_v2(c: &mut Criterion) {
    let buf = make_sine_stereo_44k(10.0);
    c.bench_function("encode_mp3_vbr_v2_10s_stereo_44k", |b| {
        b.iter(|| {
            let mut out = Cursor::new(Vec::new());
            let mut enc = LameMp3Encoder {
                bitrate: 128, // ignored in VBR mode
                mode: LameMode::Vbr { quality: 2 },
                id3_tags: None,
            };
            enc.encode(&buf, &mut out).expect("encode failed");
            std::hint::black_box(out.into_inner());
        });
    });
}

#[cfg(feature = "mp3-encode-lame")]
criterion_group!(mp3_benches, bench_cbr_128, bench_cbr_320, bench_vbr_v2);
#[cfg(feature = "mp3-encode-lame")]
criterion_main!(mp3_benches);

// When the feature is disabled the bench binary must still compile.
#[cfg(not(feature = "mp3-encode-lame"))]
fn main() {}
