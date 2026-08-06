//! Building a DSP chain directly on `oxiaudio-dsp`, without the `oxiaudio` facade.
//!
//! Crates that only need signal processing (no container decode/encode) can
//! depend on `oxiaudio-dsp` directly. This example builds a synthetic buffer,
//! resamples it, then runs it through a [`DspChain`] of a low-pass
//! [`BiquadFilter`] followed by a [`Compressor`] — both implement
//! [`oxiaudio_core::AudioFilter`], so [`DspChain::then_filter`] takes them
//! directly with no adapter closure needed. [`PeakMeter`] / [`RmsMeter`]
//! measure the buffer before and after so the effect of the chain is visible
//! in the printed output, not just asserted.
//!
//! Run with:
//! ```text
//! cargo run -p oxiaudio-dsp --example dsp_chain
//! ```

use std::f32::consts::PI;

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use oxiaudio_dsp::{resample, BiquadFilter, Compressor, DspChain, PeakMeter, RmsMeter};

/// A 1-second, 44.1 kHz mono buffer: a quiet 220 Hz fundamental plus a loud,
/// bright 6 kHz "hot" burst in the back half — enough for a low-pass filter
/// (kills the 6 kHz burst) and a compressor (evens out the level jump) to
/// produce a visibly different peak/RMS reading than the input.
fn synth_buffer_with_hot_burst(sample_rate: u32) -> AudioBuffer<f32> {
    let n_frames = sample_rate as usize;
    let mut samples = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let t = i as f32 / sample_rate as f32;
        let fundamental = (2.0 * PI * 220.0 * t).sin() * 0.2;
        let hot_burst = if i > n_frames / 2 {
            (2.0 * PI * 6_000.0 * t).sin() * 0.9
        } else {
            0.0
        };
        samples.push(fundamental + hot_burst);
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

/// Peak and RMS of an entire mono buffer, in dBFS / linear RMS.
fn measure(buf: &AudioBuffer<f32>) -> (f32, f32) {
    let mut peak_meter = PeakMeter::new(1_000.0, 0.0, buf.sample_rate);
    let peak_db = peak_meter.process_block(&buf.samples);

    let mut rms_meter = RmsMeter::new(1_000.0, buf.sample_rate);
    let mut rms = 0.0f32;
    for &s in &buf.samples {
        rms = rms_meter.process_sample(s);
    }
    (peak_db, rms)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = synth_buffer_with_hot_burst(44_100);
    let (peak_before, rms_before) = measure(&input);
    println!(
        "input:     {} frames @ {} Hz, peak {:.1} dBFS, RMS {:.4}",
        input.frame_count(),
        input.sample_rate,
        peak_before,
        rms_before
    );

    // 1. Resample up to 48 kHz (a common step before further processing/encode).
    let resampled = resample(&input, 48_000)?;
    println!(
        "resampled: {} frames @ {} Hz",
        resampled.frame_count(),
        resampled.sample_rate
    );

    // 2. Chain a low-pass filter (kills the 6 kHz burst) and a compressor
    //    (evens out what's left of the level jump) — both are AudioFilter
    //    impls, so `then_filter` takes them directly.
    let chain = DspChain::new()
        .then_filter(BiquadFilter::lowpass(
            2_000.0,
            std::f32::consts::FRAC_1_SQRT_2,
            48_000,
        ))
        .then_filter(Compressor::new(-24.0, 4.0, 5.0, 80.0));

    let processed = chain.process(&resampled)?;
    let (peak_after, rms_after) = measure(&processed);
    println!(
        "processed: {} frames @ {} Hz, peak {:.1} dBFS, RMS {:.4}",
        processed.frame_count(),
        processed.sample_rate,
        peak_after,
        rms_after
    );

    println!(
        "peak reduced by {:.1} dB; low-pass + compressor visibly tamed the hot burst",
        peak_before - peak_after
    );
    assert!(
        peak_after < peak_before,
        "low-pass + compressor chain should reduce peak level"
    );

    Ok(())
}
