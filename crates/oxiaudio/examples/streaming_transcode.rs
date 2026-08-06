//! Streaming transcode: decode in chunks → apply DSP filters → encode.
//!
//! [`TranscodeStream`] is oxiaudio's chunked transcode pipeline: it reads the
//! input in fixed-size blocks (`with_chunk_frames`), applies a list of
//! [`oxiaudio_core::AudioFilter`] implementors to the assembled buffer, and
//! writes the result in whatever format the output path's extension selects
//! (`.wav` / `.flac` / `.aiff`). It is the API to reach for when converting
//! between container formats with a couple of DSP steps in between, without
//! hand-writing the decode/process/encode loop from [`decode_dsp_encode`]
//! (see that example for the manual-chain version of the same idea).
//!
//! Run with:
//! ```text
//! cargo run -p oxiaudio --example streaming_transcode
//! ```

use std::f32::consts::PI;
use std::path::PathBuf;

use oxiaudio::dsp::{BiquadFilter, Compressor};
use oxiaudio::{
    decode_file, encode_wav, AudioBuffer, ChannelLayout, SampleFormat, TranscodeStream,
};

/// A 3-second, 44.1 kHz mono sawtooth-ish buzz (sum of two sines) at -6 dBFS —
/// enough harmonic content that a low-pass filter and a compressor visibly do
/// something, without shipping a fixture audio file.
fn synth_buzz(seconds: f32, sample_rate: u32) -> AudioBuffer<f32> {
    let n_frames = (seconds * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let t = i as f32 / sample_rate as f32;
        let fundamental = (2.0 * PI * 220.0 * t).sin();
        let harmonic = (2.0 * PI * 660.0 * t).sin() * 0.4;
        samples.push((fundamental + harmonic) * 0.4);
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp_dir = std::env::temp_dir();
    let input_wav: PathBuf = tmp_dir.join("oxiaudio_example_streaming_transcode_input.wav");
    let output_flac: PathBuf = tmp_dir.join("oxiaudio_example_streaming_transcode_output.flac");

    // ── Setup: an input file to stream from ─────────────────────────────────
    let buzz = synth_buzz(3.0, 44_100);
    encode_wav(&buzz, &input_wav)?;
    println!("wrote synthetic input: {}", input_wav.display());

    // ── Streaming transcode: WAV -> (low-pass + compressor) -> FLAC ─────────
    let low_pass = BiquadFilter::lowpass(1_500.0, std::f32::consts::FRAC_1_SQRT_2, 44_100);
    let compressor = Compressor::new(-16.0, 4.0, 5.0, 80.0);

    TranscodeStream::new(&input_wav, &output_flac)?
        .with_chunk_frames(4_096) // decode in 4096-frame blocks
        .with_filter(Box::new(low_pass))
        .with_filter(Box::new(compressor))
        .run()?;
    println!(
        "streamed transcode: {} -> {}",
        input_wav.display(),
        output_flac.display()
    );

    // Verify the output actually decodes and carries the same frame count
    // (mono WAV -> mono FLAC, filters don't change frame count).
    let original = decode_file(&input_wav)?;
    let transcoded = decode_file(&output_flac)?;
    println!(
        "original: {} frames, transcoded: {} frames, sample_rate {} -> {}",
        original.frame_count(),
        transcoded.frame_count(),
        original.sample_rate,
        transcoded.sample_rate
    );
    assert_eq!(original.frame_count(), transcoded.frame_count());

    let _ = std::fs::remove_file(&input_wav);
    Ok(())
}
