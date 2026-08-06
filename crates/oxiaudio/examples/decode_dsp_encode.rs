//! Core workflow: synthesize → encode → **decode** → DSP chain → **encode**.
//!
//! This is the canonical oxiaudio flow most users reach for first: read a file
//! ([`oxiaudio::decode_file`]), process it through a chain of DSP steps
//! ([`oxiaudio::dsp`] + [`DspChain`]), and write the result back out
//! ([`oxiaudio::encode_wav`] / [`oxiaudio::encode_flac`]).
//!
//! No fixture audio file ships with this crate, so this example first
//! synthesizes a 440 Hz stereo tone in memory and writes it to a temporary WAV
//! file — that write is only there to give `decode_file` a real file to read.
//! Everything from the `decode_file` call onward is the part worth reading.
//!
//! Run with:
//! ```text
//! cargo run -p oxiaudio --example decode_dsp_encode
//! ```

use std::f32::consts::PI;
use std::path::PathBuf;

use oxiaudio::dsp::DspChain;
use oxiaudio::{
    decode_file, dsp, encode_flac, encode_wav, AudioBuffer, ChannelLayout, SampleFormat,
};

/// Build a 2-second, 48 kHz, stereo 440 Hz sine tone at -6 dBFS.
///
/// Standing in for "the file the user actually wants to process" — every
/// oxiaudio example synthesizes its own input instead of shipping a fixture,
/// so it stays runnable with nothing but `cargo run`.
fn synth_tone(seconds: f32, sample_rate: u32) -> AudioBuffer<f32> {
    let n_frames = (seconds * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(n_frames * 2);
    for i in 0..n_frames {
        let t = i as f32 / sample_rate as f32;
        let s = (2.0 * PI * 440.0 * t).sin() * 0.5; // -6 dBFS
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp_dir = std::env::temp_dir();
    let input_wav: PathBuf = tmp_dir.join("oxiaudio_example_decode_dsp_encode_input.wav");
    let output_flac: PathBuf = tmp_dir.join("oxiaudio_example_decode_dsp_encode_output.flac");

    // ── Setup: write a synthetic tone so decode_file has something to read ──
    let tone = synth_tone(2.0, 48_000);
    encode_wav(&tone, &input_wav)?;
    println!("wrote synthetic input: {}", input_wav.display());

    // ── The actual workflow starts here ─────────────────────────────────────

    // 1. Decode.
    let decoded = decode_file(&input_wav)?;
    println!(
        "decoded: {} frames @ {} Hz, {:?}",
        decoded.frame_count(),
        decoded.sample_rate,
        decoded.channels
    );

    // 2. Build a reusable DSP chain: trim gain → parametric EQ → compressor → reverb.
    //    DspChain::then takes any `Fn(&AudioBuffer<f32>) -> Result<AudioBuffer<f32>, _>`,
    //    so in-place helpers like `dsp::gain` are wrapped in a small closure while
    //    helpers that already return a processed buffer (`dsp::eq`, `dsp::compressor`)
    //    are passed straight through.
    let chain = DspChain::new()
        .then(|buf| {
            let mut out = buf.clone();
            dsp::gain(&mut out, -3.0); // pull down 3 dB before the EQ/compressor
            Ok(out)
        })
        .then(|buf| dsp::eq(buf, &[(200.0, -2.0, 0.8), (3_000.0, 3.0, 1.2)]))
        .then(|buf| dsp::compressor(buf, -18.0, 3.0, 10.0, 120.0))
        .then(|buf| Ok(dsp::reverb(buf, 0.5, 0.35, 0.2)));

    let processed = chain.process(&decoded)?;
    println!(
        "processed: {} frames (chain: gain -> eq -> compressor -> reverb)",
        processed.frame_count()
    );

    // 3. Re-encode to a different format.
    encode_flac(&processed, &output_flac)?;
    println!("wrote processed output: {}", output_flac.display());

    // Round-trip check: the FLAC we just wrote should decode back losslessly.
    let roundtrip = decode_file(&output_flac)?;
    assert_eq!(roundtrip.frame_count(), processed.frame_count());
    println!("round-trip decode OK: {} frames", roundtrip.frame_count());

    let _ = std::fs::remove_file(&input_wav);
    Ok(())
}
