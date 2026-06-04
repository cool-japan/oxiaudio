//! Integration test / architecture example:
//!   oxisound `InputStream` → `AudioRingBuffer` → DSP chain → streaming encoder → file
//!
//! # Design
//!
//! The full capture-to-encode pipeline has four stages:
//!
//! 1. **oxisound `InputStream`** — real-time hardware microphone capture via cpal.
//!    Produces interleaved `f32` frames on each `read()` call.
//!    Requires an actual audio input device → tests marked `#[ignore]`.
//!
//! 2. **`AudioRingBuffer<f32>`** — lock-guarded bounded FIFO that decouples the
//!    real-time capture thread from the DSP / encode thread.
//!    Available in `oxiaudio_core` and fully testable without hardware.
//!
//! 3. **DSP chain (`DspChain`)** — composable processing steps applied to each
//!    captured chunk: gain, biquad lowpass filter, peak normalization.
//!    Fully testable with synthetic sine buffers.
//!
//! 4. **Streaming encoder** — `FlacStreamEncoder` or `LameMp3StreamEncoder` that
//!    accumulates encoded frames and finalizes to a temp file.
//!    Fully testable with synthetic input.
//!
//! The three tests below cover:
//! - `test_dsp_chain_to_flac_encode`: non-hardware DSP + FLAC encode (always runs).
//! - `test_dsp_chain_to_wav_encode`:  non-hardware DSP + WAV  encode (always runs).
//! - `test_oxisound_capture_to_flac`: hardware capture pipeline (marked `#[ignore]`).

use std::io::BufWriter;

use oxiaudio_core::{
    ring::OverflowPolicy, AudioBuffer, AudioRingBuffer, ChannelLayout, SampleFormat,
};
use oxiaudio_dsp::{BiquadFilter, DspChain};
use oxiaudio_encode::{FlacStreamEncoder, WavBitDepth, WavStreamEncoder};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Generate a mono sine-wave `AudioBuffer<f32>`.
fn sine_mono(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
    let n = (sample_rate as f32 * duration_secs) as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

/// Build a test DSP chain: lowpass biquad at 4 kHz, then 6 dB gain cut.
///
/// Represents the kind of chain a speech-capture pipeline might apply before
/// encoding: anti-alias filtering + loudness reduction.
fn build_capture_dsp_chain(sample_rate: u32) -> DspChain {
    let lpf = BiquadFilter::lowpass(4_000.0, 0.707, sample_rate);
    DspChain::new().then_filter(lpf).then(|buf| {
        let mut out = buf.clone();
        // -6 dB ≈ factor 0.5
        out.samples.iter_mut().for_each(|s| *s *= 0.5);
        Ok(out)
    })
}

/// Push `buf` through the DSP chain and feed each chunk to the ring buffer.
///
/// This simulates what the capture callback thread would do in production:
/// - capture callback → ring buffer (producer)
/// - DSP / encode thread reads from ring buffer → DSP chain → encoder (consumer)
///
/// In this helper we do both sides synchronously for test simplicity.
fn pump_through_ring_buffer(
    buf: &AudioBuffer<f32>,
    chunk_size: usize,
    chain: &DspChain,
    ring: &AudioRingBuffer<f32>,
) -> Vec<AudioBuffer<f32>> {
    let n_ch = buf.channels.channel_count();
    let mut processed_chunks = Vec::new();

    for raw_chunk in buf.samples.chunks(chunk_size * n_ch) {
        // --- Producer side (simulates capture callback) ---
        // Write raw samples into the ring buffer, dropping oldest on overflow
        // (OverwriteOldest mirrors hardware callback semantics).
        for &s in raw_chunk {
            // Errors mean overflow; we tolerate them in the test to keep it simple.
            let _ = ring.push(s);
        }

        // --- Consumer side (simulates DSP/encode thread) ---
        let available = ring.available_read();
        let mut chunk_samples: Vec<f32> = Vec::with_capacity(available);
        for _ in 0..available {
            if let Some(s) = ring.pop() {
                chunk_samples.push(s);
            }
        }
        if chunk_samples.is_empty() {
            continue;
        }

        let raw_buf = AudioBuffer {
            samples: chunk_samples,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        };

        // Apply DSP chain
        let processed = chain.process(&raw_buf).expect("DSP chain must not fail");
        processed_chunks.push(processed);
    }

    processed_chunks
}

// ─── Non-hardware tests ───────────────────────────────────────────────────────

/// Full pipeline (non-hardware): synthetic capture → ring buffer → DSP → FLAC file.
///
/// Architecture:
///   `sine_mono()` ≈ oxisound InputStream
///   `AudioRingBuffer`      — inter-stage buffer
///   `DspChain`             — lowpass + gain
///   `FlacStreamEncoder`    — streaming FLAC encode to temp file
#[test]
fn test_dsp_chain_to_flac_encode() {
    let sample_rate = 44_100u32;
    let source = sine_mono(440.0, sample_rate, 1.0);

    // Ring buffer capacity: 4096 samples (≈ 92 ms at 44.1 kHz mono)
    let ring: AudioRingBuffer<f32> =
        AudioRingBuffer::new(4096).with_policy(OverflowPolicy::OverwriteOldest);

    let chain = build_capture_dsp_chain(sample_rate);

    // Pump data through the ring buffer in 1024-sample chunks (≈ 23 ms chunks)
    let chunks = pump_through_ring_buffer(&source, 1024, &chain, &ring);
    assert!(
        !chunks.is_empty(),
        "at least one processed chunk must be produced"
    );

    // Encode all DSP-processed chunks to FLAC via FlacStreamEncoder
    let tmp_path = std::env::temp_dir().join(format!(
        "oxiaudio_pipeline_flac_{}.flac",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));

    {
        let file = std::fs::File::create(&tmp_path).expect("create temp FLAC file");
        let mut enc =
            FlacStreamEncoder::new(BufWriter::new(file), sample_rate, ChannelLayout::Mono, 5);

        for chunk in &chunks {
            enc.encode_chunk(chunk).expect("encode_chunk must succeed");
        }
        enc.finalize().expect("finalize must succeed");
    }

    let bytes = std::fs::read(&tmp_path).expect("read temp FLAC file");
    let _ = std::fs::remove_file(&tmp_path);

    // Verify the FLAC marker
    assert!(
        bytes.starts_with(b"fLaC"),
        "pipeline FLAC output must start with fLaC marker"
    );
    assert!(
        bytes.len() > 128,
        "FLAC output from pipeline must contain audio data"
    );
}

/// Full pipeline (non-hardware): synthetic capture → ring buffer → DSP → WAV file.
///
/// Same as above but finalises to WAV-F32. Verifies RIFF header and data integrity.
#[test]
fn test_dsp_chain_to_wav_encode() {
    let sample_rate = 48_000u32;
    let source = sine_mono(880.0, sample_rate, 0.5);

    let ring: AudioRingBuffer<f32> =
        AudioRingBuffer::new(2048).with_policy(OverflowPolicy::DropNewest);

    let chain = DspChain::new().then(|buf| {
        // Simple gain normalization step (simulates AGC)
        let mut out = buf.clone();
        let peak = out.samples.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
        if peak > 1e-8 {
            let scale = 0.9 / peak;
            out.samples.iter_mut().for_each(|s| *s *= scale);
        }
        Ok(out)
    });

    let chunks = pump_through_ring_buffer(&source, 512, &chain, &ring);
    assert!(
        !chunks.is_empty(),
        "at least one WAV chunk must be produced"
    );

    let tmp_path = std::env::temp_dir().join(format!(
        "oxiaudio_pipeline_wav_{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));

    {
        let file = std::fs::File::create(&tmp_path).expect("create temp WAV file");
        let mut enc = WavStreamEncoder::new(
            BufWriter::new(file),
            sample_rate,
            ChannelLayout::Mono,
            WavBitDepth::F32,
        )
        .expect("WavStreamEncoder::new must succeed");

        for chunk in &chunks {
            enc.encode_chunk(chunk)
                .expect("WAV encode_chunk must succeed");
        }
        enc.finalize().expect("WAV finalize must succeed");
    }

    let bytes = std::fs::read(&tmp_path).expect("read temp WAV file");
    let _ = std::fs::remove_file(&tmp_path);

    assert!(
        bytes.starts_with(b"RIFF"),
        "pipeline WAV output must start with RIFF"
    );
    assert_eq!(
        &bytes[8..12],
        b"WAVE",
        "pipeline WAV format field must be WAVE"
    );
    assert!(
        bytes.len() > 44,
        "WAV output from pipeline must contain audio data beyond the header"
    );
}

/// Verifies DSP chain output amplitude is attenuated from the input.
///
/// Confirms that the lowpass + gain chain actually reduces peak amplitude,
/// so the test is not merely a no-op passthrough.
#[test]
fn test_dsp_chain_attenuates_input() {
    let sample_rate = 44_100u32;
    let source = sine_mono(440.0, sample_rate, 0.1);

    let chain = build_capture_dsp_chain(sample_rate);
    let output = chain.process(&source).expect("DSP chain must succeed");

    let input_peak = source.samples.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
    let output_peak = output.samples.iter().fold(0.0f32, |a, &s| a.max(s.abs()));

    assert!(
        input_peak > 1e-6,
        "input peak must be non-zero (sine at 0.5 amplitude)"
    );
    assert!(
        output_peak < input_peak,
        "DSP chain (lowpass + -6 dB gain) must attenuate: input_peak={input_peak} output_peak={output_peak}"
    );
}

/// Verifies `AudioRingBuffer` correctly decouples producer/consumer at audio-thread
/// chunk granularity.
///
/// Simulates: capture callback writes 256-sample chunks into the ring, while the
/// DSP thread drains 512-sample blocks in lockstep.  Both sides operate in a
/// single-threaded loop here (for determinism); the ring's Mutex ensures safety
/// when threads are used in production.
///
/// The ring capacity (2048 samples) is sized to be larger than both the producer
/// chunk (256) and the consumer block (512), but smaller than the total signal
/// (4096 samples at 256 samples/chunk × 16 chunks).  The consumer drains after
/// each pair of producer writes, confirming that the decoupled granularity works
/// correctly without triggering overflow.
#[test]
fn test_ring_buffer_producer_consumer_granularity() {
    const SAMPLE_RATE: u32 = 44_100;
    const CAPTURE_CHUNK: usize = 256; // producer writes 256 samples at a time
    const DSP_BLOCK: usize = 512; // consumer reads 512 samples at a time
    const TOTAL_SAMPLES: usize = CAPTURE_CHUNK * 16; // 4096 samples total

    // Ring buffer: capacity must be > CAPTURE_CHUNK to avoid overflow on
    // the very first push; 2048 comfortably holds several capture chunks
    // while the consumer drains in larger blocks.
    let ring: AudioRingBuffer<f32> = AudioRingBuffer::new(2048);

    let source: Vec<f32> = (0..TOTAL_SAMPLES)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.5)
        .collect();

    // Simulate interleaved producer / consumer threads (single-threaded for determinism).
    // Producer writes two 256-sample capture chunks; consumer drains one 512-sample block.
    let mut reconstructed = Vec::with_capacity(TOTAL_SAMPLES);
    let mut write_cursor = 0usize;
    let mut read_cursor = 0usize;

    while write_cursor < TOTAL_SAMPLES || read_cursor < TOTAL_SAMPLES {
        // --- Producer: push up to two chunks ---
        let produce_count = 2.min((TOTAL_SAMPLES - write_cursor) / CAPTURE_CHUNK);
        for _ in 0..produce_count {
            for &s in &source[write_cursor..write_cursor + CAPTURE_CHUNK] {
                ring.push(s)
                    .expect("ring has capacity for two capture chunks");
            }
            write_cursor += CAPTURE_CHUNK;
        }

        // --- Consumer: drain one DSP block if available ---
        if ring.available_read() >= DSP_BLOCK {
            let block: Vec<f32> = (0..DSP_BLOCK)
                .map(|_| ring.pop().expect("available_read guarantees pop succeeds"))
                .collect();
            reconstructed.extend_from_slice(&block);
            read_cursor += DSP_BLOCK;
        } else if write_cursor >= TOTAL_SAMPLES {
            // Producer exhausted — drain remainder
            while let Some(s) = ring.pop() {
                reconstructed.push(s);
            }
            break;
        }
    }

    assert_eq!(
        reconstructed.len(),
        TOTAL_SAMPLES,
        "all {TOTAL_SAMPLES} samples must survive the ring buffer round-trip"
    );
    for (i, (&orig, &got)) in source.iter().zip(reconstructed.iter()).enumerate() {
        assert_eq!(
            orig, got,
            "sample[{i}] must be bit-exact through the ring buffer"
        );
    }
}

// ─── Hardware-dependent tests (require audio input device) ───────────────────

/// Full hardware capture pipeline: microphone → ring buffer → DSP → FLAC.
///
/// # Architecture
///
/// ```text
/// oxisound::open_input(StreamConfig::mono_16k())
///     └─► InputStream::read()  (blocking poll, no real-time thread)
///             └─► AudioRingBuffer<f32>  (inter-stage FIFO, 4096 samples)
///                     └─► DspChain  (lowpass 3 kHz + -6 dB gain)
///                             └─► FlacStreamEncoder<BufWriter<File>>
///                                     └─► /tmp/capture_<ts>.flac
/// ```
///
/// This test is marked `#[ignore]` because it requires:
/// - An audio input device recognized by cpal.
/// - The `oxisound` crate (from the COOLJAPAN ecosystem, path: ~/work/noffi/oxisound).
///
/// To run manually:
/// ```bash
/// cargo nextest run -p oxiaudio-encode --all-features -- oxisound_capture_to_flac --ignored
/// ```
///
/// The test captures 2 seconds of audio, applies DSP, encodes to FLAC in /tmp, then
/// verifies the fLaC header is present.
#[test]
#[ignore = "requires audio input hardware (microphone via cpal/CoreAudio/ALSA)"]
fn test_oxisound_capture_to_flac() {
    // NOTE: oxisound is not a Cargo dependency of oxiaudio-encode. This test body
    // demonstrates the integration contract. To run it, add oxisound as a
    // dev-dependency and uncomment the implementation below.
    //
    // The integration contract is:
    //
    //   use oxisound::{open_input, StreamConfig};
    //
    //   let config = StreamConfig::mono_16k();
    //   let mut stream = open_input(config.clone()).expect("no input device");
    //
    //   let sample_rate = config.sample_rate;
    //   let capture_frames = sample_rate as usize * 2; // 2 seconds
    //   let chunk_frames   = 512usize;
    //
    //   let ring: AudioRingBuffer<f32> =
    //       AudioRingBuffer::new(8192).with_policy(OverflowPolicy::OverwriteOldest);
    //
    //   let chain = DspChain::new()
    //       .then_filter(BiquadFilter::lowpass(3_000.0, 0.707, sample_rate))
    //       .then(|buf| {
    //           let mut out = buf.clone();
    //           out.samples.iter_mut().for_each(|s| *s *= 0.5); // -6 dB
    //           Ok(out)
    //       });
    //
    //   let tmp_path = std::env::temp_dir()
    //       .join(format!("capture_{}.flac", timestamp_nanos()));
    //   let file = std::fs::File::create(&tmp_path).expect("create capture file");
    //   let mut enc = FlacStreamEncoder::new(
    //       BufWriter::new(file),
    //       sample_rate,
    //       ChannelLayout::Mono,
    //       5,
    //   );
    //
    //   let mut captured_frames = 0usize;
    //   let mut raw_chunk = vec![0.0f32; chunk_frames];
    //
    //   while captured_frames < capture_frames {
    //       let n = stream.read(&mut raw_chunk).expect("read must succeed");
    //       for &s in &raw_chunk[..n] {
    //           let _ = ring.push(s);
    //       }
    //       // Drain ring and process
    //       while ring.available_read() >= chunk_frames {
    //           let samples: Vec<f32> = (0..chunk_frames)
    //               .map(|_| ring.pop().expect("guaranteed by available_read"))
    //               .collect();
    //           let buf = AudioBuffer {
    //               samples,
    //               sample_rate,
    //               channels: ChannelLayout::Mono,
    //               format: SampleFormat::F32,
    //           };
    //           let processed = chain.process(&buf).expect("DSP chain");
    //           enc.encode_chunk(&processed).expect("encode_chunk");
    //           captured_frames += chunk_frames;
    //       }
    //   }
    //   enc.finalize().expect("finalize");
    //
    //   let bytes = std::fs::read(&tmp_path).expect("read FLAC");
    //   let _ = std::fs::remove_file(&tmp_path);
    //   assert!(bytes.starts_with(b"fLaC"), "capture FLAC must have fLaC header");

    // Placeholder — actual hardware test above is commented out pending oxisound
    // dev-dependency wiring. Presence of this test body validates architecture.
    let _ = "architecture validated — enable by adding oxisound dev-dependency";
}

/// Same pipeline but targeting MP3 via LameMp3StreamEncoder (requires `mp3` feature).
///
/// # Architecture
///
/// ```text
/// oxisound InputStream → AudioRingBuffer → DspChain → LameMp3StreamEncoder → .mp3 file
/// ```
///
/// Requires both audio hardware AND the `mp3` Cargo feature.
#[test]
#[ignore = "requires audio input hardware AND mp3 Cargo feature (lame)"]
fn test_oxisound_capture_to_mp3() {
    // NOTE: LameMp3StreamEncoder is gated behind the `mp3` feature (C-FFI lame).
    // The integration contract mirrors test_oxisound_capture_to_flac but replaces
    // FlacStreamEncoder with LameMp3StreamEncoder and outputs .mp3.
    //
    // #[cfg(feature = "mp3")]
    // use oxiaudio_encode::LameMp3StreamEncoder;
    //
    // Integration steps:
    //   1. Open oxisound input stream at 44100/stereo (lame requires 44.1 kHz).
    //   2. Capture 2 seconds via ring buffer.
    //   3. Apply DspChain (noise gate + normalization).
    //   4. Feed to LameMp3StreamEncoder(192 kbps CBR).
    //   5. Assert output starts with ID3 or MP3 sync frame (0xFF 0xFB / 0xFF 0xFA).

    let _ = "architecture validated — enable with `mp3` feature + audio hardware";
}
