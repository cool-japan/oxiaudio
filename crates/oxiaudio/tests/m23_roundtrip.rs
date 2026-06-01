//! M23 integration tests: facade-level roundtrip and API tests.
//!
//! Tests cover:
//! 1. WAV roundtrip encode→decode (stereo, 44100 Hz)
//! 2. FLAC roundtrip encode→decode (mono, 48000 Hz)
//! 3. DSP normalize then gain convenience via facade
//! 4. decode_file_with_metadata returns valid metadata
//! 5. detect_format on WAV and FLAC files
//! 6. Streaming decode produces same total samples as one-shot decode

use oxiaudio::{
    decode_file, decode_file_with_metadata, decode_stream_with_block_size, detect_format, dsp,
    encode_flac, encode_wav, probe_metadata, AudioBuffer, ChannelLayout, SampleFormat,
};
use std::f32::consts::PI;

/// Build an interleaved sine-wave buffer.
///
/// `freq_hz` is the sine frequency, `sample_rate` the sample rate (Hz),
/// `duration_secs` the duration, and `channels` the number of channels (1 or 2).
fn make_sine_buf(
    freq_hz: f32,
    sample_rate: u32,
    duration_secs: f32,
    channels: usize,
) -> AudioBuffer<f32> {
    let frames = (sample_rate as f32 * duration_secs) as usize;
    let samples: Vec<f32> = (0..frames * channels)
        .map(|i| {
            let frame = i / channels;
            (2.0 * PI * freq_hz * frame as f32 / sample_rate as f32).sin() * 0.8
        })
        .collect();
    let layout = if channels == 1 {
        ChannelLayout::Mono
    } else {
        ChannelLayout::Stereo
    };
    AudioBuffer {
        samples,
        sample_rate,
        channels: layout,
        format: SampleFormat::F32,
    }
}

// ─── Test 1: WAV roundtrip encode→decode ──────────────────────────────────────

/// Encode a 0.5s stereo 44100 Hz 440 Hz sine wave to WAV, decode it back,
/// and verify sample rate, channel layout, sample count, and max error.
#[test]
#[cfg(feature = "pure")]
fn test_wav_roundtrip_stereo() {
    let original = make_sine_buf(440.0, 44_100, 0.5, 2);
    let path = std::env::temp_dir().join("oxiaudio_m23_wav_roundtrip.wav");

    encode_wav(&original, &path).expect("encode_wav failed");

    let decoded = decode_file(&path).expect("decode_file WAV failed");
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        decoded.sample_rate, 44_100,
        "sample_rate mismatch: got {}",
        decoded.sample_rate
    );
    assert_eq!(
        decoded.channels,
        ChannelLayout::Stereo,
        "channel layout should be Stereo"
    );

    // Sample count must be within 10% of original.
    let orig_len = original.samples.len();
    let dec_len = decoded.samples.len();
    let tolerance = orig_len / 10;
    assert!(
        dec_len.abs_diff(orig_len) <= tolerance,
        "sample count too far off: orig={orig_len} decoded={dec_len}"
    );

    // WAV f32 is lossless — max abs difference must be <= 0.001.
    let max_err = original
        .samples
        .iter()
        .zip(decoded.samples.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err <= 0.001,
        "max sample error {max_err} exceeds 0.001 tolerance (WAV f32 should be lossless)"
    );
}

// ─── Test 2: FLAC roundtrip encode→decode ─────────────────────────────────────

/// Encode a 0.5s mono 48000 Hz 220 Hz sine wave to FLAC, decode it back,
/// and verify sample rate, channel layout, sample count, and max error.
#[test]
#[cfg(feature = "pure")]
fn test_flac_roundtrip_mono() {
    let original = make_sine_buf(220.0, 48_000, 0.5, 1);
    let path = std::env::temp_dir().join("oxiaudio_m23_flac_roundtrip.flac");

    encode_flac(&original, &path).expect("encode_flac failed");

    let decoded = decode_file(&path).expect("decode_file FLAC failed");
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        decoded.sample_rate, 48_000,
        "sample_rate mismatch: got {}",
        decoded.sample_rate
    );
    assert_eq!(
        decoded.channels,
        ChannelLayout::Mono,
        "channel layout should be Mono"
    );

    // Sample count within 1%.
    let orig_len = original.samples.len();
    let dec_len = decoded.samples.len();
    let tolerance = (orig_len / 100).max(1);
    assert!(
        dec_len.abs_diff(orig_len) <= tolerance,
        "FLAC sample count too far off: orig={orig_len} decoded={dec_len}"
    );

    // FLAC is lossless — max abs difference must be <= 0.001.
    let max_err = original
        .samples
        .iter()
        .zip(decoded.samples.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err <= 0.001,
        "max FLAC sample error {max_err} exceeds 0.001 tolerance"
    );
}

// ─── Test 3: DSP normalize + gain via facade ──────────────────────────────────

/// Generate a 1s mono 44100 Hz 1000 Hz sine at full amplitude (0.8), normalize to
/// -6 dBFS (expected peak ≈ 0.501), then apply +6 dB gain (expected peak ≈ 1.0).
#[test]
#[cfg(feature = "pure")]
fn test_dsp_normalize_and_gain() {
    // Generate with amplitude 0.8 (not at max) so normalize has work to do.
    let mut buf = make_sine_buf(1000.0, 44_100, 1.0, 1);

    // Scale to 0.5 amplitude first to be well below -6 dBFS.
    for s in &mut buf.samples {
        *s *= 0.5;
    }

    // Normalize to -6 dBFS: target peak = 10^(-6/20) ≈ 0.501.
    dsp::normalize(&mut buf, -6.0);

    let peak_after_norm = buf
        .samples
        .iter()
        .cloned()
        .fold(0.0_f32, |a, s| a.max(s.abs()));
    let expected_peak_norm = 10.0_f32.powf(-6.0 / 20.0); // ≈ 0.501
    assert!(
        (peak_after_norm - expected_peak_norm).abs() <= 0.01,
        "peak after normalize to -6 dBFS: got {peak_after_norm:.4}, expected {expected_peak_norm:.4} ± 0.01"
    );

    // Apply +6 dB gain to restore to ≈ 1.0.
    dsp::gain(&mut buf, 6.0);

    let peak_after_gain = buf
        .samples
        .iter()
        .cloned()
        .fold(0.0_f32, |a, s| a.max(s.abs()));
    assert!(
        (peak_after_gain - 1.0_f32).abs() <= 0.01,
        "peak after +6 dB gain: got {peak_after_gain:.4}, expected ≈ 1.0 ± 0.01"
    );
}

// ─── Test 4: decode_file_with_metadata returns valid metadata ─────────────────

/// Encode a WAV file, then probe its metadata via both decode_file_with_metadata
/// and probe_metadata. Verify that structural fields match expectations.
#[test]
#[cfg(feature = "pure")]
fn test_decode_file_with_metadata_wav() {
    let buf = make_sine_buf(440.0, 44_100, 0.5, 2);
    let path = std::env::temp_dir().join("oxiaudio_m23_metadata.wav");

    encode_wav(&buf, &path).expect("encode_wav for metadata test");

    // decode_file_with_metadata returns (AudioBuffer, AudioMetadata).
    let (decoded, _metadata) =
        decode_file_with_metadata(&path).expect("decode_file_with_metadata failed");

    // Also verify probe_metadata does not error.
    let probed = probe_metadata(&path).expect("probe_metadata failed");

    let _ = std::fs::remove_file(&path);

    assert_eq!(
        decoded.sample_rate, 44_100,
        "decoded sample_rate should be 44100"
    );
    assert_eq!(
        decoded.channels,
        ChannelLayout::Stereo,
        "decoded channel layout should be Stereo"
    );

    // WAV files typically have no title/artist tags; only verify the probe succeeds.
    // duration_secs may or may not be set depending on the container.
    let _ = probed;
}

// ─── Test 5: detect_format on WAV and FLAC files ──────────────────────────────

/// Create a temp WAV and FLAC file, call detect_format on each, and verify that
/// the returned AudioFormat has the correct sample_rate.
#[test]
#[cfg(feature = "pure")]
fn test_detect_format_wav_and_flac() {
    let wav_buf = make_sine_buf(440.0, 44_100, 0.1, 1);
    let flac_buf = make_sine_buf(220.0, 48_000, 0.1, 2);

    let wav_path = std::env::temp_dir().join("oxiaudio_m23_detect.wav");
    let flac_path = std::env::temp_dir().join("oxiaudio_m23_detect.flac");

    encode_wav(&wav_buf, &wav_path).expect("encode_wav for detect_format test");
    encode_flac(&flac_buf, &flac_path).expect("encode_flac for detect_format test");

    let wav_fmt = detect_format(&wav_path).expect("detect_format on WAV failed");
    let flac_fmt = detect_format(&flac_path).expect("detect_format on FLAC failed");

    let _ = std::fs::remove_file(&wav_path);
    let _ = std::fs::remove_file(&flac_path);

    assert_eq!(
        wav_fmt.sample_rate, 44_100,
        "WAV detect_format: sample_rate should be 44100, got {}",
        wav_fmt.sample_rate
    );

    assert_eq!(
        flac_fmt.sample_rate, 48_000,
        "FLAC detect_format: sample_rate should be 48000, got {}",
        flac_fmt.sample_rate
    );
}

// ─── Test 6: Streaming decode produces same total samples as one-shot decode ──

/// Encode a 1s stereo WAV, decode one-shot with decode_file, decode streaming
/// with block_size=512, and verify total sample counts match and first 100
/// samples match within 0.001.
#[test]
#[cfg(feature = "pure")]
fn test_streaming_decode_matches_oneshot() {
    let original = make_sine_buf(440.0, 44_100, 1.0, 2);
    let path = std::env::temp_dir().join("oxiaudio_m23_streaming.wav");

    encode_wav(&original, &path).expect("encode_wav for streaming test");

    // One-shot decode.
    let oneshot = decode_file(&path).expect("one-shot decode_file failed");

    // Streaming decode with block_size=512 frames.
    let file = std::fs::File::open(&path).expect("open file for streaming");
    let reader = std::io::BufReader::new(file);
    let mut streamed_samples: Vec<f32> = Vec::new();
    for chunk_result in decode_stream_with_block_size(reader, 512) {
        let chunk = chunk_result.expect("streaming chunk error");
        streamed_samples.extend_from_slice(&chunk.samples);
    }

    let _ = std::fs::remove_file(&path);

    // Total sample counts must match within ±2 samples per channel (i.e. ±4 for stereo).
    let oneshot_len = oneshot.samples.len();
    let streamed_len = streamed_samples.len();
    assert!(
        oneshot_len.abs_diff(streamed_len) <= 4,
        "total sample count mismatch: one-shot={oneshot_len} streaming={streamed_len}"
    );

    // First 100 samples must match within 0.001.
    let compare_count = 100.min(oneshot_len).min(streamed_len);
    for (i, (a, b)) in oneshot
        .samples
        .iter()
        .zip(streamed_samples.iter())
        .enumerate()
        .take(compare_count)
    {
        let diff = (a - b).abs();
        assert!(
            diff <= 0.001,
            "sample[{i}] mismatch: one-shot={a} streaming={b} diff={diff}"
        );
    }
}
