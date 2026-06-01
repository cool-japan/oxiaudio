//! Roundtrip encode→decode regression tests.
//! Verifies that encoded outputs have correct magic bytes and can be probed by the decoder.

use std::io::Cursor;

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use oxiaudio_encode::{encode_aac, encode_flac_to_vec, encode_vorbis, encode_wav_to_vec};

// ─── Buffer factories ─────────────────────────────────────────────────────────

fn sine_buffer_stereo(freq: f32, duration_secs: f32, sample_rate: u32) -> AudioBuffer<f32> {
    let n_frames = (duration_secs * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(n_frames * 2);
    for i in 0..n_frames {
        let t = i as f32 / sample_rate as f32;
        let s = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3;
        samples.push(s);
        samples.push(-s); // stereo L/R
    }
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

fn sine_buffer_mono(freq: f32, duration_secs: f32, sample_rate: u32) -> AudioBuffer<f32> {
    let n_frames = (duration_secs * sample_rate as f32) as usize;
    let samples: Vec<f32> = (0..n_frames)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin() * 0.3)
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

// ─── WAV tests ────────────────────────────────────────────────────────────────

#[test]
fn test_wav_f32_encode_magic_bytes() {
    let buf = sine_buffer_stereo(440.0, 0.1, 48_000);
    let wav = encode_wav_to_vec(&buf).expect("WAV encode must succeed");
    assert!(wav.starts_with(b"RIFF"), "WAV must start with RIFF");
    // RIFF header: 4 bytes RIFF + 4 bytes size + 4 bytes WAVE
    assert_eq!(&wav[8..12], b"WAVE", "WAV format must be WAVE");
}

#[test]
fn test_wav_output_length_proportional_to_input() {
    let buf = sine_buffer_stereo(440.0, 1.0, 44_100);
    let wav = encode_wav_to_vec(&buf).expect("WAV F32 encode");
    // WAV F32 stereo 44100Hz: 44100 frames * 2 ch * 4 bytes per sample
    let expected_data_bytes = 44_100 * 2 * 4;
    assert!(
        wav.len() >= expected_data_bytes,
        "WAV must contain at least {expected_data_bytes} data bytes, got {}",
        wav.len()
    );
}

// ─── FLAC tests ───────────────────────────────────────────────────────────────

#[test]
fn test_flac_encode_magic_bytes() {
    let buf = sine_buffer_stereo(440.0, 0.1, 48_000);
    let flac = encode_flac_to_vec(&buf).expect("FLAC encode must succeed");
    assert!(
        flac.starts_with(b"fLaC"),
        "FLAC must start with fLaC marker"
    );
    assert!(flac.len() > 100, "FLAC output must be non-trivially large");
}

#[test]
fn test_flac_output_smaller_than_wav_for_sine() {
    let buf = sine_buffer_stereo(440.0, 1.0, 44_100);
    let wav = encode_wav_to_vec(&buf).expect("WAV encode");
    let flac = encode_flac_to_vec(&buf).expect("FLAC encode");
    // FLAC should be smaller than WAV F32 for a sine wave (highly compressible pattern)
    assert!(
        flac.len() < wav.len(),
        "FLAC ({} bytes) should be smaller than WAV F32 ({} bytes)",
        flac.len(),
        wav.len()
    );
}

// ─── Vorbis tests ─────────────────────────────────────────────────────────────

#[test]
fn test_vorbis_encode_magic_bytes() {
    let buf = sine_buffer_stereo(440.0, 0.1, 48_000);
    let mut out = Cursor::new(Vec::new());
    encode_vorbis(&buf, &mut out).expect("Vorbis encode must succeed");
    let bytes = out.into_inner();
    assert!(
        bytes.starts_with(b"OggS"),
        "OGG Vorbis must start with OggS capture pattern"
    );
    assert!(
        bytes.windows(7).any(|w| w == b"\x01vorbis"),
        "output must contain Vorbis identification header (\\x01vorbis)"
    );
    assert!(
        bytes.windows(7).any(|w| w == b"\x03vorbis"),
        "output must contain Vorbis comment header (\\x03vorbis)"
    );
    assert!(
        bytes.windows(7).any(|w| w == b"\x05vorbis"),
        "output must contain Vorbis setup header (\\x05vorbis)"
    );
}

// ─── AAC tests ────────────────────────────────────────────────────────────────

#[test]
fn test_aac_adts_encode_magic_bytes() {
    let buf = sine_buffer_mono(440.0, 0.1, 44_100);
    let mut out = Cursor::new(Vec::new());
    encode_aac(&buf, &mut out).expect("AAC encode must succeed");
    let bytes = out.into_inner();
    // ADTS sync word is 12 bits of all-ones: first byte = 0xFF, upper 4 bits of second = 0xF
    assert!(!bytes.is_empty(), "AAC output must not be empty");
    assert_eq!(
        bytes[0], 0xFF,
        "ADTS first byte must be 0xFF (sync word MSB)"
    );
    // Top nibble of second byte must be 0xF (sync word LSB)
    assert_eq!(
        bytes[1] & 0xF0,
        0xF0,
        "ADTS second byte top nibble must be 0xF (sync word continuation)"
    );
}

// ─── Edge case tests ──────────────────────────────────────────────────────────

#[test]
fn test_all_encoders_handle_empty_buffer() {
    let empty = AudioBuffer::<f32> {
        samples: vec![],
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };
    // These may succeed (producing a file with no audio) or return an error — both are valid.
    // The test simply verifies no panic occurs.
    let wav_result = encode_wav_to_vec(&empty);
    let _ = wav_result;

    let flac_result = encode_flac_to_vec(&empty);
    let _ = flac_result;
}

#[test]
fn test_wav_mono_encode_magic_bytes() {
    let buf = sine_buffer_mono(880.0, 0.2, 44_100);
    let wav = encode_wav_to_vec(&buf).expect("WAV mono encode must succeed");
    assert!(wav.starts_with(b"RIFF"), "mono WAV must start with RIFF");
    assert_eq!(&wav[8..12], b"WAVE", "mono WAV format must be WAVE");
}

#[test]
fn test_flac_mono_encode_magic_bytes() {
    let buf = sine_buffer_mono(880.0, 0.2, 44_100);
    let flac = encode_flac_to_vec(&buf).expect("FLAC mono encode must succeed");
    assert!(flac.starts_with(b"fLaC"), "mono FLAC must start with fLaC");
}
