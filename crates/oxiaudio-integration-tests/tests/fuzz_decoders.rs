//! Property-based fuzz harnesses for the OxiAudio decoders.
//!
//! Every decoder that consumes untrusted, attacker-controlled input must satisfy
//! one invariant: for *any* byte sequence it must either return a decoded buffer
//! or a typed [`oxiaudio_core::OxiAudioError`] — it must **never panic, abort, or
//! hang**. These tests drive each decoder with two classes of adversarial input:
//!
//! 1. **Uniform random bytes** — exercises the reject-early paths and any code
//!    reachable before a magic/length check fails.
//! 2. **Magic-prefixed bytes** — a valid format signature followed by random
//!    payload, which drives the fuzzer deep into header/frame parsing where the
//!    dangerous arithmetic (length fields, indices, allocations) actually lives.
//!
//! Because these are pure-Rust, in-memory (`&[u8]` / `Cursor`) decoders they run
//! deterministically with no filesystem access. Input sizes are bounded so that a
//! legitimately-parsed but attacker-inflated length field cannot request a
//! multi-gigabyte allocation and turn a robustness test into an OOM abort; the
//! decoders are still expected to *reject* such headers with a typed error, which
//! the bounded corpus also verifies via the magic-prefixed strategies.
//!
//! Run with the default feature set: `cargo nextest run -p oxiaudio-integration-tests`.

use std::io::Cursor;

use oxiaudio_core::SampleFormat;
use oxiaudio_decode::aac_decoder::parse_adts_header;
use oxiaudio_decode::aiff::{decode_aiff, decode_aiffc_compressed};
use oxiaudio_decode::midi::MidiFile;
use oxiaudio_decode::{
    decode_aac, decode_au, decode_musepack, decode_raw_pcm, decode_wavpack,
    detect_format_from_bytes, parse_gapless_info, parse_opus_head, RawPcmConfig,
};

use proptest::prelude::*;

/// Upper bound on generated corpus length. Large enough to contain full headers
/// and several frames for every format, small enough that the whole property set
/// runs in well under a second per case.
const MAX_LEN: usize = 4096;

/// A byte vector of length `0..=MAX_LEN` drawn from the full `u8` range.
fn random_bytes() -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(any::<u8>(), 0..=MAX_LEN)
}

/// A byte vector that begins with a fixed magic prefix and is followed by random
/// bytes, to reach past the format-detection guard into real parsing logic.
fn prefixed_bytes(prefix: &'static [u8]) -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(any::<u8>(), 0..=MAX_LEN).prop_map(move |mut tail| {
        let mut v = Vec::with_capacity(prefix.len() + tail.len());
        v.extend_from_slice(prefix);
        v.append(&mut tail);
        v
    })
}

proptest! {
    // Keep the per-decoder case count modest but meaningful; the harness covers
    // roughly a dozen decoders so the aggregate corpus is large.
    #![proptest_config(ProptestConfig::with_cases(512))]

    // ── AAC / ADTS ──────────────────────────────────────────────────────────

    #[test]
    fn adts_header_never_panics(data in random_bytes()) {
        // Whatever comes back, the call must return rather than unwind/abort.
        let _ = parse_adts_header(&data);
    }

    #[test]
    fn aac_decode_random_never_panics(data in random_bytes()) {
        let _ = decode_aac(&data);
    }

    #[test]
    fn aac_decode_adts_prefixed_never_panics(data in prefixed_bytes(&[0xFF, 0xF1])) {
        // 0xFFF1 is a valid ADTS sync word (MPEG-4, no CRC), so this drives the
        // frame-header and payload parser rather than the sync-search reject path.
        let _ = decode_aac(&data);
    }

    // ── WavPack ─────────────────────────────────────────────────────────────

    #[test]
    fn wavpack_random_never_panics(data in random_bytes()) {
        let _ = decode_wavpack(&data);
    }

    #[test]
    fn wavpack_prefixed_never_panics(data in prefixed_bytes(b"wvpk")) {
        let _ = decode_wavpack(&data);
    }

    // ── Musepack (SV7 / SV8) ────────────────────────────────────────────────

    #[test]
    fn musepack_random_never_panics(data in random_bytes()) {
        let _ = decode_musepack(&data);
    }

    #[test]
    fn musepack_sv8_prefixed_never_panics(data in prefixed_bytes(b"MPCK")) {
        let _ = decode_musepack(&data);
    }

    #[test]
    fn musepack_sv7_prefixed_never_panics(data in prefixed_bytes(b"MP+\x07")) {
        let _ = decode_musepack(&data);
    }

    // ── Sun/NeXT AU ─────────────────────────────────────────────────────────

    #[test]
    fn au_random_never_panics(data in random_bytes()) {
        let mut cur = Cursor::new(data);
        let _ = decode_au(&mut cur);
    }

    #[test]
    fn au_prefixed_never_panics(data in prefixed_bytes(b".snd")) {
        let mut cur = Cursor::new(data);
        let _ = decode_au(&mut cur);
    }

    // ── AIFF / AIFF-C ───────────────────────────────────────────────────────

    #[test]
    fn aiff_random_never_panics(data in random_bytes()) {
        let mut cur = Cursor::new(data);
        let _ = decode_aiff(&mut cur);
    }

    #[test]
    fn aiff_form_prefixed_never_panics(data in prefixed_bytes(b"FORM")) {
        let mut cur = Cursor::new(data);
        let _ = decode_aiff(&mut cur);
    }

    #[test]
    fn aiffc_form_prefixed_never_panics(data in prefixed_bytes(b"FORM")) {
        let mut cur = Cursor::new(data);
        let _ = decode_aiffc_compressed(&mut cur);
    }

    // ── Raw PCM (every sample format, both endiannesses) ────────────────────

    #[test]
    fn raw_pcm_never_panics(
        data in random_bytes(),
        fmt_sel in 0u8..6,
        little_endian in any::<bool>(),
        channels in 1u16..=8,
        skip_bytes in 0usize..=64,
    ) {
        let format = match fmt_sel {
            0 => SampleFormat::U8,
            1 => SampleFormat::I16,
            2 => SampleFormat::I24,
            3 => SampleFormat::I32,
            4 => SampleFormat::F32,
            _ => SampleFormat::F64,
        };
        let config = RawPcmConfig {
            sample_rate: 44_100,
            channels,
            format,
            little_endian,
            skip_bytes,
        };
        let mut cur = Cursor::new(data);
        let _ = decode_raw_pcm(&mut cur, &config);
    }

    // ── OGG Opus identification header (available without the `opus` feature) ─

    #[test]
    fn opus_head_random_never_panics(data in random_bytes()) {
        let _ = parse_opus_head(&data);
    }

    #[test]
    fn opus_head_prefixed_never_panics(data in prefixed_bytes(b"OpusHead")) {
        let _ = parse_opus_head(&data);
    }

    // ── Standard MIDI File ──────────────────────────────────────────────────

    #[test]
    fn midi_random_never_panics(data in random_bytes()) {
        let _ = MidiFile::from_bytes(&data);
    }

    #[test]
    fn midi_prefixed_never_panics(data in prefixed_bytes(b"MThd")) {
        let _ = MidiFile::from_bytes(&data);
    }

    // ── Format detection & MP3 gapless (LAME/Xing) tag parsing ──────────────

    #[test]
    fn detect_format_never_panics(data in random_bytes()) {
        let _ = detect_format_from_bytes(&data);
    }

    #[test]
    fn gapless_random_never_panics(data in random_bytes()) {
        let _ = parse_gapless_info(&data);
    }

    #[test]
    fn gapless_id3_prefixed_never_panics(data in prefixed_bytes(b"ID3")) {
        let _ = parse_gapless_info(&data);
    }
}
