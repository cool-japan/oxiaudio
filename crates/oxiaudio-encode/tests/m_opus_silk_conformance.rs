//! RFC 6716 SILK conformance tests for `encode_silk_frame_conformant`.
//!
//! These tests verify that `encode_silk_frame_conformant` produces SILK-mode Opus
//! packets that are decodable by the reference `opus-decoder` crate (BSD-3-Clause
//! port of libopus).
//!
//! # Test matrix
//!
//! | Test                              | Input           | Checks                               |
//! |-----------------------------------|-----------------|--------------------------------------|
//! | `silk_encode_decodable_20ms`      | 1 kHz sine      | TOC=0x08, n=960, finite samples      |
//! | `silk_silence_decodable`          | All-zero PCM    | TOC=0x08, n=960                      |
//! | `silk_toc_is_silk_mode`           | Short sine      | config in 0..=11 (SILK/Hybrid range) |
//! | `silk_packet_has_minimal_length`  | Any             | len >= 3 (TOC + 2 payload bytes)     |

use opus_decoder::OpusDecoder;
use oxiaudio_encode::encode_silk_frame_conformant;

/// Generate 960 mono samples of a 1 kHz sine at amplitude 0.5, 48 kHz.
///
/// 960 samples @ 48 kHz = 20 ms.  Internally the SILK codec processes this as
/// 160 samples @ 8 kHz (NB narrowband, 4 × 5 ms subframes).
fn sine_1khz_20ms() -> Vec<f32> {
    (0..960)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin() * 0.5)
        .collect()
}

/// Decode an Opus packet at 48 kHz mono.
///
/// Returns `(n, pcm)` where `n` is the per-channel sample count and `pcm` is
/// the decoded 960-sample output buffer.
fn decode_packet(packet: &[u8]) -> Result<(usize, Vec<f32>), opus_decoder::OpusError> {
    let mut dec = OpusDecoder::new(48_000, 1)?;
    let mut pcm = vec![0.0f32; 960];
    let n = dec.decode_float(packet, &mut pcm, false)?;
    Ok((n, pcm))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// A 1 kHz mono sine frame encoded with `encode_silk_frame_conformant` must
/// produce a packet with TOC 0x08 (SILK NB 20 ms mono) and decode to exactly
/// 960 finite samples without error.
#[test]
fn silk_encode_decodable_20ms() {
    let pcm = sine_1khz_20ms();
    let packet = encode_silk_frame_conformant(&pcm, 1);

    // TOC byte must be 0x08: config 1 = SILK-only NB 20 ms, mono, 1-frame CBR.
    assert_eq!(
        packet[0], 0x08,
        "TOC byte must be 0x08 (NB 20 ms mono), got 0x{:02X}",
        packet[0]
    );

    let (n, out) = decode_packet(&packet).expect("SILK decode must succeed");

    assert_eq!(
        n, 960,
        "decoded sample count must be 960 (20 ms @ 48 kHz), got {n}"
    );

    assert!(
        out.iter().all(|&x| x.is_finite()),
        "all decoded samples must be finite"
    );
}

/// An all-zero PCM frame must also produce a valid, decodable SILK packet.
#[test]
fn silk_silence_decodable() {
    let pcm = vec![0.0f32; 960];
    let packet = encode_silk_frame_conformant(&pcm, 1);

    assert_eq!(
        packet[0], 0x08,
        "TOC byte must be 0x08 for silence, got 0x{:02X}",
        packet[0]
    );

    let (n, out) = decode_packet(&packet).expect("silence decode must succeed");

    assert_eq!(n, 960, "silence packet must decode to 960 samples, got {n}");

    assert!(
        out.iter().all(|&x| x.is_finite()),
        "decoded silence samples must be finite"
    );
}

/// TOC `config` field (bits 7–3) must fall in the SILK / Hybrid range 0..=11.
///
/// Per RFC 6716 Table 2:
///   0..=11  SILK-only NB/MB/WB
///  12..=15  Hybrid SWB/FB
///  16..=31  CELT-only NB/WB/SWB/FB
#[test]
fn silk_toc_is_silk_mode() {
    let pcm = sine_1khz_20ms();
    let packet = encode_silk_frame_conformant(&pcm, 1);

    assert!(!packet.is_empty(), "packet must be non-empty");

    let config = (packet[0] >> 3) & 0x1F;
    assert!(
        config <= 11,
        "TOC config {config} must be in SILK range 0..=11"
    );
}

/// The encoded packet must be non-trivial: at least TOC + 2 range-coder bytes.
#[test]
fn silk_packet_has_minimal_length() {
    let pcm = sine_1khz_20ms();
    let packet = encode_silk_frame_conformant(&pcm, 1);

    assert!(
        packet.len() >= 3,
        "packet must have at least 3 bytes (TOC + 2 payload), got {} bytes",
        packet.len()
    );
}

/// Encoding the same PCM twice must yield identical bytes (deterministic encoder).
#[test]
fn silk_encode_is_deterministic() {
    let pcm = sine_1khz_20ms();
    let p1 = encode_silk_frame_conformant(&pcm, 1);
    let p2 = encode_silk_frame_conformant(&pcm, 1);
    assert_eq!(
        p1, p2,
        "repeated encode of same PCM must produce identical packets"
    );
}
