//! RFC 6716 hybrid-mode conformance tests for `encode_hybrid_frame_conformant`.
//!
//! These tests verify that `encode_hybrid_frame_conformant` produces an Opus
//! hybrid-mode packet (config 12–15) with valid structural framing, then attempt
//! to decode it via the `opus-decoder` crate.
//!
//! # Test matrix
//!
//! | Test                              | Checks                                          |
//! |-----------------------------------|-------------------------------------------------|
//! | `hybrid_toc_is_hybrid_mode`       | TOC config is in range 12–15 (hybrid)           |
//! | `hybrid_packet_structural_validity`| Packet length ≥ implied SILK layer size         |
//! | `hybrid_decode_attempt`           | Decode succeeds (finite 960 samples) or Err OK  |

use opus_decoder::OpusDecoder;
use oxiaudio_encode::encode_hybrid_frame_conformant;

/// 960 mono samples of 440 Hz sine at amplitude 0.3, 48 kHz (20 ms).
fn sine_440hz_48k() -> Vec<f32> {
    (0..960)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.3)
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// The TOC `config` field (bits 7–3) must fall in the hybrid range 12–15.
///
/// Per RFC 6716 Table 2:
///    0..=11  SILK-only NB/MB/WB
///   12..=15  Hybrid SWB/FB  ← expected
///   16..=31  CELT-only
#[test]
fn hybrid_toc_is_hybrid_mode() {
    let pcm = sine_440hz_48k();
    let packet = encode_hybrid_frame_conformant(&pcm, 1);

    assert!(!packet.is_empty(), "hybrid packet must not be empty");

    let config = (packet[0] >> 3) & 0x1F;
    assert!(
        (12..=15).contains(&config),
        "TOC config {config} must be in hybrid range 12–15"
    );

    // Verify the exact config value: 15 = Hybrid Fullband 20 ms.
    assert_eq!(
        config, 15,
        "config must be 15 (Hybrid FB 20 ms), got {config}"
    );

    // Code field (bits 1–0) must be 0 for single-frame CBR.
    let code = packet[0] & 0x03;
    assert_eq!(
        code, 0,
        "frame count code must be 0 (single frame), got {code}"
    );
}

/// Structural validity: the hybrid packet must contain a TOC byte followed by a
/// non-trivial range-coded payload.
///
/// In this implementation SILK and CELT share a single range-coder stream
/// (no separate LP-size VLC field), matching the decoder's `EcDec::new(frame)`
/// construction on the full payload.  We verify:
/// - Packet has at least 4 bytes (TOC + ≥3 payload bytes).
/// - TOC byte matches config 15 (Hybrid FB 20 ms, mono, code=0).
/// - Payload is not all-zeros (the range coder wrote meaningful bits).
#[test]
fn hybrid_packet_structural_validity() {
    let pcm = sine_440hz_48k();
    let packet = encode_hybrid_frame_conformant(&pcm, 1);

    assert!(
        packet.len() >= 4,
        "hybrid packet must have at least 4 bytes (TOC + ≥3 payload), got {}",
        packet.len()
    );

    // TOC byte must be 0x78: config=15 (Hybrid FB 20 ms), stereo=0, code=0.
    assert_eq!(packet[0], 0x78, "TOC must be 0x78, got 0x{:02X}", packet[0]);

    // Payload (packet[1..]) must contain at least one non-zero byte —
    // the range coder always produces non-trivial output for a SILK frame.
    let has_nonzero = packet[1..].iter().any(|&b| b != 0);
    assert!(
        has_nonzero,
        "payload must contain at least one non-zero byte (range coder must write bits)"
    );
}

/// Decode the hybrid packet with `OpusDecoder` and assert `Ok(960)`.
///
/// The CELT hybrid layer now uses `start_band=17` and omits the silence flag,
/// matching what the hybrid CELT decoder expects after the SILK layer has been
/// read. The decoder returns `Ok(960)` with all-finite samples.
#[test]
fn hybrid_decode_attempt() {
    let pcm = sine_440hz_48k();
    let packet = encode_hybrid_frame_conformant(&pcm, 1);

    let mut dec = OpusDecoder::new(48_000, 1).expect("OpusDecoder::new must succeed");
    let mut out = vec![0.0f32; 960];

    let n = dec
        .decode_float(&packet, &mut out, false)
        .expect("hybrid decode must succeed with conformant CELT hybrid layer");
    assert_eq!(
        n, 960,
        "decoded sample count must be 960 (20 ms @ 48 kHz), got {n}"
    );
    assert!(
        out.iter().all(|x| x.is_finite()),
        "all decoded samples must be finite"
    );
}

/// Encode a 440 Hz sine and verify the hybrid decoder returns exactly 960 samples.
///
/// This test exercises the full encode → decode round-trip for the RFC 6716
/// hybrid mode and acts as the primary conformance gate for `encode_hybrid_frame_conformant`.
#[test]
fn hybrid_decode_succeeds_960() {
    let pcm: Vec<f32> = (0..960)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
        .collect();
    let packet = encode_hybrid_frame_conformant(&pcm, 1);
    let mut dec = OpusDecoder::new(48_000, 1).expect("decoder init");
    let mut out = vec![0.0f32; 960];
    let n = dec
        .decode_float(&packet, &mut out, false)
        .expect("hybrid decode must succeed");
    assert_eq!(n, 960, "hybrid decode must return 960 samples");
}
