//! RFC 6716 CELT conformance tests for `encode_celt_frame_conformant`.
//!
//! These tests verify that `encode_celt_frame_conformant` produces Opus packets
//! that are decodable by the reference `opus-decoder` crate (BSD-3-Clause port
//! of libopus).
//!
//! # Test matrix
//!
//! | Test                           | Input          | Checks                                  |
//! |--------------------------------|----------------|-----------------------------------------|
//! | `celt_encode_decodable_mono`   | 440 Hz sine    | TOC=0xF8, n=960, energy>1e-6            |
//! | `celt_silence_packet_decodable`| All-zero PCM   | TOC=0xF8, n=960 (no energy floor check) |

use opus_decoder::OpusDecoder;
use oxiaudio_encode::encode_celt_frame_conformant;

/// Generate 960 mono samples of a 440 Hz sine wave at amplitude 0.5, 48 kHz.
fn sine_440hz() -> Vec<f32> {
    (0..960)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
        .collect()
}

/// Decode an Opus packet with a freshly initialised mono 48 kHz decoder.
///
/// Returns the decoded per-channel sample count and the decoded PCM buffer.
fn decode_packet(packet: &[u8]) -> Result<(usize, Vec<f32>), opus_decoder::OpusError> {
    let mut dec = OpusDecoder::new(48_000, 1)?;
    let mut pcm = vec![0.0f32; 960];
    let n = dec.decode_float(packet, &mut pcm, false)?;
    Ok((n, pcm))
}

/// RMS energy of a PCM slice.
fn rms_energy(pcm: &[f32]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    pcm.iter().map(|&x| x * x).sum::<f32>() / pcm.len() as f32
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// A 440 Hz mono sine frame encoded with `encode_celt_frame_conformant` must
/// produce a packet with the correct TOC byte and decode to 960 non-silent
/// samples without error.
#[test]
fn celt_encode_decodable_mono_20ms() {
    let pcm = sine_440hz();
    let packet = encode_celt_frame_conformant(&pcm, 1);

    // The packet must be non-empty (at minimum: 1 TOC + flush bytes).
    assert!(
        packet.len() >= 2,
        "packet must contain TOC byte plus at least one frame byte, got {} bytes",
        packet.len()
    );

    // TOC byte: config 31 (CELT-only, fullband, 20 ms), mono, 1 frame = 0xF8.
    assert_eq!(
        packet[0], 0xF8,
        "TOC byte must be 0xF8 (CELT-only FB 20 ms mono), got {:#04x}",
        packet[0]
    );

    // Decode and verify sample count.
    let (n, pcm_out) = decode_packet(&packet).expect("decode_float must succeed");
    assert_eq!(
        n, 960,
        "decoded sample count must be 960 (20 ms @ 48 kHz), got {n}"
    );

    // Decoded PCM must have non-trivial energy (random LCG fill from the decoder
    // scales by 2^E_MEANS[i] ≈ 16–64 per band, which far exceeds 1e-6 RMS²).
    let energy = rms_energy(&pcm_out);
    assert!(
        energy > 1e-6,
        "decoded RMS² energy must be > 1e-6, got {energy:.2e}"
    );
}

/// A 440 Hz tone encoded with the full MDCT+PVQ `encode_celt_frame_conformant`
/// must decode to a signal whose normalized cross-correlation with the input
/// exceeds 0.1 at the best lag, confirming actual 440 Hz spectral content is
/// preserved (not silence).
///
/// Note: because the decoder reconstructs audio from quantized band energies and
/// PVQ pulse shapes, perfect reconstruction is not expected — corr > 0.1 is a
/// minimal sanity gate that the encoded signal is not silence/noise. The
/// correlation is measured over a lag search window because CELT's MDCT
/// overlap-add plus the decoder's algorithmic delay shift the reconstructed
/// tone by several tens of samples; a zero-lag correlation would be dominated
/// by that phase offset rather than by spectral fidelity. (The current greedy,
/// non-RDO PVQ shape selection limits how high this correlation can climb — see
/// the conformance note in `opus_celt.rs` — so the gate is deliberately loose.)
#[test]
fn celt_encode_snr_gate_440hz() {
    let pcm = sine_440hz();
    let packet = encode_celt_frame_conformant(&pcm, 1);
    let (n, pcm_out) = decode_packet(&packet).expect("decode must succeed");
    assert_eq!(n, 960, "decoded sample count must be 960, got {n}");

    // Normalised cross-correlation over a lag window that covers the CELT/MDCT
    // reconstruction delay (up to half a frame). Measures spectral overlap
    // regardless of gain or phase alignment.
    let norm_in = pcm.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_out = pcm_out.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let denom = (norm_in * norm_out).max(1e-12);
    let mut best_corr = f32::MIN;
    let mut best_lag = 0i32;
    for lag in -480..=480i32 {
        let mut dot = 0.0f32;
        for i in 0..960i32 {
            let j = i + lag;
            if (0..960).contains(&j) {
                dot += pcm[i as usize] * pcm_out[j as usize];
            }
        }
        let corr = dot / denom;
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    assert!(
        best_corr > 0.1,
        "SNR gate: best-lag normalised correlation {best_corr:.4} (at lag {best_lag}) \
         must exceed 0.1 (expected non-silent output preserving 440 Hz content)"
    );
}

/// Best-lag normalised cross-correlation between input and decoded output over a
/// ±half-frame lag window (covers the CELT/MDCT reconstruction delay).
fn best_lag_corr(inp: &[f32], out: &[f32]) -> f32 {
    let norm_in = inp.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_out = out.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let denom = (norm_in * norm_out).max(1e-12);
    let mut best = f32::MIN;
    for lag in -480..=480i32 {
        let mut dot = 0.0f32;
        for i in 0..960i32 {
            let j = i + lag;
            if (0..960).contains(&j) {
                dot += inp[i as usize] * out[j as usize];
            }
        }
        best = best.max(dot / denom);
    }
    best
}

/// Fidelity gate for the exact-PVQ (`op_pvq_search` + `exp_rotation`) shape
/// coding. These floors are set from measured best-lag correlations and are
/// chosen to sit **above** what the previous greedy shape allocator achieved on
/// the same signals (greedy: sine440 ≈ 0.12, two-tone ≈ 0.20; exact RDO:
/// sine440 ≈ 0.17, two-tone ≈ 0.27). Passing them therefore verifies that the
/// rate-distortion-optimal shape search — not merely a decodable bitstream — is
/// in effect.
///
/// The floors are deliberately below 1.0 because this file measures a
/// *single, history-free* frame: `encode_celt_frame_conformant` assumes a zero
/// overlap buffer, so time-domain alias cancellation is incomplete by
/// construction. The stream-level quality gates (per-band level accuracy, fixed
/// -delay SNR, level tracking) live in `m_opus_celt_snr.rs`, which encodes with
/// real MDCT history. Pure high-frequency tones fall in bands the neutral
/// (non-dynalloc) allocation leaves pulse-starved, so they are intentionally
/// *not* gated here — that is an allocation limitation documented in
/// `opus_celt.rs`, not a shape-coding one.
#[test]
fn celt_encode_exact_pvq_fidelity_gate() {
    // Representative signals whose dominant energy lands in bands the neutral
    // allocation actually funds with pulses (low/mid frequencies).
    let sine440 = sine_440hz();
    let two_tone: Vec<f32> = (0..960)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 900.0 * t).sin()
        })
        .collect();

    let cases: &[(&str, &[f32], f32)] = &[
        ("sine440", &sine440, 0.15),
        ("two_tone_300_900", &two_tone, 0.22),
    ];

    for &(name, pcm, floor) in cases {
        let packet = encode_celt_frame_conformant(pcm, 1);
        let (n, out) = decode_packet(&packet).expect("decode must succeed");
        assert_eq!(n, 960, "{name}: decoded sample count must be 960, got {n}");
        let corr = best_lag_corr(pcm, &out);
        assert!(
            corr > floor,
            "{name}: exact-PVQ best-lag correlation {corr:.4} must exceed {floor:.2}"
        );
    }
}

/// A silence frame (all-zero PCM) must also produce a decodable packet with the
/// correct TOC byte and exactly 960 output samples.  No energy floor is imposed
/// on the decoded silence output.
#[test]
fn celt_silence_packet_decodable() {
    let pcm = vec![0.0f32; 960];
    let packet = encode_celt_frame_conformant(&pcm, 1);

    assert!(
        packet.len() >= 2,
        "silence packet must have at least 2 bytes, got {}",
        packet.len()
    );

    assert_eq!(
        packet[0], 0xF8,
        "TOC byte must be 0xF8, got {:#04x}",
        packet[0]
    );

    let (n, _pcm_out) = decode_packet(&packet).expect("silence decode must succeed");
    assert_eq!(n, 960, "silence packet must decode to 960 samples, got {n}");
}
