//! End-to-end verification of the full automatic Opus encode path
//! ([`encode_opus_auto`]) against the reference `opus-decoder`.
//!
//! These tests encode a matrix of signal types (tones, speech-like, music-like,
//! silence, noise) in mono and stereo at several target bitrates, demux the OGG
//! Opus stream, decode every audio packet with the workspace's pure Opus decoder,
//! and assert the stream is uncorrupted (every packet decodes to 960 finite
//! samples) plus bounded fidelity where the selected mode reconstructs signal.

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use oxiaudio_encode::{encode_opus_auto, select_conformant_mode, OpusConformantMode};

// ── OGG demux + decode helpers ──────────────────────────────────────────────

/// Minimal OGG (RFC 3533) demuxer returning audio packets (drops the two headers).
fn demux_ogg_audio_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut pos = 0usize;

    while pos + 27 <= data.len() {
        if &data[pos..pos + 4] != b"OggS" {
            pos += 1;
            continue;
        }
        let nsegs = data[pos + 26] as usize;
        let lacing_start = pos + 27;
        let lacing_end = lacing_start + nsegs;
        if lacing_end > data.len() {
            break;
        }
        let lacing = &data[lacing_start..lacing_end];
        let body_start = lacing_end;
        let body_len: usize = lacing.iter().map(|&l| l as usize).sum();
        let body_end = body_start + body_len;
        if body_end > data.len() {
            break;
        }
        let body = &data[body_start..body_end];
        let mut seg_off = 0usize;
        for &lace in lacing {
            let seg = &body[seg_off..seg_off + lace as usize];
            current.extend_from_slice(seg);
            seg_off += lace as usize;
            if lace < 255 {
                packets.push(std::mem::take(&mut current));
            }
        }
        pos = body_end;
    }
    packets
        .into_iter()
        .filter(|p| !p.starts_with(b"OpusHead") && !p.starts_with(b"OpusTags"))
        .collect()
}

fn decode_packet(packet: &[u8]) -> (usize, Vec<f32>) {
    let mut dec = opus_decoder::OpusDecoder::new(48_000, 1).expect("decoder init");
    let mut pcm = vec![0.0f32; 960];
    let n = dec
        .decode_float(packet, &mut pcm, false)
        .expect("decode_float must succeed on a conformant packet");
    (n, pcm)
}

/// Best-lag normalised cross-correlation over a small delay window (accounts for
/// SILK resampler / CELT MDCT group delay).
fn best_lag_correlation(input: &[f32], output: &[f32]) -> f64 {
    (0..320)
        .map(|d| {
            let mut num = 0.0f64;
            let mut a = 0.0f64;
            let mut b = 0.0f64;
            for i in d..output.len().min(input.len() + d) {
                if i - d >= input.len() {
                    break;
                }
                let x = input[i - d] as f64;
                let y = output[i] as f64;
                num += x * y;
                a += x * x;
                b += y * y;
            }
            if a > 0.0 && b > 0.0 {
                num / (a * b).sqrt()
            } else {
                0.0
            }
        })
        .fold(f64::MIN, f64::max)
}

// ── Signal generators (mono, 48 kHz, `frames` × 960 samples) ────────────────

fn sine(freq: f32, frames: usize, amp: f32) -> Vec<f32> {
    (0..frames * 960)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 48_000.0).sin() * amp)
        .collect()
}

/// Two-tone "music"-like signal (300 + 900 Hz). Low-frequency dominant, so at low
/// bitrates the router prefers SILK; at higher bitrates it goes to CELT.
fn music(frames: usize) -> Vec<f32> {
    (0..frames * 960)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 900.0 * t).sin()
        })
        .collect()
}

/// Speech-like signal: a 160 Hz glottal fundamental with two low formants.
fn speech(frames: usize) -> Vec<f32> {
    (0..frames * 960)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            0.4 * (2.0 * std::f32::consts::PI * 160.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 700.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 1200.0 * t).sin()
        })
        .collect()
}

/// Deterministic pseudo-random broadband noise (LCG), amplitude ~0.3.
fn noise(frames: usize) -> Vec<f32> {
    let mut s: u32 = 0x1234_5678;
    (0..frames * 960)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.6
        })
        .collect()
}

fn mono_buf(samples: Vec<f32>) -> AudioBuffer<f32> {
    AudioBuffer {
        samples,
        sample_rate: 48_000,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

fn stereo_buf(mono: &[f32]) -> AudioBuffer<f32> {
    // Interleave with a slightly attenuated right channel to exercise downmix.
    let mut s = Vec::with_capacity(mono.len() * 2);
    for &x in mono {
        s.push(x);
        s.push(x * 0.8);
    }
    AudioBuffer {
        samples: s,
        sample_rate: 48_000,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    }
}

/// Encode one buffer and return the per-packet best-lag correlation against the
/// reference mono signal. `label` is used only in assertion diagnostics.
///
/// `ref_mono` is the reference mono signal (the downmix the encoder sees).
fn roundtrip(label: &str, buf: &AudioBuffer<f32>, ref_mono: &[f32], bitrate: u32) -> Vec<f64> {
    let mut cur = std::io::Cursor::new(Vec::new());
    encode_opus_auto(buf, &mut cur, bitrate).expect("encode_opus_auto");
    let bytes = cur.into_inner();
    assert_eq!(&bytes[..4], b"OggS", "stream must start with OggS magic");

    let packets = demux_ogg_audio_packets(&bytes);
    let mut corrs = Vec::new();
    for (fi, packet) in packets.iter().enumerate() {
        let (n, pcm) = decode_packet(packet);
        assert_eq!(
            n, 960,
            "[{label} @ {bitrate}k frame {fi}] every packet must decode to 960 samples"
        );
        assert!(
            pcm.iter().all(|x| x.is_finite()),
            "[{label} @ {bitrate}k frame {fi}] decoded samples must all be finite (no corruption)"
        );
        let start = fi * 960;
        let frame_ref = &ref_mono[start..start + 960];
        corrs.push(best_lag_correlation(frame_ref, &pcm));
    }
    corrs
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// The full matrix must never corrupt the stream: every packet across every
/// signal / bitrate / channel-count decodes to 960 finite samples.
#[test]
fn full_path_matrix_no_corruption() {
    let signals: Vec<(&str, Vec<f32>)> = vec![
        ("sine_300", sine(300.0, 3, 0.5)),
        ("sine_1000", sine(1000.0, 3, 0.5)),
        ("sine_2500", sine(2500.0, 3, 0.5)),
        ("sine_6000", sine(6000.0, 3, 0.5)),
        ("music", music(3)),
        ("speech", speech(3)),
        ("noise", noise(3)),
        ("silence", vec![0.0f32; 3 * 960]),
    ];
    let bitrates = [8u32, 16, 32, 64, 128];

    for (name, mono) in &signals {
        for &br in &bitrates {
            // Mono.
            let mbuf = mono_buf(mono.clone());
            let _ = roundtrip(name, &mbuf, mono, br);
            // Stereo (downmix of L + 0.8·L = 0.9·L, so reference is scaled mono).
            let sbuf = stereo_buf(mono);
            let ref_mono: Vec<f32> = mono.iter().map(|&x| 0.9 * x).collect();
            let _ = roundtrip(name, &sbuf, &ref_mono, br);
        }
    }
}

/// Mode selection must be sane: silence and low-freq/low-bitrate → SILK; music,
/// high tones, and high bitrates → CELT.
#[test]
fn mode_selection_is_sane() {
    let silence = vec![0.0f32; 960];
    assert_eq!(
        select_conformant_mode(&silence, 64),
        OpusConformantMode::Silk,
        "silence routes to SILK"
    );

    let low_tone = sine(300.0, 1, 0.5);
    assert_eq!(
        select_conformant_mode(&low_tone, 12),
        OpusConformantMode::Silk,
        "low-freq low-bitrate routes to SILK"
    );
    assert_eq!(
        select_conformant_mode(&low_tone, 128),
        OpusConformantMode::Celt,
        "high bitrate routes to CELT even for low-freq content"
    );

    let hi_tone = sine(6000.0, 1, 0.5);
    assert_eq!(
        select_conformant_mode(&hi_tone, 12),
        OpusConformantMode::Celt,
        "high-freq content routes to CELT regardless of bitrate"
    );

    // Low-freq two-tone content: SILK at low bitrate, CELT once the bitrate rises
    // above the SILK ceiling.
    let mus = music(1);
    assert_eq!(
        select_conformant_mode(&mus, 12),
        OpusConformantMode::Silk,
        "low-freq music at low bitrate routes to SILK"
    );
    assert_eq!(
        select_conformant_mode(&mus, 64),
        OpusConformantMode::Celt,
        "music at higher bitrate routes to CELT"
    );
}

/// A low-frequency tone at a low bitrate is routed to SILK and must reconstruct
/// with real, positively-correlated signal (not silence).
#[test]
fn silk_route_reconstructs_low_tone() {
    let mono = sine(300.0, 2, 0.5);
    // Confirm the router picks SILK for the first frame.
    assert_eq!(
        select_conformant_mode(&mono[..960], 12),
        OpusConformantMode::Silk
    );
    let buf = mono_buf(mono.clone());
    let corrs = roundtrip("sine_300", &buf, &mono, 12);
    let best = corrs.iter().cloned().fold(f64::MIN, f64::max);
    assert!(
        best > 0.3,
        "SILK-routed 300 Hz tone must correlate with input (best = {best:.3})"
    );
}

/// A music-like signal is routed to CELT and must reconstruct real, correlated
/// signal (validates the PVQ shape path end-to-end through the stream).
#[test]
fn celt_route_reconstructs_music() {
    let mono = music(2);
    assert_eq!(
        select_conformant_mode(&mono[..960], 64),
        OpusConformantMode::Celt
    );
    let buf = mono_buf(mono.clone());
    let corrs = roundtrip("music", &buf, &mono, 64);
    let best = corrs.iter().cloned().fold(f64::MIN, f64::max);
    assert!(
        best > 0.2,
        "CELT-routed music must correlate with input (best = {best:.3})"
    );
}

/// Silence must decode to (near-)silence: bounded output energy, no corruption.
#[test]
fn silence_decodes_quiet() {
    let mono = vec![0.0f32; 2 * 960];
    let buf = mono_buf(mono.clone());
    let mut cur = std::io::Cursor::new(Vec::new());
    encode_opus_auto(&buf, &mut cur, 32).expect("encode");
    let bytes = cur.into_inner();
    for packet in demux_ogg_audio_packets(&bytes) {
        let (n, pcm) = decode_packet(&packet);
        assert_eq!(n, 960);
        let energy: f32 = pcm.iter().map(|&x| x * x).sum::<f32>() / 960.0;
        assert!(
            energy < 1e-3,
            "silence input must decode to low energy, got {energy}"
        );
    }
}

/// Stereo input downmixes and encodes without corruption across bitrates.
#[test]
fn stereo_roundtrip_no_corruption() {
    let mono = music(2);
    let buf = stereo_buf(&mono);
    for &br in &[16u32, 64, 128] {
        let mut cur = std::io::Cursor::new(Vec::new());
        encode_opus_auto(&buf, &mut cur, br).expect("encode stereo");
        let bytes = cur.into_inner();
        assert!(bytes.windows(8).any(|w| w == b"OpusHead"));
        for packet in demux_ogg_audio_packets(&bytes) {
            let (n, pcm) = decode_packet(&packet);
            assert_eq!(n, 960);
            assert!(pcm.iter().all(|x| x.is_finite()));
        }
    }
}
