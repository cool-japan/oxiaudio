//! RFC 6716–conformant OGG Opus encoder integration tests.
//!
//! These verify that [`encode_opus_conformant`] / [`encode_opus_conformant_file`]
//! produce structurally valid OGG Opus streams whose audio packets are accepted by
//! the reference `opus-decoder` crate. As of oxiaudio 0.2.1, [`encode_opus`] itself
//! (the default entry point) is also conformant — it delegates to
//! `encode_opus_auto`'s automatic per-frame mode selection — so this file also
//! verifies that [`encode_opus`]'s output round-trip decodes. The pre-0.2.1
//! non-conformant byte layout is preserved separately as `encode_opus_structural`
//! and pinned by its own TOC-byte test here.

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use oxiaudio_encode::{
    encode_opus, encode_opus_conformant, encode_opus_conformant_file, encode_opus_structural,
    OpusConformantMode,
};

/// Manual OGG demuxer (RFC 3533). Returns the audio packets in order, i.e. all
/// reconstructed packets except the `OpusHead` / `OpusTags` header packets.
fn demux_ogg_audio_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut pos = 0usize;

    while pos + 27 <= data.len() {
        // Find the next "OggS" capture pattern.
        if &data[pos..pos + 4] != b"OggS" {
            pos += 1;
            continue;
        }

        let nsegs = data[pos + 26] as usize;
        let lacing_start = pos + 27;
        let lacing_end = lacing_start + nsegs;
        if lacing_end > data.len() {
            break; // truncated header — stop cleanly.
        }
        let lacing = &data[lacing_start..lacing_end];

        let body_start = lacing_end;
        let body_len: usize = lacing.iter().map(|&l| l as usize).sum();
        let body_end = body_start + body_len;
        if body_end > data.len() {
            break; // truncated body — stop cleanly.
        }
        let body = &data[body_start..body_end];

        // Walk the lacing table, slicing the body into segments.
        let mut seg_off = 0usize;
        for &lace in lacing {
            let seg = body
                .get(seg_off..seg_off + lace as usize)
                .expect("lacing segment within page body");
            current.extend_from_slice(seg);
            seg_off += lace as usize;
            if lace < 255 {
                // Packet boundary: a lacing value < 255 ends the packet.
                packets.push(std::mem::take(&mut current));
            }
        }

        pos = body_end;
    }

    // Drop any dangling unterminated packet (none expected for well-formed input).
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
        .expect("decode_float");
    (n, pcm)
}

fn sine_440hz_mono(frames: usize) -> AudioBuffer<f32> {
    let n = frames * 960;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48_000.0).sin() * 0.5)
        .collect();
    AudioBuffer {
        samples,
        sample_rate: 48_000,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

fn silence_mono(frames: usize) -> AudioBuffer<f32> {
    AudioBuffer {
        samples: vec![0.0f32; frames * 960],
        sample_rate: 48_000,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

#[test]
fn conformant_celt_opus_roundtrip_decodes() {
    let buf = sine_440hz_mono(2);
    let mut cur = std::io::Cursor::new(Vec::new());
    encode_opus_conformant(&buf, &mut cur, OpusConformantMode::Celt).expect("encode celt");
    let bytes = cur.into_inner();

    assert_eq!(&bytes[..4], b"OggS", "must start with OggS magic");
    assert!(
        bytes.windows(8).any(|w| w == b"OpusHead"),
        "OpusHead magic must appear"
    );

    let audio = demux_ogg_audio_packets(&bytes);
    assert!(
        audio.len() >= 2,
        "expected >= 2 audio packets, got {}",
        audio.len()
    );

    for packet in &audio {
        let (n, pcm) = decode_packet(packet);
        assert_eq!(n, 960, "CELT packet must decode to 960 samples");
        assert!(
            pcm.iter().all(|x| x.is_finite()),
            "decoded samples must be finite"
        );
    }
}

/// This test feeds genuinely silent PCM, so it only checks decodability (960
/// finite samples) rather than signal correlation. SILK itself is a real
/// analysis-by-synthesis narrowband encoder (not silence-only) — see
/// `m_opus_silk_conformance.rs`'s `silk_tone_is_correlated_with_input` for the
/// non-silent-input correlation check.
#[test]
fn conformant_silk_opus_roundtrip_decodes() {
    let buf = silence_mono(1);
    let mut cur = std::io::Cursor::new(Vec::new());
    encode_opus_conformant(&buf, &mut cur, OpusConformantMode::Silk).expect("encode silk");
    let bytes = cur.into_inner();

    let audio = demux_ogg_audio_packets(&bytes);
    assert!(!audio.is_empty(), "expected at least one audio packet");

    for packet in &audio {
        let (n, pcm) = decode_packet(packet);
        assert_eq!(n, 960, "SILK packet must decode to 960 samples");
        assert!(
            pcm.iter().all(|x| x.is_finite()),
            "decoded samples must be finite"
        );
    }
}

#[test]
fn conformant_hybrid_opus_roundtrip_decodes() {
    let buf = sine_440hz_mono(1);
    let mut cur = std::io::Cursor::new(Vec::new());
    encode_opus_conformant(&buf, &mut cur, OpusConformantMode::Hybrid).expect("encode hybrid");
    let bytes = cur.into_inner();

    let audio = demux_ogg_audio_packets(&bytes);
    assert!(!audio.is_empty(), "expected at least one audio packet");

    for packet in &audio {
        let (n, pcm) = decode_packet(packet);
        assert_eq!(n, 960, "Hybrid packet must decode to 960 samples");
        assert!(
            pcm.iter().all(|x| x.is_finite()),
            "decoded samples must be finite"
        );
    }
}

#[test]
fn conformant_file_writes_ogg() {
    let buf = sine_440hz_mono(1);
    let path = std::env::temp_dir().join("oxiaudio_opus_conformant_test.ogg");
    encode_opus_conformant_file(&buf, &path, OpusConformantMode::Celt).expect("file");
    let bytes = std::fs::read(&path).expect("read");
    assert_eq!(&bytes[..4], b"OggS");
    let _ = std::fs::remove_file(&path);
}

/// Pins the pre-0.2.1 legacy (non-conformant) byte layout that
/// `encode_opus_structural` preserves verbatim. `encode_opus` itself was
/// rerouted to the conformant path in 0.2.1 — see
/// `encode_opus_default_is_now_conformant_and_decodes` below for that
/// invariant.
#[test]
fn encode_opus_structural_unchanged_structure() {
    let buf = silence_mono(1);
    let mut cur = std::io::Cursor::new(Vec::new());
    encode_opus_structural(&buf, &mut cur, 128).expect("encode_opus_structural mono silence");
    let bytes = cur.into_inner();

    assert_eq!(&bytes[..4], b"OggS", "must start with OggS magic");
    assert!(
        bytes.windows(8).any(|w| w == b"OpusHead"),
        "OpusHead magic must appear"
    );
    assert!(
        bytes.windows(8).any(|w| w == b"OpusTags"),
        "OpusTags magic must appear"
    );

    let audio = demux_ogg_audio_packets(&bytes);
    let first = audio.first().expect("at least one audio packet");
    let toc = *first.first().expect("audio packet has a TOC byte");
    assert_eq!(
        toc, 0xE0,
        "mono encode_opus_structural TOC must be 0xE0 ((28 << 3) | 0x00)"
    );
}

/// SHIP-ACCIDENT REGRESSION TEST (wave 1, S severity): prior to oxiaudio 0.2.1,
/// `encode_opus` — the default, most-discoverable Opus encoding entry point —
/// emitted a non-conformant 4-bit placeholder CELT payload that NO standard
/// Opus decoder would accept, while the README advertised "RFC 6716
/// conformant" with no caveat. `encode_opus` now delegates to the conformant
/// automatic-mode-selection path; this test proves its default output is
/// actually decodable by a standard-conformant decoder (not just structurally
/// valid OGG framing).
#[test]
fn encode_opus_default_is_now_conformant_and_decodes() {
    let buf = sine_440hz_mono(2);
    let mut cur = std::io::Cursor::new(Vec::new());
    encode_opus(&buf, &mut cur, 64).expect("encode_opus mono sine");
    let bytes = cur.into_inner();

    assert_eq!(&bytes[..4], b"OggS", "must start with OggS magic");
    assert!(
        bytes.windows(8).any(|w| w == b"OpusHead"),
        "OpusHead magic must appear"
    );

    let audio = demux_ogg_audio_packets(&bytes);
    assert!(
        audio.len() >= 2,
        "expected >= 2 audio packets, got {}",
        audio.len()
    );

    for packet in &audio {
        let (n, pcm) = decode_packet(packet);
        assert_eq!(
            n, 960,
            "encode_opus's default output must decode to 960 samples per packet \
             via a standard-conformant decoder"
        );
        assert!(
            pcm.iter().all(|x| x.is_finite()),
            "decoded samples must be finite"
        );
    }
}

/// Same ship-accident regression, exercised through stereo input (the other
/// branch `encode_opus` explicitly documents supporting) — decoded at the
/// MONO packet-channel count (per-packet TOC content is always mono; see
/// `encode_opus_default_stereo_input_plays_at_declared_channel_count` below
/// for the check that matters for playback: decoding at the `OpusHead`-declared
/// channel count).
#[test]
fn encode_opus_default_stereo_input_decodes() {
    let n = 2 * 960;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48_000.0).sin() * 0.5)
        .collect();
    let buf = AudioBuffer {
        samples,
        sample_rate: 48_000,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    };
    let mut cur = std::io::Cursor::new(Vec::new());
    encode_opus(&buf, &mut cur, 64).expect("encode_opus stereo sine");
    let bytes = cur.into_inner();

    let audio = demux_ogg_audio_packets(&bytes);
    assert!(!audio.is_empty(), "expected at least one audio packet");
    for packet in &audio {
        let (n, pcm) = decode_packet(packet);
        assert_eq!(
            n, 960,
            "stereo-input encode_opus packet must decode to 960 samples"
        );
        assert!(pcm.iter().all(|x| x.is_finite()));
    }
}

/// The critical playability check for stereo input: `encode_opus`'s `OpusHead`
/// advertises the INPUT channel count (2 for stereo), but every per-frame
/// packet is internally mono (downmixed, TOC S=0) — see `encode_opus_auto`'s
/// documented asymmetry. A real player initializes its decoder from
/// `OpusHead`, i.e. at 2 channels, NOT from each packet's own TOC. This test
/// proves that asymmetry is benign: a standard decoder created for the
/// `OpusHead`-declared channel count (2) still decodes the mono-flagged
/// packets cleanly (mono content duplicated to both channels internally, the
/// same behavior real Opus streams that switch mono/stereo packet-to-packet
/// rely on) rather than erroring out.
#[test]
fn encode_opus_default_stereo_input_plays_at_declared_channel_count() {
    let n = 2 * 960;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48_000.0).sin() * 0.5)
        .collect();
    let buf = AudioBuffer {
        samples,
        sample_rate: 48_000,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    };
    let mut cur = std::io::Cursor::new(Vec::new());
    encode_opus(&buf, &mut cur, 64).expect("encode_opus stereo sine");
    let bytes = cur.into_inner();

    let audio = demux_ogg_audio_packets(&bytes);
    assert!(!audio.is_empty(), "expected at least one audio packet");

    // Decode with a decoder initialized at the OpusHead-declared channel count
    // (2), exactly as a real player would, NOT at the per-packet mono count.
    let mut dec = opus_decoder::OpusDecoder::new(48_000, 2).expect("2ch decoder init");
    for packet in &audio {
        let mut pcm = vec![0.0f32; 960 * 2];
        let n = dec
            .decode_float(packet, &mut pcm, false)
            .expect("2-channel decoder must accept encode_opus's stereo-declared output");
        assert_eq!(
            n, 960,
            "must decode to 960 samples per channel at the declared channel count"
        );
        assert!(
            pcm.iter().all(|x| x.is_finite()),
            "decoded stereo samples must be finite"
        );
    }
}
