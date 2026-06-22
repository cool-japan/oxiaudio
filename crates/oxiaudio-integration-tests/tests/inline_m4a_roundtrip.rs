//! Extracted from oxiaudio-encode/src/aac_m4a.rs to break the encode↔decode dev-dependency cycle.
//!
//! These M4A container round-trip tests encode with `oxiaudio_encode` and decode (probe) with
//! `oxiaudio_decode`/Symphonia, so they live in the integration-tests crate that dev-depends on both.

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

fn silence_buf(sample_rate: u32, frames: usize) -> AudioBuffer<f32> {
    AudioBuffer {
        samples: vec![0.0f32; frames],
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

/// Verify the M4A output is a valid, Symphonia-parseable container.
///
/// This is a structural round-trip test: if `decode_reader` returns `Ok`, then:
/// - Symphonia successfully probed the container (ftyp/moov/mdat structure is valid)
/// - An audio track with valid codec params was found (stsd/mp4a/esds are valid)
/// - The AAC decoder was instantiated (AudioSpecificConfig in esds is correct)
/// - The stco offset pointed to valid packet data
///
/// We do not assert a specific sample count because ZERO_HCB silence frames may
/// decode to empty or zero output depending on the Symphonia AAC decoder version.
#[test]
fn test_m4a_symphonia_probe_decodes_mono() {
    use oxiaudio_decode::decode_reader;
    use std::io::Cursor;

    let buf = silence_buf(44_100, 4096);
    let mut out = Cursor::new(Vec::new());
    oxiaudio_encode::encode_m4a(&buf, &mut out).expect("M4A mono encode must succeed");
    let encoded = out.into_inner();

    let decoded = decode_reader(Cursor::new(encoded))
        .expect("Symphonia must successfully probe and decode the mono M4A container");
    assert_eq!(
        decoded.sample_rate, 44_100,
        "sample rate must round-trip through the M4A container"
    );
    assert_eq!(
        decoded.channels.channel_count(),
        1,
        "mono channel count must round-trip through the M4A container"
    );
}

/// Symphonia round-trip for stereo M4A at 48000 Hz.
#[test]
fn test_m4a_symphonia_probe_decodes_stereo() {
    use oxiaudio_decode::decode_reader;
    use std::io::Cursor;

    let buf = AudioBuffer {
        samples: vec![0.0f32; 4096 * 2],
        sample_rate: 48_000,
        channels: ChannelLayout::Stereo,
        format: SampleFormat::F32,
    };
    let mut out = Cursor::new(Vec::new());
    oxiaudio_encode::encode_m4a(&buf, &mut out).expect("M4A stereo encode must succeed");
    let encoded = out.into_inner();

    let decoded = decode_reader(Cursor::new(encoded))
        .expect("Symphonia must successfully probe and decode the stereo M4A container");
    assert_eq!(
        decoded.sample_rate, 48_000,
        "sample rate must round-trip for 48000 Hz stereo"
    );
    assert_eq!(
        decoded.channels.channel_count(),
        2,
        "stereo channel count must round-trip through the M4A container"
    );
}
