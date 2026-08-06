//! The RFC 6716–conformant Opus encoding entry points, and how they differ.
//!
//! `oxiaudio-encode` ships more than one Opus encoder, and picking the wrong
//! one is exactly the kind of ship-accident this example exists to prevent:
//!
//! - [`encode_opus_auto`] / [`encode_opus`] (an alias for the former) — the
//!   recommended default. Every 20 ms frame is routed through
//!   [`select_conformant_mode`] to the matching conformant per-frame encoder.
//!   Output decodes cleanly on a standard Opus decoder.
//! - [`encode_opus_conformant`] — the same conformant per-frame encoders, but
//!   with the mode ([`OpusConformantMode::Celt`] / `Silk` / `Hybrid`) chosen
//!   explicitly by the caller instead of automatically per frame.
//! - `encode_opus_structural` (not called here) — the pre-0.2.1 byte layout,
//!   kept only for byte-compatibility. It is **not** conformant: standard
//!   decoders reject its frames. New code should not use it.
//!
//! None of these are transparent-quality encoders yet — see the `opus_celt` /
//! `opus_silk_encode` module docs for the exact, measured fidelity caveats
//! (coarse CELT SNR gate; SILK is unvoiced-only with best-lag correlation
//! ≈ 0.33–0.67 against reference decode). What they guarantee is
//! *decodability*: a standard Opus decoder accepts every packet these
//! functions emit.
//!
//! Run with:
//! ```text
//! cargo run -p oxiaudio-encode --example opus_conformant
//! ```

use std::f32::consts::PI;
use std::path::PathBuf;

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use oxiaudio_encode::{encode_opus_auto_file, encode_opus_conformant_file, OpusConformantMode};

/// A 1-second, 48 kHz mono 440 Hz tone at -6 dBFS. Opus operates on 48 kHz
/// input only (no internal resampling), which is why the buffer is built at
/// that rate directly instead of using a more common rate like 44.1 kHz.
fn synth_tone_48k(seconds: f32) -> AudioBuffer<f32> {
    let sample_rate = 48_000u32;
    let n_frames = (seconds * sample_rate as f32) as usize;
    let samples = (0..n_frames)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * 440.0 * t).sin() * 0.5
        })
        .collect();
    AudioBuffer {
        samples,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tone = synth_tone_48k(1.0);
    let tmp_dir = std::env::temp_dir();

    // 1. Recommended default: automatic per-frame CELT/SILK mode selection.
    let auto_path: PathBuf = tmp_dir.join("oxiaudio_example_opus_auto.ogg");
    encode_opus_auto_file(&tone, &auto_path, 64)?;
    println!(
        "encode_opus_auto  -> {} ({} bytes)",
        auto_path.display(),
        std::fs::metadata(&auto_path)?.len()
    );

    // 2. Explicit mode selection: force CELT (real MDCT + PVQ spectral content;
    //    the only conformant mode that carries actual signal rather than an
    //    inactive/silent SILK frame).
    let celt_path: PathBuf = tmp_dir.join("oxiaudio_example_opus_celt.ogg");
    encode_opus_conformant_file(&tone, &celt_path, OpusConformantMode::Celt)?;
    println!(
        "encode_opus_conformant(Celt)   -> {} ({} bytes)",
        celt_path.display(),
        std::fs::metadata(&celt_path)?.len()
    );

    // 3. Explicit mode selection: force SILK (genuine analysis-by-synthesis
    //    narrowband encoder — not silence — but unvoiced-only; see the module
    //    docs cited above for the measured correlation caveat).
    let silk_path: PathBuf = tmp_dir.join("oxiaudio_example_opus_silk.ogg");
    encode_opus_conformant_file(&tone, &silk_path, OpusConformantMode::Silk)?;
    println!(
        "encode_opus_conformant(Silk)   -> {} ({} bytes)",
        silk_path.display(),
        std::fs::metadata(&silk_path)?.len()
    );

    // Every file above starts with the OGG "OggS" magic and is accepted by a
    // standard Opus decoder — see crates/oxiaudio-encode/tests/
    // m_opus_conformant_optin.rs for round-trip verification against the
    // `opus-decoder` reference crate.
    for path in [&auto_path, &celt_path, &silk_path] {
        let bytes = std::fs::read(path)?;
        assert_eq!(
            &bytes[..4],
            b"OggS",
            "{} must start with OggS magic",
            path.display()
        );
    }
    println!("all three packets carry valid OGG framing");

    Ok(())
}
