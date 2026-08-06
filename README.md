# OxiAudio

Pure-Rust audio processing workspace: decode, encode, DSP effects, and spectral analysis.

**Version:** 0.2.1 | **MSRV:** 1.80 | **License:** Apache-2.0

## Format Support

| Format | Decode | Encode | Pure Rust | Feature flag |
|--------|--------|--------|-----------|--------------|
| WAV / RF64 | Yes | Yes | Yes | default |
| FLAC | Yes | Yes | Yes | default |
| AIFF / AIFF-C | Yes | Yes | Yes | default |
| AU / SND | Yes | Yes | Yes | default |
| MP3 (decode) | Yes | — | Yes (symphonia) | default |
| MP3 (encode) | — | Yes (opt-in) | No (LAME FFI) | via `oxiaudio-encode-mp3-lame` crate |
| OGG Vorbis | Yes | — | Yes (symphonia) | default |
| AAC / M4A | Yes | — | Yes (symphonia) | default |
| ALAC | Yes | — | Yes (symphonia) | default |
| Opus (decode) | Yes | — | Yes (opus-decoder) | default |
| Opus (encode) | — | Yes (CELT/SILK auto-select, RFC 6716 conformant — verified via final-range match, up to 510 kbps mono)\* | Yes | default |
| WavPack | Yes | — | Yes | default |
| Musepack (SV7/SV8) | Yes | — | Yes | default |
| MIDI (SMF 0/1/2) | Yes | — | Yes | default |

\* `encode_opus` (the default entry point) routes every 20 ms frame through
automatic CELT/SILK mode selection; every emitted packet decodes cleanly on a
standard-conformant decoder, but this is **not** a transparent-quality
encoder — CELT's stereo split-band angle coding and fine-energy refinement are
still simplified, and SILK (a genuine analysis-by-synthesis narrowband
encoder, not silence) is limited to unvoiced excitation with measured best-lag
correlation ≈ 0.33–0.67 against reference decode. See the `opus_encoder` /
`opus_celt` / `opus_silk_encode` module docs for exact, measured caveats. The
pre-0.2.1 non-conformant byte layout is preserved as `encode_opus_structural`
for byte-compatibility.

## DSP Features

Biquad EQ · Parametric EQ · Butterworth/Chebyshev/Elliptic/FIR filters ·
Compressor/Limiter/Gate/Expander/De-esser · Multiband compressor ·
Chorus/Flanger/Phaser/Tremolo/Vibrato · Delay · Freeverb + convolution reverb ·
Phase vocoder (pitch shift + time stretch) · Channel vocoder ·
YIN/pYIN pitch detection · Autocorrelation pitch tracker ·
Onset detection (spectral flux, HFC, complex domain) · Beat tracking ·
Spectral subtraction + Wiener filter noise reduction ·
EBU R128 / ITU-R BS.1770 loudness (LUFS + true peak) · ReplayGain ·
MFCC · Chromagram · Spectral centroid/flux/rolloff/flatness/contrast/tonnetz ·
STFT / iSTFT · Mel-spectrogram (via OxiFFT) ·
Kaiser and FlatTop window functions

## Multi-Channel Audio

- Surround layouts: Quad, 5.1, 7.1, 5.1-Side, Atmos 7.1.4
- `ChannelMap` / `ChannelId` with SMPTE/ITU-R BS.775 ordering
- Downmix (`5.1 → stereo`, `N-ch → mono`) and upmix utilities per ITU-R BS.775
- WAVE_FORMAT_EXTENSIBLE for >2 channel WAV output

## Advanced Encoding & Tagging

- RF64/BW64 WAV for audio files exceeding 4 GB
- FLAC album art via `METADATA_BLOCK_PICTURE` (`encode_flac_with_album_art`)
- ID3v2.4 writer with UTF-8, APIC album art, USLT lyrics, extended header CRC, ReplayGain
- APEv2 tag writer for WavPack/Musepack output
- Two-pass EBU R128 loudness normalization (−14/−16/−23 LUFS targets)
- Noise-shaped (ATH-weighted) dithering for perceptually optimal bit-depth reduction

## Pipeline Architecture

- `TranscodeStream` streaming transcode pipeline (decode → optional DSP → encode)
- `transcode_batch` parallel batch format conversion via rayon
- `DspChain` composable DSP effect builder
- `AudioRingBuffer<T>` lock-free SPSC ring buffer with wait-free overflow policy
- `AudioPipeline` with parallel branches, bypass/mute, and latency reporting
- `AudioClock` with drift_ppm, elapsed_frames, elapsed_secs

## Crate Layout

```
oxiaudio/                    (facade — default = ["pure"])
  oxiaudio-core              (AudioBuffer, traits, error, IPC, ring buffer, surround layouts)
  oxiaudio-decode            (SymphoniaDecoder + AIFF/AU/Opus/WavPack/Musepack/MIDI)
  oxiaudio-encode            (WAV/RF64, FLAC, AIFF, AU, ID3v2.4, APEv2; streaming + two-pass)
  oxiaudio-encode-mp3-lame   (LAME FFI adapter — opt-in, never default)
  oxiaudio-dsp               (resample, filters, dynamics, reverb, pitch, spectral, loudness)
```

## Pure Rust Policy

Default features carry zero C/C++/Fortran dependencies.
MP3 encoding (LAME FFI, LGPL) is the sole sanctioned FFI boundary and is opt-in via the `oxiaudio-encode-mp3-lame` quarantine crate — never part of the `oxiaudio` facade's default or `--all-features` closure.

## Quick Start

```rust,no_run
use std::path::Path;

// Decode any supported format
let buf = oxiaudio::decode_file(Path::new("input.flac")).expect("decode failed");
println!("{} frames @ {} Hz", buf.frame_count(), buf.sample_rate);

// DSP: normalize, then add reverb
let mut out = buf.clone();
oxiaudio::dsp::normalize(&mut out, -1.0);
let with_reverb = oxiaudio::dsp::reverb(&out, 0.6, 0.4, 0.3);

// Re-encode as FLAC
oxiaudio::encode_flac(&with_reverb, Path::new("output.flac")).expect("encode failed");
```

## Status

All M0–M23 milestones complete.

- **1,136 tests passing / 5 skipped** (default features) / **1,242 tests passing / 6 skipped**
  (`--all-features`), plus **63 doc tests** — 0 clippy warnings, 0 rustdoc warnings
  (`cargo nextest run --workspace` / `cargo test --doc --workspace --all-features` /
  `cargo clippy --all-targets --all-features -- -D warnings`, measured 2026-08-06)
- **39,706 lines of production Rust** (`tokei crates/*/src`, Code column) across 95 source files in the
  6 published crates (a 7th workspace member, `oxiaudio-integration-tests`, is a `publish = false`
  test-only harness with an empty `src/lib.rs` — it contributes 0 lines to this count)
- All major codecs, DSP algorithms, and tagging formats implemented
- Pure-Rust Opus encoder: `encode_opus` (default) auto-selects CELT/SILK per
  frame, carries lapped-MDCT overlap history across frames, and honours
  `target_bitrate_kbps`. Its CELT bitstream is **verified** RFC 6716 conformant,
  not merely assumed: the encoder's `final_range` register matches the reference
  decoder's for every supported frame size on a corpus including full-scale
  noise (the canonical libopus conformance check), and the crate ships its own
  RFC-structured decoder (`opus_range_dec`, `opus_celt_verify`) so the check runs
  in-tree. Measured reconstruction: per-band decoded level within a few dB of
  the input, overall level within ±3 dB.
  It is still **not** transparent-quality: CELT is mono-only (stereo is
  downmixed), non-transient and uses a neutral non-dynalloc allocation. The
  frame-size ceiling (`MAX_CELT_FRAME_BYTES`) is the RFC's own 1275-byte frame
  limit — 510 kbps mono — over which bit-exactness against the reference decoder
  is swept; requests above it are clamped rather than emitted. (Before 0.2.1 the
  ceiling was 80 bytes ≈ 32 kbps, because a pulse-cache row was mis-indexed for
  four-deep band splits, which only occur at higher rates.)
  SILK is a genuine analysis-by-synthesis narrowband encoder (not silence) but
  unvoiced-only with measured correlation ≈ 0.33–0.67.
  `encode_opus_conformant` exposes explicit CELT/SILK/Hybrid mode selection with
  the same caveats — **Hybrid's low band is still SILK silence**, so hybrid
  carries no low-frequency audio and is never auto-selected;
  `encode_opus_structural` preserves the pre-0.2.1 non-conformant byte layout
  for compatibility. See the Format Support table footnote and the
  `opus_encoder` module docs for the precise, measured fidelity picture.
- Pure Rust default features (LAME FFI is opt-in only)
