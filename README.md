# OxiAudio

Pure-Rust audio processing workspace: decode, encode, DSP effects, and spectral analysis.

**Version:** 0.1.2 | **MSRV:** 1.80 | **License:** Apache-2.0

## Format Support

| Format | Decode | Encode | Pure Rust | Feature flag |
|--------|--------|--------|-----------|--------------|
| WAV / RF64 | Yes | Yes | Yes | default |
| FLAC | Yes | Yes | Yes | default |
| AIFF / AIFF-C | Yes | Yes | Yes | default |
| AU / SND | Yes | Yes | Yes | default |
| MP3 (decode) | Yes | — | Yes (symphonia) | default |
| MP3 (encode) | — | Yes | No (LAME FFI) | mp3-encode-lame |
| OGG Vorbis | Yes | — | Yes (symphonia) | default |
| AAC / M4A | Yes | — | Yes (symphonia) | default |
| ALAC | Yes | — | Yes (symphonia) | default |
| Opus (decode) | Yes | — | Yes (opus-decoder) | default |
| Opus (encode) | — | Yes (CELT/SILK/Hybrid) | Yes | default |
| WavPack | Yes | — | Yes | default |
| Musepack (SV7/SV8) | Yes | — | Yes | default |
| MIDI (SMF 0/1/2) | Yes | — | Yes | default |

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
The `mp3-encode-lame` feature is the sole sanctioned FFI boundary (LGPL, opt-in).

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

All M0–M23 milestones complete; Opus CELT/SILK/Hybrid encoders conformant as of 2026-06-10 (v0.1.2).

- **1,133 tests passing**, 0 clippy warnings
- **41,033+ production SLoC** across 6 crates
- All major codecs, DSP algorithms, and tagging formats implemented
- Pure-Rust Opus encoder: CELT-only, SILK NB/WB, and Hybrid FB modes (RFC 6716 conformant)
- Pure Rust default features (LAME FFI is opt-in only)
