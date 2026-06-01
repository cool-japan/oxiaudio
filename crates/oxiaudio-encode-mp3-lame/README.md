# oxiaudio-encode-mp3-lame — LAME MP3 encoder adapter for OxiAudio

[![Crates.io](https://img.shields.io/crates/v/oxiaudio-encode-mp3-lame.svg)](https://crates.io/crates/oxiaudio-encode-mp3-lame)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`oxiaudio-encode-mp3-lame` is the **MP3 encode adapter** for OxiAudio. It wraps `mp3lame-encoder` / `mp3lame-sys` (which link against `libmp3lame`) behind an `AudioBuffer`-shaped API, adding CBR / VBR / ABR mode selection, a fluent builder, a chunk-at-a-time streaming encoder, a hand-rolled ID3v2.4 tag writer, and ReplayGain (Xing/LAME header + TXXX) support.

> **⚠️ BOUNDED_FFI — NOT Pure Rust, LGPL.**
> This crate is the single deliberate exception to OxiAudio's Pure-Rust policy. The MP3 encoder is **opt-in only**, gated behind the `mp3-encode-lame` feature, and is **never** part of the OxiAudio facade's default feature set. With the feature disabled, the crate compiles with no C dependency and exposes only `compute_replaygain_gain_approx`.
>
> - **Licensing:** Enabling `mp3-encode-lame` links `libmp3lame`, which is **LGPL-2.1+**. Static linking imposes LGPL relinking obligations on downstream binaries — review the LGPL terms before shipping.
> - **`unsafe`:** The Rust layer is `#![forbid(unsafe_code)]`. All C FFI (including any `unsafe set_len`) is fully isolated inside `mp3lame-encoder` / `mp3lame-sys`; callers of this crate never touch `unsafe`.

## Installation

```toml
[dependencies]
# ReplayGain helper only — Pure Rust, no LAME linkage:
oxiaudio-encode-mp3-lame = "0.1.0"

# Full LAME MP3 encoding (LGPL, FFI):
oxiaudio-encode-mp3-lame = { version = "0.1.0", features = ["mp3-encode-lame"] }
```

In practice you normally enable this through the OxiAudio facade:

```toml
oxiaudio = { version = "0.1.0", features = ["mp3-encode-lame"] }
```

## Quick Start

```rust,no_run
# #[cfg(feature = "mp3-encode-lame")] {
use oxiaudio_encode_mp3_lame::lame::{LameMp3Encoder, Mp3Tags, VbrPreset};
use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

let buf = AudioBuffer {
    samples: vec![0.0f32; 1024],
    sample_rate: 44_100,
    channels: ChannelLayout::Stereo,
    format: SampleFormat::F32,
};

// Fluent builder: VBR "Music" preset + ID3v2.4 tags → Vec<u8>.
let data = LameMp3Encoder::builder(128)
    .with_vbr_preset(VbrPreset::Music)
    .with_tags(Mp3Tags::builder().title("Demo").artist("OxiAudio").build())
    .encode_to_vec(&buf)?;
assert!(!data.is_empty());
# }
# Ok::<(), oxiaudio_core::OxiAudioError>(())
```

## API Overview

### Crate root (always available)

| Item | Kind | Description |
|------|------|-------------|
| `compute_replaygain_gain_approx` | fn | Approximate ReplayGain 2.0 track gain (dB) from RMS loudness, referenced to −18 LUFS. Pure Rust; available without the FFI feature |
| `AudioBuffer`, `AudioEncoder`, `ChannelLayout`, `OxiAudioError` | re-export | Re-exported from `oxiaudio-core` |

### `lame` module (requires `mp3-encode-lame`)

#### Encoder types

| Item | Kind | Description |
|------|------|-------------|
| `LameMp3Encoder` | struct | `AudioEncoder` impl. Fields: `bitrate: u32`, `mode: LameMode`, `id3_tags: Option<Mp3Tags>`. `Default` = 128 kbps JointStereo. `builder(bitrate)` |
| `LameMp3EncoderBuilder` | struct | Fluent builder: `with_vbr_preset`, `with_abr`, `with_quality`, `with_mode`, `with_tags`, `with_ms_stereo_threshold`, then `encode` / `encode_to_vec` / `encode_to_file` |
| `LameMp3StreamEncoder<W>` | struct | Chunk-at-a-time encoder: `new`, `encode_chunk`, `finalize`, plus `frames_encoded`, `bytes_written`, `estimated_bitrate_kbps`, `elapsed_secs` |
| `LAME_ENCODER_DELAY` | const | LAME's algorithmic encoder delay (576 samples) for gapless trimming |

#### Mode / preset / tag types

| Item | Kind | Description |
|------|------|-------------|
| `LameMode` | enum | `Stereo`, `JointStereo`, `DualChannel`, `Mono`, `Vbr { quality }`, `Abr { target_kbps }`, `ForcedMono` |
| `VbrPreset` | enum | `Voice`, `Podcast`, `Music`, `HiFidelity`, `HighFidelity`, `Archival`; `quality()`, `quality_value()`, `to_mode()` |
| `Mp3Tags` | struct | ID3v2.4 fields (title/artist/album/track/year/genre/composer/comment/disc, album art, ReplayGain track+album gain/peak, lyrics, user-defined TXXX, gapless delay/padding, extended-header CRC, footer). `builder()` |
| `Mp3TagsBuilder` | struct | Fluent builder for `Mp3Tags` (`title`, `artist`, `album`, `year`/`year_int`, `track`, `genre`, `composer`, `comment`, `disc_number`, `album_art`, `replaygain_track_gain`/`_peak`, `with_replaygain_album_gain`/`_peak`, `lyrics`, `user_defined`, `encoder_delay`, `encoder_padding`, `extended_header_crc`, `write_footer`, `build`) |
| `AlbumArt` | struct | APIC payload: `mime_type: String`, `data: Vec<u8>` |

#### Convenience functions

| Item | Kind | Description |
|------|------|-------------|
| `encode_mp3_cbr_to_vec` | fn | CBR encode → `Vec<u8>` |
| `encode_mp3_cbr_to_file` | fn | CBR encode → file path |
| `encode_mp3_abr` | fn | ABR encode to a `Write+Seek` |
| `encode_mp3_with_auto_replaygain` | fn | CBR encode with auto-computed ReplayGain written to both ID3 TXXX frames and the Xing/LAME binary header |
| `write_xing_replaygain` | fn | Patch radio gain + peak into an existing Xing/Info header in a `&mut [u8]` |

#### `lame::id3v2` submodule

| Item | Kind | Description |
|------|------|-------------|
| `write_id3v2_4` | fn | Serialize an `Mp3Tags` value into ID3v2.4 tag bytes |

## Feature Flags

| Feature | Default | Pure Rust | Description |
|---------|---------|-----------|-------------|
| `mp3-encode-lame` | no | **No (LGPL FFI)** | Enables the entire `lame` module by linking `mp3lame-encoder` against `libmp3lame` |

## Supported bitrates

CBR / ABR bitrates (kbps): `8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320`. VBR quality is `0` (best/largest) … `9` (worst/smallest); `VbrPreset` maps the common cases.

## Error Variants

All fallible functions return [`oxiaudio_core::OxiAudioError`] — most failures surface as `Encode(String)` (LAME builder/encode/flush errors, unsupported bitrate or VBR quality) or `Io(std::io::Error)` (file write failures). See [`oxiaudio-core`](../oxiaudio-core) for the full enum.

## Cross-references

- [`oxiaudio-core`](../oxiaudio-core) — `AudioBuffer`, `AudioEncoder`, `OxiAudioError`
- [`oxiaudio-encode`](../oxiaudio-encode) — Pure-Rust WAV / FLAC / AIFF / AU / Vorbis / AAC / Opus encoders (enables this crate via its `mp3` feature)
- [`oxiaudio-dsp`](../oxiaudio-dsp) — for accurate EBU R128 loudness (`loudness_lufs`) instead of the RMS approximation here
- [`oxiaudio`](../oxiaudio) — the facade; enable `mp3-encode-lame` or `full` to surface MP3 encoding

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
