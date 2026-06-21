# oxiaudio-encode — Pure-Rust audio encoders for OxiAudio

[![Crates.io](https://img.shields.io/crates/v/oxiaudio-encode.svg)](https://crates.io/crates/oxiaudio-encode)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`oxiaudio-encode` is the encode-side workhorse of the OxiAudio stack. It turns an [`oxiaudio_core::AudioBuffer<f32>`] into WAV, FLAC, AIFF / AIFF-C, AU/SND, OGG Vorbis, AAC-LC / M4A, and OGG Opus byte streams — all in `#![forbid(unsafe_code)]` Pure Rust. WAV is backed by `hound`, FLAC by `flacenc`, and the remaining containers are written by hand-rolled encoders in this crate. It also provides metadata writers (ID3v2.4, APEv2, FLAC Vorbis-comments + pictures, WAV cue sheets), streaming encoders, TPDF dithering, and two-pass loudness normalization.

This crate carries no MP3 encoder. MP3 encoding is opt-in via the separate `oxiaudio-encode-mp3-lame` quarantine crate, which must be depended on directly. With default features (and without that crate) this crate is 100% C/C++/Fortran-free.

## Installation

```toml
[dependencies]
oxiaudio-encode = "0.2.0"

# MP3 encoding is NOT in this crate; depend on `oxiaudio-encode-mp3-lame` directly:
# oxiaudio-encode-mp3-lame = { version = "0.2.0", features = ["mp3-encode-lame"] }
```

## Quick Start

```rust
use std::io::Cursor;
use oxiaudio_encode::{EncoderConfig, WavBitDepth, FlacEncoder};
use oxiaudio_core::{AudioBuffer, AudioEncoder, ChannelLayout, SampleFormat};

let buf = AudioBuffer {
    samples: vec![0.0f32; 1024],
    sample_rate: 44_100,
    channels: ChannelLayout::Mono,
    format: SampleFormat::F32,
};

// Builder-style: 16-bit WAV with TPDF dithering.
let mut wav = Cursor::new(Vec::new());
EncoderConfig::new(44_100, 1)
    .with_bit_depth(WavBitDepth::I16)
    .with_dither(true)
    .encode_wav(&buf, &mut wav)?;

// Direct FLAC encoder at compression level 8.
let mut flac = Cursor::new(Vec::new());
FlacEncoder::new(8).encode(&buf, &mut flac)?;
# Ok::<(), oxiaudio_core::OxiAudioError>(())
```

### Streaming a WAV without buffering the whole file

```rust
use std::io::Cursor;
use oxiaudio_encode::{WavStreamEncoder, WavBitDepth};
use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

let mut enc = WavStreamEncoder::new(
    Cursor::new(Vec::new()),
    44_100,
    ChannelLayout::Mono,
    WavBitDepth::F32,
)?;
let chunk = AudioBuffer {
    samples: vec![0.0f32; 512],
    sample_rate: 44_100,
    channels: ChannelLayout::Mono,
    format: SampleFormat::F32,
};
enc.encode_chunk(&chunk)?;
enc.finalize()?; // patches RIFF header sizes
# Ok::<(), oxiaudio_core::OxiAudioError>(())
```

## API Overview

All functions accept an `AudioBuffer<f32>` and return `Result<_, oxiaudio_core::OxiAudioError>`. `*_file` variants take a path; the others take a `Write` (and usually `Seek`) destination, or return a `Vec<u8>`.

### WAV (`wav_core`, `wav_ext`, `wav_cue`)

| Item | Kind | Description |
|------|------|-------------|
| `WavEncoder` | struct | `AudioEncoder` impl; field `bit_depth: WavBitDepth`. `encode_to_file`, `encode_with_metadata` (LIST/INFO chunk) |
| `WavBitDepth` | enum | `F32` (default), `I16`, `I24`, `I32`, `U8` |
| `WavEncodeConfig` | struct | Lower-level WAV encode configuration |
| `encode_wav_with_config` | fn | Encode to a `Write+Seek` with an explicit `WavEncodeConfig` |
| `encode_wav_to_vec` | fn | Encode and return `Vec<u8>` |
| `encode_wav_streaming` | fn | Encode to a non-seekable `Write` (sentinel RIFF sizes) |
| `encode_wav_rf64` / `encode_wav_rf64_file` | fn | RF64 (64-bit sizes) for files > 4 GiB |
| `encode_wav_with_progress` | fn | Encode with an `EncodeProgressFn` callback |
| `WavStreamEncoder` | struct | Chunk-by-chunk WAV writer; `new`, `encode_chunk`, `frames_written`, `finalize` |
| `apply_tpdf_dither` | fn | In-place triangular dither before integer quantization |
| `encode_wav_with_cues` / `encode_wav_with_cues_file` | fn | Embed `cue ` markers |
| `CuePoint` | struct | A single WAV cue-sheet marker |

### FLAC (`flac_core`, `flac_meta`, `flac_picture`, `flac_streaming`)

| Item | Kind | Description |
|------|------|-------------|
| `FlacEncoder` | struct | `AudioEncoder` impl; fields `compression_level: u8`, `bits_per_sample: u8`. `new`, `with_bits_per_sample`, `encode_to_file` |
| `FlacBitDepth` | enum | `I16`, `I24`; `bits()` → `u8` |
| `FlacConfig` | struct | `compression: u8` (0–8), `bit_depth: FlacBitDepth`; `Default` = level 5 / I16 |
| `encode_flac` / `encode_flac_with_level` / `encode_flac_with_config` | fn | Encode to a `Write+Seek` |
| `encode_flac_to_vec` | fn | Encode and return `Vec<u8>` |
| `encode_flac_with_progress` | fn | Encode with a progress callback |
| `encode_flac_parallel` | fn | Rayon-parallel f32→i32 conversion, then sequential flacenc |
| `FlacStreamEncoder` | struct | Buffering streaming encoder (accumulates, encodes on `finalize`) |
| `FlacStreamingEncoder` | struct | True-streaming encoder (encodes frames immediately) |
| `FlacMetaConfig` | struct | FLAC + Vorbis-comment metadata configuration |
| `encode_flac_with_metadata` | fn | Encode with embedded Vorbis-comments |
| `encode_flac_with_seektable` / `encode_flac_with_seektable_file` | fn | Encode with a SEEKTABLE block |
| `encode_flac_with_md5` / `encode_flac_with_md5_file` | fn | Embed an MD5 in STREAMINFO |
| `inject_flac_md5` | fn | Patch the STREAMINFO MD5 of an existing FLAC in place |
| `FlacPicture` | struct | Cover-art / picture metadata block |
| `encode_flac_with_picture` / `_file` | fn | Encode with an embedded PICTURE block |
| `encode_flac_with_album_art` / `_file` | fn | Convenience wrappers for raw album-art bytes |
| `encode_flac_with_metadata_and_picture` | fn | Both Vorbis-comments and a picture block |

### Loudness normalization (`flac_core`)

| Item | Kind | Description |
|------|------|-------------|
| `LoudnessTarget` | enum | `Streaming`, `Podcast`, `Broadcast`, `Custom(f32)`; `lufs()` → `f32` |
| `analyze_loudness_gain` | fn | Measure integrated loudness, return the linear gain to reach a target |
| `analyze_loudness_gain` companion: `encode_normalized_wav` / `_file` | fn | Two-pass measure → gain → encode as WAV |

### AIFF / AIFF-C (`aiff`)

| Item | Kind | Description |
|------|------|-------------|
| `write_aiff` / `write_aiff_file` | fn | Write 16-bit big-endian PCM AIFF |
| `write_aiff_with_chunks` | fn | AIFF with optional NAME / AUTH / ANNO chunks |
| `encode_aiff_with_metadata` | fn | AIFF with metadata |
| `AiffBitDepth` | enum | 8 / 16 / 24-bit + 32-bit float selector |
| `AiffStreamEncoder` | struct | Chunk-by-chunk AIFF writer |
| `write_aiffc` / `write_aiffc_file` | fn | Write AIFF-C |
| `AiffcCodec` | enum | `NONE`, `ULAW`, `ALAW` compression selector |

### AU / SND (`au`)

| Item | Kind | Description |
|------|------|-------------|
| `encode_au` / `encode_au_file` | fn | Write Sun/NeXT AU |
| `AuEncoding` | enum | `I16`, `I24`, `F32` sample-format selector |

### OGG Vorbis (`vorbis`, `ogg`)

| Item | Kind | Description |
|------|------|-------------|
| `encode_vorbis` / `encode_vorbis_file` | fn | Encode OGG Vorbis I |
| `encode_vorbis_with_quality` / `encode_vorbis_quality_file` | fn | Explicit VBR quality control |
| `VorbisQuality` | struct | VBR quality (q-1…q10); `new`, `default_quality`, `from_level` |
| `OggStream` | struct | Low-level OGG page builder |
| `write_ogg_page`, `write_vorbis_comment_packet`, `ogg_crc32` | fn | OGG container primitives |

### AAC-LC / M4A (`aac`, `aac_m4a`)

| Item | Kind | Description |
|------|------|-------------|
| `encode_aac` / `encode_aac_file` | fn | Encode AAC-LC into ADTS frames |
| `encode_aac_mode` / `encode_aac_mode_file` | fn | Encode with an explicit `AacBitrateMode` |
| `encode_aac_pns` / `encode_aac_tns` | fn | PNS / TNS tool variants |
| `AacBitrateMode` | enum | AAC bitrate-mode selector |
| `encode_m4a` / `encode_m4a_file` | fn | Wrap AAC-LC in an M4A/MP4 container |

### OGG Opus (`opus_encoder`, `opus_silk`, `opus_hybrid`, `opus_celt`, `opus_mdct`, `opus_range`)

| Item | Kind | Description |
|------|------|-------------|
| `encode_opus` / `encode_opus_file` | fn | Write an OGG Opus stream (CELT-mode frames) |
| `OpusEncodeConfig` | struct | Opus bitrate / frame-size configuration |
| `OpusStreamEncoder` | struct | Frame-at-a-time Opus encoder |
| `SilkBandwidth` | enum | NB / MB / WB / SWB voice-band selector |
| `SilkLpcFrame` | struct | SILK LP frame (NLSFs, residual, pitch, gain) |
| `analyze_silk_frame` / `encode_silk_frame` | fn | SILK analysis / frame encode |
| `encode_hybrid_frame`, `hybrid_toc`, `should_use_hybrid` | fn | SILK/CELT hybrid-mode helpers |

> **Note:** The Opus and AAC encoders write conformant container framing; their audio payloads are early-stage (CELT uses non-conformant placeholder quantization) and are not yet guaranteed to decode in third-party players. WAV, FLAC, AIFF, and AU are production-quality.

### Metadata writers (`id3`, `apev2`)

| Item | Kind | Description |
|------|------|-------------|
| `Id3v24Tag` | struct | ID3v2.4 tag writer (TIT2, TPE1, TALB, TDRC, TRCK, TCON, COMM, TCOM, APIC, TXXX) |
| `ApeItem` | struct | A single APEv2 key/value item |
| `write_apev2` | fn | Write an APEv2 tag block to any writer |

### Generic streaming (`StreamEncoder`)

| Item | Kind | Description |
|------|------|-------------|
| `StreamEncoder` | trait | `Send` object-safe trait: `write_chunk(&mut self, …)` + `finalize(self: Box<Self>)`. Implemented by `WavStreamEncoder`, `FlacStreamEncoder` |
| `EncoderConfig` | struct | Unified WAV/FLAC builder with normalize + dither pre-processing (`with_bit_depth`, `with_dither`, `with_flac_compression`, `with_normalize`, `encode_wav`, `encode_flac`) |
| `EncodeProgressFn` | type | Progress-callback alias used by the `*_with_progress` functions |

`WavStreamEncoder` and `FlacStreamEncoder` also implement [`oxiaudio_core::AudioSink`].

## Error Variants

All fallible functions return [`oxiaudio_core::OxiAudioError`]:

| Variant | Description |
|---------|-------------|
| `Io(std::io::Error)` | Underlying I/O failure (file create, write) |
| `Decode(String)` | Decode failure (used by round-trip helpers) |
| `Encode(String)` | Encoder error (e.g. flacenc/hound failure, finalized stream reused) |
| `UnsupportedFormat(String)` | Format or parameter combination not supported |
| `InvalidChannelLayout(String)` | Channel layout invalid for the target codec |
| `InvalidSampleRate(String)` | Sample rate invalid for the target codec |
| `BufferOverflow(String)` | Internal buffer overflow |
| `BufferUnderflow(String)` | Internal buffer underflow |

## Cross-references

- [`oxiaudio-core`](../oxiaudio-core) — shared `AudioBuffer`, `AudioEncoder`, `AudioSink`, `ChannelLayout`, `OxiAudioError`
- [`oxiaudio-decode`](../oxiaudio-decode) — the decode-side counterpart
- [`oxiaudio-encode-mp3-lame`](../oxiaudio-encode-mp3-lame) — LAME MP3 adapter; depend on it directly for MP3 encoding
- [`oxiaudio-dsp`](../oxiaudio-dsp) — resampling, dynamics, effects, loudness analysis
- [`oxiaudio`](../oxiaudio) — the top-level facade that re-exports this crate

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
