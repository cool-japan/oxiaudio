# oxiaudio-core — Core traits and types for OxiAudio

[![Crates.io](https://img.shields.io/crates/v/oxiaudio-core.svg)](https://crates.io/crates/oxiaudio-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`oxiaudio-core` defines the foundational data model and trait surface that every OxiAudio crate is built on. It contains the interleaved sample container ([`AudioBuffer`]), the format/layout descriptors, the codec and processing traits, plus a handful of pure utilities (a thread-safe ring buffer, an audio clock, a buffer pipeline, and a compact binary IPC format). It carries **no codec or DSP logic** — decoding lives in `oxiaudio-decode`, encoding in `oxiaudio-encode`, and signal processing in `oxiaudio-dsp`.

The crate is **100% Pure Rust** with `#![forbid(unsafe_code)]`. Its only runtime dependency is `thiserror`; `serde` support is gated behind an optional feature. Sample buffers are always interleaved `f32` (or any `T`), and all conversion, mixing, fade, crossfade, and resampling helpers operate on that representation.

## Installation

```toml
[dependencies]
oxiaudio-core = "0.2.0"

# With serde derives on the public data types:
oxiaudio-core = { version = "0.2.0", features = ["serde"] }
```

## Quick Start

```rust
use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

// Build a 0.1 s stereo silence buffer at 44.1 kHz.
let mut buf = AudioBuffer::<f32>::silence(44_100, ChannelLayout::Stereo, 4410);
assert_eq!(buf.frame_count(), 4410);
assert!((buf.duration_secs() - 0.1).abs() < 1e-9);

// Fade in over 1000 frames and measure the peak.
buf.fade_in(1000);
let _peak = buf.peak_amplitude();

// Round-trip through the compact binary IPC format.
let bytes = buf.to_ipc_bytes()?;
let restored = AudioBuffer::<f32>::from_ipc_bytes(&bytes)?;
assert_eq!(restored.frame_count(), buf.frame_count());
# Ok::<(), oxiaudio_core::OxiAudioError>(())
```

## API Overview

### `AudioBuffer<T>` — interleaved sample container

The central type. Public fields: `samples: Vec<T>`, `sample_rate: u32`, `channels: ChannelLayout`, `format: SampleFormat`. Implements `Clone` for `T: Clone`.

#### Generic methods (any `T`)

| Method | Description |
|--------|-------------|
| `duration_secs()` | Duration in seconds (0.0 if no channels / rate) |
| `frame_count()` | Number of frames (samples per channel) |
| `is_empty()` | `true` when the buffer has no samples |

#### `AudioBuffer<f32>` methods

| Method | Description |
|--------|-------------|
| `silence(sample_rate, channels, frames)` | Construct a zero-filled buffer |
| `slice_frames(start, end)` | Sub-buffer over the frame range `[start, end)` |
| `append(&other)` | Append another buffer in place (layout/rate must match) |
| `split_to_planar()` | De-interleave into one `Vec<f32>` per channel |
| `to_f64()` | Convert to `AudioBuffer<f64>` |
| `to_i32_24bit()` | Convert to 24-bit signed PCM range `[-8388608, 8388607]` |
| `peak_amplitude()` | Peak absolute amplitude |
| `rms_amplitude()` | Root-mean-square amplitude |
| `peak_db()` / `rms_db()` | Peak / RMS in dBFS (`-inf` for silence) |
| `fade_in(frames)` / `fade_out(frames)` | Raised-cosine fade in / out (in place) |
| `gain_ramp(start, end)` | Linear gain ramp across the buffer (in place) |
| `mix_with(&other, gain)` | Additive in-place mix of the overlapping prefix |
| `mixed_with(&other, level)` | Out-of-place mix; output length = longer of the two |
| `reverse()` / `reversed()` | Time-reverse in place / as a copy |
| `crossfade(a, b, overlap)` | Associated fn: equal-power crossfade of two buffers |
| `linear_crossfade(&other, fade)` | Linear-envelope crossfade onto `other` |
| `resample_linear(target_rate)` | Fast linear-interpolation resampler (preview quality) |
| `to_ipc_bytes()` | Serialize to the `ABUF` v1 binary format |
| `from_ipc_bytes(data)` | Associated fn: deserialize from `ABUF` v1 bytes |

#### `AudioBuffer<f64>` methods

| Method | Description |
|--------|-------------|
| `to_f32()` | Convert back to `AudioBuffer<f32>` (lossy) |

#### `From` conversions

`AudioBuffer<f32>` converts to/from `AudioBuffer<u8>`, `AudioBuffer<i16>`, and `AudioBuffer<i32>` via `From<&_>` (integer formats are normalized against their full range; `u8` is biased at 128). `from_planar(channels_data, sample_rate, format)` is available for any `T: Clone + Default`.

### Free buffer helpers (planar / channel conversion)

| Function | Description |
|----------|-------------|
| `to_planar(&buf)` | Interleaved → planar (`Vec<Vec<f32>>`) |
| `from_planar(&channels, sample_rate)` | Planar → interleaved (validated lengths) |
| `from_planar_into(planes, sample_rate, out)` | Planar → caller-provided interleaved slice |
| `from_planar_unchecked(planes, sample_rate)` | Planar → interleaved, skipping length validation |
| `downmix_51_to_stereo(&buf)` | 5.1 → stereo (ITU-R BS.775-3 folddown) |
| `downmix_to_mono(&buf)` | Average all channels into mono |
| `upmix_mono_to_stereo(&buf)` | Duplicate a mono channel to stereo |

### `SampleFormat` enum — 6 variants

| Variant | Meaning |
|---------|---------|
| `U8` | 8-bit unsigned PCM (bias at 128) |
| `I16` | 16-bit signed PCM |
| `I24` | 24-bit packed signed PCM |
| `I32` | 32-bit signed PCM |
| `F32` | 32-bit float |
| `F64` | 64-bit float |

Methods: `normalize_i32_to_f32(raw)`, `bit_depth()`, `is_float()`, `is_integer()`, `byte_size()`. Implements `Display`, `FromStr`, and `TryFrom<&str>` (accepts aliases such as `s16`, `float`, `double`).

### `ChannelLayout` enum (`#[non_exhaustive]`)

| Variant | Channels | `Display` |
|---------|----------|-----------|
| `Mono` | 1 | `mono` |
| `Stereo` | 2 | `stereo` |
| `Quad` | 4 | `quad` |
| `Surround51` | 6 | `5.1` |
| `Surround51Side` | 6 | `5.1side` |
| `Surround71` | 8 | `7.1` |
| `Atmos714` | 12 | `7.1.4` |

Methods: `channel_count()`, plus `From<u16>` (maps a raw channel count to the best-fit layout; unknown counts fall back to `Stereo`).

### `AudioFormat` and `AudioBufferLayout`

- `AudioFormat` — lightweight probe descriptor with `sample_rate: u32`, `channels: ChannelLayout`, `format: SampleFormat`. `Copy`.
- `AudioBufferLayout` — `Interleaved` or `Planar`. `Copy`.

### `ChannelId` and `ChannelMap`

`ChannelId` is a semantic channel label: `FrontLeft`, `FrontRight`, `FrontCenter`, `LowFrequency`, `RearLeft`, `RearRight`, `SideLeft`, `SideRight`, `TopFrontLeft`, `TopFrontRight`, `TopRearLeft`, `TopRearRight`.

`ChannelMap` is an ordered index → `ChannelId` mapping.

| Method | Description |
|--------|-------------|
| `ChannelMap::new(channels)` | Construct from an ordered `Vec<ChannelId>` |
| `ChannelMap::for_layout(layout)` | Standard map for a `ChannelLayout` |
| `channel_count()` | Number of mapped channels |
| `get(idx)` | `ChannelId` at index, or `None` |
| `index_of(id)` | First index of a `ChannelId`, or `None` |
| `iter()` | Iterate over channel IDs in order |
| `ChannelMap::remap(&buf, &src, &dst)` | Reorder a buffer's channels (missing channels → silence) |

### `Sample` trait

PCM sample abstraction (`Copy + Send + Sync + 'static`) implemented for `u8`, `i16`, `i32`, `f32`, `f64`.

| Item | Description |
|------|-------------|
| `const EQUILIBRIUM` | Silence value (0.0 for float, 128 for `u8`) |
| `const MAX_AMPLITUDE` | Maximum positive amplitude |
| `to_f32(self)` | Convert to normalized `f32` in `[-1.0, 1.0]` |
| `from_f32(value)` | Construct from normalized `f32` (clamped for integers) |

> `AudioBuffer<T>` does **not** require `T: Sample`; the trait is an opt-in building block for generic conversion/DSP code.

### Codec and processing traits

| Trait | Required method | Role |
|-------|-----------------|------|
| `AudioDecoder` | `decode(&mut self, src)` | Decode a `Read + Seek` source to `AudioBuffer<f32>` |
| `AudioEncoder` | `encode(&mut self, buf, dst)` | Encode a buffer to a `Write + Seek` sink |
| `StreamingDecoder` | `decode_frames(&mut self, src, block_size)` | Chunked decode returning an iterator of buffers |
| `AudioFilter` | `apply(&self, buf)` | Pure transform producing a new buffer |
| `AudioSource` | `read_chunk(&mut self)` | Pull-based chunk source |
| `AudioSink` | `write_chunk(&mut self, buf)` | Push-based chunk sink |

### `AudioPipeline`, `AudioNode`, `ParallelBranchNode`

A linear chain of processing nodes operating on `AudioBuffer<f32>`.

`AudioNode` (trait, `Send + Sync`): `name()`, `bypass()` (default `false`), `process(input)`, `latency_frames()` (default `Some(0)`).

| Type / method | Description |
|---------------|-------------|
| `AudioPipeline::new()` | Empty pipeline |
| `push_node(Box<dyn AudioNode>)` | Append a node (builder style) |
| `process(&input)` | Run input through every non-bypassed node |
| `latency_hint()` | Latency hint in frames (currently 0) |
| `total_latency_frames()` | Sum of node latencies, or `None` if empty |
| `ParallelBranchNode::new(branches)` | Split → process each branch → mix (gain-normalized) |
| `ParallelBranchNode::with_branch(branch)` | Append a branch (builder style) |

### `AudioRingBuffer<T>` and `OverflowPolicy`

A bounded, thread-safe FIFO of `Copy + Default` elements (capacity rounded up to the next power of two). `OverflowPolicy` is one of `Error`, `OverwriteOldest`, or `DropNewest`.

| Method | Description |
|--------|-------------|
| `new(capacity)` | Construct (default policy `Error`) |
| `with_policy(p)` / `with_overflow_policy(p)` | Set overflow policy (builder style) |
| `capacity()` | Power-of-two capacity |
| `available_read()` / `available_write()` | Readable / writable element counts |
| `available_read_frames()` / `available_write_frames()` | Frame-count aliases |
| `is_empty()` / `is_full()` | Fill-state checks |
| `push(value)` / `pop()` | Single-element write / read |
| `write(&frames)` / `read(max)` | Bulk write / read |
| `read_exact(&mut out)` | Read exactly `out.len()` elements or error |
| `write_frames(&data, frames)` / `read_frames(frames)` | Exact frame write / read |
| `clear()` | Discard all queued elements |

### `AudioClock` and `Timestamp`

`Timestamp` — `Frames(u64)` or `Seconds(f64)`; converts via `to_frames(sample_rate)` / `to_seconds(sample_rate)`.

`AudioClock` — a monotonic frame counter that reports drift versus wall-clock.

| Method | Description |
|--------|-------------|
| `new(sample_rate)` | Create a clock anchored to the current instant |
| `advance(frames)` | Advance by a frame count |
| `elapsed_frames()` / `elapsed_secs()` | Elapsed audio time |
| `drift_ppm()` | Drift vs `Instant::now()` in PPM |
| `drift_ppm_from_ns(elapsed_wall_ns)` | Drift vs an explicit wall duration (testable) |

### `AudioMetadata`

Container/track metadata; every field is optional (`Default` = all `None`): `title`, `artist`, `album`, `duration_secs`, `bitrate_kbps`, `genre`, `composer`, `year`, `track_number`, `disc_number`, `comment`, `album_art` (raw image bytes).

### `ipc` module — `ABUF` v1 binary format

Compact little-endian serialization for `AudioBuffer<f32>` (19-byte header + raw f32 samples).

| Function | Description |
|----------|-------------|
| `serialize_audio_buffer_f32(buf, writer)` | Write to any `Write` |
| `deserialize_audio_buffer_f32(reader)` | Read from any `Read` |
| `to_ipc_bytes(buf)` | Serialize into a new `Vec<u8>` |
| `from_ipc_bytes(data)` | Deserialize from a byte slice |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | off | Derive `Serialize`/`Deserialize` on `AudioBuffer`, `AudioFormat`, `SampleFormat`, `AudioBufferLayout`, `ChannelLayout`, `ChannelId`, `ChannelMap`, `AudioMetadata`, and `Timestamp` |

## `OxiAudioError` variants

The crate-wide error type (`thiserror`-derived), re-exported throughout the OxiAudio ecosystem.

| Variant | Description |
|---------|-------------|
| `Io(std::io::Error)` | I/O failure (`#[from]`) |
| `Decode(String)` | Decoding failed |
| `Encode(String)` | Encoding failed |
| `UnsupportedFormat(String)` | Format / sample format not supported |
| `InvalidChannelLayout(String)` | Channel-layout mismatch or invalid layout |
| `InvalidSampleRate(String)` | Sample-rate mismatch or invalid rate |
| `BufferOverflow(String)` | Ring-buffer write exceeded capacity |
| `BufferUnderflow(String)` | Ring-buffer read had insufficient data |

## Related crates

| Crate | Role |
|-------|------|
| `oxiaudio` | Top-level façade re-exporting the ecosystem |
| `oxiaudio-decode` | Symphonia-backed decoding + native AIFF/AU/Opus/MIDI/etc. |
| `oxiaudio-encode` | Encoders (WAV, FLAC, …) |
| `oxiaudio-encode-mp3-lame` | MP3 encoding via LAME |
| `oxiaudio-dsp` | Resampling, gain, filters, spectral analysis, effects |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
