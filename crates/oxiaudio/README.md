# oxiaudio — The COOLJAPAN Pure-Rust audio codec + DSP facade

[![Crates.io](https://img.shields.io/crates/v/oxiaudio.svg)](https://crates.io/crates/oxiaudio)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`oxiaudio` is the top-level façade crate of the OxiAudio stack. It re-exports the core buffer types, decoders, encoders, and DSP under one roof so you can `decode_file → dsp → encode_*` without wiring up the member crates yourself. With the default `pure` feature, the entire pipeline is 100% Pure Rust (`#![forbid(unsafe_code)]`) — no C/C++/Fortran. MP3 *encoding* is **not** part of this facade; it is opt-in via the separate `oxiaudio-encode-mp3-lame` quarantine crate (LGPL FFI), which must be depended on directly.

Decoding (WAV/RF64, FLAC, MP3, OGG Vorbis, AAC/M4A, AIFF/AIFF-C, OGG Opus) is Pure Rust via Symphonia and OxiAudio's own readers. Encoding covers WAV/RF64, FLAC, AIFF, AU, Vorbis, AAC, and Opus in Pure Rust. MP3 encoding requires `oxiaudio-encode-mp3-lame` as a direct dependency.

## Installation

```toml
[dependencies]
# Pure-Rust decode + encode + DSP (default):
oxiaudio = "0.2"

# Add LAME MP3 encoding (LGPL, FFI):
# MP3 *encoding* (LAME, LGPL FFI) is NOT in this pure facade; depend on `oxiaudio-encode-mp3-lame` directly.
```

## Quick Start

```rust,no_run
use std::path::Path;

// Decode an audio file (format auto-detected).
let buf = oxiaudio::decode_file(Path::new("input.flac"))?;
println!("{} frames at {} Hz", buf.frame_count(), buf.sample_rate);

// Apply DSP: normalize, then add reverb.
let mut processed = buf.clone();
oxiaudio::dsp::normalize(&mut processed, -1.0);
let with_reverb = oxiaudio::dsp::reverb(&processed, 0.6, 0.4, 0.3);

// Re-encode to a different format.
oxiaudio::encode_flac(&with_reverb, Path::new("output.flac"))?;
# Ok::<(), oxiaudio::OxiAudioError>(())
```

## Crate Layout

`oxiaudio` is a thin facade over five member crates. Everything below is re-exported from the crate root (or the `dsp` module) — you rarely depend on the members directly.

| Member crate | Surfaced as | Role |
|--------------|-------------|------|
| [`oxiaudio-core`](../oxiaudio-core) | crate root | `AudioBuffer`, traits, channel maps, IPC, ring buffer |
| [`oxiaudio-decode`](../oxiaudio-decode) | `decode_*`, format types | Symphonia + Pure-Rust decoders (feature `pure`) |
| [`oxiaudio-encode`](../oxiaudio-encode) | `encode_*`, encoder types | WAV/FLAC/AIFF/AU/Vorbis/AAC/Opus encoders (feature `pure`) |
| [`oxiaudio-dsp`](../oxiaudio-dsp) | `dsp::*` | Resampling, dynamics, effects, loudness, spectral (feature `pure`) |
| [`oxiaudio-encode-mp3-lame`](../oxiaudio-encode-mp3-lame) | *(not in facade)* | LAME MP3 quarantine crate — depend on it directly for MP3 encoding |

## Supported Formats

### Decode (Pure Rust, `pure` feature)

| Format | Backend |
|--------|---------|
| WAV / RF64 | Pure Rust |
| FLAC | Pure Rust |
| MP3 (MPEG Audio) | Pure Rust (symphonia) |
| OGG Vorbis | Pure Rust |
| AAC / M4A | Pure Rust |
| AIFF / AIFF-C | Pure Rust |
| OGG Opus | Pure Rust (`opus` feature on `oxiaudio-decode`) |

### Encode

| Format | Pure Rust | Feature |
|--------|-----------|---------|
| WAV / RF64 | Yes | `pure` (default) |
| FLAC | Yes | `pure` (default) |
| AIFF / AIFF-C | Yes | `pure` (default) |
| AU / SND | Yes | `pure` (default) |
| OGG Vorbis | Yes | `pure` (default) |
| AAC-LC / M4A | Yes | `pure` (default) |
| OGG Opus | Yes | `pure` (default) |
| MP3 (via LAME) | **No (FFI, LGPL)** | not in facade — use `oxiaudio-encode-mp3-lame` crate directly |

## Top-level Entry Functions

### Decode (feature `pure`)

| Function | Returns | Description |
|----------|---------|-------------|
| `decode_file(path)` | `AudioBuffer<f32>` | Decode any supported file (format auto-detected) |
| `decode_file_f64(path)` / `decode_to_f64(path)` | `AudioBuffer<f64>` | Decode to 64-bit float |
| `decode_to_i16` / `decode_to_i32` | `AudioBuffer<_>` | Decode to integer PCM |
| `decode_files(paths)` | `Vec<Result<…>>` | Parallel (rayon) batch decode |
| `decode_reader(reader)` | `AudioBuffer<f32>` | Decode from any `Read + Seek` |
| `decode_stream(reader)` / `decode_stream_with_block_size(reader, n)` | `impl Iterator` | Chunked streaming decode |
| `decode_file_with_metadata(path)` | `(AudioBuffer, AudioMetadata)` | Decode + tags in one pass |
| `decode_file_with_options(path, opts)` | `AudioBuffer<f32>` | Decode with a `DecodeOptions` recovery policy |
| `decode_file_with_replaygain(path)` | `AudioBuffer<f32>` | Apply embedded ReplayGain on decode |
| `decode_tolerant(path)` | `AudioBuffer<f32>` | Skip corrupted frames; never panics |
| `decode_opus_file` / `decode_opus_reader` | `AudioBuffer<f32>` | Pure-Rust OGG Opus decode |
| `decode_aiff_file`, `decode_au_file`, `decode_aiffc_compressed[_file]`, `decode_raw_pcm_file` | `AudioBuffer<f32>` | Container-specific decoders |
| `detect_format(path)` / `file_format(path)` / `detect_format_file` / `detect_format_from_bytes` / `detect_format_from_path` | `AudioFormat` / hint | Probe format without decoding audio |

Plus MIDI (`synthesize_midi`, `synthesize_midi_default`, `MidiFile`, `midi_note_to_hz`), tag/cue parsers (`parse_id3v1`, `parse_replaygain`, `extract_album_art`, `extract_lyrics`, `parse_wav_cues`, `parse_flac_cue_sheet`, `parse_gapless_info`, `parse_opus_head`), and µ-law/A-law helpers (`ulaw_to_linear`, `alaw_to_linear`).

### Encode (feature `pure`)

| Function | Description |
|----------|-------------|
| `encode_wav(buf, path)` | Encode to WAV (32-bit float by default) |
| `encode_wav_with_config(buf, path, WavBitDepth)` | WAV at an explicit bit depth |
| `encode_wav_f64(buf, path)` | Encode an `AudioBuffer<f64>` as float WAV |
| `encode_wav_to_vec(buf)` | WAV → `Vec<u8>` |
| `encode_wav_rf64[_file]` | RF64 (> 4 GiB) WAV |
| `encode_wav_streaming` | WAV to a non-seekable `Write` |
| `encode_wav_with_cues[_file]` / `encode_wav_with_progress` | Cue markers / progress callback |
| `encode_flac(buf, path)` | Encode to FLAC (compression level 5) |
| `encode_flac_with_config(buf, path, &FlacConfig)` | FLAC with explicit `FlacConfig` |
| `encode_flac_with_level` / `encode_flac_to_vec` / `encode_flac_parallel` / `encode_flac_with_progress` | Level / `Vec` / parallel / progress variants |
| `encode_flac_with_metadata` / `_picture[_file]` / `_album_art[_file]` / `_metadata_and_picture` | FLAC + metadata / cover art |
| `encode_flac_with_md5[_file]` / `inject_flac_md5` / `encode_flac_with_seektable[_file]` | MD5 / seektable |
| `encode_aiff(buf, path)` / `encode_aiff_with_chunks` / `write_aiff_file` / `write_aiffc[_file]` | AIFF / AIFF-C |
| `encode_au(buf, path)` | AU/SND (16-bit) |
| `encode_vorbis[_file]` / `encode_vorbis_with_quality` / `encode_vorbis_quality_file` / `encode_vorbis_to_file` | OGG Vorbis |
| `encode_aac[_file]` / `encode_aac_to_file` / `encode_m4a[_file]` | AAC-LC / M4A |
| `encode_opus[_file]` | OGG Opus |
| `encode_stream(chunks, writer)` / `encode_stream_flac(chunks, writer, level)` | Encode an iterator/slice of chunks |
| `encode_normalized_wav[_file]` / `analyze_loudness_gain` | Two-pass loudness normalization |
| `apply_tpdf_dither` | In-place TPDF dither before quantization |

Encoder configuration/streaming types are re-exported too: `WavBitDepth`, `WavStreamEncoder`, `FlacBitDepth`, `FlacConfig`, `FlacStreamEncoder`, `FlacStreamingEncoder`, `AiffBitDepth`, `AiffStreamEncoder`, `AiffcCodec`, `AuEncoding`, `VorbisQuality`, `OpusEncodeConfig`, `OpusStreamEncoder`, `SilkBandwidth`, `SilkLpcFrame`, `EncoderConfig`, `EncodeProgressFn`, `StreamEncoder`, `CuePoint`, `FlacPicture`, `FlacMetaConfig`, `Id3v24Tag`, `ApeItem`, `LoudnessTarget`.

### Transcode / pipeline (feature `pure`)

| Item | Kind | Description |
|------|------|-------------|
| `convert(input, output)` | fn | Auto-transcode by file extension (`.wav`, `.flac`, `.aif`, `.aiff`) |
| `convert_with_dsp(input, output, f)` | fn | Decode → DSP closure → encode |
| `transcode_batch(inputs, dir, ext)` | fn | Parallel (rayon) batch transcode |
| `probe_metadata(path)` | fn | Extract `AudioMetadata` |
| `write_metadata(path, &AudioMetadata)` | fn | Embed metadata (WAV LIST/INFO) |
| `TranscodePipeline` | struct | `new` → `with_dsp` → `run` builder |
| `TranscodeStream` | struct | Streaming transcode: `new` → `with_filter` / `with_chunk_frames` → `run` |

## The `dsp` Module (feature `pure`)

```rust,no_run
let mut buf = oxiaudio::decode_file(std::path::Path::new("audio.wav"))?;
oxiaudio::dsp::gain(&mut buf, 6.0);                 // +6 dB
let buf = oxiaudio::dsp::resample(&buf, 44_100)?;   // FFT-based resample
let lufs = oxiaudio::dsp::loudness_lufs(&buf);      // EBU R128
# Ok::<(), oxiaudio::OxiAudioError>(())
```

| Category | Items |
|----------|-------|
| Gain / level | `gain`, `gain_inplace`, `normalize`, `normalize_inplace`, `normalize_loudness`, `normalize_to_lufs` |
| Sample-rate | `resample`, `resample_quality` + `ResampleQuality` |
| Channels | `mix_to_mono`, `split_channels`, `ms_encode`, `ms_decode` |
| Trim / split | `trim_silence`, `silence_split`, `short_time_energy` |
| Filters | `BiquadFilter`, `ParametricEq`, `Cascade`, `FirFilter` + `FirWindow`, `butterworth_lowpass`/`_highpass`, `chebyshev1_lowpass`/`_highpass`, `DspChain` |
| Dynamics | `Compressor`, `Limiter`, `NoiseGate`, `Expander`, `DeEsser`, `MultibandCompressor` + `BandSettings`, plus wrappers `compressor`, `gate` |
| Effects | `DelayLine`, `Chorus`, `Tremolo`, `Vibrato`, `Flanger`, `Phaser`, `Freeverb`, `EarlyReflections`, `ConvolutionReverb`, `PartitionedConvolutionReverb`, `ChannelVocoder`, wrappers `reverb`, `eq`, `delay`, `chorus` |
| Pitch / time | `pitch_shift`, `pitch_shift_pv`, `time_stretch`, `detect_pitch`, `detect_pitch_yin`, `detect_pitch_pyin`, `detect_pitch_autocorr` + `PitchFrame`, `PitchTracker` |
| Rhythm | `detect_onsets`, `complex_domain_onset`, `estimate_tempo`, `detect_tempo`, `detect_downbeats` + `TempoEstimate` |
| Loudness / meters | `loudness_lufs`, `loudness_range`, `loudness_momentary`, `loudness_momentary_windowed`, `PeakMeter`, `RmsMeter` |
| Noise reduction | `estimate_noise_profile`, `spectral_subtraction`, `wiener_filter`, `frequency_domain_noise_gate` |
| Dither | `apply_tpdf_dither`, `apply_noise_shaped_dither` |
| `dsp::spectral` | `stft`, `melspectrogram`, `mfcc`, `chromagram`(`_normalized`), `spectral_centroid`/`_bandwidth`/`_rolloff`/`_flatness`/`_flux`/`_contrast`/`_crest_factor`/`_entropy`, `harmonic_ratio`, `zero_crossing_rate`, `stft_multichannel` + `Complex`, `StftOutput`, `WindowFn` |

## Feature Flags

| Feature | Default | Pure Rust | Description |
|---------|---------|-----------|-------------|
| `pure` | **yes** | Yes | Enables `oxiaudio-decode`, `oxiaudio-encode`, `oxiaudio-dsp` + rayon — the full Pure-Rust pipeline (`decode_*`, `encode_*`, `dsp::*`, transcode) |
| `serde` | no | Yes | Enables `serde` on `oxiaudio-core` types |

## Re-exports at the crate root (from `oxiaudio-core`)

Always available (no feature required):

- Buffer + traits: `AudioBuffer`, `AudioBufferLayout`, `SampleFormat`, `ChannelLayout`, `AudioFormat`, `AudioMetadata`, `Timestamp`, `AudioClock`
- Pipeline traits: `AudioDecoder`, `AudioEncoder`, `AudioFilter`, `AudioNode`, `AudioPipeline`, `AudioSink`, `AudioSource`, `ParallelBranchNode`
- Channels: `ChannelId`, `ChannelMap`, `downmix_51_to_stereo`, `downmix_to_mono`, `upmix_mono_to_stereo`, `from_planar`, `to_planar`
- IPC + ring: `AudioRingBuffer`, `OverflowPolicy`, `serialize_audio_buffer_f32`, `deserialize_audio_buffer_f32`, `to_ipc_bytes`, `from_ipc_bytes`
- Error: `OxiAudioError`

## Error Variants

All fallible APIs return [`OxiAudioError`](../oxiaudio-core):

| Variant | Description |
|---------|-------------|
| `Io(std::io::Error)` | File open / create / read / write failure |
| `Decode(String)` | Decode failure (no track, bad codec params, corrupt data) |
| `Encode(String)` | Encoder failure |
| `UnsupportedFormat(String)` | Unknown / unsupported container or extension |
| `InvalidChannelLayout(String)` | Channel layout invalid for the operation |
| `InvalidSampleRate(String)` | Sample rate invalid for the operation |
| `BufferOverflow(String)` | Ring-buffer / internal overflow |
| `BufferUnderflow(String)` | Ring-buffer / internal underflow |

## Version

```rust
let v: &str = env!("CARGO_PKG_VERSION"); // crate version
```

## Cross-references

- [`oxiaudio-core`](../oxiaudio-core) — shared types and traits
- [`oxiaudio-decode`](../oxiaudio-decode) — decoders
- [`oxiaudio-encode`](../oxiaudio-encode) — Pure-Rust encoders
- [`oxiaudio-dsp`](../oxiaudio-dsp) — DSP, effects, analysis
- [`oxiaudio-encode-mp3-lame`](../oxiaudio-encode-mp3-lame) — LAME MP3 adapter (LGPL, opt-in)

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
