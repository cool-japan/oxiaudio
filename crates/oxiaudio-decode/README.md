# oxiaudio-decode — Symphonia-backed audio decoder for OxiAudio

[![Crates.io](https://img.shields.io/crates/v/oxiaudio-decode.svg)](https://crates.io/crates/oxiaudio-decode)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`oxiaudio-decode` is the decode layer of OxiAudio. Its primary path wraps [`symphonia`](https://crates.io/crates/symphonia) to decode every container/codec enabled by the workspace feature set (WAV, FLAC, MP3, Vorbis/OGG, AAC, ALAC, ISO-MP4/M4A, raw PCM) into a single interleaved `oxiaudio_core::AudioBuffer<f32>`. Alongside Symphonia it ships a collection of **native, pure-Rust** readers and parsers — AIFF/AIFF-C, Sun/NeXT `.au`, Opus (RFC 7845), Musepack SV7/SV8, WavPack, a standalone AAC/ADTS decoder, a Standard MIDI File parser with a built-in synthesizer, plus format detection, ReplayGain, gapless (LAME), cue sheets, and artwork extraction.

The crate is `#![deny(unsafe_code)]`; the only `unsafe` lives behind the optional `mmap` feature for `memmap2::Mmap::map`. With default features it is **100% Pure Rust**.

## Installation

```toml
[dependencies]
oxiaudio-decode = "0.2.0"

# With memory-mapped file decoding (decode_file_mmap):
oxiaudio-decode = { version = "0.2.0", features = ["mmap"] }

# With the pure-Rust OGG Opus decoder:
oxiaudio-decode = { version = "0.2.0", features = ["opus"] }
```

## Quick Start

```rust,no_run
use std::path::Path;

// Decode an entire file to an interleaved f32 buffer.
let buf = oxiaudio_decode::decode_file(Path::new("song.flac"))?;
println!("{} Hz, {} frames", buf.sample_rate, buf.frame_count());

// Decode together with embedded tags (ID3v2/Vorbis/etc., with ID3v1 fallback).
let (audio, meta) = oxiaudio_decode::decode_file_with_metadata(Path::new("song.mp3"))?;
println!("title = {:?}, duration = {:?}s", meta.title, meta.duration_secs);

// Probe the format without decoding any audio.
let fmt = oxiaudio_decode::detect_format_from_path(Path::new("clip.wav"))?;
println!("channels = {}", fmt.channels);
# Ok::<(), oxiaudio_core::OxiAudioError>(())
```

## API Overview

### Symphonia decode pipeline (crate root)

| Function | Description |
|----------|-------------|
| `decode_file(path)` | Decode a file to `AudioBuffer<f32>` |
| `decode_reader(reader)` | Decode from any `Read + Seek + Send + Sync + 'static` |
| `decode_file_with_options(path, opts)` | Decode with a configurable corrupt-packet policy |
| `decode_tolerant(path)` | Decode as much as possible; never errors (empty buffer on failure) |
| `decode_with_metadata(src)` | Decode + extract `AudioMetadata` from a reader |
| `decode_file_with_metadata(path)` | Decode + metadata with ID3v1 tail fallback |
| `decode_file_with_replaygain(path)` | Decode and apply embedded ReplayGain track gain |
| `decode_to_i16(path)` | Decode to `AudioBuffer<i16>` |
| `decode_to_i32(path)` | Decode to `AudioBuffer<i32>` (full 32-bit range) |
| `decode_to_f64(path)` | Decode to `AudioBuffer<f64>` |
| `detect_format(src)` | Probe `AudioFormat` from a reader |
| `detect_format_from_path(path)` | Probe `AudioFormat` using a file-extension hint |
| `extract_lyrics(path)` | Extract ID3v2 USLT (or `LYRIC*`) lyrics, if present |
| `parse_id3v1(path)` | Parse the trailing 128-byte ID3v1 tag, if present |
| `decode_file_mmap(path)` | *(feature `mmap`)* Decode via a memory-mapped file |

| Type | Description |
|------|-------------|
| `SymphoniaDecoder` | Zero-sized unit type implementing `oxiaudio_core::AudioDecoder` |
| `DecodeOptions` | Options struct; field `on_corrupt_packet: CorruptPacketPolicy` |
| `CorruptPacketPolicy` | `Fail` (default) or `Skip` (skip + `log::warn!`) |
| `MediaSourceWrapper<R>` | Adapts any `Read + Seek + Send + Sync` to Symphonia's `MediaSource` |

`oxiaudio_core::AudioFormat` is re-exported at the crate root for convenience.

### Streaming decode

Chunked, low-memory decoding that yields fixed-size blocks from an internal FIFO.

| Item | Description |
|------|-------------|
| `StreamingDecoder` | Pull-based block decoder; implements `Iterator` and `oxiaudio_core::AudioSource` |
| `StreamingDecoder::open(path)` | Open with the default block size |
| `StreamingDecoder::open_with_block_size(path, frames)` | Open with an explicit block size |
| `StreamingDecoder::new(...)` | Construct from a reader + format/options |
| `decode_next(...)` / `next_block()` | Decode the next block |
| `format()` / `format_info()` | Inspect the resolved `AudioFormat` |
| `metadata()` / `metadata_owned()` | Access decoded `AudioMetadata` |
| `seek(frame_offset)` / `seek_to_time(secs)` / `time_seek(secs)` | Seek by frames or time |
| `skip_frames(frames)` | Skip forward, returning frames actually skipped |
| `remaining_frames()` | Remaining frame estimate, if known |
| `StreamingDecoderBuilder::new(path)` | Builder: `block_size`, `skip_corrupt`, `track_index`, `build` |

### AIFF / AIFF-C (`aiff`)

| Item | Description |
|------|-------------|
| `decode_aiff(reader)` / `decode_aiff_file(path)` | Decode AIFF PCM |
| `decode_aiff_with_metadata(...)` / `decode_aiff_reader_with_metadata(...)` | Decode + metadata |
| `decode_aiffc_compressed(reader)` / `decode_aiffc_compressed_file(path)` | Decode AIFF-C (µ-law / A-law) |
| `ulaw_to_linear(byte)` / `alaw_to_linear(byte)` | G.711 companding → linear `i16` |

### Sun / NeXT `.au` (`au`)

| Function | Description |
|----------|-------------|
| `decode_au(reader)` / `decode_au_file(path)` | Decode `.snd` / `.au` audio |

### AAC / ADTS (`aac_decoder`)

| Item | Description |
|------|-------------|
| `AdtsFrame<'a>` | Parsed ADTS frame: `channels`, `sample_rate`, `pcm_samples`, `payload: &[u8]` |
| `parse_adts_header(data)` | Parse a single ADTS header (borrowing payload) |
| `AacDecoder` | AAC-LC decoder: `new()`, `decode_frame(data)`, `sample_rate()`, `channels()` |
| `decode_aac(data)` | Decode a full ADTS stream to `AudioBuffer<f32>` |

### Opus (`opus`)

| Item | Description |
|------|-------------|
| `OpusHead` | Parsed `OpusHead`: `channels`, `pre_skip`, `input_sample_rate`, `output_gain`, `mapping_family` |
| `parse_opus_head(packet)` | Parse the `OpusHead` identification packet |
| `decode_opus_file(path)` / `decode_opus_reader(reader)` | Decode an OGG Opus stream |
| `OpusDecoder` | *(feature `opus`)* `new(sample_rate, channels)`, `from_opus_head(head, rate)`, `decode_packet(data)` |

> Without the `opus` feature, `decode_opus_file` / `decode_opus_reader` are present but return an error indicating the decoder is unavailable.

### Musepack (`musepack`)

| Item | Description |
|------|-------------|
| `MpcVersion` | `Sv7` (frame-based) or `Sv8` (packet-based) |
| `MusepackDecoder` | `new(data)` — detects SV7/SV8 and decodes |
| `decode_musepack(data)` / `decode_musepack_file(path)` | Decode Musepack to `AudioBuffer<f32>` |
| `MUSEPACK_MAGIC_SV7` / `MUSEPACK_MAGIC_SV8` | Magic-byte constants |

### WavPack (`wavpack`)

| Item | Description |
|------|-------------|
| `decode_wavpack(data)` / `decode_wavpack_file(path)` | Decode WavPack to `AudioBuffer<f32>` |
| `WAVPACK_MAGIC` | `wvpk` magic-byte constant |

### Raw PCM (`raw`)

| Item | Description |
|------|-------------|
| `RawPcmConfig` | `sample_rate`, `channels`, `format`, `little_endian`, `skip_bytes` |
| `decode_raw_pcm(reader, &config)` | Decode headerless PCM from a reader |
| `decode_raw_pcm_file(path, &config)` | Decode headerless PCM from a file |

> Supports `U8`, `I16`, `I32`, `F32`. `I24` and `F64` are not supported for raw PCM.

### Format detection (`detect`)

| Item | Description |
|------|-------------|
| `AudioFormatHint` | `Wav`, `Flac`, `Mp3`, `Ogg`, `Aiff`, `Au`, `Musepack`, `WavPack`, `Aac`, `M4a` |
| `detect_format_from_bytes(header)` | Magic-byte sniffing from a header slice (`Option`) |
| `detect_format_file(path)` | Detect a file's format from its leading bytes |

### MIDI parsing (`midi`)

| Item | Description |
|------|-------------|
| `MidiFile` | `format: SmfFormat`, `ticks_per_quarter: u16`, `tracks: Vec<MidiTrack>` |
| `MidiFile::from_bytes(data)` / `from_path(path)` | Parse a Standard MIDI File |
| `SmfFormat` | `SingleTrack`, `MultiTrack`, `Patterns` |
| `MidiTrack` | `events: Vec<TimedEvent>` |
| `TimedEvent` | `tick: u64`, `event: TrackEvent` |
| `TrackEvent` | `Midi(MidiEvent)`, `Meta(MetaEvent)`, `SysEx(Vec<u8>)` |
| `MidiEvent` | Channel messages: `NoteOff`, `NoteOn`, `PolyKeyPressure`, `ControlChange`, `ProgramChange`, `ChannelPressure`, `PitchBend` |
| `MetaEvent` | `Tempo`, `TimeSignature`, `KeySignature`, `TrackName`, `InstrumentName`, `Lyric`, `Marker`, `CuePoint`, `EndOfTrack`, `Other` |

### MIDI synthesis (`midi_synth`)

| Item | Description |
|------|-------------|
| `synthesize_midi(midi, sample_rate, &config)` | Render a `MidiFile` to `AudioBuffer<f32>` |
| `synthesize_midi_default(midi, sample_rate)` | Render with default `SynthConfig` |
| `SynthConfig` | `waveform`, `adsr`, `polyphony` (32), `master_volume` (0.5) |
| `Waveform` | `Sine` (default), `Square`, `Sawtooth`, `Triangle` |
| `Adsr` | `attack_ms`, `decay_ms`, `sustain`, `release_ms` |
| `midi_note_to_hz(note)` | MIDI note number → frequency in Hz |

### Metadata helpers

| Item | Description |
|------|-------------|
| `parse_replaygain(path)` → `ReplayGainMetadata` | Track/album gain + peak (case-insensitive tags) |
| `ReplayGainMetadata` | `track_gain_db`, `track_peak`, `album_gain_db`, `album_peak` |
| `parse_gapless_info(mp3_bytes)` → `Option<GaplessInfo>` | LAME encoder delay/padding header |
| `GaplessInfo` | `encoder_delay`, `encoder_padding`, `total_samples` |
| `apply_gapless_trim(...)` | Trim encoder delay/padding from a decoded buffer |
| `extract_album_art(path)` → `Option<AlbumArtwork>` | Extract embedded cover art |
| `AlbumArtwork` | `data`, `mime_type`, `picture_type`, `description` + `is_jpeg()`, `is_png()`, `extension()` |
| `parse_flac_cue_sheet(path)` → `Vec<FlacCuePoint>` | FLAC CUESHEET metadata block |
| `FlacCuePoint` | `track_number`, `offset_samples`, `isrc`, `is_audio` |
| `parse_wav_cues(path)` / `parse_wav_cues_reader(reader)` → `Vec<WavCuePoint>` | WAV `cue ` chunk points |
| `WavCuePoint` | `id`, `position`, `label` |

### Low-level OGG (`ogg_reader`)

| Item | Description |
|------|-------------|
| `OggReader<R>` | Page-aware OGG demuxer: `new(reader)`, `read_packet()` |
| `ogg_crc32(data)` | OGG page CRC-32 |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `mmap` | off | Enables `decode_file_mmap` via `memmap2` (one targeted `unsafe` block) |
| `opus` | off | Enables the pure-Rust `OpusDecoder` backed by `opus-decoder` |

## Errors

All fallible functions return [`oxiaudio_core::OxiAudioError`]. The variants most commonly produced here are:

| Variant | When |
|---------|------|
| `Io(std::io::Error)` | File cannot be opened or read |
| `Decode(String)` | Probing fails, no audio track, codec setup/decode fails |
| `UnsupportedFormat(String)` | Container/codec or sample format unsupported (e.g. raw `I24`) |

## Related crates

| Crate | Role |
|-------|------|
| `oxiaudio-core` | `AudioBuffer`, `AudioFormat`, `AudioMetadata`, traits, errors |
| `oxiaudio` | Top-level façade re-exporting the ecosystem |
| `oxiaudio-encode` | Encoders (WAV, FLAC, …) |
| `oxiaudio-dsp` | Resampling, gain, filters, spectral analysis, effects |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
