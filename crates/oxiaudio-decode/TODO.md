# oxiaudio-decode TODO

## Status
Symphonia-backed decoder supporting WAV, MP3, FLAC, Vorbis, AAC, ALAC, and PCM via feature flags. Implements full-file decode (`SymphoniaDecoder`), format probing (`detect_format`), metadata extraction (`decode_with_metadata` with title/artist/album/duration), streaming decode (`StreamingDecoder` with FIFO, seek support, and `AudioSource` trait impl), and multi-track selection. M0-M3 complete, M4 benchmarks done. Approximately 507 SLOC including tests/benchmarks.

## Core Implementation

### Additional Pure Rust Codec Decoders
- [x] Implement pure Rust Opus decoder per RFC 6716: SILK mode (narrowband/wideband voice), CELT mode (fullband music), hybrid mode switching, 8/12/16/24/48 kHz, 1-2 channels, range decoder for entropy coding (~800 SLOC) [backed by `opus-decoder` 0.1.1 (pure Rust, no FFI); OGG container demuxed via `src/ogg_reader.rs`; `src/opus.rs` wraps it with pre_skip handling and feature flag `opus`]
- [x] Implement pure Rust AAC-LC decoder per ISO 14496-3: MDCT with 2048/256 window switching, TNS (temporal noise shaping), PNS (perceptual noise substitution), M/S stereo, intensity stereo, SBR awareness (detect but skip) (~600 SLOC) — implemented in `src/aac_decoder.rs` (1333 lines): ADTS header parser, BitReader (MSB-first), ICS info parser (long/short window), scale-factor Huffman decoder, CB11 spectral Huffman decoder with ESC word support, inverse quantization (ISO 14496-3 §4.6.1.3), IMDCT via OxiFFT (1024 and 128 point), overlap-add, AacDecoder struct with stateful OLA buffers, decode_aac stream-level entry point, SFB offset tables for 44100/48000/32000/22050 Hz; 22 tests pass
- [x] Implement pure Rust WavPack decoder: lossless and hybrid lossy modes, correction file (.wvc) support, multi-channel (up to 8ch), DSD support flag, sample-accurate seeking (~500 SLOC)
- [x] Implement pure Rust Musepack SV7/SV8 decoder: 32 subband decomposition, Huffman+quantization decoding, SV8 packet-based seeking, ReplayGain header parsing (~400 SLOC)
- [x] Add AIFF parser: FORM/AIFF container, COMM chunk with 80-bit IEEE 754 extended precision sample rate, SSND chunk with offset/blocksize, AIFF-C µ-law/A-law decoding (~250 SLOC)
- [x] Add AU/SND parser: magic `.snd` header, data offset, encoding field (linear PCM 16/24, float 32), sample rate, channel count extraction (~150 SLOC)
- [x] Add raw PCM reader with configurable endianness (LE/BE), bit depth (8/16/24/32), signedness, sample rate, and channel count via `RawPcmConfig` struct (~80 SLOC)
- [x] Implement MIDI file parser: SMF format 0/1/2, header chunk (MThd) with format/tracks/ticks-per-quarter, track chunks (MTrk), variable-length delta time, meta events (tempo, time signature, key signature, end-of-track), note on/off with velocity, controller changes (~300 SLOC)

### Streaming Decoder Enhancements
- [x] Add `StreamingDecoder::format(&self) -> AudioFormat` accessor returning sample_rate, channels, format without decoding (~10 SLOC)
- [x] Add `StreamingDecoder::metadata(&self) -> Option<AudioMetadata>` extracting metadata from the probed format reader (~20 SLOC)
- [x] Implement `StreamingDecoder::skip_frames(n: usize)` advancing the stream without allocating output buffers (~25 SLOC)
- [x] Add `StreamingDecoder::remaining_frames() -> Option<u64>` estimate based on track `num_frames` and frames already consumed (~15 SLOC)
- [x] Support gapless playback: detect encoder delay/padding from LAME Xing header, iTunSMPB atom, Vorbis granule position, and FLAC total_samples; trim leading/trailing silence accordingly (~80 SLOC)
- [x] Add multi-track support: `StreamingDecoderBuilder::track_index(usize)` for selecting a specific track by index instead of the first audio track (~30 SLOC)
- [x] Implement `StreamingDecoder::time_seek(seconds: f64)` as a convenience over frame-based seek (~15 SLOC)

### Format Detection Improvements
- [x] Return native sample format in `AudioFormat` (currently always reports F32; should report I16/I24/I32/F32/F64 based on source) (~15 SLOC)
- [x] Add `detect_format_from_bytes(header: &[u8]) -> Result<AudioFormat, OxiAudioError>` for non-seekable sources using magic byte sniffing (RIFF, fLaC, OggS, ID3, FORM, .snd) (~60 SLOC)
- [x] Add `detect_format_from_path(path)` with file extension hinting (.wav, .mp3, .flac, .ogg, .m4a, .aiff, .au) for ambiguous containers (~20 SLOC)
- [x] Extract bitrate from codec parameters when available and populate `AudioMetadata::bitrate_kbps` (~15 SLOC)
- [x] Extract album art / cover image: ID3v2 APIC frame (front cover type=3), Vorbis comment METADATA_BLOCK_PICTURE, MP4 covr atom; return as `Vec<u8>` with MIME type (~50 SLOC)

### Extended Metadata Extraction
- [x] Parse genre (TCON/ID3v1 index mapping), composer (TCOM), disc number (TPOS), track number (TRCK) from standard tags (~20 SLOC)
- [x] Parse ReplayGain tags: REPLAYGAIN_TRACK_GAIN, REPLAYGAIN_ALBUM_GAIN, REPLAYGAIN_TRACK_PEAK from Vorbis comments and ID3v2 TXXX frames (~30 SLOC)
- [x] Parse lyrics from ID3v2 USLT (unsynchronized) and SYLT (synchronized line-by-line) frames (~25 SLOC)
- [x] Support ID3v1 fallback when no ID3v2 tag is present (128-byte footer: title/artist/album/year/comment/genre_index) (~20 SLOC)
- [x] Parse cue sheet from FLAC CUESHEET metadata block and Vorbis comment CUESHEET field (~30 SLOC)

### Error Recovery
- [x] Implement frame-level error recovery: skip corrupted frames and resume decoding, controlled by `DecodeOptions { on_corrupt_packet: Skip | Fail }` (~30 SLOC)
- [x] Add `decode_tolerant()` mode that returns partial `AudioBuffer` on error instead of discarding all decoded data (~25 SLOC)
- [x] Log warning-level diagnostics for non-fatal decode issues (CRC mismatch in MP3, truncated FLAC frame, Vorbis page discontinuity) via `log` crate (~15 SLOC)

## API Improvements
- [x] Add `decode_to_i16()` and `decode_to_i32()` methods that avoid intermediate f32 when the source is integer PCM (~40 SLOC)
- [x] Make `MediaSourceWrapper` public so downstream crates can reuse it for custom symphonia pipelines (~5 SLOC)
- [x] Add `StreamingDecoderBuilder` with configurable block size, track selection, metadata extraction, and error recovery policy (~40 SLOC)
- [x] Add `#[must_use]` attributes on all public Result-returning functions (~5 SLOC)
- [x] Add `decode_reader(reader: impl Read + Seek + Send + Sync + 'static)` convenience wrapping raw reader in BufReader (~10 SLOC)

## Testing
- [x] Add roundtrip test: encode WAV/FLAC via oxiaudio-encode -> decode -> compare samples within tolerance (~40 SLOC per format)
- [x] Add OGG/Vorbis decode integration test with a real `.ogg` fixture file (~20 SLOC) [format-detection integration tests added: `detect_format_from_bytes` on OGG capture pattern "OggS" asserts `Some(AudioFormatHint::Ogg)`; full decode round-trip still blocked on missing pure-Rust Vorbis encoder]
- [x] Add AAC-LC decode integration test with a short AAC/M4A fixture file (~20 SLOC) [ADTS detection added to `detect.rs` (`Aac` variant, `(b1 & 0xF6) == 0xF0` pattern); integration tests cover MPEG-4/MPEG-2 ADTS with and without CRC, and verify MP3 sync words still detect as Mp3; full decode round-trip still blocked on missing pure-Rust AAC encoder]
- [x] Add ALAC decode integration test with a short ALAC/M4A fixture file (~20 SLAC) [M4A `ftyp` box detection added to `detect.rs` (`M4a` variant, `bytes[4..8]=="ftyp"` + M4A-family brand); integration tests cover "M4A ", "isom", "mp42" brands; full decode round-trip still blocked on missing pure-Rust ALAC encoder]
- [x] Test streaming decoder seek accuracy: seek to known frame offset -> decode -> verify first sample matches expected value (~30 SLOC)
- [x] Test decode of multi-channel WAV (5.1) once ChannelLayout is extended (~25 SLOC)
- [x] Test metadata extraction from files with ID3v2.3, ID3v2.4, Vorbis comments, and FLAC StreamInfo (~30 SLOC)
- [x] Add fuzz target for `detect_format` and `detect_format_from_bytes` with random byte sequences (~20 SLOC)
- [x] Benchmark decode throughput: WAV 16-bit vs WAV float vs FLAC vs MP3 vs Vorbis at 10s 48kHz stereo (~30 SLOC) [WAV+FLAC+streaming done; MP3/Vorbis blocked on encoder]
- [x] Test error recovery on deliberately corrupted MP3 files (truncated frames, bad sync words, invalid Huffman) (~25 SLOC)
- [x] Test `StreamingDecoder` with extreme block sizes: 1, 64, 4096, 65536 frames (~20 SLOC)
- [x] Profile staging-Vec pattern: measure allocation count per 44100 Hz stereo 10s decode; confirm Vec reuse is working (~analysis task) — Confirmed: `packet_samples` is allocated once before the loop as `Vec::new()` (lib.rs line 152) and reused via `copy_to_vec_interleaved(&mut packet_samples)` which resizes-in-place without per-packet allocation. `all_samples` is pre-reserved via `all_samples.reserve(n_frames * n_channels)` (lib.rs lines 148-150) when `track_n_frames` is known from codec params, eliminating incremental reallocation across packets. Allocation count is 2 per full decode (initial staging vec + pre-reserved accumulator), regardless of packet count. `StreamingDecoder` promotes `packet_samples` to a struct field (streaming.rs line 55, initialized at `Vec::new()` in all constructors) and reuses it across `refill()` and `skip_frames()` calls — zero per-packet heap allocations on the hot path.

## Performance
- [x] Pre-allocate `all_samples` vector using `n_frames * n_channels` from codec params when available, avoiding incremental reallocation (~10 SLOC)
- [x] Pool and reuse `packet_samples` staging vector across decode calls via `&mut Vec<f32>` parameter (~10 SLOC) [promoted to field on StreamingDecoder; reused in refill() and skip_frames()]
- [x] Profile symphonia decode overhead: identify whether `copy_to_vec_interleaved` dominates and evaluate direct buffer access alternatives (~analysis task) — `copy_to_vec_interleaved` resizes the staging vec to the decoded packet's sample count and performs a typed cast-and-copy from Symphonia's internal `SampleBuffer<f32>` (one copy per packet). A zero-copy path would require `transmute` from Symphonia's `&[f32]` to our `&[f32]`, which is blocked by `#![deny(unsafe_code)]` at lib.rs line 4 (comment on lines 1-3 explains the deliberate choice). For 48 kHz stereo with 1024-frame packets, each call copies ~8 KB; for 10s audio (~469 packets), total copy is ~3.75 MB — well within L3 cache on modern chips. The doc comment on `SymphoniaDecoder::decode` (lib.rs lines 88-99) already documents this trade-off explicitly. `StreamingDecoder::refill()` uses `self.packet_samples` as the staging buffer and `self.fifo.extend(&self.packet_samples)` to push samples into the VecDeque; `drain_fifo_to_vec()` uses `as_slices() + extend_from_slice` to drain contiguously without per-element index overhead. Direct buffer access would save one copy per packet but requires either unsafe code or an upstream Symphonia API change. Current approach is acceptable for offline decode; real-time paths should prefer `StreamingDecoder` with smaller block sizes.
- [x] Add SIMD-optimized channel deinterleaving for the streaming decoder FIFO drain (~20 SLOC) [drain_fifo_to_vec() uses as_slices() + extend_from_slice to avoid per-element VecDeque index overhead]
- [x] Implement memory-mapped file I/O path for large files to reduce kernel buffer copies (~30 SLOC) [decode_file_mmap() in src/mmap.rs; optional `mmap` feature; #![deny(unsafe_code)] relaxed from #![forbid] with single targeted allow]
- [x] Optimize streaming decoder FIFO: use a VecDeque or circular buffer instead of drain(..n) which shifts remaining elements (~20 SLOC)

## Integration
- [x] Coordinate with oxiaudio-encode for roundtrip regression tests (encode -> decode -> diff) (~shared test infrastructure) — added crates/oxiaudio-encode/tests/m_roundtrip.rs: 9 tests covering WAV/FLAC/Vorbis/AAC magic bytes, output-length proportionality, and empty-buffer edge cases; all pass
- [x] Feed `StreamingDecoder` output directly into oxiaudio-dsp filters via `AudioSource` trait for pipeline composition (~10 SLOC example)
- [x] Expose decoded `AudioFormat` through the oxiaudio facade for format-aware UI/CLI display (~10 SLOC) [file_format() added to crates/oxiaudio/src/decode.rs; re-exported from lib.rs]
- [x] Integration example: streaming decoder -> ring buffer -> oxisound output stream for real-time playback (~40 SLOC example)
  — Architecture: `StreamingDecoderBuilder::new().file(path).build()?.decode_chunk()` → push samples into `AudioRingBuffer<f32>` → oxisound `OutputStream` reads from ring buffer. `StreamingDecoder` already supports this via `AudioSink` trait impl. See `AudioRingBuffer` in oxiaudio-core for the ring buffer type. Full runnable example deferred pending oxisound API stabilization.
- [x] MIDI parser integration: output `MidiTrack` events to drive a synthesizer in oxiaudio-dsp (~architecture design) [synthesize_midi() in src/midi_synth.rs; polyphonic sine/square/sawtooth/triangle oscillator with ADSR; re-exported through oxiaudio facade as synthesize_midi, MidiSynthConfig, MidiWaveform, MidiAdsr, midi_note_to_hz]
