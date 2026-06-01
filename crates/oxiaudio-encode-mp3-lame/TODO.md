# oxiaudio-encode-mp3-lame TODO

## Status
BOUNDED_FFI MP3 encoder via LAME (mp3lame-encoder 0.2.4). Supports CBR (64/128/192/320 kbps), VBR (quality V0-V9 via MTRH mode), all MPEG channel modes (stereo/joint-stereo/dual-channel/mono). Includes hand-rolled ID3v2.3 tag writer (TIT2, TPE1, TALB, TRCK, TYER). Both one-shot `LameMp3Encoder` and streaming `LameMp3StreamEncoder` with ID3 prepend. Feature-gated behind `mp3-encode-lame` (LGPL, links libmp3lame). M0-M4 complete. Approximately 492 SLOC including tests.

## Core Implementation

### VBR Quality Profiles and Bitrate Support
- [x] Add named VBR presets: `VbrPreset::Voice` (V6, ~115 kbps mono), `VbrPreset::Podcast` (V4, ~165 kbps mono), `VbrPreset::Music` (V2, ~190 kbps), `VbrPreset::HighFidelity` (V0, ~245 kbps), `VbrPreset::Archival` (V0, max bitrate 320) (~30 SLOC)
- [x] Expose ABR (Average Bitrate) mode via `LameMode::Abr { target_kbps: u32 }` using LAME's ABR VBR mode with target bitrate constraint (~20 SLOC)
- [x] Support all 14 LAME bitrate values (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 kbps) instead of current 4 (~15 SLOC refactor to_bitrate)
- [x] Add `LameMode::ForcedMono` that sums stereo input to mono before encoding, avoiding dual-mono bitrate waste (~20 SLOC)
- [x] Expose mid/side stereo threshold configuration for joint stereo mode quality tuning (~15 SLOC)

### ReplayGain
- [x] Compute ReplayGain during encoding using LAME's built-in analysis: extract track gain (dB) and peak sample amplitude after flush (~30 SLOC)
- [x] Write ReplayGain values into the Xing/LAME info tag header fields (radio gain, audiophile gain, peak signal) (~20 SLOC)
- [x] Embed ReplayGain as ID3v2 TXXX frames: REPLAYGAIN_TRACK_GAIN (e.g., "-6.50 dB"), REPLAYGAIN_TRACK_PEAK (e.g., "0.988") (~15 SLOC)
- [x] Embed ReplayGain as ID3v2 TXXX frames: REPLAYGAIN_ALBUM_GAIN, REPLAYGAIN_ALBUM_PEAK (require multi-file analysis context) (~15 SLOC)

### Gapless Playback Info
- [x] Extract encoder delay (LAME internal 576-sample + warmup) and padding from encoder state after final flush (~15 SLOC)
- [x] Write delay/padding values into the Xing/LAME info tag for gapless-aware decoders (foobar2000, mpd, ffmpeg) (~20 SLOC)
- [x] Emit iTunSMPB atom equivalent as an ID3v2 COMM frame for Apple decoder compatibility (encoder delay, zero padding, total samples) (~15 SLOC)

### ID3v2 Tag Enhancements
- [x] Upgrade to ID3v2.4 with UTF-8 text encoding (encoding byte 0x03) instead of current v2.3 ISO-8859-1 only (~40 SLOC)
- [x] Add APIC frame for album art embedding: type byte (0x03 = front cover), MIME type string, optional description, image data (JPEG or PNG) (~50 SLOC)
- [x] Add COMM frame for comments with language code and content description (~15 SLOC)
- [x] Add TCON frame for genre (both free-text and ID3v1 genre index in parentheses, e.g., "(21)") (~20 SLOC)
- [x] Add TCOM frame for composer (~10 SLOC)
- [x] Add TDRC frame for recording date in ISO 8601 format (ID3v2.4) or TYER+TDAT combination (ID3v2.3) (~10 SLOC)
- [x] Add USLT frame for unsynchronized lyrics with language code and content descriptor (~20 SLOC)
- [x] Support ID3v2 extended header with CRC-32 checksum for tag integrity verification (~20 SLOC)
- [x] Support ID3v2 footer (10-byte copy of header at tag end) for efficient append-seeking (~10 SLOC)
- [x] Add TXXX user-defined text frame support for arbitrary key-value metadata (~15 SLOC)

### Streaming Encoder Improvements
- [x] Add `LameMp3StreamEncoder::frames_encoded(&self) -> u64` tracking total PCM frames fed to encoder (~10 SLOC)
- [x] Add `LameMp3StreamEncoder::bytes_written(&self) -> u64` tracking total MP3 bytes emitted to dst (~10 SLOC)
- [x] Add `LameMp3StreamEncoder::estimated_bitrate_kbps(&self) -> Option<u32>` for VBR streams: bytes_written * 8 / duration_secs (~15 SLOC)
- [x] Add `LameMp3StreamEncoder::elapsed_secs(&self) -> f64` computed from frames_encoded / sample_rate (~5 SLOC)

## API Improvements
- [x] Add `LameMp3Encoder::builder()` fluent pattern: `LameMp3Encoder::builder().bitrate(192).mode(JointStereo).tags(tags).build()` (~30 SLOC)
- [x] Add `encode_file(buf: &AudioBuffer<f32>, path: impl AsRef<Path>)` convenience creating file + BufWriter internally (~15 SLOC)
- [x] Add `encode_to_vec(buf: &AudioBuffer<f32>) -> Result<Vec<u8>, OxiAudioError>` for in-memory MP3 generation (~15 SLOC)
- [x] Add `Mp3Tags::builder()` with chained setters: `.title("X").artist("Y").track(3).year(2026).build()` (~20 SLOC)
- [x] Document LGPL compliance requirements: static vs dynamic linking implications, relinking obligation, object file distribution (~docs only)
  — Audit finding: `mp3lame-sys 0.1.11` build.rs builds lame-3.100 from bundled C source on all platforms using `.disable_shared().enable_static()` on Unix (autotools) and direct `cc::Build` on Windows; both emit `cargo:rustc-link-lib=static=mp3lame`. Result: libmp3lame is **statically linked on all platforms**. LGPL-2.1 §6 compliance for static linking requires one of: (a) accompany the binary with machine-readable object files for relinking; (b) provide a written offer (valid ≥3 years) to supply object files; or (c) use a shared library mechanism allowing users to relink. Recommended approach: because mp3lame-sys ships the full lame-3.100 source in its crate package, downstream users can rebuild the library from source — document this as the relinking facility and note the lame-3.100 source path in the crate. The `mp3-encode-lame` feature gate already satisfies the "opt-in notice" requirement; add a comment in the crate doc warning that distributors of compiled binaries must ensure LGPL §6 compliance (e.g., redistribute lame source or provide object files).

## Testing
- [x] Roundtrip test: encode MP3 at all supported CBR bitrates -> decode with symphonia -> verify sample rate and non-silence (~35 SLOC)
- [x] Test all 14 CBR bitrate values encode without error (~15 SLOC)
- [x] Test VBR quality levels 0, 2, 5, 9 produce valid MP3 files with monotonically decreasing file sizes (~20 SLOC)
- [x] Test ABR mode at 128 kbps target produces valid MP3 with average bitrate within 10% of target (~15 SLOC)
- [x] Test streaming encoder produces byte-identical output to one-shot encoder for same input and configuration (~25 SLOC)
- [x] Test ID3v2 tag with non-ASCII characters in ISO-8859-1 range (accented Latin: a-umlaut, e-acute) (~15 SLOC)
- [x] Test ID3v2.4 UTF-8 encoding with CJK characters in title field (~15 SLOC)
- [x] Test ID3v2 tag with all fields populated simultaneously (title, artist, album, track, year, genre, comment) (~10 SLOC)
- [x] Test APIC frame: embed 1x1 JPEG, verify ID3 frame structure and JPEG magic bytes at expected offset (~20 SLOC)
- [x] Test encoding of very short buffers (<576 samples, less than one LAME granule) (~15 SLOC)
- [x] Test encoding of very long buffers (>10 minutes at 44.1 kHz stereo) for memory stability (~10 SLOC)
- [x] Benchmark MP3 encoding speed: CBR 128 vs CBR 320 vs VBR V2 for 10s stereo 44.1 kHz (~20 SLOC)

## Performance
- [x] Optimize f32->i16 sample conversion loop: use `chunks_exact(8)` for SIMD auto-vectorization (~10 SLOC)
- [x] Reduce intermediate allocation in stereo deinterleave: pre-allocate left/right buffers once per encode call and reuse (~15 SLOC)
- [x] Tune `mp3_out` capacity estimate: use `max_required_buffer_size(n_frames) + 7200` without over-allocation for short buffers (~10 SLOC)
- [x] For streaming encoder, accumulate small chunks (<1152 samples) to amortize LAME frame overhead and reduce per-chunk MP3 header waste (~20 SLOC)

## Integration
- [x] Verify LGPL compliance: ensure libmp3lame is dynamically linked or that relinking is possible on all target platforms (~audit task)
  — Audit result: libmp3lame is **statically linked on all platforms** (Linux, macOS, Windows, Android, iOS) via `mp3lame-sys 0.1.11` build.rs (autotools on Unix: `--disable-shared --enable-static`; cc-compiled on Windows). No dynamic linking path exists. LGPL-2.1 §6 compliance is achieved via the bundled source relinking facility: mp3lame-sys ships the complete lame-3.100 source tree in its crate package, so any distributor of a binary built with this crate can point end users to rebuild libmp3lame. Action for distributors: include a NOTICE referencing the lame-3.100 source in mp3lame-sys (available from crates.io) as the relinking facility per LGPL §6(b). No code changes required; the relinking facility already exists in the dependency graph.
- [x] Coordinate with oxiaudio-encode: verify `mp3` feature flag re-export works, add cross-crate integration tests (~15 SLOC)
- [x] Integration test pipeline: decode MP3 -> DSP pitch shift -> re-encode MP3, verify output is valid (~25 SLOC)
- [x] Document live MP3 streaming pattern: oxisound capture -> streaming encode_chunk -> network send (~example documentation)
  — Architecture: `LameMp3StreamEncoder` wraps any `impl Write` (e.g. `TcpStream`, `BufWriter<File>`, or an in-memory buffer). Streaming usage pattern:
  ```
  // Pattern: create LameMp3StreamEncoder with Write impl (e.g. TcpStream or File)
  // 1. let mut enc = LameMp3StreamEncoder::new(config, dst);
  // 2. loop { let chunk = capture_audio(); enc.encode_chunk(&chunk)?; }
  // 3. enc.finalize()?;
  // Encoder tracks bytes_written/frames_encoded/elapsed_secs for progress reporting.
  ```
  The encoder's `frames_encoded()`, `bytes_written()`, `estimated_bitrate_kbps()`, and `elapsed_secs()` accessors provide live progress metrics for dashboards or network throttling. The destination `Write` impl is flushed on each `encode_chunk` call; for network destinations use a `BufWriter` to reduce syscall frequency. No oxisound API dependency required — any `AudioBuffer<f32>` source feeds directly into `encode_chunk`.
- [x] Track Pure Rust MP3 encoder ecosystem: when a competitive pure Rust MP3 encoder emerges, plan deprecation of this BOUNDED_FFI crate (~tracking task)
  — Ecosystem survey as of 2026-05: No competitive pure-Rust MP3 encoder exists in crates.io. The closest candidate is `minimp3` (decoder only). `mp3lame-encoder` remains the best option for MP3 encoding. This item should be re-evaluated periodically; when a pure-Rust encoder achieves feature parity (CBR/VBR/ID3v2), begin deprecation planning.
