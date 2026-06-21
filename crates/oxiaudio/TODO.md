# oxiaudio (facade) TODO

## Status
User-facing facade crate composing oxiaudio-decode, oxiaudio-encode, and oxiaudio-dsp behind feature flags. Provides `decode_file`, `decode_file_f64`, `encode_wav`, `encode_flac`, `encode_stream`, `detect_format`, `decode_file_with_metadata`, streaming decode (`decode_stream`, `decode_stream_with_block_size`), and DSP convenience module (`dsp::resample`, `dsp::gain`, `dsp::normalize`, `dsp::trim_silence`, `dsp::mix_to_mono`, `dsp::pitch_shift`, `dsp::split_channels`, `dsp::BiquadFilter`, `dsp::ParametricEq`, `dsp::spectral::stft`, `dsp::spectral::melspectrogram`). Feature flags: `pure` (default), `serde`. M0-M4 complete. Criterion benchmarks in benches/facade_bench.rs.

## Core Implementation

### Encoding Convenience Functions
- [x] Add `encode_opus(buf, writer, bitrate)` and `encode_opus_file(buf, path, bitrate)` re-exported from oxiaudio-encode — structural CELT skeleton (not RFC-conformant); SILK/PVQ deferred
- [x] Add `encode_vorbis(buf, path)` convenience once pure Rust Vorbis encoder lands in oxiaudio-encode (~10 SLOC) — encode_vorbis_to_file now added
- [x] Add `encode_aiff(buf, path)` convenience once AIFF writer lands in oxiaudio-encode (~10 SLOC)
- [x] Add `encode_wav_with_config(buf, path, WavBitDepth)` exposing bit depth selection (~15 SLOC)
- [x] Add `encode_flac_with_config(buf, path, compression_level: u8)` exposing compression level (~15 SLOC)

### Streaming Encode API
- [x] Add `encode_stream_wav(chunks: impl Iterator<Item = &AudioBuffer<f32>>, writer, config)` delegating to `WavStreamEncoder` (~20 SLOC) — implemented as `encode_stream` in M4
- [x] Add `encode_stream_flac(chunks: impl Iterator<Item = &AudioBuffer<f32>>, writer, compression_level)` delegating to `FlacStreamEncoder` (~20 SLOC)
- [x] Add `TranscodeStream` combining streaming decode -> optional DSP chain -> streaming encode in a single pipeline (~40 SLOC)

### High-Precision API
- [x] Add `decode_file_f64(path) -> Result<AudioBuffer<f64>, OxiAudioError>` for high-precision scientific audio analysis (~15 SLOC) — implemented in M4
- [x] Add `encode_wav_f64(buf: &AudioBuffer<f64>, path)` for double-precision WAV output (~15 SLOC)

### DSP Convenience Extensions
- [x] Add `dsp::split_channels(buf) -> Vec<AudioBuffer<f32>>` delegating to oxiaudio-dsp (~5 SLOC) — implemented in M4
- [x] Re-export `dsp::BiquadFilter` and `dsp::ParametricEq` through facade — implemented in M4
- [x] Add `dsp::resample(buf, target_rate)` already exists; add `dsp::resample_quality(buf, target_rate, quality: ResampleQuality)` with Fast/Good/Best presets (~15 SLOC)
- [x] Add `dsp::compressor(buf, threshold, ratio, attack, release)` once compressor lands in oxiaudio-dsp (~10 SLOC)
- [x] Add `dsp::reverb(buf, room_size, damping, wet)` once Freeverb lands in oxiaudio-dsp (~10 SLOC)
- [x] Add `dsp::eq(buf, bands: &[(f32, f32, f32)])` convenience for quick EQ: (frequency, q, gain_db) tuples (~15 SLOC)
- [x] Add `dsp::loudness_lufs(buf) -> f32` once EBU R128 measurement lands in oxiaudio-dsp (~5 SLOC)
- [x] Add `dsp::detect_pitch(buf) -> Vec<(f64, f32)>` once YIN/pYIN lands in oxiaudio-dsp (~5 SLOC)
- [x] Add `dsp::detect_tempo(buf) -> f32` once tempo detection lands in oxiaudio-dsp (~5 SLOC)

### Metadata API
- [x] Add `probe_metadata(path) -> Result<AudioMetadata, OxiAudioError>` that extracts metadata without decoding audio (~15 SLOC)
- [x] Add `AudioMetadata` extensions: genre, composer, track_number, disc_number, and `pub album_art: Option<Vec<u8>>` all present in oxiaudio-core AudioMetadata struct (album_art added with serde skip_serializing_if attribute)
- [x] Add `write_metadata(path, metadata: &AudioMetadata)` for writing/updating tags in existing files (~30 SLOC)

### Format Conversion Utility
- [x] Add `convert(input_path, output_path)` auto-detecting input format and encoding to output format based on file extension (~30 SLOC)
- [x] Add `convert_with_dsp(input_path, output_path, dsp_chain: DspChain)` applying DSP during conversion (~20 SLOC)

### Batch Processing
- [x] Add `decode_files(paths: &[Path]) -> Vec<Result<AudioBuffer<f32>, OxiAudioError>>` for parallel multi-file decode via rayon (~20 SLOC)
- [x] Add `transcode_batch(input_paths, output_dir, format, dsp_chain)` for batch format conversion (~30 SLOC)

## API Improvements
- [x] Add `#[must_use]` on all Result-returning public functions (~5 SLOC)
- [x] Add comprehensive rustdoc examples for every public function with `# use oxiaudio::*; # fn main() -> Result<(), OxiAudioError> {` blocks (~50 SLOC)
- [x] Add COOLJAPAN format-support matrix as top-level module doc: table of decode/encode formats, Pure vs BOUNDED_FFI, default vs feature-gated (~docs)
- [x] Ensure `cargo doc --no-deps --all-features` builds with zero broken links and zero warnings (~verification)

## Testing
- [x] Add FLAC roundtrip test at all compression levels (0, 3, 5, 8) via facade functions (~25 SLOC)
- [x] Test `decode_file_with_metadata` returns valid metadata from tagged WAV/MP3/FLAC files (~20 SLOC)
- [x] Test `detect_format` on WAV, FLAC, MP3, OGG file headers (~20 SLOC)
- [x] Test streaming decode with multiple block sizes (64, 512, 4096, 32768) produces same total output (~20 SLOC)
- [x] Test DSP convenience functions produce same results as direct oxiaudio-dsp calls (~15 SLOC)
- [x] Benchmark facade overhead: `decode_file` vs direct `SymphoniaDecoder.decode` to quantify abstraction cost (~10 SLOC) — benchmarks in benches/ are the appropriate venue; overhead measured there
- [x] Test `convert` utility: WAV->FLAC->WAV roundtrip preserves sample values within tolerance (~20 SLOC)

## Performance
- [x] Profile facade function overhead (file open, BufReader/BufWriter wrapping) to ensure minimal abstraction cost (~analysis) — benchmarks in benches/ are the appropriate venue; overhead measured there
- [x] Add parallel decode support using rayon for batch processing scenarios (~20 SLOC)
- [x] Implement lazy format detection in `convert` to avoid probing twice (~10 SLOC)

## Integration
- [~] Integration test with oxisound: `decode_file` -> `dsp::resample` -> play via oxisound output stream (~30 SLOC example) — pending oxisound integration; oxiaudio side is ready
- [x] Integration test: `decode_stream` -> `dsp::pitch_shift` each chunk -> `encode_stream_wav` pipeline (~25 SLOC)
- [x] Coordinate re-exports: ensure all new types from sub-crates (BiquadFilter, Compressor, PitchTracker, etc.) are accessible through facade (~re-export audit)
- [x] Final FFI audit: `ffi-audit.sh` passes; `cargo deny check` passes with `mp3lame-sys` excluded from default closure (~gate check)
- [x] CHANGELOG.md covering all milestones in Keep-a-Changelog format (~documentation)
