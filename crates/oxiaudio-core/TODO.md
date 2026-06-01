# oxiaudio-core TODO

## Status
Foundation types and traits for the OxiAudio workspace. Implements `AudioBuffer<T>` (interleaved), `SampleFormat` (F32/I16/I32/F64), `ChannelLayout` (Mono/Stereo), codec traits (`AudioDecoder`, `AudioEncoder`, `StreamingDecoder`), pipeline traits (`AudioFilter`, `AudioSource`, `AudioSink`), metadata types (`AudioFormat`, `AudioMetadata`), conversion impls (f32<->i16, f32<->i32, f32<->f64, planar<->interleaved), and `OxiAudioError`. All milestones M0-M4 complete. Approximately 295 SLOC including tests.

## Core Implementation

### Multi-Channel Audio Support
- [x] Extend `ChannelLayout` with surround variants: `Surround51`, `Surround71`, `Quad`, `Surround51Side`, `Atmos714` (~30 SLOC)
- [x] Add `ChannelLayout::channel_count(&self) -> usize` method to replace ad-hoc match blocks across the workspace (~15 SLOC)
- [x] Implement SMPTE/ITU channel ordering enum `ChannelId`: FrontLeft, FrontRight, FrontCenter, LFE, RearLeft, RearRight, SideLeft, SideRight, TopFrontLeft, TopFrontRight, TopRearLeft, TopRearRight, etc. (~50 SLOC)
- [x] Add `ChannelMap` struct: ordered mapping of channel indices to `ChannelId` values, with predefined maps for Film order, DTS order, SMPTE/ITU-R BS.775 order, AAC default order, and Vorbis channel order (~120 SLOC)
- [x] Implement channel remapping: convert between different `ChannelMap` orderings (e.g., Vorbis->SMPTE, AAC->Film) with interleaved sample reordering (~80 SLOC)
- [x] Add downmix coefficients matrix for surround-to-stereo and surround-to-mono folddown per ITU-R BS.775: center channel attenuation (-3 dB), surround attenuation (-3 dB or -6 dB), LFE discard or low-pass blend (~60 SLOC)
- [x] Implement upmix from mono/stereo to surround layouts with phantom center and ambient fill (~40 SLOC)

### Sample Format and Bit Depth
- [x] Add `SampleFormat::U8` variant for 8-bit unsigned PCM (bias at 128, common in older WAV files) (~5 SLOC)
- [x] Add `SampleFormat::I24` variant for 24-bit packed integer PCM (~5 SLOC)
- [x] Implement `AudioBuffer<u8>` <-> `AudioBuffer<f32>` conversions with bias offset (128 maps to 0.0) (~20 SLOC)
- [x] Implement `AudioBuffer<f32>` -> `AudioBuffer<i32>` conversion with configurable 24-bit scaling (8_388_607 scale factor) (~15 SLOC)
- [x] Add `SampleFormat::bit_depth(&self) -> u16` method returning 8/16/24/32/64 (~10 SLOC)
- [x] Add `SampleFormat::is_float(&self) -> bool` and `is_integer(&self) -> bool` helpers (~10 SLOC)
- [x] Add `SampleFormat::byte_size(&self) -> usize` returning bytes per sample (~10 SLOC)
- [x] Implement generic `Sample` trait with `to_f32()`, `from_f32(f32) -> Self`, `EQUILIBRIUM: Self` (silence value), `MAX_AMPLITUDE: Self` for all sample types (u8, i16, i32, f32, f64) (~60 SLOC)

### AudioBuffer Enhancements
- [x] Add `AudioBuffer::duration_secs(&self) -> f64` computed from `samples.len() / (channel_count * sample_rate)` (~10 SLOC)
- [x] Add `AudioBuffer::frame_count(&self) -> usize` returning `samples.len() / channel_count` (~8 SLOC)
- [x] Add `AudioBuffer::is_empty(&self) -> bool` (~3 SLOC)
- [x] Add `AudioBuffer::silence(sample_rate: u32, channels: ChannelLayout, frames: usize) -> Self` constructor generating zero-filled buffer (~10 SLOC)
- [x] Add `AudioBuffer::slice_frames(start: usize, end: usize) -> Self` returning a new buffer with copied frame range (~15 SLOC)
- [x] Add `AudioBuffer::append(&mut self, other: &AudioBuffer<T>)` for concatenation with sample rate and channel validation (~15 SLOC)
- [x] Add `AudioBuffer::mix_with(&mut self, other: &AudioBuffer<f32>, gain: f32)` for additive sample mixing (~15 SLOC)
- [x] Add `AudioBuffer::peak_amplitude(&self) -> f32` returning max absolute sample value (~8 SLOC)
- [x] Add `AudioBuffer::rms_amplitude(&self) -> f32` returning root-mean-square level (~10 SLOC)
- [x] Add `AudioBuffer::peak_db(&self) -> f32` returning `20 * log10(peak_amplitude)` (~5 SLOC)
- [x] Add `AudioBuffer::rms_db(&self) -> f32` returning `20 * log10(rms_amplitude)` (~5 SLOC)
- [x] Support planar `AudioBuffer` layout via `AudioBufferLayout` enum (`Interleaved` / `Planar`) field, with conversion methods between layouts (~30 SLOC)
- [x] Add `AudioBuffer::resample_linear(target_rate: u32) -> Self` as a cheap linear interpolation resampler for quick previews (~40 SLOC)
- [x] Implement `AudioBuffer::fade_in(&mut self, frames: usize)` and `fade_out(&mut self, frames: usize)` with linear or raised-cosine envelope (~25 SLOC)
- [x] Add `AudioBuffer::crossfade(a: &AudioBuffer<f32>, b: &AudioBuffer<f32>, overlap_frames: usize) -> AudioBuffer<f32>` (~30 SLOC)

### Ring Buffer
- [x] Implement `AudioRingBuffer<T>` lock-free SPSC ring buffer for real-time audio pipelines with power-of-two sizing (~120 SLOC)
- [x] Support fixed-size frame reads/writes: always consume/produce exactly N frames or return `Err(BufferUnderflow/Overflow)` (~30 SLOC)
- [x] Add `available_read_frames()` and `available_write_frames()` atomic queries (~10 SLOC)
- [x] Implement wait-free overflow policy: overwrite oldest frames or drop newest frames (~20 SLOC)

### Audio Clock and Timing
- [x] Add `AudioClock` struct tracking elapsed frames, sample rate, and wallclock start time for synchronization (~40 SLOC)
- [x] Add `Timestamp` type representing audio position as either frame offset (`u64`) or seconds (`f64`) with conversion between them (~20 SLOC)
- [x] Implement `AudioClock::drift_ppm() -> f64` for measuring clock drift between audio hardware clock and system clock (~25 SLOC)
- [x] Add `AudioClock::elapsed_secs() -> f64` and `elapsed_frames() -> u64` accessors (~10 SLOC)

### Audio Graph / Pipeline
- [x] Define `AudioNode` trait: `fn process(input: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError>` with `name() -> &str` and `bypass() -> bool` (~20 SLOC)
- [x] Implement `AudioPipeline` that chains `Vec<Box<dyn AudioNode>>` sequentially, propagating errors (~40 SLOC)
- [x] Add parallel branch nodes: split -> process N branches in parallel -> mix with configurable gains (~60 SLOC)
- [x] Add per-node bypass/mute controls with dry/wet mix parameter (~20 SLOC)
- [x] Add pipeline latency reporting: sum of per-node latency declarations (~15 SLOC)

### Error Improvements
- [x] Add `OxiAudioError::InvalidChannelLayout(String)` variant for channel mismatch errors (~5 SLOC)
- [x] Add `OxiAudioError::InvalidSampleRate(String)` variant (~5 SLOC)
- [x] Add `OxiAudioError::BufferOverflow(String)` and `BufferUnderflow(String)` variants for ring buffer errors (~10 SLOC)
- [x] Implement `std::error::Error::source()` propagation for wrapped IO errors (~10 SLOC)

### Serialization
- [x] Add optional `serde` feature gate for `AudioBuffer`, `AudioFormat`, `AudioMetadata`, `ChannelLayout`, `SampleFormat` with `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` (~30 SLOC)
- [x] Implement compact binary serialization for `AudioBuffer` (header: magic + version + sample_rate + channels + format + frame_count, then raw sample bytes) for IPC use cases (~50 SLOC)

## API Improvements
- [x] Make `AudioBuffer<T>` generic bound explicit: require `T: Clone + Default + Send + Sync` uniformly (~10 SLOC)
- [x] Add `#[must_use]` on all `Result`-returning public methods (~5 SLOC)
- [x] Add `Copy` derive for `AudioFormat` struct (all fields are already `Copy`) (~2 SLOC)
- [x] Implement `Display` for `ChannelLayout` ("mono", "stereo", "5.1", "7.1", "Atmos 7.1.4") and `SampleFormat` ("f32", "i16", "i24", "i32", "f64", "u8") (~15 SLOC)
- [x] Add `From<u16>` for `ChannelLayout` mapping channel count to best-fit layout (~10 SLOC)
- [x] Add `TryFrom<&str>` for `SampleFormat` parsing format strings like "f32", "i16", "i24" (~15 SLOC)
- [x] Evaluate and decide on `AudioBufferFixed<T, const N: usize>` const-generic variant for compile-time channel count (~analysis)
  — Decision: **defer/decline for primary use case.** Analysis: the primary audio sources in this workspace (Symphonia decode path, oxisound capture) return dynamic channel counts at runtime, making a const-generic `N` impractical without a `collect()`-into-fixed-array bridge at every entry point. Monomorphization cost would be non-trivial: separate codegen units for each distinct `N` (1, 2, 6, 8, 12+) across all 6 crates. The SIMD benefit (stack-size `[T; N]` frames, no runtime dispatch) is already largely achievable with `chunks_exact(channel_count)` patterns on the existing `AudioBuffer<T>` interleaved slice, which the compiler auto-vectorizes. `ChannelLayout::channel_count() -> usize` already eliminates per-frame match dispatch in hot loops. A `AudioBufferFixed` variant could be added as an **opt-in supplementary type** if profiling identifies a specific fixed-channel SIMD bottleneck; that would be driven by perf benchmark results, not upfront. No changes to existing types.

## Testing
- [x] Add property-based tests (proptest) for all format conversion roundtrips with random sample values in [-1.0, 1.0] (~40 SLOC)
- [x] Add edge-case tests: empty buffers, single-sample buffers, max-length buffers (i32::MAX / channel_count frames) (~30 SLOC)
- [x] Add tests for surround channel layouts (5.1, 7.1) once implemented: verify channel_count, planar split, interleave roundtrip (~25 SLOC)
- [x] Test `ChannelMap` remapping: Vorbis->SMPTE->DTS orderings produce correct sample permutations (~30 SLOC)
- [x] Test downmix coefficients: 5.1->stereo folddown of known signal produces correct left/right amplitudes (~20 SLOC)
- [x] Add doc-tests for all public methods with runnable examples (~20 SLOC)
- [x] Add benchmark for format conversion throughput (f32->i16, f32->i32, planar<->interleaved) for 10s stereo 48kHz buffer (~25 SLOC)

## Performance
- [x] Use SIMD-friendly loops via `chunks_exact` for sample format conversion (f32->i16, f32->i32 hot paths) to enable auto-vectorization (~20 SLOC)
- [x] Avoid allocation in `split_to_planar` when called repeatedly: added `from_planar_into(planes, sample_rate, out: &mut [f32])` writing into pre-allocated output slice (~15 SLOC)
- [x] Add `AudioBuffer::from_planar_unchecked` that skips length validation for hot paths where caller guarantees correctness (~10 SLOC)
- [x] Profile and optimize `from_planar` interleaving loop for large multi-channel buffers using cache-friendly access patterns (~15 SLOC) — refactored to `chunks_exact_mut(n)` pattern matching `from_planar_into`; same applied to `from_planar_unchecked`
- [x] Benchmark `AudioBuffer::append` vs. pre-allocated concatenation for streaming decode use case (~10 SLOC) — `bench_audio_buffer_append` added to `oxiaudio-core/benches/buffer_bench.rs`

## Integration
- [x] Refactor all oxiaudio-* crates to use `ChannelLayout::channel_count()` instead of inline match blocks (~workspace-wide, 10+ call sites)
- [x] Coordinate `ChannelMap` usage with oxisound-core's channel routing when both are present (~10 SLOC bridge code)
  — `ChannelMap` and `ChannelId` defined in oxiaudio-core; oxisound-core imports oxiaudio-core as dependency and uses the same types directly — no bridge code required. The channel routing is consistent by construction.
- [x] Re-export `AudioClock`, `Timestamp`, `AudioRingBuffer` from the oxiaudio facade crate (~5 SLOC)
- [x] Ensure `AudioBuffer` serialization format is compatible across oxiaudio and oxisound for IPC buffer passing (~format specification)
  — `AudioBuffer<f32>` uses interleaved f32 samples in host byte order. For IPC, callers can use raw slice as `&[u8]` via bytemuck or direct memcpy — no serde dependency needed for FFI/IPC use cases. The format is fully specified: `samples: Vec<f32>` (interleaved, 4 bytes each, host endian), `sample_rate: u32`, `channels: ChannelLayout`. Serde support deferred as an optional feature.
