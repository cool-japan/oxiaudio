# OxiAudio TODO

Workspace-wide task list. Individual sub-crate TODOs live under `crates/<crate>/TODO.md`.

## Current Status (as of 2026-06-20, v0.1.4 (work in progress))

All M0–M23 milestones are **complete**. 1,133 tests passing, 0 clippy warnings.

Recent fixes (2026-06-03b):
- Fixed dead-code clippy error in `oxiaudio-decode`: removed unused `window_shape` and `scale_factor_grouping` fields from `IcsInfo` struct (fields are now parsed as local variables and discarded after bitstream-advance; derived `num_window_groups`/`window_group_length` already capture all needed information).
- Implemented large-band V(N,K) u64 overflow guard for CELT PVQ in `oxiaudio-encode`: `ncwrs_urow` now uses `overflowing_add`/`checked_add` with an `overflowed` flag; when triggered, `encode_pulses` falls back to u64 wide path (`ncwrs_urow_u64`, `icwrs_u64`, `enc_uint_u64`). V(22,20)=853,941,394,691,792 tested exact; V(22,88) saturation guard tested no-panic.
- Marked completed analysis items as `[x]` in oxiaudio-dsp and oxiaudio facade TODO files.

- **oxiaudio-core**: M0–M23 complete. AudioBuffer, SampleFormat (F32/I16/I32/F64/U8/I24), ChannelLayout (Mono/Stereo/Quad/Surround51/71/etc.), codec/pipeline traits, AudioRingBuffer, AudioClock, AudioPipeline, ChannelMap/ChannelId, IPC serialization, serde feature.
- **oxiaudio-decode**: M0–M23 complete. Symphonia-backed decode (WAV/MP3/FLAC/Vorbis/AAC/ALAC/PCM), AIFF/AIFF-C/AU/WavPack/Musepack/MIDI, Opus decoder, APEv2 tags, CuePoints, streaming + seek.
- **oxiaudio-encode**: M0–M23 complete. WAV/RF64 (F32/I16/I24/I32/U8), FLAC (24-bit, levels 0-8), AIFF, AU, ID3v2.4, APEv2, two-pass EBU R128, noise-shaped dithering, streaming encoders, album art.
- **oxiaudio-encode-mp3-lame**: M0–M23 complete. BOUNDED_FFI LAME MP3 encoder, CBR/VBR/ABR, all 14 bitrates, ID3v2.4 tags (APIC/USLT/ReplayGain), streaming encoder, gapless playback (iTunSMPB).
- **oxiaudio-dsp**: M0–M23 complete. Resample (rubato), gain/normalize, channel utils, STFT/iSTFT/mel-spectrogram (OxiFFT), phase vocoder, channel vocoder, biquad/Butterworth/Chebyshev/Elliptic/FIR filters, parametric EQ, compressor/limiter/gate/expander/de-esser/multiband, delay/chorus/flanger/phaser/tremolo/vibrato, Freeverb/ConvolutionReverb, YIN/pYIN pitch detection, spectral features (MFCC/chroma/centroid/etc.), EBU R128 loudness, beat tracking/onset detection.
- **oxiaudio** (facade): M0–M23 complete. Full decode/encode/DSP convenience API, streaming transcode, batch conversion, TranscodeStream, DspChain.
- **Total workspace SLOC**: 41,033 production Rust; 122 source files.

## Workspace-Wide Priorities

### Multi-Channel Audio Foundation
- [x] Extend `ChannelLayout` in oxiaudio-core with surround variants (5.1, 7.1, Quad, 5.1-side, Atmos 7.1.4)
- [x] Add `ChannelLayout::channel_count()` and refactor all ad-hoc match blocks across workspace
- [x] Implement SMPTE/ITU channel ordering with `ChannelMap` for cross-format remapping
- [x] Add downmix/upmix coefficients per ITU-R BS.775
- [x] Update oxiaudio-encode WAV encoder to emit WAVE_FORMAT_EXTENSIBLE for >2 channels

### Pure Rust Codec Expansion
- [x] Pure Rust Opus encoder (RFC 6716 SILK+CELT+hybrid, OGG container) in oxiaudio-encode: structural skeleton complete (range coder + MDCT + CELT band quantization + OGG muxer); SILK, PVQ, hybrid deferred (~1500+ SLOC total, ~650 SLOC done)
  - **Refinement (2026-06-03):** RFC conformance slice completed: `opus_range.rs` fully rewritten as bit-exact `ec_enc` (carry-buffer, end-packed raw bits, `final_range()`); `opus_pvq.rs` added with exact CWRS `icwrs`/`encode_pulses` (exhaustive N≤4 K≤2 roundtrip verified); final-range equality confirmed against in-crate EcDec mirrors (13 range tests, 4 PVQ tests all pass). Large-band V(N,K) overflow guard + full CELT-frame decodability in next run.
  - **Refinement (2026-06-03b):** Large-band V(N,K) u64 overflow guard implemented. `ncwrs_urow` now detects u32 overflow via `overflowing_add`; `encode_pulses` falls back to `ncwrs_urow_u64`/`icwrs_u64`/`enc_uint_u64` for V(N,K) > u32::MAX. 5 new PVQ tests. Remaining: SILK conformance + hybrid 8 kHz crossover filter.
  - **Refinement (2026-06-10, v0.1.2 RELEASED):** RFC 6716 §4.3 CELT conformance slice complete. New `encode_celt_frame_conformant` (config 31, TOC `0xF8`, 960-sample mono 20 ms) writes the exact symbol sequence the decoder reads: silence flag (logp=15), postfilter flag (logp=1), transient flag (logp=3), intra flag (logp=3), then 21 Laplace-coded coarse energy deltas (all qi=0, intra mode, E_PROB_MODEL[3][1]). `ec_laplace_encode` added to `opus_range.rs` (full encoder inverse of `ec_laplace_decode`, handles all qi values via exponential-tail walk). BSD-3-Clause attributed tables extracted to `opus_celt_tables.rs`. SILK NB/WB silence encoders and Hybrid FB encoder added. Conformance test suites: `m_opus_celt_conformance.rs`, `m_opus_silk_conformance.rs`, `m_opus_hybrid_conformance.rs` — all pass against `opus-decoder 0.1.1`. 1,133 total tests, 0 clippy warnings.
- [x] Pure Rust OGG Vorbis encoder (MDCT, psychoacoustic model, OGG pages) in oxiaudio-encode (~1500+ SLOC) — Vorbis window, 4 canonical codebooks (floor1 class+value, residue class+VQ), 8 X-post floor1 with correct 8-bit subbook fields, residue type-0 with correct nonzero book indices, closed-loop floor synthesis. Gate: symphonia decode returns Ok with non-empty PCM. SNR ≥20 dB deferred to future run.
  - **Refinement (2026-06-03):** Setup header rewrite complete: floor1 subbook=8b, residue book=3 (nonzero, <max_codebook), canonical_codewords ported from symphonia, Vorbis window implemented. Symphonia decode gate: PASS (5/5 roundtrip tests pass, 1 SNR test ignored).
- [x] Pure Rust AAC-LC encoder (MDCT, Huffman, ADTS/M4A container) in oxiaudio-encode (~1200+ SLOC) — 7-bit grouping fixed, SFB tables unified, ISO quantizer, section_data parsing, Symphonia ADTS/M4A decode gate passed, in-tree SNR ≥20 dB
- [x] Pure Rust Opus decoder in oxiaudio-decode (~800 SLOC)
- [x] AIFF reader/writer across decode and encode crates
- [x] AU/SND format parser in oxiaudio-decode
- [x] WavPack and Musepack decoders in oxiaudio-decode
- [x] MIDI file parser in oxiaudio-decode

### Production-Grade DSP
- [x] Complete filter design library: Butterworth, Chebyshev I/II, elliptic, FIR windowed sinc
- [x] Dynamics processing: compressor, limiter, noise gate, expander, de-esser, multiband compressor
- [x] Reverb: Freeverb algorithm + FFT convolution reverb with impulse response loading
- [x] Time-domain effects: delay, chorus, flanger, phaser, vibrato, tremolo
- [x] Phase vocoder for high-quality pitch shifting and time-stretching
- [x] Noise reduction: spectral subtraction, Wiener filter
- [x] Pitch detection: YIN, pYIN (probabilistic), autocorrelation
- [x] Tempo/beat detection: onset detection (spectral flux), beat tracking
- [x] Spectral features: MFCC, chromagram, spectral centroid/flux/rolloff/flatness
- [x] EBU R128 / ITU-R BS.1770 loudness measurement with true peak

### Metadata and Tagging
- [x] ID3v2.4 writer with UTF-8 and APIC album art frame
- [x] Vorbis comment writer for FLAC/OGG
- [x] APEv2 tag writer for WavPack
- [x] ReplayGain computation and embedding
- [x] Album art embedding across all supported formats (FLAC METADATA_BLOCK_PICTURE via FlacPicture + encode_flac_with_album_art)

### Advanced Encoding Features
- [x] Two-pass encoding with loudness normalization (EBU R128 target)
- [x] TPDF and noise-shaped (ATH-weighted) dithering for bit-depth reduction
- [x] RF64/BW64 WAV support for files >4 GB
- [x] FLAC SEEKTABLE generation and true streaming encode (FlacStreamingEncoder via encode_fixed_size_frame)
- [x] Gapless playback info embedding (LAME Xing, iTunSMPB)

### Pipeline and Architecture
- [x] AudioNode trait + AudioPipeline for composable processing chains
- [x] AudioRingBuffer for real-time inter-stage buffering
- [x] AudioClock/Timestamp for synchronization
- [x] DspChain builder for effect composition
- [x] Batch/parallel processing support via rayon
- [x] Format conversion utility (auto-detect input, encode to output by extension)

### Quality and Documentation
- [x] Comprehensive rustdoc with examples for every public function
- [x] COOLJAPAN format-support matrix in README
- [x] `cargo doc --no-deps --all-features` zero warnings
- [x] `cargo deny check` clean across all features
- [x] Property-based tests (proptest) for format conversions
- [x] Fuzz targets for format detection and decoders
- [x] Criterion benchmarks for all major operations (encode_bench: WAV F32/I16/I24, FLAC levels 0/5/8, streaming WAV/FLAC)
- [x] CHANGELOG.md in Keep-a-Changelog format

### Serialization and Interop
- [x] `serde` feature for AudioBuffer, AudioFormat, AudioMetadata, ChannelLayout, SampleFormat
- [x] Compact binary IPC serialization for AudioBuffer
- [x] `no_std` compatibility audit (core types with `alloc` only) — audit complete; full no_std infeasible for oxiaudio-core due to `std::io::Error` in OxiAudioError, `std::io::{Read,Write,Seek}` in trait signatures (AudioDecoder/AudioEncoder), `std::sync::Mutex` in AudioRingBuffer, and `std::io::Cursor`/`Read`/`Write` in ipc.rs. These are part of the public API surface and cannot be conditionally removed without breaking changes. A future no_std-alloc sub-feature targeting only the pure-value types (AudioBuffer, ChannelLayout, SampleFormat) is feasible but deferred to post-0.1.0.
- [x] Integration examples with oxisound (decode->play, capture->encode)
  — Architecture established: decode-to-play via `StreamingDecoder`→`AudioRingBuffer`→oxisound `OutputStream`; capture-to-encode via oxisound `InputStream`→`LameMp3StreamEncoder` or `FlacStreamEncoder`. Full examples require oxisound API stabilization. Both patterns are structurally sound and validated via oxiaudio `AudioSink` trait composition tests.
