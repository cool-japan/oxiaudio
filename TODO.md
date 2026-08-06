# OxiAudio TODO

Workspace-wide task list. Individual sub-crate TODOs live under `crates/<crate>/TODO.md`.

## Current Status (as of 2026-08-06, v0.2.2)

All M0–M23 milestones are **complete**. Full release-gate sweep re-measured 2026-08-06 against the
working tree:

- 1,136 tests passing / 5 skipped with default features and 1,242 passing / 6 skipped with
  `--all-features` (`cargo nextest run --no-fail-fast --workspace`), zero compiler warnings in both
  runs. `--all-features` activates the opt-in `mp3-encode-lame` feature and builds `mp3lame-sys`
  from vendored C (autotools); it succeeds on a machine with a C toolchain, so no `--exclude`
  workaround is needed.
- 63 doc tests passing / 4 ignored (`cargo test --doc --workspace --all-features`).
- 0 clippy warnings (`cargo clippy --workspace --all-targets --all-features -- -D warnings`).
- 0 rustdoc warnings (`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps`) —
  four public doc comments in `oxiaudio-encode` that linked to private items were converted to plain
  code spans as part of this sweep.
- `cargo fmt --all -- --check` clean; `cargo deny check bans` reports `bans ok`.
- `cargo publish --dry-run --allow-dirty -p oxiaudio-core` passes. The five downstream crates fail
  the dry-run only on `oxiaudio-core = "^0.2.1"` not yet being on crates.io — the expected
  intra-workspace ordering artefact, resolved by publishing in topological order.

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
- **Total workspace SLOC**: 39,706 lines of code across 95 Rust source files in `crates/*/src`
  (`tokei crates/*/src`, Code column, production code only — excludes `tests/`, `benches/`, and
  `examples/`; re-measured 2026-08-06) across the 6 published
  crates (core/decode/encode/encode-mp3-lame/dsp/oxiaudio). The 7th workspace member,
  `oxiaudio-integration-tests`, is `publish = false` with an empty `src/lib.rs` and contributes 0
  lines to this count.

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
- [x] `cargo doc --no-deps --all-features` zero warnings (also verified with `RUSTDOCFLAGS="-D warnings"`)
- [x] `cargo deny check` clean across all features (bans/licenses/sources/advisories all `ok`; the
  two RUSTSEC findings present as of 2026-08-04 — `RUSTSEC-2026-0204` (crossbeam-epoch) and a
  yanked `spin` — were both resolved by an in-range `cargo update -p crossbeam-epoch -p spin`,
  0.9.18→0.9.20 and 0.12.1→0.12.2, no manifest changes, full test suite re-verified green after)
- [x] Property-based tests (proptest) for format conversions
- [x] Fuzz targets for format detection and decoders
- [x] Criterion benchmarks for all major operations (encode_bench: WAV F32/I16/I24, FLAC levels 0/5/8, streaming WAV/FLAC)
- [x] CHANGELOG.md in Keep-a-Changelog format

### Serialization and Interop
- [x] `serde` feature for AudioBuffer, AudioFormat, AudioMetadata, ChannelLayout, SampleFormat
- [x] Compact binary IPC serialization for AudioBuffer
- [x] `no_std` compatibility audit (core types with `alloc` only) — audit complete; full no_std infeasible for oxiaudio-core due to `std::io::Error` in OxiAudioError, `std::io::{Read,Write,Seek}` in trait signatures (AudioDecoder/AudioEncoder), `std::sync::Mutex` in AudioRingBuffer, and `std::io::Cursor`/`Read`/`Write` in ipc.rs. These are part of the public API surface and cannot be conditionally removed without breaking changes. A future no_std-alloc sub-feature targeting only the pure-value types (AudioBuffer, ChannelLayout, SampleFormat) is feasible but deferred to post-0.1.0.
- [x] Integration examples with oxisound (decode->play, capture->encode)
  — Architecture established: decode-to-play via `StreamingDecoder`→`AudioRingBuffer`→oxisound `OutputStream`; capture-to-encode via oxisound `InputStream`→`LameMp3StreamEncoder` or `FlacStreamEncoder`. Both patterns are structurally sound and validated via oxiaudio `AudioSink` trait composition tests. A runnable `oxisound`-integrated example specifically is **not** included in this workspace's `examples/` directories: `oxisound` depends on `oxiaudio`/`oxiaudio-core` (see `oxisound/Cargo.toml`), so an example living in this repo that also depends on `oxisound` would be a genuine workspace-level dependency cycle, not just an API-stability question — it would need to live in the `oxisound` repo instead (as a consumer of `oxiaudio`), or wait for a separate top-level integration-examples crate outside both workspaces.
- [x] Runnable examples for the core in-workspace flows (2026-08-04): `crates/oxiaudio/examples/decode_dsp_encode.rs` (decode → `DspChain` → encode, with a round-trip decode check), `crates/oxiaudio/examples/streaming_transcode.rs` (`TranscodeStream` chunked decode → filter → encode), `crates/oxiaudio-encode/examples/opus_conformant.rs` (`encode_opus_auto` / `encode_opus_conformant` — the RFC 6716–conformant, opt-in-discoverable Opus API), `crates/oxiaudio-dsp/examples/dsp_chain.rs` (resample + filter chain on `oxiaudio-dsp` directly, no facade). Each synthesizes its own input in-code (no fixture files, no hardcoded paths — all I/O goes through `std::env::temp_dir()`); verified via `cargo build --workspace --examples` and `cargo run --example <name>`.


---

<!-- production-readiness-backlog 2026-07-16 -->
## Production-Readiness Backlog — 2026-07-16

_Consolidated from static audit + Opus adversarial bug-hunt (48 verified defects across noffi) + baseline nextest/clippy + design investigation. See `../NOFFI_PRODUCTION_BACKLOG.md` for the full cross-project list and severity/model legend._
_Status reconciled 2026-08-04 against the actual working tree (commit `20c2a63` "production" + uncommitted wave fixes) — see `CHANGELOG.md`'s `[0.2.1]` section for the full, categorized list of what landed._

**Confirmed bugs — Opus-verified:**
- [x] **S · high** `oxiaudio-decode/src/wavpack.rs:691` — `data[pos+32..block_end]` with `block_end` from unvalidated `block_size`; `block_size < 24` → start>end slice panic. **Fixed**: `block_size < 24` now rejected with a typed `Decode` error before the slice is computed; regression test `test_wavpack_undersized_block_size_rejected_not_panicking`.
- [x] **S · med** `oxiaudio-decode/src/aiff.rs:146` — `num_frames*num_channels` from COMM header → `Vec::with_capacity` petabyte alloc/abort from a tiny file. **Fixed**: both the PCM and AIFF-C mu-law/A-law decode paths now compute `required_bytes` and return a `Decode` error before allocating.
- [x] **S · med** `oxiaudio-decode/src/musepack.rs:674` — `n_frames*FRAME_SAMPLES_STEREO` from SV7 header → ~34GB+ reservation/abort. **Fixed**: `max_plausible_frames` guard rejects the header before the over-large reservation. (A lower-severity residual — the guard still permits a ~9200x byte→reservation amplification on a crafted-but-plausible frame count — is tracked separately, not in this checklist.)
- Pattern: validate container header size/count fields before allocation/slicing; cap reservations. Applied consistently across all three sites above.

**Designed (Opus RFC 6716 conformance — encoder decodability, not yet transparent quality):**
- [x] **S/hard/Opus · A1** RFC-compliant range coder (entenc/entdec, bit-reversal packing) replacing private variant. **Done**: `opus_range.rs` is a faithful, bit-exact port of libopus `celt/entenc.c` (carry buffer, end-packed raw-bit window, `ec_laplace_encode`); shared by both the default and conformant CELT paths.
- [x] **A/hard/Opus · A2** exact CELT PVQ (V(N,K) combinatorial index, §4.3.4.6) + band allocation verify. **Done (2026-08-04)**: split-band `itheta` coding is real (measured angle, `qn` quantisation, triangular range coding, both halves coded recursively with the RFC's `delta`/rebalance rules); fine-energy quantisation and the ±½-LSB finalise pass are real; band allocation is *verified*, not just ported — the encoder's `final_range` matches the reference decoder's for every supported frame size (`tests/m_opus_celt_snr.rs`). Four bugs found in the process: `enc_bit_logp` inverted, `enc_bits` not advancing `nbits_total`, an unfaithful `ec_laplace_encode`, and a mis-charged band-skip bit in `interp_bits2pulses`. **Residual cleared (2026-08-04, wave 5)**: `MAX_CELT_FRAME_BYTES` is now the RFC's 1275-byte frame limit (510 kbps mono). The old 80-byte cap was caused by `celt_bits2pulses`/`celt_pulses2bits` clamping the *partition* scale `lm` to 0 before the `lm + 1` cache-row lookup, so the `lm == -1` leaves that four-deep band splits produce read row 1 instead of row 0 — and four-deep splits only happen above ≈90 bytes/frame. Two further bugs fell out of the same investigation: `finish_to_size_checked` did not share the boundary byte between the range and raw-bit streams the way libopus's `ec_enc_done` does (costing up to 36 bytes/frame of requested CBR rate to needless shrink-retries), and `bitexact_cos`/`bitexact_log2tan` could shift by −1.
- [x] **A/hard/Opus · A3** real SILK encoder (replace facade structural stub: LP analysis/quantization/coding). **Done**: `opus_silk_encode.rs` (~1,326 lines, commit `20c2a63`) — real LP analysis, two-stage NLSF VQ, analysis-by-synthesis excitation, wired into `encode_silk_frame_conformant`. Scope: NB only, unvoiced-only, independently-coded frames (measured best-lag correlation ≈ 0.33–0.67 against reference decode — see the module doc, not a claim of transparent quality).
- [ ] **A/med/Opus · A4** hybrid integration + round-trip vs own pure decoder + RFC test vectors if available. **Partial.** The "round-trip vs own pure decoder" half is **done** for CELT: `opus_range_dec.rs` + `opus_celt_verify.rs` are an in-crate RFC-structured decoder that re-parses every field of a CELT frame and agrees with the encoder on the final range register, and (2026-08-04, wave 5) the encoder now records the *same* named stage positions (`CeltEncodeTrace` / `encode_celt_frame_conformant_traced`) so the two are compared stage-by-stage rather than only on the final register. The hybrid **bitstream** is also done: the missing redundancy flag (`logp = 12`) and the hardcoded 512-bit CELT-layer budget are fixed, the hybrid writer is real CBR, and `final_range` matches the reference decoder across `[20, 1275]` bytes — before this it never matched at any size, which no "does it decode?" test could see. Still open: the hybrid low band is `encode_silk_wb_silence_into` (inactive/zero-excitation — the real SILK encoder is NB-only), so hybrid carries no low-frequency audio and `select_conformant_mode` never selects it; and no RFC 6716 test-vector corpus exists in-tree (the RFC's vectors are not distributed with the spec text and must not be fetched from the network).
  - **Remaining work, scoped and de-risked (2026-08-04, wave 5).** The investigation below was done;
    only the implementation is left, and it is smaller than it first looks.
    - The obvious framing — "port the whole WB NLSF two-stage VQ" — needs `NLSF_CB1_WB_Q8` (512 B),
      `NLSF_CB1_WGHT_WB_Q9` (512 × i16), `NLSF_PRED_WB_Q8`, `NLSF_CB2_SELECT_WB` (256 B),
      `NLSF_CB2_ICDF_WB`, `NLSF_DELTA_MIN_WB_Q15` and the WB quant step (9830).
    - **It is not necessary.** Pinning `cb1_index = 0` with **neutral stage-2 residuals** (symbol 4 =
      amplitude 0, which the existing `encode_silk_wb_silence_into` already writes) makes
      `silk_NLSF_residual_dequant` produce an all-zero residual, so the decoder reconstructs exactly
      `nlsf_q15[i] = NLSF_CB1_WB_Q8[i] << 7`. The weights table is then never consulted, and the only
      table data needed is **33 values**:
      `NLSF_CB1_WB_Q8[0..16] = [7, 23, 38, 54, 69, 85, 100, 116, 131, 147, 162, 178, 193, 208, 223, 239]`
      and `NLSF_DELTA_MIN_WB_Q15 = [100, 3, 40, 3, 3, 3, 5, 14, 14, 10, 11, 3, 8, 9, 7, 3, 347]`.
      That row is very nearly uniform, so the resulting LPC is close to flat — the excitation then
      carries the whole waveform, which is crude but real audio rather than silence. A later step can
      widen this to a 32-vector stage-1 search by adding `NLSF_CB1_WB_Q8` in full (512 B),
      `NLSF_CB2_SELECT_WB` (256 B) and `NLSF_CB2_ICDF_WB` (72 B) — still under ~100 lines of tables,
      still no weights table needed while stage-2 stays neutral.
    - `nlsf2a` is order-generic apart from one table: the coefficient ordering, which for order 16 is
      `[0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1]` (order 10 uses
      `[0, 9, 6, 3, 4, 5, 8, 1, 2, 7]`). Everything else (`nlsf2a_find_poly`, `lpc_fit`,
      `bw_expand_32`, `lpc_inv_pred_gain`, `nlsf_stabilize`) is already order-parametric arithmetic.
    - The shell-code tables, sign iCDF, rate-level iCDF, pulses-per-block iCDF, gain iCDFs, delta-gain
      iCDF, interp-factor iCDF and the seed iCDF are **bandwidth-independent** and already present in
      `opus_silk_encode.rs` / `opus_silk_conform.rs`. So is the WB stage-1 iCDF (`NLSF_CB1_ICDF_WB`)
      and the WB stage-2 row-0 iCDF (`NLSF_CB2_ROW0_WB`).
    - Real blocker: `opus_silk_encode.rs` is hard-wired to NB through module constants and
      fixed-size arrays (`[i8; ORDER]`, `[i16; FRAME_LEN]`, `[i32; SHELL_BLOCKS]`). The LPC math
      (`nlsf2a`, `nlsf_stabilize`, `lpc_inv_pred_gain`, `lpc_fit`, `bw_expand_32`) and the
      analysis-by-synthesis loop (`choose_gain_and_excitation`, `quantize_excitation`,
      `decode_sim_peak`) must move to slice-based helpers in a shared `opus_silk_core.rs` that both
      NB and WB call. The file is ~1330 lines, so the split is required anyway by the COOLJAPAN
      2000-line rule.
    - Also required: a 48 kHz → 16 kHz decimator (by 3) alongside the existing NB one, `nb_subfr = 4`
      with an 80-sample subframe, and 20 shell blocks (`320 / 16`).
    - Verification bar for the next wave: hybrid `final_range` must stay equal to the reference
      decoder's across `[MIN_HYBRID_FRAME_BYTES, MAX_HYBRID_FRAME_BYTES]` (already pinned by
      `hybrid_packet_layer_budget_is_consistent`), **plus** a new gate that the decoded output has
      real, input-correlated energy below the 8 kHz crossover — `final_range` alone cannot see a
      silent low band, which is precisely how this gap survived.
    - Not attempted in this wave: the refactor touches a verified NB path and the mode is off the
      default route (`select_conformant_mode` never returns `Hybrid`), so it was judged a worse use
      of the remaining budget than making the CELT rate ceiling airtight.
- [x] **B/easy · A5** non-test unwrap reduction (~124, core/dsp); reconcile "not supported" error docs. **Done**: 0 non-test `.unwrap()`/`.expect(` across all `crates/*/src` (verified by suppressing everything after each file's first `#[cfg(test)]` marker); feature-off decode paths (e.g. `opus.rs`) return a typed `UnsupportedFormat` instead of panicking.
- [x] **B/med · A6** fuzz harnesses for decoders (none today); examples. **Done**: fuzz half via `crates/oxiaudio-integration-tests/tests/fuzz_decoders.rs` (~20 `proptest!` properties across all container decoders, landed in `20c2a63`); examples half via the four runnable `examples/` directories added 2026-08-04 (see the "Integration examples with oxisound" item above for the full list and the one deliberately-excluded case).

**Wave 5 — deferral recovery (2026-08-04):**
- [x] **S/hard** CELT frames above 80 B: root-caused (`lm == -1` pulse-cache row) and the cap raised to the RFC's 1275-byte frame limit. Bit-exactness swept over `[16, 1275]` × 10 fixtures × 2 frames against the reference decoder, now also asserting the *physical* stream fit (`final_range` cannot see a dropped raw-bit byte).
- [x] **S/med** `RangeEncoder::finish_to_size_checked` now shares the boundary byte between the range and raw-bit streams exactly as libopus's `ec_enc_done` does. Measured CBR shrink across every request from 16 to 1275 bytes: zero (was up to 36 bytes/frame).
- [x] **A/med** Hybrid CELT-layer bit budget verified **and fixed**: the layer's hardcoded 512-bit budget is replaced by the emitted CBR length, and the previously-missing hybrid redundancy flag (`logp = 12`) is written. Hybrid `final_range` now matches the reference decoder across `[20, 1275]` — it had never matched at any size.
- [x] **B/easy** `bitexact_cos` / `bitexact_log2tan` negative-shift panic (debug builds) fixed; `theta_params_preserve_unit_energy` was red on it.
- [ ] **A/hard** Real SILK wideband low band for hybrid mode — see the scoped note under A4 below. Not attempted in this wave; the hybrid layer's *bitstream* correctness was landed instead, which is the prerequisite for it.

**Cross-cutting hygiene (2026-08-04 wave):**
- [x] **B/easy** `rustfmt.toml` + `clippy.toml` at the workspace root (MSRV pinned to `rust-version = "1.80"`, matching `Cargo.toml`).
- [x] **B/easy** `deny.toml` extended with the full COOLJAPAN banned-crate table; documented, dated exception for the `rubato → realfft → rustfft` transitive edge.
- [x] **B/easy** Two `oxiaudio-encode-mp3-lame` dev-dependencies switched from raw `path =` to `{ workspace = true }`.
- [x] **B/easy** `AudioPipeline::latency_hint()` reimplemented on top of `total_latency_frames()` instead of a hardcoded-0 stub.
- [x] **B/easy** Stale "SILK is silence-only" documentation corrected in the four sites the 2026-08 audit identified (`opus_silk_conform.rs`, `oxiaudio/src/encode.rs` ×2, `opus_encoder.rs`), plus the unrelated `analyze_silk_frame`/`encode_silk_frame` "structural stub" mislabel.
- [x] **B/easy** `CHANGELOG.md`'s `[0.2.1]` section filled in (was a literal `(placeholder)`).
- [x] **B/easy** Baseline re-measured: `target/` (`-> /tmp/target/oxiaudio`) recreated; `cargo build --workspace --examples`, `cargo clippy --workspace --all-targets` (default + `--all-features`), `cargo nextest run --workspace` (default + `--all-features`), `cargo fmt --all --check`, and `cargo deny check` all run clean (see the Current Status line above for exact numbers). No longer UNMEASURED.
- [x] **B/med** Preventive split of the largest actively-growing source file: `opus_celt.rs` (1,908 lines) split into `opus_celt.rs` (1,148) + `opus_celt_bands.rs` (split-band/PVQ coding) + `opus_celt_rate.rs` (bit allocation). The other four near-limit files (`aac_decoder.rs` 1,924, `aac.rs` 1,882, `vorbis.rs` 1,830, `spectral.rs` 1,766) are unchanged in this wave and remain under the limit.
