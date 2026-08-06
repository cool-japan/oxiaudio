# Changelog

All notable changes to OxiAudio are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [0.2.2] - Unreleased

## [0.2.1] - 2026-08-06

### Added
- **The CELT frame-size ceiling is now the RFC's own 1275-byte frame limit
  (510 kbps mono), not 80 bytes (≈32 kbps).** The previous wave clamped
  `MAX_CELT_FRAME_BYTES` to 80 because the encoder's `final_range` stopped
  matching the reference decoder's somewhere above ≈90 bytes/frame and the cause
  had not been found. It was `celt_bits2pulses` / `celt_pulses2bits` clamping the
  **partition** scale `lm` to 0 before indexing the pulse cache with `lm + 1`.
  A band that splits four times reaches `lm = -1` (band 20 at LM = 3 goes
  176 → 88 → 44 → 22 → 11 bins), which libopus indexes as cache **row 0**;
  clamping selected row 1 instead, giving those leaves a different pulse count
  than the decoder derives. Four-deep splits only happen once a band's budget
  clears the cache maximum, i.e. only at higher rates — which is exactly why the
  80-byte cap hid the bug and why raising the cap without fixing it would have
  shipped corrupt frames. The in-crate verifier shared the same helper, so it
  agreed with the encoder and could not see the defect; the reference decoder
  was the only witness. `final_range` equality is now swept from
  `MIN_CELT_FRAME_BYTES` to 1275 over ten fixtures (tones from 120 Hz to 19 kHz,
  speech, music, full-scale and near-silent noise, an impulse train and digital
  silence) × two frames each (stream-start and with-history).
- **CBR frames now deliver the requested size instead of silently shrinking.**
  `RangeEncoder::finish_to_size_checked` wrote the partial raw-bit window at its
  own byte index, so a frame always needed one byte more than libopus's
  `ec_enc_done`, which `|=`-merges that window into `buf[storage - end_offs - 1]`
  — the *last range byte* when the two streams meet, whose low bits the range
  coder's termination leaves free. Because the CELT rate allocator fills its
  budget to the last bit, that phantom byte made the fit check fail for ~80 % of
  frame sizes and the CBR writer retried one byte smaller, giving up as much as
  36 bytes/frame (≈14 kbps) of requested rate. With the boundary byte shared as
  libopus does it, the measured shrink across every request from 16 to 1275
  bytes on tonal, noise and silent fixtures is **zero**.
- **Hybrid packets are entropy-exact for the first time.** Two defects, both
  invisible to a "does it decode?" test — the packets decoded to 960 samples
  throughout, they just decoded different symbols:
  - The encoder never wrote the hybrid **redundancy flag** (`logp = 12`, guarded
    by `tell + 17 + 20 <= total_bits`) that a decoder reads between the SILK and
    CELT layers, so every CELT symbol after it was one bit position out and
    `final_range` never matched at any size.
  - `encode_celt_hybrid_layer_into` hardcoded `TARGET_BITS_HYBRID = 512` while
    the packet was assembled with a variable-length `finish()`, so the decoder's
    `total_bits = payload_len · 8` matched the layer's budget only by
    coincidence. The hybrid writer is now real CBR
    (`encode_hybrid_frame_conformant_sized`, `finish_to_size_checked`) and passes
    the emitted length into the layer.
  `final_range` equality against the reference decoder is now pinned across
  `[MIN_HYBRID_FRAME_BYTES, MAX_HYBRID_FRAME_BYTES]` = `[20, 1275]`.
- **Encoder-side bitstream trace** (`CeltEncodeTrace`,
  `encode_celt_frame_conformant_traced`). The previous wave built a verifier that
  records the range-coder position after every named decode stage; that is only
  half an instrument, because there was nothing on the encoder side to compare
  it against. The encoder now records the same stages under the same names, plus
  the split depth actually reached (`min_partition_lm`), the largest `K` handed
  to a PVQ leaf, how many leaves saw `V(N, K)` wrap `u32`, and whether the two
  stream halves physically fitted the packet. `celt_encoder_stage_tells_match_verifier`
  asserts the two agree stage-by-stage, which turns a future desynchronisation
  into a pinpointed stage rather than a mismatched register at the end.
- New tests: `celt_final_range_matches_reference_low_rates` /
  `..._high_rates` (the split sweep above, now also asserting the *physical* fit
  — `final_range` is a range-coder property and cannot see a dropped raw-bit
  byte), `celt_encoder_stage_tells_match_verifier`,
  `celt_high_rate_frames_exercise_deep_splits` (proves the `lm == -1` partition
  is actually reached at high rates, so the regression coverage cannot silently
  evaporate), `celt_higher_bitrate_improves_snr` (the raised cap must buy real
  quality: >6 dB of time-domain SNR at 320 B/frame over the old 80-byte
  ceiling), plus `opus_celt_rate` unit tests for the cache-row indexing.

- **RFC 6716 CELT conformance is now proven, not asserted.** Three real bugs in the range coder and
  one in the rate allocator were found and fixed, and the encoder's `final_range` register now
  matches the reference decoder's for **every** supported frame size on a corpus that includes
  full-scale white noise — the canonical libopus conformance check
  (`tests/m_opus_celt_snr.rs::celt_final_range_matches_reference_low_rates` / `..._high_rates`). The four fixes:
  - `RangeEncoder::enc_bit_logp` encoded the **logical complement** of its argument. The stream
    stayed byte-synchronised, so "does it decode?" tests passed, but every CELT header flag
    (silence, intra, transient, TF, band-skip) and every SILK VAD/LBRR flag was inverted.
  - `RangeEncoder::enc_bits` did not advance `nbits_total`, so `tell()` under-reported once any raw
    bits were written. CELT's whole rate allocation is driven by `tell()`, so encoder and decoder
    drifted apart from the first fine-energy field onward.
  - `ec_laplace_encode` was not a faithful port: it could emit an out-of-range symbol and, unlike
    libopus, never reported the **clamped** value it actually coded, so the coarse-energy `prev`
    chain diverged from the decoder's.
  - `interp_bits2pulses` charged the band-skip bit to `psum` on the break path; libopus charges it
    only on the "skip this band" path. The inflated `psum` shrank the final bit distribution and
    silently gave every band fewer pulses than the decoder expected.
- **In-crate RFC-structured CELT decoder and bitstream verifier** (`opus_range_dec.rs`,
  `opus_celt_verify.rs`, new files). `RangeDecoder` is a faithful `ec_dec` port with primitive-level
  round-trip tests against `RangeEncoder`; `parse_celt_frame` re-reads a whole CELT frame in the
  decoder's exact field order (headers → Laplace coarse energy → TF → spread → dynalloc → trim →
  allocation with the skip bit → fine energy → PVQ bands with recursive `itheta` splits → finalise)
  and reports per-stage bit positions plus the final range. This is what makes the conformance
  claims checkable inside the crate rather than only against a third-party decoder.
- **Real split-band (`itheta`) coding** (RFC 6716 §4.3.4.3, `opus_celt_bands.rs`): the split angle is
  measured from the two half-bands' actual energies, quantised to `qn` levels, range-coded with the
  triangular distribution, and *both* halves are coded recursively with the RFC's `delta`-driven
  `mbits`/`sbits` split, visit order and post-recursion rebalance. Previously `itheta` was pinned to
  0, which forced the upper half of every split band to reconstruct as zeros.
- **Real fine-energy quantisation** (§4.3.2.1): the coarse-energy residual is now refined with the
  allocator's per-band fine bits and the final ±½-LSB priority pass. Previously those bits were
  written as zeros, pinning every band's correction to a systematic −0.5 log2 (≈ −3 dB) offset.
- **Lapped MDCT analysis with overlap history** (`opus_mdct::celt_mdct_960_overlap`,
  `celt_analysis_spectrum`) plus the **CELT pre-emphasis filter** and the `CELT_SIG_SCALE · 2/N`
  analysis gain. CELT's MDCT is lapped and its decoder applies a mandatory de-emphasis; encoding
  each frame as if surrounded by silence and skipping pre-emphasis left the decode ~11–19× quiet and
  tilted by ~11× from 100 Hz to 18 kHz. Per-band decoded energy is now within ≈1 dB of the input on
  white noise, and the overall level within ±3 dB. `encode_opus`, `encode_opus_auto`,
  `encode_opus_conformant` and `OpusStreamEncoder` all carry the history across frames.
- **Real CELT bitrate control**: `celt_frame_bytes_for_bitrate` turns `target_bitrate_kbps` into a
  CBR payload size, wired into `encode_opus`/`encode_opus_auto` and the new
  `OpusStreamEncoder::with_bitrate`. Sizes are clamped to `MAX_CELT_FRAME_BYTES`, the largest frame
  proven bit-exact against the reference decoder.
- `RangeEncoder::finish_to_size_checked`, `range_bytes`, `raw_bytes`: the range-coded stream grows
  from the front of a CBR packet and the raw-bit stream from the back; on a near-full frame they can
  collide and bytes had to be dropped *silently*. The collision is now detected and the CELT writer
  retries at a smaller frame size (which keeps encoder and decoder in agreement, since the decoder
  derives its budget from the emitted packet length) instead of shipping a corrupt frame.
- **Stream-level CELT quality gates** (`tests/m_opus_celt_snr.rs`, 9 tests): per-band level accuracy
  over speech-like, music-like, tonal and noise fixtures; time-domain SNR at the fixed 540-sample
  CELT reconstruction delay; absolute level tracking; split-band and fine-energy
  information-content checks; and the full-range `final_range` sweep.
- **End-to-end OGG coverage for the flagship path**: `encode_opus` → OGG stream → reference decoder,
  including a SILK↔CELT mode transition (silent lead-in followed by music), plus
  `OpusStreamEncoder::with_bitrate` history/rate coverage. Previously every Opus quality test fed
  raw per-frame packets to the decoder and never exercised the OGG entry points.
- `encode_pulses` now mirrors `decode_pulses` exactly — wrapping `u32` `V(N,K)` with the decoder's
  own `.max(2)` guard. The previous `u64` fallback (and the two silent `return`s that skipped
  writing a symbol entirely) were divergent by construction: the reference decoder has no wide path,
  so any of those branches would have desynchronised the bitstream. The `u64` helpers are retained
  as `#[cfg(test)]` cross-checks that the wrapping path computes `V(N,K)` correctly.
- `opus_celt.rs` split into `opus_celt.rs` + `opus_celt_bands.rs` + `opus_celt_rate.rs` (COOLJAPAN
  2000-line rule; the file was at 1,908 lines and growing).
- **New workspace member `oxiaudio-integration-tests`** (`publish = false`, `crates/oxiaudio-integration-tests`).
  Twelve cross-crate round-trip test files (`m2_flac`, `m3_bitdepth`, `m4_streaming`, `m6_features`,
  `m10_decode`, `m11_flac_config`, `m12_wav_config`, `m23_decode`, `m23_encode`, `m_aac_roundtrip`,
  `m_vorbis_roundtrip`) plus `decode_bench` moved out of `oxiaudio-encode` / `oxiaudio-decode` into
  this crate, joined by the new `inline_*` round-trips (AAC, AU, FLAC streaming, M4A, OGG CRC, WAV
  cue) and the `fuzz_decoders` proptest harness — 130 tests in total. Hosting them in an unpublished
  member keeps encode↔decode integration coverage out of the published codec crates' dev-dependency
  graph.

### Known gaps (unchanged or newly measured this wave)
- **Hybrid mode still has a silent low band.** `encode_hybrid_frame_conformant`'s SILK layer is
  `encode_silk_wb_silence_into` (inactive/zero-excitation); the real SILK encoder is narrowband-only.
  `select_conformant_mode` never returns `Hybrid`, so it is off the default path. The *bitstream*
  half of that gap is closed — the hybrid CELT layer's budget and the redundancy flag are both fixed
  and `final_range` now matches the reference decoder across the full size range (see Added) — but
  the low band still carries no audio. Wiring a wideband SILK encoder in (16 kHz internal rate,
  order-16 NLSF codebooks, 20 shell blocks, WB stage-1/stage-2 codebooks) remains open.
- ~~**CELT frame sizes above `MAX_CELT_FRAME_BYTES` (80 B ≈ 32 kbps) are clamped, not emitted.**~~
  **Resolved** — root cause was the `lm == -1` pulse-cache row (see the Added section above). The
  ceiling is now the RFC's 1275-byte frame limit and bit-exactness is swept over the whole range.
- **Pure-Rust SILK narrowband (NB) Opus encoder** (`oxiaudio-encode/src/opus_silk_encode.rs`,
  new file, ~1,326 lines): real analysis-by-synthesis speech coding — LP analysis (autocorrelation +
  Levinson-Durbin), two-stage NLSF VQ that reproduces the decoder's `nlsf_decode` bit-for-bit,
  log-domain gain selection, and shell-coded excitation. Wired into the public conformant API via
  `encode_silk_frame_conformant`. Scope is honestly bounded: NB only, unvoiced excitation only, no
  inter-frame prediction, independently-coded frames (measured best-lag correlation ≈ 0.33–0.67
  against reference decode — see the module's own doc for the full caveat list). Bit-exactness is
  checked by the new `crates/oxiaudio-encode/tests/silk_internal_roundtrip.rs` (739 lines).
- **Exact CELT PVQ shape search** (`opus_celt.rs`): the conformant path's greedy pulse allocator
  replaced by `op_pvq_search` (libopus's rate-distortion search, maximizing `⟨x,y⟩²/⟨y,y⟩`) plus the
  matching forward `exp_rotation`/`exp_rotation1`, so the encoder's rotation is bit-for-bit the
  inverse of the decoder's. The already-existing exact CWRS combinatorial index (`opus_pvq.rs`,
  §4.3.4.6, added in 0.1.1) is unchanged; what's new here is what feeds it.
- **Range coder carry-shift fix** (`opus_range.rs`): `carry_out` was shifting by
  `EC_CODE_BITS - EC_SYM_BITS` (24) where libopus's `ec_enc` uses `EC_CODE_BITS - EC_SYM_BITS - 1`
  (23, now a named `EC_CODE_SHIFT` constant) — the off-by-one dropped bit 23 of every emitted byte.
  New bit-exact symbol-recovery round-trip tests (decode-and-compare, not just `final_range`
  equality, which cannot by itself detect this class of bug) cover `encode`/`decode`,
  `enc_icdf`/`dec_icdf`, and a mixed uint/raw-bit stream.
- Property-based fuzz harness for the container decoders
  (`crates/oxiaudio-integration-tests/tests/fuzz_decoders.rs`, ~20 `proptest!` properties):
  wavpack, musepack, aiff/aiff-c, au, aac/adts, midi, raw-pcm, format detection, OpusHead, and
  gapless parsing, all asserted panic/abort/hang-free under randomized and magic-prefixed input.
- Runnable examples: `oxiaudio` (`decode_dsp_encode`, `streaming_transcode`),
  `oxiaudio-encode` (`opus_conformant`), `oxiaudio-dsp` (`dsp_chain`) — each synthesizes its own
  input in-code and is verified by `cargo build --examples` / `cargo run --example`.
- Workspace hygiene: `rustfmt.toml`, `clippy.toml` (MSRV pinned to the workspace's `rust-version`),
  `SECURITY.md`, `CONTRIBUTING.md`; `deny.toml` extended with the full COOLJAPAN banned-crate table
  (bincode/rusqlite/quick-xml/zip/flate2/zstd/bzip2/lz4/tar/snap/brotli/miniz_oxide/openblas), plus
  a documented, dated `wrappers`-scoped exception for the `rubato → realfft → rustfft` transitive
  edge (no oxiaudio crate depends on rustfft directly; `oxifft` remains the direct dependency used
  for the workspace's own spectral/MDCT code).

### Security
- `crossbeam-epoch` 0.9.18 → 0.9.20 (in-range `cargo update`, no manifest change): resolves
  `RUSTSEC-2026-0204` (invalid pointer dereference in `fmt::Display`/`fmt::Pointer` for `Atomic`/
  `Shared`), reached transitively via `rayon` → `rayon-core` → `crossbeam-deque` →
  `crossbeam-epoch`. Not reachable from any oxiaudio code path that formats an `Atomic`/`Shared`
  pointer, but resolved anyway rather than carrying a live advisory.
- `spin` 0.12.1 → 0.12.2 (in-range `cargo update`, no manifest change): the pinned 0.12.1 had been
  yanked from crates.io; reached transitively via `oxifft`. `cargo deny check` now reports
  `advisories ok` (previously `FAILED` on both findings above).

### Changed
- **`encode_opus` / `OpusStreamEncoder`** (the default, most-discoverable Opus entry points) now
  route every 20 ms frame through the RFC 6716–conformant per-frame encoders (automatic CELT/SILK
  selection via `select_conformant_mode` for `encode_opus`; CELT-only for `OpusStreamEncoder`) —
  previously they emitted non-conformant 4-bit placeholder quantization that no standard decoder
  would accept at all. The pre-0.2.1 byte layout is preserved verbatim as `encode_opus_structural`
  / `encode_opus_structural_file` for byte-compatibility and OGG-framing-only tests.
  **This is still not a transparent-quality encoder** — CELT is mono-only (stereo input is
  downmixed), non-transient, uses a neutral non-dynalloc allocation and is capped at
  `MAX_CELT_FRAME_BYTES` (1275 B/frame ≈ 510 kbps mono — the RFC's own frame limit, raised from the
  earlier 80-byte ≈32 kbps ceiling in this same release, see Added) where bit-exactness is
  proven; SILK is
  unvoiced-only, and `OpusConformantMode::Hybrid`'s low band is still an inactive SILK silence
  frame pending a WB-capable SILK encoder. See the `opus_encoder` / `opus_celt` /
  `opus_silk_encode` module docs for the exact, measured caveats.
- `AudioPipeline::latency_hint()` now delegates to `total_latency_frames()` (collapsing the
  empty-pipeline `None` to `0`) instead of being a hardcoded stub that always returned `0`.
- `oxiaudio-encode-mp3-lame`'s dev-dependencies on `oxiaudio-decode` / `oxiaudio-dsp` now use
  `{ workspace = true }` instead of a raw `path =` entry, matching every other manifest in the
  workspace and picking up the `0.2.1` version constraint from `[workspace.dependencies]`.

### Dependencies
- `oxifft` 0.3.2 → 0.4.2 (upstream releases, three bumps since 0.2.0: 0.3.2 → 0.4.0 → 0.4.1 → 0.4.2).

### Fixed
- **`wavpack.rs`**: a block whose stored `block_size` field was smaller than the minimum valid
  WavPack block header (24 bytes) could drive an unvalidated slice computation
  (`pos + block_size + 8`) past the buffer; now rejected with a typed `Decode` error before use.
- **`aiff.rs`**: `num_frames * num_channels` from the file header was used to pre-size a `Vec`
  (`Vec::with_capacity`) without checking it against the bytes actually available in the chunk,
  allowing a crafted small file to request a huge allocation; both the PCM and the AIFF-C
  mu-law/A-law decode paths now compute `required_bytes` and return a `Decode` error first.
  Regression-tested in both places.
- **`musepack.rs`**: an SV7 frame-count header field with no upper bound could drive
  `n_frames * FRAME_SAMPLES_STEREO` into a many-gigabyte up-front reservation from a tiny input
  file; now capped by a plausibility ceiling derived from the input length before reserving.
- **OGG page CRC-32 is now validated** (`ogg_reader.rs`): pages are checksummed with the stored
  CRC field zeroed and compared against the recorded value before their packets reach the
  Opus/Vorbis decoders; a mismatch returns a typed `Decode` error. Previously the checksum was
  parsed and unconditionally discarded, so corrupted pages reached the codec decoders unverified.
- `mdct_forward` (`opus_mdct.rs`) and `aac_mdct_forward` (`aac.rs`) no longer `assert_eq!` on input
  length (a release-active panic reachable from any external caller of these `pub fn`s); both now
  zero-pad short input and truncate long input instead, and are panic-free for any input length.
- **`bitexact_log2tan` could shift by a negative amount** (`opus_celt_bands.rs`), which is a panic in
  a debug build. `bitexact_cos` returned 32768 for `|x| < 64` — libopus asserts its own intermediate
  stays at or below 32766, so `EC_ILOG(result) <= 15` and the `15 - EC_ILOG` shift is non-negative —
  and the shift then went to `-1`. `bitexact_cos` now enforces the `[1, 32767]` range libopus
  documents and both shifts saturate at 0. The clamp is unobservable for every reachable `itheta`
  (a multiple of `16384 / qn` with `qn <= 256`, so the argument is never below 64), confirmed by the
  full-range `final_range` sweep. The existing `theta_params_preserve_unit_energy` unit test — which
  probes `itheta = 16383` — was failing on this before the fix.

### Documentation
- Corrected four sites that still described the conformant SILK path as "silence-only" /
  "deferred to a future run" after the real analysis-by-synthesis encoder landed:
  `opus_silk_conform.rs` module doc, `oxiaudio/src/encode.rs` (×2), and `opus_encoder.rs`
  (`OpusConformantMode::Silk` doc, `encode_opus_conformant` doc). Also corrected the unrelated
  `analyze_silk_frame` / `encode_silk_frame` facade doc comments, which mislabeled a real (if
  non-RFC-6716-bitstream) LP-analysis encoder as a "structural stub".
- README format-support table and Status section corrected to state the `encode_opus` conformance
  change above, with the same not-yet-transparent caveats (previously claimed unqualified "RFC
  6716 conformant" with no mention of the default path's prior non-decodability).
- Four public doc comments in `oxiaudio-encode` linked to private items
  (`encode_band_with_splits` in `opus_celt_bands.rs`, `encode_silk_wb_silence_into` in
  `opus_hybrid_conform.rs`, `celt_lapped_window` ×2 in `opus_mdct.rs`), which made
  `RUSTDOCFLAGS="-D warnings" cargo doc` fail on `rustdoc::private_intra_doc_links`. They are now
  plain code spans, matching how the surrounding prose already refers to private helpers.

## [0.2.0] - 2026-06-21

### Changed
- **Pure Rust Policy v2 compliance** (`oxiaudio`): the facade's `--all-features` dependency
  closure is now L1-clean (zero `-sys` / FFI). MP3 *encoding* (LAME / libmp3lame, LGPL FFI)
  is no longer reachable from the pure crates and is opt-in only by depending on the
  `oxiaudio-encode-mp3-lame` quarantine crate directly. MP3 *decode* remains Pure Rust in
  the facade.
- Workspace version bumped from 0.1.3 to 0.2.0 (`version` in `[workspace.package]`); all
  internal path-dependency versions updated 0.1.3 → 0.2.0.
- Documentation updated for the new MP3-encode boundary: `oxiaudio` crate-level docs,
  `crates/oxiaudio/README.md`, and `crates/oxiaudio-encode-mp3-lame/README.md` now state that
  MP3 encoding lives only in the quarantine crate (the pure facade exposes MP3 decode only).

### Removed
- **`oxiaudio` facade:** removed the `mp3-encode-lame` and `full` Cargo features and the
  optional `oxiaudio-encode-mp3-lame` dependency.
- **`oxiaudio` facade:** removed the `mp3-encode-lame`-gated crate-root re-exports `AlbumArt`,
  `compute_replaygain_gain_approx`, and `encode_mp3_with_auto_replaygain`.
- **`oxiaudio-encode`:** removed the `mp3` Cargo feature and the optional
  `oxiaudio-encode-mp3-lame` back-edge dependency (its entire `[features]` block was deleted).
- **`oxiaudio-encode`:** removed the `mp3`-gated re-exports `LameMode`, `LameMp3Encoder`, and
  `VbrPreset`.

### Security
- Eliminated the libmp3lame C FFI from the `oxiaudio` facade's `--all-features` dependency
  closure, removing the C / `unsafe` attack surface (and LGPL obligation) from the pure
  facade. MP3-encode FFI is now isolated behind the opt-in `oxiaudio-encode-mp3-lame`
  quarantine crate.

## [0.1.3] - 2026-06-19

### Added
- **RFC 6716–conformant OGG Opus encoder** (`oxiaudio-encode`, `oxiaudio`): opt-in
  `encode_opus_conformant<W: Write>(buf, writer, mode)` and `encode_opus_conformant_file`
  that route each 20 ms frame through conformant SILK / CELT / Hybrid per-frame writers
  and produce a structurally valid OGG Opus stream accepted by standard Opus decoders.
- **`OpusConformantMode` enum** (`oxiaudio-encode`, `oxiaudio`): `Celt` (full MDCT + PVQ;
  decoded output correlates >0.1 with a 440 Hz input tone), `Silk` (silence-only —
  zero-excitation inactive frame; decodes cleanly), `Hybrid` (SILK WB silence + CELT
  high-band bands 17–20); re-exported from the `oxiaudio` facade crate.
- **Opt-in conformance integration tests** (`crates/oxiaudio-encode/tests/m_opus_conformant_optin.rs`):
  203-line integration test suite verifying that each mode roundtrips through `opus-decoder`;
  confirms CELT 440 Hz correlation, SILK silence decodability, Hybrid finite-sample output,
  file-write path, and backward-compatibility of the legacy `encode_opus` TOC byte (`0xE0`).

### Changed
- **Doc-comment updates** (`oxiaudio/src/encode.rs`): `encode_vorbis_to_file` and
  `encode_aac_to_file` doc-strings updated to reflect that MDCT encoding is implemented
  (Vorbis: floor type-1 + residue VQ; AAC-LC: CB11/ESC_HCB Huffman spectral encoding).

---

## [0.1.2] - 2026-06-10

### Added
- **SILK NB encoder** (`oxiaudio-encode`, `opus_silk_conform`): RFC 6716–conformant
  pure-Rust SILK narrowband (8 kHz) encoder producing decodable Opus packets.
  Exposes `encode_silk_frame_conformant(pcm) -> Vec<u8>` (TOC `0x08`, config 1 = SILK-only
  NB 20 ms mono). Encodes iCDF tables for signal type, gain, delta-gain, NLSF stage-1/2
  (NB and WB codebooks), interpolation factor, excitation seed, pulse rate-level,
  and pulses-per-block; zero-excitation (silence) path verified against `opus-decoder 0.1.1`.
- **SILK WB encoder** (`oxiaudio-encode`, `opus_silk_conform`): RFC 6716–conformant
  wideband (16 kHz) SILK layer encoder; exposes internal `encode_silk_wb_silence_into`
  for use by the hybrid encoder path. Uses WB NLSF codebook (order=16, 20 shell blocks,
  320-sample frame).
- **CELT conformant encoder** (`oxiaudio-encode`, `opus_celt`): complete RFC 6716
  §4.3 CELT-only conformance slice.  `encode_celt_frame_conformant` (config 31,
  TOC `0xF8`, 960-sample mono 20 ms) writes the exact symbol sequence the decoder
  reads: silence flag (logp=15), postfilter flag (logp=1), transient/intra flags,
  21 Laplace-coded coarse energy deltas (intra mode, `E_PROB_MODEL[3][1]`),
  TF/spread/dynalloc/trim headers, `clt_compute_allocation`-based rate allocation,
  fine energy (zeros), and PVQ CWRS shapes per band.
- **Hybrid FB encoder** (`oxiaudio-encode`, `opus_hybrid_conform`): RFC 6716
  hybrid mode (config 15, TOC `0x78`, Hybrid Fullband 20 ms mono) encoder combining
  a SILK WB silence layer and a CELT high-band layer (bands 17–20, `start_band=17`)
  in a single shared range-coder stream.
- **`ec_laplace_encode`** (`opus_range.rs`): full encoder inverse of `ec_laplace_decode`;
  handles all `qi` values via exponential-tail walk; used for Laplace-coded coarse
  energy quantization in CELT.
- **`celt_mdct_960`** (`opus_mdct.rs`): 960-sample MDCT analysis via OxiFFT for
  CELT frame energy and PVQ shape computation.
- **CELT BSD-3-Clause tables** (`opus_celt_tables.rs`): rate-allocation constants
  extracted to a dedicated module — `EBAND_5MS`, `LOG_N_400`, `ALLOC_TRIM_COEFS`,
  `ALLOC_TABLE_CELT`, `CACHE_BITS_50`, `CACHE_INDEX_50`; attributed to Xiph.Org Foundation.
- **Conformance test suites**: `m_opus_celt_conformance.rs`, `m_opus_hybrid_conformance.rs`,
  `m_opus_silk_conformance.rs` (3 × ~130 lines each) in `oxiaudio-encode`, verifying
  TOC bytes and decodability against `opus-decoder 0.1.1`.
- **AAC decoder short-window `section_data`** (`oxiaudio-decode`): `decode_section_data`
  now handles `EIGHT_SHORT_SEQUENCE` windows; reads 8 groups × up to `max_sfb` sections
  with `sect_bits=3` / `sect_esc=7` (ISO 14496-3 §4.6.8.2.3); sections are offset by
  `group * max_sfb` so callers can index a flat `num_window_groups * max_sfb` scale-factor
  array.
- **AAC decoder short-window scale-factor array** (`oxiaudio-decode`): `decode_scale_factors`
  now allocates `num_window_groups * max_sfb` entries for short-window frames, eliminating
  out-of-bounds indexing on grouped short windows.

### Changed
- `encode_celt_body_into` refactored to a shared inner function used by both
  CELT-only (`start_band=0`, silence flag written) and hybrid (`start_band=17`, silence
  flag omitted) paths, eliminating code duplication between the two modes.
- `transcode_batch` example updated to use `std::env::temp_dir()` for output paths
  (no more hardcoded absolute paths).

### Fixed
- **AAC decoder `decode_section_data` long-window zero-length guard** (`oxiaudio-decode`):
  the error message now reads `"AAC: section_data has zero-length section"` consistently
  for both long-window and short-window paths.
- **Rustdoc broken intra-doc links** (`opus_celt_tables.rs`, `opus_hybrid_conform.rs`):
  fixed `entry[0]` bracket escape and removed private-item cross-links that caused
  `RUSTDOCFLAGS="-D warnings"` build failures.

## [0.1.1] - 2026-06-04

### Added
- **`opus_pvq` module** (`oxiaudio-encode`): CWRS (Combinatorial Number System / CWRS)
  PVQ encoder — bit-exact inverse of `decode_pulses` from `opus-decoder`.
  Public functions: `encode_pulses(enc, y)`, `ncwrs_urow`, `icwrs`, plus u64-wide
  fallback variants (`ncwrs_urow_u64`, `icwrs_u64`, `enc_uint_u64`) for large
  bands where V(N,K) exceeds u32::MAX (e.g. CELT band 20 at high bitrates).
- **`OpusDecoder::final_range()`** (`oxiaudio-decode`): exposes the range coder's
  final range value for RFC 6716 conformance testing against the encoder's
  `final_range()`.
- **AAC decoder `decode_ics_data`** (`oxiaudio-decode`): inner ICS decoder
  extracted from `decode_sce`, now shared by both SCE and CPE element decoders.
- **AAC decoder `Section` / `decode_section_data`** (`oxiaudio-decode`): proper
  `section_data()` bitfield parser — reads 4-bit codebook + escape-coded section
  lengths; used by scale-factor and spectral-data decoders.
- **AAC decoder SFB offset tables** (`oxiaudio-decode`): canonical ISO 14496-3
  Table 4.138 tables added for 24 kHz/22.05 kHz, 16 kHz, 64 kHz, 96 kHz/88.2 kHz,
  and 8 kHz; `sfb_offsets_long` now uses a threshold-based lookup matching the
  encoder exactly.
- **AAC decoder CB11 canonical table** (`oxiaudio-decode`): the minimal 28-entry
  CB11 stub replaced by the full 289-entry `HCB11_LENS`/`HCB11_CODES` arrays from
  ISO 14496-3 Annex A, enabling correct high-energy spectral coefficient decoding.
- **Test suites**: `m_aac_roundtrip.rs` (249 lines), `m_oxisound_pipeline.rs`
  (478 lines), `m_vorbis_roundtrip.rs` (321 lines), `m_stream_pitch.rs` (230 lines)
  added to `oxiaudio-encode` and `oxiaudio` crates.
- `oxiaudio-dsp` added as dev-dependency in `oxiaudio-encode` for cross-crate
  pipeline integration tests.

### Changed
- **`RangeEncoder` rewritten** (`oxiaudio-encode`): `opus_range.rs` is now a
  faithful RFC 6716 §4.1 port of libopus `ec_enc`, bit-exact with the
  `EcDec` decoder in `opus-decoder`; the previous self-consistent-but-non-standard
  encoding is replaced.  Raw bits are packed from the physical end of the buffer
  (LSB-first) and stitched with range bytes on `finish()`.
- **`opus_celt` PVQ shape encoding** (`oxiaudio-encode`): `encode_pvq_shape`
  replaced by `opus_pvq::encode_pulses` — CWRS combinatorial coding instead of
  magnitude+sign per-coefficient encoding.
- `compute_global_gain` (`oxiaudio-encode`): demoted from `pub` to
  `#[cfg(test)]`-private; only used in unit tests.
- `oxiaudio-encode` restored as dev-dependency in `oxiaudio-decode` (was
  temporarily removed for publish; circular dev-dep now resolved).

### Fixed
- **AAC decoder SFB tables** (`oxiaudio-decode`): the 48 kHz and 32 kHz SFB
  boundary arrays were wrong (only 33 entries, not matching the encoder).  They
  now carry the full canonical ISO 14496-3 entries (50 and 52 boundaries
  respectively), eliminating spectral misalignment on common sample rates.
- **AAC decoder scale-factor parsing** (`oxiaudio-decode`): `decode_scale_factors`
  was unconditionally reading one delta per SFB; it now skips `ZERO_HCB` sections
  (no bits in bitstream) and correctly accumulates deltas only over live sections,
  preventing bitstream misalignment.
- **AAC decoder spectral data** (`oxiaudio-decode`): `decode_spectral_data` was
  decoding all SFBs with CB11 regardless of the codebook; it now reads only
  sections that have spectral data in the bitstream (`cb != 0, 13, 14, 15`),
  eliminating systematic decode errors for streams with zero-coded bands.
- **AAC decoder TNS parsing** (`oxiaudio-decode`): `tns_data_present` block was
  a coarse 8-bit skip that mis-aligned the bitstream; it now parses
  `n_filt`, `coef_res`, per-filter `length`/`order`/`direction`/`coef_compress`,
  and reads exactly the right number of coefficient bits.
- **AAC decoder CPE element** (`oxiaudio-decode`): channel-pair elements now
  correctly read the 4-bit `element_instance_tag` and 1-bit `common_window` flag
  before decoding each channel's ICS data; previously these bits were silently
  consumed as audio data.
- **AAC decoder `IcsInfo`** (`oxiaudio-decode`): unused fields `window_shape` and
  `scale_factor_grouping` removed; the `predictor_data_present` bit (always present
  per ISO 14496-3) is now correctly parsed and discarded.
- **AAC encoder `scale_factor_grouping`** (`oxiaudio-encode`): the 7-bit
  `scale_factor_grouping` field is only written for `EIGHT_SHORT_SEQUENCE`; it
  was incorrectly written for `ONLY_LONG_SEQUENCE` frames, producing a 7-bit
  bitstream offset that caused decoder misalignment on all long-window frames.
- **AAC encoder `compute_global_gain_and_inv_scale`** (`oxiaudio-encode`): gain
  formula corrected to ISO standard (`gain = 100 − (16/3)·log2(target/peak_q)`,
  `inv_scale = 2^(−3·(gain−100)/16)`); the previous formula included an erroneous
  `+16.0` offset and a redundant reciprocal, causing systematic over-quantization.

## [0.1.0] - 2026-06-01 (M0–M23 combined release, 1079 tests)

### oxiaudio-core
#### Added
- `AudioBuffer<T>` generic interleaved sample buffer with rich utility methods:
  `duration_secs`, `frame_count`, `is_empty`, `silence`, `slice_frames`, `append`,
  `peak_amplitude`, `rms_amplitude`, `peak_db`, `rms_db`, `fade_in`, `fade_out`
- `ChannelLayout` enum: `Mono`, `Stereo` with `channel_count()` and `Display`
- `SampleFormat` enum: `F32`, `I16`, `I32`, `F64`, `U8`, `I24` with `bit_depth()`,
  `is_float()`, `is_integer()`, `byte_size()`, `Display`, `TryFrom<&str>`
- `OxiAudioError` with `Io`, `UnsupportedFormat`, `InvalidChannelLayout`,
  `InvalidSampleRate`, `BufferOverflow`, `BufferUnderflow` variants
- `AudioDecoder`, `AudioEncoder`, `StreamingDecoder` codec traits
- `AudioFilter`, `AudioSource`, `AudioSink` pipeline traits
- `AudioMetadata` with title, artist, album, year, genre, track_number, disc_number, comment
- `AudioFormat` with sample_rate, channels, format, duration_secs, bitrate_kbps
- Sample format conversions: `f32↔i16`, `f32↔i32`, `f32↔f64`, planar↔interleaved
- `Sample` trait with `to_f32`, `from_f32`, `EQUILIBRIUM`, `MAX_AMPLITUDE`
- `AudioBufferLayout` enum (Interleaved/Planar) with `to_planar`/`from_planar`
- `AudioRingBuffer<T>` with `write_frames`, `read_frames`, `available_read_frames`,
  `available_write_frames`
- `AudioClock` with `advance`, `elapsed_frames`, `elapsed_secs`, `drift_ppm`
- `Timestamp` enum (Frames/Seconds) with conversion methods
- `AudioNode` trait and `AudioPipeline` with sequential node chaining
- Optional `serde` feature gate for core types
- `#[must_use]` on all Result-returning methods
- `ChannelMap`, `ChannelId`, downmix/upmix utilities

### oxiaudio-decode
#### Added
- Symphonia-based decoder for WAV, FLAC, MP3 (CBR/VBR), OGG/Vorbis, AIFF, AU
- `decode_file(path) -> Result<AudioBuffer<f32>>` — full decode to memory
- `decode_file_with_metadata(path) -> Result<(AudioBuffer<f32>, AudioMetadata)>`
- `decode_file_f64(path) -> Result<AudioBuffer<f64>>` — double-precision
- `decode_stream(path) -> Result<impl Iterator<Item=Result<AudioBuffer<f32>>>>`
- `decode_stream_with_block_size(path, block_size)` — configurable block size
- `StreamingDecoder` trait with `next_block`, `format_info`, `metadata`,
  `skip_frames`, `remaining_frames`, `seek_to_time`
- Pure-Rust AIFF parser (8/16/24-bit PCM, 80-bit IEEE extended sample rate)
- Pure-Rust AU/SND parser (encodings: i16, i24, f32)
- Raw PCM reader with `RawPcmConfig` (format, endianness, header skip)
- `detect_format_from_bytes(header: &[u8]) -> Option<AudioFormatHint>`
- `detect_format_file(path) -> Result<AudioFormatHint>`
- `AudioFormatHint` enum: Wav, Flac, Mp3, Ogg, Aiff, Au
- Extended metadata: genre, track_number, disc_number, comment
- `#[must_use]` on all Result-returning functions

### oxiaudio-encode
#### Added
- `encode_wav(buf, path)` — 16-bit signed PCM WAV
- `encode_flac(buf, path)` — FLAC at compression level 5
- `encode_flac_with_level(buf, writer, level)` — configurable 0–8 compression
- Pure-Rust AIFF writer (`write_aiff`, `write_aiff_file`) — 16-bit BE PCM
- WAV 8-bit unsigned PCM output (`WavBitDepth::U8`)
- TPDF dithering (`apply_tpdf_dither`) for quantization noise reduction
- `encode_wav_to_vec` / `encode_flac_to_vec` — in-memory encoding
- `StreamEncoder` trait with `write_chunk` / `finalize`
- `WavStreamEncoder<W>` and `FlacStreamEncoder<W>` streaming encoders
- `EncoderConfig` builder with `with_bit_depth`, `with_dither`, `with_flac_compression`,
  `with_normalize`, `encode_wav`, `encode_flac`
- `WavBitDepth` enum: Pcm16, Pcm8U, Float32

### oxiaudio-encode-mp3-lame (feature-gated, requires LGPL libmp3lame)
#### Added
- `LameMp3Encoder` with CBR and VBR modes, all 14 bitrate values
- `Mp3Tags` struct for ID3v2 metadata (title, artist, album, year, track)
- `LameMp3StreamEncoder` for chunk-by-chunk streaming encode
- `encode_mp3_cbr` / `encode_mp3_vbr` convenience functions

### oxiaudio-dsp
#### Added
- **Resampling**: High-quality sinc interpolation via rubato (SIMD: SSE2/AVX/NEON)
- **Gain/normalize**: `gain(buf, db)`, `normalize(buf, target_db)`
- **Channel utilities**: `mix_to_mono`, `split_channels`
- **Silence**: `trim_silence(buf, threshold_db)`
- **STFT/iSTFT**: `stft`, `istft`, `StftOutput`, configurable `WindowFn` (Hann, Blackman, Hamming, Kaiser, FlatTop)
- **Mel spectrogram**: `melspectrogram(buf, n_fft, hop, n_mels)`
- **Pitch shifting**: `pitch_shift` (simple), `pitch_shift_pv` (phase vocoder)
- **Time stretching**: `time_stretch(buf, ratio, n_fft, hop_a)`
- **Biquad filters** (RBJ Audio EQ Cookbook): `BiquadFilter::lowpass/highpass/bandpass/notch/allpass/peaking`
- **Parametric EQ**: `ParametricEq` with cascaded biquads, `frequency_response`, `phase_response`, `group_delay`
- **Butterworth filters**: `butterworth_lowpass/highpass` as cascaded SOS
- **FIR filters**: `FirFilter` with `design_lowpass` (windowed sinc) and `design_hilbert`
- **Dynamics**: `Compressor`, `Limiter`, `NoiseGate`, `Expander`, `DeEsser`, `MultibandCompressor`
- **Time effects**: `DelayLine`, `Chorus`, `Tremolo`, `Vibrato`, `Flanger`, `Phaser`
- **Reverb**: `Freeverb` (Jezar algorithm), `ConvolutionReverb` (OxiFFT overlap-save)
- **Pitch detection**: YIN (`detect_pitch_yin`), pYIN (`detect_pitch_pyin`) with Viterbi
- **Spectral features**: centroid, flux, rolloff, flatness, ZCR, bandwidth, chromagram,
  contrast, tonnetz
- **MFCC**: `mfcc(buf, n_mfcc, n_mels, n_fft, hop_size)`
- **Noise reduction**: `estimate_noise_profile`, `spectral_subtraction`, `wiener_filter`
- **Onset/rhythm**: `onset_strength_spectral_flux/hfc`, `pick_onset_peaks`,
  `estimate_tempo`, `detect_onsets`, `TempoEstimate`
- **Loudness (EBU R128)**: `k_weight`, `loudness_integrated`, `loudness_momentary`,
  `loudness_range` (LRA), `true_peak` (4× oversampling)
- **DspChain** builder: `DspChain::new().then(f).process(buf)`
- `AudioFilter` trait implemented for all effect types

### oxiaudio (facade)
#### Added
- `decode_file`, `decode_file_f64`, `decode_file_with_metadata`, `decode_stream`,
  `decode_stream_with_block_size`
- `encode_wav`, `encode_flac`, `encode_stream`, `encode_wav_with_config`,
  `encode_flac_with_config`, `encode_wav_f64`, `encode_aiff_file`
- `detect_format` — format detection from file header
- `convert(input, output)` — auto-detect format from extension
- `transcode_batch` — parallel batch conversion
- `probe_metadata(path)` — metadata without full decode
- `decode_files(paths)` — rayon parallel multi-file decode
- `dsp` module re-exporting all DSP types and functions
- `dsp::detect_tempo`, `dsp::detect_pitch` convenience wrappers
- `dsp::resample_quality(buf, rate, ResampleQuality)` with Fast/Good/Best
- `dsp::eq(buf, bands)` — quick parametric EQ
- `dsp::reverb` — convenience reverb wrapper
- `#[must_use]` on all Result-returning public functions
- All new core types re-exported: `AudioRingBuffer`, `AudioClock`, `Timestamp`,
  `AudioNode`, `AudioPipeline`, `ChannelMap`, `ChannelId`

### Added (M8–M23 incremental milestones, also in 0.1.0)

#### oxiaudio-decode (M8–M23)
- WavPack decoder: lossless/hybrid lossy, multi-channel (up to 8ch), correction file (.wvc), sample-accurate seek
- Musepack SV7/SV8 decoder: 32 subband decomposition, Huffman+quantization, ReplayGain header parsing
- MIDI file parser: SMF format 0/1/2, MThd/MTrk, variable-length delta time, meta events, note on/off, controller changes
- Streaming FLAC decoder improvements: gapless-playback trim via FLAC total_samples and granule position
- AIFF-C decoding: µ-law and A-law variants; 80-bit extended precision sample rate support
- APEv2 tag reading from WavPack/Musepack streams
- CuePoints extraction from FLAC CUESHEET metadata block and Vorbis comment CUESHEET field

#### oxiaudio-encode (M8–M23)
- RF64/BW64 WAV support for audio files exceeding 4 GB (ds64 chunk with 64-bit sizes)
- FLAC `METADATA_BLOCK_PICTURE` for album art (`FlacPicture`, `encode_flac_with_album_art`)
- APEv2 tag writer for WavPack/Musepack output (header/footer, UTF-8 key-value items)
- ID3v2.4 tag writer with UTF-8 encoding, APIC album art, USLT lyrics, extended header CRC, footer
- AIFF writer with NAME/AUTH/ANNO metadata chunks; streaming AIFF encoder with FORM size backfill
- Noise-shaped (ATH-weighted) dithering for perceptually optimal bit-depth reduction
- Two-pass encoding with EBU R128 loudness normalization (−14 LUFS, −16 LUFS, −23 LUFS targets)

#### oxiaudio-dsp (M8–M23)
- Surround channel layouts: `ChannelLayout::Quad`, `Surround51`, `Surround71`, `Surround51Side`, `Atmos714`
- `ChannelId` enum and `ChannelMap` with SMPTE/ITU-R BS.775 ordering (Vorbis, DTS, AAC, Film)
- `downmix_51_to_stereo`, `downmix_to_mono`, `upmix_mono_to_stereo` per ITU-R BS.775
- Phase vocoder: instantaneous frequency estimation, phase-locked `pitch_shift_pv`, `time_stretch`
- Channel vocoder (robotic voice effect): modulator spectral envelope applied to carrier
- `Expander`, `DeEsser`, `MultibandCompressor` dynamics processors; sidechain input support
- Early reflections model: image-source method, configurable room dimensions and reflection coefficients
- `ConvolutionReverb` via FFT-based overlap-save partitioned convolution (OxiFFT)
- Spectral noise reduction: `spectral_subtraction`, `wiener_filter`, `estimate_noise_profile`, per-bin frequency-domain noise gate
- pYIN probabilistic pitch detection with Viterbi decoding (`detect_pitch_pyin`)
- Autocorrelation-based pitch detection; `PitchTracker` with per-frame confidence and voiced/unvoiced
- Onset detection: `onset_strength_spectral_flux`, `onset_strength_hfc`, complex domain; adaptive peak picking
- Beat tracking, `TempoEstimator`, downbeat detection for bar/measure alignment
- `ParametricEq::frequency_response`, `phase_response`, `group_delay` analysis methods
- `WindowFn::Kaiser { beta }` and `WindowFn::FlatTop` window functions
- `spectral_contrast` and `tonnetz` spectral feature extractors; spectral bandwidth
- `MultichannelStftOutput` for per-channel independent STFT processing

#### oxiaudio facade (M8–M23)
- `TranscodeStream` streaming transcode pipeline (decode → optional DSP → encode)
- `transcode_batch` parallel batch format conversion via rayon
- `dsp::reverb` convenience wrapper; `dsp::detect_tempo`, `dsp::detect_pitch`
- `encode_aiff`, `probe_metadata`, `write_metadata`, `file_format` additions
- `convert_with_dsp(input, output, dsp_chain)` applying DSP during format conversion
- Re-exports: `AudioClock`, `Timestamp`, `AudioRingBuffer`, `BiquadFilter`, `ParametricEq`, `Compressor`, `PitchTracker`

#### oxiaudio-core (M8–M23)
- Compact binary IPC serialization for `AudioBuffer<f32>` (`ABUF` v1 magic+version header)
- `AudioBuffer::crossfade`, `mix_with`, `resample_linear`, `fade_in`, `fade_out`
- `AudioRingBuffer<T>` lock-free SPSC with wait-free overflow policy
- `AudioClock::drift_ppm`, `elapsed_secs`, `elapsed_frames`
- `AudioPipeline` parallel branch nodes with per-node bypass/mute and latency reporting
- Optional `serde` feature: JSON-serializable core types (AudioBuffer, AudioFormat, AudioMetadata, ChannelLayout, SampleFormat)

[0.2.1]: https://github.com/cool-japan/oxiaudio/releases/tag/v0.2.1
[0.2.0]: https://github.com/cool-japan/oxiaudio/releases/tag/v0.2.0
[0.1.3]: https://github.com/cool-japan/oxiaudio/releases/tag/v0.1.3
[0.1.2]: https://github.com/cool-japan/oxiaudio/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/oxiaudio/releases/tag/v0.1.1
