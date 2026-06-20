# Changelog

All notable changes to OxiAudio are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

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

[0.1.3]: https://github.com/cool-japan/oxiaudio/releases/tag/v0.1.3
[0.1.2]: https://github.com/cool-japan/oxiaudio/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/oxiaudio/releases/tag/v0.1.1
