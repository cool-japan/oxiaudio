# Changelog

All notable changes to OxiAudio are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [Unreleased]

<!-- No changes pending. All M0–M23 milestones shipped in 0.1.0. -->

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
