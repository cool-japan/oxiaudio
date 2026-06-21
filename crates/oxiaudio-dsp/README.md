# oxiaudio-dsp — Pure-Rust DSP (resampling, gain, spectral) for OxiAudio

[![Crates.io](https://img.shields.io/crates/v/oxiaudio-dsp.svg)](https://crates.io/crates/oxiaudio-dsp)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`oxiaudio-dsp` is the signal-processing layer of OxiAudio. It operates on `oxiaudio_core::AudioBuffer<f32>` and provides high-quality resampling, gain/normalization, biquad and FIR/IIR filtering, a dynamics suite (compressor, limiter, gate, expander, de-esser, multiband), a time/frequency effects rack (reverb, delay, chorus, flanger, phaser, vibrato, tremolo, vocoder, convolution), EBU-R128-style loudness metering, spectral analysis (STFT/ISTFT, MFCC, chroma, spectral descriptors), pitch tracking (YIN / pYIN / autocorrelation), rhythm/onset/tempo detection, phase-vocoder time-stretch and pitch-shift, dithering, mid/side encoding, and noise reduction.

The crate is `#![forbid(unsafe_code)]` and **100% Pure Rust**. High-quality resampling is delegated to [`rubato`](https://crates.io/crates/rubato), and all spectral work uses the pure-Rust [`oxifft`](https://crates.io/crates/oxifft) (COOLJAPAN FFT). SIMD acceleration (SSE2/AVX/NEON) is selected automatically at runtime where the backends support it.

## Installation

```toml
[dependencies]
oxiaudio-dsp = "0.2.0"

# With the dasp::Signal adapter (MonoSignal / StereoSignal):
oxiaudio-dsp = { version = "0.2.0", features = ["dasp"] }
```

## Quick Start

```rust
use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use oxiaudio_dsp::{resample, gain, normalize, BiquadFilter};

// A 1 s mono buffer at 48 kHz.
let buf = AudioBuffer {
    samples: vec![0.25f32; 48_000],
    sample_rate: 48_000,
    channels: ChannelLayout::Mono,
    format: SampleFormat::F32,
};

// High-quality sinc resample to 44.1 kHz.
let mut out = resample(&buf, 44_100)?;
assert_eq!(out.sample_rate, 44_100);

// Apply -3 dB of gain, then peak-normalize to 0 dBFS.
gain(&mut out, -3.0);
normalize(&mut out, 0.0);

// Run a 1 kHz low-pass biquad over the buffer.
let lp = BiquadFilter::lowpass(1_000.0, 0.707, 44_100);
let filtered = lp.process(&out);
let _ = filtered.frame_count();
# Ok::<(), oxiaudio_core::OxiAudioError>(())
```

## API Overview

### Resampling & basic gain (crate root)

| Function | Description |
|----------|-------------|
| `resample(&buf, target_rate)` | High-quality sinc (cubic, Blackman) resampler via `rubato` |
| `gain(&mut buf, db)` | Apply a dB gain in place |
| `gain_inplace(&mut buf, factor)` | Apply a linear gain factor in place (no dB conversion) |
| `normalize(&mut buf, target_db)` | Peak-normalize to a target dBFS |
| `normalize_inplace(&mut buf, target_peak)` | Peak-normalize to a linear peak (0.0–1.0) |
| `mix_to_mono(&buf)` | Average all channels to mono |
| `split_channels(&buf)` | De-interleave into one mono buffer per channel |
| `trim_silence(&buf, threshold_db)` | Remove leading/trailing silent frames |

### `DspChain` (`chain`)

A composable, fallible processing chain.

| Item | Description |
|------|-------------|
| `DspChain::new()` | Empty chain |
| `then(f)` | Append a closure step |
| `then_filter(filter)` | Append an `oxiaudio_core::AudioFilter` step |
| `process(&buf)` | Run the buffer through every step |
| `DspStep` | Type alias for a boxed chain step |

### Biquad & parametric EQ (`biquad`)

`BiquadFilter` constructors: `low_shelf`, `high_shelf`, `peaking_eq`, `lowpass`, `highpass`, `bandpass`, `notch`, `allpass`. Processing: `process(&buf)`, `process_buffer(&mut [f32])`, `process_multichannel(&buf)`.

`ParametricEq`: `new(bands)`, `graphic_eq(...)`, `process(&buf)`, `frequency_response(freqs, rate)`, `phase_response(freqs, rate)`, `group_delay(freqs, rate)`.

### Filters (`filters`, `filters_fir`, `filters_iir`)

| Item | Description |
|------|-------------|
| `Cascade` | Series of biquad sections: `new(sections)`, `process(&buf)` |
| `butterworth_lowpass(order, freq, rate)` | Butterworth LP `Cascade` |
| `butterworth_highpass(order, freq, rate)` | Butterworth HP `Cascade` |
| `chebyshev1_lowpass(...)` / `chebyshev1_highpass(...)` | Chebyshev Type-I `Cascade` |
| `chebyshev2_lowpass(...)` / `chebyshev2_highpass(...)` | Chebyshev Type-II filter (`Chebyshev2Filter`) |
| `elliptic_lowpass(...)` / `elliptic_highpass(...)` | Elliptic (Cauer) filter (`EllipticFilter`) |
| `FirFilter` | FIR filter: `new(coeffs)`, `design_lowpass(...)`, `design_highpass(...)`, `design_hilbert(taps)`, `process(&buf)` |
| `FirWindow` | `Rectangular`, `Hamming`, `Hann`, `Blackman`, `Kaiser { beta }` |
| `Chebyshev2Filter` / `EllipticFilter` | IIR filter handles with `process(&buf)` |

> Note: `BiquadFilter`, `ParametricEq`, `Cascade`, `FirFilter`, `FirWindow`, `Chebyshev2Filter`, `EllipticFilter`, and the `butterworth_*`/`chebyshev*`/`elliptic_*` constructors are all re-exported at the crate root.

### Dynamics (`dynamics`)

| Type | Constructor / notes |
|------|---------------------|
| `Compressor` | `new(threshold_db, ratio, attack_ms, release_ms)`, `with_knee`, `with_makeup`, `process`, `process_with_sidechain` |
| `Limiter` | `new(threshold_db, release_ms)`, `process` |
| `NoiseGate` | `new(threshold_db)`, `process` |
| `Expander` | `new(threshold_db, ratio, attack_ms, release_ms)`, `with_range`, `process` |
| `DeEsser` | `new(threshold_db, sample_rate)`, `process` |
| `MultibandCompressor` | `three_band(low, mid, high)`, `process`; `bands: Vec<BandSettings>` |
| `BandSettings` | `crossover_hz`, `threshold_db`, `ratio`, `attack_ms`, `release_ms`, `makeup_gain_db` |

### Effects (`effects`)

All effects expose `new(...)` and `process(&buf)`.

| Type | Description |
|------|-------------|
| `DelayLine` | Feedback delay: `new(delay_ms, feedback, wet_dry)` |
| `Freeverb` | Schroeder/Freeverb reverb: `new(sample_rate)` |
| `EarlyReflections` | Early-reflection tap network: `new()` |
| `ConvolutionReverb` | Impulse-response convolution: `new(impulse_response)` |
| `PartitionedConvolutionReverb` | Block-partitioned (low-latency) convolution |
| `Chorus` | `new(rate_hz, depth_ms)` |
| `Flanger` | `new(sample_rate)` |
| `Phaser` | `new(sample_rate)` |
| `Vibrato` | `new(rate_hz, depth_cents)` |
| `Tremolo` | `new(rate_hz, depth)` |
| `ChannelVocoder` | Spectral-envelope transfer: `new(n_fft, hop_size)`, `process(modulator, carrier)` |

### Loudness & metering (`loudness`)

| Function / type | Description |
|-----------------|-------------|
| `k_weight(&buf)` | Apply the K-weighting pre-filter |
| `loudness_integrated(&buf)` | Integrated loudness (LUFS) |
| `loudness_momentary(&buf)` | Momentary loudness over default windows |
| `loudness_momentary_windowed(&buf, window_ms)` | Momentary loudness, custom window |
| `loudness_range(&buf)` | Loudness range (LRA) |
| `true_peak(&buf)` | True-peak level (oversampled) |
| `normalize_to_lufs(...)` | Normalize a buffer to a target integrated loudness |
| `PeakMeter` | `new(hold_ms, decay_db_per_second, sample_rate)`, `process_block`, `peak_db`, `reset` |
| `RmsMeter` | `new(window_ms, sample_rate)`, `process_sample`, `rms`, `rms_db` |

### Spectral analysis (`spectral`)

| Item | Description |
|------|-------------|
| `Complex` | Re-export of `oxifft::Complex` |
| `WindowFn` | `Hann`, `Hamming`, `Blackman`, `Rectangular`, `Kaiser { beta }`, `FlatTop` |
| `StftOutput` | `frames: Vec<Vec<Complex<f32>>>`, `sample_rate`, `hop_size`, `window` |
| `stft(&buf, window_size, hop_size, ...)` | Short-Time Fourier Transform (mono) |
| `stft_multichannel(...)` | Per-channel STFT |
| `istft(&stft_out, original_len)` | Inverse STFT (overlap-add) |
| `melspectrogram(...)` | Mel-scaled spectrogram |
| `mfcc(&buf, n_mfcc, n_mels, n_fft, hop_size)` | Mel-frequency cepstral coefficients |
| `chromagram(...)` / `chromagram_normalized(...)` | 12-bin chroma features |
| `tonnetz(...)` | Tonal-centroid (Tonnetz) features |
| `pitch_shift(...)` | Spectral pitch shift |
| `spectral_centroid` / `spectral_bandwidth` / `spectral_rolloff` | Spectral shape descriptors |
| `spectral_flatness` / `spectral_crest_factor` / `spectral_entropy` | Spectral flatness measures |
| `spectral_flux` / `spectral_contrast` | Flux & contrast descriptors |
| `harmonic_ratio` | Harmonic-to-total energy ratio |
| `zero_crossing_rate` / `short_time_energy` | Time-domain frame descriptors |

### Pitch detection (`pitch`)

| Item | Description |
|------|-------------|
| `PitchTracker` | `new(frame_size, hop_size)`, `with_threshold`, `with_range`, `track(&buf)` |
| `PitchFrame` | `time_seconds`, `frequency_hz`, `confidence`, `is_voiced` |
| `detect_pitch_yin(...)` / `detect_pitch_yin_simple(&buf)` | YIN fundamental-frequency estimation |
| `detect_pitch_pyin(...)` | Probabilistic YIN (pYIN) |
| `detect_pitch_autocorr(...)` | Autocorrelation-based pitch detection |

### Rhythm, onsets & tempo (`rhythm`)

| Item | Description |
|------|-------------|
| `onset_strength_spectral_flux(...)` / `onset_strength_hfc(...)` | Onset-strength envelopes |
| `complex_domain_onset(...)` | Complex-domain onset detection |
| `pick_onset_peaks(...)` | Peak-pick an onset envelope |
| `detect_onsets(...)` / `detect_downbeats(...)` | Onset / downbeat positions |
| `estimate_tempo(...)` → `TempoEstimate` | Tempo + beat positions |
| `TempoEstimate` | `bpm`, `confidence`, `beat_times` |

### Phase vocoder (`pvocoder`)

| Function | Description |
|----------|-------------|
| `time_stretch(&buf, ratio, n_fft, hop_a)` | Phase-vocoder time-stretch (pitch preserved) |
| `pitch_shift_pv(...)` | Pitch-shift by N semitones (stretch + resample) |

### Other modules

| Item | Description |
|------|-------------|
| `silence_split(...)` (`segment`) | Split a buffer into non-silent segments |
| `ms_encode(&buf)` / `ms_decode(&buf)` (`stereo`) | Stereo ↔ mid/side conversion |
| `apply_tpdf_dither(&buf, bit_depth)` (`dither`) | TPDF dithering for bit-depth reduction |
| `apply_noise_shaped_dither(...)` (`dither`) | Noise-shaped dithering |
| `spectral_subtraction(...)` (`noise`) | Spectral-subtraction noise reduction |
| `wiener_filter(...)` (`noise`) | Wiener-filter noise reduction |
| `frequency_domain_noise_gate(...)` (`noise`) | Frequency-domain noise gate |
| `estimate_noise_profile(...)` (`noise`) | Estimate a noise magnitude profile |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `dasp` | off | Enables the `dasp_adapter` module: `MonoSignal` / `StereoSignal` wrappers implementing `dasp::Signal`, plus `mono_signal_to_buffer` / `stereo_signal_to_buffer` |

## Errors

Fallible functions return [`oxiaudio_core::OxiAudioError`]. Resampling and spectral allocation failures surface as `UnsupportedFormat`; channel-layout / sample-rate mismatches (e.g. in `ms_encode`) surface as `InvalidChannelLayout` / `InvalidSampleRate`.

## Related crates

| Crate | Role |
|-------|------|
| `oxiaudio-core` | `AudioBuffer`, `AudioFilter`, errors, layout/format types |
| `oxiaudio` | Top-level façade re-exporting the ecosystem |
| `oxiaudio-decode` | Symphonia-backed decoding + native readers |
| `oxiaudio-encode` | Encoders (WAV, FLAC, …) |

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
