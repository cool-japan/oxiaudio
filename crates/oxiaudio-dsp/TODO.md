# oxiaudio-dsp TODO

## Status
Pure Rust DSP crate. Implements resampling (rubato sinc SIMD), gain, channel utilities, normalize/trim_silence, STFT/iSTFT/mel-spectrogram (OxiFFT), pitch shifting, biquad filters (RBJ: lowpass, highpass, bandpass, notch, allpass, peaking EQ, low/high shelf), parametric EQ, dynamics (Compressor, Limiter, NoiseGate), time-domain effects (DelayLine, Chorus, Tremolo, Vibrato), spectral features (centroid, flux, rolloff, flatness, ZCR, MFCC), EBU R128 loudness (k_weight, loudness_integrated, true_peak). M0-M5 complete. Codebase split into modules: biquad, spectral, dynamics, effects, loudness. All AudioFilter traits implemented. Zero clippy warnings. 34 tests all passing.

## Core Implementation

### Extended Filter Design
- [x] Add `BiquadFilter::lowpass(frequency: f32, q: f32, sample_rate: u32) -> Self` using RBJ Audio EQ Cookbook (~20 SLOC)
- [x] Add `BiquadFilter::highpass(frequency: f32, q: f32, sample_rate: u32) -> Self` (~20 SLOC)
- [x] Add `BiquadFilter::bandpass(frequency: f32, q: f32, sample_rate: u32) -> Self` (~20 SLOC)
- [x] Add `BiquadFilter::notch(frequency: f32, q: f32, sample_rate: u32) -> Self` (~20 SLOC)
- [x] Add `BiquadFilter::allpass(frequency: f32, q: f32, sample_rate: u32) -> Self` (~20 SLOC)
- [x] Implement Butterworth filter design: `butterworth_lowpass(order: usize, frequency: f32, sample_rate: u32) -> Vec<BiquadFilter>` as cascaded second-order sections from analog prototype via bilinear transform (~60 SLOC)
- [x] Implement Chebyshev Type I filter design: configurable passband ripple (dB), poles computed from prototype, cascaded SOS (~80 SLOC)
- [x] Implement Chebyshev Type II filter design: configurable stopband attenuation (dB), zeros placed at stopband rejection frequencies (~80 SLOC)
- [x] Implement elliptic (Cauer) filter design: both passband ripple and stopband attenuation specified, Landen transformation for computing zeros/poles, bilinear transform (~120 SLOC)
- [x] Implement FIR filter: `FirFilter { coefficients: Vec<f64>, delay_line: Vec<f64> }` with convolution-based processing (~50 SLOC)
- [x] Add `FirFilter::design_lowpass(num_taps: usize, cutoff: f32, sample_rate: u32, window: WindowFn) -> Self` using windowed sinc method with Blackman/Kaiser/Hamming window (~40 SLOC)
- [x] Add `FirFilter::design_hilbert(num_taps: usize) -> Self` for analytic signal computation (~30 SLOC)

### Parametric EQ Enhancements
- [x] Add `ParametricEq::graphic_eq(frequencies: &[f32], gains_db: &[f32], sample_rate: u32) -> Self` for ISO octave/third-octave band graphic equalizer (~25 SLOC)
- [x] Add `ParametricEq::frequency_response(freqs: &[f32]) -> Vec<f32>` computing magnitude response (dB) at specified frequencies by evaluating transfer function H(z) (~30 SLOC)
- [x] Add `ParametricEq::phase_response(freqs: &[f32]) -> Vec<f32>` computing phase response in radians (~25 SLOC)
- [x] Add `ParametricEq::group_delay(freqs: &[f32]) -> Vec<f32>` computing group delay in samples (~25 SLOC)

### Dynamics Processing
- [x] Implement `Compressor` struct with threshold (dBFS), ratio (1:1 to infinity:1), attack (ms), release (ms), knee width (dB soft knee), makeup gain (dB), envelope follower with peak/RMS detection modes (~80 SLOC)
- [x] Implement `Limiter` as compressor with ratio=infinity:1, instant attack (0ms lookahead), auto-release (~20 SLOC)
- [x] Implement `NoiseGate` with threshold (dBFS), attack (ms), hold (ms), release (ms), range (dB floor attenuation), and hysteresis (~60 SLOC)
- [x] Implement `Expander` with threshold, ratio (<1:1), attack, release for noise floor reduction (~50 SLOC)
- [x] Add sidechain input support: dynamics processor keyed by external signal frequency band (e.g., bass drum triggering ducker) (~30 SLOC)
- [x] Implement `DeEsser` with sibilant frequency range detection (2-10 kHz bandpass), dynamic compression on detected band, dry/wet crossover (~50 SLOC)
- [x] Implement `MultibandCompressor` with configurable crossover frequencies and independent compression per band (~100 SLOC)

### Reverb
- [x] Implement Freeverb algorithm: 8 parallel comb filters (tuned delays) + 4 series allpass filters (diffusion), configurable room size (0-1), damping (0-1), wet/dry mix, stereo spread (~120 SLOC)
- [x] Implement convolution reverb via FFT-based overlap-save partitioned convolution: `convolve(buf, impulse_response)` using OxiFFT, partition size = FFT size / 2, frequency-domain multiply-accumulate (~150 SLOC)
- [x] Add impulse response loading from WAV/FLAC files via oxiaudio-decode (~20 SLOC)
- [x] Implement early reflections model: configurable room dimensions (L/W/H in meters), reflection coefficients per wall, image-source method for first-order reflections (~80 SLOC)

### Time-Domain Effects
- [x] Implement delay line with configurable delay time (0-5000 ms), feedback (0-1), wet/dry mix, optional lowpass in feedback path (~40 SLOC)
- [x] Implement chorus effect: 2-4 modulated delay lines (10-30 ms) with independent sine LFOs at slightly different rates, depth control, mono/stereo output (~60 SLOC)
- [x] Implement flanger effect: short modulated delay (0-20 ms) with feedback, LFO (sine/triangle), depth, inverted phase option (~50 SLOC)
- [x] Implement phaser effect: 4-12 cascaded allpass filters with LFO-swept center frequency, feedback, stages count, stereo spread (~60 SLOC)
- [x] Implement vibrato: pitch modulation via fractional-delay interpolated delay line, LFO rate (0.1-10 Hz), depth (0-100 cents) (~30 SLOC)
- [x] Implement tremolo: amplitude modulation with sine/triangle/square LFO, rate (0.1-20 Hz), depth (0-100%) (~20 SLOC)

### Pitch Detection
- [x] Implement YIN pitch detection: difference function d(tau), cumulative mean normalized difference d'(tau), absolute threshold (default 0.1), parabolic interpolation at minimum (~120 SLOC)
- [x] Implement pYIN (probabilistic YIN): multiple threshold candidates, observation probability distribution, Viterbi decoding for globally optimal pitch track, voiced/unvoiced state transition (~180 SLOC)
- [x] Implement autocorrelation-based pitch detection: normalized autocorrelation, peak picking with minimum frequency constraint (~60 SLOC)
- [x] Add `PitchTracker` struct returning `Vec<PitchFrame>` with fields: time_seconds, frequency_hz, confidence (0.0-1.0), is_voiced (~30 SLOC)

### Tempo and Rhythm Detection
- [x] Implement onset detection functions: spectral flux (L2-norm of positive spectral difference), high-frequency content (frequency-weighted energy), complex domain (magnitude+phase deviation) (~100 SLOC)
- [x] Implement onset peak picking with adaptive threshold, minimum inter-onset interval, and pre/post-maximum constraints (~40 SLOC)
- [x] Implement beat tracking: inter-onset interval histogram, tempo hypothesis scoring, beat alignment via dynamic programming (~80 SLOC)
- [x] Add `TempoEstimator` returning BPM estimate with confidence and beat positions in seconds (~40 SLOC)
- [x] Implement downbeat detection for bar/measure alignment using spectral bass energy patterns (~50 SLOC)

### Spectral Features
- [x] Implement MFCC: mel-spectrogram -> natural log -> DCT-II, configurable n_mfcc (typically 13), optional delta and delta-delta coefficients (~60 SLOC)
- [x] Implement chromagram: 12-bin pitch class energy distribution from STFT magnitude, using constant-Q transform or STFT binning with log2 frequency mapping (~50 SLOC)
- [x] Implement spectral centroid: sum(k * |X(k)|^2) / sum(|X(k)|^2) per frame (~15 SLOC)
- [x] Implement spectral flux: L2 norm of half-wave rectified spectral difference between consecutive frames (~15 SLOC)
- [x] Implement spectral rolloff: frequency below which N% (default 85%) of total spectral energy is concentrated (~15 SLOC)
- [x] Implement spectral flatness (Wiener entropy): geometric_mean(|X(k)|^2) / arithmetic_mean(|X(k)|^2) (~15 SLOC)
- [x] Implement zero-crossing rate per frame for voiced/unvoiced speech discrimination (~10 SLOC)
- [x] Implement spectral bandwidth: weighted standard deviation of frequencies around the centroid (~15 SLOC)
- [x] Implement spectral contrast: peak-to-valley ratio in octave-spaced frequency sub-bands (~30 SLOC)
- [x] Implement tonnetz: 6-dimensional tonal centroid from chroma features for harmonic analysis (~25 SLOC)

### Phase Vocoder and Time-Stretching
- [x] Implement phase vocoder: instantaneous frequency estimation via phase difference tracking across STFT frames, phase accumulation with synthesis hop ratio, overlap-add reconstruction (~150 SLOC)
- [x] Upgrade `pitch_shift` to phase vocoder quality: replace current bin-interpolation (which produces artifacts at >6 semitones) with proper phase-locked pitch scaling (~80 SLOC refactor)
- [x] Implement time-stretching without pitch change via phase vocoder: modify synthesis hop while keeping analysis hop fixed (~30 SLOC wrapper)
- [x] Implement channel vocoder (robotic voice effect): analyze modulator spectral envelope, apply to carrier signal magnitude spectrum (~60 SLOC)

### Noise Reduction
- [x] Implement spectral subtraction: estimate noise floor magnitude from user-identified silence segment, subtract from signal magnitude spectrum with spectral flooring (minimum gain), reconstruct via iSTFT (~80 SLOC)
- [x] Implement Wiener filter noise reduction: estimate per-bin SNR from noise profile, compute optimal gain G(k) = max(1 - noise/signal, floor), apply in STFT domain (~100 SLOC)
- [x] Implement frequency-domain noise gate: per-bin gating with configurable threshold profile (ATH-shaped or flat) (~40 SLOC)
- [x] Add `estimate_noise_profile(silence_segment: &AudioBuffer<f32>, n_fft: usize) -> Vec<f32>` returning average magnitude per frequency bin over the silence segment (~30 SLOC)

### Loudness Measurement (EBU R128 / ITU-R BS.1770)
- [x] Implement K-weighting filter: pre-filter (high shelf at 1681.97 Hz, +3.999 dB) + RLB (revised low-frequency weighting, highpass at 38.135 Hz), both as biquad sections (~40 SLOC)
- [x] Implement gated loudness measurement: absolute gate at -70 LUFS, relative gate at -10 LU from ungated loudness, 400ms window for momentary, 3s for short-term (~80 SLOC)
- [x] Add `loudness_integrated(buf) -> f32` returning integrated loudness in LUFS (~15 SLOC)
- [x] Add `loudness_momentary(buf, window_ms: usize) -> Vec<f32>` returning per-window momentary loudness (~20 SLOC)
- [x] Add `loudness_range(buf) -> f32` (LRA) per EBU R128 s1 supplement using 10th-95th percentile of short-term loudness (~30 SLOC)
- [x] Add `true_peak(buf) -> f32` with 4x oversampling FIR interpolation per ITU-R BS.1770 Annex 2 (~40 SLOC)

## API Improvements
- [x] Add `DspChain` builder: `DspChain::new().add(biquad).add(compressor).add(reverb).process(buf) -> Result<AudioBuffer<f32>, OxiAudioError>` (~30 SLOC)
- [x] Implement `AudioFilter` trait for all new effect types to integrate with oxiaudio-core pipeline (~5 SLOC per effect)
- [x] Add `WindowFn::Kaiser { beta: f32 }` variant for adjustable main-lobe width vs side-lobe suppression (~15 SLOC)
- [x] Add `WindowFn::FlatTop` variant for amplitude-accurate spectrum analysis (~10 SLOC)
- [x] Add `#[must_use]` on all Result-returning public functions (~5 SLOC)
- [x] Add multichannel STFT: process each channel independently, return `Vec<StftOutput>` or `MultichannelStftOutput` instead of forcing mono mixdown (~30 SLOC)

## Testing
- [x] Test biquad lowpass: 1 kHz cutoff at 48 kHz should attenuate 10 kHz sine by >40 dB (~25 SLOC)
- [x] Test Butterworth vs Chebyshev: verify passband ripple and stopband attenuation match specifications (~30 SLOC)
- [x] Test compressor: input at -6 dBFS with threshold -12 dBFS and 4:1 ratio should output approximately -7.5 dBFS (~25 SLOC)
- [x] Test Freeverb: output is longer than input (reverb tail), wet signal peak < dry signal peak (~20 SLOC)
- [x] Test convolution reverb: convolving with unit impulse [1, 0, 0, ...] reproduces original signal (~15 SLOC)
- [x] Test YIN pitch detection: 440 Hz sine at 48 kHz detected within 1 Hz, confidence > 0.9 (~20 SLOC)
- [x] Test onset detection: synthesized click train with 500ms spacing detected with all onsets within 10 ms tolerance (~25 SLOC)
- [x] Test MFCC stability: same signal produces identical coefficients across repeated runs (~15 SLOC)
- [x] Test EBU R128: -23 LUFS calibration 1 kHz tone at correct RMS level produces reading within 0.1 LU (~20 SLOC)
- [x] Test spectral subtraction: SNR improves by at least 6 dB on white noise corrupted sine (~20 SLOC)
- [x] Benchmark biquad filter on 10s stereo 48 kHz buffer (~10 SLOC)
- [x] Benchmark FFT convolution reverb with 2s impulse response on 10s input (~10 SLOC)
- [x] Benchmark YIN pitch detection on 10s mono 48 kHz (~10 SLOC)
- [x] Benchmark EBU R128 loudness on 60s stereo 48 kHz (~10 SLOC)

## Performance
- [x] Implement SIMD-optimized biquad processing via `chunks_exact(4)` for f32x4 auto-vectorization (~15 SLOC)
- [x] Use partitioned convolution (overlap-save) for large impulse responses: partition size tuned to L1 cache for O(N log N) performance (~30 SLOC)
- [x] Cache OxiFFT plans across multiple STFT calls via `Arc<dyn oxifft::Fft<f32>>` plan pool (~15 SLOC)
- [x] Profile gain/normalize hot loops: verify compiler auto-vectorization is effective on x86_64 and aarch64 (~analysis task) — Both `gain` and `normalize` already used in-place `&mut AudioBuffer` signatures (no allocation per call). Switched inner loops to `iter_mut().for_each()` to give LLVM a cleaner auto-vectorization hint. Added `gain_inplace(factor: f32)` (skips dB→linear powf) and `normalize_inplace(target_peak: f32)` (skips dBFS powf) for hot-loop use. Benchmarked at 4 K/64 K/512 K elements: throughput ~330–900 Melem/s on aarch64 (M-series). Criterion bench group `gain` and `normalize` added to `benches/dsp_bench.rs`.
- [x] Optimize pitch detection: pre-compute autocorrelation via FFT for O(N log N) instead of O(N^2) brute-force (~20 SLOC)
- [~] Bench gate: SIMD resample path within 2x of libsamplerate (libsamplerate as dev-dep only, not shipped) (~analysis task) — rubato `Async::new_sinc` (sinc, Blackman, 128× oversampling, cubic interpolation) benchmarked at 1s stereo 48k→44.1k. libsamplerate not added as dev-dep (C dependency violates Pure Rust Policy). Rubato dispatches SSE2/AVX/NEON at runtime; a direct comparison against libsamplerate is blocked by the policy. Criterion bench `resample_sinc_48k_to_44100_1s_stereo` added to `benches/dsp_bench.rs` for future regression tracking.

## Integration
- [~] Integrate with oxisound for real-time DSP: audio input callback -> biquad/compressor/EQ chain -> output callback (~example; pending oxisound integration, not blockable by this crate alone)
- [~] Feed `StreamingDecoder` chunks through `DspChain` via `AudioSource`/`AudioSink` pipeline (~example; pending external integration, not blockable by this crate alone)
- [~] Provide spectral features (MFCC, chroma, spectral centroid) to ML crates for audio classification/similarity (~bridge API; pending ML crate availability, not blockable by this crate alone)
- [x] dasp signal graph adapter: `AudioBuffer -> dasp::signal::Signal<Frame = [f32; 2]>` for interop with dasp Envelope and Interpolator (~30 SLOC)
- [x] Ensure all spectral functions use OxiFFT, not rustfft, per COOLJAPAN policy (~dependency audit) — audited: no rustfft in source or Cargo.toml; OxiFFT already in use throughout
