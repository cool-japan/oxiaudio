//! DSP utilities: resampling, gain control, normalization, silence trimming, spectral analysis,
//! pitch detection, dynamics processing, and effects — all re-exported from `oxiaudio-dsp`.

use oxiaudio_core::{AudioBuffer, OxiAudioError};

/// Resample the buffer to a new sample rate using a high-quality FFT-based algorithm.
///
/// # Examples
///
/// ```no_run
/// use oxiaudio::{decode_file, dsp};
/// let buf = decode_file(std::path::Path::new("audio.wav")).unwrap();
/// let resampled = dsp::resample(&buf, 44_100).unwrap();
/// assert_eq!(resampled.sample_rate, 44_100);
/// ```
#[must_use = "discarding the Result ignores resample errors"]
pub fn resample(
    buf: &AudioBuffer<f32>,
    target_rate: u32,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    oxiaudio_dsp::resample(buf, target_rate)
}

/// Apply a gain in decibels to all samples in place.
///
/// Positive dB amplifies; negative dB attenuates. 0 dB = unity gain.
///
/// # Examples
///
/// ```no_run
/// use oxiaudio::{decode_file, dsp};
/// let mut buf = decode_file(std::path::Path::new("audio.wav")).unwrap();
/// dsp::gain(&mut buf, 6.0); // approximately 2× amplitude
/// ```
pub fn gain(buf: &mut AudioBuffer<f32>, db: f32) {
    oxiaudio_dsp::gain(buf, db)
}

/// Peak-normalize in-place to `target_db` dBFS. Silent buffers are left unchanged.
///
/// # Examples
///
/// ```no_run
/// let mut buf = oxiaudio::decode_file(std::path::Path::new("audio.wav")).unwrap();
/// oxiaudio::dsp::normalize(&mut buf, -1.0);
/// ```
pub fn normalize(buf: &mut AudioBuffer<f32>, target_db: f32) {
    oxiaudio_dsp::normalize(buf, target_db)
}

/// Apply a gain factor directly in place (linear scale, not dB).
///
/// Use this when you already have a linear factor and want to avoid the
/// dB→linear conversion overhead of [`gain`].
///
/// # Examples
///
/// ```no_run
/// let mut buf = oxiaudio::decode_file(std::path::Path::new("audio.wav")).unwrap();
/// oxiaudio::dsp::gain_inplace(&mut buf, 0.5); // halve amplitude
/// ```
pub fn gain_inplace(buf: &mut AudioBuffer<f32>, factor: f32) {
    oxiaudio_dsp::gain_inplace(buf, factor)
}

/// Peak-normalize in place so that the maximum sample magnitude equals `target_peak` (0.0–1.0).
///
/// Uses the raw peak value rather than dBFS. Silent buffers are left unchanged.
/// Prefer this over [`normalize`] when you need precise linear amplitude control.
///
/// # Examples
///
/// ```no_run
/// let mut buf = oxiaudio::decode_file(std::path::Path::new("audio.wav")).unwrap();
/// oxiaudio::dsp::normalize_inplace(&mut buf, 0.9); // peak at 90%
/// ```
pub fn normalize_inplace(buf: &mut AudioBuffer<f32>, target_peak: f32) {
    oxiaudio_dsp::normalize_inplace(buf, target_peak)
}

/// Remove leading and trailing silent frames below `threshold_db` dBFS.
///
/// # Examples
///
/// ```no_run
/// use oxiaudio::{decode_file, dsp};
/// let buf = decode_file(std::path::Path::new("audio.wav")).unwrap();
/// let trimmed = dsp::trim_silence(&buf, -60.0);
/// assert!(trimmed.frame_count() <= buf.frame_count());
/// ```
pub fn trim_silence(buf: &AudioBuffer<f32>, threshold_db: f32) -> AudioBuffer<f32> {
    oxiaudio_dsp::trim_silence(buf, threshold_db)
}

/// Average all channels into a single mono channel.
///
/// # Examples
///
/// ```no_run
/// use oxiaudio::{decode_file, dsp, ChannelLayout};
/// let buf = decode_file(std::path::Path::new("audio.wav")).unwrap();
/// let mono = dsp::mix_to_mono(&buf);
/// assert_eq!(mono.channels, ChannelLayout::Mono);
/// ```
pub fn mix_to_mono(buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
    oxiaudio_dsp::mix_to_mono(buf)
}

/// De-interleave a stereo buffer into one `AudioBuffer<f32>` per channel (all Mono).
///
/// A Mono input returns a single-element `Vec` containing a copy of the buffer.
///
/// # Examples
///
/// ```no_run
/// use oxiaudio::{decode_file, dsp};
/// let buf = decode_file(std::path::Path::new("audio.wav")).unwrap();
/// let channels = dsp::split_channels(&buf);
/// assert!(channels.len() >= 1);
/// ```
pub fn split_channels(buf: &AudioBuffer<f32>) -> Vec<AudioBuffer<f32>> {
    oxiaudio_dsp::split_channels(buf)
}

/// Biquad filter (Direct Form II Transposed) with RBJ Audio EQ Cookbook constructors.
pub use oxiaudio_dsp::BiquadFilter;

/// A chain of [`BiquadFilter`]s applied in series.
pub use oxiaudio_dsp::ParametricEq;

/// Feed-forward compressor with soft-knee support and envelope following.
pub use oxiaudio_dsp::Compressor;

/// Hard limiter implemented as a compressor with a very high ratio (1000:1).
pub use oxiaudio_dsp::Limiter;

/// Noise gate that attenuates the signal when it falls below a threshold.
pub use oxiaudio_dsp::NoiseGate;

/// Digital delay line for echo and other time-based effects.
pub use oxiaudio_dsp::DelayLine;

/// Chorus effect: multiple pitch-modulated delay taps blended with dry signal.
pub use oxiaudio_dsp::Chorus;

/// Tremolo effect: low-frequency amplitude modulation.
pub use oxiaudio_dsp::Tremolo;

/// Vibrato effect: low-frequency pitch modulation via delay-line interpolation.
pub use oxiaudio_dsp::Vibrato;

/// Downward expander for noise-floor reduction (compressor complement).
pub use oxiaudio_dsp::Expander;

/// Flanger effect: short modulated delay with feedback.
pub use oxiaudio_dsp::Flanger;

/// Phaser effect: LFO-swept cascade of allpass filters.
pub use oxiaudio_dsp::Phaser;

/// Freeverb algorithmic reverberation (8 combs + 4 allpasses per channel).
pub use oxiaudio_dsp::Freeverb;

/// A cascade of biquad second-order sections (Butterworth / Chebyshev output).
pub use oxiaudio_dsp::Cascade;

/// Windowed-sinc FIR filter and its window selector.
pub use oxiaudio_dsp::{FirFilter, FirWindow};

/// Composable DSP processing chain.
pub use oxiaudio_dsp::DspChain;

/// YIN pitch tracker, per-frame estimate type, and one-shot full-resolution detector.
pub use oxiaudio_dsp::{detect_pitch_yin, PitchFrame, PitchTracker};

/// pYIN (probabilistic YIN) pitch detector with Viterbi decoding.
pub use oxiaudio_dsp::detect_pitch_pyin;

/// Onset detection: full detect_onsets pipeline.
pub use oxiaudio_dsp::detect_onsets;

/// Tempo estimation from audio.
pub use oxiaudio_dsp::estimate_tempo;

/// Tempo estimate result type (bpm, confidence, beat_times).
pub use oxiaudio_dsp::TempoEstimate;

/// Phase-vocoder pitch shift (preserves duration, shifts pitch).
pub use oxiaudio_dsp::pitch_shift_pv;

/// Phase-vocoder time stretch (preserves pitch, changes duration).
pub use oxiaudio_dsp::time_stretch;

/// Estimate a noise profile from an audio buffer.
pub use oxiaudio_dsp::estimate_noise_profile;

/// Spectral subtraction noise reduction.
pub use oxiaudio_dsp::spectral_subtraction;

/// Wiener filter noise reduction.
pub use oxiaudio_dsp::wiener_filter;

/// Apply TPDF noise-shaped dithering before integer quantization.
///
/// `bit_depth` should match the target encoding bit depth (e.g. 16 for WAV I16).
pub use oxiaudio_dsp::apply_tpdf_dither;

// ─── M16 DSP additions ────────────────────────────────────────────────────────

/// Channel vocoder: modulates a carrier signal with the spectral envelope of a modulator.
pub use oxiaudio_dsp::ChannelVocoder;

/// Complex-domain onset detection: computes onset strength using phase and magnitude deviation.
pub use oxiaudio_dsp::complex_domain_onset;

// ─── M19 — IR loading ────────────────────────────────────────────────────────

/// Load an impulse response from raw WAV bytes into an `AudioBuffer<f32>`.
///
/// Use the returned buffer with [`ConvolutionReverb`] to apply convolution reverb.
/// Only uncompressed PCM WAV (8-bit, 16-bit, 24-bit, or 32-bit integer) is supported.
///
/// # Errors
///
/// Returns [`OxiAudioError::UnsupportedFormat`] if the bytes are not a valid PCM WAV
/// or if the format is unsupported (e.g. float WAV, compressed).
pub use oxiaudio_dsp::effects::load_ir_from_wav_bytes;

// ─── M18 DSP additions ────────────────────────────────────────────────────────

/// Detect downbeats (measure/bar onsets) from an audio buffer given pre-computed beat times.
///
/// Returns a `Vec<f32>` of downbeat times in seconds. Requires a `tempo_bpm` estimate
/// and a `beats_per_bar` setting (typically 4 for 4/4 time).
pub use oxiaudio_dsp::detect_downbeats;

// ─── M17 DSP additions ────────────────────────────────────────────────────────

/// Early reflections reverb: models the first discrete echoes in a room before the diffuse tail.
pub use oxiaudio_dsp::EarlyReflections;

/// Autocorrelation-based pitch detector: returns the dominant frequency (Hz) for a mono frame.
pub use oxiaudio_dsp::detect_pitch_autocorr;

// ─── M15 DSP additions ────────────────────────────────────────────────────────

/// Frequency-domain noise gate: suppress frequency bins below `threshold` in each STFT frame.
pub use oxiaudio_dsp::frequency_domain_noise_gate;

/// Apply noise-shaped dithering at the target `bit_depth` with the given `sample_rate`.
///
/// Reduces audible quantization artifacts by shaping dither noise away from sensitive frequencies.
pub use oxiaudio_dsp::apply_noise_shaped_dither;

/// Compute a per-channel STFT, returning `Vec<Vec<Vec<[f32; 2]>>>` indexed by
/// `[channel][frame][bin]` where each bin is `[real, imag]`.
pub use oxiaudio_dsp::stft_multichannel;

/// De-esser: frequency-selective compressor targeting sibilance.
pub use oxiaudio_dsp::DeEsser;

/// Per-band settings for a MultibandCompressor.
pub use oxiaudio_dsp::BandSettings;

/// Multiband compressor with per-band processing.
pub use oxiaudio_dsp::MultibandCompressor;

/// Convolution reverb using an impulse response.
pub use oxiaudio_dsp::ConvolutionReverb;

/// Overlap-save partitioned convolution reverb for large impulse responses.
///
/// Unlike [`ConvolutionReverb`] (direct convolution), this struct splits the IR into
/// power-of-2 partitions and processes each independently — reducing per-block latency
/// for long IRs.
pub use oxiaudio_dsp::PartitionedConvolutionReverb;

/// Momentary loudness (EBU R128: 400ms blocks, 100ms hop, no gating).
pub use oxiaudio_dsp::loudness_momentary;

/// Momentary loudness using arbitrary non-overlapping windows.
pub use oxiaudio_dsp::loudness_momentary_windowed;

/// Peak level meter with configurable hold time and decay rate.
pub use oxiaudio_dsp::PeakMeter;

/// Windowed RMS level meter.
pub use oxiaudio_dsp::RmsMeter;

/// Butterworth / Chebyshev Type I filter design (cascaded biquad sections).
pub use oxiaudio_dsp::{
    butterworth_highpass, butterworth_lowpass, chebyshev1_highpass, chebyshev1_lowpass,
};

// ─── ResampleQuality / resample_quality ──────────────────────────────────

/// Quality hint for [`resample_quality`]. All levels currently use the same high-quality
/// sinc interpolation; Fast/Good/Best are reserved for future per-level tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleQuality {
    /// Fastest processing; smallest sinc kernel (future: lower oversampling factor).
    Fast,
    /// Balanced quality and speed (future: medium oversampling factor).
    Good,
    /// Highest quality; largest sinc kernel (future: highest oversampling factor).
    Best,
}

/// Resample `buf` to `target_rate` with a quality hint.
///
/// All quality levels currently delegate to the same high-quality sinc resampler in
/// `oxiaudio-dsp`. `Fast`/`Good`/`Best` are reserved for future per-level configuration.
#[must_use = "discarding the Result ignores resample errors"]
pub fn resample_quality(
    buf: &AudioBuffer<f32>,
    target_rate: u32,
    _quality: ResampleQuality,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    oxiaudio_dsp::resample(buf, target_rate)
}

/// Apply feed-forward compression to `buf` with the given parameters.
///
/// `threshold_db` (dBFS), `ratio` (e.g. 4.0 for 4:1), `attack_ms`, `release_ms`.
///
/// # Examples
///
/// ```no_run
/// let buf = oxiaudio::AudioBuffer { samples: vec![0.5f32; 44100], sample_rate: 44_100,
///     channels: oxiaudio::ChannelLayout::Mono, format: oxiaudio::SampleFormat::F32 };
/// let compressed = oxiaudio::dsp::compressor(&buf, -20.0, 4.0, 10.0, 100.0).unwrap();
/// ```
#[must_use = "returns the compressed audio buffer"]
pub fn compressor(
    buf: &AudioBuffer<f32>,
    threshold_db: f32,
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    Ok(oxiaudio_dsp::Compressor::new(threshold_db, ratio, attack_ms, release_ms).process(buf))
}

/// Apply Freeverb reverberation to `buf`.
///
/// `room_size` and `damping` are in `0.0..=1.0`; `wet` is the wet/dry mix.
pub fn reverb(buf: &AudioBuffer<f32>, room_size: f32, damping: f32, wet: f32) -> AudioBuffer<f32> {
    let mut fv = oxiaudio_dsp::Freeverb::new(buf.sample_rate);
    fv.room_size = room_size.clamp(0.0, 1.0);
    fv.damping = damping.clamp(0.0, 1.0);
    fv.wet = wet.clamp(0.0, 1.0);
    fv.dry = (1.0 - wet).clamp(0.0, 1.0);
    fv.process(buf)
}

/// Apply a multi-band parametric EQ to an audio buffer.
///
/// `bands`: each tuple is `(center_hz, gain_db, q)`.
///
/// # Examples
///
/// ```no_run
/// let buf = oxiaudio::AudioBuffer { samples: vec![0.0f32; 4410], sample_rate: 44_100,
///     channels: oxiaudio::ChannelLayout::Mono, format: oxiaudio::SampleFormat::F32 };
/// let result = oxiaudio::dsp::eq(&buf, &[(1000.0, 3.0, 1.0)]).unwrap();
/// ```
#[must_use = "returns the EQ-processed audio buffer"]
pub fn eq(
    buf: &AudioBuffer<f32>,
    bands: &[(f32, f32, f32)],
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let filters = bands
        .iter()
        .map(|&(freq, gain_db, q)| {
            oxiaudio_dsp::BiquadFilter::peaking_eq(freq, q, gain_db, buf.sample_rate)
        })
        .collect();
    Ok(oxiaudio_dsp::ParametricEq::new(filters).process(buf))
}

/// Apply a noise gate to the audio buffer.
///
/// Attenuates the signal when it falls below `threshold_db` dBFS.
/// The gate uses a state machine (Closed → Attack → Open → Hold → Release) with
/// the specified time constants in milliseconds.
///
/// # Example
/// ```no_run
/// let buf = oxiaudio::decode_file(std::path::Path::new("audio.wav")).unwrap();
/// let gated = oxiaudio::dsp::gate(&buf, -40.0, 5.0, 50.0, 100.0).unwrap();
/// ```
#[must_use = "returns the gated audio buffer"]
pub fn gate(
    buf: &AudioBuffer<f32>,
    threshold_db: f32,
    attack_ms: f32,
    hold_ms: f32,
    release_ms: f32,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let g = oxiaudio_dsp::NoiseGate {
        threshold_db,
        attack_ms,
        hold_ms,
        release_ms,
        range_db: -80.0,
    };
    Ok(g.process(buf))
}

/// Apply a delay/echo effect to the audio buffer.
///
/// `delay_ms` sets the echo delay, `feedback` controls the number of repeats
/// (clamped to `[0, 0.999]`), and `wet_dry` sets the wet/dry mix (`0.0` = fully
/// dry, `1.0` = fully wet).
///
/// # Example
/// ```no_run
/// let buf = oxiaudio::decode_file(std::path::Path::new("audio.wav")).unwrap();
/// let delayed = oxiaudio::dsp::delay(&buf, 250.0, 0.4, 0.5).unwrap();
/// ```
#[must_use = "returns the delay-processed audio buffer"]
pub fn delay(
    buf: &AudioBuffer<f32>,
    delay_ms: f32,
    feedback: f32,
    wet_dry: f32,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let d = oxiaudio_dsp::DelayLine::new(delay_ms, feedback, wet_dry);
    Ok(d.process(buf))
}

/// Apply a chorus effect to the audio buffer.
///
/// `rate_hz` is the LFO modulation rate, `depth_ms` is the modulation depth,
/// `voices` sets the number of chorus voices (clamped to 2–4), and `wet_dry`
/// controls the wet/dry mix.
///
/// # Example
/// ```no_run
/// let buf = oxiaudio::decode_file(std::path::Path::new("audio.wav")).unwrap();
/// let chorused = oxiaudio::dsp::chorus(&buf, 0.25, 5.0, 3, 0.5).unwrap();
/// ```
#[must_use = "returns the chorus-processed audio buffer"]
pub fn chorus(
    buf: &AudioBuffer<f32>,
    rate_hz: f32,
    depth_ms: f32,
    voices: usize,
    wet_dry: f32,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let c = oxiaudio_dsp::Chorus {
        rate_hz,
        depth_ms,
        voices,
        wet_dry,
    };
    Ok(c.process(buf))
}

/// Detect pitch frames using pYIN (probabilistic YIN with Viterbi decoding).
///
/// Returns one [`PitchFrame`] per analysis hop. Unvoiced frames have
/// `frequency_hz = 0.0` and `is_voiced = false`.
pub fn detect_pitch(buf: &AudioBuffer<f32>) -> Vec<PitchFrame> {
    oxiaudio_dsp::detect_pitch_pyin(buf, 2048, 512)
}

/// Detect tempo (BPM) from audio. Returns `(bpm, confidence)`.
///
/// Uses onset detection and inter-onset interval histogram analysis.
/// Returns `(0.0, 0.0)` if tempo cannot be determined.
pub fn detect_tempo(buf: &AudioBuffer<f32>) -> (f32, f32) {
    match estimate_tempo(buf, 2048, 512) {
        Ok(est) => (est.bpm, est.confidence),
        Err(_) => (0.0, 0.0),
    }
}

/// Measure integrated loudness per EBU R128 / ITU-R BS.1770-4, returning LUFS.
///
/// Returns [`f32::NEG_INFINITY`] if the signal is silent or shorter than one
/// 400 ms gating block.
///
/// # Examples
///
/// ```no_run
/// let buf = oxiaudio::decode_file(std::path::Path::new("audio.wav")).unwrap();
/// let lufs = oxiaudio::dsp::loudness_lufs(&buf);
/// println!("Integrated loudness: {lufs:.1} LUFS");
/// ```
pub fn loudness_lufs(buf: &AudioBuffer<f32>) -> f32 {
    oxiaudio_dsp::loudness_integrated(buf)
}

/// Peak-normalize `buf` to reach approximately `target_lufs` integrated loudness.
///
/// Measures the current integrated loudness using EBU R128 / ITU-R BS.1770-4,
/// then applies a gain to bring it to the target level. Returns the normalized buffer.
///
/// Returns the input buffer unchanged if it is silent (LUFS = -∞).
///
/// # Examples
///
/// ```no_run
/// let buf = oxiaudio::decode_file(std::path::Path::new("audio.wav")).unwrap();
/// let normalized = oxiaudio::dsp::normalize_loudness(&buf, -23.0);
/// ```
pub fn normalize_loudness(buf: &AudioBuffer<f32>, target_lufs: f32) -> AudioBuffer<f32> {
    let current_lufs = oxiaudio_dsp::loudness_integrated(buf);
    if !current_lufs.is_finite() {
        return buf.clone();
    }
    let gain_db = target_lufs - current_lufs;
    let mut out = buf.clone();
    oxiaudio_dsp::gain(&mut out, gain_db);
    out
}

/// Measure the loudness range (LRA) in LU per EBU R128 supplementary spec.
pub fn loudness_range(buf: &AudioBuffer<f32>) -> f32 {
    oxiaudio_dsp::loudness_range(buf)
}

/// Shift the pitch of a buffer by `semitones` (positive = up, negative = down).
///
/// Uses frequency-domain bin-scaling. Returns a mono buffer.
///
/// # Errors
///
/// Propagates any error from the internal STFT / iSTFT calls.
#[must_use = "discarding the Result ignores pitch-shift errors"]
pub fn pitch_shift(
    buf: &AudioBuffer<f32>,
    semitones: f32,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    oxiaudio_dsp::pitch_shift(buf, semitones)
}

/// Spectral analysis utilities: STFT, mel spectrogram, and related types.
pub mod spectral {
    use oxiaudio_core::{AudioBuffer, OxiAudioError};
    pub use oxiaudio_dsp::{
        chromagram, chromagram_normalized, harmonic_ratio, mfcc, spectral_bandwidth,
        spectral_centroid, spectral_contrast, spectral_crest_factor, spectral_entropy,
        spectral_flatness, spectral_flux, spectral_rolloff, zero_crossing_rate,
    };
    pub use oxiaudio_dsp::{Complex, StftOutput, WindowFn};

    /// Compute the Short-Time Fourier Transform of an `AudioBuffer<f32>`.
    ///
    /// The buffer is mixed to mono before analysis. Returns an [`StftOutput`]
    /// whose `frames[t][k]` is the complex spectrum at time-frame `t`, bin `k`.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::UnsupportedFormat`] when OxiFFT cannot produce a
    /// valid spectrogram (e.g. plan allocation failure).
    pub fn stft(
        buf: &AudioBuffer<f32>,
        window_size: usize,
        hop_size: usize,
        window_fn: WindowFn,
    ) -> Result<StftOutput, OxiAudioError> {
        oxiaudio_dsp::stft(buf, window_size, hop_size, window_fn)
    }

    /// Compute a log-mel spectrogram of an `AudioBuffer<f32>`.
    ///
    /// The buffer is mixed to mono. Returns `Vec<Vec<f32>>` of shape `[n_frames][n_mels]`
    /// with log-mel energies (natural log, floor at 1e-10).
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::UnsupportedFormat`] when the output is empty but the
    /// input had enough samples for at least one frame.
    pub fn melspectrogram(
        buf: &AudioBuffer<f32>,
        n_mels: usize,
        f_min: f32,
        f_max: f32,
        n_fft: usize,
        hop_size: usize,
    ) -> Result<Vec<Vec<f32>>, OxiAudioError> {
        oxiaudio_dsp::melspectrogram(buf, n_mels, f_min, f_max, n_fft, hop_size)
    }
}

// ─── M22 DSP additions ────────────────────────────────────────────────────────

/// Encode a stereo L/R buffer to mid-side (M/S) representation.
///
/// `mid = (L+R)*0.5`, `side = (L-R)*0.5`.
pub use oxiaudio_dsp::ms_encode;

/// Decode a mid-side (M/S) buffer back to stereo L/R.
///
/// `L = M+S`, `R = M-S`.
pub use oxiaudio_dsp::ms_decode;

/// Split an audio buffer at silence boundaries.
///
/// Returns a `Vec` of non-silent segments. Segments separated by silence runs
/// shorter than `min_silence_frames` are not split.
pub use oxiaudio_dsp::silence_split;

/// Short-time energy per hop frame.
///
/// Returns the mean-square energy of each frame of `frame_size` samples,
/// advancing by `hop` frames each step.
pub use oxiaudio_dsp::short_time_energy;

/// Normalize `buf` to a target integrated loudness (LUFS, EBU R128).
///
/// Measures the integrated loudness and applies the required gain.
/// If the measured loudness is below –70 LUFS, returns the buffer unchanged.
pub use oxiaudio_dsp::normalize_to_lufs;
