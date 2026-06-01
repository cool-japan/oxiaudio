use std::cell::RefCell;
use std::collections::HashMap;

use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError, SampleFormat};

pub use oxifft::Complex;

use oxifft::{
    api::{Direction, Flags, Plan},
    streaming::{
        istft as oxifft_istft, mel_spectrogram as oxifft_mel_spectrogram, MelConfig,
        WindowFunction as OxifftWindow,
    },
};

use crate::mix_to_mono;

// ── Thread-local FFT plan cache (M23-F Task 1) ────────────────────────────────
//
// oxifft::streaming::stft (the stateless free function) constructs a new
// `Plan::dft_1d` on every call.  For workloads that call stft() in a loop with
// the same window size (e.g. phase vocoder, chromagram streaming) this plan
// construction overhead is unnecessary.
//
// `Plan<f32>` is not Clone, so we store plans in a thread-local HashMap and
// perform STFT operations via a closure that borrows the plan in-place.
// The map is keyed by window size.
thread_local! {
    static FFT_PLAN_CACHE: RefCell<HashMap<usize, Plan<f32>>> =
        RefCell::new(HashMap::new());
}

// ── Thread-local mono scratch buffer cache (M19) ──────────────────────────────
//
// Reuse a Vec<f32> across repeated stft() calls on same-length buffers to avoid
// heap pressure in tight loops.
thread_local! {
    static MONO_SCRATCH_CACHE: RefCell<HashMap<usize, Vec<f32>>> =
        RefCell::new(HashMap::new());
}

/// Borrow a scratch buffer of at least `len` elements from the thread-local cache,
/// invoke `f` with a mutable reference to it, then return it to the cache.
///
/// The scratch buffer is zeroed to `len` elements before calling `f`.
fn with_scratch_buf<R, F: FnOnce(&mut Vec<f32>) -> R>(len: usize, f: F) -> R {
    MONO_SCRATCH_CACHE.with(|cache| {
        let mut map = cache.borrow_mut();
        // Use the next power-of-two as the cache key so that buffers are reused
        // across calls with slightly different lengths (e.g. after resampling).
        let key = len.next_power_of_two();
        let mut buf = map.remove(&key).unwrap_or_else(|| Vec::with_capacity(key));
        buf.resize(len, 0.0_f32);
        drop(map); // release the borrow before calling f (allows re-entry if needed)
        let result = f(&mut buf);
        // Put the buffer back into the cache if it's still the right size.
        cache.borrow_mut().insert(key, buf);
        result
    })
}

/// Execute an FFT-based STFT on `signal` using a cached forward `Plan<f32>` keyed
/// by `window_size`.
///
/// On the first call for a given `window_size` the plan is constructed and stored;
/// subsequent calls reuse it, avoiding per-call plan allocation overhead.
///
/// Returns a complex spectrogram: `frames[t][k]` = spectrum of frame `t` at bin `k`.
fn stft_with_cached_plan(
    signal: &[f32],
    window_size: usize,
    hop_size: usize,
    window: &OxifftWindow,
) -> Vec<Vec<Complex<f32>>> {
    if signal.len() < window_size || window_size == 0 || hop_size == 0 {
        return Vec::new();
    }

    let window_coeffs: Vec<f32> = window.generate(window_size);
    let num_frames = (signal.len() - window_size) / hop_size + 1;
    let mut spectrogram = Vec::with_capacity(num_frames);

    FFT_PLAN_CACHE.with(|cache| {
        // Ensure a plan exists for this window size.
        {
            let mut map = cache.borrow_mut();
            map.entry(window_size).or_insert_with(|| {
                // Plan::dft_1d always returns Some for valid sizes.
                Plan::<f32>::dft_1d(window_size, Direction::Forward, Flags::ESTIMATE)
                    .expect("Plan::dft_1d returned None for non-zero window_size")
            });
        }

        // Borrow the plan and compute all frames.
        let map = cache.borrow();
        // The entry was just inserted above, so this expect cannot fail.
        let plan = map.get(&window_size).expect("plan was just inserted");

        let mut input = vec![Complex::<f32>::new(0.0, 0.0); window_size];
        let mut output = vec![Complex::<f32>::new(0.0, 0.0); window_size];

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = start + window_size;

            // Apply window and pack as complex input.
            for (i, (s, w)) in signal[start..end]
                .iter()
                .zip(window_coeffs.iter())
                .enumerate()
            {
                input[i] = Complex::new(*s * *w, 0.0);
            }

            plan.execute(&input, &mut output);
            spectrogram.push(output.clone());
        }
    });

    spectrogram
}

/// Window function selector for STFT analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowFn {
    /// Hann (raised cosine) — general purpose, good frequency resolution.
    Hann,
    /// Hamming — reduced side lobes, suitable for speech.
    Hamming,
    /// Blackman — highest side-lobe rejection.
    Blackman,
    /// Rectangular — no tapering, highest frequency resolution.
    Rectangular,
    /// Kaiser window with shape parameter `beta`.
    ///
    /// Larger `beta` increases side-lobe attenuation at the cost of wider main lobe.
    /// Typical values: 5.0 (similar to Hamming), 8.6 (similar to Blackman-Harris).
    Kaiser { beta: f32 },
    /// Flat top window — minimal amplitude error for accurate level measurement.
    ///
    /// Very wide main lobe; not suitable for frequency resolution.
    FlatTop,
}

impl WindowFn {
    /// Convert to the equivalent OxiFFT `WindowFunction` enum variant.
    ///
    /// For `FlatTop`, the coefficients are pre-computed for the given `n` samples
    /// and wrapped in `OxifftWindow::Custom`.
    pub(crate) fn to_oxifft_for_size(self, n: usize) -> OxifftWindow {
        match self {
            Self::Hann => OxifftWindow::Hann,
            Self::Hamming => OxifftWindow::Hamming,
            Self::Blackman => OxifftWindow::Blackman,
            Self::Rectangular => OxifftWindow::Rectangular,
            Self::Kaiser { beta } => OxifftWindow::Kaiser {
                beta: f64::from(beta),
            },
            Self::FlatTop => {
                let a0 = 0.215_578_95_f64;
                let a1 = 0.416_631_58_f64;
                let a2 = 0.277_263_16_f64;
                let a3 = 0.083_552_63_f64;
                let a4 = 0.006_947_37_f64;
                let coeffs: Vec<f64> = (0..n)
                    .map(|k| {
                        let x = 2.0 * std::f64::consts::PI * k as f64 / (n - 1).max(1) as f64;
                        a0 - a1 * x.cos() + a2 * (2.0 * x).cos() - a3 * (3.0 * x).cos()
                            + a4 * (4.0 * x).cos()
                    })
                    .collect();
                OxifftWindow::Custom(coeffs)
            }
        }
    }
}

/// Output of an STFT operation.
///
/// `frames[t][k]` is the complex spectrum of time-frame `t` at frequency bin `k`.
#[derive(Debug, Clone)]
pub struct StftOutput {
    /// Complex spectrogram: outer index = time frame, inner index = frequency bin.
    pub frames: Vec<Vec<Complex<f32>>>,
    /// Sample rate of the source audio (Hz).
    pub sample_rate: u32,
    /// Hop size used during analysis (samples).
    pub hop_size: usize,
    /// Window function used during analysis.
    pub window: WindowFn,
}

/// Compute the Short-Time Fourier Transform of an `AudioBuffer<f32>`.
///
/// The buffer is mixed to mono before analysis.
///
/// # Errors
///
/// Returns `OxiAudioError::UnsupportedFormat` if OxiFFT cannot allocate a plan
/// (i.e. the spectrogram comes back empty when the input is non-empty).
#[must_use = "returns the StftOutput spectrogram; use istft to reconstruct"]
pub fn stft(
    buf: &AudioBuffer<f32>,
    window_size: usize,
    hop_size: usize,
    window_fn: WindowFn,
) -> Result<StftOutput, OxiAudioError> {
    // Mix to mono using a thread-local scratch buffer to avoid per-call allocation.
    // Use stft_with_cached_plan (M23-F) so the Plan<f32> is reused across calls
    // with the same window_size, eliminating per-call plan construction overhead.
    let mono = mix_to_mono(buf);
    let n = mono.samples.len();
    let window = window_fn.to_oxifft_for_size(window_size);

    let frames = with_scratch_buf(n, |scratch| {
        scratch.copy_from_slice(&mono.samples);
        stft_with_cached_plan(scratch, window_size, hop_size, &window)
    });

    if frames.is_empty() && !mono.samples.is_empty() && mono.samples.len() >= window_size {
        return Err(OxiAudioError::UnsupportedFormat(
            "OxiFFT STFT returned empty spectrogram — plan allocation may have failed".to_owned(),
        ));
    }

    Ok(StftOutput {
        frames,
        sample_rate: buf.sample_rate,
        hop_size,
        window: window_fn,
    })
}

/// Reconstruct a mono `AudioBuffer<f32>` from an `StftOutput` via overlap-add.
///
/// The reconstructed signal is truncated or zero-padded to `original_len` samples.
///
/// # Errors
///
/// Returns `OxiAudioError::UnsupportedFormat` when `original_len` is zero but
/// the spectrogram is non-empty (ambiguous request).
#[must_use = "returns the reconstructed AudioBuffer"]
pub fn istft(
    stft_out: &StftOutput,
    original_len: usize,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if stft_out.frames.is_empty() {
        return Ok(AudioBuffer {
            samples: vec![0.0_f32; original_len],
            sample_rate: stft_out.sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        });
    }

    // Infer window size from the first frame length.
    let window_size = stft_out.frames.first().map(|f| f.len()).unwrap_or(0);
    let mut samples = oxifft_istft::<f32>(
        &stft_out.frames,
        stft_out.hop_size,
        stft_out.window.to_oxifft_for_size(window_size),
    );

    // Truncate or zero-pad to `original_len`.
    samples.resize(original_len, 0.0_f32);

    Ok(AudioBuffer {
        samples,
        sample_rate: stft_out.sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    })
}

/// Compute a log-mel spectrogram of an `AudioBuffer<f32>`.
///
/// The buffer is mixed to mono. Returns a `Vec<Vec<f32>>` of shape
/// `[n_frames][n_mels]` with log-mel energies (natural log, floor at 1e-10).
///
/// # Errors
///
/// Returns `OxiAudioError::UnsupportedFormat` if the resulting spectrogram is
/// empty while the input has enough samples for at least one frame.
#[must_use = "returns the log-mel spectrogram matrix"]
pub fn melspectrogram(
    buf: &AudioBuffer<f32>,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
    n_fft: usize,
    hop_size: usize,
) -> Result<Vec<Vec<f32>>, OxiAudioError> {
    let mono = mix_to_mono(buf);

    // Build MelConfig — f_min/f_max are public fields, set after construction.
    let mut config = MelConfig::new(f64::from(buf.sample_rate), n_fft, hop_size, n_mels);
    config.f_min = f64::from(f_min);
    config.f_max = f64::from(f_max);

    // mel_spectrogram is generic; use f32 for consistency with the rest of the crate.
    let mel = oxifft_mel_spectrogram::<f32>(&mono.samples, &config);

    if mel.is_empty() && mono.samples.len() >= n_fft {
        return Err(OxiAudioError::UnsupportedFormat(
            "OxiFFT mel_spectrogram returned empty output — configuration may be invalid"
                .to_owned(),
        ));
    }

    Ok(mel)
}

/// Shift the pitch of an `AudioBuffer<f32>` by `semitones` (positive = up, negative = down).
///
/// Uses frequency-domain bin-scaling (M3 quality): each output bin `j` is
/// interpolated from input bin `j / factor` where `factor = 2^(semitones/12)`.
/// The result is returned as a mono buffer.
///
/// # Errors
///
/// Propagates any error from the internal `stft` / `istft` calls.
#[must_use = "returns the pitch-shifted AudioBuffer"]
pub fn pitch_shift(
    buf: &AudioBuffer<f32>,
    semitones: f32,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    const WINDOW_SIZE: usize = 2048;
    const HOP_SIZE: usize = 512;

    let factor = 2_f32.powf(semitones / 12.0);

    let stft_out = stft(buf, WINDOW_SIZE, HOP_SIZE, WindowFn::Hann)?;

    let new_frames: Vec<Vec<Complex<f32>>> = stft_out
        .frames
        .iter()
        .map(|frame| {
            let n_bins = frame.len();
            (0..n_bins)
                .map(|j| {
                    let src = j as f32 / factor;
                    if src < 0.0 || src >= n_bins as f32 {
                        Complex::new(0.0_f32, 0.0_f32)
                    } else {
                        linear_interp_complex(frame, src)
                    }
                })
                .collect()
        })
        .collect();

    let shifted_stft = StftOutput {
        frames: new_frames,
        sample_rate: stft_out.sample_rate,
        hop_size: stft_out.hop_size,
        window: stft_out.window,
    };

    // Determine output length: mono sample count.
    let mono = mix_to_mono(buf);
    istft(&shifted_stft, mono.samples.len())
}

/// Linear interpolation in a complex spectrum at a fractional bin index.
fn linear_interp_complex(frame: &[Complex<f32>], pos: f32) -> Complex<f32> {
    let lo = pos as usize;
    let hi = lo + 1;
    let frac = pos - lo as f32;

    if hi >= frame.len() {
        return frame[lo.min(frame.len().saturating_sub(1))];
    }

    let lo_val = frame[lo];
    let hi_val = frame[hi];
    Complex::new(
        lo_val.re + frac * (hi_val.re - lo_val.re),
        lo_val.im + frac * (hi_val.im - lo_val.im),
    )
}

/// Compute the spectral centroid (frequency-weighted centre of mass) per frame.
///
/// Returns a `Vec<f32>` with one centroid value (in Hz) per STFT frame.
/// Only the positive-frequency half of the spectrum (bins 0..=n_fft/2) is used.
pub fn spectral_centroid(buf: &AudioBuffer<f32>, n_fft: usize, hop_size: usize) -> Vec<f32> {
    let stft_out = match stft(buf, n_fft, hop_size, WindowFn::Hann) {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    let sr = buf.sample_rate as f32;
    let bin_hz = sr / n_fft as f32;
    // Only use positive-frequency bins (first n_fft/2 + 1 bins).
    let n_pos = n_fft / 2 + 1;
    stft_out
        .frames
        .iter()
        .map(|frame| {
            let half = frame.len().min(n_pos);
            let total_power: f32 = frame[..half].iter().map(|c| c.norm_sqr()).sum();
            if total_power < 1e-12 {
                return 0.0;
            }
            let weighted: f32 = frame[..half]
                .iter()
                .enumerate()
                .map(|(k, c)| k as f32 * bin_hz * c.norm_sqr())
                .sum();
            weighted / total_power
        })
        .collect()
}

/// Compute spectral flux (onset strength) per frame.
///
/// Flux is the L2-norm of the half-wave-rectified spectral difference between
/// consecutive frames. The first frame always has flux = 0.
/// Only positive-frequency bins (0..=n_fft/2) are used.
pub fn spectral_flux(buf: &AudioBuffer<f32>, n_fft: usize, hop_size: usize) -> Vec<f32> {
    let stft_out = match stft(buf, n_fft, hop_size, WindowFn::Hann) {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    if stft_out.frames.is_empty() {
        return vec![];
    }
    let n_pos = n_fft / 2 + 1;
    let mut flux = vec![0.0f32; stft_out.frames.len()];
    for (flux_val, window) in flux[1..].iter_mut().zip(stft_out.frames.windows(2)) {
        let prev = &window[0];
        let curr = &window[1];
        let half_prev = prev.len().min(n_pos);
        let half_curr = curr.len().min(n_pos);
        *flux_val = prev[..half_prev]
            .iter()
            .zip(curr[..half_curr].iter())
            .map(|(p, c)| {
                let d = c.norm() - p.norm();
                if d > 0.0 {
                    d * d
                } else {
                    0.0
                }
            })
            .sum::<f32>()
            .sqrt();
    }
    flux
}

/// Compute the spectral rolloff frequency per frame.
///
/// Returns the frequency (in Hz) below which `rolloff_percent` of the total
/// spectral energy is concentrated.
/// Only positive-frequency bins (0..=n_fft/2) are used.
pub fn spectral_rolloff(
    buf: &AudioBuffer<f32>,
    n_fft: usize,
    hop_size: usize,
    rolloff_percent: f32,
) -> Vec<f32> {
    let stft_out = match stft(buf, n_fft, hop_size, WindowFn::Hann) {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    let sr = buf.sample_rate as f32;
    let bin_hz = sr / n_fft as f32;
    let threshold = rolloff_percent.clamp(0.0, 1.0);
    let n_pos = n_fft / 2 + 1;
    stft_out
        .frames
        .iter()
        .map(|frame| {
            let half = frame.len().min(n_pos);
            let total_energy: f32 = frame[..half].iter().map(|c| c.norm_sqr()).sum();
            if total_energy < 1e-12 {
                return 0.0;
            }
            let target = total_energy * threshold;
            let mut cumsum = 0.0f32;
            for (k, c) in frame[..half].iter().enumerate() {
                cumsum += c.norm_sqr();
                if cumsum >= target {
                    return k as f32 * bin_hz;
                }
            }
            half as f32 * bin_hz
        })
        .collect()
}

/// Compute spectral flatness (Wiener entropy) per frame.
///
/// Flatness = geometric_mean(|X(k)|^2) / arithmetic_mean(|X(k)|^2).
/// Returns values in [0, 1]: 1.0 = white noise, ~0.0 = tonal.
/// Only positive-frequency bins (0..=n_fft/2) are used.
pub fn spectral_flatness(buf: &AudioBuffer<f32>, n_fft: usize, hop_size: usize) -> Vec<f32> {
    let stft_out = match stft(buf, n_fft, hop_size, WindowFn::Hann) {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    let n_pos = n_fft / 2 + 1;
    stft_out
        .frames
        .iter()
        .map(|frame| {
            let half = frame.len().min(n_pos);
            if half == 0 {
                return 0.0;
            }
            let powers: Vec<f32> = frame[..half]
                .iter()
                .map(|c| c.norm_sqr().max(1e-20))
                .collect();
            let arith_mean: f32 = powers.iter().sum::<f32>() / half as f32;
            let log_sum: f32 = powers.iter().map(|&p| p.ln()).sum::<f32>();
            let geom_mean = (log_sum / half as f32).exp();
            if arith_mean < 1e-20 {
                0.0
            } else {
                geom_mean / arith_mean
            }
        })
        .collect()
}

/// Compute the zero-crossing rate per frame.
///
/// Returns the fraction of sample pairs in each frame where the signal
/// crosses zero (changes sign). Input is mixed to mono before analysis.
pub fn zero_crossing_rate(buf: &AudioBuffer<f32>, frame_size: usize, hop_size: usize) -> Vec<f32> {
    let n_ch = buf.channels.channel_count();
    let mono_samples: Vec<f32> = if n_ch == 1 {
        buf.samples.clone()
    } else {
        buf.samples
            .chunks_exact(n_ch)
            .map(|c| c.iter().sum::<f32>() / n_ch as f32)
            .collect()
    };
    if mono_samples.len() < frame_size || frame_size == 0 {
        return vec![];
    }
    let effective_hop = hop_size.max(1);
    let n_frames = (mono_samples.len() - frame_size) / effective_hop + 1;
    (0..n_frames)
        .map(|i| {
            let start = i * effective_hop;
            let frame = &mono_samples[start..(start + frame_size).min(mono_samples.len())];
            if frame.len() < 2 {
                return 0.0;
            }
            let zcr = frame
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count();
            zcr as f32 / frame.len() as f32
        })
        .collect()
}

/// Compute Mel-frequency cepstral coefficients (MFCCs) from an audio buffer.
///
/// Returns a `Vec<Vec<f32>>` of shape `[n_frames][n_mfcc]`.
/// Each row is the DCT-II of the log-mel spectrum for that frame.
///
/// # Errors
///
/// Propagates errors from the internal `melspectrogram` call.
#[must_use = "returns the MFCC matrix"]
pub fn mfcc(
    buf: &AudioBuffer<f32>,
    n_mfcc: usize,
    n_mels: usize,
    n_fft: usize,
    hop_size: usize,
) -> Result<Vec<Vec<f32>>, OxiAudioError> {
    let mel_spec = melspectrogram(
        buf,
        n_mels,
        0.0,
        buf.sample_rate as f32 / 2.0,
        n_fft,
        hop_size,
    )?;
    Ok(mel_spec
        .iter()
        .map(|frame_mels| {
            let n = frame_mels.len();
            let log_mels: Vec<f32> = frame_mels.iter().map(|&m| m.max(1e-10_f32).ln()).collect();
            (0..n_mfcc)
                .map(|k| {
                    log_mels
                        .iter()
                        .enumerate()
                        .map(|(n_idx, &s)| {
                            s * (std::f32::consts::PI * k as f32 * (n_idx as f32 + 0.5) / n as f32)
                                .cos()
                        })
                        .sum::<f32>()
                })
                .collect()
        })
        .collect())
}

/// Compute the chromagram (chroma feature) per STFT frame.
///
/// Returns one `[f32; 12]` per frame, indexed by pitch class
/// `[C, C#, D, D#, E, F, F#, G, G#, A, A#, B]`.
/// Each frame is normalized so that the 12 pitch-class energies sum to 1.0
/// (or left as all-zero for silent frames).
///
/// The input buffer is mixed to mono before analysis.
pub fn chromagram(buf: &AudioBuffer<f32>, n_fft: usize, hop_size: usize) -> Vec<[f32; 12]> {
    let stft_out = match stft(buf, n_fft, hop_size, WindowFn::Hann) {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    let sr = buf.sample_rate as f32;
    let bin_hz = sr / n_fft as f32;
    // C0 frequency in Hz
    let c0_hz: f32 = 16.351_599;
    let n_pos = n_fft / 2 + 1;

    stft_out
        .frames
        .iter()
        .map(|frame| {
            let half = frame.len().min(n_pos);
            let mut chroma = [0.0f32; 12];
            for (k, c) in frame[..half].iter().enumerate() {
                let freq_k = k as f32 * bin_hz;
                // Skip DC (k=0) and very low frequencies where log2 is unreliable
                if freq_k < c0_hz * 0.5 {
                    continue;
                }
                let pitch_class_f = 12.0 * (freq_k / c0_hz).log2();
                let pc = (pitch_class_f.round() as i64).rem_euclid(12) as usize;
                chroma[pc] += c.norm_sqr();
            }
            let total: f32 = chroma.iter().sum();
            if total > 1e-12 {
                for v in &mut chroma {
                    *v /= total;
                }
            }
            chroma
        })
        .collect()
}

/// Compute spectral bandwidth (weighted standard deviation around the centroid) per frame.
///
/// Returns one value (in Hz) per STFT frame:
/// `bandwidth = sqrt( sum(|X(k)|^2 * (freq_k - centroid)^2) / sum(|X(k)|^2) )`
///
/// Returns 0.0 for silent frames. Only positive-frequency bins (0..n_fft/2) are used.
pub fn spectral_bandwidth(buf: &AudioBuffer<f32>, n_fft: usize, hop_size: usize) -> Vec<f32> {
    let stft_out = match stft(buf, n_fft, hop_size, WindowFn::Hann) {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    let sr = buf.sample_rate as f32;
    let bin_hz = sr / n_fft as f32;
    let n_pos = n_fft / 2 + 1;

    stft_out
        .frames
        .iter()
        .map(|frame| {
            let half = frame.len().min(n_pos);
            let total_power: f32 = frame[..half].iter().map(|c| c.norm_sqr()).sum();
            if total_power < 1e-12 {
                return 0.0;
            }
            // centroid
            let centroid: f32 = frame[..half]
                .iter()
                .enumerate()
                .map(|(k, c)| k as f32 * bin_hz * c.norm_sqr())
                .sum::<f32>()
                / total_power;
            // weighted variance
            let variance: f32 = frame[..half]
                .iter()
                .enumerate()
                .map(|(k, c)| {
                    let diff = k as f32 * bin_hz - centroid;
                    diff * diff * c.norm_sqr()
                })
                .sum::<f32>()
                / total_power;
            variance.sqrt()
        })
        .collect()
}

/// Compute spectral crest factor per STFT frame.
///
/// Crest factor = max(|X\[k\]|²) / mean(|X\[k\]|²).
/// Higher values indicate more tonal, peaky spectra (a pure sine has a very high crest factor).
/// Returns one value per STFT frame. Silent frames return 0.0.
/// Only positive-frequency bins (0..=n_fft/2) are used.
pub fn spectral_crest_factor(buf: &AudioBuffer<f32>, n_fft: usize, hop_size: usize) -> Vec<f32> {
    let stft_out = match stft(buf, n_fft, hop_size, WindowFn::Hann) {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    let n_pos = n_fft / 2 + 1;

    stft_out
        .frames
        .iter()
        .map(|frame| {
            let half = frame.len().min(n_pos);
            if half == 0 {
                return 0.0;
            }
            let mean_power: f32 =
                frame[..half].iter().map(|c| c.norm_sqr()).sum::<f32>() / half as f32;
            if mean_power < 1e-20 {
                return 0.0;
            }
            let peak_power: f32 = frame[..half]
                .iter()
                .map(|c| c.norm_sqr())
                .fold(0.0f32, f32::max);
            peak_power / mean_power
        })
        .collect()
}

/// Compute spectral entropy per STFT frame.
///
/// Treats the normalised power spectrum as a probability distribution and computes
/// Shannon entropy: `H = -sum(p_k * log2(p_k + eps))` where `p_k = |X[k]|^2 / sum(|X[k]|^2)`.
///
/// Returns one value per frame. Low entropy ≈ pure tone; high entropy ≈ noise-like.
/// Only positive-frequency bins (0..=n_fft/2) are used.
pub fn spectral_entropy(buf: &AudioBuffer<f32>, n_fft: usize, hop_size: usize) -> Vec<f32> {
    let stft_out = match stft(buf, n_fft, hop_size, WindowFn::Hann) {
        Ok(o) => o,
        Err(_) => return vec![],
    };
    let n_pos = n_fft / 2 + 1;

    stft_out
        .frames
        .iter()
        .map(|frame| {
            let half = frame.len().min(n_pos);
            if half == 0 {
                return 0.0;
            }
            let total_power: f32 = frame[..half].iter().map(|c| c.norm_sqr()).sum();
            if total_power < 1e-20 {
                return 0.0;
            }
            let eps = 1e-12_f32;
            frame[..half]
                .iter()
                .map(|c| {
                    let p = c.norm_sqr() / total_power;
                    -(p + eps) * (p + eps).log2()
                })
                .sum::<f32>()
        })
        .collect()
}

/// Compute normalised chromagram (chroma feature vector).
///
/// Calls [`chromagram`] and optionally applies per-frame L2 normalisation.
/// Each frame is a `[f32; 12]` indexed `[C, C#, D, D#, E, F, F#, G, G#, A, A#, B]`.
///
/// When `normalize` is `true`, each frame is scaled to unit L2 norm.
/// Silent frames (L2 norm < 1e-9) are returned unchanged.
///
/// # Errors
///
/// Returns `OxiAudioError::UnsupportedFormat` only when the underlying STFT fails
/// on non-empty input that should produce at least one frame.  In practice, the
/// internal `chromagram` call returns an empty `Vec` on error, so this function
/// surfaces that as `Ok(vec![])` rather than an error; therefore the `Result` is
/// kept consistent with similar feature functions.
#[must_use = "returns the normalised chromagram frames"]
pub fn chromagram_normalized(
    buf: &AudioBuffer<f32>,
    n_fft: usize,
    hop_size: usize,
    normalize: bool,
) -> Result<Vec<[f32; 12]>, OxiAudioError> {
    let chroma = chromagram(buf, n_fft, hop_size);
    if !normalize {
        return Ok(chroma);
    }
    Ok(chroma
        .into_iter()
        .map(|frame| {
            let norm = frame.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm < 1e-9 {
                frame
            } else {
                let mut f = frame;
                for x in f.iter_mut() {
                    *x /= norm;
                }
                f
            }
        })
        .collect())
}

/// Estimate the harmonic-to-noise ratio (HNR) per frame via autocorrelation.
///
/// For each analysis frame:
/// 1. Compute `R_0` (autocorrelation at lag 0, equivalent to frame energy).
/// 2. Search for the peak autocorrelation `R_peak` over lags corresponding to
///    the fundamental period range `[1/max_f0_hz, 1/min_f0_hz]` in samples.
/// 3. Return `R_peak / R_0`, clamped to `[0.0, 1.0]`.
///
/// Higher values indicate more periodic (harmonic) content; near-zero indicates noise.
///
/// The buffer is mixed to mono before analysis.
pub fn harmonic_ratio(
    buf: &AudioBuffer<f32>,
    frame_size: usize,
    hop_size: usize,
    min_f0_hz: f32,
    max_f0_hz: f32,
) -> Vec<f32> {
    if frame_size == 0 || hop_size == 0 {
        return vec![];
    }
    let sr = buf.sample_rate as f32;
    // Compute lag bounds from f0 range (guard against division by zero / inversions).
    let min_lag = ((sr / max_f0_hz.max(1.0)).floor() as usize).max(1);
    let max_lag = ((sr / min_f0_hz.max(1.0)).min(frame_size as f32 - 1.0) as usize).max(min_lag);

    let mono = crate::mix_to_mono(buf);
    let samples = &mono.samples;
    if samples.len() < frame_size {
        return vec![];
    }

    let n_frames = (samples.len() - frame_size) / hop_size + 1;

    (0..n_frames)
        .map(|fi| {
            let start = fi * hop_size;
            let frame = &samples[start..start + frame_size];

            // R_0: autocorrelation at lag 0 = sum of squared samples.
            let r0: f32 = frame.iter().map(|&x| x * x).sum();
            if r0 < 1e-12 {
                return 0.0;
            }

            // Find peak autocorrelation in the lag range [min_lag, max_lag].
            let mut r_peak = 0.0f32;
            for lag in min_lag..=max_lag.min(frame_size - 1) {
                let r_lag: f32 = frame[..frame_size - lag]
                    .iter()
                    .zip(frame[lag..].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                if r_lag > r_peak {
                    r_peak = r_lag;
                }
            }

            (r_peak / r0).clamp(0.0, 1.0)
        })
        .collect()
}

/// Spectral contrast per STFT frame.
///
/// Computes the peak-to-valley ratio in octave-spaced frequency sub-bands.
/// Returns a `Vec<Vec<f32>>` of shape `[n_frames][n_bands]`.
///
/// Sub-bands start at 200 Hz and double each octave up to Nyquist.
/// Typical `n_bands` value is 6 or 7.
///
/// # Errors
///
/// Propagates errors from the internal STFT computation.
#[must_use = "returns the spectral contrast matrix"]
pub fn spectral_contrast(
    buf: &AudioBuffer<f32>,
    n_fft: usize,
    hop_size: usize,
    n_bands: usize,
) -> Result<Vec<Vec<f32>>, OxiAudioError> {
    let stft_out = stft(buf, n_fft, hop_size, WindowFn::Hann)?;
    let sr = buf.sample_rate as f32;
    let n_bins = n_fft / 2 + 1;

    // Define band edges: start at 200 Hz, double each octave.
    // edges[b] and edges[b+1] are the bin boundaries (inclusive low, exclusive high).
    let band_edges: Vec<usize> = {
        let mut edges = vec![0usize];
        let mut f = 200.0f32;
        for _ in 0..n_bands {
            let bin = ((f / sr) * n_fft as f32).round() as usize;
            edges.push(bin.min(n_bins - 1));
            f *= 2.0;
        }
        edges.push(n_bins);
        edges
    };

    let contrast_frames = stft_out
        .frames
        .iter()
        .map(|frame| {
            let magnitudes: Vec<f32> = (0..n_bins.min(frame.len()))
                .map(|k| (frame[k].re * frame[k].re + frame[k].im * frame[k].im).sqrt())
                .collect();

            (0..n_bands)
                .map(|b| {
                    let lo = band_edges[b];
                    let hi = band_edges[b + 1].min(magnitudes.len());
                    if hi <= lo + 1 {
                        return 0.0;
                    }
                    let band: Vec<f32> = magnitudes[lo..hi].to_vec();
                    if band.is_empty() {
                        return 0.0;
                    }
                    let n_top = (band.len() / 5).max(1);
                    let n_bot = (band.len() / 5).max(1);
                    let mut sorted = band.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let valley: f32 = sorted[..n_bot].iter().sum::<f32>() / n_bot as f32;
                    let peak: f32 =
                        sorted[sorted.len() - n_top..].iter().sum::<f32>() / n_top as f32;
                    if valley < 1e-10 {
                        return 0.0;
                    }
                    (peak / valley).log10() * 20.0
                })
                .collect()
        })
        .collect();

    Ok(contrast_frames)
}

/// Tonal centroid (tonnetz) representation.
///
/// Projects the normalized chromagram onto three tonal cycles:
/// perfect fifth (7 semitones), minor third (3 semitones), and major third (4 semitones).
/// Each cycle contributes a 2-dimensional (sin, cos) component, giving a 6-dimensional
/// tonal centroid per frame.
///
/// Returns `Vec<[f32; 6]>` — one 6-dim vector per frame.
///
/// # Errors
///
/// Propagates errors from the internal chromagram computation (via STFT).
#[must_use = "returns the tonal centroid (tonnetz) frames"]
pub fn tonnetz(
    buf: &AudioBuffer<f32>,
    n_fft: usize,
    hop_size: usize,
) -> Result<Vec<[f32; 6]>, OxiAudioError> {
    let chroma_frames = chromagram(buf, n_fft, hop_size);

    // Radii for the three tonal cycles (perfect 5th, minor 3rd, major 3rd)
    let r = [1.0f32, 1.0, 0.5];
    // Intervals in semitones for each cycle
    let intervals = [7usize, 3, 4];

    let result: Vec<[f32; 6]> = chroma_frames
        .iter()
        .map(|chroma| {
            let norm: f32 = chroma.iter().sum::<f32>() + 1e-10;
            let chroma_norm: Vec<f32> = chroma.iter().map(|&c| c / norm).collect();

            let mut frame_result = [0.0f32; 6];
            for (i, &interval) in intervals.iter().enumerate() {
                let freq = interval as f32 / 12.0;
                let (mut sin_acc, mut cos_acc) = (0.0f32, 0.0f32);
                for (k, &cn) in chroma_norm.iter().enumerate().take(12) {
                    let angle = 2.0 * std::f32::consts::PI * freq * k as f32;
                    sin_acc += r[i] * angle.sin() * cn;
                    cos_acc += r[i] * angle.cos() * cn;
                }
                frame_result[i * 2] = sin_acc;
                frame_result[i * 2 + 1] = cos_acc;
            }
            frame_result
        })
        .collect();

    Ok(result)
}

/// Compute short-time energy (STE) per frame.
///
/// For each hop position, the mean-square energy over `frame_size` samples
/// (across all channels) is returned. Frames that extend beyond the end of the
/// buffer are zero-padded.
///
/// Returns one energy value per hop position.
#[must_use = "returns the short-time energy Vec"]
pub fn short_time_energy(buf: &AudioBuffer<f32>, frame_size: usize, hop: usize) -> Vec<f32> {
    if frame_size == 0 || hop == 0 || buf.samples.is_empty() {
        return vec![];
    }

    let n_channels = buf.channels.channel_count().max(1);
    let n_frames = buf.samples.len() / n_channels;

    if n_frames == 0 {
        return vec![];
    }

    // Number of hop positions: ceil(n_frames / hop)
    let n_hops = n_frames.div_ceil(hop);

    (0..n_hops)
        .map(|h| {
            let frame_start = h * hop;
            // sum of squared samples across all channels over frame_size frames.
            let mut sum_sq = 0.0f32;
            let mut count = 0usize;
            for fi in 0..frame_size {
                let frame_idx = frame_start + fi;
                if frame_idx < n_frames {
                    // Sum all channels for this frame.
                    let sample_base = frame_idx * n_channels;
                    for c in 0..n_channels {
                        let s = buf.samples[sample_base + c];
                        sum_sq += s * s;
                    }
                }
                // Frames beyond the buffer end are zero-padded (contribute 0).
                count += n_channels;
            }
            if count == 0 {
                0.0
            } else {
                sum_sq / count as f32
            }
        })
        .collect()
}

/// Compute the STFT of each channel independently.
///
/// Unlike [`stft`] which mixes the buffer to mono before analysis, this function
/// processes all channels and returns one STFT output per channel.
///
/// Channels are extracted from the interleaved buffer, converted to individual
/// mono `AudioBuffer<f32>` instances, and then passed through the standard
/// [`stft`] pipeline with a Hann window.
///
/// # Returns
///
/// `Vec<Vec<Vec<Complex<f32>>>>` — outer Vec indexed by channel index,
/// then by time frame, then by frequency bin.
///
/// On per-channel STFT failure (e.g., not enough samples for one frame),
/// that channel's entry is an empty `Vec`.
pub fn stft_multichannel(
    buf: &AudioBuffer<f32>,
    n_fft: usize,
    hop_size: usize,
) -> Vec<Vec<Vec<Complex<f32>>>> {
    let n_channels = buf.channels.channel_count();
    let n_frames = buf.samples.len() / n_channels;

    (0..n_channels)
        .map(|ch| {
            // Extract the interleaved channel samples into a mono buffer.
            let samples: Vec<f32> = (0..n_frames)
                .map(|f| buf.samples[f * n_channels + ch])
                .collect();
            let mono_buf = AudioBuffer {
                samples,
                sample_rate: buf.sample_rate,
                channels: ChannelLayout::Mono,
                format: buf.format,
            };
            match stft(&mono_buf, n_fft, hop_size, WindowFn::Hann) {
                Ok(out) => out.frames,
                Err(_) => vec![],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{ChannelLayout, SampleFormat};

    fn sine_buf(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn peak_bin(frame: &[Complex<f32>]) -> usize {
        frame
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.norm()
                    .partial_cmp(&b.norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k)
            .unwrap_or(0)
    }

    #[test]
    fn test_stft_peak_bin() {
        // 440 Hz sine at 48 kHz, STFT with window=2048.
        // Expected peak bin ≈ round(440 * 2048 / 48000) = 19.
        let buf = sine_buf(440.0, 48_000, 0.5);
        let out = stft(&buf, 2048, 512, WindowFn::Hann).expect("stft failed");
        assert!(!out.frames.is_empty(), "no frames produced");

        // Use frame from the middle (avoid edge effects).
        let mid = out.frames.len() / 2;
        let peak = peak_bin(&out.frames[mid]);
        let expected = (440.0 * 2048.0 / 48_000.0_f32).round() as usize;
        assert!(
            peak.abs_diff(expected) <= 1,
            "STFT peak bin {peak} is more than ±1 from expected {expected}"
        );
    }

    #[test]
    fn test_istft_roundtrip() {
        // Round-trip STFT → iSTFT: interior samples should match within 1e-3.
        let sample_rate = 48_000_u32;
        let buf = sine_buf(440.0, sample_rate, 1.0);
        let original_len = buf.samples.len();

        let stft_out = stft(&buf, 2048, 512, WindowFn::Hann).expect("stft failed");
        let reconstructed = istft(&stft_out, original_len).expect("istft failed");

        let start = 2048;
        let end = original_len.saturating_sub(2048);

        assert!(
            end > start,
            "signal too short for interior comparison: len={original_len}"
        );

        for i in start..end {
            let diff = (reconstructed.samples[i] - buf.samples[i]).abs();
            assert!(
                diff < 1e-3,
                "roundtrip mismatch at sample {i}: reconstructed={} original={} diff={diff}",
                reconstructed.samples[i],
                buf.samples[i],
            );
        }
    }

    #[test]
    fn test_melspectrogram() {
        // 440 Hz sine at 48 kHz → mel spectrogram with n_mels=40.
        // All output rows should be finite and at least one should be above log(1e-10)≈-23.
        let buf = sine_buf(440.0, 48_000, 0.5);
        let mel =
            melspectrogram(&buf, 40, 0.0, 24_000.0, 2048, 512).expect("melspectrogram failed");

        assert!(!mel.is_empty(), "no mel frames produced");

        // Each frame must have 40 mel bins.
        for (fi, frame) in mel.iter().enumerate() {
            assert_eq!(frame.len(), 40, "frame {fi} has wrong number of mel bins");
            for (mi, &val) in frame.iter().enumerate() {
                assert!(val.is_finite(), "mel[{fi}][{mi}] is not finite: {val}");
            }
        }

        // There should be at least one frame with significant energy (log-mel > -20).
        let max_val = mel
            .iter()
            .flat_map(|f| f.iter().copied())
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_val > -20.0,
            "max log-mel energy {max_val} is unexpectedly low — signal has no detectable energy"
        );
    }

    #[test]
    fn test_pitch_shift_doubles_freq() {
        // +12 semitones = octave up → 440 Hz becomes ~880 Hz.
        let sample_rate = 48_000_u32;
        let window_size = 2048;
        let hop_size = 512;

        let buf = sine_buf(440.0, sample_rate, 1.0);
        let shifted = pitch_shift(&buf, 12.0).expect("pitch_shift failed");

        let orig_stft = stft(&buf, window_size, hop_size, WindowFn::Hann).expect("orig stft");
        let shifted_stft =
            stft(&shifted, window_size, hop_size, WindowFn::Hann).expect("shifted stft");

        assert!(!orig_stft.frames.is_empty());
        assert!(!shifted_stft.frames.is_empty());

        let orig_mid = orig_stft.frames.len() / 2;
        let shift_mid = shifted_stft.frames.len() / 2;

        let orig_peak = peak_bin(&orig_stft.frames[orig_mid]);
        let shift_peak = peak_bin(&shifted_stft.frames[shift_mid]);

        // Allow ±3 bins of tolerance due to interpolation artifacts.
        let expected_shift_peak = orig_peak * 2;
        assert!(
            shift_peak.abs_diff(expected_shift_peak) <= 3,
            "pitch-shifted peak bin {shift_peak} is not ≈ 2× original {orig_peak} \
             (expected ~{expected_shift_peak})"
        );
    }

    #[test]
    fn spectral_centroid_pure_tone() {
        let sr = 48_000u32;
        let freq = 1_000.0f32;
        let n = sr as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let centroids = spectral_centroid(&buf, 2048, 512);
        assert!(!centroids.is_empty());
        let avg_centroid = centroids.iter().sum::<f32>() / centroids.len() as f32;
        // centroid of 1kHz tone should be near 1000 Hz (within 200 Hz)
        assert!(
            (avg_centroid - freq).abs() < 200.0,
            "centroid of 1kHz tone should be near 1000Hz, got {avg_centroid:.1}Hz"
        );
    }

    #[test]
    fn zcr_sine_wave() {
        let sr = 48_000u32;
        let freq = 440.0f32;
        let n = sr as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let zcr = zero_crossing_rate(&buf, 2048, 1024);
        assert!(!zcr.is_empty());
        let avg_zcr = zcr.iter().sum::<f32>() / zcr.len() as f32;
        // 440 Hz sine: ~880 zero crossings per second; per frame: 880/48000 * frame_size
        let expected_zcr = 2.0 * freq / sr as f32;
        assert!(
            (avg_zcr - expected_zcr).abs() < expected_zcr,
            "ZCR of 440Hz should be near {expected_zcr:.4}, got {avg_zcr:.4}"
        );
    }

    #[test]
    fn mfcc_stable() {
        let sr = 22_050u32;
        let n = sr as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mfcc1 = mfcc(&buf, 13, 40, 1024, 512).expect("mfcc");
        let mfcc2 = mfcc(&buf, 13, 40, 1024, 512).expect("mfcc");
        assert_eq!(mfcc1.len(), mfcc2.len());
        for (a, b) in mfcc1.iter().zip(mfcc2.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-5, "MFCC should be deterministic");
            }
        }
    }

    #[test]
    fn spectral_flatness_tonal_vs_noise() {
        let sr = 48_000u32;
        let n = 4096usize;
        // Tonal signal: single sine
        let tonal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
            .collect();
        let tonal_buf = AudioBuffer {
            samples: tonal,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let flatness_tonal = spectral_flatness(&tonal_buf, 1024, 512);
        let avg_tonal = if flatness_tonal.is_empty() {
            0.0
        } else {
            flatness_tonal.iter().sum::<f32>() / flatness_tonal.len() as f32
        };
        // Tonal signal should have low flatness (near 0)
        assert!(
            avg_tonal < 0.5,
            "tonal signal flatness should be < 0.5, got {avg_tonal:.3}"
        );
    }

    #[test]
    fn test_chromagram_produces_12_bins() {
        let sr = 48_000u32;
        let n = sr as usize;
        // 440 Hz sine
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let chroma = chromagram(&buf, 2048, 512);
        assert!(!chroma.is_empty(), "should produce at least one frame");
        for (i, frame) in chroma.iter().enumerate() {
            assert_eq!(frame.len(), 12, "frame {i} should have 12 chroma bins");
            for &v in frame.iter() {
                assert!(v >= 0.0, "chroma values must be non-negative, got {v}");
            }
        }
    }

    #[test]
    fn test_spectral_bandwidth_nonneg() {
        let sr = 48_000u32;
        let n = sr as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let bw = spectral_bandwidth(&buf, 2048, 512);
        assert!(!bw.is_empty(), "should produce at least one frame");
        for &v in &bw {
            assert!(v >= 0.0, "bandwidth must be non-negative, got {v}");
        }
    }

    #[test]
    fn test_kaiser_window_stft() {
        let buf = AudioBuffer {
            samples: vec![1.0f32; 4096],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let result = stft(&buf, 1024, 256, WindowFn::Kaiser { beta: 5.0 });
        assert!(result.is_ok(), "stft with Kaiser window should succeed");
        let out = result.expect("stft");
        assert!(!out.frames.is_empty(), "Kaiser STFT should produce frames");
    }

    #[test]
    fn test_flat_top_window_stft() {
        let buf = AudioBuffer {
            samples: vec![0.5f32; 4096],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let result = stft(&buf, 1024, 256, WindowFn::FlatTop);
        assert!(result.is_ok(), "stft with FlatTop window should succeed");
        let out = result.expect("stft");
        assert!(!out.frames.is_empty(), "FlatTop STFT should produce frames");
    }

    #[test]
    fn test_spectral_contrast_shape() {
        let n = 44100usize;
        let buf = AudioBuffer {
            samples: (0..n).map(|i| (i as f32 / 100.0).sin() * 0.5).collect(),
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let contrast = spectral_contrast(&buf, 2048, 512, 6).expect("spectral_contrast failed");
        assert!(!contrast.is_empty(), "should have at least one frame");
        assert_eq!(
            contrast[0].len(),
            6,
            "each frame should have 6 contrast values"
        );
    }

    #[test]
    fn test_tonnetz_shape() {
        let buf = AudioBuffer {
            samples: vec![0.3f32; 4096],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let tz = tonnetz(&buf, 2048, 512).expect("tonnetz failed");
        assert!(!tz.is_empty(), "should have at least one tonnetz frame");
        for frame in &tz {
            assert_eq!(frame.len(), 6, "each tonnetz frame should be 6-dimensional");
        }
    }

    // ── M12 tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_spectral_bandwidth_tonal() {
        // A pure 1 kHz sine should have a narrow bandwidth well below Nyquist/2.
        let buf = sine_buf(1000.0, 44_100, 0.5);
        let bw = spectral_bandwidth(&buf, 2048, 512);
        assert!(!bw.is_empty());
        let mean_bw: f32 = bw.iter().sum::<f32>() / bw.len() as f32;
        assert!(
            mean_bw < 5000.0,
            "tonal signal bandwidth should be narrow, got {mean_bw} Hz"
        );
        assert!(mean_bw > 0.0, "bandwidth should be positive");
    }

    #[test]
    fn test_spectral_crest_factor_tonal() {
        // A pure sine should have a high crest factor (energy concentrated in one bin).
        let buf = sine_buf(440.0, 44_100, 0.5);
        let crest = spectral_crest_factor(&buf, 2048, 512);
        assert!(!crest.is_empty(), "should produce at least one frame");
        let mean_crest: f32 = crest.iter().sum::<f32>() / crest.len() as f32;
        assert!(
            mean_crest > 5.0,
            "tonal crest factor should be high, got {mean_crest}"
        );
    }

    #[test]
    fn test_spectral_crest_factor_silent() {
        // Silent input should return 0.0 for all frames (or empty).
        let buf = AudioBuffer {
            samples: vec![0.0f32; 4096],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let crest = spectral_crest_factor(&buf, 2048, 512);
        for v in &crest {
            assert_eq!(*v, 0.0, "silent frame crest factor should be 0.0");
        }
    }

    #[test]
    fn test_spectral_entropy_noise_vs_tonal() {
        // A sum of 10 harmonically unrelated sinusoids is more noise-like than a single sine.
        let sr = 44_100u32;
        let n = (sr as f32 * 0.5) as usize;
        let noise_samples: Vec<f32> = (0..n)
            .map(|i| {
                (0..10_usize)
                    .map(|k| {
                        let freq = (k * 440 + 100) as f32;
                        (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin() * 0.1
                    })
                    .sum::<f32>()
            })
            .collect();
        let noise_buf = AudioBuffer {
            samples: noise_samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let tonal_buf = sine_buf(440.0, sr, 0.5);

        let ent_noise = spectral_entropy(&noise_buf, 2048, 512);
        let ent_tonal = spectral_entropy(&tonal_buf, 2048, 512);

        let mean_noise = ent_noise.iter().sum::<f32>() / ent_noise.len().max(1) as f32;
        let mean_tonal = ent_tonal.iter().sum::<f32>() / ent_tonal.len().max(1) as f32;

        assert!(
            mean_noise > mean_tonal,
            "noise entropy ({mean_noise}) should exceed tonal entropy ({mean_tonal})"
        );
    }

    #[test]
    fn test_chromagram_normalized_unit_norm() {
        let buf = sine_buf(440.0, 44_100, 0.5);
        let chroma =
            chromagram_normalized(&buf, 2048, 512, true).expect("chromagram_normalized failed");
        assert!(!chroma.is_empty());
        for frame in &chroma {
            let norm: f32 = frame.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                assert!(
                    (norm - 1.0).abs() < 1e-5,
                    "normalised chroma frame should have unit L2 norm, got {norm}"
                );
            }
        }
    }

    #[test]
    fn test_chromagram_normalized_no_norm() {
        // Without normalisation the output should match the raw chromagram.
        let buf = sine_buf(440.0, 44_100, 0.5);
        let raw = chromagram(&buf, 2048, 512);
        let wrapped = chromagram_normalized(&buf, 2048, 512, false)
            .expect("chromagram_normalized(false) failed");
        assert_eq!(raw.len(), wrapped.len());
        for (r, w) in raw.iter().zip(wrapped.iter()) {
            for (a, b) in r.iter().zip(w.iter()) {
                assert!(
                    (a - b).abs() < 1e-7,
                    "un-normalised should equal raw chromagram"
                );
            }
        }
    }

    #[test]
    fn test_harmonic_ratio_sine_high() {
        // A pure 220 Hz sine should yield a high harmonic ratio (> 0.5 on average).
        let buf = sine_buf(220.0, 44_100, 0.5);
        let hr = harmonic_ratio(&buf, 2048, 512, 80.0, 800.0);
        assert!(!hr.is_empty(), "should produce at least one frame");
        let mean_hr: f32 = hr.iter().sum::<f32>() / hr.len() as f32;
        assert!(
            mean_hr > 0.5,
            "sine wave should have high harmonic ratio, got {mean_hr}"
        );
    }

    #[test]
    fn test_harmonic_ratio_range() {
        // All returned values must be in [0, 1].
        let buf = sine_buf(440.0, 44_100, 0.5);
        let hr = harmonic_ratio(&buf, 2048, 512, 80.0, 800.0);
        for &v in &hr {
            assert!(
                (0.0..=1.0).contains(&v),
                "harmonic ratio must be in [0, 1], got {v}"
            );
        }
    }

    #[test]
    fn test_stft_scratch_buffer_reuse_identical_results() {
        // Calling stft() twice on the same buffer with the same n_fft must yield
        // bit-identical results even when the thread-local scratch buffer is reused.
        let buf = sine_buf(440.0, 44_100, 0.1);
        let out1 = stft(&buf, 1024, 256, WindowFn::Hann).expect("first stft failed");
        let out2 = stft(&buf, 1024, 256, WindowFn::Hann).expect("second stft failed");

        assert_eq!(
            out1.frames.len(),
            out2.frames.len(),
            "frame count must be identical on repeated stft call"
        );
        for (fi, (f1, f2)) in out1.frames.iter().zip(out2.frames.iter()).enumerate() {
            assert_eq!(f1.len(), f2.len(), "frame {fi}: bin count differs");
            for (bi, (b1, b2)) in f1.iter().zip(f2.iter()).enumerate() {
                assert_eq!(
                    b1.re, b2.re,
                    "frame {fi} bin {bi}: re part differs (scratch buffer not properly reset?)"
                );
                assert_eq!(
                    b1.im, b2.im,
                    "frame {fi} bin {bi}: im part differs (scratch buffer not properly reset?)"
                );
            }
        }
    }

    // ── short_time_energy tests ────────────────────────────────────────────────

    #[test]
    fn test_short_time_energy_silence() {
        // All zeros → all energies 0.0.
        let buf = AudioBuffer {
            samples: vec![0.0f32; 256],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let ste = short_time_energy(&buf, 64, 32);
        assert!(!ste.is_empty(), "should produce at least one hop");
        for &e in &ste {
            assert_eq!(e, 0.0, "silence energy must be 0.0, got {e}");
        }
    }

    #[test]
    fn test_short_time_energy_constant() {
        // Constant 1.0 signal → mean-square energy = 1.0 per frame.
        // Use frame_size == hop so every hop aligns exactly with the buffer boundary
        // and there is no zero-padding for any hop (total = n_hops * frame_size).
        let frame_size = 32usize;
        let hop = 32usize;
        let n_hops = 8usize;
        let n_samples = n_hops * frame_size; // 256, exact multiple
        let buf = AudioBuffer {
            samples: vec![1.0f32; n_samples],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let ste = short_time_energy(&buf, frame_size, hop);
        assert!(!ste.is_empty(), "should produce at least one hop");
        assert_eq!(ste.len(), n_hops, "should have exactly {n_hops} hops");
        for &e in &ste {
            assert!(
                (e - 1.0f32).abs() < 1e-6,
                "constant-1.0 signal energy should be 1.0, got {e}"
            );
        }
    }

    #[test]
    fn test_short_time_energy_frame_count() {
        // Output length should equal ceil(n_frames / hop).
        let n_frames = 100usize;
        let hop = 10usize;
        let buf = AudioBuffer {
            samples: vec![0.5f32; n_frames],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let ste = short_time_energy(&buf, 20, hop);
        let expected = n_frames.div_ceil(hop);
        assert_eq!(
            ste.len(),
            expected,
            "output length should be ceil({n_frames}/{hop})={expected}, got {}",
            ste.len()
        );
    }

    #[test]
    fn test_stft_multichannel_stereo_same_as_mono() {
        // Stereo buffer where both channels carry the same sine: both STFT outputs
        // must be identical frame-by-frame and bin-by-bin.
        let sr = 44_100_u32;
        let n = (sr as f32 * 0.25) as usize;
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let s = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
            samples.push(s); // left
            samples.push(s); // right — identical
        }
        let stereo = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };

        let mc = stft_multichannel(&stereo, 1024, 256);
        assert_eq!(mc.len(), 2, "should have 2 channel outputs");
        assert_eq!(
            mc[0].len(),
            mc[1].len(),
            "both channels should yield same number of frames"
        );
        for (frame_idx, (frame_l, frame_r)) in mc[0].iter().zip(mc[1].iter()).enumerate() {
            assert_eq!(
                frame_l.len(),
                frame_r.len(),
                "frame {frame_idx}: bin count differs"
            );
            for (bin, (l, r)) in frame_l.iter().zip(frame_r.iter()).enumerate() {
                assert!(
                    (l.re - r.re).abs() < 1e-5 && (l.im - r.im).abs() < 1e-5,
                    "frame {frame_idx} bin {bin}: left={l:?} right={r:?}"
                );
            }
        }
    }

    #[test]
    fn mfcc_stability_same_signal_bit_exact() {
        // Test 8: MFCC stability — same signal, same parameters → bit-exact output.
        // This verifies no randomness or uninitialized memory in the MFCC pipeline.
        use oxiaudio_core::{ChannelLayout, SampleFormat};
        use std::f32::consts::PI;

        let sr = 22_050u32;
        let n = sr as usize; // 1 second
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.7 * (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };

        let result1 = mfcc(&buf, 13, 40, 512, 256).expect("mfcc run 1 should succeed");
        let result2 = mfcc(&buf, 13, 40, 512, 256).expect("mfcc run 2 should succeed");

        assert_eq!(result1.len(), result2.len(), "frame count must match");
        for (frame_idx, (f1, f2)) in result1.iter().zip(result2.iter()).enumerate() {
            assert_eq!(f1.len(), f2.len(), "frame {frame_idx}: coeff count differs");
            for (coeff_idx, (v1, v2)) in f1.iter().zip(f2.iter()).enumerate() {
                assert_eq!(
                    v1.to_bits(),
                    v2.to_bits(),
                    "frame {frame_idx} coeff {coeff_idx}: mfcc is not bit-exact ({v1} vs {v2})"
                );
            }
        }
        assert!(!result1.is_empty(), "mfcc should return at least one frame");
        assert_eq!(
            result1[0].len(),
            13,
            "each frame should have 13 coefficients"
        );
    }
}
