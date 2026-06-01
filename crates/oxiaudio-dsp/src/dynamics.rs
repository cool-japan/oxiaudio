use oxiaudio_core::{AudioBuffer, AudioFilter, OxiAudioError};

/// A feed-forward compressor with soft-knee support and envelope following.
///
/// The compressor applies gain reduction when the input level exceeds `threshold_db`.
/// Attack and release time constants control the envelope follower speed.
#[derive(Debug, Clone)]
pub struct Compressor {
    /// Threshold in dBFS above which compression begins.
    pub threshold_db: f32,
    /// Compression ratio (e.g. 4.0 = 4:1 ratio). Values below 1.0 are clamped to 1.0.
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Soft-knee width in dB. Set to 0.0 for hard knee.
    pub knee_db: f32,
    /// Makeup gain in dB applied after compression.
    pub makeup_gain_db: f32,
}

/// Compute the gain computer output in dB given an input level.
///
/// Returns the gain reduction in dB (always <= 0). Uses the same soft-knee / hard-knee
/// logic as [`Compressor::process`]. Positive `knee` enables soft knee; zero or negative
/// uses hard knee.
///
/// `target_db - level_db` is the gain reduction applied to move the signal to the
/// desired output level.
#[inline]
fn compute_gain_db(level_db: f32, threshold: f32, ratio: f32, knee: f32) -> f32 {
    let target_db = if knee <= 0.0 {
        if level_db < threshold {
            level_db
        } else {
            threshold + (level_db - threshold) / ratio
        }
    } else {
        let knee_start = threshold - knee / 2.0;
        let knee_end = threshold + knee / 2.0;
        if level_db < knee_start {
            level_db
        } else if level_db > knee_end {
            threshold + (level_db - threshold) / ratio
        } else {
            let t = (level_db - knee_start) / knee;
            level_db + (1.0 / ratio - 1.0) * t * t * knee / 2.0
        }
    };
    target_db - level_db
}

impl Compressor {
    /// Create a new `Compressor` with hard knee and no makeup gain.
    pub fn new(threshold_db: f32, ratio: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        }
    }

    /// Set soft-knee width in dB (builder pattern).
    pub fn with_knee(mut self, knee_db: f32) -> Self {
        self.knee_db = knee_db;
        self
    }

    /// Set makeup gain in dB (builder pattern).
    pub fn with_makeup(mut self, makeup_db: f32) -> Self {
        self.makeup_gain_db = makeup_db;
        self
    }

    /// Apply compression to `buf` and return the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let sr = buf.sample_rate as f32;
        let ln10 = 10.0f32.ln();
        let attack_coeff = if self.attack_ms > 0.0 {
            (-1.0 / (self.attack_ms * sr / 1000.0)).exp()
        } else {
            0.0
        };
        let release_coeff = if self.release_ms > 0.0 {
            (-1.0 / (self.release_ms * sr / 1000.0)).exp()
        } else {
            0.0
        };
        let ch = buf.channels.channel_count();
        let frames = buf.samples.len() / ch.max(1);
        let makeup_linear = (self.makeup_gain_db / 20.0 * ln10).exp();
        let mut envelope = vec![0.0f32; ch];
        let mut out = buf.samples.clone();

        let threshold = self.threshold_db;
        let ratio = self.ratio.max(1.0);
        let knee = self.knee_db;

        for frame in 0..frames {
            for c in 0..ch {
                let x = out[frame * ch + c];
                let x_abs = x.abs();
                let x_db = if x_abs > 1e-10 {
                    20.0 * x_abs.log10()
                } else {
                    -200.0_f32
                };
                // Gain computer
                let target_db = if knee <= 0.0 {
                    if x_db < threshold {
                        x_db
                    } else {
                        threshold + (x_db - threshold) / ratio
                    }
                } else {
                    let knee_start = threshold - knee / 2.0;
                    let knee_end = threshold + knee / 2.0;
                    if x_db < knee_start {
                        x_db
                    } else if x_db > knee_end {
                        threshold + (x_db - threshold) / ratio
                    } else {
                        let t = (x_db - knee_start) / knee;
                        x_db + (1.0 / ratio - 1.0) * t * t * knee / 2.0
                    }
                };
                let gain_db = target_db - x_db;
                if gain_db < envelope[c] {
                    envelope[c] = attack_coeff * envelope[c] + (1.0 - attack_coeff) * gain_db;
                } else {
                    envelope[c] = release_coeff * envelope[c] + (1.0 - release_coeff) * gain_db;
                }
                let gain_linear = (envelope[c] / 20.0 * ln10).exp();
                out[frame * ch + c] = x * gain_linear * makeup_linear;
            }
        }
        AudioBuffer {
            samples: out,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }

    /// Compress `input` using `key_signal` to drive the gain computer.
    ///
    /// The key signal controls *when* compression occurs (its level drives the
    /// envelope follower), but the gain is applied to `input`. This enables
    /// ducking (e.g., music ducked by voiceover) and frequency-selective
    /// compression (sidechain from a band-pass filter output).
    ///
    /// `key_signal` and `input` must have the same sample_rate and sample count.
    /// If lengths differ, the shorter of the two determines the output length.
    ///
    /// # Errors
    /// Returns `Err` if sample rates differ.
    #[must_use = "discarding errors ignores sidechain failure"]
    pub fn process_with_sidechain(
        &self,
        input: &AudioBuffer<f32>,
        key_signal: &AudioBuffer<f32>,
    ) -> Result<AudioBuffer<f32>, OxiAudioError> {
        if input.sample_rate != key_signal.sample_rate {
            return Err(OxiAudioError::InvalidSampleRate(format!(
                "sidechain key={} input={}",
                key_signal.sample_rate, input.sample_rate
            )));
        }

        let sr = input.sample_rate as f32;
        let ln10 = 10.0f32.ln();
        let attack_coeff = if self.attack_ms > 0.0 {
            (-1.0 / (self.attack_ms * sr / 1000.0)).exp()
        } else {
            0.0
        };
        let release_coeff = if self.release_ms > 0.0 {
            (-1.0 / (self.release_ms * sr / 1000.0)).exp()
        } else {
            0.0
        };
        let makeup_linear = (self.makeup_gain_db / 20.0 * ln10).exp();
        let ratio = self.ratio.max(1.0);
        let threshold = self.threshold_db;
        let knee = self.knee_db;

        let n = input.samples.len().min(key_signal.samples.len());
        let mut envelope = 0.0f32;
        let mut out_samples = Vec::with_capacity(n);

        for i in 0..n {
            let key_abs = key_signal.samples[i].abs();
            let key_db = if key_abs > 1e-9 {
                20.0 * key_abs.log10()
            } else {
                -200.0_f32
            };
            let gain_db = compute_gain_db(key_db, threshold, ratio, knee);
            // Smooth gain with attack/release coefficients
            if gain_db < envelope {
                envelope = attack_coeff * envelope + (1.0 - attack_coeff) * gain_db;
            } else {
                envelope = release_coeff * envelope + (1.0 - release_coeff) * gain_db;
            }
            let gain_linear = (envelope / 20.0 * ln10).exp();
            out_samples.push(input.samples[i] * gain_linear * makeup_linear);
        }

        Ok(AudioBuffer {
            samples: out_samples,
            sample_rate: input.sample_rate,
            channels: input.channels,
            format: input.format,
        })
    }
}

impl AudioFilter for Compressor {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

/// A hard limiter implemented as a compressor with a very high ratio (1000:1)
/// and near-instant attack.
#[derive(Debug, Clone)]
pub struct Limiter {
    compressor: Compressor,
}

impl Limiter {
    /// Create a new `Limiter` with the given ceiling threshold and release time.
    ///
    /// Attack is set to 0.01 ms (effectively instantaneous), ratio to 1000:1.
    pub fn new(threshold_db: f32, release_ms: f32) -> Self {
        Self {
            compressor: Compressor::new(threshold_db, 1000.0, 0.01, release_ms),
        }
    }

    /// Apply the limiter to `buf` and return the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        self.compressor.process(buf)
    }
}

impl AudioFilter for Limiter {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

/// Internal state machine for the noise gate envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
enum GateState {
    Closed,
    Attack,
    Open,
    Hold,
    Release,
}

/// A noise gate that attenuates the signal when it falls below a threshold.
///
/// The gate uses a state machine: Closed → Attack → Open → Hold → Release → Closed.
#[derive(Debug, Clone)]
pub struct NoiseGate {
    /// Threshold in dBFS. Signal below this level is gated.
    pub threshold_db: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Hold time in milliseconds (how long gate stays open after signal drops).
    pub hold_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Floor attenuation in dB when gate is closed (typically -80 dB).
    pub range_db: f32,
}

impl NoiseGate {
    /// Create a new `NoiseGate` with sensible defaults.
    pub fn new(threshold_db: f32) -> Self {
        Self {
            threshold_db,
            attack_ms: 1.0,
            hold_ms: 50.0,
            release_ms: 100.0,
            range_db: -80.0,
        }
    }

    /// Apply the gate to `buf` and return the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let sr = buf.sample_rate as f32;
        let ch = buf.channels.channel_count();
        let attack_samples = (self.attack_ms * sr / 1000.0).max(1.0) as usize;
        let hold_samples = (self.hold_ms * sr / 1000.0).max(1.0) as usize;
        let release_samples = (self.release_ms * sr / 1000.0).max(1.0) as usize;
        let threshold_linear = if self.threshold_db <= -199.0 {
            0.0
        } else {
            (self.threshold_db / 20.0 * 10.0f32.ln()).exp()
        };
        let floor_gain = (self.range_db / 20.0 * 10.0f32.ln()).exp();
        let frames = buf.samples.len() / ch.max(1);
        let mut state = GateState::Closed;
        let mut counter = 0usize;
        let mut current_gain = floor_gain;
        let mut out = buf.samples.clone();

        for frame in 0..frames {
            let peak: f32 = (0..ch)
                .map(|c| out[frame * ch + c].abs())
                .fold(0.0f32, f32::max);
            let above = peak >= threshold_linear;
            let (new_state, new_counter) = match state {
                GateState::Closed => {
                    if above {
                        (GateState::Attack, 0)
                    } else {
                        (GateState::Closed, counter)
                    }
                }
                GateState::Attack => {
                    let c = counter + 1;
                    if !above && c >= attack_samples {
                        (GateState::Closed, 0)
                    } else if c >= attack_samples {
                        (GateState::Open, 0)
                    } else {
                        (GateState::Attack, c)
                    }
                }
                GateState::Open => {
                    if !above {
                        (GateState::Hold, 0)
                    } else {
                        (GateState::Open, 0)
                    }
                }
                GateState::Hold => {
                    if above {
                        (GateState::Open, 0)
                    } else {
                        let c = counter + 1;
                        if c >= hold_samples {
                            (GateState::Release, 0)
                        } else {
                            (GateState::Hold, c)
                        }
                    }
                }
                GateState::Release => {
                    if above {
                        (GateState::Open, 0)
                    } else {
                        let c = counter + 1;
                        if c >= release_samples {
                            (GateState::Closed, 0)
                        } else {
                            (GateState::Release, c)
                        }
                    }
                }
            };
            state = new_state;
            counter = new_counter;
            let target_gain = match state {
                GateState::Closed => floor_gain,
                GateState::Open => 1.0,
                GateState::Attack => {
                    floor_gain + (1.0 - floor_gain) * counter as f32 / attack_samples.max(1) as f32
                }
                GateState::Hold => 1.0,
                GateState::Release => {
                    1.0 - (1.0 - floor_gain) * counter as f32 / release_samples.max(1) as f32
                }
            };
            current_gain = current_gain * 0.9 + target_gain * 0.1;
            for c in 0..ch {
                out[frame * ch + c] *= current_gain;
            }
        }
        AudioBuffer {
            samples: out,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }
}

impl AudioFilter for NoiseGate {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

/// A downward expander: attenuates the signal when it falls *below* `threshold_db`,
/// the complement of a compressor. Useful for reducing low-level noise without the
/// hard gating of a [`NoiseGate`].
///
/// The expansion ratio is `< 1.0` conceptually (e.g. a "1:2" expander); this struct
/// stores the reciprocal `ratio >= 1.0` for numerical convenience: a `ratio` of 2.0
/// means every dB below threshold is mapped to 2 dB of output reduction.
#[derive(Debug, Clone)]
pub struct Expander {
    /// Threshold in dBFS below which expansion (downward gain) begins.
    pub threshold_db: f32,
    /// Expansion ratio `>= 1.0` (output dB-below-threshold = ratio × input dB-below).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Maximum downward attenuation in dB (floor), e.g. -40 dB.
    pub range_db: f32,
}

impl Expander {
    /// Create a new `Expander`.
    pub fn new(threshold_db: f32, ratio: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self {
            threshold_db,
            ratio: ratio.max(1.0),
            attack_ms,
            release_ms,
            range_db: -40.0,
        }
    }

    /// Set the maximum downward attenuation floor in dB (builder style).
    pub fn with_range(mut self, range_db: f32) -> Self {
        self.range_db = range_db;
        self
    }

    /// Apply expansion to `buf` and return the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let sr = buf.sample_rate as f32;
        let ln10 = 10.0f32.ln();
        let attack_coeff = if self.attack_ms > 0.0 {
            (-1.0 / (self.attack_ms * sr / 1000.0)).exp()
        } else {
            0.0
        };
        let release_coeff = if self.release_ms > 0.0 {
            (-1.0 / (self.release_ms * sr / 1000.0)).exp()
        } else {
            0.0
        };
        let ch = buf.channels.channel_count();
        let frames = buf.samples.len() / ch.max(1);
        let ratio = self.ratio.max(1.0);
        let threshold = self.threshold_db;
        let range = self.range_db;
        let mut envelope = vec![0.0f32; ch];
        let mut out = buf.samples.clone();

        for frame in 0..frames {
            for c in 0..ch {
                let x = out[frame * ch + c];
                let x_abs = x.abs();
                let x_db = if x_abs > 1e-10 {
                    20.0 * x_abs.log10()
                } else {
                    -200.0_f32
                };
                // Downward gain: only below threshold.
                let gain_db = if x_db >= threshold {
                    0.0
                } else {
                    // Each dB below threshold maps to `ratio` dB of reduction,
                    // floored at `range`.
                    ((x_db - threshold) * (ratio - 1.0)).max(range)
                };
                // Envelope follower (attack when reducing further, release otherwise).
                if gain_db < envelope[c] {
                    envelope[c] = attack_coeff * envelope[c] + (1.0 - attack_coeff) * gain_db;
                } else {
                    envelope[c] = release_coeff * envelope[c] + (1.0 - release_coeff) * gain_db;
                }
                let gain_linear = (envelope[c] / 20.0 * ln10).exp();
                out[frame * ch + c] = x * gain_linear;
            }
        }
        AudioBuffer {
            samples: out,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }
}

impl AudioFilter for Expander {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

// ─── DeEsser ─────────────────────────────────────────────────────────────────

/// A broadband de-esser: detects sibilant energy in a configurable frequency band
/// and applies dynamic gain reduction to the full signal when the detection level
/// exceeds the threshold.
///
/// The detection path uses a bandpass filter centred at `sqrt(low_freq * high_freq)`.
/// Gain reduction is applied to the dry signal, not the filtered path — this is the
/// "broadband" topology (simpler and more phase-coherent than split-band).
#[derive(Debug, Clone)]
pub struct DeEsser {
    /// Detection threshold in dBFS (e.g. `-20.0`).
    pub threshold_db: f32,
    /// Compression ratio applied to the sibilant band (e.g. `4.0`).
    pub ratio: f32,
    /// Attack time in milliseconds for the envelope follower.
    pub attack_ms: f32,
    /// Release time in milliseconds for the envelope follower.
    pub release_ms: f32,
    /// Lower bound of the sibilant detection band in Hz (e.g. `5000.0`).
    pub low_freq: f32,
    /// Upper bound of the sibilant detection band in Hz (e.g. `10000.0`).
    pub high_freq: f32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl DeEsser {
    /// Create a new `DeEsser` with defaults: 4:1 ratio, 1 ms attack, 50 ms release,
    /// 5–10 kHz sibilant band.
    pub fn new(threshold_db: f32, sample_rate: u32) -> Self {
        Self {
            threshold_db,
            ratio: 4.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            low_freq: 5000.0,
            high_freq: 10000.0,
            sample_rate,
        }
    }

    /// Apply de-essing to `buf` and return the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        use crate::biquad::BiquadFilter;

        // Design bandpass detection filter: centre = geometric mean, Q from bandwidth.
        let center = (self.low_freq * self.high_freq).sqrt();
        let q = center / (self.high_freq - self.low_freq).max(1.0);
        let detect_filter = BiquadFilter::bandpass(center, q, self.sample_rate);

        let sr = self.sample_rate as f32;
        let attack_coeff = if self.attack_ms > 0.0 {
            (-1.0 / (self.attack_ms * sr / 1000.0)).exp()
        } else {
            0.0
        };
        let release_coeff = if self.release_ms > 0.0 {
            (-1.0 / (self.release_ms * sr / 1000.0)).exp()
        } else {
            0.0
        };

        // Apply detection filter once to get sibilant signal.
        let detected = detect_filter.process(buf);

        let mut envelope = 0.0_f32;
        let out_samples: Vec<f32> = buf
            .samples
            .iter()
            .zip(detected.samples.iter())
            .map(|(&x, &d)| {
                let det_abs = d.abs();
                if det_abs > envelope {
                    envelope = attack_coeff * envelope + (1.0 - attack_coeff) * det_abs;
                } else {
                    envelope = release_coeff * envelope + (1.0 - release_coeff) * det_abs;
                }
                let level_db = if envelope > 1e-9 {
                    20.0 * envelope.log10()
                } else {
                    -120.0_f32
                };
                let gain_db = if level_db > self.threshold_db {
                    self.threshold_db + (level_db - self.threshold_db) / self.ratio - level_db
                } else {
                    0.0
                };
                x * 10.0_f32.powf(gain_db / 20.0)
            })
            .collect();

        AudioBuffer {
            samples: out_samples,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }
}

impl AudioFilter for DeEsser {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

// ─── MultibandCompressor ──────────────────────────────────────────────────────

/// Settings for one frequency band of a [`MultibandCompressor`].
#[derive(Debug, Clone)]
pub struct BandSettings {
    /// Upper crossover frequency in Hz. `None` for the last (highest) band.
    pub crossover_hz: Option<f32>,
    /// Threshold in dBFS (e.g. `-20.0`).
    pub threshold_db: f32,
    /// Compression ratio (e.g. `4.0` for 4:1).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Makeup gain in dB applied after compression.
    pub makeup_gain_db: f32,
}

/// A flexible N-band multiband compressor using Linkwitz-Riley 4th-order crossover filters.
///
/// The signal is split into frequency bands at each crossover frequency defined in `bands`.
/// Each band is compressed independently with its own [`BandSettings`], then all bands are
/// summed back together.
///
/// **Crossover topology**: Linkwitz-Riley 4th-order (LR4) = two cascaded 2nd-order Butterworth
/// biquad sections at the same frequency (Q = 1/√2). LR4 crossovers provide a maximally-flat
/// response at the crossover point and -6 dB for both LP and HP at the crossover frequency.
#[derive(Debug, Clone)]
pub struct MultibandCompressor {
    /// Per-band settings. The last entry must have `crossover_hz: None`.
    pub bands: Vec<BandSettings>,
}

impl MultibandCompressor {
    /// Create a standard 3-band compressor with crossovers at 200 Hz and 2000 Hz.
    ///
    /// Sensible defaults are applied per band:
    /// - Low band: 3:1, 20 ms attack, 200 ms release
    /// - Mid band: 4:1, 10 ms attack, 100 ms release
    /// - High band: 6:1, 5 ms attack, 50 ms release
    pub fn three_band(low_threshold: f32, mid_threshold: f32, high_threshold: f32) -> Self {
        Self {
            bands: vec![
                BandSettings {
                    crossover_hz: Some(200.0),
                    threshold_db: low_threshold,
                    ratio: 3.0,
                    attack_ms: 20.0,
                    release_ms: 200.0,
                    makeup_gain_db: 0.0,
                },
                BandSettings {
                    crossover_hz: Some(2000.0),
                    threshold_db: mid_threshold,
                    ratio: 4.0,
                    attack_ms: 10.0,
                    release_ms: 100.0,
                    makeup_gain_db: 0.0,
                },
                BandSettings {
                    crossover_hz: None,
                    threshold_db: high_threshold,
                    ratio: 6.0,
                    attack_ms: 5.0,
                    release_ms: 50.0,
                    makeup_gain_db: 0.0,
                },
            ],
        }
    }

    /// Process `buf` through all bands and return the compressed output.
    ///
    /// The signal is split using LR4 crossover filters, each band is compressed
    /// with its own settings, and the bands are summed.
    #[must_use = "returns the multiband-compressed AudioBuffer"]
    pub fn process(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        use crate::biquad::BiquadFilter;
        use std::f32::consts::FRAC_1_SQRT_2;

        if self.bands.is_empty() {
            return Ok(buf.clone());
        }

        let sr = buf.sample_rate;
        let n_bands = self.bands.len();
        let mut band_signals: Vec<AudioBuffer<f32>> = Vec::with_capacity(n_bands);

        // Split into frequency bands using successive LR4 crossovers.
        // Each crossover splits the remaining signal into low (LP) and high (HP).
        // LR4 = two cascaded 2nd-order Butterworth biquads (Q = 1/√2 each).
        let mut remaining = buf.clone();
        for band_idx in 0..n_bands {
            match self.bands[band_idx].crossover_hz {
                Some(cf) => {
                    // LR4 lowpass: apply Butterworth LP twice at the crossover frequency.
                    let lp1 = BiquadFilter::lowpass(cf, FRAC_1_SQRT_2, sr);
                    let lp2 = BiquadFilter::lowpass(cf, FRAC_1_SQRT_2, sr);
                    let low = lp2.process(&lp1.process(&remaining));

                    // LR4 highpass: apply Butterworth HP twice at the crossover frequency.
                    let hp1 = BiquadFilter::highpass(cf, FRAC_1_SQRT_2, sr);
                    let hp2 = BiquadFilter::highpass(cf, FRAC_1_SQRT_2, sr);
                    let high = hp2.process(&hp1.process(&remaining));

                    band_signals.push(low);
                    remaining = high;
                }
                None => {
                    // Last band: push whatever remains, ignore any further entries.
                    band_signals.push(remaining.clone());
                }
            }
        }

        // Compress each band independently.
        let mut compressed_bands: Vec<AudioBuffer<f32>> = Vec::with_capacity(n_bands);
        for (band_idx, band_signal) in band_signals.iter().enumerate() {
            let settings = &self.bands[band_idx];
            let comp = Compressor::new(
                settings.threshold_db,
                settings.ratio,
                settings.attack_ms,
                settings.release_ms,
            )
            .with_makeup(settings.makeup_gain_db);
            compressed_bands.push(comp.process(band_signal));
        }

        // Sum the compressed bands sample-by-sample.
        let n_samples = buf.samples.len();
        let mut output = vec![0.0f32; n_samples];
        for band in &compressed_bands {
            for (i, &s) in band.samples.iter().enumerate() {
                if i < output.len() {
                    output[i] += s;
                }
            }
        }

        Ok(AudioBuffer {
            samples: output,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        })
    }
}

impl AudioFilter for MultibandCompressor {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        self.process(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{ChannelLayout, SampleFormat};
    use std::f32::consts::PI;

    fn make_sine(freq: f32, amplitude: f32, sr: u32, secs: f32) -> AudioBuffer<f32> {
        let n = (sr as f32 * secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect();
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn compressor_reduces_loud_signal() {
        let amplitude = 10.0f32.powf(-6.0 / 20.0); // -6 dBFS
        let buf = make_sine(1000.0, amplitude, 48_000, 0.5);
        let comp = Compressor::new(-20.0, 4.0, 5.0, 50.0);
        let out = comp.process(&buf);
        let peak_db = 20.0 * out.peak_amplitude().log10();
        assert!(
            peak_db < -6.0,
            "compressor should reduce level below -6dBFS, got {peak_db:.1}dBFS"
        );
    }

    #[test]
    fn limiter_hard_ceiling() {
        let amp = 10.0f32.powf(-3.0 / 20.0);
        let buf = make_sine(1000.0, amp, 48_000, 0.5);
        let limiter = Limiter::new(-6.0, 50.0);
        let out = limiter.process(&buf);
        let peak_db = 20.0 * out.peak_amplitude().log10();
        assert!(
            peak_db <= -5.0,
            "limiter should cap level near -6dBFS, got {peak_db:.1}dBFS"
        );
    }

    #[test]
    fn noise_gate_attenuates_quiet_signal() {
        let buf = make_sine(1000.0, 0.001, 48_000, 0.5); // -60 dBFS
        let gate = NoiseGate::new(-40.0);
        let out = gate.process(&buf);
        let peak_out = out.peak_amplitude();
        assert!(
            peak_out < 0.001,
            "gate should attenuate quiet signal, got peak={peak_out:.5}"
        );
    }

    #[test]
    fn noise_gate_passes_loud_signal() {
        let buf = make_sine(1000.0, 0.5, 48_000, 1.0);
        let gate = NoiseGate::new(-40.0);
        let out = gate.process(&buf);
        let peak_db = 20.0 * out.peak_amplitude().log10();
        assert!(
            peak_db > -12.0,
            "gate should pass loud signal, got {peak_db:.1}dBFS"
        );
    }

    #[test]
    fn expander_attenuates_quiet_signal() {
        // -50 dBFS tone, threshold -40 dBFS, 2:1 expansion -> further attenuated.
        let amp = 10.0f32.powf(-50.0 / 20.0);
        let buf = make_sine(1000.0, amp, 48_000, 0.5);
        let exp = Expander::new(-40.0, 2.0, 1.0, 50.0);
        let out = exp.process(&buf);
        let peak_db = 20.0 * out.peak_amplitude().log10();
        assert!(
            peak_db < -50.0,
            "expander should push -50dBFS quiet signal lower, got {peak_db:.1}dBFS"
        );
    }

    #[test]
    fn expander_passes_loud_signal() {
        // -6 dBFS tone, threshold -40 dBFS -> above threshold, essentially unchanged.
        let amp = 10.0f32.powf(-6.0 / 20.0);
        let buf = make_sine(1000.0, amp, 48_000, 0.5);
        let exp = Expander::new(-40.0, 2.0, 1.0, 50.0);
        let out = exp.process(&buf);
        let peak_db = 20.0 * out.peak_amplitude().log10();
        assert!(
            peak_db > -8.0,
            "expander should pass loud signal, got {peak_db:.1}dBFS"
        );
    }

    #[test]
    fn test_expander_reduces_quiet_signal() {
        // Input at -50 dBFS (below -40 dB threshold), expander with ratio 4:1.
        // Output should be significantly quieter than input.
        let quiet_level = 10.0f32.powf(-50.0 / 20.0);
        let buf = AudioBuffer {
            samples: vec![quiet_level; 44100],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: oxiaudio_core::SampleFormat::F32,
        };
        let exp = Expander::new(-40.0, 4.0, 10.0, 100.0);
        let out = exp.process(&buf);
        let out_rms =
            (out.samples.iter().map(|&s| s * s).sum::<f32>() / out.samples.len() as f32).sqrt();
        assert!(
            out_rms < quiet_level * 0.5,
            "expander should have reduced quiet signal: in_rms={quiet_level:.6} out_rms={out_rms:.6}"
        );
    }

    #[test]
    fn test_expander_passes_loud_signal() {
        // Input at -10 dBFS (above -40 dB threshold): gain ≈ 0 dB after settling.
        let loud_level = 10.0f32.powf(-10.0 / 20.0);
        let buf = AudioBuffer {
            samples: vec![loud_level; 44100],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: oxiaudio_core::SampleFormat::F32,
        };
        let exp = Expander::new(-40.0, 4.0, 10.0, 100.0);
        let out = exp.process(&buf);
        // Check tail after envelope has settled.
        let tail = &out.samples[44000..];
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!(
            (mean - loud_level).abs() < 0.01,
            "expander should pass loud signal: mean={mean:.4} expected≈{loud_level:.4}"
        );
    }

    #[test]
    fn test_deesser_reduces_sibilant() {
        // High-frequency sine (7 kHz) above threshold should be attenuated.
        let sr = 44100_u32;
        let n = sr as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (2.0 * PI * 7000.0 * t).sin() * 0.3
            })
            .collect();
        let in_rms = (samples.iter().map(|&s| s * s).sum::<f32>() / n as f32).sqrt();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: oxiaudio_core::SampleFormat::F32,
        };
        let de = DeEsser::new(-20.0, sr);
        let out = de.process(&buf);
        let out_rms = (out.samples.iter().map(|&s| s * s).sum::<f32>() / n as f32).sqrt();
        assert!(
            out_rms < in_rms,
            "deesser should reduce sibilant: in={in_rms:.4} out={out_rms:.4}"
        );
    }

    fn silent_buf(sr: u32, n_frames: usize, ch: ChannelLayout) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.0; n_frames * ch.channel_count()],
            sample_rate: sr,
            channels: ch,
            format: SampleFormat::F32,
        }
    }

    fn loud_buf(sr: u32, n_frames: usize, amplitude: f32) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![amplitude; n_frames],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_multiband_compressor_output_not_silent() {
        let sr = 44100_u32;
        let n = sr as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: oxiaudio_core::SampleFormat::F32,
        };
        let mbc = MultibandCompressor::three_band(-20.0, -20.0, -20.0);
        let out = mbc.process(&buf).expect("three_band process");
        let rms = (out.samples.iter().map(|&s| s * s).sum::<f32>() / n as f32).sqrt();
        assert!(
            rms > 0.01,
            "multiband output should not be silent: rms={rms:.4}"
        );
    }

    #[test]
    fn test_multiband_compressor_three_band_processes() {
        let buf = loud_buf(44_100, 4096, 0.8);
        let mbc = MultibandCompressor::three_band(-20.0, -20.0, -20.0);
        let out = mbc.process(&buf).expect("multiband process");
        assert_eq!(out.samples.len(), buf.samples.len());
        let rms_in: f32 =
            (buf.samples.iter().map(|&s| s * s).sum::<f32>() / buf.samples.len() as f32).sqrt();
        let rms_out: f32 =
            (out.samples.iter().map(|&s| s * s).sum::<f32>() / out.samples.len() as f32).sqrt();
        assert!(
            rms_out < rms_in * 1.1,
            "multiband should not significantly increase level: in={rms_in:.4} out={rms_out:.4}"
        );
    }

    #[test]
    fn test_multiband_compressor_silence_passthrough() {
        let buf = silent_buf(44_100, 4096, ChannelLayout::Mono);
        let mbc = MultibandCompressor::three_band(-20.0, -20.0, -20.0);
        let out = mbc.process(&buf).expect("process silence");
        assert_eq!(out.samples.len(), buf.samples.len());
        let max: f32 = out.samples.iter().copied().fold(0.0f32, f32::max);
        assert!(
            max.abs() < 0.01,
            "near-silence should pass through: max={max}"
        );
    }

    #[test]
    fn test_multiband_compressor_audio_filter_trait() {
        let buf = loud_buf(44_100, 1024, 0.5);
        let mbc = MultibandCompressor::three_band(-30.0, -30.0, -30.0);
        let out = mbc.apply(&buf).expect("apply via AudioFilter");
        assert_eq!(out.samples.len(), buf.samples.len());
    }

    // ── Sidechain compressor tests ────────────────────────────────────────────

    #[test]
    fn test_sidechain_compressor_ducks_signal() {
        let sr = 48_000u32;
        let n = sr as usize;
        // Loud input (all 1.0) and loud key (all 1.0): compression should duck input
        let input_loud = AudioBuffer {
            samples: vec![1.0f32; n],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let key_loud = AudioBuffer {
            samples: vec![1.0f32; n],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let comp = Compressor::new(-20.0, 4.0, 5.0, 50.0);
        let out = comp
            .process_with_sidechain(&input_loud, &key_loud)
            .expect("sidechain loud/loud should not error");
        let tail = &out.samples[(n * 3 / 4)..];
        let mean_out: f32 = tail.iter().sum::<f32>() / tail.len() as f32;
        // With loud key driving compression, output should be significantly less than 1.0
        assert!(
            mean_out < 0.9,
            "loud key should duck loud input: mean_tail={mean_out:.4}"
        );

        // Silent key (all 0.0): input should pass through with minimal gain reduction
        let key_silent = AudioBuffer {
            samples: vec![0.0f32; n],
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let out_silent_key = comp
            .process_with_sidechain(&input_loud, &key_silent)
            .expect("sidechain with silent key should not error");
        let tail_silent = &out_silent_key.samples[(n * 3 / 4)..];
        let mean_silent: f32 = tail_silent.iter().sum::<f32>() / tail_silent.len() as f32;
        // Silent key → no compression trigger → output ≈ input (≈ 1.0)
        assert!(
            mean_silent > 0.9,
            "silent key should let signal through: mean_tail={mean_silent:.4}"
        );
    }

    #[test]
    fn test_sidechain_sample_rate_mismatch_returns_err() {
        let input = AudioBuffer {
            samples: vec![1.0f32; 1024],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let key = AudioBuffer {
            samples: vec![1.0f32; 1024],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let comp = Compressor::new(-20.0, 4.0, 5.0, 50.0);
        let result = comp.process_with_sidechain(&input, &key);
        assert!(result.is_err(), "mismatched sample rates should return Err");
    }

    #[test]
    fn test_multiband_compressor_band_settings_custom() {
        let mbc = MultibandCompressor {
            bands: vec![
                BandSettings {
                    crossover_hz: Some(500.0),
                    threshold_db: -12.0,
                    ratio: 2.0,
                    attack_ms: 10.0,
                    release_ms: 100.0,
                    makeup_gain_db: 0.0,
                },
                BandSettings {
                    crossover_hz: None,
                    threshold_db: -6.0,
                    ratio: 8.0,
                    attack_ms: 1.0,
                    release_ms: 50.0,
                    makeup_gain_db: 2.0,
                },
            ],
        };
        let buf = loud_buf(48_000, 2048, 0.9);
        let out = mbc.process(&buf).expect("custom 2-band");
        assert_eq!(out.samples.len(), buf.samples.len());
    }

    #[test]
    fn compressor_4to1_at_minus6dbfs_with_threshold_minus12() {
        // Compressor: threshold=-12 dBFS, ratio=4.0, attack=1ms, release=100ms.
        // Input: 440 Hz sine, peak amplitude 0.5 (-6 dBFS).
        // Input is 6 dB above threshold (-6 - (-12) = 6 dB).
        // After steady-state: gain reduction = 6*(1 - 1/4) = 4.5 dB.
        // Output level ≈ -6 - 4.5 = -10.5 dBFS => amplitude ≈ 10^(-10.5/20) ≈ 0.299.
        // We check output peak is in (0.15, 0.50) — covers envelope follower convergence.
        let sr = 48_000u32;
        let n = (sr as f32 * 0.5) as usize;
        let input_amp = 0.5f32; // peak = -6 dBFS
        let samples: Vec<f32> = (0..n)
            .map(|i| input_amp * (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let comp = Compressor::new(-12.0, 4.0, 1.0, 100.0);
        let out = comp.process(&buf);
        let peak_out = out.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        // Output must be less than input (compression applied)
        assert!(
            peak_out < input_amp,
            "compressor should reduce level: in_peak={input_amp:.3} out_peak={peak_out:.3}"
        );
        // Rough lower bound: envelope follower converges after attack time
        assert!(
            peak_out > 0.15,
            "compressor should not silence signal: out_peak={peak_out:.3}"
        );
    }
}
