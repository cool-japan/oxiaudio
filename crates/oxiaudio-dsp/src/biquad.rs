use std::f32::consts::PI;

use oxiaudio_core::{AudioBuffer, AudioFilter, OxiAudioError};

/// Biquad filter coefficients in Direct Form II Transposed.
///
/// Coefficients follow the RBJ Audio EQ Cookbook normalization:
/// `b0`, `b1`, `b2` are the feed-forward coefficients (divided by `a0`),
/// `a1`, `a2` are the feedback coefficients (divided by `a0`, with sign from the standard IIR equation).
#[derive(Debug, Clone, Copy)]
pub struct BiquadFilter {
    /// Feed-forward coefficient b0 / a0.
    pub b0: f32,
    /// Feed-forward coefficient b1 / a0.
    pub b1: f32,
    /// Feed-forward coefficient b2 / a0.
    pub b2: f32,
    /// Feedback coefficient a1 / a0 (used as `y[n-1]` coefficient in the recurrence).
    pub a1: f32,
    /// Feedback coefficient a2 / a0 (used as `y[n-2]` coefficient in the recurrence).
    pub a2: f32,
}

impl BiquadFilter {
    /// Low-shelf filter with shelf slope S = 1 (RBJ Audio EQ Cookbook).
    ///
    /// - `frequency`: shelf corner frequency in Hz.
    /// - `gain_db`: shelf gain in dB (positive = boost, negative = cut).
    /// - `sample_rate`: sample rate in Hz.
    pub fn low_shelf(frequency: f32, gain_db: f32, sample_rate: u32) -> Self {
        let a = 10_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * frequency / sample_rate as f32;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        // S = 1 simplification: alpha = sin(w0)/2 * sqrt(2)
        let alpha = sin_w0 * 0.5 * (2.0_f32).sqrt();
        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// High-shelf filter with shelf slope S = 1 (RBJ Audio EQ Cookbook).
    ///
    /// - `frequency`: shelf corner frequency in Hz.
    /// - `gain_db`: shelf gain in dB (positive = boost, negative = cut).
    /// - `sample_rate`: sample rate in Hz.
    pub fn high_shelf(frequency: f32, gain_db: f32, sample_rate: u32) -> Self {
        let a = 10_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * frequency / sample_rate as f32;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        // S = 1 simplification: alpha = sin(w0)/2 * sqrt(2)
        let alpha = sin_w0 * 0.5 * (2.0_f32).sqrt();
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Peaking EQ filter (RBJ Audio EQ Cookbook).
    ///
    /// - `frequency`: center frequency in Hz.
    /// - `q`: quality factor (bandwidth control; 0.707 = 1 octave).
    /// - `gain_db`: peak gain in dB (positive = boost, negative = cut).
    /// - `sample_rate`: sample rate in Hz.
    pub fn peaking_eq(frequency: f32, q: f32, gain_db: f32, sample_rate: u32) -> Self {
        let a = 10_f32.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * frequency / sample_rate as f32;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha / a;
        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Lowpass filter (RBJ Audio EQ Cookbook).
    ///
    /// - `frequency`: cutoff frequency in Hz.
    /// - `q`: quality factor (0.707 = Butterworth maximally flat).
    /// - `sample_rate`: sample rate in Hz.
    pub fn lowpass(frequency: f32, q: f32, sample_rate: u32) -> Self {
        let w0 = 2.0 * PI * frequency / sample_rate as f32;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Highpass filter (RBJ Audio EQ Cookbook).
    ///
    /// - `frequency`: cutoff frequency in Hz.
    /// - `q`: quality factor (0.707 = Butterworth maximally flat).
    /// - `sample_rate`: sample rate in Hz.
    pub fn highpass(frequency: f32, q: f32, sample_rate: u32) -> Self {
        let w0 = 2.0 * PI * frequency / sample_rate as f32;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Bandpass filter with constant 0 dB peak gain (RBJ Audio EQ Cookbook, bandwidth form).
    ///
    /// - `frequency`: center frequency in Hz.
    /// - `q`: quality factor.
    /// - `sample_rate`: sample rate in Hz.
    pub fn bandpass(frequency: f32, q: f32, sample_rate: u32) -> Self {
        let w0 = 2.0 * PI * frequency / sample_rate as f32;
        let alpha = w0.sin() / (2.0 * q);
        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * w0.cos();
        let a2 = 1.0 - alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Notch (band-reject) filter (RBJ Audio EQ Cookbook).
    ///
    /// - `frequency`: center (notch) frequency in Hz.
    /// - `q`: quality factor (higher Q = narrower notch).
    /// - `sample_rate`: sample rate in Hz.
    pub fn notch(frequency: f32, q: f32, sample_rate: u32) -> Self {
        let w0 = 2.0 * PI * frequency / sample_rate as f32;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = 1.0;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Allpass filter (RBJ Audio EQ Cookbook).
    ///
    /// Flat magnitude response; introduces frequency-dependent phase shift.
    ///
    /// - `frequency`: center frequency in Hz.
    /// - `q`: quality factor (controls phase transition width).
    /// - `sample_rate`: sample rate in Hz.
    pub fn allpass(frequency: f32, q: f32, sample_rate: u32) -> Self {
        let w0 = 2.0 * PI * frequency / sample_rate as f32;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = 1.0 - alpha;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 + alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Apply the filter to an interleaved `AudioBuffer<f32>` using Direct Form II Transposed.
    ///
    /// Filter state (z1, z2) is initialized to zero on each call — this is a stateless
    /// (block-mode) interface. For real-time streaming, maintain state externally.
    #[inline]
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let n_channels = buf.channels.channel_count();
        // Per-channel state for Direct Form II Transposed
        let mut z1 = vec![0.0_f32; n_channels];
        let mut z2 = vec![0.0_f32; n_channels];
        let mut out = Vec::with_capacity(buf.samples.len());
        for (i, &x) in buf.samples.iter().enumerate() {
            let ch = i % n_channels;
            let y = self.b0 * x + z1[ch];
            z1[ch] = self.b1 * x - self.a1 * y + z2[ch];
            z2[ch] = self.b2 * x - self.a2 * y;
            out.push(y);
        }
        AudioBuffer {
            samples: out,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }

    /// Process a mono sample buffer in-place using 4-step unrolled loop.
    ///
    /// Uses `chunks_exact_mut(4)` to hint LLVM to unroll and pipeline the
    /// scalar IIR recurrence, which aids instruction-level parallelism on
    /// x86_64 (SSE/AVX) and aarch64 (NEON) targets even without explicit SIMD
    /// intrinsics. True SIMD auto-vectorisation is limited for IIR filters due
    /// to the data dependency chain (each output depends on the previous two);
    /// the 4-step unroll reduces loop overhead and exposes pipeline slots.
    ///
    /// Filter state (z1, z2) is initialised to zero on each call — this is the
    /// stateless block-mode interface. For streaming, maintain state externally.
    pub fn process_buffer(&self, buf: &mut [f32]) {
        let mut z1 = 0.0_f32;
        let mut z2 = 0.0_f32;
        let mut iter = buf.chunks_exact_mut(4);
        for chunk in iter.by_ref() {
            // Unrolled 4-step biquad (Direct Form II Transposed).
            let y0 = self.b0 * chunk[0] + z1;
            z1 = self.b1 * chunk[0] - self.a1 * y0 + z2;
            z2 = self.b2 * chunk[0] - self.a2 * y0;
            chunk[0] = y0;

            let y1 = self.b0 * chunk[1] + z1;
            z1 = self.b1 * chunk[1] - self.a1 * y1 + z2;
            z2 = self.b2 * chunk[1] - self.a2 * y1;
            chunk[1] = y1;

            let y2 = self.b0 * chunk[2] + z1;
            z1 = self.b1 * chunk[2] - self.a1 * y2 + z2;
            z2 = self.b2 * chunk[2] - self.a2 * y2;
            chunk[2] = y2;

            let y3 = self.b0 * chunk[3] + z1;
            z1 = self.b1 * chunk[3] - self.a1 * y3 + z2;
            z2 = self.b2 * chunk[3] - self.a2 * y3;
            chunk[3] = y3;
        }
        for s in iter.into_remainder() {
            let y = self.b0 * *s + z1;
            z1 = self.b1 * *s - self.a1 * y + z2;
            z2 = self.b2 * *s - self.a2 * y;
            *s = y;
        }
    }

    /// Apply this biquad filter to all channels of a buffer independently,
    /// using `chunks_exact` to iterate over interleaved frames.
    ///
    /// Each channel gets its own z1/z2 state, preventing inter-channel crosstalk.
    /// The inner loop deinterleaves, filters, and reinterleaves in a single pass per
    /// channel, which exposes the per-frame stride pattern to LLVM's auto-vectoriser.
    ///
    /// # Behaviour
    /// - The filter state is reset to zero on each call (stateless block-mode interface).
    /// - Samples that do not form a complete frame (i.e. `buf.samples.len() % ch != 0`)
    ///   are copied unmodified via `chunks_exact.remainder()`.
    #[inline]
    pub fn process_multichannel(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let ch = buf.channels.channel_count();
        if ch == 0 || buf.samples.is_empty() {
            return buf.clone();
        }
        let mut out = vec![0.0_f32; buf.samples.len()];

        // Process each channel independently with its own z1/z2 state.
        for c in 0..ch {
            let mut z1 = 0.0_f32;
            let mut z2 = 0.0_f32;
            // Deinterleave + filter + reinterleave in one pass using chunks_exact(ch).
            for (frame_idx, frame) in buf.samples.chunks_exact(ch).enumerate() {
                let x = frame[c];
                let y = self.b0 * x + z1;
                z1 = self.b1 * x - self.a1 * y + z2;
                z2 = self.b2 * x - self.a2 * y;
                out[frame_idx * ch + c] = y;
            }
            // Copy any remainder samples unmodified (malformed / non-frame-aligned buffers).
            let full_frames = (buf.samples.len() / ch) * ch;
            for (i, &s) in buf.samples[full_frames..].iter().enumerate() {
                out[full_frames + i] = s;
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

impl AudioFilter for BiquadFilter {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

// ─── Parametric EQ ───────────────────────────────────────────────────────────

/// A chain of [`BiquadFilter`]s applied in series.
///
/// Each band is processed sequentially, passing its output as the input of the next.
#[derive(Debug, Clone)]
pub struct ParametricEq {
    /// Ordered list of biquad filter bands.
    pub bands: Vec<BiquadFilter>,
}

impl ParametricEq {
    /// Create a new `ParametricEq` from a list of biquad bands.
    pub fn new(bands: Vec<BiquadFilter>) -> Self {
        Self { bands }
    }

    /// Apply all bands in series to the buffer. Returns the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let mut current = buf.clone();
        for band in &self.bands {
            current = band.process(&current);
        }
        current
    }
}

impl AudioFilter for ParametricEq {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        Ok(self.process(buf))
    }
}

impl ParametricEq {
    /// Compute magnitude response in dB at each frequency (Hz).
    ///
    /// Uses `10 * log10(|H(e^jw)|^2) = 20 * log10(|H(e^jw)|)`.
    /// Returns `f32::NEG_INFINITY` for frequencies where the gain is zero.
    pub fn frequency_response(&self, freqs: &[f32], sample_rate: u32) -> Vec<f32> {
        freqs
            .iter()
            .map(|&f| {
                let w = 2.0 * PI * f / sample_rate as f32;
                let mag_sq = self.transfer_mag_sq(w);
                if mag_sq <= 0.0 {
                    return f32::NEG_INFINITY;
                }
                10.0 * mag_sq.log10()
            })
            .collect()
    }

    /// Compute phase response in radians at each frequency (Hz).
    pub fn phase_response(&self, freqs: &[f32], sample_rate: u32) -> Vec<f32> {
        freqs
            .iter()
            .map(|&f| {
                let w = 2.0 * PI * f / sample_rate as f32;
                self.transfer_phase(w)
            })
            .collect()
    }

    /// Compute group delay in samples at each frequency (Hz).
    ///
    /// Uses numerical differentiation: `-d(phase)/dw`.
    pub fn group_delay(&self, freqs: &[f32], sample_rate: u32) -> Vec<f32> {
        let eps = 1e-4f32;
        freqs
            .iter()
            .map(|&f| {
                let w = 2.0 * PI * f / sample_rate as f32;
                let p1 = self.transfer_phase(w + eps);
                let p0 = self.transfer_phase(w - eps);
                let mut dp = p1 - p0;
                while dp > PI {
                    dp -= 2.0 * PI;
                }
                while dp < -PI {
                    dp += 2.0 * PI;
                }
                -dp / (2.0 * eps)
            })
            .collect()
    }

    /// |H(e^jw)|^2 — product of all biquad section magnitude-squared values.
    fn transfer_mag_sq(&self, w: f32) -> f32 {
        self.bands
            .iter()
            .fold(1.0f32, |acc, filt| acc * biquad_mag_sq(filt, w))
    }

    /// arg(H(e^jw)) — sum of all biquad section phase values.
    fn transfer_phase(&self, w: f32) -> f32 {
        self.bands
            .iter()
            .fold(0.0f32, |acc, filt| acc + biquad_phase(filt, w))
    }
}

/// Compute |H(e^jw)|^2 for a single biquad section.
///
/// `H(e^jw) = (b0 + b1*e^-jw + b2*e^-2jw) / (1 + a1*e^-jw + a2*e^-2jw)`
fn biquad_mag_sq(f: &BiquadFilter, w: f32) -> f32 {
    let cos_w = w.cos();
    let cos_2w = (2.0 * w).cos();
    let sin_w = w.sin();
    let sin_2w = (2.0 * w).sin();

    let num_re = f.b0 + f.b1 * cos_w + f.b2 * cos_2w;
    let num_im = -(f.b1 * sin_w + f.b2 * sin_2w);
    let den_re = 1.0 + f.a1 * cos_w + f.a2 * cos_2w;
    let den_im = -(f.a1 * sin_w + f.a2 * sin_2w);

    let num_sq = num_re * num_re + num_im * num_im;
    let den_sq = den_re * den_re + den_im * den_im;
    if den_sq < 1e-30 {
        return 0.0;
    }
    num_sq / den_sq
}

/// Compute arg(H(e^jw)) for a single biquad section.
///
/// `arg(H) = arg(numerator) - arg(denominator)`
fn biquad_phase(f: &BiquadFilter, w: f32) -> f32 {
    let cos_w = w.cos();
    let cos_2w = (2.0 * w).cos();
    let sin_w = w.sin();
    let sin_2w = (2.0 * w).sin();

    let num_re = f.b0 + f.b1 * cos_w + f.b2 * cos_2w;
    let num_im = -(f.b1 * sin_w + f.b2 * sin_2w);
    let den_re = 1.0 + f.a1 * cos_w + f.a2 * cos_2w;
    let den_im = -(f.a1 * sin_w + f.a2 * sin_2w);

    num_im.atan2(num_re) - den_im.atan2(den_re)
}

impl ParametricEq {
    /// Create a graphic equalizer from per-band gain values at ISO standard centre frequencies.
    ///
    /// - `gains_db`: gain in dB for each band (positive = boost, negative = cut).
    /// - `n_bands`: `10` for 1-octave ISO bands, `31` for ⅓-octave ISO bands.
    /// - `sample_rate`: sample rate in Hz.
    ///
    /// Returns an error if `gains_db.len() != n_bands` or `n_bands` is not `10` or `31`.
    #[must_use = "returns the configured ParametricEq"]
    pub fn graphic_eq(
        gains_db: &[f32],
        n_bands: usize,
        sample_rate: u32,
    ) -> Result<Self, OxiAudioError> {
        if gains_db.len() != n_bands {
            return Err(OxiAudioError::UnsupportedFormat(format!(
                "gains_db.len() ({}) must equal n_bands ({})",
                gains_db.len(),
                n_bands
            )));
        }

        const ISO_10_BAND: [f32; 10] = [
            31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
        ];
        // Q for 1-octave: sqrt(2) ≈ 1.414
        const Q_10: f32 = 1.414_f32;

        const ISO_31_BAND: [f32; 31] = [
            20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0,
            400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0,
            5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
        ];
        // Q for ⅓-octave: 2^(1/6) / (2^(1/3) − 1) ≈ 4.318
        const Q_31: f32 = 4.318_f32;

        let (freqs, q): (&[f32], f32) = match n_bands {
            10 => (&ISO_10_BAND, Q_10),
            31 => (&ISO_31_BAND, Q_31),
            _ => {
                return Err(OxiAudioError::UnsupportedFormat(format!(
                    "n_bands must be 10 or 31, got {n_bands}"
                )))
            }
        };

        let bands: Vec<BiquadFilter> = freqs
            .iter()
            .zip(gains_db.iter())
            .map(|(&freq, &gain_db)| BiquadFilter::peaking_eq(freq, q, gain_db, sample_rate))
            .collect();

        Ok(ParametricEq { bands })
    }
}

/// Design an `order`-pole Butterworth **lowpass** and return it as a cascade of
/// [`BiquadFilter`] second-order sections.
///
/// Apply all returned sections in series (each section's output feeds the next).
/// See also [`crate::filters::butterworth_lowpass`] for a [`crate::filters::Cascade`]-typed
/// variant that wraps the same sections.
pub fn butterworth_lowpass(order: usize, frequency: f32, sample_rate: u32) -> Vec<BiquadFilter> {
    crate::filters::butterworth_lowpass(order, frequency, sample_rate).sections
}

/// Design an `order`-pole Butterworth **highpass** and return it as a cascade of
/// [`BiquadFilter`] second-order sections.
pub fn butterworth_highpass(order: usize, frequency: f32, sample_rate: u32) -> Vec<BiquadFilter> {
    crate::filters::butterworth_highpass(order, frequency, sample_rate).sections
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
                (2.0 * PI * freq_hz * t).sin()
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_biquad_passthrough() {
        // peaking_eq with 0 dB gain must be a unity-gain filter.
        // Steady-state: skip first 1024 samples (transient settling).
        let buf = sine_buf(440.0, 48_000, 0.5);
        let filter = BiquadFilter::peaking_eq(1000.0, 0.707, 0.0, 48_000);
        let out = filter.process(&buf);
        assert_eq!(out.samples.len(), buf.samples.len());
        for i in 1024..buf.samples.len() {
            let diff = (out.samples[i] - buf.samples[i]).abs();
            assert!(
                diff < 1e-4,
                "sample {i}: expected {} got {} (diff={diff})",
                buf.samples[i],
                out.samples[i]
            );
        }
    }

    #[test]
    fn test_biquad_low_shelf_boosts_low_freq() {
        // A +6 dB low-shelf at 1 kHz applied to a 200 Hz sine (below shelf) should
        // produce higher RMS energy than the original in steady state.
        let buf = sine_buf(200.0, 48_000, 0.5);
        let filter = BiquadFilter::low_shelf(1000.0, 6.0, 48_000);
        let out = filter.process(&buf);
        // Skip first 2048 samples (transient settling)
        let start = 2048;
        let orig_rms: f32 = {
            let sq_sum: f32 = buf.samples[start..].iter().map(|&s| s * s).sum();
            (sq_sum / (buf.samples.len() - start) as f32).sqrt()
        };
        let out_rms: f32 = {
            let sq_sum: f32 = out.samples[start..].iter().map(|&s| s * s).sum();
            (sq_sum / (out.samples.len() - start) as f32).sqrt()
        };
        assert!(
            out_rms > orig_rms * 1.3,
            "expected boosted RMS ({out_rms:.4}) to be > 1.3× original ({orig_rms:.4})"
        );
    }

    #[test]
    fn test_parametric_eq_applies_all_bands() {
        // 2-band ParametricEq: low-shelf boost + high-shelf cut.
        // Output must differ from input.
        let buf = sine_buf(440.0, 48_000, 0.1);
        let eq = ParametricEq::new(vec![
            BiquadFilter::low_shelf(500.0, 6.0, 48_000),
            BiquadFilter::high_shelf(8000.0, -6.0, 48_000),
        ]);
        let out = eq.process(&buf);
        assert_eq!(out.samples.len(), buf.samples.len());
        // Verify at least some samples differ
        let any_diff = buf
            .samples
            .iter()
            .zip(out.samples.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-6);
        assert!(
            any_diff,
            "ParametricEq output is identical to input — bands had no effect"
        );
    }

    #[test]
    fn lowpass_attenuates_high_freq() {
        let sr = 48_000u32;
        let n = sr as usize;
        // 10kHz sine through 1kHz lowpass
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 10_000.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let filter = BiquadFilter::lowpass(1_000.0, 0.707, sr);
        let out = filter.process(&buf);
        // Skip initial transient (first 100 samples), measure RMS attenuation
        let rms_in: f32 =
            (buf.samples[100..].iter().map(|s| s * s).sum::<f32>() / (n - 100) as f32).sqrt();
        let rms_out: f32 =
            (out.samples[100..].iter().map(|s| s * s).sum::<f32>() / (n - 100) as f32).sqrt();
        let attenuation_db = 20.0 * (rms_out / rms_in.max(1e-10)).log10();
        assert!(
            attenuation_db < -40.0,
            "lowpass 1kHz should attenuate 10kHz by >40dB, got {attenuation_db:.1}dB"
        );
    }

    #[test]
    fn highpass_passes_high_freq() {
        let sr = 48_000u32;
        let n = sr as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 10_000.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let filter = BiquadFilter::highpass(1_000.0, 0.707, sr);
        let out = filter.process(&buf);
        let rms_in: f32 =
            (buf.samples[200..].iter().map(|s| s * s).sum::<f32>() / (n - 200) as f32).sqrt();
        let rms_out: f32 =
            (out.samples[200..].iter().map(|s| s * s).sum::<f32>() / (n - 200) as f32).sqrt();
        let ratio_db = 20.0 * (rms_out / rms_in.max(1e-10)).log10();
        assert!(
            ratio_db > -3.0,
            "highpass should pass 10kHz with < 3dB loss, got {ratio_db:.1}dB"
        );
    }

    #[test]
    fn notch_removes_center() {
        let sr = 48_000u32;
        let n = sr as usize;
        let fc = 1_000.0f32;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * fc * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let filter = BiquadFilter::notch(fc, 10.0, sr); // high Q for sharp notch
        let out = filter.process(&buf);
        let rms_in: f32 =
            (buf.samples[500..].iter().map(|s| s * s).sum::<f32>() / (n - 500) as f32).sqrt();
        let rms_out: f32 =
            (out.samples[500..].iter().map(|s| s * s).sum::<f32>() / (n - 500) as f32).sqrt();
        let attenuation_db = 20.0 * (rms_out / rms_in.max(1e-10)).log10();
        assert!(
            attenuation_db < -10.0,
            "notch should attenuate center freq, got {attenuation_db:.1}dB"
        );
    }

    #[test]
    fn allpass_preserves_magnitude() {
        let sr = 48_000u32;
        let n = 10_000usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * 1000.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let filter = BiquadFilter::allpass(1_000.0, 0.707, sr);
        let out = filter.process(&buf);
        let rms_in: f32 =
            (buf.samples[200..].iter().map(|s| s * s).sum::<f32>() / (n - 200) as f32).sqrt();
        let rms_out: f32 =
            (out.samples[200..].iter().map(|s| s * s).sum::<f32>() / (n - 200) as f32).sqrt();
        let ratio_db = 20.0 * (rms_out / rms_in.max(1e-10)).log10();
        assert!(
            ratio_db.abs() < 1.0,
            "allpass should preserve magnitude, got {ratio_db:.1}dB change"
        );
    }

    #[test]
    fn test_butterworth_lowpass_order2() {
        // 1 kHz cutoff at 48 kHz, order 2; 10 kHz sine should be attenuated by >24 dB.
        let sr = 48_000u32;
        let n = sr as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 10_000.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let sections = butterworth_lowpass(2, 1_000.0, sr);
        assert!(!sections.is_empty(), "should produce sections");
        // Apply cascade
        let mut current = buf.clone();
        for s in &sections {
            current = s.process(&current);
        }
        let rms_in: f32 =
            (buf.samples[500..].iter().map(|s| s * s).sum::<f32>() / (n - 500) as f32).sqrt();
        let rms_out: f32 =
            (current.samples[500..].iter().map(|s| s * s).sum::<f32>() / (n - 500) as f32).sqrt();
        let atten_db = 20.0 * (rms_out / rms_in.max(1e-10)).log10();
        assert!(
            atten_db < -24.0,
            "butterworth LP order=2 at 1kHz should attenuate 10kHz by >24dB, got {atten_db:.1}dB"
        );
    }

    #[test]
    fn test_fir_design_lowpass() {
        // 101-tap lowpass at 1kHz/48kHz; 10kHz should be attenuated by >40dB.
        use crate::filters::{FirFilter, FirWindow};
        let sr = 48_000u32;
        let n = sr as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 10_000.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let fir = FirFilter::design_lowpass(101, 1_000.0, sr, FirWindow::Blackman);
        let out = fir.process(&buf);
        let rms_in: f32 =
            (buf.samples[200..].iter().map(|s| s * s).sum::<f32>() / (n - 200) as f32).sqrt();
        let rms_out: f32 =
            (out.samples[200..].iter().map(|s| s * s).sum::<f32>() / (n - 200) as f32).sqrt();
        let atten_db = 20.0 * (rms_out / rms_in.max(1e-10)).log10();
        assert!(
            atten_db < -40.0,
            "FIR LP 101-tap at 1kHz should attenuate 10kHz by >40dB, got {atten_db:.1}dB"
        );
    }

    #[test]
    fn test_parametriceq_frequency_response_unity() {
        // Empty ParametricEq (no bands) → 0 dB at all frequencies
        let eq = ParametricEq::new(vec![]);
        let freqs = vec![100.0f32, 1000.0, 10000.0];
        let resp = eq.frequency_response(&freqs, 44100);
        for &r in &resp {
            assert!((r - 0.0).abs() < 0.1, "empty eq should be 0 dB, got {r}");
        }
    }

    #[test]
    fn test_parametriceq_lowpass_attenuates_high_freqs() {
        let lp = BiquadFilter::lowpass(1000.0, 0.707, 44100);
        let eq = ParametricEq::new(vec![lp]);
        let freqs = vec![100.0f32, 10000.0];
        let resp = eq.frequency_response(&freqs, 44100);
        // 10kHz should be significantly attenuated compared to 100Hz
        assert!(
            resp[0] > resp[1] + 10.0,
            "lowpass should attenuate 10kHz: 100Hz={:.1} 10kHz={:.1}",
            resp[0],
            resp[1]
        );
    }

    #[test]
    fn test_parametriceq_phase_response_returns_correct_length() {
        let lp = BiquadFilter::lowpass(1000.0, 0.707, 44100);
        let eq = ParametricEq::new(vec![lp]);
        let freqs: Vec<f32> = (1..=10).map(|i| i as f32 * 1000.0).collect();
        let phase = eq.phase_response(&freqs, 44100);
        assert_eq!(phase.len(), freqs.len());
    }

    #[test]
    fn test_parametriceq_group_delay_positive_lowpass() {
        let lp = BiquadFilter::lowpass(1000.0, 0.707, 44100);
        let eq = ParametricEq::new(vec![lp]);
        let freqs = vec![100.0f32, 500.0, 1000.0];
        let gd = eq.group_delay(&freqs, 44100);
        // Group delay should be positive for a causal filter at low frequencies
        assert!(
            gd.iter().all(|&d| d >= 0.0),
            "group delay should be non-negative, got: {gd:?}"
        );
    }

    // ─── M9 graphic_eq tests ────────────────────────────────────────────────

    fn make_mono_sine(freq: f32, sr: u32, duration: f32) -> AudioBuffer<f32> {
        let n = (sr as f32 * duration) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn rms_of(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn test_graphic_eq_flat_response() {
        let sr = 48_000u32;
        // All-zero gains: output should equal input within 1e-3 per sample
        let gains = [0.0f32; 10];
        let eq = ParametricEq::graphic_eq(&gains, 10, sr).expect("graphic_eq should succeed");
        let buf = make_mono_sine(1000.0, sr, 0.1);
        let out = eq.process(&buf);
        assert_eq!(out.samples.len(), buf.samples.len());
        // Skip initial transient
        for i in 512..buf.samples.len() {
            let diff = (out.samples[i] - buf.samples[i]).abs();
            assert!(
                diff < 1e-3,
                "flat graphic EQ sample {i}: expected {}, got {} (diff={diff})",
                buf.samples[i],
                out.samples[i]
            );
        }
    }

    #[test]
    fn test_graphic_eq_boost_at_1khz() {
        let sr = 48_000u32;
        // 10-band: boost +6dB at index 5 (1000 Hz)
        let mut gains = [0.0f32; 10];
        gains[5] = 6.0; // 1kHz band
        let eq = ParametricEq::graphic_eq(&gains, 10, sr).expect("graphic_eq should succeed");
        let buf = make_mono_sine(1000.0, sr, 0.5);
        let out = eq.process(&buf);
        let skip = (sr as f32 * 0.05) as usize;
        let in_rms = rms_of(&buf.samples[skip..]);
        let out_rms = rms_of(&out.samples[skip..]);
        assert!(
            out_rms > in_rms,
            "+6dB boost at 1kHz should increase RMS: in={in_rms:.4} out={out_rms:.4}"
        );
    }

    #[test]
    fn test_graphic_eq_wrong_band_count_errors() {
        let gains = [0.0f32; 5]; // wrong: 5 gains for 10 bands
        let result = ParametricEq::graphic_eq(&gains, 10, 48_000);
        assert!(
            result.is_err(),
            "mismatched gains/n_bands should return error"
        );
    }

    #[test]
    fn test_graphic_eq_invalid_n_bands_errors() {
        let gains = [0.0f32; 20];
        let result = ParametricEq::graphic_eq(&gains, 20, 48_000);
        assert!(result.is_err(), "n_bands=20 should return error");
    }

    #[test]
    fn test_graphic_eq_31_band() {
        let sr = 48_000u32;
        let gains = [0.0f32; 31];
        let eq =
            ParametricEq::graphic_eq(&gains, 31, sr).expect("31-band graphic_eq should succeed");
        assert_eq!(eq.bands.len(), 31);
    }

    #[test]
    fn test_biquad_process_buffer_matches_process() {
        // process_buffer on a mono slice must produce identical output to process()
        // on the same data wrapped in a Mono AudioBuffer, because both initialise
        // z1/z2 to zero and apply the same Direct Form II Transposed recurrence.
        use oxiaudio_core::SampleFormat;
        let filter = BiquadFilter::lowpass(1000.0, 0.707, 48000);

        let samples: Vec<f32> = (0..1000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();

        // process() path — via AudioBuffer
        let buf_in = AudioBuffer {
            samples: samples.clone(),
            sample_rate: 48000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let expected = filter.process(&buf_in);

        // process_buffer() path — direct slice
        let mut actual = samples;
        filter.process_buffer(&mut actual);

        for (i, (&a, &b)) in actual.iter().zip(expected.samples.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "process_buffer diverges from process at sample {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn biquad_lowpass_1khz_attenuates_10khz_by_40db() {
        // 1 kHz lowpass at 48 kHz, Q=0.707: passband (1 kHz) vs stopband (10 kHz)
        // ratio must exceed 40 dB. Use 1s buffers to get sufficient steady-state data.
        let sr = 48_000u32;
        let n = sr as usize; // 1 second for good steady-state measurement
        let filter = BiquadFilter::lowpass(1_000.0, 0.707, sr);

        // Generate 1 kHz sine (passband)
        let passband: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1_000.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf_pass = AudioBuffer {
            samples: passband,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let out_pass = filter.process(&buf_pass);

        // Generate 10 kHz sine (stopband)
        let stopband: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 10_000.0 * i as f32 / sr as f32).sin())
            .collect();
        let buf_stop = AudioBuffer {
            samples: stopband,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let out_stop = filter.process(&buf_stop);

        // Skip settling transient (first 5000 samples ≈ 100ms)
        let skip = 5000usize;
        // Passband gain at 1 kHz with Q=0.707 Butterworth is ~0 dB (unity), so
        // we can normalize by the input RMS rather than the filtered 1 kHz RMS.
        // Input RMS for a unit sine = 1/sqrt(2) ≈ 0.707.
        let input_rms = (0.5f32).sqrt(); // RMS of unit sine
        let rms_stop = {
            let sq: f32 = out_stop.samples[skip..].iter().map(|&s| s * s).sum();
            (sq / (out_stop.samples.len() - skip) as f32).sqrt()
        };
        // Measure passband gain at 1 kHz (should be ~0 dB, i.e. ~0.707 RMS)
        let rms_pass = {
            let sq: f32 = out_pass.samples[skip..].iter().map(|&s| s * s).sum();
            (sq / (out_pass.samples.len() - skip) as f32).sqrt()
        };
        // Use the passband RMS as reference; it should be close to input_rms
        let _ = input_rms; // kept for documentation
        let attenuation_db = 20.0 * (rms_stop / rms_pass.max(1e-10)).log10();
        assert!(
            attenuation_db < -39.0,
            "1kHz LP should attenuate 10kHz by >39dB relative to 1kHz passband, got {attenuation_db:.2}dB"
        );
    }

    #[test]
    fn bandpass_passes_center() {
        let sr = 48_000u32;
        let n = sr as usize;
        let fc = 2_000.0f32;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * fc * i as f32 / sr as f32).sin())
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let filter = BiquadFilter::bandpass(fc, 1.0, sr);
        let out = filter.process(&buf);
        let rms_in: f32 =
            (buf.samples[500..].iter().map(|s| s * s).sum::<f32>() / (n - 500) as f32).sqrt();
        let rms_out: f32 =
            (out.samples[500..].iter().map(|s| s * s).sum::<f32>() / (n - 500) as f32).sqrt();
        let ratio_db = 20.0 * (rms_out / rms_in.max(1e-10)).log10();
        assert!(
            ratio_db > -12.0,
            "bandpass should pass center freq with <12dB loss, got {ratio_db:.1}dB"
        );
    }

    // ─── process_multichannel tests ──────────────────────────────────────────

    /// Helper: create an interleaved stereo buffer containing a high-frequency sine on
    /// both channels.
    fn stereo_sine_buf(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
        let frames = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..frames)
            .flat_map(|i| {
                let t = i as f32 / sample_rate as f32;
                let s = (2.0 * PI * freq_hz * t).sin();
                [s, s]
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_biquad_multichannel_stereo() {
        // A 1 kHz lowpass applied to a 10 kHz stereo sine via process_multichannel
        // must attenuate both channels.
        let sr = 48_000u32;
        let buf = stereo_sine_buf(10_000.0, sr, 0.5);
        let filter = BiquadFilter::lowpass(1_000.0, 0.707, sr);
        let out = filter.process_multichannel(&buf);

        assert_eq!(out.samples.len(), buf.samples.len());
        assert_eq!(out.channels, buf.channels);

        // Skip settling transient (first 200 stereo frames = 400 samples)
        let skip = 400usize;
        // Left channel (even indices)
        let rms_in_left: f32 = {
            let sq: f32 = buf.samples[skip..].iter().step_by(2).map(|&s| s * s).sum();
            let n = buf.samples[skip..].len() / 2;
            (sq / n as f32).sqrt()
        };
        let rms_out_left: f32 = {
            let sq: f32 = out.samples[skip..].iter().step_by(2).map(|&s| s * s).sum();
            let n = out.samples[skip..].len() / 2;
            (sq / n as f32).sqrt()
        };
        // Right channel (odd indices)
        let rms_out_right: f32 = {
            let sq: f32 = out.samples[(skip + 1)..]
                .iter()
                .step_by(2)
                .map(|&s| s * s)
                .sum();
            let n = out.samples[(skip + 1)..].len() / 2;
            (sq / n as f32).sqrt()
        };

        let atten_left_db = 20.0 * (rms_out_left / rms_in_left.max(1e-10)).log10();
        let atten_right_db = 20.0 * (rms_out_right / rms_in_left.max(1e-10)).log10();

        assert!(
            atten_left_db < -20.0,
            "multichannel: left channel should be attenuated >20 dB, got {atten_left_db:.1} dB"
        );
        assert!(
            atten_right_db < -20.0,
            "multichannel: right channel should be attenuated >20 dB, got {atten_right_db:.1} dB"
        );
    }

    #[test]
    fn test_biquad_multichannel_equals_mono() {
        // On a mono buffer process_multichannel must produce sample-for-sample identical
        // output to process().
        let sr = 48_000u32;
        let buf = sine_buf(440.0, sr, 0.1);
        let filter = BiquadFilter::lowpass(1_000.0, 0.707, sr);

        let out_process = filter.process(&buf);
        let out_multi = filter.process_multichannel(&buf);

        assert_eq!(out_process.samples.len(), out_multi.samples.len());
        for (i, (&a, &b)) in out_process
            .samples
            .iter()
            .zip(out_multi.samples.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-7,
                "mono: sample {i} diverges: process={a} multichannel={b}"
            );
        }
    }

    #[test]
    fn test_biquad_multichannel_equals_process_stereo() {
        // On a stereo buffer both methods must be sample-for-sample identical,
        // because both maintain per-channel state independently.
        let sr = 48_000u32;
        let buf = stereo_sine_buf(440.0, sr, 0.1);
        let filter = BiquadFilter::lowpass(1_000.0, 0.707, sr);

        let out_process = filter.process(&buf);
        let out_multi = filter.process_multichannel(&buf);

        assert_eq!(out_process.samples.len(), out_multi.samples.len());
        for (i, (&a, &b)) in out_process
            .samples
            .iter()
            .zip(out_multi.samples.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "stereo: sample {i} diverges: process={a} multichannel={b}"
            );
        }
    }
}
