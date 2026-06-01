//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use oxiaudio_core::{AudioBuffer, OxiAudioError};
use std::f32::consts::PI;

use super::functions::{ALLPASS_TUNINGS, COMB_TUNINGS};

/// A channel vocoder that transfers the spectral envelope of a modulator signal
/// onto a carrier signal.
///
/// The modulator (e.g. voice) determines *what* spectral shape is imposed;
/// the carrier (e.g. synthesizer tone) provides the *excitation* signal.
/// The output sounds like the carrier "talking" with the voice's articulation.
///
/// ## Algorithm
/// 1. Compute STFT of both modulator and carrier (Hann window).
/// 2. Per frame, per bin: replace the carrier magnitude with the modulator magnitude.
///    `out[k] = carrier[k] / (|carrier[k]| + ε) * |modulator[k]|`
/// 3. Reconstruct via ISTFT (overlap-add).
#[derive(Debug, Clone)]
pub struct ChannelVocoder {
    /// FFT window size in samples (e.g. 1024).
    pub n_fft: usize,
    /// Hop size between consecutive frames in samples (e.g. 256).
    pub hop_size: usize,
    /// Small constant added to the carrier magnitude to avoid division by zero.
    pub epsilon: f32,
}
impl ChannelVocoder {
    /// Create a new `ChannelVocoder` with the given FFT and hop sizes.
    ///
    /// `epsilon` defaults to `1e-8`.
    pub fn new(n_fft: usize, hop_size: usize) -> Self {
        Self {
            n_fft,
            hop_size,
            epsilon: 1e-8,
        }
    }
    /// Apply the vocoder: impose the modulator's spectral envelope on the carrier.
    ///
    /// Both `modulator` and `carrier` must have the same `sample_rate`.
    /// The output length in samples matches the mono length of the shorter input.
    ///
    /// # Errors
    ///
    /// Returns `OxiAudioError::UnsupportedFormat` if the two buffers have different
    /// sample rates, or if the internal STFT/ISTFT fails.
    #[must_use = "returns the vocoded output buffer"]
    pub fn process(
        &self,
        modulator: &oxiaudio_core::AudioBuffer<f32>,
        carrier: &oxiaudio_core::AudioBuffer<f32>,
    ) -> Result<oxiaudio_core::AudioBuffer<f32>, oxiaudio_core::OxiAudioError> {
        use crate::mix_to_mono;
        use crate::spectral::{stft, StftOutput, WindowFn};
        if modulator.sample_rate != carrier.sample_rate {
            return Err(oxiaudio_core::OxiAudioError::UnsupportedFormat(format!(
                "ChannelVocoder: modulator sample_rate ({}) != carrier sample_rate ({})",
                modulator.sample_rate, carrier.sample_rate
            )));
        }
        let mod_stft = stft(modulator, self.n_fft, self.hop_size, WindowFn::Hann)?;
        let car_stft = stft(carrier, self.n_fft, self.hop_size, WindowFn::Hann)?;
        if mod_stft.frames.is_empty() || car_stft.frames.is_empty() {
            let mono_mod = mix_to_mono(modulator);
            let mono_car = mix_to_mono(carrier);
            let out_len = mono_mod.samples.len().min(mono_car.samples.len());
            return Ok(oxiaudio_core::AudioBuffer {
                samples: vec![0.0f32; out_len],
                sample_rate: carrier.sample_rate,
                channels: oxiaudio_core::ChannelLayout::Mono,
                format: oxiaudio_core::SampleFormat::F32,
            });
        }
        let n_frames = mod_stft.frames.len().min(car_stft.frames.len());
        let eps = self.epsilon;
        let output_frames: Vec<Vec<crate::spectral::Complex<f32>>> = (0..n_frames)
            .map(|i| {
                let mod_frame = &mod_stft.frames[i];
                let car_frame = &car_stft.frames[i];
                let n_bins = mod_frame.len().min(car_frame.len());
                (0..n_bins)
                    .map(|k| {
                        let mod_mag = (mod_frame[k].re * mod_frame[k].re
                            + mod_frame[k].im * mod_frame[k].im)
                            .sqrt();
                        let car_mag = (car_frame[k].re * car_frame[k].re
                            + car_frame[k].im * car_frame[k].im)
                            .sqrt();
                        let scale = mod_mag / (car_mag + eps);
                        crate::spectral::Complex::new(
                            car_frame[k].re * scale,
                            car_frame[k].im * scale,
                        )
                    })
                    .collect()
            })
            .collect();
        let out_stft = StftOutput {
            frames: output_frames,
            sample_rate: carrier.sample_rate,
            hop_size: self.hop_size,
            window: WindowFn::Hann,
        };
        let mono_mod = mix_to_mono(modulator);
        let mono_car = mix_to_mono(carrier);
        let original_len = mono_mod.samples.len().min(mono_car.samples.len());
        crate::spectral::istft(&out_stft, original_len)
    }
}
/// Early reflections reverb using the image-source method.
///
/// Models first-order reflections from a rectangular room (6 walls).
/// Each reflection arrives as an attenuated, delayed copy of the direct signal.
///
/// # Parameters
/// - `room_l`, `room_w`, `room_h` — room dimensions in meters
/// - `src_x`, `src_y`, `src_z` — source position (fraction of room dimensions, 0.0–1.0)
/// - `mic_x`, `mic_y`, `mic_z` — microphone position (fraction of room dimensions, 0.0–1.0)
/// - `reflection_coeff` — wall absorption coefficient (0.0 = perfect absorption, 1.0 = perfect reflection)
/// - `dry_wet` — 0.0 = dry only, 1.0 = reflections only
#[derive(Debug, Clone)]
pub struct EarlyReflections {
    /// Room length in meters.
    pub room_l: f32,
    /// Room width in meters.
    pub room_w: f32,
    /// Room height in meters.
    pub room_h: f32,
    /// Source x position as a fraction of room_l (0.0–1.0).
    pub src_x: f32,
    /// Source y position as a fraction of room_w (0.0–1.0).
    pub src_y: f32,
    /// Source z position as a fraction of room_h (0.0–1.0).
    pub src_z: f32,
    /// Microphone x position as a fraction of room_l (0.0–1.0).
    pub mic_x: f32,
    /// Microphone y position as a fraction of room_w (0.0–1.0).
    pub mic_y: f32,
    /// Microphone z position as a fraction of room_h (0.0–1.0).
    pub mic_z: f32,
    /// Wall absorption coefficient in [0.0, 1.0].
    pub reflection_coeff: f32,
    /// Wet/dry mix: 0.0 = dry only, 1.0 = reflections only.
    pub dry_wet: f32,
}
impl EarlyReflections {
    /// Create with default parameters: 10x8x3m room, centered source, offset mic, coeff=0.7, dry_wet=0.5.
    pub fn new() -> Self {
        Self {
            room_l: 10.0,
            room_w: 8.0,
            room_h: 3.0,
            src_x: 0.5,
            src_y: 0.5,
            src_z: 0.5,
            mic_x: 0.75,
            mic_y: 0.5,
            mic_z: 0.5,
            reflection_coeff: 0.7,
            dry_wet: 0.5,
        }
    }
    /// Precompute the 6 first-order image source (delay_samples, gain) pairs
    /// for the current room/source/mic configuration.
    fn compute_reflections(&self, sample_rate: u32) -> [(usize, f32); 6] {
        const SPEED_OF_SOUND: f32 = 343.0;
        let src_xm = self.src_x * self.room_l;
        let src_ym = self.src_y * self.room_w;
        let src_zm = self.src_z * self.room_h;
        let mic_xm = self.mic_x * self.room_l;
        let mic_ym = self.mic_y * self.room_w;
        let mic_zm = self.mic_z * self.room_h;
        let images: [(f32, f32, f32); 6] = [
            (-src_xm, src_ym, src_zm),
            (2.0 * self.room_l - src_xm, src_ym, src_zm),
            (src_xm, -src_ym, src_zm),
            (src_xm, 2.0 * self.room_w - src_ym, src_zm),
            (src_xm, src_ym, -src_zm),
            (src_xm, src_ym, 2.0 * self.room_h - src_zm),
        ];
        let sr_f = sample_rate as f32;
        let mut result = [(0usize, 0.0f32); 6];
        for (i, &(im_x, im_y, im_z)) in images.iter().enumerate() {
            let dx = im_x - mic_xm;
            let dy = im_y - mic_ym;
            let dz = im_z - mic_zm;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.001);
            let delay_samples = (dist / SPEED_OF_SOUND * sr_f).round() as usize;
            let gain = self.reflection_coeff / dist;
            result[i] = (delay_samples, gain);
        }
        result
    }
    /// Process buffer applying early reflections.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let ch = buf.channels.channel_count().max(1);
        let frames = buf.samples.len() / ch;
        let reflections = self.compute_reflections(buf.sample_rate);
        let mut out = Vec::with_capacity(buf.samples.len());
        for i in 0..frames {
            for c in 0..ch {
                let x = buf.samples[i * ch + c];
                let mut wet = 0.0f32;
                for &(delay_samp, gain) in &reflections {
                    if i >= delay_samp {
                        let src_idx = (i - delay_samp) * ch + c;
                        wet += gain * buf.samples[src_idx];
                    }
                }
                let mixed = self.dry_wet * wet + (1.0 - self.dry_wet) * x;
                out.push(mixed.clamp(-1.0, 1.0));
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
/// A chorus effect: multiple modulated delay lines blended with the dry signal.
#[derive(Debug, Clone)]
pub struct Chorus {
    /// LFO rate in Hz.
    pub rate_hz: f32,
    /// Modulation depth in milliseconds.
    pub depth_ms: f32,
    /// Number of chorus voices (clamped to 2-4).
    pub voices: usize,
    /// Wet/dry mix: 0.0 = fully dry, 1.0 = fully wet.
    pub wet_dry: f32,
}
impl Chorus {
    /// Create a new `Chorus` with 2 voices and 50% wet/dry mix.
    pub fn new(rate_hz: f32, depth_ms: f32) -> Self {
        Self {
            rate_hz,
            depth_ms,
            voices: 2,
            wet_dry: 0.5,
        }
    }
    /// Apply the chorus effect to `buf` and return the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let sr = buf.sample_rate as f32;
        let ch = buf.channels.channel_count();
        let frames = buf.samples.len() / ch.max(1);
        let voices = self.voices.clamp(2, 4);
        let base_delay_ms = (self.depth_ms * 2.0).max(1.0);
        let max_delay_samp = ((base_delay_ms + self.depth_ms) * sr / 1000.0) as usize + 2;
        let buf_len = max_delay_samp + 1;
        let mut ring = vec![vec![0.0f32; buf_len]; ch];
        let mut write_pos = 0usize;
        let mut out = vec![0.0f32; buf.samples.len()];
        for frame in 0..frames {
            let t = frame as f32 / sr;
            let mut voice_sum = vec![0.0f32; ch];
            for v in 0..voices {
                let phase = 2.0 * PI * v as f32 / voices as f32;
                let lfo = (2.0 * PI * self.rate_hz * t + phase).sin();
                let d_ms = base_delay_ms + self.depth_ms * lfo;
                let d_f = (d_ms * sr / 1000.0).max(0.0);
                let d_int = d_f as usize;
                let frac = d_f - d_int as f32;
                let r0 = if write_pos >= d_int {
                    write_pos - d_int
                } else {
                    buf_len - (d_int - write_pos)
                };
                let r1 = if r0 + 1 < buf_len { r0 + 1 } else { 0 };
                for c in 0..ch {
                    voice_sum[c] += ring[c][r0] + frac * (ring[c][r1] - ring[c][r0]);
                }
            }
            for c in 0..ch {
                let x = buf.samples[frame * ch + c];
                ring[c][write_pos] = x;
                out[frame * ch + c] =
                    (1.0 - self.wet_dry) * x + self.wet_dry * voice_sum[c] / voices as f32;
            }
            write_pos = (write_pos + 1) % buf_len;
        }
        AudioBuffer {
            samples: out,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }
}
/// A phaser effect using a cascade of first-order allpass filters swept by an LFO.
///
/// The LFO modulates the allpass coefficient, sweeping the phase response and
/// creating the characteristic notch-comb pattern that shifts over time.
#[derive(Debug, Clone)]
pub struct Phaser {
    /// LFO rate in Hz (0.1–2.0 Hz typical).
    pub rate_hz: f32,
    /// Modulation depth in [0.0, 1.0].
    pub depth: f32,
    /// Feedback coefficient in [0.0, 0.99].
    pub feedback: f32,
    /// Number of allpass stages (4–12, even numbers recommended).
    pub stages: usize,
    /// Wet/dry mix in [0.0, 1.0].
    pub wet_dry: f32,
    /// Sample rate for this instance.
    pub sample_rate: u32,
}
impl Phaser {
    /// Create a new `Phaser` with defaults: rate=0.5 Hz, depth=1.0,
    /// feedback=0.7, stages=4, wet_dry=0.5.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            rate_hz: 0.5,
            depth: 1.0,
            feedback: 0.7,
            stages: 4,
            wet_dry: 0.5,
            sample_rate,
        }
    }
    /// Apply the phaser to `buf` and return the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let sr = buf.sample_rate as f32;
        let ch = buf.channels.channel_count();
        let frames = buf.samples.len() / ch.max(1);
        let stages = self.stages.clamp(2, 12);
        let min_freq = 100.0f32;
        let max_freq = 2000.0f32;
        let fb = self.feedback.clamp(0.0, 0.99);
        let mut z1 = vec![vec![0.0f32; stages]; ch];
        let mut feedback_state = vec![0.0f32; ch];
        let mut out = vec![0.0f32; buf.samples.len()];
        for frame in 0..frames {
            let t = frame as f32 / sr;
            let lfo = (2.0 * PI * self.rate_hz * t).sin();
            let sweep = min_freq + (max_freq - min_freq) * 0.5 * (1.0 + lfo) * self.depth;
            let tan_val = (PI * sweep / sr).tan();
            let b = (tan_val - 1.0) / (tan_val + 1.0);
            for c in 0..ch {
                let x = buf.samples[frame * ch + c];
                let input = x + fb * feedback_state[c];
                let phased = (0..stages).fold(input, |acc, stage| {
                    let y = -acc + (1.0 + b) * z1[c][stage];
                    z1[c][stage] = acc + b * z1[c][stage];
                    y
                });
                feedback_state[c] = phased;
                let dry = 1.0 - self.wet_dry;
                out[frame * ch + c] = dry * x + self.wet_dry * phased;
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
/// Schroeder allpass filter for diffusion.
#[derive(Debug, Clone)]
struct AllpassFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
}
impl AllpassFilter {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0f32; size],
            write_pos: 0,
            feedback: 0.5,
        }
    }
    fn process(&mut self, input: f32) -> f32 {
        let buf_out = self.buffer[self.write_pos];
        let output = -input + buf_out;
        self.buffer[self.write_pos] = input + buf_out * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }
}
/// A simple feedback delay line with wet/dry control.
#[derive(Debug, Clone)]
pub struct DelayLine {
    /// Delay time in milliseconds.
    pub delay_ms: f32,
    /// Feedback coefficient in [0, 0.999].
    pub feedback: f32,
    /// Wet/dry mix: 0.0 = fully dry, 1.0 = fully wet.
    pub wet_dry: f32,
}
impl DelayLine {
    /// Create a new `DelayLine`.
    ///
    /// `feedback` is clamped to [0, 0.999] and `wet_dry` to [0, 1].
    pub fn new(delay_ms: f32, feedback: f32, wet_dry: f32) -> Self {
        Self {
            delay_ms,
            feedback: feedback.clamp(0.0, 0.999),
            wet_dry: wet_dry.clamp(0.0, 1.0),
        }
    }
    /// Apply the delay to `buf` and return the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let sr = buf.sample_rate as f32;
        let ch = buf.channels.channel_count();
        let delay_samples = ((self.delay_ms * sr / 1000.0) as usize).max(1);
        let buf_len = delay_samples + 1;
        let frames = buf.samples.len() / ch.max(1);
        let mut ring = vec![vec![0.0f32; buf_len]; ch];
        let mut write_pos = 0usize;
        let mut out = vec![0.0f32; buf.samples.len()];
        for frame in 0..frames {
            let read_pos = if write_pos >= delay_samples {
                write_pos - delay_samples
            } else {
                buf_len - (delay_samples - write_pos)
            };
            for c in 0..ch {
                let x = buf.samples[frame * ch + c];
                let delayed = ring[c][read_pos];
                ring[c][write_pos] = x + self.feedback * delayed;
                out[frame * ch + c] = (1.0 - self.wet_dry) * x + self.wet_dry * delayed;
            }
            write_pos = (write_pos + 1) % buf_len;
        }
        AudioBuffer {
            samples: out,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }
}
/// Feedback comb filter with a first-order lowpass in the feedback path.
#[derive(Debug, Clone)]
struct CombFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
    damp: f32,
    filterstore: f32,
}
impl CombFilter {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0f32; size],
            write_pos: 0,
            feedback: 0.5,
            damp: 0.5,
            filterstore: 0.0,
        }
    }
    fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.write_pos];
        self.filterstore = output * (1.0 - self.damp) + self.filterstore * self.damp;
        self.buffer[self.write_pos] = input + self.filterstore * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }
}
/// Freeverb algorithmic reverb (Jezar at Dreampoint).
///
/// 8 parallel comb filters with lowpass feedback, followed by 4 series allpass
/// filters for diffusion. Delay lengths are tuned for 44100 Hz and scaled to the
/// actual sample rate.
#[derive(Debug, Clone)]
pub struct Freeverb {
    /// Room size in [0.0, 1.0]: controls comb-filter feedback (larger = longer reverb).
    pub room_size: f32,
    /// Damping in [0.0, 1.0]: higher values damp high frequencies faster.
    pub damping: f32,
    /// Wet level in [0.0, 1.0].
    pub wet: f32,
    /// Dry level in [0.0, 1.0].
    pub dry: f32,
    /// Stereo width in [0.0, 1.0] (currently applies to stereo processing).
    pub width: f32,
    /// Sample rate this instance was created for.
    pub sample_rate: u32,
}
impl Freeverb {
    /// Create a new `Freeverb` with defaults: room_size=0.5, damping=0.5,
    /// wet=0.33, dry=1.0, width=1.0.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            room_size: 0.5,
            damping: 0.5,
            wet: 0.33,
            dry: 1.0,
            width: 1.0,
            sample_rate,
        }
    }
    /// Scale a 44100 Hz delay length to the current sample rate.
    fn scale_delay(&self, delay_44100: usize) -> usize {
        let scaled = delay_44100 as f64 * self.sample_rate as f64 / 44_100.0;
        (scaled as usize).max(1)
    }
    /// Build fresh comb filter state using current parameters.
    fn build_combs(&self) -> Vec<CombFilter> {
        let feedback = (0.7 * self.room_size + 0.5).min(0.98);
        let damp = self.damping;
        COMB_TUNINGS
            .iter()
            .map(|&tuning| {
                let mut c = CombFilter::new(self.scale_delay(tuning));
                c.feedback = feedback;
                c.damp = damp;
                c
            })
            .collect()
    }
    /// Build fresh allpass filter state.
    fn build_allpasses(&self) -> Vec<AllpassFilter> {
        ALLPASS_TUNINGS
            .iter()
            .map(|&tuning| AllpassFilter::new(self.scale_delay(tuning)))
            .collect()
    }
    /// Apply the reverb to `buf` and return the wet+dry mix.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let ch = buf.channels.channel_count();
        let frames = buf.samples.len() / ch.max(1);
        let mut out = vec![0.0f32; buf.samples.len()];
        for c in 0..ch {
            let mut combs = self.build_combs();
            let mut allpasses = self.build_allpasses();
            for frame in 0..frames {
                let x = buf.samples[frame * ch + c];
                let reverb_out: f32 = combs.iter_mut().map(|comb| comb.process(x)).sum();
                let diffused = allpasses
                    .iter_mut()
                    .fold(reverb_out, |acc, ap| ap.process(acc));
                out[frame * ch + c] = self.wet * diffused + self.dry * x;
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
/// A flanger effect using a modulated delay line with feedback.
///
/// The LFO sweeps the delay between `base_delay` and `base_delay + depth`,
/// creating the characteristic comb-filtering sweep effect.
#[derive(Debug, Clone)]
pub struct Flanger {
    /// LFO modulation rate in Hz (0.1–2.0 Hz typical).
    pub rate_hz: f32,
    /// Modulation depth in milliseconds (0.5–5.0 ms typical).
    pub depth_ms: f32,
    /// Feedback coefficient in [−0.99, 0.99]. Negative values invert feedback.
    pub feedback: f32,
    /// Wet/dry mix in [0.0, 1.0].
    pub wet_dry: f32,
    /// If `true`, the flanged signal is phase-inverted before mixing.
    pub inverted: bool,
    /// Sample rate for this instance.
    pub sample_rate: u32,
}
impl Flanger {
    /// Create a new `Flanger` with defaults: rate=0.3 Hz, depth=2.5 ms,
    /// feedback=0.5, wet_dry=0.5, inverted=false.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            rate_hz: 0.3,
            depth_ms: 2.5,
            feedback: 0.5,
            wet_dry: 0.5,
            inverted: false,
            sample_rate,
        }
    }
    /// Apply the flanger to `buf` and return the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let sr = buf.sample_rate as f32;
        let ch = buf.channels.channel_count();
        let frames = buf.samples.len() / ch.max(1);
        let base_delay_samps = self.depth_ms * sr / 1000.0;
        let max_delay_samps = (base_delay_samps * 2.0).ceil() as usize + 2;
        let buf_len = max_delay_samps + 1;
        let sign = if self.inverted { -1.0f32 } else { 1.0 };
        let fb = self.feedback.clamp(-0.99, 0.99);
        let mut ring = vec![vec![0.0f32; buf_len]; ch];
        let mut write_pos = 0usize;
        let mut out = vec![0.0f32; buf.samples.len()];
        for frame in 0..frames {
            let t = frame as f32 / sr;
            let lfo = (2.0 * PI * self.rate_hz * t).sin();
            let delay_f = base_delay_samps + base_delay_samps * lfo;
            let delay_f = delay_f.max(0.0);
            let d_int = delay_f as usize;
            let frac = delay_f - d_int as f32;
            let r0 = if write_pos >= d_int {
                write_pos - d_int
            } else {
                buf_len - (d_int - write_pos) % buf_len
            };
            let r1 = if r0 + 1 < buf_len { r0 + 1 } else { 0 };
            for c in 0..ch {
                let x = buf.samples[frame * ch + c];
                let delayed = ring[c][r0] + frac * (ring[c][r1] - ring[c][r0]);
                ring[c][write_pos] = x + fb * delayed;
                out[frame * ch + c] = (1.0 - self.wet_dry) * x + self.wet_dry * sign * delayed;
            }
            write_pos = (write_pos + 1) % buf_len;
        }
        AudioBuffer {
            samples: out,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }
}
/// A vibrato effect: pitch modulation via a fractional-delay interpolated delay line.
#[derive(Debug, Clone)]
pub struct Vibrato {
    /// LFO rate in Hz.
    pub rate_hz: f32,
    /// Modulation depth in cents.
    pub depth_cents: f32,
}
impl Vibrato {
    /// Create a new `Vibrato`.
    pub fn new(rate_hz: f32, depth_cents: f32) -> Self {
        Self {
            rate_hz,
            depth_cents,
        }
    }
    /// Apply the vibrato effect to `buf` and return the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let sr = buf.sample_rate as f32;
        let ch = buf.channels.channel_count();
        let frames = buf.samples.len() / ch.max(1);
        let semitones = self.depth_cents / 100.0;
        let max_delay = ((2.0f32.powf(semitones / 12.0) - 1.0) * sr
            / (2.0 * PI * self.rate_hz.max(0.01)))
        .max(1.0);
        let buf_len = (max_delay as usize + 2).max(4);
        let mut ring = vec![vec![0.0f32; buf_len]; ch];
        let mut write_pos = 0usize;
        let mut out = vec![0.0f32; buf.samples.len()];
        for frame in 0..frames {
            let t = frame as f32 / sr;
            let lfo = (2.0 * PI * self.rate_hz * t).sin();
            let d_f = max_delay * 0.5 * (1.0 + lfo);
            let d_int = d_f as usize;
            let frac = d_f - d_int as f32;
            let r0 = if write_pos >= d_int {
                write_pos - d_int
            } else {
                buf_len - (d_int - write_pos)
            };
            let r1 = if r0 + 1 < buf_len { r0 + 1 } else { 0 };
            for c in 0..ch {
                ring[c][write_pos] = buf.samples[frame * ch + c];
                out[frame * ch + c] = ring[c][r0] + frac * (ring[c][r1] - ring[c][r0]);
            }
            write_pos = (write_pos + 1) % buf_len;
        }
        AudioBuffer {
            samples: out,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }
}
/// A tremolo effect: amplitude modulation via a sine LFO.
#[derive(Debug, Clone)]
pub struct Tremolo {
    /// LFO rate in Hz.
    pub rate_hz: f32,
    /// Modulation depth in [0, 1]. At depth=1.0, amplitude reaches zero.
    pub depth: f32,
}
impl Tremolo {
    /// Create a new `Tremolo`. `depth` is clamped to [0, 1].
    pub fn new(rate_hz: f32, depth: f32) -> Self {
        Self {
            rate_hz,
            depth: depth.clamp(0.0, 1.0),
        }
    }
    /// Apply the tremolo effect to `buf` and return the processed buffer.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let sr = buf.sample_rate as f32;
        let ch = buf.channels.channel_count();
        let frames = buf.samples.len() / ch.max(1);
        let mut out = buf.samples.clone();
        for frame in 0..frames {
            let t = frame as f32 / sr;
            let lfo = (2.0 * PI * self.rate_hz * t).sin();
            let gain = 1.0 - self.depth * 0.5 * (1.0 - lfo);
            for c in 0..ch {
                out[frame * ch + c] *= gain;
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
/// Convolution reverb using FFT-based linear convolution (overlap-save equivalent).
///
/// Convolves each channel of the input audio with the provided impulse response
/// to produce a physically accurate room reverberation effect.
///
/// The wet output includes the full reverb tail (length = input + IR - 1 frames).
///
/// # Example
/// ```ignore
/// // Load your IR samples via oxiaudio::decode_file(), then:
/// let reverb = ConvolutionReverb::new(ir_samples)
///     .with_wet(0.3)
///     .with_dry(0.7);
/// let result = reverb.apply(&audio_buffer)?;
/// ```
#[derive(Debug, Clone)]
pub struct ConvolutionReverb {
    /// The impulse response (mono). Users should decode an IR file and pass raw f32 samples.
    pub impulse_response: Vec<f32>,
    /// Wet level in [0.0, 1.0]: amount of convolved signal mixed in.
    pub wet: f32,
    /// Dry level in [0.0, 1.0]: amount of original signal mixed in.
    pub dry: f32,
    /// Partition size hint for overlap-save processing (informational; default 2048).
    /// The actual FFT size is determined by `oxifft::conv::convolve`.
    pub partition_size: usize,
}
impl ConvolutionReverb {
    /// Create a new `ConvolutionReverb` with the given impulse response and default wet=0.5, dry=0.5.
    pub fn new(impulse_response: Vec<f32>) -> Self {
        Self {
            impulse_response,
            wet: 0.5,
            dry: 0.5,
            partition_size: 2048,
        }
    }
    /// Set the wet (convolved) level, clamped to [0.0, 1.0].
    pub fn with_wet(mut self, wet: f32) -> Self {
        self.wet = wet.clamp(0.0, 1.0);
        self
    }
    /// Set the dry (original) level, clamped to [0.0, 1.0].
    pub fn with_dry(mut self, dry: f32) -> Self {
        self.dry = dry.clamp(0.0, 1.0);
        self
    }
    /// Set the partition size hint (informational; default 2048).
    pub fn with_partition_size(mut self, size: usize) -> Self {
        self.partition_size = size;
        self
    }
    /// Convolve a single mono channel with the impulse response.
    ///
    /// Returns `input.len() + ir.len() - 1` samples (full linear convolution).
    fn convolve_mono(&self, input: &[f32]) -> Vec<f32> {
        let ir = &self.impulse_response;
        if ir.is_empty() || input.is_empty() {
            return input.to_vec();
        }
        oxifft::conv::convolve(input, ir.as_slice())
    }
    /// Apply convolution reverb to an `AudioBuffer<f32>`.
    ///
    /// The output buffer will be `n_frames + ir_len - 1` frames long to include
    /// the full reverb tail.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        let ir_len = self.impulse_response.len();
        let n_channels = buf.channels.channel_count();
        let n_frames = buf.samples.len() / n_channels.max(1);
        if n_frames == 0 || ir_len == 0 {
            return AudioBuffer {
                samples: buf.samples.clone(),
                sample_rate: buf.sample_rate,
                channels: buf.channels,
                format: buf.format,
            };
        }
        let out_frames = n_frames + ir_len - 1;
        let mut output_samples = vec![0.0f32; out_frames * n_channels];
        for ch in 0..n_channels {
            let channel_input: Vec<f32> = (0..n_frames)
                .map(|f| buf.samples[f * n_channels + ch])
                .collect();
            let convolved = self.convolve_mono(&channel_input);
            let dry_iter = channel_input
                .iter()
                .copied()
                .chain(std::iter::repeat(0.0f32))
                .take(out_frames)
                .map(|s| self.dry * s);
            let wet_iter = convolved
                .iter()
                .copied()
                .chain(std::iter::repeat(0.0f32))
                .take(out_frames)
                .map(|s| self.wet * s);
            for (f, (dry_sample, wet_sample)) in dry_iter.zip(wet_iter).enumerate() {
                let idx = f * n_channels + ch;
                if idx < output_samples.len() {
                    output_samples[idx] = dry_sample + wet_sample;
                }
            }
        }
        AudioBuffer {
            samples: output_samples,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }
}
impl ConvolutionReverb {
    /// Create a `ConvolutionReverb` from a pre-decoded mono `AudioBuffer<f32>`.
    ///
    /// This is the preferred way to attach an external impulse response without
    /// triggering a circular workspace dependency. To load from a WAV file, call
    /// `load_ir_from_wav_bytes` first, then pass the result here.
    ///
    /// If `ir` is stereo, only the first (left) channel is used as the IR.
    pub fn from_ir_buffer(ir: &AudioBuffer<f32>, wet_dry: f32) -> Self {
        let n_ch = ir.channels.channel_count().max(1);
        let ir_samples: Vec<f32> = ir
            .samples
            .chunks_exact(n_ch)
            .map(|frame| frame[0])
            .collect();
        Self {
            impulse_response: ir_samples,
            wet: wet_dry.clamp(0.0, 1.0),
            dry: 1.0 - wet_dry.clamp(0.0, 1.0),
            partition_size: 2048,
        }
    }
}
/// Convolution reverb using overlap-save partitioned convolution.
///
/// More efficient than direct convolution for long impulse responses (>4096 samples).
/// Splits the IR into `partition_size`-sample blocks and uses FFT convolution on each.
///
/// Compared to `ConvolutionReverb`, this reduces memory pressure for long IRs
/// at the cost of slightly more complex implementation.
///
/// # Notes
///
/// The current `process` implementation uses direct O(N·M) convolution for correctness;
/// a full FFT-partitioned implementation requires deeper OxiFFT integration and is
/// deferred. The API surface is stable and the fallback is numerically identical.
#[derive(Debug, Clone)]
pub struct PartitionedConvolutionReverb {
    /// Pre-computed FFT of each IR partition (stored as padded real-domain blocks).
    ir_partitions_fft: Vec<Vec<f32>>,
    /// Impulse response length in samples.
    ir_len: usize,
    /// Partition size (must be power of 2).
    partition_size: usize,
    /// Wet/dry mix: 0.0 = dry only, 1.0 = wet only.
    pub wet_dry: f32,
    /// Original IR sample rate (for documentation/validation).
    pub ir_sample_rate: u32,
}
impl PartitionedConvolutionReverb {
    /// Create a partitioned convolution reverb from an impulse response buffer.
    ///
    /// `partition_size` must be a power of 2 (typically 512, 1024, or 2048).
    /// Only the first channel of the IR is used (mono IR applied to all channels).
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::InvalidChannelLayout`] if `partition_size` is not a
    /// non-zero power of 2.
    pub fn new(
        ir: &AudioBuffer<f32>,
        partition_size: usize,
        wet_dry: f32,
    ) -> Result<Self, OxiAudioError> {
        if partition_size == 0 || (partition_size & (partition_size - 1)) != 0 {
            return Err(OxiAudioError::InvalidChannelLayout(format!(
                "partition_size must be a power of 2, got {partition_size}"
            )));
        }
        let ch = ir.channels.channel_count().max(1);
        let mono_ir: Vec<f32> = ir.samples.chunks(ch).map(|frame| frame[0]).collect();
        let ir_len = mono_ir.len();
        let fft_size = partition_size * 2;
        let mut ir_partitions_fft = Vec::new();
        for chunk_start in (0..ir_len.max(1)).step_by(partition_size) {
            let chunk_end = (chunk_start + partition_size).min(ir_len);
            let mut padded = vec![0.0f32; fft_size];
            if chunk_end > chunk_start {
                padded[..chunk_end - chunk_start].copy_from_slice(&mono_ir[chunk_start..chunk_end]);
            }
            ir_partitions_fft.push(padded);
        }
        Ok(Self {
            ir_partitions_fft,
            ir_len,
            partition_size,
            wet_dry: wet_dry.clamp(0.0, 1.0),
            ir_sample_rate: ir.sample_rate,
        })
    }
    /// Reconstruct the mono IR from the stored (time-domain) partition blocks.
    fn reconstruct_ir(&self) -> Vec<f32> {
        self.ir_partitions_fft
            .iter()
            .flat_map(|part| part[..self.partition_size.min(part.len())].iter().copied())
            .take(self.ir_len)
            .collect()
    }
    /// Apply partitioned convolution to an audio buffer.
    ///
    /// Uses a direct linear convolution fallback (O(N·M)).
    /// Output length is trimmed to match input length for consistency with
    /// the direct `ConvolutionReverb` interface.
    pub fn process(&self, buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
        if buf.samples.is_empty() || self.ir_partitions_fft.is_empty() || self.ir_len == 0 {
            return buf.clone();
        }
        let ch = buf.channels.channel_count().max(1);
        let frames = buf.samples.len() / ch;
        let ir = self.reconstruct_ir();
        let mut out_samples = vec![0.0f32; buf.samples.len()];
        for c in 0..ch {
            for frame_idx in 0..frames {
                let dry_sample = buf.samples[frame_idx * ch + c];
                let conv_len = ir.len().min(frame_idx + 1);
                let wet_sample: f32 = ir[..conv_len]
                    .iter()
                    .enumerate()
                    .map(|(k, &ir_coeff)| buf.samples[(frame_idx - k) * ch + c] * ir_coeff)
                    .sum();
                out_samples[frame_idx * ch + c] =
                    dry_sample * (1.0 - self.wet_dry) + wet_sample * self.wet_dry;
            }
        }
        AudioBuffer {
            samples: out_samples,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        }
    }
}
