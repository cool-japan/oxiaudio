use crate::error::OxiAudioError;
use crate::format::SampleFormat;
use crate::layout::ChannelLayout;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioBuffer<T> {
    pub samples: Vec<T>,
    pub sample_rate: u32,
    pub channels: ChannelLayout,
    pub format: SampleFormat,
}

impl<T: Clone> Clone for AudioBuffer<T> {
    fn clone(&self) -> Self {
        AudioBuffer {
            samples: self.samples.clone(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: self.format,
        }
    }
}

impl<T> AudioBuffer<T> {
    /// Duration of the buffer in seconds.
    ///
    /// ```
    /// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    /// let buf = AudioBuffer::<f32>::silence(44100, ChannelLayout::Stereo, 4410);
    /// assert!((buf.duration_secs() - 0.1).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        let ch = self.channels.channel_count();
        if ch == 0 || self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / (ch as f64 * self.sample_rate as f64)
    }

    /// Number of frames (samples per channel) in the buffer.
    ///
    /// ```
    /// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    /// let buf = AudioBuffer::<f32>::silence(44100, ChannelLayout::Stereo, 4410);
    /// assert_eq!(buf.frame_count(), 4410);
    /// ```
    #[must_use]
    pub fn frame_count(&self) -> usize {
        let ch = self.channels.channel_count();
        if ch == 0 {
            return 0;
        }
        self.samples.len() / ch
    }

    /// Returns `true` if the buffer contains no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl AudioBuffer<f32> {
    /// Convert to f64 precision. The format tag becomes `SampleFormat::F64`.
    pub fn to_f64(&self) -> AudioBuffer<f64> {
        AudioBuffer {
            samples: self.samples.iter().map(|&s| s as f64).collect(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: SampleFormat::F64,
        }
    }

    /// De-interleave into one `Vec<f32>` per channel (planar layout).
    ///
    /// Returns `[channel_0_samples, channel_1_samples, ...]` where each inner vec has
    /// `self.samples.len() / n_channels` elements.
    pub fn split_to_planar(&self) -> Vec<Vec<f32>> {
        let n = self.channels.channel_count();
        if n == 1 {
            return vec![self.samples.clone()];
        }
        let frames = self.samples.len() / n;
        let mut planes: Vec<Vec<f32>> = (0..n).map(|_| Vec::with_capacity(frames)).collect();
        for (i, &s) in self.samples.iter().enumerate() {
            planes[i % n].push(s);
        }
        planes
    }

    /// Create a buffer filled with zeros (silence) of the given duration.
    ///
    /// ```
    /// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    /// let buf = AudioBuffer::<f32>::silence(44100, ChannelLayout::Stereo, 4410);
    /// assert_eq!(buf.frame_count(), 4410);
    /// assert!((buf.duration_secs() - 0.1).abs() < 1e-9);
    /// assert!(buf.samples.iter().all(|&s| s == 0.0));
    /// ```
    #[must_use]
    pub fn silence(sample_rate: u32, channels: ChannelLayout, frames: usize) -> Self {
        AudioBuffer {
            samples: vec![0.0f32; frames * channels.channel_count()],
            sample_rate,
            channels,
            format: SampleFormat::F32,
        }
    }

    /// Return a sub-buffer containing frames `[start, end)`.
    ///
    /// ```
    /// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    /// let buf = AudioBuffer::<f32>::silence(44100, ChannelLayout::Stereo, 1000);
    /// let slice = buf.slice_frames(100, 200);
    /// assert_eq!(slice.frame_count(), 100);
    /// assert_eq!(slice.sample_rate, 44100);
    /// ```
    #[must_use]
    pub fn slice_frames(&self, start: usize, end: usize) -> Self {
        let ch = self.channels.channel_count();
        let s = start.saturating_mul(ch);
        let e = end.saturating_mul(ch).min(self.samples.len());
        AudioBuffer {
            samples: self.samples[s..e].to_vec(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: self.format,
        }
    }

    /// Append samples from `other` to this buffer.
    ///
    /// Both buffers must share the same channel layout and sample rate.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::InvalidChannelLayout`] or
    /// [`OxiAudioError::InvalidSampleRate`] on mismatch.
    pub fn append(&mut self, other: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
        if self.channels != other.channels {
            return Err(OxiAudioError::InvalidChannelLayout(format!(
                "cannot append {} buffer to {} buffer",
                other.channels, self.channels
            )));
        }
        if self.sample_rate != other.sample_rate {
            return Err(OxiAudioError::InvalidSampleRate(format!(
                "cannot append {}Hz buffer to {}Hz buffer",
                other.sample_rate, self.sample_rate
            )));
        }
        self.samples.extend_from_slice(&other.samples);
        Ok(())
    }

    /// Peak absolute amplitude across all samples.
    ///
    /// ```
    /// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    /// let buf = AudioBuffer {
    ///     samples: vec![0.3f32, -0.7, 0.5],
    ///     sample_rate: 44100,
    ///     channels: ChannelLayout::Mono,
    ///     format: SampleFormat::F32,
    /// };
    /// assert!((buf.peak_amplitude() - 0.7).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn peak_amplitude(&self) -> f32 {
        self.samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()))
    }

    /// Root-mean-square amplitude across all samples.
    ///
    /// ```
    /// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    /// let buf = AudioBuffer {
    ///     samples: vec![1.0f32, -1.0],
    ///     sample_rate: 44100,
    ///     channels: ChannelLayout::Mono,
    ///     format: SampleFormat::F32,
    /// };
    /// assert!((buf.rms_amplitude() - 1.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn rms_amplitude(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = self.samples.iter().map(|&s| s * s).sum();
        (sum_sq / self.samples.len() as f32).sqrt()
    }

    /// Peak level in dBFS. Returns [`f32::NEG_INFINITY`] for silence.
    #[must_use]
    pub fn peak_db(&self) -> f32 {
        let p = self.peak_amplitude();
        if p <= 0.0 {
            return f32::NEG_INFINITY;
        }
        20.0 * p.log10()
    }

    /// RMS level in dBFS. Returns [`f32::NEG_INFINITY`] for silence.
    #[must_use]
    pub fn rms_db(&self) -> f32 {
        let r = self.rms_amplitude();
        if r <= 0.0 {
            return f32::NEG_INFINITY;
        }
        20.0 * r.log10()
    }

    /// Raised-cosine fade in: `0.5 * (1 - cos(PI * i / n))` for `i in 0..frames`
    ///
    /// ```
    /// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    /// let mut buf = AudioBuffer {
    ///     samples: vec![1.0f32; 200],
    ///     sample_rate: 44100,
    ///     channels: ChannelLayout::Mono,
    ///     format: SampleFormat::F32,
    /// };
    /// buf.fade_in(100);
    /// // First sample should be near zero (fade starts at silence)
    /// assert!(buf.samples[0].abs() < 0.01);
    /// // Last sample (after fade region) should still be near 1.0
    /// assert!((buf.samples[199] - 1.0).abs() < 1e-6);
    /// ```
    pub fn fade_in(&mut self, frames: usize) {
        let ch = self.channels.channel_count();
        let n = frames.min(self.samples.len() / ch.max(1));
        for i in 0..n {
            let gain = 0.5 * (1.0 - (std::f32::consts::PI * i as f32 / n as f32).cos());
            for c in 0..ch {
                self.samples[i * ch + c] *= gain;
            }
        }
    }

    /// Raised-cosine fade out: `0.5 * (1 + cos(PI * i / n))` applied to last `frames` frames
    ///
    /// ```
    /// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    /// let mut buf = AudioBuffer {
    ///     samples: vec![1.0f32; 200],
    ///     sample_rate: 44100,
    ///     channels: ChannelLayout::Mono,
    ///     format: SampleFormat::F32,
    /// };
    /// buf.fade_out(100);
    /// // First sample should still be near 1.0 (before fade region)
    /// assert!((buf.samples[0] - 1.0).abs() < 1e-6);
    /// // Last sample should be near zero (fade ends at silence)
    /// assert!(buf.samples[199].abs() < 0.01);
    /// ```
    pub fn fade_out(&mut self, frames: usize) {
        let ch = self.channels.channel_count();
        let total_frames = self.samples.len() / ch.max(1);
        let n = frames.min(total_frames);
        let start = total_frames.saturating_sub(n);
        for i in 0..n {
            let gain = 0.5 * (1.0 + (std::f32::consts::PI * i as f32 / n as f32).cos());
            let frame = start + i;
            for c in 0..ch {
                self.samples[frame * ch + c] *= gain;
            }
        }
    }

    /// Mix `other` into `self` at the given `level` (0.0 = silent, 1.0 = full), returning a
    /// new [`AudioBuffer<f32>`] that contains the result.
    ///
    /// Both buffers must share the same channel layout and sample rate. When the buffers differ
    /// in length the shorter one is zero-padded during mixing, so the output length is
    /// `max(self.samples.len(), other.samples.len())`.
    ///
    /// For in-place additive mixing use [`mix_with`][Self::mix_with] instead.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::InvalidChannelLayout`] or
    /// [`OxiAudioError::InvalidSampleRate`] on mismatch.
    pub fn mixed_with(
        &self,
        other: &AudioBuffer<f32>,
        level: f32,
    ) -> Result<AudioBuffer<f32>, OxiAudioError> {
        if self.channels != other.channels {
            return Err(OxiAudioError::InvalidChannelLayout(format!(
                "mixed_with: channel mismatch {:?} vs {:?}",
                self.channels, other.channels
            )));
        }
        if self.sample_rate != other.sample_rate {
            return Err(OxiAudioError::InvalidSampleRate(format!(
                "mixed_with: sample rate mismatch {} vs {}",
                self.sample_rate, other.sample_rate
            )));
        }
        let len = self.samples.len().max(other.samples.len());
        let mut samples = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.samples.get(i).copied().unwrap_or(0.0);
            let b = other.samples.get(i).copied().unwrap_or(0.0);
            samples.push(a + b * level);
        }
        Ok(AudioBuffer {
            samples,
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: self.format,
        })
    }

    /// Reverse the audio in place (time reversal), maintaining interleaved channel ordering.
    pub fn reverse(&mut self) {
        let n_ch = self.channels.channel_count();
        if n_ch == 0 {
            return;
        }
        let n_frames = self.samples.len() / n_ch;
        for i in 0..n_frames / 2 {
            let j = n_frames - 1 - i;
            for c in 0..n_ch {
                self.samples.swap(i * n_ch + c, j * n_ch + c);
            }
        }
    }

    /// Return a new [`AudioBuffer<f32>`] that is the time-reversed copy of `self`.
    #[must_use]
    pub fn reversed(&self) -> AudioBuffer<f32> {
        let mut out = self.clone();
        out.reverse();
        out
    }

    /// Crossfade (linear envelope) from `self` (end) to `other` (start) over `fade_frames`.
    ///
    /// The output length is `self.frame_count() + other.frame_count() - fade_frames`. In the
    /// overlap region `self` fades linearly from 1 → 0 while `other` fades from 0 → 1.
    ///
    /// For an equal-power (constant-energy) crossfade between two independent buffers, use the
    /// associated function [`AudioBuffer::crossfade`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::InvalidChannelLayout`] / [`OxiAudioError::InvalidSampleRate`]
    /// if `self` and `other` differ in channel layout or sample rate.
    pub fn linear_crossfade(
        &self,
        other: &AudioBuffer<f32>,
        fade_frames: usize,
    ) -> Result<AudioBuffer<f32>, OxiAudioError> {
        if self.channels != other.channels {
            return Err(OxiAudioError::InvalidChannelLayout(
                "linear_crossfade: channel mismatch".into(),
            ));
        }
        if self.sample_rate != other.sample_rate {
            return Err(OxiAudioError::InvalidSampleRate(
                "linear_crossfade: sample rate mismatch".into(),
            ));
        }
        let n_ch = self.channels.channel_count();
        let self_frames = self.frame_count();
        let other_frames = other.frame_count();
        let fade = fade_frames.min(self_frames).min(other_frames);
        let out_frames = self_frames + other_frames - fade;
        let mut samples = vec![0.0f32; out_frames * n_ch];

        // Copy self verbatim (includes the fade region — will be overwritten below).
        for f in 0..self_frames {
            for c in 0..n_ch {
                samples[f * n_ch + c] = self.samples[f * n_ch + c];
            }
        }

        // Blend end of self with start of other in the fade region.
        let fade_start = self_frames.saturating_sub(fade);
        for f in 0..fade {
            let t = f as f32 / fade.max(1) as f32;
            for c in 0..n_ch {
                let s = self.samples[(fade_start + f) * n_ch + c];
                let o = other.samples[f * n_ch + c];
                samples[(fade_start + f) * n_ch + c] = s * (1.0 - t) + o * t;
            }
        }

        // Copy the non-overlapping tail of other.
        for f in fade..other_frames {
            for c in 0..n_ch {
                samples[(self_frames + f - fade) * n_ch + c] = other.samples[f * n_ch + c];
            }
        }

        Ok(AudioBuffer {
            samples,
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: self.format,
        })
    }

    /// Apply a linear gain ramp from `start_gain` to `end_gain` over the entire buffer,
    /// modifying samples in place.
    pub fn gain_ramp(&mut self, start_gain: f32, end_gain: f32) {
        let n = self.samples.len();
        if n == 0 {
            return;
        }
        for (i, s) in self.samples.iter_mut().enumerate() {
            let t = i as f32 / (n - 1).max(1) as f32;
            *s *= start_gain + (end_gain - start_gain) * t;
        }
    }

    /// Additively mix `other` into `self`, scaling `other` by `gain` (linear).
    ///
    /// The two buffers must share the same channel layout and sample rate. Only
    /// the overlapping prefix is mixed: if `other` is shorter, trailing samples of
    /// `self` are untouched; if `other` is longer, its tail is ignored.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::InvalidChannelLayout`] or
    /// [`OxiAudioError::InvalidSampleRate`] on mismatch.
    pub fn mix_with(&mut self, other: &AudioBuffer<f32>, gain: f32) -> Result<(), OxiAudioError> {
        if self.channels != other.channels {
            return Err(OxiAudioError::InvalidChannelLayout(format!(
                "cannot mix {} buffer into {} buffer",
                other.channels, self.channels
            )));
        }
        if self.sample_rate != other.sample_rate {
            return Err(OxiAudioError::InvalidSampleRate(format!(
                "cannot mix {}Hz buffer into {}Hz buffer",
                other.sample_rate, self.sample_rate
            )));
        }
        for (dst, &src) in self.samples.iter_mut().zip(other.samples.iter()) {
            *dst += src * gain;
        }
        Ok(())
    }

    /// Crossfade from `a` into `b` over `overlap_frames` frames using an
    /// equal-power (raised-cosine) envelope.
    ///
    /// The result is `a` (faded out over the overlap region at its tail) followed
    /// by `b` (faded in over the overlap region at its head), with the two overlap
    /// regions summed. Total output length is
    /// `a.frame_count() + b.frame_count() - overlap`.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::InvalidChannelLayout`] / [`OxiAudioError::InvalidSampleRate`]
    /// if `a` and `b` differ in channel layout or sample rate.
    pub fn crossfade(
        a: &AudioBuffer<f32>,
        b: &AudioBuffer<f32>,
        overlap_frames: usize,
    ) -> Result<AudioBuffer<f32>, OxiAudioError> {
        if a.channels != b.channels {
            return Err(OxiAudioError::InvalidChannelLayout(format!(
                "crossfade channel mismatch: {} vs {}",
                a.channels, b.channels
            )));
        }
        if a.sample_rate != b.sample_rate {
            return Err(OxiAudioError::InvalidSampleRate(format!(
                "crossfade sample-rate mismatch: {}Hz vs {}Hz",
                a.sample_rate, b.sample_rate
            )));
        }
        let ch = a.channels.channel_count();
        let a_frames = a.frame_count();
        let b_frames = b.frame_count();
        let overlap = overlap_frames.min(a_frames).min(b_frames);
        let out_frames = a_frames + b_frames - overlap;
        let mut samples = vec![0.0f32; out_frames * ch];

        // Copy the non-overlapping head of `a`.
        let a_head = a_frames - overlap;
        samples[..a_head * ch].copy_from_slice(&a.samples[..a_head * ch]);

        // Overlap region: equal-power crossfade.
        for i in 0..overlap {
            let t = if overlap > 1 {
                i as f32 / (overlap - 1) as f32
            } else {
                1.0
            };
            // Equal-power (constant energy) gains.
            let gain_a = (std::f32::consts::FRAC_PI_2 * t).cos();
            let gain_b = (std::f32::consts::FRAC_PI_2 * t).sin();
            for c in 0..ch {
                let sa = a.samples[(a_head + i) * ch + c];
                let sb = b.samples[i * ch + c];
                samples[(a_head + i) * ch + c] = sa * gain_a + sb * gain_b;
            }
        }

        // Copy the non-overlapping tail of `b`.
        let b_tail_start = overlap;
        let out_tail_start = a_head + overlap;
        for frame in b_tail_start..b_frames {
            for c in 0..ch {
                samples[(out_tail_start + frame - b_tail_start) * ch + c] =
                    b.samples[frame * ch + c];
            }
        }

        Ok(AudioBuffer {
            samples,
            sample_rate: a.sample_rate,
            channels: a.channels,
            format: SampleFormat::F32,
        })
    }

    /// Fast linear-interpolation resampler (preview quality; not high-fidelity).
    pub fn resample_linear(&self, target_rate: u32) -> AudioBuffer<f32> {
        if self.sample_rate == target_rate || self.samples.is_empty() {
            return self.clone();
        }
        let ch = self.channels.channel_count();
        let src_frames = self.frame_count();
        let ratio = self.sample_rate as f64 / target_rate as f64;
        let dst_frames = (src_frames as f64 / ratio).round() as usize;
        let mut samples = Vec::with_capacity(dst_frames * ch);

        for di in 0..dst_frames {
            let src_pos = di as f64 * ratio;
            let src_idx = src_pos as usize;
            let frac = src_pos - src_idx as f64;
            let next_idx = (src_idx + 1).min(src_frames.saturating_sub(1));
            for c in 0..ch {
                let s0 = self.samples[src_idx * ch + c];
                let s1 = self.samples[next_idx * ch + c];
                samples.push(s0 + (s1 - s0) * frac as f32);
            }
        }

        AudioBuffer {
            samples,
            sample_rate: target_rate,
            channels: self.channels,
            format: SampleFormat::F32,
        }
    }
}

impl AudioBuffer<f64> {
    /// Convert to f32 (lossy). The format tag becomes `SampleFormat::F32`.
    pub fn to_f32(&self) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: self.samples.iter().map(|&s| s as f32).collect(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: SampleFormat::F32,
        }
    }
}

impl<T: Clone + Default> AudioBuffer<T> {
    /// Interleave planar channel data into an `AudioBuffer<T>`.
    ///
    /// `channels_data` must be non-empty; all inner vecs must have the same length.
    /// `ChannelLayout`: 1 channel → `Mono`, otherwise → `Stereo`.
    pub fn from_planar(channels_data: Vec<Vec<T>>, sample_rate: u32, format: SampleFormat) -> Self {
        let n = channels_data.len();
        let layout = if n == 1 {
            ChannelLayout::Mono
        } else {
            ChannelLayout::Stereo
        };
        let frames = channels_data.first().map(|c| c.len()).unwrap_or(0);
        let mut samples = Vec::with_capacity(frames * n);
        for frame in 0..frames {
            for ch in &channels_data {
                if let Some(s) = ch.get(frame) {
                    samples.push(s.clone());
                }
            }
        }
        AudioBuffer {
            samples,
            sample_rate,
            channels: layout,
            format,
        }
    }
}

impl From<&AudioBuffer<i16>> for AudioBuffer<f32> {
    fn from(buf: &AudioBuffer<i16>) -> Self {
        AudioBuffer {
            samples: buf
                .samples
                .iter()
                .map(|&s| s as f32 / i16::MAX as f32)
                .collect(),
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: SampleFormat::F32,
        }
    }
}

impl From<&AudioBuffer<f32>> for AudioBuffer<i16> {
    fn from(buf: &AudioBuffer<f32>) -> Self {
        let scale = i16::MAX as f32;
        let n = buf.samples.len();
        let mut samples = Vec::with_capacity(n);
        // chunks_exact(8) hints the auto-vectorizer for SIMD throughput.
        let chunks = buf.samples.chunks_exact(8);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for &s in chunk {
                samples.push((s.clamp(-1.0, 1.0) * scale) as i16);
            }
        }
        for &s in remainder {
            samples.push((s.clamp(-1.0, 1.0) * scale) as i16);
        }
        AudioBuffer {
            samples,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: SampleFormat::I16,
        }
    }
}

impl From<&AudioBuffer<i32>> for AudioBuffer<f32> {
    fn from(buf: &AudioBuffer<i32>) -> Self {
        AudioBuffer {
            samples: buf
                .samples
                .iter()
                .map(|&s| s as f32 / i32::MAX as f32)
                .collect(),
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: SampleFormat::F32,
        }
    }
}

impl From<&AudioBuffer<f32>> for AudioBuffer<i32> {
    fn from(buf: &AudioBuffer<f32>) -> Self {
        let scale = i32::MAX as f32;
        let n = buf.samples.len();
        let mut samples = Vec::with_capacity(n);
        // chunks_exact(8) hints the auto-vectorizer for SIMD throughput.
        let chunks = buf.samples.chunks_exact(8);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for &s in chunk {
                samples.push((s.clamp(-1.0, 1.0) * scale) as i32);
            }
        }
        for &s in remainder {
            samples.push((s.clamp(-1.0, 1.0) * scale) as i32);
        }
        AudioBuffer {
            samples,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: SampleFormat::I32,
        }
    }
}

impl AudioBuffer<f32> {
    /// Convert f32 [-1.0, 1.0] to 24-bit signed integer range [-8388608, 8388607].
    ///
    /// This is distinct from the [`From`] impl which uses the full 32-bit range.
    /// Use this when writing 24-bit PCM (e.g., WAV I24 or AIFF 24-bit).
    ///
    /// Values are clamped to [-1.0, 1.0] before scaling.
    pub fn to_i32_24bit(&self) -> AudioBuffer<i32> {
        const SCALE: f32 = 8_388_607.0; // 2^23 − 1
        const NEG_SCALE: f32 = 8_388_608.0; // −2^23 magnitude
        const NEG_MAX: f32 = -8_388_608.0; // −2^23

        let n = self.samples.len();
        let mut samples = Vec::with_capacity(n);
        // chunks_exact(8) hints the auto-vectorizer for SIMD throughput.
        let chunks = self.samples.chunks_exact(8);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for &s in chunk {
                let clamped = s.clamp(-1.0, 1.0);
                samples.push(if clamped < 0.0 {
                    (clamped * NEG_SCALE).round().max(NEG_MAX) as i32
                } else {
                    (clamped * SCALE).round() as i32
                });
            }
        }
        for &s in remainder {
            let clamped = s.clamp(-1.0, 1.0);
            samples.push(if clamped < 0.0 {
                (clamped * NEG_SCALE).round().max(NEG_MAX) as i32
            } else {
                (clamped * SCALE).round() as i32
            });
        }
        AudioBuffer {
            samples,
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: SampleFormat::I24,
        }
    }

    /// Serialize this buffer to the compact binary IPC format (`ABUF` v1).
    ///
    /// The returned bytes begin with the `ABUF` magic header followed by sample
    /// metadata and raw little-endian f32 samples. See [`crate::ipc`] for the
    /// full wire-format specification.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError`] if the in-memory write fails (highly unlikely).
    #[must_use = "discarding the Result ignores serialize errors"]
    pub fn to_ipc_bytes(&self) -> Result<Vec<u8>, OxiAudioError> {
        crate::ipc::to_ipc_bytes(self)
    }

    /// Deserialize an [`AudioBuffer<f32>`] from bytes produced by [`Self::to_ipc_bytes`].
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError`] if the data is truncated, has an invalid magic
    /// header or version, or contains an unsupported channel-layout byte.
    #[must_use = "discarding the Result ignores deserialize errors"]
    pub fn from_ipc_bytes(data: &[u8]) -> Result<Self, OxiAudioError> {
        crate::ipc::from_ipc_bytes(data)
    }
}

impl From<&AudioBuffer<u8>> for AudioBuffer<f32> {
    /// Convert 8-bit unsigned PCM to f32.
    ///
    /// The bias for u8 PCM is at 128 (silence). Mapping:
    /// `(sample as f32 - 128.0) / 128.0`, producing [-1.0, ~1.0].
    fn from(buf: &AudioBuffer<u8>) -> Self {
        AudioBuffer {
            samples: buf
                .samples
                .iter()
                .map(|&s| (s as f32 - 128.0) / 128.0)
                .collect(),
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: SampleFormat::F32,
        }
    }
}

impl From<&AudioBuffer<f32>> for AudioBuffer<u8> {
    /// Convert f32 [-1.0, 1.0] to 8-bit unsigned PCM.
    ///
    /// Mapping: `((sample * 128.0) + 128.0).clamp(0.0, 255.0) as u8`.
    /// The output format tag is `SampleFormat::U8`.
    fn from(buf: &AudioBuffer<f32>) -> Self {
        AudioBuffer {
            samples: buf
                .samples
                .iter()
                .map(|&s| ((s * 128.0) + 128.0).clamp(0.0, 255.0) as u8)
                .collect(),
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: SampleFormat::U8,
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone planar conversion helpers (supplement split_to_planar / from_planar)
// ---------------------------------------------------------------------------

/// Convert an interleaved `AudioBuffer<f32>` to planar format.
///
/// Returns one `Vec<f32>` per channel.
pub fn to_planar(buf: &AudioBuffer<f32>) -> Vec<Vec<f32>> {
    buf.split_to_planar()
}

/// Convert planar channel data into an interleaved `AudioBuffer<f32>`.
///
/// # Errors
///
/// Returns [`OxiAudioError::InvalidChannelLayout`] if `channels` is empty or
/// if the inner vectors differ in length.
pub fn from_planar(
    channels: &[Vec<f32>],
    sample_rate: u32,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if channels.is_empty() {
        return Err(OxiAudioError::InvalidChannelLayout(
            "from_planar: no channels provided".into(),
        ));
    }
    let frame_len = channels[0].len();
    if channels.iter().any(|c| c.len() != frame_len) {
        return Err(OxiAudioError::InvalidChannelLayout(
            "from_planar: channel vectors have different lengths".into(),
        ));
    }
    let n = channels.len();
    let layout = if n == 1 {
        ChannelLayout::Mono
    } else {
        ChannelLayout::Stereo
    };
    // Allocate interleaved output and fill using chunks_exact_mut(n) for
    // cache-friendly write access and LLVM auto-vectorisation hints.
    let mut samples = vec![0.0f32; frame_len * n];
    for (frame_idx, chunk) in samples.chunks_exact_mut(n).enumerate() {
        for (ch_idx, dst) in chunk.iter_mut().enumerate() {
            *dst = channels[ch_idx][frame_idx];
        }
    }
    Ok(AudioBuffer {
        samples,
        sample_rate,
        channels: layout,
        format: SampleFormat::F32,
    })
}

/// Like [`from_planar`], but writes into a pre-allocated interleaved output slice
/// instead of allocating a new `Vec`. Caller must ensure `out.len() >= n_channels * n_frames`.
///
/// # Errors
///
/// Returns [`OxiAudioError::InvalidChannelLayout`] if `planes` is empty,
/// if the inner slices differ in length, or if `out` is too small.
///
/// # Examples
///
/// ```
/// use oxiaudio_core::from_planar_into;
/// let left = vec![0.0f32, 0.5, 1.0];
/// let right = vec![0.1f32, 0.6, 0.9];
/// let planes: &[&[f32]] = &[&left, &right];
/// let mut out = vec![0.0f32; 6];
/// from_planar_into(planes, 44100, &mut out).unwrap();
/// assert_eq!(&out, &[0.0, 0.1, 0.5, 0.6, 1.0, 0.9]);
/// ```
pub fn from_planar_into(
    planes: &[&[f32]],
    _sample_rate: u32,
    out: &mut [f32],
) -> Result<(), OxiAudioError> {
    let n_channels = planes.len();
    if n_channels == 0 {
        return Ok(());
    }
    let n_frames = planes[0].len();
    if planes.iter().any(|p| p.len() != n_frames) {
        return Err(OxiAudioError::InvalidChannelLayout(
            "from_planar_into: mismatched plane lengths".into(),
        ));
    }
    if out.len() < n_channels * n_frames {
        return Err(OxiAudioError::InvalidChannelLayout(
            "from_planar_into: output slice too small".into(),
        ));
    }
    for (frame, chunk) in out.chunks_exact_mut(n_channels).enumerate() {
        for (ch, s) in chunk.iter_mut().zip(planes.iter().map(|p| p[frame])) {
            *ch = s;
        }
    }
    Ok(())
}

/// Like [`from_planar`], but skips length validation checks for hot paths where
/// the caller guarantees all planes have the same length.
///
/// Unlike the name might suggest, this is purely safe Rust — it simply omits the
/// length-validation check and panics (rather than returning `Err`) on out-of-bounds
/// access if the invariant is violated.
///
/// For inputs that satisfy the invariant, this produces the same result as [`from_planar`].
///
/// # Panics
///
/// Panics in both debug and release builds if the planes differ in length (index out-of-bounds).
///
/// # Examples
///
/// ```
/// use oxiaudio_core::from_planar_unchecked;
/// let left = vec![0.0f32, 0.5, 1.0];
/// let right = vec![0.1f32, 0.6, 0.9];
/// let buf = from_planar_unchecked(&[&left, &right], 44100);
/// assert_eq!(buf.frame_count(), 3);
/// assert_eq!(&buf.samples, &[0.0, 0.1, 0.5, 0.6, 1.0, 0.9]);
/// ```
pub fn from_planar_unchecked(planes: &[&[f32]], sample_rate: u32) -> AudioBuffer<f32> {
    let n_channels = planes.len();
    if n_channels == 0 {
        return AudioBuffer {
            samples: Vec::new(),
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
    }
    let n_frames = planes[0].len();
    // Use chunks_exact_mut(n_channels) for cache-friendly writes and LLVM
    // auto-vectorisation, consistent with from_planar_into.
    let mut samples = vec![0.0f32; n_channels * n_frames];
    for (frame_idx, chunk) in samples.chunks_exact_mut(n_channels).enumerate() {
        for (ch_idx, dst) in chunk.iter_mut().enumerate() {
            *dst = planes[ch_idx][frame_idx];
        }
    }
    let channels = match n_channels {
        1 => ChannelLayout::Mono,
        2 => ChannelLayout::Stereo,
        _ => ChannelLayout::Stereo,
    };
    AudioBuffer {
        samples,
        sample_rate,
        channels,
        format: SampleFormat::F32,
    }
}

// ---------------------------------------------------------------------------
// Channel conversion helpers (downmix / upmix)
// ---------------------------------------------------------------------------

/// Downmix a 5.1 (`Surround51`) `AudioBuffer<f32>` to stereo.
///
/// Channel order: FL, FR, FC, LFE, RL, RR.
/// Uses ITU-R BS.775-3 folddown: center at −3 dB, surround at −3 dB, LFE discarded.
///
/// # Errors
///
/// Returns [`OxiAudioError::InvalidChannelLayout`] if `buf` is not `Surround51`.
pub fn downmix_51_to_stereo(buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if buf.channels != ChannelLayout::Surround51 {
        return Err(OxiAudioError::InvalidChannelLayout(format!(
            "expected Surround51, got {:?}",
            buf.channels
        )));
    }
    let n_frames = buf.samples.len() / 6;
    let mut out = vec![0.0f32; n_frames * 2];
    for f in 0..n_frames {
        let fl = buf.samples[f * 6];
        let fr = buf.samples[f * 6 + 1];
        let fc = buf.samples[f * 6 + 2];
        // LFE (buf.samples[f * 6 + 3]) is discarded.
        let rl = buf.samples[f * 6 + 4];
        let rr = buf.samples[f * 6 + 5];
        let l = (fl + 0.707 * fc + 0.707 * rl).clamp(-1.0, 1.0);
        let r = (fr + 0.707 * fc + 0.707 * rr).clamp(-1.0, 1.0);
        out[f * 2] = l;
        out[f * 2 + 1] = r;
    }
    Ok(AudioBuffer {
        samples: out,
        sample_rate: buf.sample_rate,
        channels: ChannelLayout::Stereo,
        format: buf.format,
    })
}

/// Downmix any multi-channel `AudioBuffer<f32>` to mono by averaging all channels.
pub fn downmix_to_mono(buf: &AudioBuffer<f32>) -> AudioBuffer<f32> {
    let n_ch = buf.channels.channel_count();
    if n_ch == 1 {
        return buf.clone();
    }
    let n_frames = buf.samples.len() / n_ch;
    let samples = (0..n_frames)
        .map(|f| {
            let sum: f32 = (0..n_ch).map(|c| buf.samples[f * n_ch + c]).sum();
            sum / n_ch as f32
        })
        .collect();
    AudioBuffer {
        samples,
        sample_rate: buf.sample_rate,
        channels: ChannelLayout::Mono,
        format: buf.format,
    }
}

/// Upmix a mono `AudioBuffer<f32>` to stereo by duplicating the single channel.
///
/// # Errors
///
/// Returns [`OxiAudioError::InvalidChannelLayout`] if `buf` is not `Mono`.
pub fn upmix_mono_to_stereo(buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if buf.channels != ChannelLayout::Mono {
        return Err(OxiAudioError::InvalidChannelLayout(format!(
            "expected Mono, got {:?}",
            buf.channels
        )));
    }
    let n_frames = buf.samples.len();
    let mut out = vec![0.0f32; n_frames * 2];
    for f in 0..n_frames {
        out[f * 2] = buf.samples[f];
        out[f * 2 + 1] = buf.samples[f];
    }
    Ok(AudioBuffer {
        samples: out,
        sample_rate: buf.sample_rate,
        channels: ChannelLayout::Stereo,
        format: buf.format,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod prop_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_f32_to_i16_roundtrip_in_range(
            s in -1.0f32..=1.0f32
        ) {
            // f32 → i16 → f32 should be within 1 LSB (1/32767 ≈ 3e-5)
            let i = (s * 32767.0f32).round() as i16;
            let back = i as f32 / 32767.0f32;
            prop_assert!((back - s).abs() <= 2.0f32 / 32767.0f32,
                "roundtrip error too large: {s} → {i} → {back}");
        }

        #[test]
        fn prop_u8_to_f32_range(byte in 0u8..=255u8) {
            let f = (byte as f32 - 128.0) / 128.0;
            prop_assert!((-1.0f32..=1.0f32).contains(&f), "f32 out of range: {f}");
        }

        #[test]
        fn prop_f32_to_i32_24bit_range(s in -1.0f32..=1.0f32) {
            // AudioBuffer<f32>::to_i32_24bit() scaling
            let i = (s.clamp(-1.0, 1.0) * 8_388_607.0).round() as i32;
            prop_assert!((-8_388_608..=8_388_607i32).contains(&i),
                "i32 24-bit out of range: {s} → {i}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_to_i32_24bit_scaling() {
        let buf = AudioBuffer {
            samples: vec![1.0f32, -1.0, 0.0],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let out = buf.to_i32_24bit();
        assert_eq!(out.format, SampleFormat::I24);
        // +1.0 → 8388607 (max positive 24-bit)
        assert_eq!(out.samples[0], 8_388_607, "+1.0 must map to 8388607");
        // −1.0 → −8388608 (min 24-bit, clamped)
        assert_eq!(out.samples[1], -8_388_608, "-1.0 must map to -8388608");
        // 0.0 → 0
        assert_eq!(out.samples[2], 0, "0.0 must map to 0");
    }

    #[test]
    fn test_u8_to_f32_conversion() {
        let buf_u8: AudioBuffer<u8> = AudioBuffer {
            samples: vec![128, 0, 255],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::U8,
        };
        let buf_f32 = AudioBuffer::<f32>::from(&buf_u8);

        // 128 → 0.0
        assert!(
            buf_f32.samples[0].abs() < 1e-6,
            "expected ~0.0, got {}",
            buf_f32.samples[0]
        );
        // 0 → -1.0
        assert!(
            (buf_f32.samples[1] - (-1.0)).abs() < 1e-6,
            "expected ~-1.0, got {}",
            buf_f32.samples[1]
        );
        // 255 → (255 - 128) / 128 = 127/128 ≈ 0.9921875
        let expected = (255.0f32 - 128.0) / 128.0;
        assert!(
            (buf_f32.samples[2] - expected).abs() < 1e-6,
            "expected ~{expected}, got {}",
            buf_f32.samples[2]
        );
    }

    #[test]
    fn test_f32_to_u8_conversion() {
        // Roundtrip: f32 → u8 → f32, tolerance within 1/128.
        let original_f32: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let buf_f32: AudioBuffer<f32> = AudioBuffer {
            samples: original_f32.clone(),
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };

        let buf_u8 = AudioBuffer::<u8>::from(&buf_f32);
        assert_eq!(buf_u8.format, SampleFormat::U8);

        let buf_roundtrip = AudioBuffer::<f32>::from(&buf_u8);

        let tolerance = 1.0 / 128.0;
        for (i, (&orig, &rt)) in original_f32
            .iter()
            .zip(buf_roundtrip.samples.iter())
            .enumerate()
        {
            assert!(
                (orig - rt).abs() <= tolerance,
                "sample {i}: original={orig}, roundtrip={rt}, diff={}",
                (orig - rt).abs()
            );
        }
    }

    #[test]
    fn test_audio_buffer_ipc_method_roundtrip_stereo() {
        // 100-frame stereo f32 buffer roundtripped through the method-level IPC API.
        let n_frames = 100usize;
        let sample_rate = 48_000u32;
        let samples: Vec<f32> = (0..n_frames * 2)
            .map(|i| (i as f32 / (n_frames * 2) as f32) * 2.0 - 1.0)
            .collect();
        let buf = AudioBuffer {
            samples: samples.clone(),
            sample_rate,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };

        let bytes = buf.to_ipc_bytes().expect("to_ipc_bytes must succeed");
        let decoded =
            AudioBuffer::<f32>::from_ipc_bytes(&bytes).expect("from_ipc_bytes must succeed");

        assert_eq!(decoded.sample_rate, sample_rate, "sample_rate must match");
        assert_eq!(
            decoded.channels,
            ChannelLayout::Stereo,
            "channels must match"
        );
        assert_eq!(decoded.format, SampleFormat::F32, "format must match");
        assert_eq!(decoded.frame_count(), n_frames, "frame_count must match");
        assert_eq!(
            decoded.samples.len(),
            samples.len(),
            "sample count must match"
        );
        for (i, (&a, &b)) in samples.iter().zip(decoded.samples.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-7,
                "sample {i}: original={a}, decoded={b}"
            );
        }
    }

    #[test]
    fn test_empty_buffer_operations() {
        let buf: AudioBuffer<f32> = AudioBuffer {
            samples: vec![],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        assert!(buf.is_empty());
        assert_eq!(buf.frame_count(), 0);
        assert_eq!(buf.duration_secs(), 0.0);
        assert_eq!(buf.peak_amplitude(), 0.0);
        assert_eq!(buf.rms_amplitude(), 0.0);
    }

    #[test]
    fn test_single_sample_buffer() {
        let buf = AudioBuffer {
            samples: vec![0.5f32],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        assert!(!buf.is_empty());
        assert_eq!(buf.frame_count(), 1);
        assert!((buf.duration_secs() - 1.0 / 44100.0).abs() < 1e-10);
        assert!((buf.peak_amplitude() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_from_planar_into_matches_from_planar() {
        let left = vec![0.1f32, 0.3, 0.5];
        let right = vec![0.2f32, 0.4, 0.6];
        let planes: &[&[f32]] = &[&left, &right];

        // Reference via from_planar (Vec-based)
        let ref_buf =
            from_planar(&[left.clone(), right.clone()], 44100).expect("from_planar failed");

        // New function writing into pre-allocated slice
        let mut out = vec![0.0f32; 6];
        from_planar_into(planes, 44100, &mut out).expect("from_planar_into failed");

        assert_eq!(
            &out,
            ref_buf.samples.as_slice(),
            "from_planar_into must match from_planar"
        );
    }

    #[test]
    fn test_from_planar_into_error_too_small() {
        let left = vec![0.0f32; 3];
        let right = vec![0.0f32; 3];
        let planes: &[&[f32]] = &[&left, &right];
        let mut out = vec![0.0f32; 5]; // needs 6
        assert!(from_planar_into(planes, 44100, &mut out).is_err());
    }

    #[test]
    fn test_from_planar_into_mismatched_lengths() {
        let left = vec![0.0f32; 3];
        let right = vec![0.0f32; 4]; // different length
        let planes: &[&[f32]] = &[&left, &right];
        let mut out = vec![0.0f32; 8];
        assert!(from_planar_into(planes, 44100, &mut out).is_err());
    }

    #[test]
    fn test_from_planar_unchecked_matches_from_planar() {
        let left = vec![0.0f32, 0.5, 1.0];
        let right = vec![0.1f32, 0.6, 0.9];
        let planes: &[&[f32]] = &[&left, &right];

        let ref_buf =
            from_planar(&[left.clone(), right.clone()], 44100).expect("from_planar failed");

        let unchecked = from_planar_unchecked(planes, 44100);

        assert_eq!(
            unchecked.samples, ref_buf.samples,
            "from_planar_unchecked must produce same interleaving as from_planar"
        );
        assert_eq!(unchecked.frame_count(), 3);
        assert_eq!(unchecked.channels, ChannelLayout::Stereo);
    }
}
