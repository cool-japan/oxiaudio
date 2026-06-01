use oxiaudio_core::{AudioBuffer, OxiAudioError};

/// A boxed DSP processing step: an `AudioBuffer<f32>` transform that may fail.
pub type DspStep =
    Box<dyn Fn(&AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> + Send + Sync>;

/// A builder for chaining DSP processing steps.
///
/// Each step is a closure that takes an [`AudioBuffer<f32>`] and returns a
/// `Result<AudioBuffer<f32>, OxiAudioError>`. Steps are applied in the order
/// they were appended via [`DspChain::then`].
///
/// # Example
///
/// ```ignore
/// let chain = DspChain::new()
///     .then(|buf| Ok(gain_step(buf)))
///     .then(|buf| Ok(normalize_step(buf)));
/// let out = chain.process(&input)?;
/// ```
pub struct DspChain {
    steps: Vec<DspStep>,
}

impl DspChain {
    /// Create an empty `DspChain`.
    pub fn new() -> Self {
        Self { steps: vec![] }
    }

    /// Append a processing step to the chain (builder style).
    pub fn then<F>(mut self, f: F) -> Self
    where
        F: Fn(&AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> + Send + Sync + 'static,
    {
        self.steps.push(Box::new(f));
        self
    }

    /// Append a filter implementing [`oxiaudio_core::AudioFilter`] as a step.
    #[must_use]
    pub fn then_filter<F>(self, filter: F) -> Self
    where
        F: oxiaudio_core::AudioFilter + Send + Sync + 'static,
    {
        self.then(move |buf| filter.apply(buf))
    }

    /// Process `buf` through all steps in order.
    ///
    /// Stops and returns the first error encountered.
    #[must_use = "returns the processed AudioBuffer"]
    pub fn process(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        let mut current = buf.clone();
        for step in &self.steps {
            current = step(&current)?;
        }
        Ok(current)
    }
}

impl Default for DspChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{ChannelLayout, SampleFormat};

    fn make_buf(samples: Vec<f32>) -> AudioBuffer<f32> {
        AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_dspchain_passthrough() {
        let buf = make_buf(vec![1.0, 2.0, 3.0, 4.0]);
        let chain = DspChain::new();
        let out = chain.process(&buf).expect("passthrough should succeed");
        assert_eq!(out.samples, buf.samples);
        assert_eq!(out.sample_rate, buf.sample_rate);
    }

    #[test]
    fn test_dspchain_two_steps() {
        // Gain doubler then halver should produce roughly the original values.
        let buf = make_buf(vec![0.5, -0.5, 0.25]);
        let chain = DspChain::new()
            .then(|b| {
                let mut out = b.clone();
                for s in &mut out.samples {
                    *s *= 2.0;
                }
                Ok(out)
            })
            .then(|b| {
                let mut out = b.clone();
                for s in &mut out.samples {
                    *s *= 0.5;
                }
                Ok(out)
            });
        let out = chain.process(&buf).expect("two-step chain should succeed");
        for (orig, got) in buf.samples.iter().zip(out.samples.iter()) {
            assert!((orig - got).abs() < 1e-6, "expected {orig}, got {got}");
        }
    }

    #[test]
    fn test_dspchain_error_propagates() {
        let buf = make_buf(vec![1.0]);
        let chain = DspChain::new()
            .then(|_| Err(OxiAudioError::UnsupportedFormat("test error".to_owned())));
        assert!(chain.process(&buf).is_err());
    }
}
