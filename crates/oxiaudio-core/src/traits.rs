use crate::buffer::AudioBuffer;
use crate::error::OxiAudioError;

pub trait AudioDecoder {
    fn decode(
        &mut self,
        src: impl std::io::Read + std::io::Seek + Send + Sync + 'static,
    ) -> Result<AudioBuffer<f32>, OxiAudioError>;
}

pub trait AudioEncoder {
    fn encode(
        &mut self,
        buf: &AudioBuffer<f32>,
        dst: impl std::io::Write + std::io::Seek,
    ) -> Result<(), OxiAudioError>;
}

/// Chunked-decode contract (M2 declares the trait; decode-side impl lands in M3).
pub trait StreamingDecoder {
    fn decode_frames(
        &mut self,
        src: impl std::io::Read + std::io::Seek + Send + Sync + 'static,
        block_size: usize,
    ) -> impl Iterator<Item = Result<AudioBuffer<f32>, OxiAudioError>>;
}

/// In-place-style transform producing a new buffer (pure, no I/O).
///
/// # Examples
///
/// ```
/// use oxiaudio_core::{AudioBuffer, AudioFilter, ChannelLayout, OxiAudioError, SampleFormat};
///
/// struct GainFilter { factor: f32 }
/// impl AudioFilter for GainFilter {
///     fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
///         let samples = buf.samples.iter().map(|&s| s * self.factor).collect();
///         Ok(AudioBuffer { samples, ..*buf })
///     }
/// }
///
/// let buf = AudioBuffer {
///     samples: vec![0.5f32],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
/// let out = GainFilter { factor: 2.0 }.apply(&buf).unwrap();
/// assert!((out.samples[0] - 1.0).abs() < 1e-6);
/// ```
pub trait AudioFilter {
    fn apply(&self, buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError>;
}

/// Pull-based chunk source (e.g. a streaming decoder).
///
/// # Examples
///
/// ```
/// use oxiaudio_core::{AudioBuffer, AudioSource, ChannelLayout, OxiAudioError, SampleFormat};
///
/// struct VecSource { chunks: Vec<AudioBuffer<f32>> }
/// impl AudioSource for VecSource {
///     fn read_chunk(&mut self) -> Result<Option<AudioBuffer<f32>>, OxiAudioError> {
///         if self.chunks.is_empty() { return Ok(None); }
///         Ok(Some(self.chunks.remove(0)))
///     }
/// }
///
/// let chunk = AudioBuffer {
///     samples: vec![0.1f32, 0.2],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Stereo,
///     format: SampleFormat::F32,
/// };
/// let mut src = VecSource { chunks: vec![chunk] };
/// let got = src.read_chunk().unwrap();
/// assert!(got.is_some());
/// assert!(src.read_chunk().unwrap().is_none());
/// ```
pub trait AudioSource {
    fn read_chunk(&mut self) -> Result<Option<AudioBuffer<f32>>, OxiAudioError>;
}

/// Push-based chunk sink (e.g. a streaming encoder).
///
/// # Examples
///
/// ```
/// use oxiaudio_core::{AudioBuffer, AudioSink, ChannelLayout, OxiAudioError, SampleFormat};
///
/// struct VecSink { received: Vec<AudioBuffer<f32>> }
/// impl AudioSink for VecSink {
///     fn write_chunk(&mut self, buf: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
///         self.received.push(AudioBuffer {
///             samples: buf.samples.clone(),
///             ..*buf
///         });
///         Ok(())
///     }
/// }
///
/// let buf = AudioBuffer {
///     samples: vec![0.3f32],
///     sample_rate: 48_000,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
/// let mut sink = VecSink { received: vec![] };
/// sink.write_chunk(&buf).unwrap();
/// assert_eq!(sink.received.len(), 1);
/// ```
pub trait AudioSink {
    fn write_chunk(&mut self, buf: &AudioBuffer<f32>) -> Result<(), OxiAudioError>;
}
