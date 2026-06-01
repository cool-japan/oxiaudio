//! Raw PCM reader with explicit format configuration.

use std::io::{self, Read};
use std::path::Path;

use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError, SampleFormat};

/// Configuration for decoding raw PCM audio data.
#[derive(Debug, Clone)]
pub struct RawPcmConfig {
    /// Target sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Sample format of the raw data.
    pub format: SampleFormat,
    /// Whether sample bytes are in little-endian order (`true`) or big-endian (`false`).
    pub little_endian: bool,
    /// Number of bytes to skip at the start of the stream (e.g. a raw header).
    pub skip_bytes: usize,
}

/// Skip exactly `n` bytes by draining via a 512-byte scratch buffer.
fn skip_bytes<R: Read>(reader: &mut R, mut n: usize) -> io::Result<()> {
    let mut scratch = [0u8; 512];
    while n > 0 {
        let to_read = n.min(scratch.len());
        reader.read_exact(&mut scratch[..to_read])?;
        n -= to_read;
    }
    Ok(())
}

/// Decode raw PCM from `reader` using the provided `config`.
///
/// Reads samples until the stream is exhausted (EOF). Each sample is converted
/// to `f32` and normalized to the range `[-1.0, 1.0]`.
///
/// # Supported formats
///
/// | [`SampleFormat`] | Bytes | Conversion                      |
/// |------------------|-------|---------------------------------|
/// | `U8`             | 1     | `(byte - 128) / 128.0`          |
/// | `I16`            | 2     | `value / 32767.0`               |
/// | `I32`            | 4     | `value / 2147483647.0`          |
/// | `F32`            | 4     | passthrough (no normalization)  |
///
/// `I24` is **not** supported for raw PCM (use the AIFF or AU decoders).
///
/// # Errors
///
/// Returns [`OxiAudioError::UnsupportedFormat`] for `I24` and `F64`,
/// [`OxiAudioError::Io`] on I/O failure.
pub fn decode_raw_pcm<R: Read>(
    reader: &mut R,
    config: &RawPcmConfig,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if config.skip_bytes > 0 {
        skip_bytes(reader, config.skip_bytes)?;
    }

    let samples = match config.format {
        SampleFormat::U8 => {
            let mut out = Vec::new();
            let mut buf = [0u8; 1];
            loop {
                match reader.read_exact(&mut buf) {
                    Ok(()) => {
                        let s = (buf[0] as f32 - 128.0) / 128.0;
                        out.push(s);
                    }
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(OxiAudioError::Io(e)),
                }
            }
            out
        }
        SampleFormat::I16 => {
            let mut out = Vec::new();
            let mut buf = [0u8; 2];
            loop {
                match reader.read_exact(&mut buf) {
                    Ok(()) => {
                        let raw = if config.little_endian {
                            i16::from_le_bytes(buf)
                        } else {
                            i16::from_be_bytes(buf)
                        };
                        let s = raw as f32 / i16::MAX as f32;
                        out.push(s);
                    }
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(OxiAudioError::Io(e)),
                }
            }
            out
        }
        SampleFormat::I32 => {
            let mut out = Vec::new();
            let mut buf = [0u8; 4];
            loop {
                match reader.read_exact(&mut buf) {
                    Ok(()) => {
                        let raw = if config.little_endian {
                            i32::from_le_bytes(buf)
                        } else {
                            i32::from_be_bytes(buf)
                        };
                        let s = raw as f32 / i32::MAX as f32;
                        out.push(s);
                    }
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(OxiAudioError::Io(e)),
                }
            }
            out
        }
        SampleFormat::F32 => {
            let mut out = Vec::new();
            let mut buf = [0u8; 4];
            loop {
                match reader.read_exact(&mut buf) {
                    Ok(()) => {
                        let s = if config.little_endian {
                            f32::from_le_bytes(buf)
                        } else {
                            f32::from_be_bytes(buf)
                        };
                        out.push(s);
                    }
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(OxiAudioError::Io(e)),
                }
            }
            out
        }
        SampleFormat::I24 => {
            return Err(OxiAudioError::UnsupportedFormat(
                "raw PCM decode does not support I24; use decode_aiff or decode_au instead".into(),
            ));
        }
        SampleFormat::F64 => {
            return Err(OxiAudioError::UnsupportedFormat(
                "raw PCM decode does not support F64".into(),
            ));
        }
    };

    let layout = ChannelLayout::from(config.channels);

    Ok(AudioBuffer {
        samples,
        sample_rate: config.sample_rate,
        channels: layout,
        format: SampleFormat::F32,
    })
}

/// Convenience: decode raw PCM from a file at `path` using the provided `config`.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on file open failure, or any error from [`decode_raw_pcm`].
pub fn decode_raw_pcm_file(
    path: &Path,
    config: &RawPcmConfig,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    decode_raw_pcm(&mut reader, config)
}
