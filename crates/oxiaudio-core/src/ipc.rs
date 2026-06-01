//! Compact binary IPC serialization for [`AudioBuffer`].
//!
//! # Wire format — `ABUF` v1
//!
//! ```text
//! Offset  Size   Field
//! 0       4      magic  = b"ABUF"
//! 4       1      version = 1
//! 5       1      format  (0=F32)
//! 6       1      channels (0=Mono, 1=Stereo, 255=other)
//! 7       4 LE   sample_rate (u32)
//! 11      8 LE   sample_count (u64) — number of individual samples, NOT frames
//! 19      N*4    samples (little-endian f32)
//! ```
//!
//! Total header: 19 bytes; total size = 19 + `sample_count * 4`.

use std::io::{Read, Write};

use crate::{AudioBuffer, ChannelLayout, OxiAudioError, SampleFormat};

const MAGIC: &[u8; 4] = b"ABUF";
const VERSION: u8 = 1;
const FORMAT_F32: u8 = 0;

// ── helpers ──────────────────────────────────────────────────────────────────

fn channel_to_byte(layout: ChannelLayout) -> u8 {
    match layout {
        ChannelLayout::Mono => 0,
        ChannelLayout::Stereo => 1,
        _ => 255,
    }
}

fn byte_to_channel(byte: u8) -> Result<ChannelLayout, OxiAudioError> {
    match byte {
        0 => Ok(ChannelLayout::Mono),
        1 => Ok(ChannelLayout::Stereo),
        other => Err(OxiAudioError::Decode(format!(
            "unsupported channel layout byte: {other}"
        ))),
    }
}

// ── public API ────────────────────────────────────────────────────────────────

/// Serialize an [`AudioBuffer<f32>`] to a compact binary IPC format.
///
/// The writer receives the full byte stream beginning with the `ABUF` magic.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the underlying write fails.
#[must_use = "discarding the Result ignores serialize errors"]
pub fn serialize_audio_buffer_f32<W: Write>(
    buf: &AudioBuffer<f32>,
    writer: &mut W,
) -> Result<(), OxiAudioError> {
    writer.write_all(MAGIC).map_err(OxiAudioError::Io)?;
    writer.write_all(&[VERSION]).map_err(OxiAudioError::Io)?;
    writer.write_all(&[FORMAT_F32]).map_err(OxiAudioError::Io)?;
    writer
        .write_all(&[channel_to_byte(buf.channels)])
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&buf.sample_rate.to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    let count = buf.samples.len() as u64;
    writer
        .write_all(&count.to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    for &s in &buf.samples {
        writer
            .write_all(&s.to_le_bytes())
            .map_err(OxiAudioError::Io)?;
    }
    Ok(())
}

/// Deserialize an [`AudioBuffer<f32>`] from the compact binary IPC format.
///
/// Reads exactly the number of bytes required by the embedded header.
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] on magic / version / format mismatch,
/// or [`OxiAudioError::Io`] on read failure.
#[must_use = "discarding the Result ignores deserialize errors"]
pub fn deserialize_audio_buffer_f32<R: Read>(
    reader: &mut R,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    // magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(OxiAudioError::Io)?;
    if &magic != MAGIC {
        return Err(OxiAudioError::Decode(
            "invalid magic bytes for AudioBuffer IPC".to_string(),
        ));
    }
    // version + format + channels
    let mut header = [0u8; 3];
    reader.read_exact(&mut header).map_err(OxiAudioError::Io)?;
    if header[0] != VERSION {
        return Err(OxiAudioError::Decode(format!(
            "unsupported AudioBuffer IPC version: {}",
            header[0]
        )));
    }
    if header[1] != FORMAT_F32 {
        return Err(OxiAudioError::Decode(format!(
            "unsupported sample format byte: {} (only F32={FORMAT_F32} supported)",
            header[1]
        )));
    }
    let channels = byte_to_channel(header[2])?;
    // sample_rate
    let mut sr_bytes = [0u8; 4];
    reader
        .read_exact(&mut sr_bytes)
        .map_err(OxiAudioError::Io)?;
    let sample_rate = u32::from_le_bytes(sr_bytes);
    // sample_count
    let mut count_bytes = [0u8; 8];
    reader
        .read_exact(&mut count_bytes)
        .map_err(OxiAudioError::Io)?;
    let count = usize::try_from(u64::from_le_bytes(count_bytes)).map_err(|_| {
        OxiAudioError::Decode("sample count exceeds platform address space".to_string())
    })?;
    // samples
    let mut samples = Vec::with_capacity(count);
    let mut sample_buf = [0u8; 4];
    for _ in 0..count {
        reader
            .read_exact(&mut sample_buf)
            .map_err(OxiAudioError::Io)?;
        samples.push(f32::from_le_bytes(sample_buf));
    }
    Ok(AudioBuffer {
        samples,
        sample_rate,
        channels,
        format: SampleFormat::F32,
    })
}

/// Convenience: serialize an [`AudioBuffer<f32>`] into a new `Vec<u8>`.
///
/// # Errors
///
/// Propagates any [`OxiAudioError::Io`] from the in-memory write (unlikely in practice).
#[must_use = "discarding the Result ignores serialize errors"]
pub fn to_ipc_bytes(buf: &AudioBuffer<f32>) -> Result<Vec<u8>, OxiAudioError> {
    // Header = 4 + 1 + 1 + 1 + 4 + 8 = 19 bytes; each sample = 4 bytes.
    let mut out = Vec::with_capacity(buf.samples.len() * 4 + 19);
    serialize_audio_buffer_f32(buf, &mut out)?;
    Ok(out)
}

/// Convenience: deserialize an [`AudioBuffer<f32>`] from a byte slice.
///
/// # Errors
///
/// Propagates any error from [`deserialize_audio_buffer_f32`].
#[must_use = "discarding the Result ignores deserialize errors"]
pub fn from_ipc_bytes(data: &[u8]) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let mut cursor = std::io::Cursor::new(data);
    deserialize_audio_buffer_f32(&mut cursor)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod prop_tests {
    use super::*;
    use crate::{AudioBuffer, ChannelLayout, SampleFormat};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_ipc_roundtrip_mono(
            samples in proptest::collection::vec(-1.0f32..=1.0f32, 0..1000),
            sample_rate in 8000u32..=192000u32,
        ) {
            let buf = AudioBuffer {
                samples: samples.clone(),
                sample_rate,
                channels: ChannelLayout::Mono,
                format: SampleFormat::F32,
            };
            let bytes = to_ipc_bytes(&buf).expect("serialize");
            let decoded = from_ipc_bytes(&bytes).expect("deserialize");
            prop_assert_eq!(decoded.sample_rate, sample_rate);
            prop_assert_eq!(decoded.channels, ChannelLayout::Mono);
            prop_assert_eq!(decoded.samples.len(), samples.len());
            for (a, b) in decoded.samples.iter().zip(samples.iter()) {
                prop_assert!((a - b).abs() < 1e-6, "sample mismatch: {a} vs {b}");
            }
        }

        #[test]
        fn prop_ipc_roundtrip_stereo(
            samples in proptest::collection::vec(-1.0f32..=1.0f32, 0..2000),
            sample_rate in 8000u32..=192000u32,
        ) {
            let buf = AudioBuffer {
                samples: samples.clone(),
                sample_rate,
                channels: ChannelLayout::Stereo,
                format: SampleFormat::F32,
            };
            let bytes = to_ipc_bytes(&buf).expect("serialize");
            let decoded = from_ipc_bytes(&bytes).expect("deserialize");
            prop_assert_eq!(decoded.sample_rate, sample_rate);
            prop_assert_eq!(decoded.channels, ChannelLayout::Stereo);
            prop_assert_eq!(decoded.samples.len(), samples.len());
        }

        #[test]
        fn prop_ipc_byte_count(
            n_samples in 0usize..500,
            sample_rate in 8000u32..=48000u32,
        ) {
            let buf = AudioBuffer {
                samples: vec![0.0f32; n_samples],
                sample_rate,
                channels: ChannelLayout::Mono,
                format: SampleFormat::F32,
            };
            let bytes = to_ipc_bytes(&buf).expect("serialize");
            // Header: 4(magic) + 1(version) + 1(format) + 1(channels) + 4(rate) + 8(count) = 19
            // Samples: N * 4 bytes
            let expected_len = 19 + n_samples * 4;
            prop_assert_eq!(bytes.len(), expected_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioBuffer, ChannelLayout, SampleFormat};

    fn make_mono(samples: Vec<f32>) -> AudioBuffer<f32> {
        AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn make_stereo(samples: Vec<f32>) -> AudioBuffer<f32> {
        AudioBuffer {
            samples,
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_ipc_roundtrip_mono() {
        let buf = make_mono(vec![0.0, 0.5, -0.5, 1.0, -1.0]);
        let bytes = to_ipc_bytes(&buf).unwrap();
        let decoded = from_ipc_bytes(&bytes).unwrap();
        assert_eq!(decoded.sample_rate, buf.sample_rate);
        assert_eq!(decoded.channels, buf.channels);
        assert_eq!(decoded.format, buf.format);
        assert_eq!(decoded.samples.len(), buf.samples.len());
        for (a, b) in buf.samples.iter().zip(decoded.samples.iter()) {
            assert!((a - b).abs() < 1e-7, "sample mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_ipc_roundtrip_stereo() {
        let samples: Vec<f32> = (0..20).map(|i| i as f32 / 20.0).collect();
        let buf = make_stereo(samples);
        let bytes = to_ipc_bytes(&buf).unwrap();
        let decoded = from_ipc_bytes(&bytes).unwrap();
        assert_eq!(decoded.sample_rate, 48_000);
        assert_eq!(decoded.channels, ChannelLayout::Stereo);
        assert_eq!(decoded.samples, buf.samples);
    }

    #[test]
    fn test_ipc_magic_check() {
        // Corrupt the magic bytes.
        let buf = make_mono(vec![0.0]);
        let mut bytes = to_ipc_bytes(&buf).unwrap();
        bytes[0] = b'X';
        match from_ipc_bytes(&bytes) {
            Ok(_) => panic!("should fail on bad magic"),
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("magic"), "error should mention magic: {msg}");
            }
        }
    }

    #[test]
    fn test_ipc_version_check() {
        // Tamper with the version byte (offset 4).
        let buf = make_mono(vec![0.0]);
        let mut bytes = to_ipc_bytes(&buf).unwrap();
        bytes[4] = 99;
        match from_ipc_bytes(&bytes) {
            Ok(_) => panic!("should fail on unsupported version"),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("version"),
                    "error should mention version: {msg}"
                );
            }
        }
    }

    #[test]
    fn test_ipc_empty_buffer() {
        let buf = make_mono(vec![]);
        let bytes = to_ipc_bytes(&buf).unwrap();
        let decoded = from_ipc_bytes(&bytes).unwrap();
        assert_eq!(decoded.samples.len(), 0);
        assert_eq!(decoded.sample_rate, 44_100);
        assert_eq!(decoded.channels, ChannelLayout::Mono);
    }

    #[test]
    fn test_ipc_byte_count() {
        // Header = 4 (magic) + 1 (version) + 1 (format) + 1 (channels) + 4 (sr) + 8 (count) = 19
        // Each sample = 4 bytes.
        let n = 7usize;
        let buf = make_mono(vec![0.0f32; n]);
        let bytes = to_ipc_bytes(&buf).unwrap();
        let expected = 19 + n * 4;
        assert_eq!(
            bytes.len(),
            expected,
            "expected {expected} bytes, got {}",
            bytes.len()
        );
    }
}
