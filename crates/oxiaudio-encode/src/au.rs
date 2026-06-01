//! Pure-Rust AU/SND audio encoder.
//!
//! Encodes `AudioBuffer<f32>` to the Sun/NeXT AU format (`.au`, `.snd`).
//! Magic: `.snd` (0x2E736E64).  All fields use big-endian byte order.
//!
//! Supported output encodings:
//! - [`AuEncoding::I16`]: encoding type 3, 16-bit signed big-endian PCM
//! - [`AuEncoding::I24`]: encoding type 4, 24-bit signed big-endian PCM
//! - [`AuEncoding::F32`]: encoding type 6, 32-bit IEEE 754 float big-endian
//!
//! Encoding type numbers follow the Sun/NeXT AU specification:
//!   3 = 16-bit signed linear PCM (big-endian)
//!   4 = 24-bit signed linear PCM (big-endian)
//!   6 = 32-bit IEEE 754 float (big-endian)

use std::io::Write;

use oxiaudio_core::{AudioBuffer, OxiAudioError};

/// Bit depth / sample format selection for AU encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuEncoding {
    /// 16-bit signed linear PCM, big-endian. Best compatibility (AU encoding type 3).
    I16,
    /// 24-bit signed linear PCM, big-endian (AU encoding type 4).
    I24,
    /// 32-bit IEEE 754 float, big-endian (AU encoding type 6).
    F32,
}

/// Encode an `AudioBuffer<f32>` as AU/SND audio to `writer`.
///
/// The output always uses big-endian byte order.  The header is always 24 bytes
/// (no annotation string), so `data_offset` is fixed at 24.
///
/// Samples are interleaved (as stored in `AudioBuffer`) and written in the
/// chosen `encoding` format.
///
/// # Errors
///
/// Returns `OxiAudioError::Io` on any write failure.
pub fn encode_au<W: Write>(
    buf: &AudioBuffer<f32>,
    mut writer: W,
    encoding: AuEncoding,
) -> Result<(), OxiAudioError> {
    let n_samples = buf.samples.len();
    let ch = buf.channels.channel_count();

    let (au_encoding_id, bytes_per_sample): (u32, usize) = match encoding {
        AuEncoding::I16 => (3, 2),
        AuEncoding::I24 => (4, 3),
        AuEncoding::F32 => (6, 4),
    };

    let data_size = (n_samples * bytes_per_sample) as u32;
    let data_offset = 24u32; // fixed header, no annotation

    // ── AU header (24 bytes, all big-endian) ──
    // Magic ".snd" = 0x2E736E64
    writer.write_all(&0x2E736E64u32.to_be_bytes())?;
    writer.write_all(&data_offset.to_be_bytes())?;
    writer.write_all(&data_size.to_be_bytes())?;
    writer.write_all(&au_encoding_id.to_be_bytes())?;
    writer.write_all(&buf.sample_rate.to_be_bytes())?;
    writer.write_all(&(ch as u32).to_be_bytes())?;

    // ── Audio data ──
    match encoding {
        AuEncoding::I16 => {
            for &s in &buf.samples {
                let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                writer.write_all(&v.to_be_bytes())?;
            }
        }
        AuEncoding::I24 => {
            for &s in &buf.samples {
                let raw = (s.clamp(-1.0, 1.0) * 8_388_607.0_f32) as i32;
                // Write big-endian 24-bit: the upper 3 bytes of the sign-extended i32.
                // i32::to_be_bytes() = [byte3, byte2, byte1, byte0]; skip byte3 (index 0)
                // to get bytes [byte2, byte1, byte0] = the 24-bit big-endian value.
                let b = raw.to_be_bytes();
                writer.write_all(&b[1..4])?;
            }
        }
        AuEncoding::F32 => {
            for &s in &buf.samples {
                writer.write_all(&s.to_be_bytes())?;
            }
        }
    }

    Ok(())
}

/// Encode `buf` to a file at `path` using AU/SND format.
///
/// For 16-bit PCM (the most compatible option), pass [`AuEncoding::I16`].
/// For 24-bit or float output, pass [`AuEncoding::I24`] or [`AuEncoding::F32`].
///
/// # Errors
///
/// Returns `OxiAudioError::Io` on file creation or write failure.
pub fn encode_au_file(
    buf: &AudioBuffer<f32>,
    path: impl AsRef<std::path::Path>,
    encoding: AuEncoding,
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path.as_ref()).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    encode_au(buf, writer, encoding)
}

// ─── Internal unit tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    use super::*;

    fn sine_mono(sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        let samples = (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    /// Encode a mono 44100 Hz sine, verify magic bytes are ".snd".
    #[test]
    fn test_au_i16_magic_bytes() {
        let buf = sine_mono(44_100, 0.1);
        let mut out = Cursor::new(Vec::new());
        encode_au(&buf, &mut out, AuEncoding::I16).expect("encode_au I16 failed");
        let bytes = out.into_inner();
        assert_eq!(&bytes[..4], b".snd", "AU magic bytes mismatch");
    }

    /// Encode a stereo 48 kHz buffer and verify sample_rate and channels in the header.
    #[test]
    fn test_au_i16_header_fields() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 1024], // 512 stereo frames
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let mut out = Cursor::new(Vec::new());
        encode_au(&buf, &mut out, AuEncoding::I16).expect("encode_au header test failed");
        let bytes = out.into_inner();

        // data_offset at bytes [4..8] must be 24
        let data_offset = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(data_offset, 24, "data_offset should be 24");

        // encoding type at bytes [12..16] must be 3 (AU I16)
        let enc_id = u32::from_be_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(enc_id, 3, "encoding ID for I16 should be 3");

        // sample_rate at bytes [16..20] must be 48000 big-endian
        let sr = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        assert_eq!(sr, 48_000, "sample_rate field should be 48000");

        // channels at bytes [20..24] must be 2 big-endian
        let ch = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!(ch, 2, "channels field should be 2");
    }

    /// Encode a mono 44100 Hz sine and decode it back, verifying sample_rate and fidelity.
    #[test]
    fn test_au_roundtrip_i16() {
        let buf = sine_mono(44_100, 0.25);
        let tmp = std::env::temp_dir().join("oxiaudio_au_roundtrip_i16.au");

        encode_au_file(&buf, &tmp, AuEncoding::I16).expect("encode_au_file failed");

        let decoded = oxiaudio_decode::decode_au_file(&tmp).expect("decode_au_file failed");
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(
            decoded.sample_rate, buf.sample_rate,
            "sample_rate mismatch after roundtrip"
        );
        assert_eq!(
            decoded.samples.len(),
            buf.samples.len(),
            "sample count mismatch after roundtrip"
        );

        // Find the first non-silent sample and verify it is within 2e-4 (I16 quantisation noise).
        let first_nonsilent = buf
            .samples
            .iter()
            .zip(decoded.samples.iter())
            .find(|(&orig, _)| orig.abs() > 1e-3);
        if let Some((&orig, &dec)) = first_nonsilent {
            assert!(
                (orig - dec).abs() < 2e-4,
                "roundtrip error too large: orig={orig} dec={dec} diff={}",
                (orig - dec).abs()
            );
        }
    }
}
