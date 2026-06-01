//! Pure-Rust AU/SND audio decoder.
//!
//! Supports the Sun/NeXT AU format (`.au`, `.snd`).
//! Magic: `.snd` (0x2E736E64).
//!
//! Supported encodings (per Sun/NeXT AU specification):
//! - 3: 16-bit signed big-endian PCM
//! - 4: 24-bit signed big-endian PCM
//! - 6: 32-bit IEEE 754 float big-endian

use std::io::{self, Read};
use std::path::Path;

use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError, SampleFormat};

/// AU encoding type constants (Sun/NeXT AU specification).
///
/// Per the AU spec:
///   2 = 8-bit linear PCM (unsigned)
///   3 = 16-bit signed linear PCM (big-endian)
///   4 = 24-bit signed linear PCM (big-endian)
///   5 = 32-bit signed linear PCM (big-endian)
///   6 = 32-bit IEEE 754 float (big-endian)
///   7 = 64-bit IEEE 754 float (big-endian)
const AU_ENCODING_I16: u32 = 3;
const AU_ENCODING_I24: u32 = 4;
const AU_ENCODING_F32: u32 = 6;

/// AU file header: all fields are big-endian.
struct AuHeader {
    /// Byte offset from the start of the file to the audio data.
    _data_offset: u32,
    /// Encoding type (1–6).
    encoding: u32,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Number of channels.
    channels: u32,
}

/// Read a 4-byte big-endian u32.
#[inline]
fn read_u32_be<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

/// Skip exactly `n` bytes by draining via a 512-byte scratch buffer.
fn skip_bytes<R: Read>(reader: &mut R, mut n: u64) -> io::Result<()> {
    let mut scratch = [0u8; 512];
    while n > 0 {
        let to_read = n.min(scratch.len() as u64) as usize;
        reader.read_exact(&mut scratch[..to_read])?;
        n -= to_read as u64;
    }
    Ok(())
}

/// Parse the AU file header.
///
/// Leaves the reader positioned at the start of the audio data
/// (i.e. `data_offset` bytes from the start of the stream).
fn parse_header<R: Read>(reader: &mut R) -> Result<AuHeader, OxiAudioError> {
    // Magic: ".snd" == 0x2E736E64
    let magic = read_u32_be(reader)?;
    if magic != 0x2E736E64 {
        return Err(OxiAudioError::Decode(format!(
            "AU: invalid magic 0x{magic:08X} (expected 0x2E736E64)"
        )));
    }

    let data_offset = read_u32_be(reader)?;
    let _data_size = read_u32_be(reader)?; // may be 0xFFFFFFFF if unknown — ignored
    let encoding = read_u32_be(reader)?;
    let sample_rate = read_u32_be(reader)?;
    let channels = read_u32_be(reader)?;

    // The fixed header is 24 bytes. Any bytes between byte 24 and `data_offset`
    // are an optional annotation string — skip them via Read (no Seek required).
    if data_offset < 24 {
        return Err(OxiAudioError::Decode(format!(
            "AU: data_offset {data_offset} is less than minimum header size (24)"
        )));
    }
    let annotation_bytes = data_offset as u64 - 24;
    if annotation_bytes > 0 {
        skip_bytes(reader, annotation_bytes)?;
    }

    Ok(AuHeader {
        _data_offset: data_offset,
        encoding,
        sample_rate,
        channels,
    })
}

/// Convert a 3-byte big-endian byte sequence to a sign-extended `i32`.
#[inline]
fn i24_be_to_i32(b: &[u8; 3]) -> i32 {
    ((b[0] as i32) << 24 | (b[1] as i32) << 16 | (b[2] as i32) << 8) >> 8
}

/// Decode AU audio data from `reader`.
///
/// Supports encodings 3 (i16 BE), 4 (i24 BE), and 6 (f32 BE) per the Sun/NeXT AU spec.
/// Returns [`OxiAudioError::UnsupportedFormat`] for all other encodings.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on I/O failure, malformed header, or unsupported encoding.
pub fn decode_au<R: Read>(reader: &mut R) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let header = parse_header(reader)?;

    if header.channels == 0 {
        return Err(OxiAudioError::Decode(
            "AU: header reports 0 channels".into(),
        ));
    }
    if header.sample_rate == 0 {
        return Err(OxiAudioError::Decode(
            "AU: header reports 0 sample_rate".into(),
        ));
    }

    let samples: Vec<f32> = match header.encoding {
        AU_ENCODING_I16 => {
            let mut samples = Vec::new();
            let mut tmp = [0u8; 2];
            loop {
                match reader.read_exact(&mut tmp) {
                    Ok(()) => {
                        let s = i16::from_be_bytes(tmp) as f32 / i16::MAX as f32;
                        samples.push(s);
                    }
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(OxiAudioError::Io(e)),
                }
            }
            samples
        }
        AU_ENCODING_I24 => {
            let mut samples = Vec::new();
            let mut tmp = [0u8; 3];
            loop {
                match reader.read_exact(&mut tmp) {
                    Ok(()) => {
                        let raw = i24_be_to_i32(&tmp);
                        let s = raw as f32 / 8_388_607.0_f32;
                        samples.push(s);
                    }
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(OxiAudioError::Io(e)),
                }
            }
            samples
        }
        AU_ENCODING_F32 => {
            let mut samples = Vec::new();
            let mut tmp = [0u8; 4];
            loop {
                match reader.read_exact(&mut tmp) {
                    Ok(()) => {
                        let s = f32::from_be_bytes(tmp);
                        samples.push(s);
                    }
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(OxiAudioError::Io(e)),
                }
            }
            samples
        }
        enc => {
            return Err(OxiAudioError::UnsupportedFormat(format!(
                "AU encoding {enc} is not supported (supported: 3=i16, 4=i24, 6=f32)"
            )));
        }
    };

    let layout = ChannelLayout::from(header.channels as u16);

    Ok(AudioBuffer {
        samples,
        sample_rate: header.sample_rate,
        channels: layout,
        format: SampleFormat::F32,
    })
}

/// Convenience: decode an AU file at `path` to `AudioBuffer<f32>`.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on file open failure, or any error from [`decode_au`].
pub fn decode_au_file(path: &Path) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    decode_au(&mut reader)
}
