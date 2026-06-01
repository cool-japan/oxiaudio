//! Pure-Rust AIFF/AIFF-C decoder.
//!
//! Supports AIFF (not AIFF-C) with 8-bit unsigned, 16-bit signed big-endian,
//! and 24-bit signed big-endian sample data.
//!
//! Supports AIFF-C with µ-law (G.711 µ-law) and A-law (G.711 A-law) compressed
//! sample data via [`decode_aiffc_compressed`] and [`decode_aiffc_compressed_file`].
//!
//! Text metadata chunks (`NAME`, `AUTH`, `ANNO`) are decoded when using
//! [`decode_aiff_with_metadata`].

use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use oxiaudio_core::{AudioBuffer, AudioMetadata, ChannelLayout, OxiAudioError, SampleFormat};

/// AIFF COMM chunk: the only mandatory chunk describing the audio parameters.
struct CommChunk {
    num_channels: u16,
    num_frames: u32,
    bit_depth: u16,
    sample_rate: u32,
}

/// AIFF SSND chunk location within the reader.
struct SsndInfo {
    /// Byte offset from the start of the chunk data (after "SSND" + size) to PCM data.
    data_start: u64,
}

/// Decode an 80-bit IEEE 754 extended-precision float (big-endian, 10 bytes).
///
/// This is the sample-rate encoding used by AIFF.
fn extended_to_f64(bytes: &[u8; 10]) -> f64 {
    let sign = if bytes[0] & 0x80 != 0 {
        -1.0_f64
    } else {
        1.0_f64
    };
    let exp = (((bytes[0] & 0x7F) as i32) << 8 | bytes[1] as i32) - 16383;
    let mantissa_bytes: [u8; 8] = bytes[2..10].try_into().unwrap_or([0u8; 8]);
    let mantissa = u64::from_be_bytes(mantissa_bytes);
    sign * (mantissa as f64) * 2.0_f64.powi(exp - 63)
}

/// Read a 4-byte chunk ID and its 32-bit big-endian size.
/// Returns `(id, size)`.
fn read_chunk_header<R: Read>(reader: &mut R) -> io::Result<([u8; 4], u32)> {
    let mut id = [0u8; 4];
    reader.read_exact(&mut id)?;
    let mut sz = [0u8; 4];
    reader.read_exact(&mut sz)?;
    Ok((id, u32::from_be_bytes(sz)))
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

/// Read exactly `size` bytes of a text chunk payload and return it as a UTF-8 `String`.
///
/// Trailing NUL bytes are stripped. The AIFF spec states chunks are padded to even byte
/// boundaries; if `size` is odd, one extra pad byte is consumed (best-effort, since the
/// pad byte may not be present at EOF).
fn read_text_chunk<R: Read>(reader: &mut R, size: u32) -> io::Result<String> {
    let mut buf = vec![0u8; size as usize];
    reader.read_exact(&mut buf)?;

    // Consume pad byte if the declared payload size is odd (best-effort: ignore EOF).
    if size % 2 != 0 {
        let mut pad = [0u8; 1];
        let _ = reader.read(&mut pad);
    }

    // Strip trailing NUL bytes then convert lossily to UTF-8.
    let trimmed = buf
        .iter()
        .rposition(|&b| b != 0)
        .map_or(&b""[..], |last| &buf[..=last]);
    Ok(String::from_utf8_lossy(trimmed).into_owned())
}

/// Parse the COMM chunk payload (without the 4-byte ID and size already consumed).
fn parse_comm<R: Read>(reader: &mut R, comm_size: u32) -> Result<CommChunk, OxiAudioError> {
    // Minimum COMM payload: 2 + 4 + 2 + 10 = 18 bytes.
    if comm_size < 18 {
        return Err(OxiAudioError::Decode(format!(
            "AIFF COMM chunk too small: {comm_size} bytes (minimum 18)"
        )));
    }

    let mut tmp2 = [0u8; 2];
    let mut tmp4 = [0u8; 4];
    let mut tmp10 = [0u8; 10];

    reader.read_exact(&mut tmp2)?;
    let num_channels = i16::from_be_bytes(tmp2) as u16;

    reader.read_exact(&mut tmp4)?;
    let num_frames = u32::from_be_bytes(tmp4);

    reader.read_exact(&mut tmp2)?;
    let bit_depth = i16::from_be_bytes(tmp2) as u16;

    reader.read_exact(&mut tmp10)?;
    let sample_rate_f64 = extended_to_f64(&tmp10);
    let sample_rate = sample_rate_f64.round() as u32;

    // Skip any extra bytes (e.g. AIFF-C compression type).
    let extra = comm_size as u64 - 18;
    if extra > 0 {
        skip_bytes(reader, extra)?;
    }

    Ok(CommChunk {
        num_channels,
        num_frames,
        bit_depth,
        sample_rate,
    })
}

/// Convert a 3-byte big-endian sequence to a sign-extended `i32`.
#[inline]
fn i24_be_to_i32(b: &[u8; 3]) -> i32 {
    // Pack into the upper 24 bits of i32 then arithmetic-shift right by 8.
    ((b[0] as i32) << 24 | (b[1] as i32) << 16 | (b[2] as i32) << 8) >> 8
}

/// Decode AIFF PCM data from `reader` given the parsed COMM and SSND metadata.
///
/// PCM data is already positioned at the start of the SSND payload
/// (after the 8-byte offset/blockAlign header has been consumed).
fn decode_pcm_samples<R: Read>(
    reader: &mut R,
    comm: &CommChunk,
) -> Result<Vec<f32>, OxiAudioError> {
    let total_samples = comm.num_frames as usize * comm.num_channels as usize;
    let mut samples = Vec::with_capacity(total_samples);

    match comm.bit_depth {
        8 => {
            // 8-bit unsigned PCM; bias at 128.
            let mut buf = vec![0u8; total_samples];
            reader.read_exact(&mut buf)?;
            for byte in buf {
                let s = (byte as f32 - 128.0) / 128.0;
                samples.push(s);
            }
        }
        16 => {
            let mut tmp = [0u8; 2];
            for _ in 0..total_samples {
                reader.read_exact(&mut tmp)?;
                let s = i16::from_be_bytes(tmp) as f32 / i16::MAX as f32;
                samples.push(s);
            }
        }
        24 => {
            let mut tmp = [0u8; 3];
            for _ in 0..total_samples {
                reader.read_exact(&mut tmp)?;
                let raw = i24_be_to_i32(&tmp);
                let s = raw as f32 / 8_388_607.0_f32;
                samples.push(s);
            }
        }
        d => {
            return Err(OxiAudioError::UnsupportedFormat(format!(
                "AIFF bit depth {d} is not supported (supported: 8, 16, 24)"
            )));
        }
    }

    Ok(samples)
}

/// Internal decode implementation that reads all chunks and returns decoded audio + metadata.
///
/// Scans the full FORM/AIFF container, collecting `COMM`, `SSND`, `NAME`, `AUTH`, and
/// `ANNO` chunks. The early-exit optimisation is intentionally omitted so text chunks
/// appearing anywhere in the container (including after `SSND`) are captured.
fn decode_aiff_inner<R: Read + Seek>(
    reader: &mut R,
) -> Result<(AudioBuffer<f32>, AudioMetadata), OxiAudioError> {
    // ── FORM header ────────────────────────────────────────────────────────────
    let mut form_id = [0u8; 4];
    reader.read_exact(&mut form_id)?;
    if &form_id != b"FORM" {
        return Err(OxiAudioError::Decode(format!(
            "AIFF: expected 'FORM' at offset 0, got {:?}",
            form_id
        )));
    }

    let mut form_size_bytes = [0u8; 4];
    reader.read_exact(&mut form_size_bytes)?;
    // We don't validate total size here — we just parse chunks until we have what we need.
    let _ = u32::from_be_bytes(form_size_bytes);

    let mut form_type = [0u8; 4];
    reader.read_exact(&mut form_type)?;
    if &form_type != b"AIFF" {
        return Err(OxiAudioError::Decode(format!(
            "AIFF: expected 'AIFF' form type, got {:?}",
            form_type
        )));
    }

    // ── Chunk scan ─────────────────────────────────────────────────────────────
    let mut comm: Option<CommChunk> = None;
    let mut ssnd: Option<SsndInfo> = None;

    // Metadata accumulators.
    let mut title: Option<String> = None;
    let mut artist: Option<String> = None;
    let mut anno_parts: Vec<String> = Vec::new();

    loop {
        let (chunk_id, chunk_size) = match read_chunk_header(reader) {
            Ok(h) => h,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(OxiAudioError::Io(e)),
        };

        match &chunk_id {
            b"COMM" => {
                comm = Some(parse_comm(reader, chunk_size)?);
            }
            b"SSND" => {
                // SSND layout:  offset(u32 BE) + blockAlign(u32 BE) + PCM data
                let mut hdr8 = [0u8; 8];
                reader.read_exact(&mut hdr8)?;
                let ssnd_offset = u32::from_be_bytes(hdr8[..4].try_into().unwrap_or([0u8; 4]));
                // Skip `ssnd_offset` additional bytes if non-zero.
                if ssnd_offset > 0 {
                    skip_bytes(reader, ssnd_offset as u64)?;
                }
                // Record the current stream position (== start of PCM data).
                let pcm_start = reader.stream_position().map_err(OxiAudioError::Io)?;
                ssnd = Some(SsndInfo {
                    data_start: pcm_start,
                });
                // Consume the rest of the SSND chunk so we can continue scanning.
                // PCM size = chunk_size - 8 (offset/blockAlign header) - ssnd_offset.
                let already_consumed = 8u64 + ssnd_offset as u64;
                let remaining = (chunk_size as u64).saturating_sub(already_consumed);
                skip_bytes(reader, remaining)?;
            }
            b"NAME" => {
                // NAME chunk: title string, size bytes of text.
                let text = read_text_chunk(reader, chunk_size).map_err(OxiAudioError::Io)?;
                title = Some(text);
            }
            b"AUTH" => {
                // AUTH chunk: author/artist string.
                let text = read_text_chunk(reader, chunk_size).map_err(OxiAudioError::Io)?;
                artist = Some(text);
            }
            b"ANNO" => {
                // ANNO chunk: annotation; multiple ANNO chunks may appear.
                let text = read_text_chunk(reader, chunk_size).map_err(OxiAudioError::Io)?;
                anno_parts.push(text);
            }
            _ => {
                // Unknown/unneeded chunk — skip it.
                // AIFF chunks are padded to even byte boundaries.
                // The pad byte is NOT included in `chunk_size`; add it manually.
                let padded = chunk_size as u64 + (chunk_size as u64 & 1);
                skip_bytes(reader, padded)?;
            }
        }
    }

    let comm = comm.ok_or_else(|| OxiAudioError::Decode("AIFF: missing COMM chunk".into()))?;
    let ssnd = ssnd.ok_or_else(|| OxiAudioError::Decode("AIFF: missing SSND chunk".into()))?;

    // Validate channel count
    if comm.num_channels == 0 {
        return Err(OxiAudioError::Decode(
            "AIFF: COMM reports 0 channels".into(),
        ));
    }

    // ── Decode PCM ─────────────────────────────────────────────────────────────
    reader
        .seek(SeekFrom::Start(ssnd.data_start))
        .map_err(OxiAudioError::Io)?;

    let samples = decode_pcm_samples(reader, &comm)?;

    let layout = ChannelLayout::from(comm.num_channels);

    let audio_buf = AudioBuffer {
        samples,
        sample_rate: comm.sample_rate,
        channels: layout,
        format: SampleFormat::F32,
    };

    // Collate metadata: join multiple ANNO chunks with newline.
    let comment = if anno_parts.is_empty() {
        None
    } else {
        Some(anno_parts.join("\n"))
    };

    let metadata = AudioMetadata {
        title,
        artist,
        comment,
        ..AudioMetadata::default()
    };

    Ok((audio_buf, metadata))
}

/// Decode an AIFF file from a reader to `AudioBuffer<f32>`.
///
/// Supports 8-bit unsigned, 16-bit signed big-endian, and 24-bit signed big-endian
/// sample data. Requires both `COMM` and `SSND` chunks in the container.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on I/O failure, malformed headers, or unsupported bit depth.
pub fn decode_aiff<R: Read + Seek>(reader: &mut R) -> Result<AudioBuffer<f32>, OxiAudioError> {
    decode_aiff_inner(reader).map(|(buf, _meta)| buf)
}

/// Decode an AIFF stream from a reader, returning audio samples and embedded text metadata.
///
/// In addition to all audio data decoded by [`decode_aiff`], this function extracts:
/// - `NAME` chunk → [`AudioMetadata::title`]
/// - `AUTH` chunk → [`AudioMetadata::artist`]
/// - `ANNO` chunk(s) → [`AudioMetadata::comment`] (multiple `ANNO` chunks joined with `\n`)
///
/// # Errors
///
/// Returns [`OxiAudioError`] on I/O failure, malformed headers, or unsupported bit depth.
pub fn decode_aiff_reader_with_metadata<R: Read + Seek>(
    reader: &mut R,
) -> Result<(AudioBuffer<f32>, AudioMetadata), OxiAudioError> {
    decode_aiff_inner(reader)
}

/// Convenience: decode an AIFF file at `path` to `AudioBuffer<f32>`.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on file open failure, or any error from [`decode_aiff`].
pub fn decode_aiff_file(path: &Path) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let mut file = std::fs::File::open(path)?;
    decode_aiff(&mut file)
}

/// Decode an AIFF file at `path`, returning audio samples and embedded metadata.
///
/// Extracts text metadata chunks:
/// - `NAME` chunk → [`AudioMetadata::title`]
/// - `AUTH` chunk → [`AudioMetadata::artist`]
/// - `ANNO` chunk(s) → [`AudioMetadata::comment`] (multiple chunks joined with `\n`)
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on file open failure, or [`OxiAudioError::Decode`]
/// on malformed AIFF data or unsupported bit depth.
pub fn decode_aiff_with_metadata(
    path: &Path,
) -> Result<(AudioBuffer<f32>, AudioMetadata), OxiAudioError> {
    let mut file = std::fs::File::open(path)?;
    decode_aiff_inner(&mut file)
}

// ── AIFF-C µ-law / A-law support ────────────────────────────────────────────

/// Decode a single G.711 µ-law (mu-law) byte to a 16-bit signed linear PCM sample.
///
/// Implements the ITU G.711 µ-law decompression algorithm.
/// Each 8-bit µ-law codeword maps to a 16-bit linear PCM value.
pub fn ulaw_to_linear(byte: u8) -> i16 {
    // Invert all bits (µ-law encoding inverts all bits).
    let byte = !byte;
    let sign = (byte & 0x80) != 0;
    let exponent = (byte >> 4) & 0x07;
    let mantissa = byte & 0x0F;
    // The max value here is ((15+16) << (7+3)) - 144 = (31 << 10) - 144 = 31_744 - 144 = 31_600
    // which fits in i16 (max 32_767), so the truncation is safe.
    #[allow(clippy::cast_possible_truncation)]
    let magnitude = (((mantissa as i32 + 16) << (exponent + 3)) - 144) as i16;
    if sign {
        -magnitude
    } else {
        magnitude
    }
}

/// Decode a single G.711 A-law byte to a 16-bit signed linear PCM sample.
///
/// Implements the ITU G.711 A-law decompression algorithm.
/// Each 8-bit A-law codeword maps to a 16-bit linear PCM value.
pub fn alaw_to_linear(byte: u8) -> i16 {
    let byte = byte ^ 0x55;
    let sign = (byte & 0x80) != 0;
    let segment = (byte >> 4) & 0x07;
    let mantissa = (byte & 0x0F) as i32;
    // Compute magnitude in i32 to avoid overflow during shift.
    let magnitude_i32: i32 = if segment == 0 {
        (mantissa << 1) + 1
    } else {
        ((mantissa + 16) << segment) + (1 << segment) - 1
    };
    // Scale by 8 (<<3). Max: segment=7 → ((15+16)<<7)+(128-1) = 3968+127 = 4095; *8 = 32_760 < 32_767.
    #[allow(clippy::cast_possible_truncation)]
    let magnitude = (magnitude_i32 << 3) as i16;
    if sign {
        magnitude
    } else {
        -magnitude
    }
}

/// AIFF-C COMM chunk including the compressionType field.
struct AifcCommChunk {
    num_channels: u16,
    num_frames: u32,
    sample_rate: u32,
    /// 4-byte compression type OSType (e.g. `b"ulaw"`, `b"alaw"`, `b"NONE"`).
    compression_type: [u8; 4],
}

/// Parse an AIFC COMM chunk payload (without the 4-byte ID and size already consumed).
///
/// AIFC COMM minimum size is 22 bytes (18 base + 4 compressionType).
/// A pascal-string compressionName follows, which we skip.
fn parse_aifc_comm<R: Read>(
    reader: &mut R,
    comm_size: u32,
) -> Result<AifcCommChunk, OxiAudioError> {
    // Minimum AIFC COMM payload: 18 (base AIFF) + 4 (compressionType) = 22 bytes.
    if comm_size < 22 {
        return Err(OxiAudioError::Decode(format!(
            "AIFF-C COMM chunk too small: {comm_size} bytes (minimum 22 for AIFC)"
        )));
    }

    let mut tmp2 = [0u8; 2];
    let mut tmp4 = [0u8; 4];
    let mut tmp10 = [0u8; 10];

    reader.read_exact(&mut tmp2)?;
    let num_channels = i16::from_be_bytes(tmp2) as u16;

    reader.read_exact(&mut tmp4)?;
    let num_frames = u32::from_be_bytes(tmp4);

    // sampleSize field (bit depth) — not used for µ-law/A-law (always 8-bit per spec).
    reader.read_exact(&mut tmp2)?;
    let _bit_depth = i16::from_be_bytes(tmp2) as u16;

    reader.read_exact(&mut tmp10)?;
    let sample_rate_f64 = extended_to_f64(&tmp10);
    let sample_rate = sample_rate_f64.round() as u32;

    // Read 4-byte compressionType OSType.
    let mut compression_type = [0u8; 4];
    reader.read_exact(&mut compression_type)?;

    // Skip remaining bytes (pascal-string compressionName + any padding).
    let consumed: u64 = 18 + 4; // 22 bytes parsed above
    let extra = comm_size as u64 - consumed;
    if extra > 0 {
        skip_bytes(reader, extra)?;
    }

    Ok(AifcCommChunk {
        num_channels,
        num_frames,
        sample_rate,
        compression_type,
    })
}

/// Decode an AIFF-C file with µ-law or A-law encoding.
///
/// Supports `compressionType` `"ulaw"` (G.711 µ-law) and `"alaw"` (G.711 A-law).
/// Both lowercase (`b"ulaw"`, `b"alaw"`) and uppercase (`b"ULAW"`, `b"ALAW"`) are
/// accepted for maximum compatibility with real-world files.
///
/// Returns `Err` for other AIFF-C compression types or non-AIFC containers.
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] if the container is not `FORM/AIFC`, the COMM
/// chunk is missing, the `compressionType` is not µ-law or A-law, or the audio data
/// chunk is absent.
pub fn decode_aiffc_compressed<R: Read + Seek>(
    reader: &mut R,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    // ── FORM header ────────────────────────────────────────────────────────────
    let mut form_id = [0u8; 4];
    reader.read_exact(&mut form_id)?;
    if &form_id != b"FORM" {
        return Err(OxiAudioError::Decode(format!(
            "AIFF-C: expected 'FORM' at offset 0, got {:?}",
            form_id
        )));
    }

    // Skip the 4-byte FORM size (we scan chunks until EOF).
    let mut _form_size = [0u8; 4];
    reader.read_exact(&mut _form_size)?;

    let mut form_type = [0u8; 4];
    reader.read_exact(&mut form_type)?;
    if &form_type != b"AIFC" {
        return Err(OxiAudioError::Decode(format!(
            "AIFF-C: expected 'AIFC' form type, got {:?} (use decode_aiff for plain AIFF)",
            form_type
        )));
    }

    // ── Chunk scan ─────────────────────────────────────────────────────────────
    let mut comm: Option<AifcCommChunk> = None;
    let mut ssnd_start: Option<u64> = None;

    loop {
        let (chunk_id, chunk_size) = match read_chunk_header(reader) {
            Ok(h) => h,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(OxiAudioError::Io(e)),
        };

        match &chunk_id {
            b"COMM" => {
                comm = Some(parse_aifc_comm(reader, chunk_size)?);
            }
            b"SSND" => {
                // SSND layout:  offset(u32 BE) + blockSize(u32 BE) + audio data
                let mut hdr8 = [0u8; 8];
                reader.read_exact(&mut hdr8)?;
                let ssnd_offset = u32::from_be_bytes(hdr8[..4].try_into().unwrap_or([0u8; 4]));
                // Skip ssnd_offset additional bytes if non-zero.
                if ssnd_offset > 0 {
                    skip_bytes(reader, ssnd_offset as u64)?;
                }
                // Record the stream position (start of audio data).
                let audio_start = reader.stream_position().map_err(OxiAudioError::Io)?;
                ssnd_start = Some(audio_start);
                // Consume the rest of the SSND chunk so we can keep scanning.
                let already_consumed = 8u64 + ssnd_offset as u64;
                let remaining_chunk = (chunk_size as u64).saturating_sub(already_consumed);
                skip_bytes(reader, remaining_chunk)?;
            }
            _ => {
                // Unknown chunk — skip with even-byte padding.
                let padded = chunk_size as u64 + (chunk_size as u64 & 1);
                skip_bytes(reader, padded)?;
            }
        }
    }

    let comm = comm.ok_or_else(|| OxiAudioError::Decode("AIFF-C: missing COMM chunk".into()))?;
    let ssnd_start =
        ssnd_start.ok_or_else(|| OxiAudioError::Decode("AIFF-C: missing SSND chunk".into()))?;

    if comm.num_channels == 0 {
        return Err(OxiAudioError::Decode(
            "AIFF-C: COMM reports 0 channels".into(),
        ));
    }
    if comm.sample_rate == 0 {
        return Err(OxiAudioError::Decode(
            "AIFF-C: COMM reports 0 sample rate".into(),
        ));
    }

    // Determine compression type (accept both lower and upper case per Apple spec).
    let ct_lower: [u8; 4] = [
        comm.compression_type[0].to_ascii_lowercase(),
        comm.compression_type[1].to_ascii_lowercase(),
        comm.compression_type[2].to_ascii_lowercase(),
        comm.compression_type[3].to_ascii_lowercase(),
    ];

    let use_ulaw = ct_lower == *b"ulaw";
    let use_alaw = ct_lower == *b"alaw";

    if !use_ulaw && !use_alaw {
        return Err(OxiAudioError::UnsupportedFormat(format!(
            "AIFF-C compressionType {:?} is not supported (supported: ulaw, alaw)",
            comm.compression_type
        )));
    }

    // Seek to audio data and decode.
    reader
        .seek(SeekFrom::Start(ssnd_start))
        .map_err(OxiAudioError::Io)?;

    let total_samples = comm.num_frames as usize * comm.num_channels as usize;
    let mut samples = Vec::with_capacity(total_samples);

    let mut byte_buf = [0u8; 1];
    for _ in 0..total_samples {
        match reader.read_exact(&mut byte_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(OxiAudioError::Io(e)),
        }
        let s = if use_ulaw {
            ulaw_to_linear(byte_buf[0]) as f32 / 32768.0
        } else {
            alaw_to_linear(byte_buf[0]) as f32 / 32768.0
        };
        samples.push(s);
    }

    let layout = ChannelLayout::from(comm.num_channels);

    Ok(AudioBuffer {
        samples,
        sample_rate: comm.sample_rate,
        channels: layout,
        format: SampleFormat::F32,
    })
}

/// Convenience: decode an AIFF-C file at `path` with µ-law or A-law encoding.
///
/// Opens the file and delegates to [`decode_aiffc_compressed`].
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on file open failure, or any error from
/// [`decode_aiffc_compressed`].
pub fn decode_aiffc_compressed_file(path: &Path) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    let mut reader = std::io::BufReader::new(file);
    decode_aiffc_compressed(&mut reader)
}

#[cfg(test)]
mod aiffc_tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal AIFF-C byte stream with the given compressionType and raw audio bytes.
    ///
    /// Produces a valid FORM/AIFC container with COMM and SSND chunks.
    fn build_aiffc(
        num_channels: u16,
        sample_rate: u32,
        num_frames: u32,
        compression_type: &[u8; 4],
        audio_bytes: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();

        // Helper: append big-endian u16
        let be_u16 = |v: u16| v.to_be_bytes();
        // Helper: append big-endian u32
        let be_u32 = |v: u32| v.to_be_bytes();

        // Build the 80-bit extended sample rate.
        // Encode sample_rate as an 80-bit float (sign=0, exp biased, mantissa).
        fn f64_to_extended(val: f64) -> [u8; 10] {
            if val == 0.0 {
                return [0u8; 10];
            }
            let sign: u8 = 0;
            let mut exp = 16383i32;
            let mut mantissa = val;
            while mantissa >= 2.0 {
                mantissa /= 2.0;
                exp += 1;
            }
            while mantissa < 1.0 {
                mantissa *= 2.0;
                exp -= 1;
            }
            // mantissa is in [1.0, 2.0); scale to u64 with explicit integer bit.
            let mantissa_u64 = (mantissa * (u64::MAX as f64 / 2.0).round()) as u64;
            // sign is 0 here (we only handle positive values in this helper).
            let exp_u16 = ((sign as u16) << 15) | (exp as u16 & 0x7FFF);
            let mut result = [0u8; 10];
            result[0..2].copy_from_slice(&exp_u16.to_be_bytes());
            result[2..10].copy_from_slice(&mantissa_u64.to_be_bytes());
            result
        }

        // COMM chunk payload for AIFC:
        // numChannels (2) + numFrames (4) + sampleSize (2) + sampleRate (10) + compressionType (4)
        // + compressionName pascal string (at minimum 1 byte: 0 = empty string)
        let sr_bytes = f64_to_extended(sample_rate as f64);
        let mut comm_payload: Vec<u8> = Vec::new();
        comm_payload.extend_from_slice(&be_u16(num_channels));
        comm_payload.extend_from_slice(&be_u32(num_frames));
        comm_payload.extend_from_slice(&be_u16(8)); // sampleSize = 8 (1 byte per sample)
        comm_payload.extend_from_slice(&sr_bytes);
        comm_payload.extend_from_slice(compression_type);
        // Pascal string: length byte + content. Empty name: just 0.
        comm_payload.push(0); // pascal string length = 0

        // SSND chunk payload: offset(u32) + blockSize(u32) + audio_bytes
        let mut ssnd_payload: Vec<u8> = Vec::new();
        ssnd_payload.extend_from_slice(&be_u32(0)); // offset = 0
        ssnd_payload.extend_from_slice(&be_u32(0)); // blockSize = 0
        ssnd_payload.extend_from_slice(audio_bytes);

        // Calculate FORM size: "AIFC"(4) + "COMM"(4) + comm_size(4) + comm_payload
        //                       + "SSND"(4) + ssnd_size(4) + ssnd_payload
        let comm_size = comm_payload.len() as u32;
        let ssnd_size = ssnd_payload.len() as u32;
        let form_size = 4 + 4 + 4 + comm_size as usize + 4 + 4 + ssnd_size as usize;

        // Write FORM header
        out.extend_from_slice(b"FORM");
        out.extend_from_slice(&(form_size as u32).to_be_bytes());
        out.extend_from_slice(b"AIFC");

        // Write COMM chunk
        out.extend_from_slice(b"COMM");
        out.extend_from_slice(&comm_size.to_be_bytes());
        out.extend_from_slice(&comm_payload);

        // Write SSND chunk
        out.extend_from_slice(b"SSND");
        out.extend_from_slice(&ssnd_size.to_be_bytes());
        out.extend_from_slice(&ssnd_payload);

        out
    }

    #[test]
    fn test_ulaw_to_linear_sign() {
        // µ-law sign analysis (all bits are inverted first):
        // Byte 0x80 → inverted = 0x7F, bit7=0 → sign=false → result positive (large +31600)
        // Byte 0x00 → inverted = 0xFF, bit7=1 → sign=true  → result negative (-31600)
        let pos = ulaw_to_linear(0x80);
        let neg = ulaw_to_linear(0x00);
        assert!(
            pos > 0,
            "ulaw_to_linear(0x80) should be positive, got {pos}"
        );
        assert!(
            neg < 0,
            "ulaw_to_linear(0x00) should be negative, got {neg}"
        );
        // Symmetry: 0x80 and 0x00 should be symmetric (opposite signs, equal magnitude).
        assert_eq!(
            pos, -neg,
            "µ-law should be symmetric: 0x80={pos}, 0x00={neg}"
        );
    }

    #[test]
    fn test_alaw_to_linear_sign() {
        // A-law: XOR with 0x55 then check high bit.
        // Byte 0x80 XOR 0x55 = 0xD5 → high bit set → sign=true → result positive.
        // Byte 0x00 XOR 0x55 = 0x55 → high bit clear → sign=false → result negative.
        let pos = alaw_to_linear(0x80);
        let neg = alaw_to_linear(0x00);
        assert!(
            pos >= 0,
            "alaw_to_linear(0x80) should be non-negative, got {pos}"
        );
        assert!(
            neg <= 0,
            "alaw_to_linear(0x00) should be non-positive, got {neg}"
        );
    }

    #[test]
    fn test_decode_aiffc_ulaw_roundtrip() {
        // Build 4 mono µ-law samples at 8000 Hz.
        // Use bytes that are known to produce valid output.
        let raw_bytes: Vec<u8> = vec![0xFF, 0x7F, 0x80, 0x00];
        let data = build_aiffc(1, 8000, 4, b"ulaw", &raw_bytes);
        let mut cursor = Cursor::new(data);
        let buf = decode_aiffc_compressed(&mut cursor).expect("decode_aiffc_compressed ulaw");
        assert_eq!(buf.sample_rate, 8000);
        assert_eq!(buf.samples.len(), 4);
        // All samples should be in the valid f32 PCM range [-1.0, 1.0].
        for s in &buf.samples {
            assert!(
                s.abs() <= 1.0,
                "µ-law decoded sample {s} out of [-1.0, 1.0]"
            );
        }
    }

    #[test]
    fn test_decode_aiffc_alaw_roundtrip() {
        // Build 4 mono A-law samples at 8000 Hz.
        let raw_bytes: Vec<u8> = vec![0xD5, 0x55, 0x80, 0x00];
        let data = build_aiffc(1, 8000, 4, b"alaw", &raw_bytes);
        let mut cursor = Cursor::new(data);
        let buf = decode_aiffc_compressed(&mut cursor).expect("decode_aiffc_compressed alaw");
        assert_eq!(buf.sample_rate, 8000);
        assert_eq!(buf.samples.len(), 4);
        for s in &buf.samples {
            assert!(
                s.abs() <= 1.0,
                "A-law decoded sample {s} out of [-1.0, 1.0]"
            );
        }
    }

    #[test]
    fn test_decode_aiffc_uppercase_compression_type() {
        // Test that uppercase "ULAW" is also accepted.
        let raw_bytes: Vec<u8> = vec![0xFF, 0x7F];
        let data = build_aiffc(1, 8000, 2, b"ULAW", &raw_bytes);
        let mut cursor = Cursor::new(data);
        let buf = decode_aiffc_compressed(&mut cursor).expect("uppercase ULAW should be accepted");
        assert_eq!(buf.samples.len(), 2);
    }

    #[test]
    fn test_decode_aiffc_rejects_plain_aiff() {
        // A plain AIFF container should be rejected (expects AIFC).
        let mut data = Vec::new();
        data.extend_from_slice(b"FORM");
        data.extend_from_slice(&100u32.to_be_bytes());
        data.extend_from_slice(b"AIFF");
        let mut cursor = Cursor::new(data);
        let result = decode_aiffc_compressed(&mut cursor);
        assert!(
            result.is_err(),
            "plain AIFF container should return Err from decode_aiffc_compressed"
        );
    }

    #[test]
    fn test_decode_aiffc_rejects_unsupported_codec() {
        // compressionType "NONE" (uncompressed) should return UnsupportedFormat.
        let raw_bytes: Vec<u8> = vec![0x00, 0x01];
        let data = build_aiffc(1, 44100, 2, b"NONE", &raw_bytes);
        let mut cursor = Cursor::new(data);
        let result = decode_aiffc_compressed(&mut cursor);
        assert!(result.is_err(), "compressionType NONE should return Err");
    }

    #[test]
    fn test_decode_aiffc_stereo_ulaw() {
        // Stereo µ-law: 2 channels, 2 frames → 4 bytes total.
        let raw_bytes: Vec<u8> = vec![0xFF, 0x80, 0x7F, 0x00];
        let data = build_aiffc(2, 8000, 2, b"ulaw", &raw_bytes);
        let mut cursor = Cursor::new(data);
        let buf = decode_aiffc_compressed(&mut cursor).expect("stereo ulaw");
        assert_eq!(buf.samples.len(), 4, "2 channels * 2 frames = 4 samples");
    }
}
