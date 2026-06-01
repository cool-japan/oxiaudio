//! WAV `cue ` chunk and `LIST adtl` label parsing.
//!
//! Reads the `cue ` chunk (and optional `LIST adtl` labels) from a RIFF/WAVE
//! file without decoding any audio data.  The parser is a pure-Rust, zero-copy
//! scanner that skips every non-cue chunk efficiently.
//!
//! Reference: <https://www.recordingblogs.com/wiki/cue-chunk-of-a-wave-file>

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use oxiaudio_core::OxiAudioError;

// ─── Public types ─────────────────────────────────────────────────────────────

/// A cue point read from a WAV file's `cue ` chunk.
///
/// Cue points mark named (or unnamed) positions in the audio timeline, commonly
/// used as loop points, region boundaries, or chapter markers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavCuePoint {
    /// Cue point identifier (unique within the file).
    pub id: u32,
    /// Position in samples from the start of the audio data.
    ///
    /// Derived from the `sample_offset` field of the cue entry, which stores the
    /// byte/sample position within the referenced data chunk.
    pub position: u32,
    /// Optional label from the `labl` sub-chunk of the `LIST adtl` chunk.
    pub label: Option<String>,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Parse cue points from a WAV file at `path`.
///
/// Returns an empty [`Vec`] if the file contains no `cue ` chunk.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened or read, or
/// [`OxiAudioError::UnsupportedFormat`] if the bytes are not a valid RIFF/WAVE file.
#[must_use = "discarding the Result ignores parse errors"]
pub fn parse_wav_cues(path: &std::path::Path) -> Result<Vec<WavCuePoint>, OxiAudioError> {
    let mut file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    parse_wav_cues_reader(&mut file)
}

/// Parse cue points from any [`Read`] + [`Seek`] reader containing a WAV stream.
///
/// The reader must be positioned at the start of the RIFF header (byte 0).
/// Returns an empty [`Vec`] if the stream contains no `cue ` chunk.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on I/O failure, or
/// [`OxiAudioError::UnsupportedFormat`] if the stream is not a valid RIFF/WAVE.
#[must_use = "discarding the Result ignores parse errors"]
pub fn parse_wav_cues_reader<R: Read + Seek>(
    reader: &mut R,
) -> Result<Vec<WavCuePoint>, OxiAudioError> {
    verify_riff_wave_header(reader)?;

    let mut cue_points: Vec<(u32, u32)> = Vec::new(); // (id, sample_offset)
    let mut labels: HashMap<u32, String> = HashMap::new();

    // Scan top-level RIFF chunks.
    loop {
        let mut chunk_id = [0u8; 4];
        match reader.read_exact(&mut chunk_id) {
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(OxiAudioError::Io(e)),
            Ok(()) => {}
        }

        let mut size_buf = [0u8; 4];
        reader
            .read_exact(&mut size_buf)
            .map_err(OxiAudioError::Io)?;
        let chunk_size = u32::from_le_bytes(size_buf);

        match &chunk_id {
            b"cue " => parse_cue_chunk(reader, chunk_size, &mut cue_points)?,
            b"LIST" => parse_list_chunk(reader, chunk_size, &mut labels)?,
            _ => skip_chunk(reader, chunk_size)?,
        }
    }

    let result = cue_points
        .into_iter()
        .map(|(id, position)| WavCuePoint {
            id,
            position,
            label: labels.remove(&id),
        })
        .collect();

    Ok(result)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Read and validate the 12-byte RIFF/WAVE header.
fn verify_riff_wave_header<R: Read>(reader: &mut R) -> Result<(), OxiAudioError> {
    let mut riff = [0u8; 4];
    reader.read_exact(&mut riff).map_err(OxiAudioError::Io)?;
    if &riff != b"RIFF" {
        return Err(OxiAudioError::UnsupportedFormat(
            "not a RIFF/WAV file".to_string(),
        ));
    }

    // Skip the 4-byte RIFF chunk size (not needed for scanning).
    let mut _size = [0u8; 4];
    reader.read_exact(&mut _size).map_err(OxiAudioError::Io)?;

    let mut wave = [0u8; 4];
    reader.read_exact(&mut wave).map_err(OxiAudioError::Io)?;
    if &wave != b"WAVE" {
        return Err(OxiAudioError::UnsupportedFormat(
            "RIFF file is not WAVE format".to_string(),
        ));
    }

    Ok(())
}

/// Parse a `cue ` chunk and append `(id, sample_offset)` pairs to `out`.
///
/// Layout (after the 8-byte chunk header):
/// ```text
/// num_cue_points(u32 LE)
/// Per cue point (24 bytes):
///   id(u32) + position(u32) + data_chunk_id(4) + chunk_start(u32) + block_start(u32) + sample_offset(u32)
/// ```
fn parse_cue_chunk<R: Read>(
    reader: &mut R,
    _chunk_size: u32,
    out: &mut Vec<(u32, u32)>,
) -> Result<(), OxiAudioError> {
    let mut count_buf = [0u8; 4];
    reader
        .read_exact(&mut count_buf)
        .map_err(OxiAudioError::Io)?;
    let count = u32::from_le_bytes(count_buf) as usize;

    for _ in 0..count {
        let mut entry = [0u8; 24];
        reader.read_exact(&mut entry).map_err(OxiAudioError::Io)?;
        let id = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
        // sample_offset is the last u32 in the 24-byte entry (bytes 20–23).
        let sample_offset = u32::from_le_bytes([entry[20], entry[21], entry[22], entry[23]]);
        out.push((id, sample_offset));
    }

    Ok(())
}

/// Parse a `LIST` chunk and, if it is an `adtl` list, extract `labl` labels.
fn parse_list_chunk<R: Read + Seek>(
    reader: &mut R,
    chunk_size: u32,
    labels: &mut HashMap<u32, String>,
) -> Result<(), OxiAudioError> {
    // The LIST chunk payload begins with a 4-byte sub-type identifier.
    if chunk_size < 4 {
        // Too small to hold the sub-type — skip entirely.
        return skip_chunk(reader, chunk_size);
    }

    let mut sub_type = [0u8; 4];
    reader
        .read_exact(&mut sub_type)
        .map_err(OxiAudioError::Io)?;

    if &sub_type != b"adtl" {
        // Not an associated data list — skip the remaining bytes.
        let remaining = chunk_size.saturating_sub(4);
        return skip_chunk(reader, remaining);
    }

    // Scan adtl sub-chunks.  `remaining` tracks bytes left in the LIST payload
    // (after the 4-byte "adtl" type).
    let mut remaining = chunk_size.saturating_sub(4);

    while remaining >= 8 {
        let mut sub_id = [0u8; 4];
        reader.read_exact(&mut sub_id).map_err(OxiAudioError::Io)?;
        let mut sub_size_buf = [0u8; 4];
        reader
            .read_exact(&mut sub_size_buf)
            .map_err(OxiAudioError::Io)?;
        let sub_size = u32::from_le_bytes(sub_size_buf);

        // Account for the 8-byte sub-chunk header.
        remaining = remaining.saturating_sub(8);

        if &sub_id == b"labl" && sub_size >= 4 {
            let mut id_buf = [0u8; 4];
            reader.read_exact(&mut id_buf).map_err(OxiAudioError::Io)?;
            let label_id = u32::from_le_bytes(id_buf);

            let text_len = (sub_size - 4) as usize;
            let mut text_bytes = vec![0u8; text_len];
            reader
                .read_exact(&mut text_bytes)
                .map_err(OxiAudioError::Io)?;

            // Strip trailing null terminator and any further nulls.
            let end = text_bytes.iter().position(|&b| b == 0).unwrap_or(text_len);
            let label = String::from_utf8_lossy(&text_bytes[..end]).into_owned();
            labels.insert(label_id, label);

            // Account for the sub-chunk payload (sub_size bytes after the header).
            remaining = remaining.saturating_sub(sub_size);

            // RIFF chunks are padded to even size on disk; the padding byte is NOT
            // counted in sub_size, so we consume it here if the payload is odd-length.
            if sub_size % 2 != 0 {
                reader
                    .seek(SeekFrom::Current(1))
                    .map_err(OxiAudioError::Io)?;
                remaining = remaining.saturating_sub(1);
            }
        } else {
            // Unknown sub-chunk — skip its payload.
            reader
                .seek(SeekFrom::Current(sub_size as i64))
                .map_err(OxiAudioError::Io)?;
            remaining = remaining.saturating_sub(sub_size);

            if sub_size % 2 != 0 {
                reader
                    .seek(SeekFrom::Current(1))
                    .map_err(OxiAudioError::Io)?;
                remaining = remaining.saturating_sub(1);
            }
        }
    }

    Ok(())
}

/// Skip `chunk_size` bytes (plus an optional RIFF alignment pad byte when odd).
fn skip_chunk<R: Seek>(reader: &mut R, chunk_size: u32) -> Result<(), OxiAudioError> {
    reader
        .seek(SeekFrom::Current(chunk_size as i64))
        .map_err(OxiAudioError::Io)?;
    if chunk_size % 2 != 0 {
        reader
            .seek(SeekFrom::Current(1))
            .map_err(OxiAudioError::Io)?;
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    use oxiaudio_encode::{encode_wav_with_cues, CuePoint};

    use super::{parse_wav_cues, parse_wav_cues_reader, WavCuePoint};

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_audio_buffer(frames: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.0f32; frames],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn build_wav_bytes_no_cues(frames: usize) -> Vec<u8> {
        let buf = make_audio_buffer(frames);
        let mut cursor = Cursor::new(Vec::new());
        encode_wav_with_cues(&buf, &mut cursor, &[]).expect("encode_wav_with_cues");
        cursor.into_inner()
    }

    fn build_wav_bytes_with_cues(frames: usize, cues: &[CuePoint]) -> Vec<u8> {
        let buf = make_audio_buffer(frames);
        let mut cursor = Cursor::new(Vec::new());
        encode_wav_with_cues(&buf, &mut cursor, cues).expect("encode_wav_with_cues");
        cursor.into_inner()
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// Encoding a WAV without cues and parsing it should yield an empty Vec.
    #[test]
    fn test_parse_wav_cues_no_cues() {
        let bytes = build_wav_bytes_no_cues(4096);
        let mut cursor = Cursor::new(bytes);
        let result = parse_wav_cues_reader(&mut cursor).expect("parse_wav_cues_reader");
        assert!(
            result.is_empty(),
            "expected empty cue list for a WAV without a cue chunk, got {:?}",
            result
        );
    }

    /// Passing non-WAV bytes should return an Err, not panic.
    #[test]
    fn test_parse_wav_cues_non_wav() {
        let junk = b"this is not a wav file at all!";
        let mut cursor = Cursor::new(junk.as_slice());
        let result = parse_wav_cues_reader(&mut cursor);
        assert!(
            result.is_err(),
            "expected Err for non-WAV bytes, got Ok({:?})",
            result
        );
    }

    /// Write a WAV with cues via oxiaudio-encode, parse it back, verify positions and labels.
    #[test]
    fn test_parse_wav_cues_roundtrip() {
        let cues_in = vec![
            CuePoint::with_label(1, 100, "Intro"),
            CuePoint::new(2, 500),
            CuePoint::with_label(3, 1000, "Verse"),
        ];
        let bytes = build_wav_bytes_with_cues(4096, &cues_in);
        let mut cursor = Cursor::new(bytes);
        let parsed = parse_wav_cues_reader(&mut cursor).expect("parse_wav_cues_reader");

        assert_eq!(
            parsed.len(),
            cues_in.len(),
            "number of parsed cue points must match encoded count"
        );

        for (expected, got) in cues_in.iter().zip(parsed.iter()) {
            assert_eq!(
                got.id, expected.id,
                "cue id mismatch: expected {}, got {}",
                expected.id, got.id
            );
            assert_eq!(
                got.position, expected.position,
                "cue position mismatch for id={}: expected {}, got {}",
                expected.id, expected.position, got.position
            );
            assert_eq!(
                got.label.as_deref(),
                expected.label.as_deref(),
                "cue label mismatch for id={}: expected {:?}, got {:?}",
                expected.id,
                expected.label,
                got.label
            );
        }
    }

    /// File-path API: write a WAV to a temp file and parse via `parse_wav_cues`.
    #[test]
    fn test_parse_wav_cues_file_path() {
        use std::io::Write;

        let cues_in = vec![CuePoint::with_label(7, 44_100, "LoopStart")];
        let bytes = build_wav_bytes_with_cues(88_200, &cues_in);

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("oxiaudio_decode_wav_cue_{ts}.wav"));
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            f.write_all(&bytes).expect("write wav bytes");
        }

        let parsed = parse_wav_cues(&path).expect("parse_wav_cues");
        let _ = std::fs::remove_file(&path);

        assert_eq!(parsed.len(), 1, "expected exactly 1 cue point");
        assert_eq!(
            parsed[0],
            WavCuePoint {
                id: 7,
                position: 44_100,
                label: Some("LoopStart".to_string()),
            }
        );
    }

    /// RIFF magic bytes present but `WAVE` identifier missing → UnsupportedFormat.
    #[test]
    fn test_parse_wav_cues_riff_not_wave() {
        // Build a fake RIFF header with "AIFF" instead of "WAVE".
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&0u32.to_le_bytes()); // size placeholder
        bytes.extend_from_slice(b"AIFF");
        let mut cursor = Cursor::new(bytes);
        let result = parse_wav_cues_reader(&mut cursor);
        assert!(
            result.is_err(),
            "expected Err for RIFF/AIFF (not RIFF/WAVE), got Ok"
        );
    }
}
