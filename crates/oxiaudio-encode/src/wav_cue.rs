/// WAV `cue ` chunk and `LIST adtl` (Associated Data List) support.
///
/// Writes a standard RIFF/WAVE file extended with:
///   - a `cue ` chunk (RIFF chunk type) containing cue point descriptors, and
///   - optionally a `LIST adtl` chunk containing `labl` and/or `note` sub-chunks.
///
/// Reference: <https://www.recordingblogs.com/wiki/cue-chunk-of-a-wave-file>
use std::io::{Seek, SeekFrom, Write};

use oxiaudio_core::{AudioBuffer, OxiAudioError};

// ─── helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn write_u16_le<W: Write>(w: &mut W, v: u16) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

#[inline]
fn write_u32_le<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

// ─── CuePoint ────────────────────────────────────────────────────────────────

/// A single cue point (marker) in a WAV file.
///
/// Each cue point occupies 24 bytes in the `cue ` chunk payload.
/// Optional [`label`][Self::label] and [`note`][Self::note] strings are stored
/// in a separate `LIST adtl` chunk.
#[derive(Debug, Clone)]
pub struct CuePoint {
    /// Unique cue point identifier within the file (must be > 0 for most DAWs).
    pub id: u32,
    /// Position in samples from the start of the audio data.
    pub position: u32,
    /// Optional label for this cue point (`labl` sub-chunk in `LIST adtl`).
    pub label: Option<String>,
    /// Optional note/comment for this cue point (`note` sub-chunk in `LIST adtl`).
    pub note: Option<String>,
}

impl CuePoint {
    /// Create a simple cue point with no label or note.
    #[must_use]
    pub fn new(id: u32, position: u32) -> Self {
        Self {
            id,
            position,
            label: None,
            note: None,
        }
    }

    /// Create a labeled cue point.
    #[must_use]
    pub fn with_label(id: u32, position: u32, label: impl Into<String>) -> Self {
        Self {
            id,
            position,
            label: Some(label.into()),
            note: None,
        }
    }
}

// ─── fmt chunk (16-bit PCM) ───────────────────────────────────────────────────

/// Write a `fmt ` chunk for 16-bit signed integer PCM (WAVE_FORMAT_PCM = 0x0001).
///
/// Chunk size is always 16 bytes (standard PCM, no extension).
fn write_fmt_chunk_i16<W: Write>(
    w: &mut W,
    channels: u16,
    sample_rate: u32,
) -> std::io::Result<()> {
    let bits_per_sample: u16 = 16;
    let block_align = channels * bits_per_sample / 8;
    let avg_bytes_per_sec = sample_rate * u32::from(block_align);

    w.write_all(b"fmt ")?;
    write_u32_le(w, 16)?; // chunk size
    write_u16_le(w, 0x0001)?; // WAVE_FORMAT_PCM
    write_u16_le(w, channels)?;
    write_u32_le(w, sample_rate)?;
    write_u32_le(w, avg_bytes_per_sec)?;
    write_u16_le(w, block_align)?;
    write_u16_le(w, bits_per_sample)?;
    Ok(())
}

// ─── data chunk (i16 PCM) ────────────────────────────────────────────────────

/// Convert f32 → i16 PCM and write the `data` chunk.
///
/// Each sample is multiplied by `i16::MAX` (32767), clamped to `[i16::MIN, i16::MAX]`,
/// and written as little-endian i16.
fn write_data_chunk_i16<W: Write>(w: &mut W, samples: &[f32]) -> std::io::Result<()> {
    let byte_count = (samples.len() * 2) as u32;
    w.write_all(b"data")?;
    write_u32_le(w, byte_count)?;
    for &s in samples {
        let pcm = (s * 32767.0)
            .clamp(i16::MIN as f32, i16::MAX as f32)
            .round() as i16;
        w.write_all(&pcm.to_le_bytes())?;
    }
    Ok(())
}

// ─── cue chunk ───────────────────────────────────────────────────────────────

/// Write the `cue ` chunk for the given cue points.
///
/// Layout:
/// ```text
/// "cue " + chunk_size(u32 LE) + num_cue_points(u32 LE)
/// Per cue point (24 bytes):
///   id(u32 LE) + position(u32 LE) + "data"(4 bytes)
///   + chunk_start(0u32 LE) + block_start(0u32 LE) + sample_offset(u32 LE)
/// ```
fn write_cue_chunk<W: Write>(w: &mut W, cues: &[CuePoint]) -> std::io::Result<()> {
    let num = cues.len() as u32;
    // Each cue point = 24 bytes; plus 4 bytes for num_cue_points field.
    let chunk_size = 4 + num * 24;

    w.write_all(b"cue ")?;
    write_u32_le(w, chunk_size)?;
    write_u32_le(w, num)?;

    for cue in cues {
        write_u32_le(w, cue.id)?;
        write_u32_le(w, cue.position)?;
        w.write_all(b"data")?; // data_chunk_id
        write_u32_le(w, 0)?; // chunk_start (single data chunk)
        write_u32_le(w, 0)?; // block_start
        write_u32_le(w, cue.position)?; // sample_offset
    }
    Ok(())
}

// ─── LIST adtl chunk ─────────────────────────────────────────────────────────

/// Write a `labl` or `note` sub-chunk inside a `LIST adtl` chunk.
///
/// Layout: `chunk_id(4) + size(u32 LE) + id(u32 LE) + text + '\0' [+ pad_byte]`
///
/// The chunk size includes `id` (4 bytes) + text bytes + null terminator (1 byte).
/// A pad byte is appended when `(4 + text_len + 1)` is odd to keep even alignment.
fn write_adtl_sub_chunk<W: Write>(
    w: &mut W,
    chunk_id: &[u8; 4],
    cue_id: u32,
    text: &str,
) -> std::io::Result<()> {
    let text_bytes = text.as_bytes();
    // chunk_size = id(4) + text(len) + null(1)
    let payload_len = 4 + text_bytes.len() + 1;
    let chunk_size = payload_len as u32;

    w.write_all(chunk_id)?;
    write_u32_le(w, chunk_size)?;
    write_u32_le(w, cue_id)?;
    w.write_all(text_bytes)?;
    w.write_all(&[0u8])?; // null terminator

    // Pad to even alignment (total bytes on disk = 8 header + payload_len)
    if payload_len % 2 != 0 {
        w.write_all(&[0u8])?;
    }
    Ok(())
}

/// Compute the total byte count for a single adtl sub-chunk (header included).
///
/// Returns `8 (header) + id(4) + text_len + 1 (null) + pad`.
fn adtl_sub_chunk_size(text: &str) -> u32 {
    let payload_len = 4 + text.len() + 1;
    let pad = if payload_len % 2 != 0 { 1 } else { 0 };
    (8 + payload_len + pad) as u32
}

/// Write the `LIST adtl` chunk when at least one cue point has a label or note.
///
/// Layout:
/// ```text
/// "LIST" + total_size(u32 LE) + "adtl"
/// Per cue with label: labl sub-chunk
/// Per cue with note:  note sub-chunk
/// ```
fn write_list_adtl_chunk<W: Write>(w: &mut W, cues: &[CuePoint]) -> std::io::Result<()> {
    // Calculate size of `adtl` form type + all sub-chunks.
    let mut adtl_payload_size: u32 = 4; // "adtl" form type
    for cue in cues {
        if let Some(label) = &cue.label {
            adtl_payload_size += adtl_sub_chunk_size(label);
        }
        if let Some(note) = &cue.note {
            adtl_payload_size += adtl_sub_chunk_size(note);
        }
    }

    w.write_all(b"LIST")?;
    write_u32_le(w, adtl_payload_size)?;
    w.write_all(b"adtl")?;

    for cue in cues {
        if let Some(label) = &cue.label {
            write_adtl_sub_chunk(w, b"labl", cue.id, label)?;
        }
        if let Some(note) = &cue.note {
            write_adtl_sub_chunk(w, b"note", cue.id, note)?;
        }
    }
    Ok(())
}

/// Return `true` if any cue point carries a label or note.
fn has_adtl(cues: &[CuePoint]) -> bool {
    cues.iter().any(|c| c.label.is_some() || c.note.is_some())
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Encode an [`AudioBuffer<f32>`] to WAV with embedded cue points.
///
/// Produces a standard RIFF/WAVE file (16-bit signed integer PCM) extended with:
///   - a `cue ` chunk when `cues` is non-empty, and
///   - a `LIST adtl` chunk when any cue carries a label or note.
///
/// The RIFF total-size field is backfilled via a seek after all data is written.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on any I/O failure.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_wav_with_cues<W: Write + Seek>(
    buf: &AudioBuffer<f32>,
    mut writer: W,
    cues: &[CuePoint],
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count() as u16;
    let sample_rate = buf.sample_rate;

    // ── 1. RIFF header (placeholder size) ───────────────────────────────────
    writer.write_all(b"RIFF").map_err(OxiAudioError::Io)?;
    // Placeholder for RIFF chunk size (4 bytes); we seek back to patch it later.
    let riff_size_offset = writer.stream_position().map_err(OxiAudioError::Io)?;
    writer
        .write_all(&0u32.to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    writer.write_all(b"WAVE").map_err(OxiAudioError::Io)?;

    // ── 2. fmt  chunk ────────────────────────────────────────────────────────
    write_fmt_chunk_i16(&mut writer, channels, sample_rate).map_err(OxiAudioError::Io)?;

    // ── 3. data chunk ────────────────────────────────────────────────────────
    write_data_chunk_i16(&mut writer, &buf.samples).map_err(OxiAudioError::Io)?;

    // ── 4. cue  chunk ────────────────────────────────────────────────────────
    if !cues.is_empty() {
        write_cue_chunk(&mut writer, cues).map_err(OxiAudioError::Io)?;
    }

    // ── 5. LIST adtl chunk ───────────────────────────────────────────────────
    if has_adtl(cues) {
        write_list_adtl_chunk(&mut writer, cues).map_err(OxiAudioError::Io)?;
    }

    // ── 6. Backfill RIFF chunk size ──────────────────────────────────────────
    let end_pos = writer.stream_position().map_err(OxiAudioError::Io)?;
    // RIFF chunk size = total bytes after the 8-byte RIFF header (id + size field).
    let riff_size = (end_pos - riff_size_offset - 4) as u32;
    writer
        .seek(SeekFrom::Start(riff_size_offset))
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&riff_size.to_le_bytes())
        .map_err(OxiAudioError::Io)?;

    Ok(())
}

/// File-based convenience wrapper around [`encode_wav_with_cues`].
///
/// Creates (or truncates) the file at `path` and encodes with embedded cue points.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be created, or any encode error
/// from [`encode_wav_with_cues`].
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_wav_with_cues_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    cues: &[CuePoint],
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    encode_wav_with_cues(buf, writer, cues)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    use super::{encode_wav_with_cues, encode_wav_with_cues_file, CuePoint};

    fn make_buf(samples: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.0f32; samples],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    /// Search for a 4-byte tag in a byte slice.
    fn contains_tag(bytes: &[u8], tag: &[u8; 4]) -> bool {
        bytes.windows(4).any(|w| w == tag)
    }

    #[test]
    fn test_wav_cue_point_constructor() {
        let plain = CuePoint::new(1, 100);
        assert_eq!(plain.id, 1);
        assert_eq!(plain.position, 100);
        assert!(plain.label.is_none());
        assert!(plain.note.is_none());

        let labeled = CuePoint::with_label(2, 200, "Verse");
        assert_eq!(labeled.id, 2);
        assert_eq!(labeled.position, 200);
        assert_eq!(labeled.label.as_deref(), Some("Verse"));
        assert!(labeled.note.is_none());
    }

    #[test]
    fn test_wav_cue_no_cues() {
        let buf = make_buf(4096);
        let mut cursor = Cursor::new(Vec::new());
        encode_wav_with_cues(&buf, &mut cursor, &[])
            .expect("encode_wav_with_cues with no cues must succeed");
        let bytes = cursor.into_inner();

        // Must start with RIFF magic.
        assert_eq!(&bytes[..4], b"RIFF", "must start with RIFF");
        // Must contain a data chunk.
        assert!(contains_tag(&bytes, b"data"), "must contain 'data' chunk");
        // Must NOT contain a cue chunk (no cues given).
        assert!(
            !contains_tag(&bytes, b"cue "),
            "must NOT contain 'cue ' chunk when no cues"
        );
    }

    #[test]
    fn test_wav_cue_single_cue() {
        let buf = make_buf(4096);
        let cues = vec![CuePoint::new(1, 100)];
        let mut cursor = Cursor::new(Vec::new());
        encode_wav_with_cues(&buf, &mut cursor, &cues).expect("encode_wav_with_cues must succeed");
        let bytes = cursor.into_inner();

        assert_eq!(&bytes[..4], b"RIFF", "must start with RIFF");
        assert!(contains_tag(&bytes, b"cue "), "must contain 'cue ' chunk");
        // No labels → no LIST adtl.
        assert!(
            !contains_tag(&bytes, b"LIST"),
            "must NOT have LIST chunk when no labels"
        );
    }

    #[test]
    fn test_wav_cue_with_label() {
        let buf = make_buf(4096);
        let cues = vec![CuePoint::with_label(1, 100, "Intro")];
        let mut cursor = Cursor::new(Vec::new());
        encode_wav_with_cues(&buf, &mut cursor, &cues)
            .expect("encode_wav_with_cues with label must succeed");
        let bytes = cursor.into_inner();

        assert!(contains_tag(&bytes, b"cue "), "must contain 'cue ' chunk");
        assert!(
            contains_tag(&bytes, b"LIST"),
            "must contain 'LIST' chunk for labels"
        );
        assert!(
            contains_tag(&bytes, b"labl"),
            "must contain 'labl' sub-chunk"
        );

        // The label text "Intro" must appear in the file bytes.
        let needle = b"Intro";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(found, "label text 'Intro' must be present in the output");
    }

    #[test]
    fn test_wav_cue_with_note() {
        let buf = make_buf(2048);
        let mut cue = CuePoint::new(3, 512);
        cue.note = Some("Bridge section".to_string());
        let cues = vec![cue];

        let mut cursor = Cursor::new(Vec::new());
        encode_wav_with_cues(&buf, &mut cursor, &cues)
            .expect("encode_wav_with_cues with note must succeed");
        let bytes = cursor.into_inner();

        assert!(
            contains_tag(&bytes, b"note"),
            "must contain 'note' sub-chunk"
        );
        let needle = b"Bridge section";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(
            found,
            "note text 'Bridge section' must be present in the output"
        );
    }

    #[test]
    fn test_wav_cue_file_roundtrip() {
        let buf = make_buf(4096);
        let cues = vec![CuePoint::with_label(1, 44, "Start"), CuePoint::new(2, 1000)];

        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "oxiaudio_encode_wav_cue_{}.wav",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));

        encode_wav_with_cues_file(&buf, &tmp, &cues)
            .expect("encode_wav_with_cues_file must succeed");

        let bytes = std::fs::read(&tmp).expect("read output file");
        assert_eq!(&bytes[..4], b"RIFF", "file must start with RIFF");
        assert!(
            contains_tag(&bytes, b"cue "),
            "file must contain 'cue ' chunk"
        );
        assert!(contains_tag(&bytes, b"LIST"), "file must contain LIST adtl");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_wav_cue_riff_size_consistent() {
        // The RIFF size field at offset 4 must equal total_bytes - 8.
        let buf = make_buf(2048);
        let cues = vec![CuePoint::with_label(1, 100, "Mark")];
        let mut cursor = Cursor::new(Vec::new());
        encode_wav_with_cues(&buf, &mut cursor, &cues).expect("encode_wav_with_cues must succeed");
        let bytes = cursor.into_inner();

        let riff_size = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes")) as usize;
        // Total file = RIFF header (8 bytes) + RIFF size.
        assert_eq!(
            riff_size + 8,
            bytes.len(),
            "RIFF size field must equal total_file_size - 8"
        );
    }
}
