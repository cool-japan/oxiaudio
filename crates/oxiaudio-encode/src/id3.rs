//! ID3v2.4 tag writer (RFC / ID3.org informal spec).
//!
//! Produces a complete ID3v2.4 binary tag block suitable for prepending to an MP3 stream.
//!
//! ## Key differences from ID3v2.3
//!
//! * **Syncsafe frame sizes** — in ID3v2.4 each frame's 4-byte size field is syncsafe
//!   (7 bits per byte), **not** a raw big-endian 32-bit integer as it is in v2.3.
//! * All text frames use UTF-8 (encoding byte `0x03`) rather than Latin-1 or UTF-16.
//!
//! ## Supported frames
//!
//! | Field         | Frame ID | Notes                                         |
//! |---------------|----------|-----------------------------------------------|
//! | title         | `TIT2`   | Text (UTF-8)                                  |
//! | artist        | `TPE1`   | Text (UTF-8)                                  |
//! | album         | `TALB`   | Text (UTF-8)                                  |
//! | year          | `TDRC`   | Text (UTF-8, recording time)                  |
//! | track_number  | `TRCK`   | Text (UTF-8)                                  |
//! | genre         | `TCON`   | Text (UTF-8)                                  |
//! | comment       | `COMM`   | Full COMM frame with `eng` language           |
//! | composer      | `TCOM`   | Text (UTF-8)                                  |
//! | album_art     | `APIC`   | Attached picture (front cover, type `0x03`)   |
//! | extra_tags    | `TXXX`   | User-defined text frames for arbitrary keys   |

use oxiaudio_core::OxiAudioError;
use std::io::Write;

// ─── Public API ───────────────────────────────────────────────────────────────

/// A builder for ID3v2.4 tags.
///
/// Call [`Id3v24Tag::write_to`] or [`Id3v24Tag::to_vec`] to serialise.
#[derive(Debug, Clone, Default)]
pub struct Id3v24Tag {
    /// Track title (`TIT2`).
    pub title: Option<String>,
    /// Lead artist / performer (`TPE1`).
    pub artist: Option<String>,
    /// Album name (`TALB`).
    pub album: Option<String>,
    /// Recording year as a string (`TDRC`). E.g. `"2024"`.
    pub year: Option<String>,
    /// Track number / total tracks (`TRCK`). E.g. `"3"` or `"3/12"`.
    pub track_number: Option<String>,
    /// Genre string (`TCON`). E.g. `"Rock"` or `"(17)"`.
    pub genre: Option<String>,
    /// Comment, encoded in a `COMM` frame with `eng` language.
    pub comment: Option<String>,
    /// Composer (`TCOM`).
    pub composer: Option<String>,
    /// Cover art: (MIME type, image bytes). Embedded as an `APIC` frame, picture type `0x03`
    /// (Cover (front)).
    pub album_art: Option<(String, Vec<u8>)>,
    /// Arbitrary user-defined text frames (`TXXX`): list of `(description, value)` pairs.
    pub extra_tags: Vec<(String, String)>,
}

impl Id3v24Tag {
    /// Create a new empty tag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialise the tag to a `Vec<u8>`.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Encode`] if any frame cannot be built (e.g. an oversize field).
    pub fn to_vec(&self) -> Result<Vec<u8>, OxiAudioError> {
        let mut out = Vec::new();
        self.write_to(&mut out)?;
        Ok(out)
    }

    /// Serialise the tag, writing directly into `dst`.
    ///
    /// The 10-byte ID3v2.4 header is emitted first, then all frames.  `dst` is not
    /// flushed automatically.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Io`] on write failure, or [`OxiAudioError::Encode`] if a
    /// tag frame is too large (≥ 256 MiB) to fit in a syncsafe integer.
    pub fn write_to(&self, dst: &mut impl Write) -> Result<(), OxiAudioError> {
        // Accumulate all frame bytes first so we know the total size.
        let frames = self.build_frames()?;

        // ── ID3v2.4 header (10 bytes) ─────────────────────────────────────────
        //   "ID3" + version (0x04 0x00) + flags (0x00) + syncsafe size (4 bytes)
        dst.write_all(b"ID3").map_err(OxiAudioError::Io)?;
        dst.write_all(&[0x04, 0x00, 0x00])
            .map_err(OxiAudioError::Io)?; // v2.4, no flags

        let total = frames.len();
        let ss = syncsafe(total as u32)?;
        dst.write_all(&ss).map_err(OxiAudioError::Io)?;

        // ── Frames ────────────────────────────────────────────────────────────
        dst.write_all(&frames).map_err(OxiAudioError::Io)
    }

    // ── Frame assembly ────────────────────────────────────────────────────────

    fn build_frames(&self) -> Result<Vec<u8>, OxiAudioError> {
        let mut buf = Vec::new();

        if let Some(v) = &self.title {
            push_text_frame(&mut buf, b"TIT2", v)?;
        }
        if let Some(v) = &self.artist {
            push_text_frame(&mut buf, b"TPE1", v)?;
        }
        if let Some(v) = &self.album {
            push_text_frame(&mut buf, b"TALB", v)?;
        }
        if let Some(v) = &self.year {
            push_text_frame(&mut buf, b"TDRC", v)?;
        }
        if let Some(v) = &self.track_number {
            push_text_frame(&mut buf, b"TRCK", v)?;
        }
        if let Some(v) = &self.genre {
            push_text_frame(&mut buf, b"TCON", v)?;
        }
        if let Some(v) = &self.composer {
            push_text_frame(&mut buf, b"TCOM", v)?;
        }
        if let Some(v) = &self.comment {
            push_comm_frame(&mut buf, v)?;
        }
        if let Some((mime, data)) = &self.album_art {
            push_apic_frame(&mut buf, mime, data)?;
        }
        for (desc, value) in &self.extra_tags {
            push_txxx_frame(&mut buf, desc, value)?;
        }

        Ok(buf)
    }
}

// ─── Syncsafe integer ─────────────────────────────────────────────────────────

/// Encode a 28-bit value as a 4-byte syncsafe integer (7 bits per byte, MSB always 0).
///
/// Returns [`OxiAudioError::Encode`] if `value >= 2^28` (268 MiB limit).
fn syncsafe(value: u32) -> Result<[u8; 4], OxiAudioError> {
    if value >= 0x1000_0000 {
        return Err(OxiAudioError::Encode(format!(
            "ID3v2.4: value {value} exceeds the 268 MiB syncsafe limit"
        )));
    }
    Ok([
        ((value >> 21) & 0x7F) as u8,
        ((value >> 14) & 0x7F) as u8,
        ((value >> 7) & 0x7F) as u8,
        (value & 0x7F) as u8,
    ])
}

// ─── Frame helpers ────────────────────────────────────────────────────────────

/// Push a standard text frame (T***) with UTF-8 encoding.
///
/// Frame layout: 4-byte ID | 4-byte syncsafe size | 2-byte flags | 0x03 (UTF-8) | text bytes
fn push_text_frame(buf: &mut Vec<u8>, id: &[u8; 4], text: &str) -> Result<(), OxiAudioError> {
    let text_bytes = text.as_bytes();
    // 1 byte encoding prefix + text bytes
    let content_len = 1 + text_bytes.len();
    let ss = syncsafe(content_len as u32)?;

    buf.extend_from_slice(id);
    buf.extend_from_slice(&ss);
    buf.extend_from_slice(&[0x00, 0x00]); // frame flags
    buf.push(0x03); // UTF-8 encoding byte
    buf.extend_from_slice(text_bytes);
    Ok(())
}

/// Push a `COMM` (comment) frame.
///
/// Layout: 4b ID | 4b syncsafe size | 2b flags | 0x03 | "eng" | 0x00 (short desc null) | comment
fn push_comm_frame(buf: &mut Vec<u8>, comment: &str) -> Result<(), OxiAudioError> {
    let comment_bytes = comment.as_bytes();
    // encoding(1) + language(3) + short_desc_null(1) + comment_bytes
    let content_len = 1 + 3 + 1 + comment_bytes.len();
    let ss = syncsafe(content_len as u32)?;

    buf.extend_from_slice(b"COMM");
    buf.extend_from_slice(&ss);
    buf.extend_from_slice(&[0x00, 0x00]); // frame flags
    buf.push(0x03); // UTF-8
    buf.extend_from_slice(b"eng"); // language
    buf.push(0x00); // null-terminated short description (empty)
    buf.extend_from_slice(comment_bytes);
    Ok(())
}

/// Push a `TXXX` (user-defined text) frame.
///
/// Layout: 4b ID | 4b syncsafe size | 2b flags | 0x03 | description | 0x00 | value
fn push_txxx_frame(buf: &mut Vec<u8>, description: &str, value: &str) -> Result<(), OxiAudioError> {
    let desc_bytes = description.as_bytes();
    let val_bytes = value.as_bytes();
    // encoding(1) + desc_bytes + null(1) + val_bytes
    let content_len = 1 + desc_bytes.len() + 1 + val_bytes.len();
    let ss = syncsafe(content_len as u32)?;

    buf.extend_from_slice(b"TXXX");
    buf.extend_from_slice(&ss);
    buf.extend_from_slice(&[0x00, 0x00]); // frame flags
    buf.push(0x03); // UTF-8
    buf.extend_from_slice(desc_bytes);
    buf.push(0x00); // null separator
    buf.extend_from_slice(val_bytes);
    Ok(())
}

/// Push an `APIC` (attached picture) frame for the front cover.
///
/// Layout: 4b ID | 4b syncsafe size | 2b flags | 0x03 | MIME | 0x00 | 0x03 (front cover) |
///         0x00 (empty description) | image bytes
fn push_apic_frame(buf: &mut Vec<u8>, mime: &str, image: &[u8]) -> Result<(), OxiAudioError> {
    let mime_bytes = mime.as_bytes();
    // encoding(1) + mime + null(1) + picture_type(1) + description_null(1) + image
    let content_len = 1 + mime_bytes.len() + 1 + 1 + 1 + image.len();
    let ss = syncsafe(content_len as u32)?;

    buf.extend_from_slice(b"APIC");
    buf.extend_from_slice(&ss);
    buf.extend_from_slice(&[0x00, 0x00]); // frame flags
    buf.push(0x03); // UTF-8 encoding for MIME
    buf.extend_from_slice(mime_bytes);
    buf.push(0x00); // null after MIME
    buf.push(0x03); // picture type: Cover (front)
    buf.push(0x00); // empty description (null terminator only)
    buf.extend_from_slice(image);
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the 10-byte ID3 header is correctly formed for an empty tag.
    #[test]
    fn test_empty_tag_header() {
        let tag = Id3v24Tag::new();
        let bytes = tag.to_vec().expect("empty tag must serialise");

        // Minimum: 10-byte header only (no frames).
        assert!(bytes.len() >= 10, "too short");
        assert_eq!(&bytes[0..3], b"ID3", "magic mismatch");
        assert_eq!(bytes[3], 0x04, "major version must be 4 (ID3v2.4)");
        assert_eq!(bytes[4], 0x00, "minor version must be 0");
        assert_eq!(bytes[5], 0x00, "flags must be 0");

        // Syncsafe size bytes must all have MSB clear.
        for (offset, &b) in bytes[6..10].iter().enumerate() {
            let i = offset + 6;
            assert_eq!(b & 0x80, 0, "byte {i} has MSB set (not syncsafe)");
        }

        // Size must decode to 0 (no frames).
        let size = decode_syncsafe(&bytes[6..10]);
        assert_eq!(size, 0, "empty tag must have size=0");
    }

    /// Verify that a TIT2 frame is present and uses syncsafe size encoding.
    #[test]
    fn test_title_frame() {
        let mut tag = Id3v24Tag::new();
        tag.title = Some("Test Track".into());
        let bytes = tag.to_vec().expect("serialise");

        // Find TIT2 frame after the 10-byte header.
        let frame_area = &bytes[10..];
        let pos = frame_area
            .windows(4)
            .position(|w| w == b"TIT2")
            .expect("TIT2 not found");

        // Frame size bytes must all have MSB clear (syncsafe).
        for (offset, &b) in frame_area[pos + 4..pos + 8].iter().enumerate() {
            let i = pos + 4 + offset;
            assert_eq!(b & 0x80, 0, "TIT2 size byte {i} is not syncsafe");
        }

        // Encoding byte must be 0x03 (UTF-8).
        assert_eq!(frame_area[pos + 10], 0x03, "encoding must be UTF-8 (0x03)");

        // Text content must match.
        let text_start = pos + 11;
        let content = &frame_area[text_start..text_start + "Test Track".len()];
        assert_eq!(content, b"Test Track");
    }

    /// Verify that the header size syncsafe encodes the total frame byte length.
    #[test]
    fn test_header_size_reflects_frames() {
        let mut tag = Id3v24Tag::new();
        tag.title = Some("A".into());
        tag.artist = Some("B".into());
        let bytes = tag.to_vec().expect("serialise");

        let declared_size = decode_syncsafe(&bytes[6..10]) as usize;
        assert_eq!(
            declared_size,
            bytes.len() - 10,
            "header size must equal total_length - 10"
        );
    }

    /// Verify a fully-populated tag round-trips without error.
    #[test]
    fn test_all_fields() {
        let mut tag = Id3v24Tag::new();
        tag.title = Some("Song".into());
        tag.artist = Some("Artist".into());
        tag.album = Some("Album".into());
        tag.year = Some("2024".into());
        tag.track_number = Some("1/10".into());
        tag.genre = Some("Electronic".into());
        tag.comment = Some("Great track".into());
        tag.composer = Some("Composer".into());
        tag.album_art = Some(("image/jpeg".into(), vec![0xFF, 0xD8, 0xFF, 0xE0]));
        tag.extra_tags
            .push(("REPLAYGAIN_TRACK_GAIN".into(), "-6.5 dB".into()));

        let bytes = tag.to_vec().expect("all-fields tag must serialise");
        // Must start with ID3 header.
        assert_eq!(&bytes[..3], b"ID3");
        // All size bytes in header must be syncsafe.
        for (offset, &b) in bytes[6..10].iter().enumerate() {
            let i = offset + 6;
            assert_eq!(b & 0x80, 0, "header size byte {i} not syncsafe");
        }
        // Sanity: output must contain TIT2, TPE1, TALB, APIC, TXXX.
        assert!(bytes.windows(4).any(|w| w == b"TIT2"), "TIT2 missing");
        assert!(bytes.windows(4).any(|w| w == b"TPE1"), "TPE1 missing");
        assert!(bytes.windows(4).any(|w| w == b"TALB"), "TALB missing");
        assert!(bytes.windows(4).any(|w| w == b"APIC"), "APIC missing");
        assert!(bytes.windows(4).any(|w| w == b"TXXX"), "TXXX missing");
        assert!(bytes.windows(4).any(|w| w == b"COMM"), "COMM missing");
    }

    // Helper: decode a syncsafe 4-byte big-endian value.
    fn decode_syncsafe(bytes: &[u8]) -> u32 {
        assert_eq!(bytes.len(), 4);
        (u32::from(bytes[0]) << 21)
            | (u32::from(bytes[1]) << 14)
            | (u32::from(bytes[2]) << 7)
            | u32::from(bytes[3])
    }
}
