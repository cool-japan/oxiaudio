use std::path::Path;

use oxiaudio_core::OxiAudioError;

/// Embedded album artwork extracted from audio file tags.
#[derive(Debug, Clone)]
pub struct AlbumArtwork {
    /// Raw image bytes (JPEG, PNG, etc.).
    pub data: Vec<u8>,
    /// MIME type of the image (e.g., "image/jpeg", "image/png").
    pub mime_type: String,
    /// Picture type (3 = front cover per ID3v2/FLAC convention).
    pub picture_type: u32,
    /// Optional text description.
    pub description: String,
}

impl AlbumArtwork {
    /// Returns true if the MIME type suggests this is a JPEG image.
    pub fn is_jpeg(&self) -> bool {
        self.mime_type.eq_ignore_ascii_case("image/jpeg")
            || self.mime_type.eq_ignore_ascii_case("image/jpg")
    }

    /// Returns true if the MIME type suggests this is a PNG image.
    pub fn is_png(&self) -> bool {
        self.mime_type.eq_ignore_ascii_case("image/png")
    }

    /// Returns the file extension for this image type ("jpg", "png", or "bin").
    pub fn extension(&self) -> &'static str {
        if self.is_jpeg() {
            "jpg"
        } else if self.is_png() {
            "png"
        } else {
            "bin"
        }
    }
}

/// Extract embedded album artwork from an audio file.
///
/// Searches ID3v2 APIC frames (MP3), Vorbis METADATA_BLOCK_PICTURE (FLAC/OGG),
/// and generic visual tags from symphonia metadata.
///
/// Returns `Ok(None)` if no artwork is found, `Ok(Some(art))` if found.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened, or
/// [`OxiAudioError::Decode`] if format probing fails.
#[must_use = "discarding the Result ignores extraction errors"]
pub fn extract_album_art(path: &Path) -> Result<Option<AlbumArtwork>, OxiAudioError> {
    use symphonia::core::formats::{probe::Hint, FormatOptions};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| OxiAudioError::Decode(format!("probe failed: {e}")))?;

    if let Some(rev) = format.metadata().current() {
        if let Some(art) = extract_from_metadata_revision(rev) {
            return Ok(Some(art));
        }
    }

    Ok(None)
}

/// Extract artwork from a single metadata revision, preferring parsed visuals over raw binary tags.
fn extract_from_metadata_revision(
    rev: &symphonia::core::meta::MetadataRevision,
) -> Option<AlbumArtwork> {
    use symphonia::core::meta::RawValue;

    // Primary path: symphonia parsed visuals (ID3v2 APIC / FLAC PICTURE blocks).
    // Return the first visual if any are present.
    if let Some(visual) = rev.media.visuals.first() {
        let mime_type = visual.media_type.clone().unwrap_or_default();
        let picture_type = visual.usage.map(|u| u as u32).unwrap_or(0);
        return Some(AlbumArtwork {
            data: visual.data.to_vec(),
            mime_type,
            picture_type,
            description: String::new(),
        });
    }

    // Fallback: scan raw tags for binary picture blobs.
    for tag in &rev.media.tags {
        let key_upper = tag.raw.key.to_ascii_uppercase();
        if key_upper.contains("APIC")
            || key_upper.contains("PICTURE")
            || key_upper.contains("COVERART")
        {
            if let RawValue::Binary(ref arc_data) = tag.raw.value {
                // arc_data is Arc<Box<[u8]>> — deref twice to get &[u8].
                if let Some(art) = decode_picture_binary(arc_data.as_ref().as_ref()) {
                    return Some(art);
                }
            }
        }
    }

    None
}

/// Attempt to identify image format by magic bytes and decode as an `AlbumArtwork`.
///
/// Recognises JPEG (`FF D8 FF`), PNG (`89 50 4E 47`), and the FLAC
/// `METADATA_BLOCK_PICTURE` binary layout.
fn decode_picture_binary(data: &[u8]) -> Option<AlbumArtwork> {
    if data.len() < 4 {
        return None;
    }
    // JPEG magic: FF D8 FF
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(AlbumArtwork {
            data: data.to_vec(),
            mime_type: "image/jpeg".to_string(),
            picture_type: 3,
            description: String::new(),
        });
    }
    // PNG magic: 89 50 4E 47
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        return Some(AlbumArtwork {
            data: data.to_vec(),
            mime_type: "image/png".to_string(),
            picture_type: 3,
            description: String::new(),
        });
    }
    // Try FLAC METADATA_BLOCK_PICTURE layout.
    if data.len() >= 36 {
        if let Some(art) = try_decode_flac_picture_block(data) {
            return Some(art);
        }
    }
    None
}

/// Decode a FLAC `METADATA_BLOCK_PICTURE` binary blob.
///
/// Layout (big-endian):
/// ```text
/// picture_type(4) + mime_len(4) + mime(mime_len) + desc_len(4) + desc(desc_len)
///   + width(4) + height(4) + color_depth(4) + colors(4) + data_len(4) + data(data_len)
/// ```
fn try_decode_flac_picture_block(data: &[u8]) -> Option<AlbumArtwork> {
    if data.len() < 32 {
        return None;
    }
    let picture_type = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    // Sanity-check: picture_type must be 0..=20 per spec.
    if picture_type > 20 {
        return None;
    }
    let mime_len = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let mime_end = 8usize.checked_add(mime_len)?;
    if mime_end + 4 > data.len() {
        return None;
    }
    let mime = std::str::from_utf8(&data[8..mime_end]).ok()?.to_string();
    let desc_len = u32::from_be_bytes([
        data[mime_end],
        data[mime_end + 1],
        data[mime_end + 2],
        data[mime_end + 3],
    ]) as usize;
    // skip desc + width(4) + height(4) + depth(4) + colors(4)
    let skip = mime_end
        .checked_add(4)?
        .checked_add(desc_len)?
        .checked_add(16)?;
    if skip + 4 > data.len() {
        return None;
    }
    let img_data_len =
        u32::from_be_bytes([data[skip], data[skip + 1], data[skip + 2], data[skip + 3]]) as usize;
    let img_start = skip.checked_add(4)?;
    let img_end = img_start.checked_add(img_data_len)?;
    if img_end > data.len() {
        return None;
    }
    let img_data = data[img_start..img_end].to_vec();
    Some(AlbumArtwork {
        data: img_data,
        mime_type: mime,
        picture_type,
        description: String::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Helper: build an AlbumArtwork with given data and mime_type for unit tests.
    // -------------------------------------------------------------------------
    fn make_artwork(mime: &str) -> AlbumArtwork {
        AlbumArtwork {
            data: vec![0u8; 4],
            mime_type: mime.to_string(),
            picture_type: 3,
            description: String::new(),
        }
    }

    // -------------------------------------------------------------------------
    // is_jpeg / is_png / extension tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_album_artwork_is_jpeg_true() {
        let art = make_artwork("image/jpeg");
        assert!(art.is_jpeg());
    }

    #[test]
    fn test_album_artwork_is_jpeg_false() {
        let art = make_artwork("image/png");
        assert!(!art.is_jpeg());
    }

    #[test]
    fn test_album_artwork_extension_jpeg() {
        let art = make_artwork("image/jpeg");
        assert_eq!(art.extension(), "jpg");
    }

    #[test]
    fn test_album_artwork_extension_png() {
        let art = make_artwork("image/png");
        assert_eq!(art.extension(), "png");
    }

    #[test]
    fn test_album_artwork_extension_other() {
        let art = make_artwork("image/bmp");
        assert_eq!(art.extension(), "bin");
    }

    // -------------------------------------------------------------------------
    // extract_album_art: error on nonexistent file
    // -------------------------------------------------------------------------

    #[test]
    fn test_extract_album_art_nonexistent_file() {
        let path = std::env::temp_dir().join("oxiaudio_nonexistent_artwork_xyz_12345.mp3");
        let result = extract_album_art(&path);
        assert!(result.is_err(), "expected Err for missing file, got Ok");
    }

    // -------------------------------------------------------------------------
    // decode_picture_binary magic-byte tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_decode_picture_binary_jpeg_magic() {
        let mut data = vec![0u8; 16];
        data[0] = 0xFF;
        data[1] = 0xD8;
        data[2] = 0xFF;
        let result = decode_picture_binary(&data);
        assert!(result.is_some(), "expected Some for JPEG magic bytes");
        let art = result.expect("already checked is_some");
        assert_eq!(art.mime_type, "image/jpeg");
    }

    #[test]
    fn test_decode_picture_binary_png_magic() {
        let mut data = vec![0u8; 16];
        data[0] = 0x89;
        data[1] = 0x50;
        data[2] = 0x4E;
        data[3] = 0x47;
        let result = decode_picture_binary(&data);
        assert!(result.is_some(), "expected Some for PNG magic bytes");
        let art = result.expect("already checked is_some");
        assert_eq!(art.mime_type, "image/png");
    }

    #[test]
    fn test_decode_picture_binary_unknown() {
        let data = [0x00u8, 0x01, 0x02, 0x03];
        let result = decode_picture_binary(&data);
        assert!(result.is_none(), "expected None for unknown magic bytes");
    }
}
