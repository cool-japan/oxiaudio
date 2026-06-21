/// FLAC METADATA_BLOCK_PICTURE (type 6) — album art embedding.
///
/// The FLAC picture block follows the FLAC specification section 9.3.7.
/// Payload layout (all big-endian):
/// ```text
/// picture_type:        u32
/// mime_length:         u32
/// mime_type:           [u8; mime_length]
/// description_length:  u32
/// description:         [u8; description_length]  (UTF-8)
/// width:               u32
/// height:              u32
/// color_depth:         u32
/// color_count:         u32
/// data_length:         u32
/// data:                [u8; data_length]
/// ```
use std::io::{Seek, Write};

use flacenc::bitsink::ByteSink;
use flacenc::component::{BitRepr, MetadataBlockData};
use flacenc::error::Verify;
use flacenc::source::MemSource;
use oxiaudio_core::{AudioBuffer, OxiAudioError};

use crate::flac_core::{block_size_for_level, clamp_flac_bits, flac_full_scale};
use crate::flac_meta::FlacMetaConfig;

// ─── FlacPicture ─────────────────────────────────────────────────────────────

/// FLAC METADATA_BLOCK_PICTURE for embedding cover art in FLAC streams.
///
/// Block type 6 per the FLAC specification.  Picture type values follow the
/// ID3v2 `APIC` frame spec (type 3 = front cover is the most common).
#[derive(Debug, Clone)]
pub struct FlacPicture {
    /// Picture type per FLAC/ID3v2 spec (3 = front cover, 0 = other).
    pub picture_type: u32,
    /// MIME type string (e.g. `"image/jpeg"`, `"image/png"`).
    pub mime_type: String,
    /// Description (usually empty; UTF-8).
    pub description: String,
    /// Image width in pixels (0 if unknown).
    pub width: u32,
    /// Image height in pixels (0 if unknown).
    pub height: u32,
    /// Colour depth in bits per pixel (0 if unknown).
    pub color_depth: u32,
    /// Number of colours for indexed images; 0 for non-indexed.
    pub color_count: u32,
    /// Raw image bytes (JPEG, PNG, …).
    pub data: Vec<u8>,
}

impl FlacPicture {
    /// Convenience constructor: front cover from JPEG bytes.
    ///
    /// Sets `picture_type = 3` and `mime_type = "image/jpeg"`.
    /// Width/height/color fields default to 0 (unknown).
    #[must_use]
    pub fn front_cover_jpeg(data: Vec<u8>) -> Self {
        Self {
            picture_type: 3,
            mime_type: "image/jpeg".to_string(),
            description: String::new(),
            width: 0,
            height: 0,
            color_depth: 0,
            color_count: 0,
            data,
        }
    }

    /// Convenience constructor: front cover from PNG bytes.
    ///
    /// Sets `picture_type = 3` and `mime_type = "image/png"`.
    /// Width/height/color fields default to 0 (unknown).
    #[must_use]
    pub fn front_cover_png(data: Vec<u8>) -> Self {
        Self {
            picture_type: 3,
            mime_type: "image/png".to_string(),
            description: String::new(),
            width: 0,
            height: 0,
            color_depth: 0,
            color_count: 0,
            data,
        }
    }

    /// Serialize to a FLAC `METADATA_BLOCK_PICTURE` binary payload (block type 6).
    ///
    /// The returned `Vec<u8>` is the raw payload that is passed to
    /// `MetadataBlockData::new_unknown(6, &payload)`.
    pub(crate) fn to_block_payload(&self) -> Vec<u8> {
        let mime_bytes = self.mime_type.as_bytes();
        let desc_bytes = self.description.as_bytes();

        let capacity = 4 // picture_type
            + 4 + mime_bytes.len()   // mime
            + 4 + desc_bytes.len()   // description
            + 4 * 4                  // width, height, color_depth, color_count
            + 4 + self.data.len(); // data

        let mut payload = Vec::with_capacity(capacity);

        payload.extend_from_slice(&self.picture_type.to_be_bytes());

        payload.extend_from_slice(&(mime_bytes.len() as u32).to_be_bytes());
        payload.extend_from_slice(mime_bytes);

        payload.extend_from_slice(&(desc_bytes.len() as u32).to_be_bytes());
        payload.extend_from_slice(desc_bytes);

        payload.extend_from_slice(&self.width.to_be_bytes());
        payload.extend_from_slice(&self.height.to_be_bytes());
        payload.extend_from_slice(&self.color_depth.to_be_bytes());
        payload.extend_from_slice(&self.color_count.to_be_bytes());

        payload.extend_from_slice(&(self.data.len() as u32).to_be_bytes());
        payload.extend_from_slice(&self.data);

        payload
    }
}

// ─── Shared encode helper ─────────────────────────────────────────────────────

/// Build a FLAC `Stream` from `buf` using `config`, then invoke `extra_blocks`
/// to add any extra metadata blocks before serialising to `writer`.
fn encode_flac_with_extra_blocks<W, F>(
    buf: &AudioBuffer<f32>,
    mut writer: W,
    config: &FlacMetaConfig,
    extra_blocks: F,
) -> Result<(), OxiAudioError>
where
    W: Write + Seek,
    F: FnOnce(&mut flacenc::component::Stream) -> Result<(), OxiAudioError>,
{
    let channels = buf.channels.channel_count();
    let block_size = block_size_for_level(config.compression_level);
    let bits = clamp_flac_bits(config.bits_per_sample);
    let scale = flac_full_scale(bits);

    let pcm: Vec<i32> = buf
        .samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * scale) as i32)
        .collect();

    let source = MemSource::from_samples(&pcm, channels, bits as usize, buf.sample_rate as usize);

    let mut cfg = flacenc::config::Encoder::default();
    cfg.block_size = block_size;
    let cfg = cfg
        .into_verified()
        .map_err(|(_, e)| OxiAudioError::Encode(e.to_string()))?;

    let mut stream = flacenc::encode_with_fixed_block_size(&cfg, source, block_size)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    stream
        .stream_info_mut()
        .set_block_sizes(block_size, block_size)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    // Inject Vorbis comment block (type 4) if present.
    if !config.comments.is_empty() {
        let payload = build_vorbis_comment_payload(&config.comments);
        let block = MetadataBlockData::new_unknown(4, &payload)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
        stream.add_metadata_block(block);
    }

    // Let the caller inject additional blocks (e.g. PICTURE).
    extra_blocks(&mut stream)?;

    let mut sink = ByteSink::with_capacity(stream.count_bits());
    stream
        .write(&mut sink)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    writer
        .write_all(sink.as_slice())
        .map_err(OxiAudioError::Io)?;

    Ok(())
}

/// Build a raw VORBIS_COMMENT (type 4) payload (little-endian, Ogg Vorbis layout).
fn build_vorbis_comment_payload(comments: &[(String, String)]) -> Vec<u8> {
    const VENDOR: &str = concat!("OxiAudio ", env!("CARGO_PKG_VERSION"));
    let vendor_bytes = VENDOR.as_bytes();

    let mut payload = Vec::new();
    let vlen = vendor_bytes.len() as u32;
    payload.extend_from_slice(&vlen.to_le_bytes());
    payload.extend_from_slice(vendor_bytes);

    payload.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for (key, value) in comments {
        let entry = format!("{key}={value}");
        let entry_bytes = entry.as_bytes();
        payload.extend_from_slice(&(entry_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(entry_bytes);
    }

    payload
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Encode an [`AudioBuffer<f32>`] to FLAC with an embedded cover-art picture.
///
/// Uses the default [`FlacMetaConfig`] (compression level 5, 16-bit, no Vorbis
/// comments).  The picture is serialised as a FLAC `METADATA_BLOCK_PICTURE`
/// (block type 6) and appended after `STREAMINFO`.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on configuration, encode, or I/O failure.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_with_picture<W: Write + Seek>(
    buf: &AudioBuffer<f32>,
    writer: W,
    picture: &FlacPicture,
) -> Result<(), OxiAudioError> {
    let meta = FlacMetaConfig::default();
    encode_flac_with_metadata_and_picture(buf, writer, &meta, picture)
}

/// Encode an [`AudioBuffer<f32>`] to FLAC with both Vorbis comments and cover art.
///
/// Blocks are written in order: `STREAMINFO` → `VORBIS_COMMENT` (if any
/// comments are set) → `PICTURE`.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on configuration, encode, or I/O failure.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_with_metadata_and_picture<W: Write + Seek>(
    buf: &AudioBuffer<f32>,
    writer: W,
    meta: &FlacMetaConfig,
    picture: &FlacPicture,
) -> Result<(), OxiAudioError> {
    let payload = picture.to_block_payload();
    encode_flac_with_extra_blocks(buf, writer, meta, |stream| {
        let block = MetadataBlockData::new_unknown(6, &payload)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
        stream.add_metadata_block(block);
        Ok(())
    })
}

/// File-based convenience wrapper around [`encode_flac_with_picture`].
///
/// Creates (or truncates) the file at `path` and encodes with embedded cover art.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be created, or any encode
/// error from [`encode_flac_with_picture`].
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_with_picture_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    picture: &FlacPicture,
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    encode_flac_with_picture(buf, writer, picture)
}

/// Embed raw album-art bytes in a FLAC stream.
///
/// This is a thin convenience wrapper around [`encode_flac_with_picture`] for
/// callers who already have raw image bytes and a MIME type string, without
/// needing to construct a [`FlacPicture`] manually.
///
/// `picture_type` follows the ID3v2 / FLAC specification (3 = front cover is
/// the most common value; 0 = other).  Pass `3` for standard front-cover art.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on configuration, encode, or I/O failure.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use oxiaudio_encode::encode_flac_with_album_art;
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let buf = AudioBuffer {
///     samples: vec![0.0f32; 4096],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
/// // Minimal 1×1 PNG bytes (for illustration only; not a real image).
/// let fake_png = vec![0x89u8, 0x50, 0x4E, 0x47];
/// let mut out = Cursor::new(Vec::new());
/// // In practice this would succeed only with valid FLAC config; the doc-test
/// // just shows the call site.
/// let _ = encode_flac_with_album_art(&buf, &mut out, 5, 3, "image/png", &fake_png);
/// ```
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_with_album_art<W: Write + Seek>(
    buf: &AudioBuffer<f32>,
    writer: W,
    compression_level: u8,
    picture_type: u32,
    mime_type: &str,
    image_data: &[u8],
) -> Result<(), OxiAudioError> {
    let picture = FlacPicture {
        picture_type,
        mime_type: mime_type.to_string(),
        description: String::new(),
        width: 0,
        height: 0,
        color_depth: 0,
        color_count: 0,
        data: image_data.to_vec(),
    };
    let meta = FlacMetaConfig {
        compression_level,
        bits_per_sample: 16,
        comments: Vec::new(),
    };
    encode_flac_with_metadata_and_picture(buf, writer, &meta, &picture)
}

/// File-based convenience wrapper around [`encode_flac_with_album_art`].
///
/// Creates (or truncates) the file at `path` and encodes with embedded album art.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be created, or any encode
/// error from [`encode_flac_with_album_art`].
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_with_album_art_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    compression_level: u8,
    picture_type: u32,
    mime_type: &str,
    image_data: &[u8],
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    encode_flac_with_album_art(
        buf,
        writer,
        compression_level,
        picture_type,
        mime_type,
        image_data,
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    use super::{
        encode_flac_with_metadata_and_picture, encode_flac_with_picture,
        encode_flac_with_picture_file, FlacPicture,
    };
    use crate::FlacMetaConfig;

    /// Minimal valid 1×1 white PNG (69 bytes).
    const MINIMAL_PNG: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR length + type
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // 8-bit RGB + CRC
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT length + type
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, // IDAT data
        0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC, // IDAT CRC
        0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND length + type
        0x44, 0xAE, 0x42, 0x60, 0x82, // IEND CRC
    ];

    fn make_buf(samples: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.0f32; samples],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_flac_picture_payload_structure() {
        let picture = FlacPicture::front_cover_png(MINIMAL_PNG.to_vec());
        let payload = picture.to_block_payload();

        // First 4 bytes: picture_type (3) big-endian.
        assert_eq!(&payload[..4], &3u32.to_be_bytes(), "picture_type must be 3");

        // Next 4 bytes: mime_length.
        let mime_len = u32::from_be_bytes(payload[4..8].try_into().expect("4 bytes")) as usize;
        let mime_bytes = b"image/png";
        assert_eq!(
            mime_len,
            mime_bytes.len(),
            "mime_length must equal len of 'image/png'"
        );

        // Immediately after: mime string bytes.
        assert_eq!(
            &payload[8..8 + mime_len],
            mime_bytes,
            "mime bytes must be 'image/png'"
        );
    }

    #[test]
    fn test_flac_picture_front_cover_jpeg() {
        let picture = FlacPicture::front_cover_jpeg(b"fake-jpeg-data".to_vec());
        assert_eq!(
            picture.picture_type, 3,
            "front cover must have picture_type = 3"
        );
        assert_eq!(picture.mime_type, "image/jpeg");
    }

    #[test]
    fn test_encode_flac_with_picture_creates_file() {
        let buf = make_buf(4096);
        let picture = FlacPicture::front_cover_png(MINIMAL_PNG.to_vec());

        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "oxiaudio_encode_flac_picture_{}.flac",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));

        encode_flac_with_picture_file(&buf, &tmp, &picture)
            .expect("encode_flac_with_picture_file must succeed");

        let meta = std::fs::metadata(&tmp).expect("output file must exist");
        assert!(meta.len() > 0, "output file must be non-empty");

        // Verify fLaC magic.
        let bytes = std::fs::read(&tmp).expect("read output file");
        assert_eq!(&bytes[..4], b"fLaC", "output must start with fLaC marker");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_encode_flac_with_picture_cursor() {
        let buf = make_buf(4096);
        let picture = FlacPicture::front_cover_png(MINIMAL_PNG.to_vec());

        let mut cursor = Cursor::new(Vec::new());
        encode_flac_with_picture(&buf, &mut cursor, &picture)
            .expect("encode_flac_with_picture must succeed");

        let bytes = cursor.into_inner();
        assert!(!bytes.is_empty(), "output must not be empty");
        assert_eq!(&bytes[..4], b"fLaC", "must start with fLaC");
    }

    #[test]
    fn test_encode_flac_with_album_art_convenience() {
        use super::{encode_flac_with_album_art, encode_flac_with_album_art_file};

        let buf = make_buf(4096);

        // ── in-memory path ────────────────────────────────────────────────
        let mut cursor = Cursor::new(Vec::new());
        encode_flac_with_album_art(
            &buf,
            &mut cursor,
            5,
            3, // front cover
            "image/png",
            MINIMAL_PNG,
        )
        .expect("encode_flac_with_album_art must succeed");

        let bytes = cursor.into_inner();
        assert_eq!(&bytes[..4], b"fLaC", "output must start with fLaC");

        // Verify METADATA_BLOCK_PICTURE signature bytes are present.
        // picture_type=3 serialised as 4 BE bytes: [0,0,0,3]
        let picture_type_marker = [0u8, 0, 0, 3];
        let found = bytes
            .windows(picture_type_marker.len())
            .any(|w| w == picture_type_marker);
        assert!(
            found,
            "METADATA_BLOCK_PICTURE picture_type=3 bytes must appear in FLAC output"
        );

        // The MIME type 'image/png' must appear as raw bytes.
        let mime_marker = b"image/png";
        let has_mime = bytes.windows(mime_marker.len()).any(|w| w == mime_marker);
        assert!(
            has_mime,
            "'image/png' MIME bytes must appear in FLAC output"
        );

        // ── file path ─────────────────────────────────────────────────────
        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "oxiaudio_album_art_{}.flac",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        encode_flac_with_album_art_file(&buf, &tmp, 5, 3, "image/png", MINIMAL_PNG)
            .expect("encode_flac_with_album_art_file must succeed");
        let file_bytes = std::fs::read(&tmp).expect("read album art flac file");
        assert_eq!(&file_bytes[..4], b"fLaC", "file must start with fLaC");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_encode_flac_with_metadata_and_picture() {
        let buf = make_buf(4096);
        let picture = FlacPicture::front_cover_jpeg(b"fake-jpeg".to_vec());
        let meta = FlacMetaConfig {
            compression_level: 3,
            bits_per_sample: 16,
            comments: vec![("TITLE".to_string(), "Album Art Test".to_string())],
        };

        let mut cursor = Cursor::new(Vec::new());
        encode_flac_with_metadata_and_picture(&buf, &mut cursor, &meta, &picture)
            .expect("encode_flac_with_metadata_and_picture must succeed");

        let bytes = cursor.into_inner();
        assert_eq!(&bytes[..4], b"fLaC", "must start with fLaC");

        // Vorbis comment tag must be present.
        let needle = b"TITLE=Album Art Test";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(
            found,
            "Vorbis comment 'TITLE=Album Art Test' must be present"
        );
    }
}
