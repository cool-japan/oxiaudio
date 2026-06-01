/// APEv2 tag writer.
///
/// APEv2 tags are used by WavPack, Musepack, Monkey's Audio, and other formats.
/// This module implements the binary format serialization per the APEv2 specification.
///
/// Format reference: <https://wiki.hydrogenaud.io/index.php?title=APEv2_specification>
use oxiaudio_core::OxiAudioError;

/// Preamble string for APEv2 header and footer.
const APE_PREAMBLE: &[u8; 8] = b"APETAGEX";
/// APEv2 version number (2000).
const APE_VERSION: u32 = 2000;
/// Header/footer size in bytes.
const APE_HEADER_SIZE: usize = 32;
/// Flags for the APEv2 header (bit 31 = has header, bit 29 = this is a header).
const APE_FLAG_HEADER: u32 = 0xA000_0000;
/// Flags for the APEv2 footer (bit 31 = has header, bit 30 = has no footer is clear = footer present).
const APE_FLAG_FOOTER: u32 = 0x8000_0000;
/// Item type flag: UTF-8 text (0 = UTF-8 text).
const APE_ITEM_TEXT: u32 = 0;

/// An APEv2 tag item (key-value pair).
///
/// Keys must be ASCII (7-bit), case-insensitive. Standard keys are:
/// "Title", "Artist", "Album", "Track", "Year", "Genre", "Comment".
///
/// Values are UTF-8 encoded strings.
///
/// # Examples
///
/// ```
/// use oxiaudio_encode::ApeItem;
///
/// let item = ApeItem::new("Title", "My Song");
/// assert_eq!(item.key, "Title");
/// assert_eq!(item.value, "My Song");
/// ```
#[derive(Debug, Clone)]
pub struct ApeItem {
    /// Tag key (ASCII, case-insensitive; e.g. "Title", "Artist").
    pub key: String,
    /// Tag value (UTF-8 encoded).
    pub value: String,
}

impl ApeItem {
    /// Create a new `ApeItem` with the given key and value.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiaudio_encode::ApeItem;
    ///
    /// let item = ApeItem::new("Artist", "OxiAudio");
    /// assert_eq!(item.key, "Artist");
    /// ```
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// Serialize a single APEv2 item to bytes.
///
/// Layout:
/// ```text
/// value_size: u32 LE  (byte length of value)
/// item_flags: u32 LE  (0 = UTF-8 text)
/// key:        ASCII bytes
/// null:       0x00    (key terminator)
/// value:      UTF-8 bytes (no null terminator)
/// ```
fn serialize_item(item: &ApeItem) -> Vec<u8> {
    let value_bytes = item.value.as_bytes();
    let key_bytes = item.key.as_bytes();

    let mut buf = Vec::with_capacity(8 + key_bytes.len() + 1 + value_bytes.len());

    // value_size: u32 LE
    let value_size = value_bytes.len() as u32;
    buf.extend_from_slice(&value_size.to_le_bytes());

    // item_flags: u32 LE (0 = UTF-8 text)
    buf.extend_from_slice(&APE_ITEM_TEXT.to_le_bytes());

    // key (ASCII) + null terminator
    buf.extend_from_slice(key_bytes);
    buf.push(0x00);

    // value (UTF-8, no null)
    buf.extend_from_slice(value_bytes);

    buf
}

/// Write a 32-byte APEv2 header or footer block.
///
/// Layout:
/// ```text
/// preamble:   8 bytes  (b"APETAGEX")
/// version:    4 bytes  (2000 as u32 LE)
/// tag_size:   4 bytes  (items_bytes + 32, as u32 LE) — footer size, NOT including header
/// item_count: 4 bytes  (number of items, as u32 LE)
/// flags:      4 bytes  (as u32 LE)
/// reserved:   8 bytes  (all zero)
/// ```
fn write_ape_block<W: std::io::Write>(
    writer: &mut W,
    tag_size: u32,
    item_count: u32,
    flags: u32,
) -> Result<(), OxiAudioError> {
    writer.write_all(APE_PREAMBLE).map_err(OxiAudioError::Io)?;
    writer
        .write_all(&APE_VERSION.to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&tag_size.to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&item_count.to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&flags.to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    // Reserved: 8 zero bytes.
    writer.write_all(&[0u8; 8]).map_err(OxiAudioError::Io)?;
    Ok(())
}

/// Write APEv2 tags to `writer` (header + items + footer).
///
/// Keys must be ASCII (7-bit); values are UTF-8. Keys are case-insensitive by spec.
/// Standard keys: "Title", "Artist", "Album", "Track", "Year", "Genre", "Comment".
///
/// The output layout is:
/// ```text
/// [32-byte header]
/// [item 0][item 1]...[item N-1]
/// [32-byte footer]
/// ```
///
/// The `tag_size` field in both the header and footer equals the total byte size
/// of the items plus the 32-byte footer (i.e. it does NOT count the header itself).
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on any write failure.
///
/// # Examples
///
/// ```
/// use oxiaudio_encode::{ApeItem, write_apev2};
///
/// let items = vec![
///     ApeItem::new("Title", "Test"),
///     ApeItem::new("Artist", "OxiAudio"),
/// ];
/// let mut out = Vec::new();
/// write_apev2(&mut out, &items).unwrap();
/// assert_eq!(&out[..8], b"APETAGEX");
/// ```
#[must_use = "discarding errors ignores write failure"]
pub fn write_apev2<W: std::io::Write>(
    writer: &mut W,
    items: &[ApeItem],
) -> Result<(), OxiAudioError> {
    // Serialize all items to bytes first so we know the total size.
    let serialized: Vec<Vec<u8>> = items.iter().map(serialize_item).collect();
    let items_total_bytes: usize = serialized.iter().map(|b| b.len()).sum();

    // tag_size = items_total_bytes + footer size (32). NOT including the header.
    let tag_size = (items_total_bytes + APE_HEADER_SIZE) as u32;
    let item_count = items.len() as u32;

    // Write header.
    write_ape_block(writer, tag_size, item_count, APE_FLAG_HEADER)?;

    // Write all items.
    for item_bytes in &serialized {
        writer.write_all(item_bytes).map_err(OxiAudioError::Io)?;
    }

    // Write footer.
    write_ape_block(writer, tag_size, item_count, APE_FLAG_FOOTER)?;

    Ok(())
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{write_apev2, ApeItem, APE_HEADER_SIZE, APE_VERSION};

    fn make_items() -> Vec<ApeItem> {
        vec![
            ApeItem::new("Title", "Test"),
            ApeItem::new("Artist", "OxiAudio"),
        ]
    }

    #[test]
    fn test_apev2_header_magic() {
        let items = make_items();
        let mut out = Vec::new();
        write_apev2(&mut out, &items).expect("write_apev2 should succeed");
        assert_eq!(&out[..8], b"APETAGEX", "first 8 bytes must be 'APETAGEX'");
    }

    #[test]
    fn test_apev2_version_and_count() {
        let items = make_items();
        let mut out = Vec::new();
        write_apev2(&mut out, &items).expect("write_apev2 should succeed");

        // Bytes 8..12: version = 2000 as u32 LE
        let version = u32::from_le_bytes(out[8..12].try_into().expect("version slice"));
        assert_eq!(version, APE_VERSION, "version must be 2000");

        // Bytes 16..20: item_count = 2 as u32 LE (in the header)
        let item_count = u32::from_le_bytes(out[16..20].try_into().expect("item_count slice"));
        assert_eq!(item_count, 2, "item_count must be 2");
    }

    #[test]
    fn test_apev2_roundtrip_title() {
        let items = vec![
            ApeItem::new("Title", "Hello"),
            ApeItem::new("Artist", "OxiAudio"),
        ];
        let mut out = Vec::new();
        write_apev2(&mut out, &items).expect("write_apev2 should succeed");

        // The key "Title" (ASCII) must appear somewhere in the output.
        let needle = b"Title";
        let found = out.windows(needle.len()).any(|w| w == needle);
        assert!(found, "'Title' key must appear in APEv2 output");
    }

    #[test]
    fn test_apev2_minimum_size() {
        // Even with no items: 32 (header) + 0 (items) + 32 (footer) = 64 bytes
        let items: Vec<ApeItem> = vec![];
        let mut out = Vec::new();
        write_apev2(&mut out, &items).expect("write_apev2 with no items should succeed");
        assert_eq!(
            out.len(),
            2 * APE_HEADER_SIZE,
            "empty APEv2 must be exactly 64 bytes"
        );
        // Both header and footer start with preamble
        assert_eq!(&out[..8], b"APETAGEX");
        assert_eq!(&out[32..40], b"APETAGEX");
    }

    #[test]
    fn test_apev2_tag_size_field() {
        // tag_size = items_bytes + 32 (footer). The same value appears in header and footer.
        let items = make_items();
        let mut out = Vec::new();
        write_apev2(&mut out, &items).expect("write_apev2");

        let header_tag_size = u32::from_le_bytes(out[12..16].try_into().expect("slice"));
        let footer_offset = out.len() - APE_HEADER_SIZE;
        let footer_tag_size = u32::from_le_bytes(
            out[footer_offset + 12..footer_offset + 16]
                .try_into()
                .expect("slice"),
        );

        assert_eq!(
            header_tag_size, footer_tag_size,
            "header and footer tag_size must match"
        );
        // tag_size includes footer (32 bytes) but NOT header.
        // total = header(32) + items + footer(32) => tag_size = total - 32
        let expected_tag_size = (out.len() - APE_HEADER_SIZE) as u32;
        assert_eq!(
            header_tag_size, expected_tag_size,
            "tag_size must equal items + footer"
        );
    }

    #[test]
    fn test_apev2_footer_magic() {
        let items = make_items();
        let mut out = Vec::new();
        write_apev2(&mut out, &items).expect("write_apev2");
        // Footer starts at out.len() - 32
        let footer_start = out.len() - APE_HEADER_SIZE;
        assert_eq!(
            &out[footer_start..footer_start + 8],
            b"APETAGEX",
            "footer must also start with APETAGEX"
        );
    }
}
