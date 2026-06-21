//! OGG container writer per RFC 3533.
//!
//! Implements page segmentation, CRC-32 computation (OGG-specific polynomial),
//! granule position tracking, and a high-level stream writer with automatic page
//! sequencing. Supports multi-packet pages and large packets split across pages.
//!
//! This module also provides a Vorbis comment packet writer used by both Vorbis
//! and Opus streams (OGG headers share the same comment vector structure).

use std::io::Write;

use oxiaudio_core::OxiAudioError;

// ─── CRC-32 (OGG-specific polynomial 0x04C11DB7) ─────────────────────────────

/// Pre-computed CRC-32 lookup table for the OGG polynomial `0x04C11DB7`.
///
/// The table is computed at compile time via `const fn` to avoid any runtime
/// allocation or lazy-init overhead.
const CRC_TABLE: [u32; 256] = build_crc_table();

const fn build_crc_table() -> [u32; 256] {
    let poly: u32 = 0x04C11DB7;
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = (i as u32) << 24;
        let mut j = 0;
        while j < 8 {
            if crc & 0x8000_0000 != 0 {
                crc = (crc << 1) ^ poly;
            } else {
                crc <<= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Compute the OGG CRC-32 over `data`.
///
/// Uses the OGG-specific polynomial `0x04C11DB7` (big-endian / MSB-first).
/// The initial register value is 0 and no final XOR is applied, matching
/// the specification in RFC 3533 §6.3.
pub fn ogg_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0;
    for &byte in data {
        let idx = ((crc >> 24) ^ u32::from(byte)) as usize;
        crc = (crc << 8) ^ CRC_TABLE[idx];
    }
    crc
}

// ─── Segment table helpers ────────────────────────────────────────────────────

/// Build the OGG lacing (segment table) for a packet of `data_len` bytes.
///
/// Per RFC 3533 §2: each segment is at most 255 bytes. A segment of exactly
/// 255 means "continued"; a segment of < 255 terminates the packet. If the
/// last segment happens to be exactly 255 bytes, an additional 0-byte segment
/// is appended to signal end-of-packet.
fn build_segment_table(data_len: usize) -> Vec<u8> {
    if data_len == 0 {
        return vec![0u8];
    }
    let full_segments = data_len / 255;
    let remainder = data_len % 255;
    let mut table = vec![255u8; full_segments];
    // Always append the remainder (may be 0, which terminates the packet).
    table.push(remainder as u8);
    table
}

// ─── Low-level page writer ────────────────────────────────────────────────────

/// Header type flags used in OGG page headers (RFC 3533 §6.1).
const HEADER_TYPE_CONTINUATION: u8 = 0x01;
const HEADER_TYPE_BOS: u8 = 0x02;
const HEADER_TYPE_EOS: u8 = 0x04;

/// Write one complete OGG page to `writer`.
///
/// The page layout (RFC 3533 §6):
///
/// ```text
/// capture_pattern:      b"OggS"            (4 bytes)
/// version:              0x00               (1 byte)
/// header_type_flag:     bitmask            (1 byte)
/// granule_position:     i64 LE             (8 bytes)
/// stream_serial_number: u32 LE             (4 bytes)
/// page_sequence_number: u32 LE             (4 bytes)
/// checksum:             u32 LE CRC-32      (4 bytes, computed over page with field zeroed)
/// page_segments:        u8                 (1 byte)
/// segment_table:        [u8; page_segments](page_segments bytes)
/// page_data:            raw bytes          (sum of segment_table values)
/// ```
///
/// # Arguments
///
/// * `writer`      — Output sink.
/// * `data`        — Packet bytes for this page (may be a slice of a larger packet).
/// * `serial`      — Stream serial number (constant for all pages of one logical stream).
/// * `granule_pos` — Cumulative sample position at the last complete packet on this page;
///   `-1` (`0xFFFF_FFFF_FFFF_FFFF`) when no complete packet ends here.
/// * `seq_num`     — Monotonically increasing page sequence number (0-based).
/// * `header_type` — Bitmask of `HEADER_TYPE_*` constants.
///
/// # Errors
///
/// Returns [`OxiAudioError::Encode`] if `data` does not fit in a single OGG page
/// (max 65 025 bytes = 255 segments × 255 bytes each), or on any I/O failure.
pub fn write_ogg_page_raw<W: Write>(
    writer: &mut W,
    data: &[u8],
    serial: u32,
    granule_pos: i64,
    seq_num: u32,
    header_type: u8,
) -> Result<(), OxiAudioError> {
    let segment_table = build_segment_table(data.len());
    // An OGG page can hold at most 255 segment-table entries.
    if segment_table.len() > 255 {
        return Err(OxiAudioError::Encode(format!(
            "OGG page data too large: {} bytes (max {})",
            data.len(),
            MAX_PAGE_PAYLOAD,
        )));
    }
    let n_segments = segment_table.len() as u8;

    // Build the full page in memory to compute the CRC in one pass.
    let header_size = 27 + usize::from(n_segments);
    let total_size = header_size + data.len();
    let mut page = Vec::with_capacity(total_size);

    // capture_pattern
    page.extend_from_slice(b"OggS");
    // version
    page.push(0x00);
    // header_type_flag
    page.push(header_type);
    // granule_position (i64 LE)
    page.extend_from_slice(&granule_pos.to_le_bytes());
    // stream_serial_number (u32 LE)
    page.extend_from_slice(&serial.to_le_bytes());
    // page_sequence_number (u32 LE)
    page.extend_from_slice(&seq_num.to_le_bytes());
    // checksum placeholder (4 zero bytes)
    page.extend_from_slice(&[0u8; 4]);
    // page_segments
    page.push(n_segments);
    // segment_table
    page.extend_from_slice(&segment_table);
    // page_data
    page.extend_from_slice(data);

    // Compute CRC over the complete page (with the checksum field still zero).
    let crc = ogg_crc32(&page);
    // Overwrite the checksum field at offset 22 (little-endian).
    let crc_bytes = crc.to_le_bytes();
    page[22] = crc_bytes[0];
    page[23] = crc_bytes[1];
    page[24] = crc_bytes[2];
    page[25] = crc_bytes[3];

    writer.write_all(&page).map_err(OxiAudioError::Io)
}

/// Write one OGG page with explicit BOS / EOS / continuation control.
///
/// This is the public API for single-page packet writes. For large packets that
/// must span multiple pages, use [`OggStream::write_packet`] instead.
///
/// # Errors
///
/// Returns [`OxiAudioError::Encode`] if `data` exceeds the maximum OGG page
/// payload (65 025 bytes), or [`OxiAudioError::Io`] on write failure.
pub fn write_ogg_page<W: Write>(
    writer: &mut W,
    data: &[u8],
    serial: u32,
    granule_pos: i64,
    seq_num: u32,
    is_first: bool,
    is_last: bool,
) -> Result<(), OxiAudioError> {
    let mut header_type: u8 = 0;
    if is_first {
        header_type |= HEADER_TYPE_BOS;
    }
    if is_last {
        header_type |= HEADER_TYPE_EOS;
    }
    write_ogg_page_raw(writer, data, serial, granule_pos, seq_num, header_type)
}

// ─── Maximum bytes per page ───────────────────────────────────────────────────
//
// An OGG page holds at most 255 segments. The last segment terminates the packet
// with a value < 255. When the data is a multiple of 255 bytes a zero-byte
// terminator segment is added, so the worst case is:
//   254 segments × 255 bytes + 1 segment × 254 bytes = 64 770 + 254 = 65 024
// For a cleaner split we use 255 * 254 + 254 = 65_024.
// If data is exactly a multiple of 255, the table gains an extra 0-byte
// terminator and would overflow 255 entries, so we cap at 65_024.
const MAX_PAGE_PAYLOAD: usize = 255 * 254 + 254; // 65_024

// ─── High-level OGG stream writer ─────────────────────────────────────────────

/// High-level OGG logical stream writer.
///
/// Manages page sequencing, granule-position tracking, and BOS/EOS page emission.
/// Large packets are automatically split across multiple pages.
///
/// # Usage
///
/// ```no_run
/// use std::io::Cursor;
/// use oxiaudio_encode::ogg::OggStream;
///
/// let mut buf = Cursor::new(Vec::new());
/// let mut stream = OggStream::new(&mut buf, 0x12345678);
/// stream.write_packet(b"OpusHead\x01\x02...", 0, false).unwrap();
/// stream.write_packet(b"audio frame data", 960, true).unwrap();
/// stream.finish().unwrap();
/// ```
pub struct OggStream<W: Write> {
    writer: W,
    serial: u32,
    seq_num: u32,
    granule_pos: i64,
    first_page: bool,
    eos_written: bool,
}

impl<W: Write> OggStream<W> {
    /// Create a new `OggStream` writing to `writer` with the given `serial` number.
    ///
    /// The serial number must be unique across all logical streams multiplexed into
    /// the same physical OGG bitstream (for single-stream files any value works).
    pub fn new(writer: W, serial: u32) -> Self {
        Self {
            writer,
            serial,
            seq_num: 0,
            granule_pos: 0,
            first_page: true,
            eos_written: false,
        }
    }

    /// Write one logical packet to the stream.
    ///
    /// Large packets are automatically split across multiple OGG pages. Granule
    /// position advances by `granule_delta` after this packet (typically the number
    /// of audio frames encoded in the packet).
    ///
    /// Set `is_last` to `true` for the final packet; an EOS page is written.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Io`] on write failure or
    /// [`OxiAudioError::Encode`] if a page exceeds the OGG size limit.
    pub fn write_packet(
        &mut self,
        data: &[u8],
        granule_delta: i64,
        is_last: bool,
    ) -> Result<(), OxiAudioError> {
        let mut remaining = data;
        let mut first_chunk = true;

        loop {
            let chunk_len = remaining.len().min(MAX_PAGE_PAYLOAD);
            let chunk = &remaining[..chunk_len];
            remaining = &remaining[chunk_len..];
            let is_final_chunk = remaining.is_empty();

            let mut header_type: u8 = 0;
            if !first_chunk {
                header_type |= HEADER_TYPE_CONTINUATION;
            }
            if self.first_page {
                header_type |= HEADER_TYPE_BOS;
                self.first_page = false;
            }

            // Granule position is only set on the last page of a packet; intermediate
            // pages carry −1 (all bits set) per RFC 3533 §3.
            let page_granule = if is_final_chunk {
                self.granule_pos + granule_delta
            } else {
                -1i64
            };

            if is_last && is_final_chunk {
                header_type |= HEADER_TYPE_EOS;
                self.eos_written = true;
            }

            write_ogg_page_raw(
                &mut self.writer,
                chunk,
                self.serial,
                page_granule,
                self.seq_num,
                header_type,
            )?;

            self.seq_num = self.seq_num.wrapping_add(1);
            first_chunk = false;

            if is_final_chunk {
                self.granule_pos += granule_delta;
                break;
            }
        }

        Ok(())
    }

    /// Finalise the stream.
    ///
    /// If no packet with `is_last = true` has been written yet, writes an empty
    /// EOS page to cleanly terminate the logical stream. Returns the underlying
    /// writer so callers can inspect or re-use it.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Io`] on write failure.
    pub fn finish(mut self) -> Result<W, OxiAudioError> {
        if !self.eos_written {
            // Write an empty EOS page.
            let header_type = HEADER_TYPE_EOS | if self.first_page { HEADER_TYPE_BOS } else { 0 };
            write_ogg_page_raw(
                &mut self.writer,
                &[],
                self.serial,
                self.granule_pos,
                self.seq_num,
                header_type,
            )?;
        }
        Ok(self.writer)
    }

    /// Return the current granule position (cumulative frames written so far).
    pub fn granule_pos(&self) -> i64 {
        self.granule_pos
    }

    /// Return the next page sequence number to be written.
    pub fn seq_num(&self) -> u32 {
        self.seq_num
    }
}

// ─── Vorbis comment packet ─────────────────────────────────────────────────────

/// Write a Vorbis comment packet (used in both Vorbis and Opus streams).
///
/// The format of the returned `Vec<u8>` depends on `is_opus`:
///
/// * `is_opus = false` (Vorbis): packet type byte `0x03`, then `b"vorbis"`,
///   then the comment vector.
/// * `is_opus = true` (Opus): magic `b"OpusTags"`, then the comment vector.
///
/// The comment vector structure (per Vorbis I spec §5.2.1):
/// ```text
/// vendor_length: u32 LE
/// vendor_string: [u8; vendor_length]
/// comment_count: u32 LE
/// for each comment:
///     length: u32 LE
///     "KEY=value": [u8; length] (UTF-8)
/// ```
///
/// # Examples
///
/// ```
/// use oxiaudio_encode::ogg::write_vorbis_comment_packet;
///
/// let pkt = write_vorbis_comment_packet("OxiAudio 0.2.0", &[("TITLE", "Demo")], false);
/// assert!(pkt.starts_with(&[0x03])); // Vorbis packet type
/// ```
pub fn write_vorbis_comment_packet(
    vendor_string: &str,
    comments: &[(&str, &str)],
    is_opus: bool,
) -> Vec<u8> {
    let vendor_bytes = vendor_string.as_bytes();
    let vendor_len = vendor_bytes.len();

    // Pre-compute total comment bytes to avoid repeated allocation.
    let comments_data_len: usize = comments
        .iter()
        .map(|(k, v)| {
            // Each comment is encoded as "KEY=value"
            let entry_len = k.len() + 1 + v.len(); // +1 for '='
            4 + entry_len // u32 length prefix
        })
        .sum();

    let prefix_len = if is_opus {
        8 // "OpusTags"
    } else {
        7 // 0x03 + "vorbis"
    };
    let total = prefix_len + 4 + vendor_len + 4 + comments_data_len;
    let mut pkt = Vec::with_capacity(total);

    // Write prefix / magic
    if is_opus {
        pkt.extend_from_slice(b"OpusTags");
    } else {
        pkt.push(0x03); // comment header type
        pkt.extend_from_slice(b"vorbis");
    }

    // vendor_length + vendor_string
    pkt.extend_from_slice(&(vendor_len as u32).to_le_bytes());
    pkt.extend_from_slice(vendor_bytes);

    // comment_count
    pkt.extend_from_slice(&(comments.len() as u32).to_le_bytes());

    // individual comments
    for (key, value) in comments {
        let entry = format!("{}={}", key, value);
        let entry_bytes = entry.as_bytes();
        pkt.extend_from_slice(&(entry_bytes.len() as u32).to_le_bytes());
        pkt.extend_from_slice(entry_bytes);
    }

    pkt
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CRC-32 ───────────────────────────────────────────────────────────────

    #[test]
    fn test_ogg_crc32_deterministic() {
        // RFC 3533 does not specify a test vector; verify determinism.
        let data = b"OggS\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x01";
        let crc1 = ogg_crc32(data);
        let crc2 = ogg_crc32(data);
        assert_eq!(crc1, crc2, "CRC must be deterministic for identical input");
        // Different data must yield a different CRC (with overwhelming probability).
        let other = b"OggS\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xFF\xFF";
        let crc3 = ogg_crc32(other);
        assert_ne!(crc1, crc3, "CRC must differ for different input");
    }

    #[test]
    fn test_ogg_crc32_empty() {
        // CRC of empty input must be zero (initial register = 0, no bytes processed).
        assert_eq!(ogg_crc32(&[]), 0);
    }

    // ── write_ogg_page ────────────────────────────────────────────────────────

    #[test]
    fn test_ogg_page_starts_with_oggs_magic() {
        let mut out = Vec::new();
        write_ogg_page(&mut out, b"test data", 12345, 0, 0, true, false)
            .expect("write_ogg_page must succeed");
        assert_eq!(
            &out[..4],
            b"OggS",
            "OGG page must start with capture pattern 'OggS'"
        );
    }

    #[test]
    fn test_ogg_page_version_byte_is_zero() {
        let mut out = Vec::new();
        write_ogg_page(&mut out, b"hello", 1, 0, 0, true, false)
            .expect("write_ogg_page must succeed");
        assert_eq!(out[4], 0x00, "OGG page version byte must be 0");
    }

    #[test]
    fn test_ogg_page_bos_flag_set_on_first() {
        let mut out = Vec::new();
        write_ogg_page(&mut out, b"x", 0, 0, 0, true, false).expect("write_ogg_page must succeed");
        assert_eq!(
            out[5] & HEADER_TYPE_BOS,
            HEADER_TYPE_BOS,
            "BOS flag must be set on first page"
        );
    }

    #[test]
    fn test_ogg_page_eos_flag_set_on_last() {
        let mut out = Vec::new();
        write_ogg_page(&mut out, b"x", 0, 0, 0, false, true).expect("write_ogg_page must succeed");
        assert_eq!(
            out[5] & HEADER_TYPE_EOS,
            HEADER_TYPE_EOS,
            "EOS flag must be set on last page"
        );
    }

    #[test]
    fn test_ogg_page_crc_field_matches_recomputed_crc() {
        let mut out = Vec::new();
        write_ogg_page(
            &mut out,
            b"some packet data here",
            99,
            48000,
            3,
            false,
            false,
        )
        .expect("write_ogg_page must succeed");

        // Extract the stored CRC from bytes 22-25.
        let stored_crc = u32::from_le_bytes([out[22], out[23], out[24], out[25]]);

        // Zero the CRC field and recompute.
        let mut page_zeroed = out.clone();
        page_zeroed[22] = 0;
        page_zeroed[23] = 0;
        page_zeroed[24] = 0;
        page_zeroed[25] = 0;
        let computed_crc = ogg_crc32(&page_zeroed);

        assert_eq!(
            stored_crc, computed_crc,
            "stored CRC {stored_crc:#010X} must match recomputed CRC {computed_crc:#010X}"
        );
    }

    #[test]
    fn test_ogg_page_granule_position_written_correctly() {
        let granule: i64 = 0x0102_0304_0506_0708;
        let mut out = Vec::new();
        write_ogg_page(&mut out, b"y", 0, granule, 0, false, false)
            .expect("write_ogg_page must succeed");

        // Granule position is at bytes 6-13 (i64 LE).
        let stored = i64::from_le_bytes([
            out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13],
        ]);
        assert_eq!(
            stored, granule,
            "granule position must round-trip through OGG page header"
        );
    }

    #[test]
    fn test_ogg_page_serial_number_written_correctly() {
        let serial: u32 = 0xDEAD_BEEF;
        let mut out = Vec::new();
        write_ogg_page(&mut out, b"z", serial, 0, 0, false, false)
            .expect("write_ogg_page must succeed");

        // Serial number at bytes 14-17 (u32 LE).
        let stored = u32::from_le_bytes([out[14], out[15], out[16], out[17]]);
        assert_eq!(
            stored, serial,
            "serial number must round-trip through OGG page header"
        );
    }

    #[test]
    fn test_ogg_page_sequence_number_written_correctly() {
        let seq: u32 = 42;
        let mut out = Vec::new();
        write_ogg_page(&mut out, b"w", 0, 0, seq, false, false)
            .expect("write_ogg_page must succeed");

        // Sequence number at bytes 18-21 (u32 LE).
        let stored = u32::from_le_bytes([out[18], out[19], out[20], out[21]]);
        assert_eq!(
            stored, seq,
            "sequence number must round-trip through OGG page header"
        );
    }

    #[test]
    fn test_ogg_page_empty_data_writes_valid_page() {
        let mut out = Vec::new();
        write_ogg_page(&mut out, &[], 0, 0, 0, true, true)
            .expect("write_ogg_page must succeed on empty data");
        assert_eq!(&out[..4], b"OggS", "empty OGG page must have OggS magic");
        // page_segments = 1, segment_table = [0], no page data
        assert_eq!(out[26], 1, "empty page must have exactly 1 segment");
        assert_eq!(out[27], 0, "empty page segment must be 0 (0-byte packet)");
    }

    // ── OggStream ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ogg_stream_single_packet_roundtrip() {
        let mut buf = Vec::new();
        {
            let mut stream = OggStream::new(&mut buf, 0xABCD_EF01);
            stream
                .write_packet(b"hello ogg stream", 100, true)
                .expect("write_packet must succeed");
            stream.finish().expect("finish must succeed");
        }
        assert_eq!(&buf[..4], b"OggS", "stream output must start with OggS");
        // BOS flag on the first page
        assert_eq!(buf[5] & HEADER_TYPE_BOS, HEADER_TYPE_BOS, "BOS must be set");
        // EOS flag must appear somewhere in the output
        let has_eos = buf
            .windows(28) // minimum page size with 1 segment
            .any(|w| w[0..4] == *b"OggS" && w[5] & HEADER_TYPE_EOS == HEADER_TYPE_EOS);
        assert!(has_eos, "EOS flag must appear in stream output");
    }

    #[test]
    fn test_ogg_stream_multiple_packets_sequence_numbers() {
        let mut buf = Vec::new();
        {
            let mut stream = OggStream::new(&mut buf, 1);
            for i in 0u8..5 {
                stream
                    .write_packet(&[i; 10], 100, i == 4)
                    .expect("write_packet");
            }
            stream.finish().expect("finish");
        }

        // Collect all page sequence numbers and verify they are monotonically
        // increasing (0, 1, 2, …).
        let mut seq_nums: Vec<u32> = Vec::new();
        let mut pos = 0usize;
        while pos + 27 <= buf.len() {
            if &buf[pos..pos + 4] == b"OggS" {
                let n_seg = buf[pos + 26] as usize;
                let seg_table_end = pos + 27 + n_seg;
                if seg_table_end > buf.len() {
                    break;
                }
                let data_len: usize = buf[pos + 27..seg_table_end]
                    .iter()
                    .map(|&b| b as usize)
                    .sum();
                let seq = u32::from_le_bytes([
                    buf[pos + 18],
                    buf[pos + 19],
                    buf[pos + 20],
                    buf[pos + 21],
                ]);
                seq_nums.push(seq);
                pos = seg_table_end + data_len;
            } else {
                break;
            }
        }

        assert!(!seq_nums.is_empty(), "must find at least one OGG page");
        for (i, (&a, &b)) in seq_nums.iter().zip(seq_nums.iter().skip(1)).enumerate() {
            assert_eq!(
                b,
                a + 1,
                "page sequence numbers must be consecutive: seq[{i}]={a} seq[{}]={b}",
                i + 1
            );
        }
    }

    #[test]
    fn test_ogg_stream_large_packet_split_across_pages() {
        // A packet larger than MAX_PAGE_PAYLOAD must be split.
        let large_data: Vec<u8> = (0..=255u8).cycle().take(70_000).collect();
        let mut buf = Vec::new();
        {
            let mut stream = OggStream::new(&mut buf, 7);
            stream
                .write_packet(&large_data, 48_000, true)
                .expect("write_packet must succeed for large packet");
            stream.finish().expect("finish");
        }

        // Count the number of pages written.
        let mut page_count = 0usize;
        let mut pos = 0usize;
        while pos + 27 <= buf.len() {
            if &buf[pos..pos + 4] == b"OggS" {
                page_count += 1;
                let n_seg = buf[pos + 26] as usize;
                let seg_table_end = pos + 27 + n_seg;
                if seg_table_end > buf.len() {
                    break;
                }
                let data_len: usize = buf[pos + 27..seg_table_end]
                    .iter()
                    .map(|&b| b as usize)
                    .sum();
                pos = seg_table_end + data_len;
            } else {
                break;
            }
        }

        assert!(
            page_count >= 2,
            "70 000-byte packet must span at least 2 pages; got {page_count}"
        );
    }

    #[test]
    fn test_ogg_stream_finish_writes_eos_when_not_yet_written() {
        let mut buf = Vec::new();
        {
            let mut stream = OggStream::new(&mut buf, 5);
            // Write packet without EOS.
            stream
                .write_packet(b"data", 100, false)
                .expect("write_packet");
            // finish() must emit an EOS page.
            stream.finish().expect("finish must succeed");
        }
        let has_eos = buf
            .windows(27)
            .any(|w| w.len() >= 6 && &w[..4] == b"OggS" && w[5] & HEADER_TYPE_EOS != 0);
        assert!(has_eos, "finish() must write an EOS page if not yet done");
    }

    // ── write_vorbis_comment_packet ───────────────────────────────────────────

    #[test]
    fn test_vorbis_comment_packet_vorbis_type_byte() {
        let pkt = write_vorbis_comment_packet("VendorX", &[], false);
        assert_eq!(pkt[0], 0x03, "Vorbis comment packet must start with 0x03");
        assert_eq!(
            &pkt[1..7],
            b"vorbis",
            "Vorbis packet must have 'vorbis' identifier"
        );
    }

    #[test]
    fn test_vorbis_comment_packet_opus_magic() {
        let pkt = write_vorbis_comment_packet("OxiAudio", &[], true);
        assert_eq!(
            &pkt[..8],
            b"OpusTags",
            "Opus comment packet must start with 'OpusTags'"
        );
    }

    #[test]
    fn test_vorbis_comment_packet_vendor_string_roundtrip() {
        let vendor = "OxiAudio 0.2.0";
        let pkt = write_vorbis_comment_packet(vendor, &[], true);
        // After "OpusTags" (8 bytes): vendor_length (u32 LE) at bytes 8-11, vendor string at 12+
        let vlen = u32::from_le_bytes([pkt[8], pkt[9], pkt[10], pkt[11]]) as usize;
        assert_eq!(vlen, vendor.len(), "vendor string length must match");
        assert_eq!(
            &pkt[12..12 + vlen],
            vendor.as_bytes(),
            "vendor string bytes must match"
        );
    }

    #[test]
    fn test_vorbis_comment_packet_comment_count_and_content() {
        let comments = [("TITLE", "My Track"), ("ARTIST", "Cool Japan")];
        let pkt = write_vorbis_comment_packet("OxiAudio", &comments, true);

        // After "OpusTags" (8): vendor_length(4) + vendor_bytes(8) = offset 20
        let vendor_len = u32::from_le_bytes([pkt[8], pkt[9], pkt[10], pkt[11]]) as usize;
        let comment_count_offset = 12 + vendor_len;
        let comment_count = u32::from_le_bytes([
            pkt[comment_count_offset],
            pkt[comment_count_offset + 1],
            pkt[comment_count_offset + 2],
            pkt[comment_count_offset + 3],
        ]);
        assert_eq!(
            comment_count, 2,
            "comment count must be 2, got {comment_count}"
        );

        // Verify the literal bytes "TITLE=My Track" appear in the packet.
        let title_entry = b"TITLE=My Track";
        let has_title = pkt.windows(title_entry.len()).any(|w| w == title_entry);
        assert!(has_title, "'TITLE=My Track' must appear in packet bytes");
    }

    #[test]
    fn test_vorbis_comment_packet_empty_comments() {
        // Zero comments: comment_count field must be 0.
        let pkt = write_vorbis_comment_packet("V", &[], false);
        // After 0x03 "vorbis" (7 bytes) + vendor_length(4) + vendor_bytes(1) = offset 12
        let comment_count = u32::from_le_bytes([pkt[12], pkt[13], pkt[14], pkt[15]]);
        assert_eq!(comment_count, 0, "empty comment list must have count 0");
    }

    // ── build_segment_table helper ─────────────────────────────────────────────

    #[test]
    fn test_segment_table_zero_length() {
        let t = build_segment_table(0);
        assert_eq!(
            t,
            vec![0u8],
            "zero-length packet must have [0] segment table"
        );
    }

    #[test]
    fn test_segment_table_exact_255() {
        // A 255-byte packet: one segment of 255, then a 0-byte terminator.
        let t = build_segment_table(255);
        assert_eq!(t, vec![255u8, 0u8], "255-byte packet: [255, 0]");
    }

    #[test]
    fn test_segment_table_256() {
        // A 256-byte packet: [255, 1].
        let t = build_segment_table(256);
        assert_eq!(t, vec![255u8, 1u8], "256-byte packet: [255, 1]");
    }

    #[test]
    fn test_segment_table_small() {
        // A 100-byte packet: one segment of 100.
        let t = build_segment_table(100);
        assert_eq!(t, vec![100u8], "100-byte packet: [100]");
    }
}
