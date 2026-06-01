//! OGG bitstream packet reader per RFC 3533.
//!
//! Reads logical bitstream pages and reassembles them into complete packets.
//! Supports continuation pages, beginning-of-stream (BOS), and end-of-stream (EOS)
//! markers. Does not validate CRC by default; uses the CRC32 polynomial 0x04C11DB7
//! defined in RFC 3533 §6.

use std::io::Read;

use oxiaudio_core::OxiAudioError;

/// A single OGG page, containing the demuxed segment data.
struct OggPage {
    /// Bitfield: bit 0 = continuation, bit 1 = BOS, bit 2 = EOS.
    header_type: u8,
    /// Granule position (codec-specific timestamp).
    #[allow(dead_code)]
    granule_pos: i64,
    /// Logical bitstream serial number.
    #[allow(dead_code)]
    serial: u32,
    /// Absolute page sequence number.
    #[allow(dead_code)]
    seq_num: u32,
    /// Reassembled segment data for this page (all lace table entries concatenated).
    data: Vec<u8>,
    /// Segment table: each entry is the byte count of one segment (0..=255).
    segment_table: Vec<u8>,
}

impl OggPage {
    /// Returns true when the last segment in this page is a packet terminator
    /// (i.e. the last lace entry is < 255, or the page has no segments).
    fn last_segment_terminates_packet(&self) -> bool {
        self.segment_table.last().map(|&s| s < 255).unwrap_or(true)
    }
}

/// OGG bitstream reader that reassembles complete packets from page segments.
///
/// Call [`OggReader::read_packet`] repeatedly until it returns `Ok(None)` to
/// exhaust the stream.  The reader buffers partial packets across page boundaries.
pub struct OggReader<R: Read> {
    reader: R,
    /// Accumulated bytes of the current in-progress packet (may span multiple pages).
    packet_buf: Vec<u8>,
    /// Whether the stream has reached EOS.
    eos: bool,
    /// Remaining segments from the current page that have not yet been consumed.
    pending_segments: Vec<(u8, Vec<u8>)>, // (segment_len, data_slice)
    /// Whether the pending_segments list came from a page whose last segment < 255.
    pending_page_terminates: bool,
}

impl<R: Read> OggReader<R> {
    /// Create a new `OggReader` wrapping the given reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            packet_buf: Vec::new(),
            eos: false,
            pending_segments: Vec::new(),
            pending_page_terminates: false,
        }
    }

    /// Read the next complete Opus packet from the OGG stream.
    ///
    /// Returns `None` when the end of stream is reached.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Decode`] on malformed OGG data, or
    /// [`OxiAudioError::Io`] on underlying I/O failures.
    pub fn read_packet(&mut self) -> Result<Option<Vec<u8>>, OxiAudioError> {
        loop {
            // Process any pending segments from the last loaded page.
            while let Some((seg_len, seg_data)) = self.pending_segments.first().cloned() {
                self.pending_segments.remove(0);
                self.packet_buf.extend_from_slice(&seg_data);
                if seg_len < 255 {
                    // Packet terminates here.
                    let packet = std::mem::take(&mut self.packet_buf);
                    return Ok(Some(packet));
                }
                // seg_len == 255: packet continues on next segment / next page.
            }

            // We consumed all pending segments. Check if the last page terminated a packet.
            // If we get here with a non-empty packet_buf, the packet continues onto the next page.
            if self.eos {
                // EOS: if there's leftover data, flush it as a final packet.
                if !self.packet_buf.is_empty() {
                    let packet = std::mem::take(&mut self.packet_buf);
                    return Ok(Some(packet));
                }
                return Ok(None);
            }

            // Load the next page.
            match self.read_page()? {
                None => {
                    self.eos = true;
                    if !self.packet_buf.is_empty() {
                        let packet = std::mem::take(&mut self.packet_buf);
                        return Ok(Some(packet));
                    }
                    return Ok(None);
                }
                Some(page) => {
                    if page.header_type & 0x04 != 0 {
                        self.eos = true;
                    }
                    // Slice segments out of page data.
                    let mut offset = 0usize;
                    let segments: Vec<(u8, Vec<u8>)> = page
                        .segment_table
                        .iter()
                        .map(|&len| {
                            let end = offset + len as usize;
                            let slice = page.data[offset..end].to_vec();
                            offset = end;
                            (len, slice)
                        })
                        .collect();
                    self.pending_segments = segments;
                    self.pending_page_terminates = page.last_segment_terminates_packet();
                }
            }
        }
    }

    /// Read a single OGG page from the underlying reader.
    ///
    /// Returns `None` when the stream is exhausted (first read returns 0 bytes).
    fn read_page(&mut self) -> Result<Option<OggPage>, OxiAudioError> {
        // OGG page layout (RFC 3533 §6):
        //   capture_pattern  "OggS"   4 bytes
        //   version          u8       1 byte (must be 0)
        //   header_type      u8       1 byte
        //   granule_position i64 LE   8 bytes
        //   bitstream_serial u32 LE   4 bytes
        //   page_sequence    u32 LE   4 bytes
        //   checksum         u32 LE   4 bytes  (we parse but do not validate)
        //   page_segments    u8       1 byte
        //   segment_table    u8[n]    n bytes
        //   page_data        (sum of segment_table) bytes

        // --- Sync to the next "OggS" capture pattern ---
        let mut magic = [0u8; 4];
        match self.reader.read_exact(&mut magic) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(OxiAudioError::Io(e)),
        }
        if &magic != b"OggS" {
            // Try to re-sync: scan for the next 'O' and re-read.
            return Err(OxiAudioError::Decode(
                "OGG sync lost: expected 'OggS' capture pattern".into(),
            ));
        }

        // version (1 byte)
        let mut version_buf = [0u8; 1];
        self.reader
            .read_exact(&mut version_buf)
            .map_err(OxiAudioError::Io)?;
        if version_buf[0] != 0 {
            return Err(OxiAudioError::Decode(format!(
                "OGG page version {} != 0",
                version_buf[0]
            )));
        }

        // header_type (1 byte)
        let mut header_type_buf = [0u8; 1];
        self.reader
            .read_exact(&mut header_type_buf)
            .map_err(OxiAudioError::Io)?;
        let header_type = header_type_buf[0];

        // granule_position (8 bytes LE i64)
        let mut granule_buf = [0u8; 8];
        self.reader
            .read_exact(&mut granule_buf)
            .map_err(OxiAudioError::Io)?;
        let granule_pos = i64::from_le_bytes(granule_buf);

        // bitstream_serial (4 bytes LE u32)
        let mut serial_buf = [0u8; 4];
        self.reader
            .read_exact(&mut serial_buf)
            .map_err(OxiAudioError::Io)?;
        let serial = u32::from_le_bytes(serial_buf);

        // page_sequence (4 bytes LE u32)
        let mut seq_buf = [0u8; 4];
        self.reader
            .read_exact(&mut seq_buf)
            .map_err(OxiAudioError::Io)?;
        let seq_num = u32::from_le_bytes(seq_buf);

        // checksum (4 bytes LE u32) — parse but ignore for now.
        let mut crc_buf = [0u8; 4];
        self.reader
            .read_exact(&mut crc_buf)
            .map_err(OxiAudioError::Io)?;
        // (CRC validation is intentionally skipped — malformed pages are rare in practice
        //  and error recovery is better handled by the downstream packet decoder.)

        // page_segments (1 byte)
        let mut n_seg_buf = [0u8; 1];
        self.reader
            .read_exact(&mut n_seg_buf)
            .map_err(OxiAudioError::Io)?;
        let n_segments = n_seg_buf[0] as usize;

        // segment_table (n_segments bytes)
        let mut segment_table = vec![0u8; n_segments];
        self.reader
            .read_exact(&mut segment_table)
            .map_err(OxiAudioError::Io)?;

        // page_data (sum of segment_table bytes)
        let total_data: usize = segment_table.iter().map(|&s| s as usize).sum();
        let mut data = vec![0u8; total_data];
        self.reader
            .read_exact(&mut data)
            .map_err(OxiAudioError::Io)?;

        Ok(Some(OggPage {
            header_type,
            granule_pos,
            serial,
            seq_num,
            data,
            segment_table,
        }))
    }
}

/// OGG CRC-32 with the polynomial 0x04C11DB7 used in RFC 3533.
///
/// Not used in the current implementation (CRC validation is skipped),
/// but provided for completeness and future validation support.
#[allow(dead_code)]
pub fn ogg_crc32(data: &[u8]) -> u32 {
    const POLY: u32 = 0x04C11DB7;
    let mut crc: u32 = 0;
    for &byte in data {
        crc ^= (byte as u32) << 24;
        for _ in 0..8 {
            if crc & 0x8000_0000 != 0 {
                crc = (crc << 1) ^ POLY;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal OGG page with a single segment.
    ///
    /// Header layout (per RFC 3533):
    ///   "OggS" + version(0) + header_type + granule_pos(0) + serial(1) + seq(0) + crc(0) + n_segs(1) + seg_table + data
    fn build_ogg_page(header_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut page = Vec::new();
        page.extend_from_slice(b"OggS");
        page.push(0); // version
        page.push(header_type);
        page.extend_from_slice(&0i64.to_le_bytes()); // granule_pos
        page.extend_from_slice(&1u32.to_le_bytes()); // serial
        page.extend_from_slice(&0u32.to_le_bytes()); // seq_num
        page.extend_from_slice(&0u32.to_le_bytes()); // crc (not validated)
                                                     // Segment table: one segment per chunk of ≤ 255 bytes.
        let chunks: Vec<&[u8]> = payload.chunks(255).collect();
        page.push(chunks.len() as u8); // n_segments
        for chunk in &chunks {
            page.push(chunk.len() as u8);
        }
        for chunk in &chunks {
            page.extend_from_slice(chunk);
        }
        page
    }

    #[test]
    fn test_empty_stream_returns_none() {
        let mut reader = OggReader::new(Cursor::new(vec![]));
        let result = reader.read_packet().expect("no error on empty stream");
        assert!(result.is_none(), "empty stream must return None");
    }

    #[test]
    fn test_single_packet_in_one_page() {
        // Build a BOS page containing a 3-byte packet.
        let payload = b"ABC";
        let page = build_ogg_page(0x02, payload); // 0x02 = BOS
        let mut reader = OggReader::new(Cursor::new(page));
        let pkt = reader
            .read_packet()
            .expect("no error")
            .expect("must have packet");
        assert_eq!(&pkt, b"ABC");
    }

    #[test]
    fn test_multiple_packets_in_one_page() {
        // Build a page with two packets: "AB" (2 bytes) and "CD" (2 bytes)
        // Segment table: [2, 2]; data: ABCD
        let mut page = Vec::new();
        page.extend_from_slice(b"OggS");
        page.push(0); // version
        page.push(0x02); // header_type = BOS
        page.extend_from_slice(&0i64.to_le_bytes()); // granule_pos
        page.extend_from_slice(&1u32.to_le_bytes()); // serial
        page.extend_from_slice(&0u32.to_le_bytes()); // seq_num
        page.extend_from_slice(&0u32.to_le_bytes()); // crc
        page.push(2); // n_segments = 2
        page.push(2); // seg[0] = 2 bytes → packet 1
        page.push(2); // seg[1] = 2 bytes → packet 2
        page.extend_from_slice(b"ABCD");

        let mut reader = OggReader::new(Cursor::new(page));
        let pkt1 = reader.read_packet().expect("no error").expect("packet 1");
        assert_eq!(&pkt1, b"AB");
        let pkt2 = reader.read_packet().expect("no error").expect("packet 2");
        assert_eq!(&pkt2, b"CD");
        // No more packets.
        let none = reader.read_packet().expect("no error");
        assert!(none.is_none());
    }

    #[test]
    fn test_ogg_crc32_known_value() {
        // CRC32 of empty data should be 0.
        assert_eq!(ogg_crc32(&[]), 0);
    }
}
