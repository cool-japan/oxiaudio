//! OGG bitstream packet reader per RFC 3533.
//!
//! Reads logical bitstream pages and reassembles them into complete packets.
//! Supports continuation pages, beginning-of-stream (BOS), and end-of-stream (EOS)
//! markers. Every page's CRC-32 checksum (polynomial 0x04C11DB7, RFC 3533 §6.3) is
//! validated against a freshly computed value before the page is accepted; a page
//! whose stored checksum does not match is rejected with [`OxiAudioError::Decode`]
//! rather than being handed to the downstream codec. This closes off attacker- or
//! corruption-induced bit flips reaching the Opus/Vorbis decoders unverified.

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
    /// Returns [`OxiAudioError::Decode`] on malformed OGG data — including a page
    /// whose stored CRC-32 checksum does not match the checksum recomputed over
    /// the received bytes (RFC 3533 §6.3); such a page is rejected outright and
    /// never reaches the caller, even partially — or [`OxiAudioError::Io`] on
    /// underlying I/O failures.
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
        //   checksum         u32 LE   4 bytes  (validated against a recomputed CRC-32)
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

        // checksum (4 bytes LE u32) — validated below once the whole page is read.
        let mut crc_buf = [0u8; 4];
        self.reader
            .read_exact(&mut crc_buf)
            .map_err(OxiAudioError::Io)?;
        let stored_crc = u32::from_le_bytes(crc_buf);

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

        // --- Validate the RFC 3533 §6.3 CRC-32 ---
        //
        // The checksum is computed over the entire page (header + segment table +
        // page data) with the checksum field itself zeroed. Reconstruct that exact
        // byte layout from the fields already parsed above and compare against the
        // value stored on the wire; reject the page on any mismatch instead of
        // handing unverified bytes to the downstream Opus/Vorbis decoder.
        let mut page_for_crc = Vec::with_capacity(27 + n_segments + total_data);
        page_for_crc.extend_from_slice(&magic);
        page_for_crc.push(version_buf[0]);
        page_for_crc.push(header_type);
        page_for_crc.extend_from_slice(&granule_buf);
        page_for_crc.extend_from_slice(&serial_buf);
        page_for_crc.extend_from_slice(&seq_buf);
        page_for_crc.extend_from_slice(&[0u8; 4]); // checksum field zeroed for computation
        page_for_crc.push(n_seg_buf[0]);
        page_for_crc.extend_from_slice(&segment_table);
        page_for_crc.extend_from_slice(&data);

        let computed_crc = ogg_crc32(&page_for_crc);
        if computed_crc != stored_crc {
            return Err(OxiAudioError::Decode(format!(
                "OGG page CRC-32 mismatch at sequence {seq_num} (serial {serial}): \
                 stored={stored_crc:#010X}, computed={computed_crc:#010X}"
            )));
        }

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
/// Used by `OggReader::read_page` to validate every page's checksum field
/// (bit-by-bit, MSB-first, initial value 0, no final XOR — cross-checked against
/// the table-driven implementation in `oxiaudio_encode::ogg::ogg_crc32`, which
/// produces identical output for the same input).
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

    /// Build a minimal OGG page with a single segment and a correct CRC-32.
    ///
    /// Header layout (per RFC 3533):
    ///   "OggS" + version(0) + header_type + granule_pos(0) + serial(1) + seq(0) + crc + n_segs(1) + seg_table + data
    ///
    /// The checksum field is written as 0 while building the page, then patched
    /// in-place with the real CRC-32 computed over the whole page (matching the
    /// encoder-side convention in `oxiaudio_encode::ogg::write_ogg_page_raw`), so
    /// pages built by this helper pass [`OggReader`]'s CRC validation.
    fn build_ogg_page(header_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut page = Vec::new();
        page.extend_from_slice(b"OggS");
        page.push(0); // version
        page.push(header_type);
        page.extend_from_slice(&0i64.to_le_bytes()); // granule_pos
        page.extend_from_slice(&1u32.to_le_bytes()); // serial
        page.extend_from_slice(&0u32.to_le_bytes()); // seq_num
        page.extend_from_slice(&0u32.to_le_bytes()); // crc placeholder, patched below
                                                     // Segment table: one segment per chunk of ≤ 255 bytes.
        let chunks: Vec<&[u8]> = payload.chunks(255).collect();
        page.push(chunks.len() as u8); // n_segments
        for chunk in &chunks {
            page.push(chunk.len() as u8);
        }
        for chunk in &chunks {
            page.extend_from_slice(chunk);
        }
        let crc = ogg_crc32(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());
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
        page.extend_from_slice(&0u32.to_le_bytes()); // crc placeholder, patched below
        page.push(2); // n_segments = 2
        page.push(2); // seg[0] = 2 bytes → packet 1
        page.push(2); // seg[1] = 2 bytes → packet 2
        page.extend_from_slice(b"ABCD");
        let crc = ogg_crc32(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());

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
        // Known-answer vector independently cross-checked against a table-driven
        // reference implementation of the same RFC 3533 §6.3 polynomial (also used
        // by `oxiaudio_encode::ogg::ogg_crc32::test_ogg_crc32_deterministic`).
        let data = b"OggS\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x01";
        assert_eq!(ogg_crc32(data), 0x49a1_d5fd);
    }

    // ─── CRC-32 validation regression tests ───────────────────────────────────

    #[test]
    fn test_valid_page_with_correct_crc_is_accepted() {
        // A page built with the real, patched-in CRC-32 must be read successfully.
        // (This is also exercised implicitly by every other test in this module
        // now that `build_ogg_page` embeds a correct checksum, but this test names
        // the acceptance path explicitly.)
        let page = build_ogg_page(0x02 | 0x04, b"valid payload"); // BOS|EOS
        let mut reader = OggReader::new(Cursor::new(page));
        let pkt = reader
            .read_packet()
            .expect("a page with a correct CRC-32 must be accepted")
            .expect("packet must be present");
        assert_eq!(&pkt, b"valid payload");
    }

    #[test]
    fn test_corrupted_payload_byte_rejected_by_crc_check() {
        // Build a page with a correct CRC-32, then flip one bit in the payload
        // (the last byte of the page) without updating the checksum. The reader
        // must reject the page instead of handing corrupted bytes to the caller.
        let mut page = build_ogg_page(0x02, b"ABC");
        let last = page.len() - 1;
        page[last] ^= 0xFF;

        let mut reader = OggReader::new(Cursor::new(page));
        let err = reader
            .read_packet()
            .expect_err("a payload bit-flip must invalidate the CRC-32 and be rejected");
        assert!(
            matches!(err, OxiAudioError::Decode(_)),
            "expected Decode error for CRC mismatch, got {err:?}"
        );
    }

    #[test]
    fn test_corrupted_header_field_rejected_by_crc_check() {
        // Corrupt the serial-number field (bytes 14..18) directly, leaving the
        // stored CRC-32 pointing at the original (uncorrupted) header. The CRC
        // covers the whole page, so header tampering must be caught too, not
        // just payload tampering.
        let mut page = build_ogg_page(0x02, b"XYZ");
        page[14] ^= 0xFF;

        let mut reader = OggReader::new(Cursor::new(page));
        let err = reader
            .read_packet()
            .expect_err("a corrupted header field must invalidate the CRC-32 and be rejected");
        assert!(
            matches!(err, OxiAudioError::Decode(_)),
            "expected Decode error for CRC mismatch, got {err:?}"
        );
    }

    #[test]
    fn test_crc_mismatch_error_message_contains_diagnostics() {
        // The error message should carry enough context (sequence number, stored
        // vs. computed CRC) to debug a real corrupted stream.
        let mut page = build_ogg_page(0x02, b"diagnostic payload");
        let last = page.len() - 1;
        page[last] ^= 0xFF;

        let mut reader = OggReader::new(Cursor::new(page));
        let err = reader
            .read_packet()
            .expect_err("must reject corrupted page");
        let OxiAudioError::Decode(msg) = err else {
            panic!("expected Decode error, got {err:?}");
        };
        assert!(
            msg.contains("CRC-32") && msg.contains("stored") && msg.contains("computed"),
            "error message should mention CRC-32 stored/computed values for debuggability, got: {msg}"
        );
    }
}
