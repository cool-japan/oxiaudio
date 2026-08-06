//! Cross-crate regression test for OGG page CRC-32 validation.
//!
//! `oxiaudio_decode::ogg_reader::OggReader` validates every page's CRC-32
//! checksum (RFC 3533 §6.3) before handing packet data to the caller. This test
//! exercises the full production round trip: pages are framed with
//! `oxiaudio_encode::OggStream` (table-driven CRC-32 in `oxiaudio_encode::ogg`)
//! and read back with `oxiaudio_decode::ogg_reader::OggReader` (bit-by-bit
//! CRC-32), proving the two independently-implemented checksums agree, and that
//! a page corrupted after encoding is rejected instead of being silently handed
//! to the downstream Opus/Vorbis decoder unverified.

use oxiaudio_core::OxiAudioError;
use oxiaudio_decode::ogg_reader::OggReader;
use oxiaudio_encode::OggStream;
use std::io::Cursor;

/// Two packets written through the real encoder must round-trip byte-for-byte
/// through the real decoder, with the CRC-32 on both pages validating cleanly.
#[test]
fn test_ogg_stream_encode_decode_roundtrip_crc_valid() {
    let mut buf = Vec::new();
    {
        let mut stream = OggStream::new(&mut buf, 0xC0FF_EE01);
        stream
            .write_packet(b"OpusHead-like first packet", 0, false)
            .expect("write_packet 1 must succeed");
        stream
            .write_packet(b"second packet with a different length", 960, true)
            .expect("write_packet 2 must succeed");
        stream.finish().expect("finish must succeed");
    }

    let mut reader = OggReader::new(Cursor::new(buf));
    let pkt1 = reader
        .read_packet()
        .expect("CRC-32 must validate for an unmodified encoder-written page")
        .expect("packet 1 must be present");
    assert_eq!(&pkt1, b"OpusHead-like first packet");

    let pkt2 = reader
        .read_packet()
        .expect("CRC-32 must validate for the second encoder-written page")
        .expect("packet 2 must be present");
    assert_eq!(&pkt2, b"second packet with a different length");

    let end = reader
        .read_packet()
        .expect("clean EOS must not be an error");
    assert!(end.is_none(), "stream must be exhausted after two packets");
}

/// Flipping a single payload byte after encoding must be caught by CRC-32
/// validation rather than silently reaching the caller with corrupted data.
#[test]
fn test_ogg_stream_corrupted_payload_byte_rejected() {
    let payload = b"a payload that will be corrupted after encoding";
    let mut buf = Vec::new();
    {
        let mut stream = OggStream::new(&mut buf, 42);
        stream
            .write_packet(payload, 960, true)
            .expect("write_packet must succeed");
        stream.finish().expect("finish must succeed");
    }

    // Single-segment page layout: 27-byte fixed header + 1-byte segment table
    // (payload < 255 bytes) precede the page data. Flip one payload byte,
    // leaving every structural field (lengths, magic, header_type) untouched.
    const HEADER_LEN: usize = 27 + 1;
    assert!(
        buf.len() > HEADER_LEN + 10,
        "test payload must be long enough to corrupt a byte inside page data"
    );
    buf[HEADER_LEN + 10] ^= 0xFF;

    let mut reader = OggReader::new(Cursor::new(buf));
    let err = reader
        .read_packet()
        .expect_err("a corrupted payload byte must fail CRC-32 validation");
    assert!(
        matches!(err, OxiAudioError::Decode(_)),
        "expected a typed Decode error for CRC mismatch, got {err:?}"
    );
}

/// Flipping a header field (the granule position) after encoding must also be
/// caught, since the CRC-32 covers the whole page, not just the payload.
#[test]
fn test_ogg_stream_corrupted_header_field_rejected() {
    let mut buf = Vec::new();
    {
        let mut stream = OggStream::new(&mut buf, 7);
        stream
            .write_packet(b"packet with a header that gets corrupted", 480, true)
            .expect("write_packet must succeed");
        stream.finish().expect("finish must succeed");
    }

    // Granule position occupies bytes 6..14 (i64 LE) of the page header.
    buf[6] ^= 0xFF;

    let mut reader = OggReader::new(Cursor::new(buf));
    let err = reader
        .read_packet()
        .expect_err("a corrupted header field must fail CRC-32 validation");
    assert!(
        matches!(err, OxiAudioError::Decode(_)),
        "expected a typed Decode error for CRC mismatch, got {err:?}"
    );
}
