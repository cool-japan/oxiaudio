//! Hand-rolled ID3v2.4 tag writer (pure Rust, no unsafe).
//!
//! This module exists because LAME's built-in ID3 support has no `TRCK`
//! (track number) frame, so we craft the tag bytes ourselves and prepend
//! them to the MP3 stream.

use super::types::{AlbumArt, Mp3Tags};

/// Encode a 28-bit integer in ID3v2 synchsafe form (4 bytes, 7 bits each).
/// Used only for the tag-header size field (bytes 6–9).
fn encode_synchsafe(n: u32) -> [u8; 4] {
    [
        ((n >> 21) & 0x7F) as u8,
        ((n >> 14) & 0x7F) as u8,
        ((n >> 7) & 0x7F) as u8,
        (n & 0x7F) as u8,
    ]
}

/// Build one ID3v2.4 text frame (standard Txxx or similar).
///
/// Frame layout:
/// ```text
/// [frame_id: 4 bytes]   — ASCII tag name, e.g. b"TIT2"
/// [size:     4 bytes]   — payload length, plain big-endian u32 (NOT synchsafe in v2.4)
/// [flags:    2 bytes]   — 0x00 0x00
/// [encoding: 1 byte]    — 0x03 = UTF-8
/// [text:     N bytes]   — field.as_bytes() (UTF-8 encoded)
/// ```
fn build_text_frame(frame_id: &[u8; 4], text: &str) -> Vec<u8> {
    let text_bytes = text.as_bytes();
    // payload = encoding byte (1) + text bytes (UTF-8)
    let payload_len = 1usize + text_bytes.len();
    let size_be = (payload_len as u32).to_be_bytes();

    let mut frame = Vec::with_capacity(10 + text_bytes.len());
    frame.extend_from_slice(frame_id); // 4 bytes: frame ID
    frame.extend_from_slice(&size_be); // 4 bytes: size (plain big-endian)
    frame.extend_from_slice(&[0x00, 0x00]); // 2 bytes: flags
    frame.push(0x03); // 1 byte: encoding (UTF-8)
    frame.extend_from_slice(text_bytes); // N bytes: text (UTF-8)
    frame
}

/// Build a TXXX (user-defined text) ID3v2.4 frame.
///
/// Frame layout:
/// ```text
/// [frame_id: 4 bytes]       — b"TXXX"
/// [size:     4 bytes]       — payload length, plain big-endian u32
/// [flags:    2 bytes]       — 0x00 0x00
/// [encoding: 1 byte]        — 0x03 = UTF-8
/// [description: N bytes]    — key string bytes (UTF-8)
/// [null:     1 byte]        — 0x00 separator
/// [value:    M bytes]       — value string bytes (UTF-8)
/// ```
fn build_txxx_frame(key: &str, value: &str) -> Vec<u8> {
    let key_bytes = key.as_bytes();
    let val_bytes = value.as_bytes();
    // payload = encoding(1) + key_bytes(N) + null(1) + val_bytes(M)
    let payload_len = 1usize + key_bytes.len() + 1 + val_bytes.len();
    let size_be = (payload_len as u32).to_be_bytes();

    let mut frame = Vec::with_capacity(10 + payload_len);
    frame.extend_from_slice(b"TXXX"); // 4 bytes: frame ID
    frame.extend_from_slice(&size_be); // 4 bytes: size
    frame.extend_from_slice(&[0x00, 0x00]); // 2 bytes: flags
    frame.push(0x03); // 1 byte: encoding UTF-8
    frame.extend_from_slice(key_bytes); // N bytes: description key
    frame.push(0x00); // 1 byte: null separator
    frame.extend_from_slice(val_bytes); // M bytes: value
    frame
}

/// Build an APIC (attached picture) ID3v2.4 frame.
///
/// Frame layout:
/// ```text
/// [frame_id: 4 bytes]   — b"APIC"
/// [size:     4 bytes]   — payload length, plain big-endian u32
/// [flags:    2 bytes]   — 0x00 0x00
/// [encoding: 1 byte]    — 0x03 = UTF-8
/// [mime_type: N bytes]  — MIME type string bytes (UTF-8)
/// [null:     1 byte]    — 0x00 MIME terminator
/// [pic_type: 1 byte]    — 0x03 = front cover
/// [desc:     1 byte]    — 0x00 empty description (null-terminated)
/// [data:     M bytes]   — raw image bytes
/// ```
fn build_apic_frame(art: &AlbumArt) -> Vec<u8> {
    let mime_bytes = art.mime_type.as_bytes();
    // payload = encoding(1) + mime(N) + null(1) + pic_type(1) + desc_null(1) + data(M)
    let payload_len = 1usize + mime_bytes.len() + 1 + 1 + 1 + art.data.len();
    let size_be = (payload_len as u32).to_be_bytes();

    let mut frame = Vec::with_capacity(10 + payload_len);
    frame.extend_from_slice(b"APIC"); // 4 bytes: frame ID
    frame.extend_from_slice(&size_be); // 4 bytes: size
    frame.extend_from_slice(&[0x00, 0x00]); // 2 bytes: flags
    frame.push(0x03); // 1 byte: encoding UTF-8
    frame.extend_from_slice(mime_bytes); // N bytes: MIME type
    frame.push(0x00); // 1 byte: MIME null terminator
    frame.push(0x03); // 1 byte: picture type (front cover)
    frame.push(0x00); // 1 byte: empty description (null-terminated)
    frame.extend_from_slice(&art.data); // M bytes: raw image data
    frame
}

/// Build a USLT (unsynchronised lyrics) ID3v2.4 frame.
///
/// Frame layout:
/// ```text
/// [frame_id: 4 bytes]      — b"USLT"
/// [size:     4 bytes]      — payload length, plain big-endian u32
/// [flags:    2 bytes]      — 0x00 0x00
/// [encoding: 1 byte]       — 0x03 = UTF-8
/// [language: 3 bytes]      — b"eng"
/// [content_desc: 1 byte]   — 0x00 (empty description, null-terminated)
/// [lyrics:   N bytes]      — lyrics text as UTF-8
/// ```
fn build_uslt_frame(lyrics: &str) -> Vec<u8> {
    let text_bytes = lyrics.as_bytes();
    // payload = encoding(1) + language(3) + desc_null(1) + text(N)
    let payload_len = 1usize + 3 + 1 + text_bytes.len();
    let size_be = (payload_len as u32).to_be_bytes();

    let mut frame = Vec::with_capacity(10 + payload_len);
    frame.extend_from_slice(b"USLT"); // 4 bytes: frame ID
    frame.extend_from_slice(&size_be); // 4 bytes: size
    frame.extend_from_slice(&[0x00, 0x00]); // 2 bytes: flags
    frame.push(0x03); // 1 byte: encoding UTF-8
    frame.extend_from_slice(b"eng"); // 3 bytes: language
    frame.push(0x00); // 1 byte: empty description (null-terminated)
    frame.extend_from_slice(text_bytes); // N bytes: lyrics text
    frame
}

/// Build a COMM (comments) ID3v2.4 frame with a non-empty description field.
///
/// Unlike the inline COMM frame used for plain comments (which has an empty
/// description), this builder is used for named COMM frames such as `iTunSMPB`
/// (Apple gapless playback metadata).
///
/// Frame layout:
/// ```text
/// [frame_id:  4 bytes]   — b"COMM"
/// [size:      4 bytes]   — payload length, plain big-endian u32
/// [flags:     2 bytes]   — 0x00 0x00
/// [encoding:  1 byte]    — 0x03 = UTF-8
/// [language:  3 bytes]   — b"eng"
/// [desc:      N bytes]   — description.as_bytes() (UTF-8)
/// [desc_null: 1 byte]    — 0x00 null terminator
/// [text:      M bytes]   — text.as_bytes() (UTF-8)
/// ```
fn build_comm_frame_with_desc(description: &str, text: &str) -> Vec<u8> {
    let desc_bytes = description.as_bytes();
    let text_bytes = text.as_bytes();
    // payload = encoding(1) + language(3) + desc(N) + null(1) + text(M)
    let payload_len = 1usize + 3 + desc_bytes.len() + 1 + text_bytes.len();
    let size_be = (payload_len as u32).to_be_bytes();

    let mut frame = Vec::with_capacity(10 + payload_len);
    frame.extend_from_slice(b"COMM"); // 4 bytes: frame ID
    frame.extend_from_slice(&size_be); // 4 bytes: size (plain big-endian)
    frame.extend_from_slice(&[0x00, 0x00]); // 2 bytes: flags
    frame.push(0x03); // 1 byte: encoding UTF-8
    frame.extend_from_slice(b"eng"); // 3 bytes: language
    frame.extend_from_slice(desc_bytes); // N bytes: description
    frame.push(0x00); // 1 byte: null terminator for description
    frame.extend_from_slice(text_bytes); // M bytes: text
    frame
}

/// Compute the IEEE 802.3 CRC-32 (polynomial 0xEDB88320, reversed form).
///
/// Used for the optional ID3v2 extended header CRC field.
fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let mut b = byte;
        for _ in 0..8 {
            if (crc ^ u32::from(b)) & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
            b >>= 1;
        }
    }
    crc ^ 0xFFFF_FFFF
}

/// Serialise an [`Mp3Tags`] value into a complete ID3v2.4 byte sequence.
///
/// The returned bytes should be written to the output **before** the LAME
/// MP3 data so that players find the tag at offset 0.
///
/// When `tags.extended_header_crc` is `true`, an ID3v2 extended header
/// (with CRC-32 of the frame bytes) is inserted between the main header and
/// the frames. When `tags.write_footer` is `true`, a 10-byte footer ("3DI")
/// is appended after the frames.
pub fn write_id3v2_4(tags: &Mp3Tags) -> Vec<u8> {
    // Collect all frame bytes first so we know the total frame-data size.
    let mut frames: Vec<u8> = Vec::new();

    if let Some(ref title) = tags.title {
        frames.extend_from_slice(&build_text_frame(b"TIT2", title));
    }
    if let Some(ref artist) = tags.artist {
        frames.extend_from_slice(&build_text_frame(b"TPE1", artist));
    }
    if let Some(ref album) = tags.album {
        frames.extend_from_slice(&build_text_frame(b"TALB", album));
    }
    if let Some(track) = tags.track_number {
        frames.extend_from_slice(&build_text_frame(b"TRCK", &track.to_string()));
    }
    // Also handle the `track` field from the builder API (u32).
    if tags.track_number.is_none() {
        if let Some(track) = tags.track {
            frames.extend_from_slice(&build_text_frame(b"TRCK", &track.to_string()));
        }
    }
    if let Some(year) = tags.year {
        frames.extend_from_slice(&build_text_frame(b"TDRC", &year.to_string()));
    }
    if let Some(ref genre) = tags.genre {
        frames.extend_from_slice(&build_text_frame(b"TCON", genre));
    }
    if let Some(ref composer) = tags.composer {
        frames.extend_from_slice(&build_text_frame(b"TCOM", composer));
    }
    if let Some(ref comment) = tags.comment {
        // Write as a COMM frame (simplified: language=eng, short description empty).
        // COMM payload: encoding(1) + language(3) + short_desc(1) + text(N)
        let text_bytes = comment.as_bytes();
        let payload_len = 1 + 3 + 1 + text_bytes.len();
        let size_be = (payload_len as u32).to_be_bytes();
        frames.extend_from_slice(b"COMM");
        frames.extend_from_slice(&size_be);
        frames.extend_from_slice(&[0x00, 0x00]); // flags
        frames.push(0x03); // encoding UTF-8
        frames.extend_from_slice(b"eng"); // language
        frames.push(0x00); // short description (empty, null-terminated)
        frames.extend_from_slice(text_bytes); // comment text (UTF-8)
    }
    if let Some(disc) = tags.disc_number {
        frames.extend_from_slice(&build_text_frame(b"TPOS", &disc.to_string()));
    }

    // Feature 1: APIC album art frame.
    if let Some(ref art) = tags.album_art {
        frames.extend_from_slice(&build_apic_frame(art));
    }

    // Feature 2: ReplayGain TXXX frames.
    if let Some(gain) = tags.replaygain_track_gain {
        let value = format!("{gain:+.2} dB");
        frames.extend_from_slice(&build_txxx_frame("REPLAYGAIN_TRACK_GAIN", &value));
    }
    if let Some(peak) = tags.replaygain_track_peak {
        let value = format!("{peak:.6}");
        frames.extend_from_slice(&build_txxx_frame("REPLAYGAIN_TRACK_PEAK", &value));
    }
    if let Some(gain) = tags.replaygain_album_gain {
        let value = format!("{gain:+.2} dB");
        frames.extend_from_slice(&build_txxx_frame("REPLAYGAIN_ALBUM_GAIN", &value));
    }
    if let Some(peak) = tags.replaygain_album_peak {
        let value = format!("{peak:.6}");
        frames.extend_from_slice(&build_txxx_frame("REPLAYGAIN_ALBUM_PEAK", &value));
    }

    // Feature 3: USLT lyrics frame.
    if let Some(ref lyrics) = tags.lyrics {
        frames.extend_from_slice(&build_uslt_frame(lyrics));
    }

    // Feature 4: User-defined TXXX frames.
    for (key, value) in &tags.user_defined {
        frames.extend_from_slice(&build_txxx_frame(key, value));
    }

    // Feature 5: iTunSMPB gapless playback COMM frame.
    // Written only when both encoder_delay and encoder_padding are present.
    if let (Some(delay), Some(padding)) = (tags.encoder_delay, tags.encoder_padding) {
        // iTunSMPB format: " %08X %08X %08X %016X"
        // Fields: pad_byte (always 0), start_delay, end_padding, total_samples (0 = unknown).
        let itun_value = format!(
            " {:08X} {:08X} {:08X} {:016X}",
            0u32,    // pad byte (always 0)
            delay,   // encoder start delay
            padding, // end padding
            0u64,    // total samples (0 = unknown/not computed)
        );
        frames.extend_from_slice(&build_comm_frame_with_desc("iTunSMPB", &itun_value));
    }

    // Build optional extended header (CRC-32).
    // Extended header layout (v2.4, minimal with CRC):
    //   size:         4 bytes BE — 10 (includes the size field itself)
    //   num_flags:    1 byte — 0x01 (one flag byte follows)
    //   flags_byte:   1 byte — 0x40 (bit 6 = CRC present)
    //   CRC_data_len: 1 byte — 0x05 (5-byte CRC field)
    //   CRC:          5 bytes — synchsafe-encoded CRC-32 (low 35 bits, 7 bits per byte)
    let ext_header: Vec<u8> = if tags.extended_header_crc {
        let crc = crc32_ieee(&frames);
        // Encode CRC as 5 synchsafe bytes (ID3v2.4 style, 7 bits each).
        let crc_bytes = [
            ((crc >> 28) & 0x7F) as u8,
            ((crc >> 21) & 0x7F) as u8,
            ((crc >> 14) & 0x7F) as u8,
            ((crc >> 7) & 0x7F) as u8,
            (crc & 0x7F) as u8,
        ];
        // Extended header is 10 bytes: 4 (size) + 1 (num_flags) + 1 (flags_byte)
        // + 1 (CRC_data_len) + ... but v2.4 spec sizes it differently.
        // Follow the spec: size field = big-endian 4-byte synchsafe count of
        // *all* bytes in the extended header (including the size field itself).
        // Minimal structure with CRC = 4 + 1 + 1 + 1 + 5 = 12 bytes total (synchsafe 12).
        let ext_size = encode_synchsafe(12u32);
        let mut h = Vec::with_capacity(12);
        h.extend_from_slice(&ext_size); // 4 bytes: extended header size (synchsafe)
        h.push(0x01); // 1 byte: number of flag bytes
        h.push(0x40); // 1 byte: flags (bit 6 = CRC present)
        h.push(0x05); // 1 byte: CRC data length (5 bytes)
        h.extend_from_slice(&crc_bytes); // 5 bytes: synchsafe CRC-32
        h
    } else {
        Vec::new()
    };

    // Total tag payload = extended header (if any) + frame bytes.
    // The synchsafe size field covers everything after the 10-byte main header.
    let payload_size = (ext_header.len() + frames.len()) as u32;
    let synchsafe_size = encode_synchsafe(payload_size);

    // Compute header flags byte:
    //   bit 6 (0x40) = extended header present
    //   bit 4 (0x10) = footer present
    let flags_byte = {
        let mut f = 0u8;
        if tags.extended_header_crc {
            f |= 0x40;
        }
        if tags.write_footer {
            f |= 0x10;
        }
        f
    };

    // ID3v2.4 header: "ID3" + version (0x04 0x00) + flags + size (synchsafe).
    let capacity = 10 + ext_header.len() + frames.len() + if tags.write_footer { 10 } else { 0 };
    let mut tag: Vec<u8> = Vec::with_capacity(capacity);
    tag.extend_from_slice(b"ID3"); // 3 bytes: magic
    tag.extend_from_slice(&[0x04, 0x00]); // 2 bytes: version 2.4.0
    tag.push(flags_byte); // 1 byte: flags
    tag.extend_from_slice(&synchsafe_size); // 4 bytes: synchsafe size
    tag.extend_from_slice(&ext_header); // optional extended header
    tag.extend_from_slice(&frames); // frame data

    // Append ID3v2 footer ("3DI") when requested.
    if tags.write_footer {
        tag.extend_from_slice(b"3DI"); // 3 bytes: footer magic
        tag.extend_from_slice(&[0x04, 0x00]); // 2 bytes: version 2.4.0
        tag.push(flags_byte); // 1 byte: flags (same as header)
        tag.extend_from_slice(&synchsafe_size); // 4 bytes: same synchsafe size
    }

    tag
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lame::Mp3Tags;

    #[test]
    fn test_synchsafe_encoding() {
        // 0 → all zeros
        assert_eq!(encode_synchsafe(0), [0, 0, 0, 0]);
        // 128 = 0b1000_0000 → synchsafe: [0, 0, 1, 0] (bit 7 carries to next byte)
        assert_eq!(encode_synchsafe(128), [0, 0, 1, 0]);
        // 255 = 0b1111_1111 → synchsafe: [0, 0, 1, 0x7F]
        assert_eq!(encode_synchsafe(255), [0, 0, 1, 0x7F]);
    }

    #[test]
    fn test_id3_header_magic_and_version() {
        let tags = Mp3Tags {
            title: Some("X".into()),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        assert_eq!(&bytes[0..3], b"ID3");
        assert_eq!(bytes[3], 0x04, "ID3v2.4 major version");
        assert_eq!(bytes[4], 0x00, "ID3v2.4 minor version");
        assert_eq!(bytes[5], 0x00, "flags byte");
    }

    #[test]
    fn test_id3_size_field_roundtrip() {
        let tags = Mp3Tags {
            title: Some("Hello".into()),
            track_number: Some(7),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        // Decode synchsafe size from bytes 6..10.
        let s = &bytes[6..10];
        let decoded =
            ((s[0] as u32) << 21) | ((s[1] as u32) << 14) | ((s[2] as u32) << 7) | (s[3] as u32);
        // The size must equal the remaining bytes after the 10-byte header.
        assert_eq!(decoded as usize, bytes.len() - 10);
    }

    #[test]
    fn test_trck_frame_present() {
        let tags = Mp3Tags {
            track_number: Some(3),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        let pos = bytes
            .windows(4)
            .position(|w| w == b"TRCK")
            .expect("TRCK frame must be present");
        // Encoding byte after frame header (10 bytes) should be 0x03 (UTF-8).
        assert_eq!(bytes[pos + 10], 0x03);
        // Text "3" follows the encoding byte.
        assert_eq!(bytes[pos + 11], b'3');
    }

    #[test]
    fn test_empty_tags_produces_10_byte_header_only() {
        let tags = Mp3Tags::default(); // all None
        let bytes = write_id3v2_4(&tags);
        assert_eq!(bytes.len(), 10, "header-only tag must be exactly 10 bytes");
        // Size synchsafe = 0.
        assert_eq!(&bytes[6..10], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_genre_frame_written() {
        let tags = Mp3Tags {
            genre: Some("Electronic".into()),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        assert!(
            bytes.windows(4).any(|w| w == b"TCON"),
            "TCON (genre) frame must be present"
        );
    }

    #[test]
    fn test_composer_frame_written() {
        let tags = Mp3Tags {
            composer: Some("Bach".into()),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        assert!(
            bytes.windows(4).any(|w| w == b"TCOM"),
            "TCOM (composer) frame must be present"
        );
    }

    #[test]
    fn test_comment_frame_written() {
        let tags = Mp3Tags {
            comment: Some("Test comment".into()),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        assert!(
            bytes.windows(4).any(|w| w == b"COMM"),
            "COMM (comment) frame must be present"
        );
    }

    #[test]
    fn test_disc_number_frame_written() {
        let tags = Mp3Tags {
            disc_number: Some(2),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        assert!(
            bytes.windows(4).any(|w| w == b"TPOS"),
            "TPOS (disc number) frame must be present"
        );
    }

    #[test]
    fn test_utf8_encoding_byte_is_0x03() {
        // Non-ASCII title: UTF-8 bytes must be correctly tagged with encoding 0x03.
        let title = "日本語タイトル";
        let tags = Mp3Tags {
            title: Some(title.into()),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);

        // Locate the TIT2 frame.
        let pos = bytes
            .windows(4)
            .position(|w| w == b"TIT2")
            .expect("TIT2 frame must be present");

        // Byte at frame_start + 10 is the encoding byte (after 4 ID + 4 size + 2 flags).
        assert_eq!(
            bytes[pos + 10],
            0x03,
            "TIT2 encoding byte must be 0x03 (UTF-8)"
        );

        // The text bytes following the encoding byte must match the UTF-8 representation.
        let text_start = pos + 11;
        let text_bytes = title.as_bytes();
        assert_eq!(
            &bytes[text_start..text_start + text_bytes.len()],
            text_bytes,
            "TIT2 text bytes must be valid UTF-8"
        );
    }

    #[test]
    fn test_tdrc_frame_replaces_tyer() {
        // TDRC (ID3v2.4 recording time) must be used instead of legacy TYER.
        let tags = Mp3Tags {
            year: Some(2024),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        assert!(
            bytes.windows(4).any(|w| w == b"TDRC"),
            "TDRC frame must be present for year field"
        );
        assert!(
            !bytes.windows(4).any(|w| w == b"TYER"),
            "legacy TYER frame must not appear in ID3v2.4 output"
        );
    }

    #[test]
    fn test_apic_frame_structure() {
        use crate::lame::AlbumArt;
        // Minimal JPEG magic bytes: FF D8 FF E0
        let art = AlbumArt {
            mime_type: "image/jpeg".into(),
            data: vec![0xFF, 0xD8, 0xFF, 0xE0],
        };
        let tags = Mp3Tags {
            album_art: Some(art),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);

        let apic_pos = bytes
            .windows(4)
            .position(|w| w == b"APIC")
            .expect("APIC frame must be present");

        // After the 10-byte frame header: encoding(1) + "image/jpeg"(10) + null(1) +
        // pic_type(1) + desc_null(1) = 14 bytes before image data.
        let data_start = apic_pos + 10 + 1 + "image/jpeg".len() + 1 + 1 + 1;
        assert_eq!(
            bytes[data_start], 0xFF,
            "JPEG magic byte 0xFF must be present"
        );
        assert_eq!(
            bytes[data_start + 1],
            0xD8,
            "JPEG magic byte 0xD8 must be present"
        );
    }

    #[test]
    fn test_replaygain_txxx_frames() {
        let tags = Mp3Tags {
            replaygain_track_gain: Some(-6.5),
            replaygain_track_peak: Some(0.988),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);

        // At least one TXXX frame must be present.
        assert!(
            bytes.windows(4).any(|w| w == b"TXXX"),
            "TXXX frame must be present for ReplayGain"
        );
        // The key "REPLAYGAIN_TRACK_GAIN" must appear in the bytes.
        let key_bytes = b"REPLAYGAIN_TRACK_GAIN";
        assert!(
            bytes.windows(key_bytes.len()).any(|w| w == key_bytes),
            "REPLAYGAIN_TRACK_GAIN key must be present"
        );
        // Check format: -6.50 dB
        let gain_str = b"-6.50 dB";
        assert!(
            bytes.windows(gain_str.len()).any(|w| w == gain_str),
            "ReplayGain gain value must be formatted as '-6.50 dB'"
        );
    }

    #[test]
    fn test_uslt_frame_present() {
        let tags = Mp3Tags {
            lyrics: Some("Hello world".into()),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);

        assert!(
            bytes.windows(4).any(|w| w == b"USLT"),
            "USLT frame must be present when lyrics are set"
        );
        let text_bytes = b"Hello world";
        assert!(
            bytes
                .windows(text_bytes.len())
                .any(|w| w == text_bytes.as_slice()),
            "lyrics text must appear in USLT frame"
        );
    }

    #[test]
    fn test_txxx_user_defined() {
        let mut tags = Mp3Tags::default();
        tags.user_defined.push(("MY_KEY".into(), "MY_VAL".into()));
        let bytes = write_id3v2_4(&tags);

        assert!(
            bytes.windows(4).any(|w| w == b"TXXX"),
            "TXXX frame must be present for user-defined tags"
        );
        let key_bytes = b"MY_KEY";
        assert!(
            bytes.windows(key_bytes.len()).any(|w| w == key_bytes),
            "user-defined key MY_KEY must appear in TXXX frame"
        );
        let val_bytes = b"MY_VAL";
        assert!(
            bytes.windows(val_bytes.len()).any(|w| w == val_bytes),
            "user-defined value MY_VAL must appear in TXXX frame"
        );
    }

    #[test]
    fn test_itun_smpb_frame_present() {
        // Both encoder_delay and encoder_padding set → iTunSMPB COMM frame emitted.
        let tags = Mp3Tags {
            encoder_delay: Some(576),
            encoder_padding: Some(0),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);

        // A COMM frame must be present.
        assert!(
            bytes.windows(4).any(|w| w == b"COMM"),
            "COMM frame must be present when encoder_delay and encoder_padding are set"
        );
        // The description "iTunSMPB" must appear in the bytes.
        let desc = b"iTunSMPB";
        assert!(
            bytes.windows(desc.len()).any(|w| w == desc),
            "iTunSMPB description must appear in the COMM frame"
        );
    }

    #[test]
    fn test_itun_smpb_absent_without_delay() {
        // encoder_delay = None → no iTunSMPB frame.
        let tags = Mp3Tags {
            encoder_delay: None,
            encoder_padding: Some(0),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        let desc = b"iTunSMPB";
        assert!(
            !bytes.windows(desc.len()).any(|w| w == desc),
            "iTunSMPB must not appear when encoder_delay is None"
        );
    }

    #[test]
    fn test_lame_encoder_delay_constant() {
        assert_eq!(
            crate::lame::LAME_ENCODER_DELAY,
            576,
            "LAME_ENCODER_DELAY must equal 576"
        );
    }

    #[test]
    fn test_extended_header_crc_flag_set() {
        // extended_header_crc=true must set bit 6 (0x40) of the header flags byte.
        let tags = Mp3Tags {
            extended_header_crc: true,
            title: Some("CRC Test".into()),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        // Byte 5 of the tag is the ID3v2 header flags byte.
        assert_eq!(
            bytes[5] & 0x40,
            0x40,
            "header flags byte must have extended header bit (0x40) set"
        );
    }

    #[test]
    fn test_extended_header_crc_not_set_by_default() {
        let tags = Mp3Tags {
            title: Some("No CRC".into()),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        assert_eq!(
            bytes[5] & 0x40,
            0,
            "extended header flag must not be set by default"
        );
    }

    #[test]
    fn test_write_footer_appends_3di() {
        let tags = Mp3Tags {
            write_footer: true,
            title: Some("Footer Test".into()),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        // "3DI" must appear somewhere after the main header.
        let found = bytes
            .windows(3)
            .skip(10) // skip the main header
            .any(|w| w == b"3DI");
        assert!(
            found,
            "footer magic '3DI' must be present when write_footer=true"
        );
        // The footer flag bit 4 (0x10) must be set in the header flags byte.
        assert_eq!(
            bytes[5] & 0x10,
            0x10,
            "header flags byte must have footer bit (0x10) set"
        );
    }

    #[test]
    fn test_footer_not_written_by_default() {
        let tags = Mp3Tags {
            title: Some("No Footer".into()),
            ..Default::default()
        };
        let bytes = write_id3v2_4(&tags);
        let found = bytes.windows(3).skip(10).any(|w| w == b"3DI");
        assert!(!found, "footer must not appear when write_footer=false");
    }

    #[test]
    fn test_crc32_ieee_known_value() {
        // CRC-32 of b"123456789" is the well-known test vector 0xCBF43926.
        assert_eq!(
            crc32_ieee(b"123456789"),
            0xCBF4_3926,
            "CRC-32 of '123456789' must equal 0xCBF43926"
        );
    }
}
