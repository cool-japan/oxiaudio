//! FLAC CUESHEET metadata block parser.
//!
//! Parses the binary CUESHEET block (block type 5) defined in the FLAC specification:
//! <https://xiph.org/flac/format.html#metadata_block_cuesheet>
//!
//! The parser reads the raw byte stream after the `fLaC` stream marker, iterates over
//! metadata blocks, and decodes the CUESHEET block when found. It never allocates into
//! a temporary string heap except for ISRC values.

use std::path::Path;

use oxiaudio_core::OxiAudioError;

/// A single track entry parsed from a FLAC CUESHEET metadata block.
#[derive(Debug, Clone, PartialEq)]
pub struct FlacCuePoint {
    /// Track number as stored in the cue sheet (1–99 for audio, 170 for lead-out).
    pub track_number: u8,
    /// Sample offset from the start of the audio data where this track begins.
    pub offset_samples: u64,
    /// The 12-character ISRC code if present and non-zero; `None` otherwise.
    pub isrc: Option<String>,
    /// `true` when the track type flag indicates an audio track (bit 7 of the flags byte = 0).
    pub is_audio: bool,
}

/// FLAC stream marker: the first 4 bytes of every valid FLAC file.
const FLAC_MARKER: &[u8; 4] = b"fLaC";

/// FLAC metadata block type for CUESHEET.
const BLOCK_TYPE_CUESHEET: u8 = 5;

/// Parse cue sheet entries from a FLAC file at `path`.
///
/// Reads the raw bytes of the file, locates the CUESHEET metadata block (block type 5),
/// and decodes it according to the FLAC specification. If the file does not begin with the
/// `fLaC` stream marker or contains no CUESHEET block, an empty [`Vec`] is returned.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened or read.
/// Returns [`OxiAudioError::Decode`] if the CUESHEET block is malformed (truncated data).
#[must_use = "discarding the Result ignores parse errors"]
pub fn parse_flac_cue_sheet(path: &Path) -> Result<Vec<FlacCuePoint>, OxiAudioError> {
    let data = std::fs::read(path).map_err(OxiAudioError::Io)?;
    parse_flac_cue_sheet_from_bytes(&data)
}

/// Parse cue sheet entries from an in-memory FLAC byte slice.
///
/// Returns an empty `Vec` when no CUESHEET block is present or the data is not a FLAC stream.
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] if the CUESHEET block is malformed.
pub(crate) fn parse_flac_cue_sheet_from_bytes(
    data: &[u8],
) -> Result<Vec<FlacCuePoint>, OxiAudioError> {
    // Must start with the fLaC marker.
    if data.len() < 4 || &data[..4] != FLAC_MARKER {
        return Ok(Vec::new());
    }

    let mut pos = 4usize; // cursor after the stream marker

    loop {
        // Each metadata block header is 4 bytes:
        //   byte 0: bit7 = last-metadata-block flag, bits 6..0 = block type
        //   bytes 1-3: block length (24-bit big-endian, NOT including the 4-byte header)
        if pos + 4 > data.len() {
            break;
        }
        let header_byte = data[pos];
        let is_last = (header_byte & 0x80) != 0;
        let block_type = header_byte & 0x7F;

        let block_len = (u32::from(data[pos + 1]) << 16)
            | (u32::from(data[pos + 2]) << 8)
            | u32::from(data[pos + 3]);
        let block_len = block_len as usize;
        pos += 4;

        if pos + block_len > data.len() {
            break;
        }

        if block_type == BLOCK_TYPE_CUESHEET {
            let block = &data[pos..pos + block_len];
            return parse_cuesheet_block(block);
        }

        pos += block_len;

        if is_last {
            break;
        }
    }

    // No CUESHEET block found — return empty.
    Ok(Vec::new())
}

/// Parse the payload of a CUESHEET metadata block.
///
/// Layout per FLAC spec (all multi-byte fields are big-endian):
/// - 128 bytes: media catalog number (null-padded ASCII)
/// - 8 bytes:   lead-in samples
/// - 1 byte:    bit 7 = is_CD flag, bits 6..0 = reserved (must be 0)
/// - 258 bytes: reserved (must be zero)
/// - 1 byte:    num_tracks
/// - For each track:
///   - 8 bytes: track offset (samples from start)
///   - 1 byte:  track number (1–99 audio, 170 lead-out)
///   - 12 bytes: ISRC (ASCII, null-padded)
///   - 1 byte:  flags (bit7 = non-audio type, bit6 = pre-emphasis)
///   - 13 bytes: reserved (must be zero)
///   - 1 byte:  num_indices
///   - For each index: 4 bytes offset + 1 byte index number + 3 bytes reserved
fn parse_cuesheet_block(block: &[u8]) -> Result<Vec<FlacCuePoint>, OxiAudioError> {
    // Minimum required before the track list: 128 + 8 + 1 + 258 + 1 = 396 bytes.
    const HEADER_SIZE: usize = 128 + 8 + 1 + 258 + 1;
    if block.len() < HEADER_SIZE {
        return Err(OxiAudioError::Decode(
            "FLAC CUESHEET block too short for header".into(),
        ));
    }

    let mut cursor = 0usize;

    // Skip media catalog number (128 bytes).
    cursor += 128;

    // Skip lead-in samples (8 bytes).
    cursor += 8;

    // is_CD flag byte — consume but ignore.
    cursor += 1;

    // Skip 258 reserved bytes.
    cursor += 258;

    // num_tracks (1 byte).
    let num_tracks = block[cursor] as usize;
    cursor += 1;

    let mut cue_points = Vec::with_capacity(num_tracks.min(256));

    for _ in 0..num_tracks {
        // Each track entry is at minimum 36 bytes:
        //   8 (offset) + 1 (track_number) + 12 (ISRC) + 1 (flags) + 13 (reserved) + 1 (num_indices)
        // plus 8 bytes per index point.
        const TRACK_HEADER_SIZE: usize = 8 + 1 + 12 + 1 + 13 + 1;
        if cursor + TRACK_HEADER_SIZE > block.len() {
            return Err(OxiAudioError::Decode(
                "FLAC CUESHEET track entry truncated".into(),
            ));
        }

        // 8 bytes: track offset in samples.
        let offset_samples = u64::from_be_bytes(
            block[cursor..cursor + 8]
                .try_into()
                .map_err(|_| OxiAudioError::Decode("CUESHEET offset slice error".into()))?,
        );
        cursor += 8;

        // 1 byte: track number.
        let track_number = block[cursor];
        cursor += 1;

        // 12 bytes: ISRC (ASCII, null-padded).
        let isrc_bytes = &block[cursor..cursor + 12];
        cursor += 12;

        // 1 byte: flags (bit 7 = non-audio, bit 6 = pre-emphasis).
        let flags = block[cursor];
        cursor += 1;
        // is_audio: bit 7 of flags being 0 means audio track.
        let is_audio = (flags & 0x80) == 0;

        // 13 bytes: reserved.
        cursor += 13;

        // 1 byte: num_indices.
        let num_indices = block[cursor] as usize;
        cursor += 1;

        // Skip num_indices × 8 bytes (4-byte sample offset + 1 index number + 3 reserved).
        let indices_size = num_indices * 8;
        if cursor + indices_size > block.len() {
            return Err(OxiAudioError::Decode(
                "FLAC CUESHEET index points truncated".into(),
            ));
        }
        cursor += indices_size;

        // Decode ISRC: non-zero bytes form the code; all-zero → None.
        let isrc = if isrc_bytes.iter().any(|&b| b != 0) {
            let s = String::from_utf8_lossy(isrc_bytes);
            // Trim trailing NUL chars.
            let trimmed = s.trim_end_matches('\0');
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_owned())
            }
        } else {
            None
        };

        cue_points.push(FlacCuePoint {
            track_number,
            offset_samples,
            isrc,
            is_audio,
        });
    }

    Ok(cue_points)
}

#[cfg(test)]
mod cue_tests {
    use super::*;

    // ─── Helper: build minimal FLAC metadata block ────────────────────────────

    /// Wrap `payload` in a FLAC metadata block header.
    ///
    /// `block_type` is the 7-bit block type; `is_last` sets the MSB of byte 0.
    fn make_metadata_block(block_type: u8, is_last: bool, payload: &[u8]) -> Vec<u8> {
        let mut block = Vec::with_capacity(4 + payload.len());
        let header_byte = if is_last {
            0x80 | (block_type & 0x7F)
        } else {
            block_type & 0x7F
        };
        block.push(header_byte);
        let len = payload.len() as u32;
        block.push(((len >> 16) & 0xFF) as u8);
        block.push(((len >> 8) & 0xFF) as u8);
        block.push((len & 0xFF) as u8);
        block.extend_from_slice(payload);
        block
    }

    /// Build a minimal CUESHEET block payload with `num_tracks` dummy tracks.
    ///
    /// Each track has offset = `track_index * 44100`, track_number = track_index + 1,
    /// blank ISRC, and the is-audio flag (bit 7 of flags = 0).
    fn make_cuesheet_payload(tracks: &[(u64, u8, bool)]) -> Vec<u8> {
        let mut payload = Vec::new();
        // 128-byte media catalog number (all zero).
        payload.extend_from_slice(&[0u8; 128]);
        // 8-byte lead-in samples.
        payload.extend_from_slice(&0u64.to_be_bytes());
        // is_CD byte.
        payload.push(0);
        // 258 reserved bytes.
        payload.extend_from_slice(&[0u8; 258]);
        // num_tracks.
        payload.push(tracks.len() as u8);

        for &(offset, track_num, is_audio) in tracks {
            // 8-byte offset.
            payload.extend_from_slice(&offset.to_be_bytes());
            // track_number.
            payload.push(track_num);
            // 12-byte ISRC (all zero = no ISRC).
            payload.extend_from_slice(&[0u8; 12]);
            // flags: bit 7 = non-audio (set if !is_audio).
            let flags: u8 = if is_audio { 0x00 } else { 0x80 };
            payload.push(flags);
            // 13 reserved bytes.
            payload.extend_from_slice(&[0u8; 13]);
            // num_indices = 0.
            payload.push(0);
        }

        payload
    }

    // ─── Tests ────────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_flac_cue_non_flac_returns_empty() {
        // Random bytes that do NOT start with fLaC.
        let data = b"RIFF\x00\x00\x00\x00WAVE";
        let result = parse_flac_cue_sheet_from_bytes(data).expect("must not error");
        assert!(result.is_empty(), "expected empty Vec for non-FLAC data");
    }

    #[test]
    fn test_parse_flac_cue_no_cuesheet_block_returns_empty() {
        // A FLAC stream with only a STREAMINFO block (block type 0) and no CUESHEET.
        let streaminfo_payload = vec![0u8; 34]; // minimal STREAMINFO size
        let mut data = Vec::new();
        data.extend_from_slice(FLAC_MARKER);
        data.extend_from_slice(&make_metadata_block(0, true, &streaminfo_payload));

        let result = parse_flac_cue_sheet_from_bytes(&data).expect("must not error");
        assert!(
            result.is_empty(),
            "expected empty Vec when no CUESHEET block"
        );
    }

    #[test]
    fn test_parse_flac_cue_with_two_tracks() {
        // Build a FLAC stream: STREAMINFO (last=false) + CUESHEET (last=true).
        let streaminfo_payload = vec![0u8; 34];
        let tracks: &[(u64, u8, bool)] = &[
            (0, 1, true),        // track 1: offset 0, audio
            (44100, 2, true),    // track 2: offset 44100, audio
            (0xFFFF, 170, true), // lead-out track (track number 170)
        ];
        let cuesheet_payload = make_cuesheet_payload(tracks);

        let mut data = Vec::new();
        data.extend_from_slice(FLAC_MARKER);
        data.extend_from_slice(&make_metadata_block(0, false, &streaminfo_payload));
        data.extend_from_slice(&make_metadata_block(
            BLOCK_TYPE_CUESHEET,
            true,
            &cuesheet_payload,
        ));

        let result = parse_flac_cue_sheet_from_bytes(&data).expect("must not error");
        assert_eq!(result.len(), 3, "expected 3 cue points");

        assert_eq!(result[0].track_number, 1);
        assert_eq!(result[0].offset_samples, 0);
        assert!(result[0].isrc.is_none());
        assert!(result[0].is_audio);

        assert_eq!(result[1].track_number, 2);
        assert_eq!(result[1].offset_samples, 44100);
        assert!(result[1].is_audio);

        assert_eq!(result[2].track_number, 170); // lead-out
    }

    #[test]
    fn test_parse_flac_cue_non_audio_track() {
        // A track with bit 7 of flags = 1 → is_audio = false.
        let tracks: &[(u64, u8, bool)] = &[(0, 1, false)];
        let cuesheet_payload = make_cuesheet_payload(tracks);

        let mut data = Vec::new();
        data.extend_from_slice(FLAC_MARKER);
        data.extend_from_slice(&make_metadata_block(
            BLOCK_TYPE_CUESHEET,
            true,
            &cuesheet_payload,
        ));

        let result = parse_flac_cue_sheet_from_bytes(&data).expect("must not error");
        assert_eq!(result.len(), 1);
        assert!(!result[0].is_audio, "expected non-audio track");
    }

    #[test]
    fn test_parse_flac_cue_file_nonexistent_is_error() {
        let path = std::env::temp_dir().join("oxiaudio_cue_nonexistent_xyz.flac");
        let result = parse_flac_cue_sheet(&path);
        assert!(result.is_err(), "expected Err for missing file");
    }

    #[test]
    fn test_parse_flac_cue_wav_file_returns_empty() {
        // Write a tiny WAV file (RIFF header) and call parse_flac_cue_sheet on it.
        // The function must return Ok(empty) since the file doesn't start with fLaC.
        let mut path = std::env::temp_dir();
        path.push("oxiaudio_cue_test_wav.wav");

        let wav_bytes: &[u8] = &[
            b'R', b'I', b'F', b'F', // RIFF
            36, 0, 0, 0, // file size - 8
            b'W', b'A', b'V', b'E', // WAVE
            b'f', b'm', b't', b' ', // fmt chunk
            16, 0, 0, 0, // chunk size
            1, 0, // PCM
            1, 0, // mono
            0x44, 0xAC, 0, 0, // 44100 Hz
            0x88, 0x58, 1, 0, // byte rate
            2, 0, // block align
            16, 0, // bits per sample
            b'd', b'a', b't', b'a', // data chunk
            0, 0, 0, 0, // no samples
        ];
        std::fs::write(&path, wav_bytes).expect("write tmp wav");

        let result = parse_flac_cue_sheet(&path).expect("must not error on WAV");
        assert!(result.is_empty(), "expected empty Vec for WAV file");

        let _ = std::fs::remove_file(&path);
    }
}
