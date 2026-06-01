//! Format detection by magic-byte inspection.

use std::path::Path;

use oxiaudio_core::OxiAudioError;

/// A hint about the container/codec format of an audio file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioFormatHint {
    Wav,
    Flac,
    Mp3,
    Ogg,
    Aiff,
    Au,
    /// Musepack SV7 (`MP+`) or SV8 (`MPCK`) audio.
    Musepack,
    /// WavPack lossless/lossy audio (`wvpk`).
    WavPack,
    /// AAC audio in ADTS framing (sync word `0xFF 0xF?` with ADTS layer bits).
    Aac,
    /// AAC or ALAC audio in an ISO Base Media File Format container (M4A/MP4).
    /// Detected via the `ftyp` box with an M4A-family compatible brand.
    M4a,
}

/// Detect audio format from initial bytes (magic number inspection).
///
/// Returns `None` if format cannot be determined from the provided bytes.
/// Requires at least 12 bytes for reliable detection of all formats.
///
/// Magic byte patterns checked:
/// - **WAV**: `RIFF....WAVE` — bytes[0..4] == `b"RIFF"` && bytes[8..12] == `b"WAVE"`
/// - **FLAC**: `fLaC` — bytes[0..4] == `b"fLaC"`
/// - **AAC/ADTS**: ADTS sync word — `bytes[0] == 0xFF` and `(bytes[1] & 0xF6) == 0xF0`
///   (12 sync bits + 2 ADTS layer bits = 00, distinguishing it from MPEG audio)
/// - **MP3**: `ID3` header or MPEG sync word — `bytes[0..3] == b"ID3"` or (`bytes[0] == 0xFF`
///   and `(bytes[1] & 0xE0) == 0xE0`, checked after ADTS to avoid misidentifying ADTS as MP3)
/// - **OGG**: `OggS` — bytes[0..4] == `b"OggS"`
/// - **M4A/MP4** (`ftyp` box): `bytes[4..8] == b"ftyp"` and a M4A-family compatible brand
///   at bytes[8..12] (`M4A `, `M4B `, `M4P `, `M4V `, `isom`, `iso2`, `mp42`, `f4v `)
/// - **AIFF**: `FORM....AIFF` — bytes[0..4] == `b"FORM"` && bytes[8..12] == `b"AIFF"`
/// - **AU/SND**: `.snd` magic — bytes[0..4] == `b".snd"`
/// - **Musepack SV7**: `MP+` — bytes[0..3] == `b"MP+"`
/// - **Musepack SV8**: `MPCK` — bytes[0..4] == `b"MPCK"`
/// - **WavPack**: `wvpk` — bytes[0..4] == `b"wvpk"`
pub fn detect_format_from_bytes(header: &[u8]) -> Option<AudioFormatHint> {
    // AU/SND: 4 bytes minimum
    if header.len() >= 4 && &header[..4] == b".snd" {
        return Some(AudioFormatHint::Au);
    }

    // FLAC: 4 bytes minimum
    if header.len() >= 4 && &header[..4] == b"fLaC" {
        return Some(AudioFormatHint::Flac);
    }

    // OGG: 4 bytes minimum
    if header.len() >= 4 && &header[..4] == b"OggS" {
        return Some(AudioFormatHint::Ogg);
    }

    // WavPack: "wvpk" (4 bytes)
    if header.len() >= 4 && &header[..4] == b"wvpk" {
        return Some(AudioFormatHint::WavPack);
    }

    // Musepack SV8: "MPCK" (4 bytes)
    if header.len() >= 4 && &header[..4] == b"MPCK" {
        return Some(AudioFormatHint::Musepack);
    }

    // Musepack SV7: "MP+" (3 bytes) — check before MP3 ID3 to avoid false match
    if header.len() >= 3 && &header[..3] == b"MP+" {
        return Some(AudioFormatHint::Musepack);
    }

    // ADTS AAC: sync word 0xFFF (12 bits) + ADTS layer = 00 (bits 1..0 of byte 1 must be 0b?0).
    // Pattern: byte0 == 0xFF, (byte1 & 0xF6) == 0xF0.
    // This is checked before the generic MPEG sync word to avoid misidentifying ADTS as MP3:
    //   ADTS nibble = 0xF? where (b1 & 0xF6) == 0xF0 (layer bits clear).
    //   MP3 MPEG-1/2/2.5 with layer 3: e.g. 0xFA, 0xFB — these satisfy (b1 & 0xF6) != 0xF0.
    if header.len() >= 2 && header[0] == 0xFF && (header[1] & 0xF6) == 0xF0 {
        return Some(AudioFormatHint::Aac);
    }

    // MP3: ID3 tag header (3 bytes) or MPEG sync word (2 bytes, checked after ADTS)
    if header.len() >= 3 && &header[..3] == b"ID3" {
        return Some(AudioFormatHint::Mp3);
    }
    if header.len() >= 2 && header[0] == 0xFF && (header[1] & 0xE0) == 0xE0 {
        return Some(AudioFormatHint::Mp3);
    }

    // WAV, AIFF, and M4A/MP4 all need 12 bytes for reliable detection
    if header.len() >= 12 {
        if &header[..4] == b"RIFF" && &header[8..12] == b"WAVE" {
            return Some(AudioFormatHint::Wav);
        }
        // M4A/MP4/ALAC container: ISO Base Media File Format with `ftyp` box.
        // The ftyp box starts at byte 4 (after a 4-byte big-endian box size).
        // Bytes 8..12 hold the major brand; we accept common M4A-family brands.
        if &header[4..8] == b"ftyp" {
            let brand = &header[8..12];
            if matches!(
                brand,
                b"M4A " | b"M4B " | b"M4P " | b"M4V " | b"isom" | b"iso2" | b"mp42" | b"f4v "
            ) {
                return Some(AudioFormatHint::M4a);
            }
        }
        if &header[..4] == b"FORM" && &header[8..12] == b"AIFF" {
            return Some(AudioFormatHint::Aiff);
        }
    }

    None
}

/// Detect audio format from the first 12 bytes of a file.
///
/// Opens the file, reads up to 12 bytes from the start, and delegates to
/// [`detect_format_from_bytes`].
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on file I/O failure, or
/// [`OxiAudioError::UnsupportedFormat`] when the magic bytes are not recognised.
pub fn detect_format_file(path: &Path) -> Result<AudioFormatHint, OxiAudioError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut header = [0u8; 12];
    let n = file.read(&mut header)?;
    detect_format_from_bytes(&header[..n]).ok_or_else(|| {
        OxiAudioError::UnsupportedFormat(format!(
            "unrecognised audio format in file: {}",
            path.display()
        ))
    })
}
