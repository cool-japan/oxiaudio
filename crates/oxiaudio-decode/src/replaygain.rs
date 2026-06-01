//! ReplayGain tag extraction from audio file metadata.
//!
//! Supports reading ReplayGain values from Vorbis comment tags (OGG/FLAC) and
//! ID3v2 TXXX frames (MP3), as provided by symphonia's unified tag interface.

use oxiaudio_core::OxiAudioError;
use symphonia::core::{
    formats::{probe::Hint, FormatOptions},
    io::{MediaSource, MediaSourceStream},
    meta::{MetadataOptions, RawValue},
};

/// ReplayGain metadata extracted from audio file tags.
///
/// All fields use dB values for gains and linear peak values for peaks,
/// following the ReplayGain 2.0 specification.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReplayGainMetadata {
    /// Track-level replay gain in dB (e.g. `-6.5 dB`). Positive means louder than reference.
    pub track_gain_db: Option<f64>,
    /// Track-level peak sample value (0.0–1.0+ linear scale).
    pub track_peak: Option<f64>,
    /// Album-level replay gain in dB.
    pub album_gain_db: Option<f64>,
    /// Album-level peak sample value (0.0–1.0+ linear scale).
    pub album_peak: Option<f64>,
}

/// Parse a gain string like `"-6.500 dB"` or `"-6.5"` into an `f64` dB value.
///
/// Strips any trailing ` dB` suffix before parsing. Returns `None` if the string
/// cannot be parsed as a floating-point number.
fn parse_gain_db(s: &str) -> Option<f64> {
    let trimmed = s
        .trim()
        .trim_end_matches("dB")
        .trim()
        .trim_end_matches("db")
        .trim();
    trimmed.parse::<f64>().ok()
}

/// Parse a peak string like `"0.998"` into an `f64` linear peak value.
fn parse_peak(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

/// Extract a string value from a symphonia `RawValue`, if it is a string variant.
fn raw_value_as_str(v: &RawValue) -> Option<&str> {
    match v {
        RawValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Parse ReplayGain metadata from an audio file's tags.
///
/// Reads the file using symphonia's format probing and extracts ReplayGain tags
/// from any tag format it supports (Vorbis comments, ID3v2 TXXX, APEv2, etc.).
///
/// Tag key matching is case-insensitive, following the ReplayGain 2.0 specification.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened, or
/// [`OxiAudioError::Decode`] if format probing fails.
#[must_use = "discarding the Result ignores parse errors"]
pub fn parse_replaygain(path: &std::path::Path) -> Result<ReplayGainMetadata, OxiAudioError> {
    let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mss = MediaSourceStream::new(Box::new(ReplayGainMediaWrapper(file)), Default::default());

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

    let mut rg = ReplayGainMetadata::default();

    // Check metadata from format reader.
    if let Some(rev) = format.metadata().current() {
        extract_replaygain_from_tags(rev.media.tags.iter(), &mut rg);
    }

    Ok(rg)
}

/// Parse ReplayGain metadata directly from a slice of symphonia tags.
///
/// This function is the core extraction logic, separated so it can be used in
/// unit tests without needing to create real audio files.
pub(crate) fn extract_replaygain_from_tags<'a>(
    tags: impl Iterator<Item = &'a symphonia::core::meta::Tag>,
    rg: &mut ReplayGainMetadata,
) {
    for tag in tags {
        // Use the raw key for matching — case-insensitive, trim whitespace.
        let key = tag.raw.key.trim().to_ascii_uppercase();
        let value_str = raw_value_as_str(&tag.raw.value);

        match key.as_str() {
            "REPLAYGAIN_TRACK_GAIN" => {
                rg.track_gain_db = value_str.and_then(parse_gain_db);
            }
            "REPLAYGAIN_TRACK_PEAK" => {
                rg.track_peak = value_str.and_then(parse_peak);
            }
            "REPLAYGAIN_ALBUM_GAIN" => {
                rg.album_gain_db = value_str.and_then(parse_gain_db);
            }
            "REPLAYGAIN_ALBUM_PEAK" => {
                rg.album_peak = value_str.and_then(parse_peak);
            }
            _ => {}
        }
    }
}

/// Thin wrapper implementing `MediaSource` for `std::fs::File`.
struct ReplayGainMediaWrapper(std::fs::File);

impl std::io::Read for ReplayGainMediaWrapper {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

impl std::io::Seek for ReplayGainMediaWrapper {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.0.seek(pos)
    }
}

impl MediaSource for ReplayGainMediaWrapper {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        self.0.metadata().ok().map(|m| m.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symphonia::core::meta::{RawTag, RawValue, Tag};

    /// Build a `Tag` from a raw key and string value (no standard tag mapping).
    fn make_str_tag(key: &str, value: &str) -> Tag {
        Tag::new(RawTag::new(
            key,
            RawValue::String(std::sync::Arc::new(value.to_owned())),
        ))
    }

    /// Verify that exact-case ReplayGain keys are parsed correctly.
    #[test]
    fn test_replaygain_exact_keys() {
        let tags = [
            make_str_tag("REPLAYGAIN_TRACK_GAIN", "-6.500 dB"),
            make_str_tag("REPLAYGAIN_TRACK_PEAK", "0.998244"),
            make_str_tag("REPLAYGAIN_ALBUM_GAIN", "-7.000 dB"),
            make_str_tag("REPLAYGAIN_ALBUM_PEAK", "0.999512"),
        ];
        let mut rg = ReplayGainMetadata::default();
        extract_replaygain_from_tags(tags.iter(), &mut rg);

        let tg = rg.track_gain_db.expect("track_gain_db should be set");
        assert!(
            (tg - (-6.5)).abs() < 1e-9,
            "track_gain_db: expected -6.5, got {tg}"
        );

        let tp = rg.track_peak.expect("track_peak should be set");
        assert!(
            (tp - 0.998244).abs() < 1e-9,
            "track_peak: expected 0.998244, got {tp}"
        );

        let ag = rg.album_gain_db.expect("album_gain_db should be set");
        assert!(
            (ag - (-7.0)).abs() < 1e-9,
            "album_gain_db: expected -7.0, got {ag}"
        );

        let ap = rg.album_peak.expect("album_peak should be set");
        assert!(
            (ap - 0.999512).abs() < 1e-9,
            "album_peak: expected 0.999512, got {ap}"
        );
    }

    /// Verify that lowercase key variants are normalised and parsed.
    #[test]
    fn test_replaygain_lowercase_keys() {
        let tags = [
            make_str_tag("replaygain_track_gain", "+1.23 dB"),
            make_str_tag("replaygain_track_peak", "0.5"),
        ];
        let mut rg = ReplayGainMetadata::default();
        extract_replaygain_from_tags(tags.iter(), &mut rg);

        let tg = rg.track_gain_db.expect("track_gain_db from lowercase key");
        assert!(
            (tg - 1.23).abs() < 1e-9,
            "track_gain_db: expected +1.23, got {tg}"
        );

        let tp = rg.track_peak.expect("track_peak from lowercase key");
        assert!(
            (tp - 0.5).abs() < 1e-9,
            "track_peak: expected 0.5, got {tp}"
        );
    }

    /// Verify that mixed-case key variants are normalised and parsed.
    #[test]
    fn test_replaygain_mixed_case_keys() {
        let tags = [
            make_str_tag("Replaygain_Track_Gain", "-3.0 dB"),
            make_str_tag("Replaygain_Album_Peak", "0.75"),
        ];
        let mut rg = ReplayGainMetadata::default();
        extract_replaygain_from_tags(tags.iter(), &mut rg);

        assert!(
            rg.track_gain_db.is_some(),
            "track_gain_db from mixed-case key"
        );
        assert!(rg.album_peak.is_some(), "album_peak from mixed-case key");
    }

    /// Verify that unrelated tags do not affect the output.
    #[test]
    fn test_replaygain_ignores_unrelated_tags() {
        let tags = [
            make_str_tag("TITLE", "Some Track"),
            make_str_tag("ARTIST", "Some Artist"),
        ];
        let mut rg = ReplayGainMetadata::default();
        extract_replaygain_from_tags(tags.iter(), &mut rg);

        assert!(rg.track_gain_db.is_none());
        assert!(rg.track_peak.is_none());
        assert!(rg.album_gain_db.is_none());
        assert!(rg.album_peak.is_none());
    }

    /// Verify that a gain string without " dB" suffix parses correctly.
    #[test]
    fn test_replaygain_gain_without_db_suffix() {
        let tags = [make_str_tag("REPLAYGAIN_TRACK_GAIN", "-5.25")];
        let mut rg = ReplayGainMetadata::default();
        extract_replaygain_from_tags(tags.iter(), &mut rg);

        let tg = rg
            .track_gain_db
            .expect("should parse gain without dB suffix");
        assert!((tg - (-5.25)).abs() < 1e-9);
    }

    /// Verify that an unparseable gain string yields `None`, not a panic.
    #[test]
    fn test_replaygain_invalid_gain_value() {
        let tags = [make_str_tag("REPLAYGAIN_TRACK_GAIN", "not_a_number")];
        let mut rg = ReplayGainMetadata::default();
        extract_replaygain_from_tags(tags.iter(), &mut rg);
        assert!(
            rg.track_gain_db.is_none(),
            "invalid gain string should yield None"
        );
    }

    /// Verify that default `ReplayGainMetadata` has all-`None` fields.
    #[test]
    fn test_replaygain_default_all_none() {
        let rg = ReplayGainMetadata::default();
        assert!(rg.track_gain_db.is_none());
        assert!(rg.track_peak.is_none());
        assert!(rg.album_gain_db.is_none());
        assert!(rg.album_peak.is_none());
    }
}
