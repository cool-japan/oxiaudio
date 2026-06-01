//! Type definitions for the LAME MP3 encoder: modes, presets, tags, and builders.

/// MPEG channel mode for the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LameMode {
    Stereo,
    JointStereo,
    /// Note: upstream `mp3lame-encoder` spells this `DaulChannel` (typo preserved).
    DualChannel,
    Mono,
    /// Variable Bit Rate mode. `quality` is 0 (best/largest) to 9 (worst/smallest).
    Vbr {
        quality: u8,
    },
    /// Average Bit Rate mode targeting `target_kbps` (one of the 14 LAME bitrates).
    Abr {
        target_kbps: u32,
    },
    /// Force a mono encode by summing stereo input to one channel before
    /// encoding (avoids dual-mono bitrate waste for mono sources fed as stereo).
    ForcedMono,
}

/// Named VBR quality presets mapping to LAME `-V` quality levels.
///
/// Presets pick a sensible VBR quality for common use cases. Use
/// [`VbrPreset::quality`] to obtain the underlying `0..=9` quality value,
/// [`VbrPreset::quality_value`] for the `i32` form used by builder APIs, or
/// [`VbrPreset::to_mode`] for a ready-to-use [`LameMode::Vbr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VbrPreset {
    /// V6 (~115 kbps mono) — speech / voice memos.
    Voice,
    /// V5 (~130 kbps) — podcasts.
    Podcast,
    /// V2 (~190 kbps) — transparent music for most listeners.
    Music,
    /// V1 (~225 kbps) — high fidelity.
    HiFidelity,
    /// V0 (~245 kbps) — highest VBR fidelity.
    HighFidelity,
    /// V0 (~245 kbps) — archival (alias of `HighFidelity`, max quality).
    Archival,
}

impl VbrPreset {
    /// The LAME VBR quality value (`0` = best, `9` = worst) for this preset.
    pub fn quality(self) -> u8 {
        self.quality_value() as u8
    }

    /// The LAME VBR quality value as `i32` (`0` = best, `9` = worst) for this preset.
    ///
    /// This is the form expected by builder APIs such as [`super::encoder::LameMp3EncoderBuilder::with_vbr_preset`].
    pub fn quality_value(self) -> i32 {
        match self {
            VbrPreset::Voice => 6,
            VbrPreset::Podcast => 5,
            VbrPreset::Music => 2,
            VbrPreset::HiFidelity => 1,
            VbrPreset::HighFidelity | VbrPreset::Archival => 0,
        }
    }

    /// Convert this preset to a [`LameMode::Vbr`] value.
    pub fn to_mode(self) -> LameMode {
        LameMode::Vbr {
            quality: self.quality(),
        }
    }
}

/// Album art payload for the ID3v2 APIC frame (attached picture).
#[derive(Debug, Clone, Default)]
pub struct AlbumArt {
    /// MIME type string, e.g. "image/jpeg" or "image/png".
    pub mime_type: String,
    /// Raw image bytes (JPEG or PNG).
    pub data: Vec<u8>,
}

/// ID3v2.4 tag fields (UTF-8 text encoding).
#[derive(Debug, Clone, Default)]
pub struct Mp3Tags {
    /// Song title → ID3 frame `TIT2`.
    pub title: Option<String>,
    /// Artist name → ID3 frame `TPE1`.
    pub artist: Option<String>,
    /// Album name → ID3 frame `TALB`.
    pub album: Option<String>,
    /// Track number → ID3 frame `TRCK`.
    pub track_number: Option<u16>,
    /// Year → ID3 frame `TDRC` (ID3v2.4 recording time).
    pub year: Option<i32>,
    /// Genre → ID3 frame `TCON`.
    pub genre: Option<String>,
    /// Composer → ID3 frame `TCOM`.
    pub composer: Option<String>,
    /// Comment → ID3 frame `COMM` (simplified; written as a `COMM` text frame).
    pub comment: Option<String>,
    /// Disc number → ID3 frame `TPOS`.
    pub disc_number: Option<u32>,
    /// Track number as `u32` (alias for builder API compatibility; preferred over `track_number`).
    pub track: Option<u32>,
    /// Album art → ID3 frame `APIC` (attached picture, front cover).
    pub album_art: Option<AlbumArt>,
    /// ReplayGain track gain in dB (e.g. -6.50) → ID3 `TXXX` frame.
    pub replaygain_track_gain: Option<f64>,
    /// ReplayGain track peak amplitude (linear, e.g. 0.988) → ID3 `TXXX` frame.
    pub replaygain_track_peak: Option<f64>,
    /// ReplayGain album gain in dB (e.g. -7.20) → ID3 `TXXX` frame `REPLAYGAIN_ALBUM_GAIN`.
    ///
    /// Requires a multi-file analysis pass to compute (unlike track gain which is per-file).
    pub replaygain_album_gain: Option<f64>,
    /// ReplayGain album peak amplitude (linear, e.g. 0.995) → ID3 `TXXX` frame `REPLAYGAIN_ALBUM_PEAK`.
    pub replaygain_album_peak: Option<f64>,
    /// Lyrics text → ID3 frame `USLT`.
    pub lyrics: Option<String>,
    /// Arbitrary user-defined key-value text frames (TXXX).
    pub user_defined: Vec<(String, String)>,
    /// Encoder delay in samples for gapless playback (stored in iTunSMPB COMM frame).
    ///
    /// LAME's algorithmic delay is [`super::LAME_ENCODER_DELAY`] (576 samples). Set to
    /// `Some(576)` for standard LAME encodes, or `None` to omit the iTunSMPB frame.
    /// Both `encoder_delay` and `encoder_padding` must be `Some` for the frame to be written.
    pub encoder_delay: Option<u32>,
    /// End-padding samples for gapless playback (stored in iTunSMPB COMM frame).
    ///
    /// Set to `Some(0)` if unknown, or compute precisely from the encoded output.
    /// Both `encoder_delay` and `encoder_padding` must be `Some` for the frame to be written.
    pub encoder_padding: Option<u32>,
    /// When `true`, include an ID3v2 extended header with a CRC-32 of the frame data.
    ///
    /// The extended header flag (bit 6 = 0x40) is set in the ID3v2 header flags byte.
    /// The CRC covers all frame bytes using the IEEE 802.3 polynomial (0xEDB88320).
    pub extended_header_crc: bool,
    /// When `true`, append an ID3v2 footer ("3DI") after the tag frames.
    ///
    /// The footer is a 10-byte mirror of the header (identifier "3DI" instead of "ID3"),
    /// enabling tag discovery when the tag is appended to the end of a file.
    /// Setting this also sets the footer flag (bit 4 = 0x10) in the header flags byte.
    pub write_footer: bool,
}

impl Mp3Tags {
    /// Return a new [`Mp3TagsBuilder`] for fluent tag construction.
    pub fn builder() -> Mp3TagsBuilder {
        Mp3TagsBuilder::new()
    }
}

/// Fluent builder for [`Mp3Tags`].
///
/// # Example
/// ```rust,no_run
/// # use oxiaudio_encode_mp3_lame::lame::{Mp3Tags};
/// let tags = Mp3Tags::builder()
///     .title("My Song")
///     .artist("The Artist")
///     .genre("Electronic")
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct Mp3TagsBuilder {
    tags: Mp3Tags,
}

impl Mp3TagsBuilder {
    /// Create a new builder with all fields unset.
    pub fn new() -> Self {
        Self {
            tags: Mp3Tags::default(),
        }
    }

    /// Set the song title.
    pub fn title(mut self, v: impl Into<String>) -> Self {
        self.tags.title = Some(v.into());
        self
    }

    /// Set the artist name.
    pub fn artist(mut self, v: impl Into<String>) -> Self {
        self.tags.artist = Some(v.into());
        self
    }

    /// Set the album name.
    pub fn album(mut self, v: impl Into<String>) -> Self {
        self.tags.album = Some(v.into());
        self
    }

    /// Set the year (as a string for maximum flexibility).
    pub fn year(mut self, v: impl Into<String>) -> Self {
        // Parse if numeric; store as string via i32 round-trip where possible.
        let s: String = v.into();
        if let Ok(n) = s.parse::<i32>() {
            self.tags.year = Some(n);
        }
        self
    }

    /// Set the year as an integer.
    pub fn year_int(mut self, v: i32) -> Self {
        self.tags.year = Some(v);
        self
    }

    /// Set the track number.
    pub fn track(mut self, v: u32) -> Self {
        self.tags.track = Some(v);
        self
    }

    /// Set the genre.
    pub fn genre(mut self, v: impl Into<String>) -> Self {
        self.tags.genre = Some(v.into());
        self
    }

    /// Set the composer.
    pub fn composer(mut self, v: impl Into<String>) -> Self {
        self.tags.composer = Some(v.into());
        self
    }

    /// Set the comment.
    pub fn comment(mut self, v: impl Into<String>) -> Self {
        self.tags.comment = Some(v.into());
        self
    }

    /// Set the disc number.
    pub fn disc_number(mut self, v: u32) -> Self {
        self.tags.disc_number = Some(v);
        self
    }

    /// Attach album art (APIC frame, front cover).
    pub fn album_art(mut self, art: AlbumArt) -> Self {
        self.tags.album_art = Some(art);
        self
    }

    /// Set the ReplayGain track gain in dB (e.g. -6.5).
    pub fn replaygain_track_gain(mut self, db: f64) -> Self {
        self.tags.replaygain_track_gain = Some(db);
        self
    }

    /// Set the ReplayGain track peak amplitude (linear, e.g. 0.988).
    pub fn replaygain_track_peak(mut self, peak: f64) -> Self {
        self.tags.replaygain_track_peak = Some(peak);
        self
    }

    /// Set the ReplayGain album gain in dB (e.g. -7.2).
    ///
    /// Requires a multi-file analysis pass to compute. Stored as a TXXX frame
    /// with key `REPLAYGAIN_ALBUM_GAIN`.
    pub fn with_replaygain_album_gain(mut self, db: f64) -> Self {
        self.tags.replaygain_album_gain = Some(db);
        self
    }

    /// Set the ReplayGain album peak amplitude (linear, e.g. 0.995).
    ///
    /// Stored as a TXXX frame with key `REPLAYGAIN_ALBUM_PEAK`.
    pub fn with_replaygain_album_peak(mut self, peak: f64) -> Self {
        self.tags.replaygain_album_peak = Some(peak);
        self
    }

    /// Set the lyrics text (USLT frame).
    pub fn lyrics(mut self, text: impl Into<String>) -> Self {
        self.tags.lyrics = Some(text.into());
        self
    }

    /// Add a user-defined key-value TXXX frame.
    pub fn user_defined(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.user_defined.push((key.into(), value.into()));
        self
    }

    /// Set the encoder delay in samples for gapless playback (iTunSMPB).
    ///
    /// Use [`super::LAME_ENCODER_DELAY`] (576) for standard LAME encodes. Both
    /// `encoder_delay` and `encoder_padding` must be set for the iTunSMPB COMM
    /// frame to be written into the ID3v2.4 tag.
    pub fn encoder_delay(mut self, delay: u32) -> Self {
        self.tags.encoder_delay = Some(delay);
        self
    }

    /// Set the end-padding samples for gapless playback (iTunSMPB).
    ///
    /// Set to `0` if the exact value is unknown. Both `encoder_delay` and
    /// `encoder_padding` must be set for the iTunSMPB COMM frame to be written.
    pub fn encoder_padding(mut self, padding: u32) -> Self {
        self.tags.encoder_padding = Some(padding);
        self
    }

    /// Enable an ID3v2 extended header containing a CRC-32 of the frame data.
    ///
    /// The IEEE 802.3 CRC-32 covers all frame bytes written after the extended header.
    pub fn extended_header_crc(mut self, enable: bool) -> Self {
        self.tags.extended_header_crc = enable;
        self
    }

    /// Enable an ID3v2 footer ("3DI") appended after all tag frames.
    ///
    /// Useful when the tag is appended to a file end; decoders can scan backwards.
    pub fn write_footer(mut self, enable: bool) -> Self {
        self.tags.write_footer = enable;
        self
    }

    /// Consume the builder and return the [`Mp3Tags`].
    pub fn build(self) -> Mp3Tags {
        self.tags
    }
}
