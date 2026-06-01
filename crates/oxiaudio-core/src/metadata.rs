/// Container/track metadata. All fields optional; `Default` = all `None`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AudioMetadata {
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub duration_secs: Option<f64>,
    pub bitrate_kbps: Option<u32>,
    pub genre: Option<String>,
    pub composer: Option<String>,
    pub year: Option<u32>,
    pub track_number: Option<u32>,
    pub disc_number: Option<u32>,
    pub comment: Option<String>,
    /// Raw album art image bytes (e.g. JPEG or PNG), extracted from embedded tags.
    /// `None` if no artwork is present or artwork extraction was not requested.
    #[cfg_attr(
        feature = "serde",
        serde(skip_serializing_if = "Option::is_none", default)
    )]
    pub album_art: Option<Vec<u8>>,
}
