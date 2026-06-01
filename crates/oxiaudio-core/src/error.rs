/// All errors that can be produced by the OxiAudio crate.
#[derive(Debug, thiserror::Error)]
pub enum OxiAudioError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode failed: {0}")]
    Decode(String),
    #[error("encode failed: {0}")]
    Encode(String),
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("invalid channel layout: {0}")]
    InvalidChannelLayout(String),
    #[error("invalid sample rate: {0}")]
    InvalidSampleRate(String),
    #[error("buffer overflow: {0}")]
    BufferOverflow(String),
    #[error("buffer underflow: {0}")]
    BufferUnderflow(String),
}
