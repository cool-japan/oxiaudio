//! LAME-backed MP3 encoder: types, encoder, streaming encoder, and ID3v2 tag writer.

mod encoder;
pub mod id3v2;
mod stream;
mod types;

// Re-export all public items at the `lame` module level so the public API is unchanged.
pub use encoder::{
    encode_mp3_abr, encode_mp3_cbr_to_file, encode_mp3_cbr_to_vec, encode_mp3_with_auto_replaygain,
    write_xing_replaygain, LameMp3Encoder, LameMp3EncoderBuilder, LAME_ENCODER_DELAY,
};
pub use oxiaudio_core::{AudioBuffer, AudioEncoder, ChannelLayout, OxiAudioError};
pub use stream::LameMp3StreamEncoder;
pub use types::{AlbumArt, LameMode, Mp3Tags, Mp3TagsBuilder, VbrPreset};

#[cfg(test)]
#[path = "lame_tests.rs"]
mod tests;
