//! Decode-side helpers: file decode, streaming decode, metadata-aware decode, format detection.

use oxiaudio_core::{AudioBuffer, AudioDecoder, AudioFormat, AudioMetadata, OxiAudioError};

/// Default block size (frames) used by `decode_stream`.
pub(crate) const DEFAULT_STREAM_BLOCK: usize = 4096;

/// Internal enum to represent either a live streaming decoder or a deferred init error.
///
/// This lets `decode_stream` / `decode_stream_with_block_size` return a concrete `impl Iterator`
/// without `Box<dyn Iterator>` by unifying the two states in a single enum.
pub(crate) enum DecodeStream {
    // Boxed: `StreamingDecoder` is much larger than the `Init` variant.
    Active(Box<oxiaudio_decode::StreamingDecoder>),
    Init(Option<OxiAudioError>),
}

impl Iterator for DecodeStream {
    type Item = Result<AudioBuffer<f32>, OxiAudioError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DecodeStream::Active(d) => d.next(),
            DecodeStream::Init(e) => e.take().map(Err),
        }
    }
}

/// Decode an audio file from disk into an `AudioBuffer<f32>`.
///
/// Supports all formats enabled via the `symphonia` feature flags (WAV, MP3, FLAC, Vorbis, AAC, ALAC).
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// let buf = oxiaudio::decode_file(Path::new("input.flac")).unwrap();
/// println!("Decoded {} frames at {} Hz", buf.frame_count(), buf.sample_rate);
/// ```
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_file(path: impl AsRef<std::path::Path>) -> Result<AudioBuffer<f32>, OxiAudioError> {
    use oxiaudio_decode::SymphoniaDecoder;
    let file = std::fs::File::open(path.as_ref()).map_err(OxiAudioError::Io)?;
    let reader = std::io::BufReader::new(file);
    SymphoniaDecoder.decode(reader)
}

// ─── Additional decode re-exports ────────────────────────────────────────────

/// Detect the audio format of a file by path without decoding audio frames.
pub use oxiaudio_decode::{detect_format_file, detect_format_from_bytes, AudioFormatHint};

/// Probe the audio format of a file by path, using the file extension as a hint.
///
/// This is a convenience wrapper around the decode-crate's path-aware format detection.
/// Supports `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`, `.aiff`, `.au`, and more.
pub use oxiaudio_decode::detect_format_from_path;

/// Decode an AIFF file from disk.
pub use oxiaudio_decode::decode_aiff_file;

/// Decode an AU (Sun/NeXT) file from disk.
pub use oxiaudio_decode::decode_au_file;

/// Decode raw PCM from a file with explicit format config.
pub use oxiaudio_decode::{decode_raw_pcm_file, RawPcmConfig};

/// Probe a file's audio format without decoding any audio frames.
///
/// Returns an [`AudioFormat`] describing the sample rate, channel layout, and sample format
/// of the first audio track found in the container.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// let fmt = oxiaudio::detect_format(Path::new("input.flac")).unwrap();
/// println!("Sample rate: {} Hz, channels: {}", fmt.sample_rate, fmt.channels);
/// ```
#[must_use = "discarding the Result ignores detection errors"]
pub fn detect_format(path: impl AsRef<std::path::Path>) -> Result<AudioFormat, OxiAudioError> {
    let file = std::fs::File::open(path.as_ref()).map_err(OxiAudioError::Io)?;
    let reader = std::io::BufReader::new(file);
    oxiaudio_decode::detect_format(reader)
}

/// Return the [`AudioFormat`] (sample rate, channel layout, native bit depth) of an audio file.
///
/// Equivalent to [`detect_format`] but uses the name `file_format` to match ergonomic
/// UI/CLI patterns where format-aware display is the primary concern.  Internally probes
/// the file through the full Symphonia pipeline so the result reflects the actual codec
/// parameters (e.g. `SampleFormat::I16` for a 16-bit WAV) rather than magic-byte guesses.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// let fmt = oxiaudio::file_format(Path::new("input.wav")).unwrap();
/// println!("{}Hz {} {:?}", fmt.sample_rate, fmt.channels, fmt.format);
/// ```
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened, or [`OxiAudioError::Decode`]
/// if no audio track is found or codec parameters are missing.
#[must_use = "discarding the Result ignores detection errors"]
pub fn file_format(path: impl AsRef<std::path::Path>) -> Result<AudioFormat, OxiAudioError> {
    detect_format(path)
}

/// Stream-decode an audio source in chunks, returning an iterator of `AudioBuffer<f32>` blocks.
///
/// Each yielded buffer has at most `DEFAULT_STREAM_BLOCK` (4096) frames. The last chunk
/// may be smaller. Errors during construction (format probe failure, etc.) are surfaced as
/// the first — and only — `Err` item yielded by the iterator.
///
/// # Example
///
/// ```no_run
/// # use oxiaudio::decode_stream;
/// let file = std::fs::File::open("audio.wav").unwrap();
/// let reader = std::io::BufReader::new(file);
/// for chunk in decode_stream(reader) {
///     let buf = chunk.unwrap();
///     println!("{} samples", buf.samples.len());
/// }
/// ```
pub fn decode_stream(
    reader: impl std::io::Read + std::io::Seek + Send + Sync + 'static,
) -> impl Iterator<Item = Result<AudioBuffer<f32>, OxiAudioError>> {
    decode_stream_with_block_size(reader, DEFAULT_STREAM_BLOCK)
}

/// Stream-decode an audio source with a configurable block size (number of frames per chunk).
///
/// Errors during construction are surfaced as the first yielded `Err` item.
pub fn decode_stream_with_block_size(
    reader: impl std::io::Read + std::io::Seek + Send + Sync + 'static,
    block_size: usize,
) -> impl Iterator<Item = Result<AudioBuffer<f32>, OxiAudioError>> {
    match oxiaudio_decode::StreamingDecoder::new(reader, block_size) {
        Ok(d) => DecodeStream::Active(Box::new(d)),
        Err(e) => DecodeStream::Init(Some(e)),
    }
}

/// Decode a file and extract its metadata in a single pass.
///
/// Returns the decoded `AudioBuffer<f32>` together with any embedded [`AudioMetadata`]
/// (title, artist, album, duration). Avoids opening the file twice compared to calling
/// `decode_file` and a separate metadata probe.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_file_with_metadata(
    path: impl AsRef<std::path::Path>,
) -> Result<(AudioBuffer<f32>, AudioMetadata), OxiAudioError> {
    let file = std::fs::File::open(path.as_ref()).map_err(OxiAudioError::Io)?;
    let reader = std::io::BufReader::new(file);
    oxiaudio_decode::decode_with_metadata(reader)
}

/// Decode a file path to an `AudioBuffer<f64>`.
///
/// Decodes the file using the standard Symphonia pipeline (same formats as [`decode_file`]),
/// then converts the resulting `f32` samples to `f64` precision.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_file_f64(
    path: impl AsRef<std::path::Path>,
) -> Result<AudioBuffer<f64>, OxiAudioError> {
    decode_file(path).map(|buf| buf.to_f64())
}

/// Decode multiple files in parallel using rayon, returning one `Result` per path in order.
pub fn decode_files(paths: &[&std::path::Path]) -> Vec<Result<AudioBuffer<f32>, OxiAudioError>> {
    use rayon::prelude::*;
    paths.par_iter().map(decode_file).collect()
}

// ─── M18 decode additions ─────────────────────────────────────────────────────

/// Gapless playback metadata extracted from a LAME/Xing MP3 header.
pub use oxiaudio_decode::GaplessInfo;

/// Parse LAME gapless playback info (encoder delay, padding, total samples) from raw MP3 bytes.
pub use oxiaudio_decode::parse_gapless_info;

/// Decode an AIFF-C compressed stream (ULAW, ALAW, or uncompressed NONE) from any `Read + Seek`.
pub use oxiaudio_decode::decode_aiffc_compressed;

/// Decode an AIFF-C compressed file by path.
pub use oxiaudio_decode::decode_aiffc_compressed_file;

/// Convert a μ-law encoded byte to a linear f32 sample in the range `[-1.0, 1.0]`.
pub use oxiaudio_decode::ulaw_to_linear;

/// Convert an A-law encoded byte to a linear f32 sample in the range `[-1.0, 1.0]`.
pub use oxiaudio_decode::alaw_to_linear;

// M19 — decode_reader, ReplayGain, gapless trim
pub use oxiaudio_decode::{
    apply_gapless_trim, decode_reader, parse_replaygain, ReplayGainMetadata,
};

// M20 — lyrics, ID3v1, i16/i32 decode
pub use oxiaudio_decode::{decode_to_i16, decode_to_i32, extract_lyrics, parse_id3v1};

// ─── M21 decode additions ─────────────────────────────────────────────────────

/// Embedded album artwork extracted from audio file tags (JPEG, PNG, etc.).
pub use oxiaudio_decode::AlbumArtwork;

/// Extract embedded album artwork from an audio file (ID3v2 APIC, Vorbis PICTURE, etc.).
pub use oxiaudio_decode::extract_album_art;

/// Fluent builder for `StreamingDecoder` with configurable block size.
pub use oxiaudio_decode::StreamingDecoderBuilder;

/// A cue point parsed from a WAV `cue ` chunk.
pub use oxiaudio_decode::WavCuePoint;

/// Parse cue points from a WAV file at a path.
pub use oxiaudio_decode::parse_wav_cues;

/// Parse cue points from any `Read + Seek` WAV reader.
pub use oxiaudio_decode::parse_wav_cues_reader;

// ─── M22 decode additions ─────────────────────────────────────────────────────

/// Decode an audio file and apply the embedded ReplayGain track gain if found.
pub use oxiaudio_decode::decode_file_with_replaygain;

/// Decode an audio file to 64-bit float samples.
pub use oxiaudio_decode::decode_to_f64;

// ─── M23-K decode additions ───────────────────────────────────────────────────

/// Policy controlling what happens when a corrupted or undecodable packet is encountered.
pub use oxiaudio_decode::CorruptPacketPolicy;

/// Options for `decode_file_with_options`.
pub use oxiaudio_decode::DecodeOptions;

/// Decode an audio file with configurable error-recovery policy.
pub use oxiaudio_decode::decode_file_with_options;

/// Decode as much audio as possible from a file, skipping corrupted frames; never panics.
pub use oxiaudio_decode::decode_tolerant;

// ─── OGG Opus pure-Rust decoder ──────────────────────────────────────────────

/// Decode an OGG Opus file from any [`std::io::Read`] reader to an `AudioBuffer<f32>`.
///
/// Requires the `opus` feature on `oxiaudio-decode`; returns
/// [`OxiAudioError::UnsupportedFormat`] when the feature is not enabled.
pub use oxiaudio_decode::decode_opus_reader;

/// Decode an OGG Opus file at `path` to an `AudioBuffer<f32>`.
///
/// Requires the `opus` feature on `oxiaudio-decode`; returns
/// [`OxiAudioError::UnsupportedFormat`] when the feature is not enabled.
pub use oxiaudio_decode::decode_opus_file;

/// Parsed OGG Opus identification header (RFC 7845 §5.1).
pub use oxiaudio_decode::OpusHead;

/// Parse a raw `OpusHead` binary packet (RFC 7845 §5.1).
pub use oxiaudio_decode::parse_opus_head;

// ─── FLAC cue sheet ───────────────────────────────────────────────────────────

/// A single cue point parsed from a FLAC CUESHEET metadata block.
pub use oxiaudio_decode::FlacCuePoint;

/// Parse the CUESHEET metadata block from a FLAC file.
///
/// Returns an empty `Vec` when no CUESHEET block is present (e.g. non-FLAC or untagged files).
pub use oxiaudio_decode::parse_flac_cue_sheet;

// ─── MIDI SMF parser ─────────────────────────────────────────────────────────

/// A fully parsed Standard MIDI File (SMF), including format, ticks-per-quarter, and tracks.
pub use oxiaudio_decode::MidiFile;

/// A single `MTrk` chunk: ordered list of time-stamped track events.
pub use oxiaudio_decode::MidiTrack;

/// A MIDI channel message (NoteOn/Off, ControlChange, ProgramChange, PitchBend, etc.).
pub use oxiaudio_decode::MidiEvent;

/// A meta event within an SMF track (Tempo, TimeSignature, KeySignature, TrackName, etc.).
pub use oxiaudio_decode::MetaEvent;

/// A track event payload — either a MIDI channel message, meta event, or SysEx.
pub use oxiaudio_decode::TrackEvent;

/// A time-stamped event (absolute tick position + event payload).
pub use oxiaudio_decode::TimedEvent;

/// SMF format selector: SingleTrack (0), MultiTrack (1), or Patterns (2).
pub use oxiaudio_decode::SmfFormat;

// ─── MIDI synthesizer ─────────────────────────────────────────────────────────

/// Synthesize a `MidiFile` to a mono `AudioBuffer<f32>` using the given `MidiSynthConfig`.
pub use oxiaudio_decode::synthesize_midi;

/// Synthesize a `MidiFile` to a mono `AudioBuffer<f32>` with default synthesizer settings.
pub use oxiaudio_decode::synthesize_midi_default;

/// Configuration for the MIDI polyphonic synthesizer (waveform, ADSR, polyphony, volume).
pub use oxiaudio_decode::SynthConfig as MidiSynthConfig;

/// Oscillator waveform shape used by the MIDI synthesizer.
pub use oxiaudio_decode::Waveform as MidiWaveform;

/// ADSR envelope parameters used by the MIDI synthesizer.
pub use oxiaudio_decode::Adsr as MidiAdsr;

/// Convert a MIDI note number (0–127) to frequency in Hz (A4=69=440 Hz).
pub use oxiaudio_decode::midi_note_to_hz;
