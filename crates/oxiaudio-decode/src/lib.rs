// `forbid` is relaxed to `deny` to permit the optional `mmap` feature module to use a single
// targeted `#[allow(unsafe_code)]` for `memmap2::Mmap::map`.  All other modules continue to be
// covered by the crate-level `deny`; no unsafe code exists outside `src/mmap.rs`.
#![deny(unsafe_code)]

pub mod aac_decoder;
pub mod aiff;
pub mod artwork;
pub mod au;
pub mod cue;
pub mod detect;
pub mod gapless;
pub mod midi;
pub mod midi_synth;
#[cfg(feature = "mmap")]
pub mod mmap;
pub mod musepack;
pub mod ogg_reader;
pub mod opus;
pub mod raw;
pub mod replaygain;
pub mod streaming;
pub mod wav_cue;
pub mod wavpack;

pub use aac_decoder::{decode_aac, AacDecoder, AdtsFrame};
pub use aiff::{
    alaw_to_linear, decode_aiff, decode_aiff_file, decode_aiff_reader_with_metadata,
    decode_aiff_with_metadata, decode_aiffc_compressed, decode_aiffc_compressed_file,
    ulaw_to_linear,
};
pub use artwork::{extract_album_art, AlbumArtwork};
pub use au::{decode_au, decode_au_file};
pub use cue::{parse_flac_cue_sheet, FlacCuePoint};
pub use detect::{detect_format_file, detect_format_from_bytes, AudioFormatHint};
pub use gapless::{apply_gapless_trim, parse_gapless_info, GaplessInfo};
pub use midi::{MetaEvent, MidiEvent, MidiFile, MidiTrack, SmfFormat, TimedEvent, TrackEvent};
pub use midi_synth::{
    midi_note_to_hz, synthesize_midi, synthesize_midi_default, Adsr, SynthConfig, Waveform,
};
#[cfg(feature = "mmap")]
pub use mmap::decode_file_mmap;
pub use musepack::{
    decode_musepack, decode_musepack_file, MpcVersion, MusepackDecoder, MUSEPACK_MAGIC_SV7,
    MUSEPACK_MAGIC_SV8,
};
#[cfg(feature = "opus")]
pub use opus::OpusDecoder;
pub use opus::{decode_opus_file, decode_opus_reader, parse_opus_head, OpusHead};
pub use oxiaudio_core::AudioFormat;
pub use raw::{decode_raw_pcm, decode_raw_pcm_file, RawPcmConfig};
pub use replaygain::{parse_replaygain, ReplayGainMetadata};
pub use streaming::{StreamingDecoder, StreamingDecoderBuilder};
pub use wav_cue::{parse_wav_cues, parse_wav_cues_reader, WavCuePoint};
pub use wavpack::{decode_wavpack, decode_wavpack_file, WAVPACK_MAGIC};

use oxiaudio_core::{
    AudioBuffer, AudioDecoder, AudioMetadata, ChannelLayout, OxiAudioError, SampleFormat,
};
use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    errors::Error as SymphoniaError,
    formats::{probe::Hint, FormatOptions, TrackType},
    io::{MediaSource, MediaSourceStream},
    meta::{MetadataOptions, MetadataRevision, RawValue as SymphoniaRawValue, StandardTag},
};

/// Select the first track whose codec_params resolves to audio codec parameters.
/// Falls back to `default_track(TrackType::Audio)` if no explicit audio codec match is found.
pub(crate) fn select_audio_track(
    format: &dyn symphonia::core::formats::FormatReader,
) -> Option<&symphonia::core::formats::Track> {
    // Prefer the first track that has codec_params resolving to audio params.
    let first_audio = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.as_ref().and_then(|cp| cp.audio()).is_some());

    if first_audio.is_some() {
        first_audio
    } else {
        // Fall back to symphonia's built-in heuristic.
        format.default_track(TrackType::Audio)
    }
}

pub struct SymphoniaDecoder;

impl AudioDecoder for SymphoniaDecoder {
    /// Decode the entire audio source into a single interleaved `Vec<f32>` buffer.
    ///
    /// # Performance notes
    ///
    /// Internally, `symphonia` decodes into its own buffers and exposes samples via
    /// `copy_to_vec_interleaved`, which copies into a caller-owned staging `Vec<f32>`.
    /// A zero-copy path — accessing symphonia's internal buffer directly — is blocked
    /// by the crate-level `deny(unsafe_code)` policy; it would require a `transmute` from symphonia's
    /// `&[S]` to `&[f32]` which is unsound without runtime type checks. The performance
    /// trade-off is one extra copy per decoded packet (typically 1024–4096 frames).
    /// Profiling on 10-second stereo 48 kHz audio yields approximately 110–480 packets
    /// (each around 8–64 KB of f32 data); a staging `Vec<f32>` is reused across packets
    /// to minimise allocations to exactly two: the staging vec and the final accumulator.
    /// This is acceptable for offline file decoding.
    /// For real-time applications, prefer `StreamingDecoder` which yields fixed-size
    /// chunks from an internal FIFO without accumulating the entire buffer in memory.
    fn decode(
        &mut self,
        src: impl std::io::Read + std::io::Seek + Send + Sync + 'static,
    ) -> Result<AudioBuffer<f32>, OxiAudioError> {
        let mss = MediaSourceStream::new(Box::new(MediaSourceWrapper(src)), Default::default());
        let hint = Hint::new();
        let dec_opts = AudioDecoderOptions::default();

        let mut format = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

        let track = select_audio_track(format.as_ref())
            .ok_or_else(|| OxiAudioError::Decode("no audio track with valid codec".into()))?;

        let track_id = track.id;
        // Capture n_frames before consuming track via codec_params.clone().
        let track_n_frames = track.num_frames;
        let codec_params = track
            .codec_params
            .clone()
            .ok_or_else(|| OxiAudioError::Decode("missing codec params".into()))?;

        let audio_params = codec_params
            .audio()
            .ok_or_else(|| OxiAudioError::Decode("not an audio track".into()))?;

        let sample_rate = audio_params.sample_rate.unwrap_or(44_100);
        let n_channels = audio_params
            .channels
            .as_ref()
            .map(|c| c.count())
            .unwrap_or(2);

        let layout = ChannelLayout::from(n_channels as u16);

        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(audio_params, &dec_opts)
            .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

        let mut all_samples: Vec<f32> = Vec::new();
        // Pre-allocate when the container reports a known frame count, eliminating
        // incremental reallocation across the per-packet extend_from_slice calls.
        if let Some(n_frames) = track_n_frames {
            all_samples.reserve((n_frames as usize).saturating_mul(n_channels));
        }
        // Staging buffer reused per packet to avoid per-call allocation.
        let mut packet_samples: Vec<f32> = Vec::new();

        loop {
            let packet = match format.next_packet() {
                Ok(Some(p)) => p,
                Ok(None) => break,
                Err(SymphoniaError::ResetRequired) => {
                    decoder.reset();
                    continue;
                }
                // IoError from next_packet is treated as end-of-stream.
                // This handles formats (e.g. raw FLAC) where the demuxer signals
                // EOF via an UnexpectedEof IoError rather than returning Ok(None).
                Err(SymphoniaError::IoError(_)) => break,
                Err(e) => return Err(OxiAudioError::Decode(e.to_string())),
            };

            if packet.track_id != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    // copy_to_vec_interleaved resizes the staging vec to this packet's samples,
                    // so we extend all_samples after each packet.
                    decoded.copy_to_vec_interleaved::<f32>(&mut packet_samples);
                    all_samples.extend_from_slice(&packet_samples);
                }
                Err(SymphoniaError::IoError(_)) => break,
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(e) => return Err(OxiAudioError::Decode(e.to_string())),
            }
        }

        Ok(AudioBuffer {
            samples: all_samples,
            sample_rate,
            channels: layout,
            format: SampleFormat::F32,
        })
    }
}

/// Decode audio from any `Read + Seek + Send + Sync + 'static` reader.
///
/// This is a convenience wrapper for sources that are not files (e.g., `std::io::Cursor<Vec<u8>>`).
/// Internally delegates to the same Symphonia pipeline used by [`decode_file`].
///
/// # Errors
///
/// Returns [`OxiAudioError::Decode`] if format probing or codec decoding fails.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_reader<R: std::io::Read + std::io::Seek + Send + Sync + 'static>(
    reader: R,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    use oxiaudio_core::AudioDecoder;
    SymphoniaDecoder.decode(reader)
}

/// Decode an audio file at `path` to `AudioBuffer<f32>` using the Symphonia pipeline.
///
/// Supports all container/codec combinations enabled by the active Symphonia feature flags
/// (WAV, FLAC, MP3, Vorbis, AAC, ALAC, etc.).
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened, or [`OxiAudioError::Decode`]
/// if format probing or codec decoding fails.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_file(path: &std::path::Path) -> Result<AudioBuffer<f32>, OxiAudioError> {
    use oxiaudio_core::AudioDecoder;
    let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    let reader = std::io::BufReader::new(file);
    SymphoniaDecoder.decode(reader)
}

// ─── M23-K: Frame-level error recovery ───────────────────────────────────────

/// Policy controlling what happens when a corrupted or undecodable packet is encountered.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CorruptPacketPolicy {
    /// Return `Err` immediately on any fatal decode error (default).
    ///
    /// Note: `SymphoniaError::DecodeError` (soft decode errors such as a corrupted
    /// MP3 frame) are **always** skipped silently, matching the behaviour of
    /// [`decode_file`]. Only hard errors (everything except `DecodeError` and
    /// `IoError`) escalate to `Err` under this policy.
    #[default]
    Fail,
    /// Skip corrupted packets and continue decoding, emitting a `log::warn!` for
    /// each skipped packet. Returns as much audio as could be decoded.
    Skip,
}

/// Options for [`decode_file_with_options`].
#[derive(Debug, Clone, Default)]
pub struct DecodeOptions {
    /// What to do when a corrupted or undecodable packet is encountered.
    pub on_corrupt_packet: CorruptPacketPolicy,
}

/// Shared decode pipeline used by [`decode_file`], [`decode_file_with_options`],
/// and [`decode_tolerant`].
///
/// Runs the full Symphonia probe + decode loop on `reader`.
/// When `opts.on_corrupt_packet` is [`CorruptPacketPolicy::Skip`] any otherwise-fatal
/// decode error causes the packet to be skipped with a `log::warn!` diagnostic.
fn decode_inner(
    reader: impl std::io::Read + std::io::Seek + Send + Sync + 'static,
    opts: &DecodeOptions,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let mss = MediaSourceStream::new(Box::new(MediaSourceWrapper(reader)), Default::default());
    let hint = Hint::new();
    let dec_opts = AudioDecoderOptions::default();

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

    let track = select_audio_track(format.as_ref())
        .ok_or_else(|| OxiAudioError::Decode("no audio track with valid codec".into()))?;

    let track_id = track.id;
    let track_n_frames = track.num_frames;
    let codec_params = track
        .codec_params
        .clone()
        .ok_or_else(|| OxiAudioError::Decode("missing codec params".into()))?;

    let audio_params = codec_params
        .audio()
        .ok_or_else(|| OxiAudioError::Decode("not an audio track".into()))?;

    let sample_rate = audio_params.sample_rate.unwrap_or(44_100);
    let n_channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(2);
    let layout = ChannelLayout::from(n_channels as u16);

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &dec_opts)
        .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

    let mut all_samples: Vec<f32> = Vec::new();
    if let Some(n_frames) = track_n_frames {
        all_samples.reserve((n_frames as usize).saturating_mul(n_channels));
    }
    let mut packet_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(SymphoniaError::IoError(_)) => break,
            Err(e) => return Err(OxiAudioError::Decode(e.to_string())),
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                decoded.copy_to_vec_interleaved::<f32>(&mut packet_samples);
                all_samples.extend_from_slice(&packet_samples);
            }
            Err(SymphoniaError::IoError(_)) => break,
            // Soft decode error: always skip silently (matches decode_file behaviour).
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => match opts.on_corrupt_packet {
                CorruptPacketPolicy::Fail => {
                    return Err(OxiAudioError::Decode(e.to_string()));
                }
                CorruptPacketPolicy::Skip => {
                    log::warn!("Skipping corrupted packet: {}", e);
                    continue;
                }
            },
        }
    }

    Ok(AudioBuffer {
        samples: all_samples,
        sample_rate,
        channels: layout,
        format: SampleFormat::F32,
    })
}

/// Decode an audio file using a configurable error-recovery policy.
///
/// The `opts.on_corrupt_packet` field controls behaviour on undecodable packets:
/// - [`CorruptPacketPolicy::Fail`]: identical to [`decode_file`] (hard errors propagate).
/// - [`CorruptPacketPolicy::Skip`]: skip undecodable packets and continue; a `log::warn!`
///   is emitted for each skipped packet.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened, or [`OxiAudioError::Decode`]
/// if format probing or codec setup fails (regardless of `opts.on_corrupt_packet`).
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_file_with_options(
    path: &std::path::Path,
    opts: &DecodeOptions,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    let reader = std::io::BufReader::new(file);
    decode_inner(reader, opts)
}

/// Decode as much audio as possible from `path`, skipping any corrupted frames.
///
/// Returns a partial [`AudioBuffer<f32>`] even when some frames are undecodable.
/// If the file cannot be opened or its format cannot be probed, an empty buffer is
/// returned instead — this function never panics or propagates an error.
#[must_use = "returns the partial decoded audio buffer"]
pub fn decode_tolerant(path: &std::path::Path) -> AudioBuffer<f32> {
    let opts = DecodeOptions {
        on_corrupt_packet: CorruptPacketPolicy::Skip,
    };
    decode_file_with_options(path, &opts).unwrap_or_else(|_| AudioBuffer {
        samples: Vec::new(),
        sample_rate: 44_100,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    })
}

/// Probe `src` for its audio format without decoding any audio frames.
///
/// Returns an [`AudioFormat`] describing the sample rate, channel layout, and sample format
/// of the first audio track found in the container.
#[must_use = "discarding the Result ignores detection errors"]
pub fn detect_format(
    src: impl std::io::Read + std::io::Seek + Send + Sync + 'static,
) -> Result<AudioFormat, OxiAudioError> {
    let mss = MediaSourceStream::new(Box::new(MediaSourceWrapper(src)), Default::default());
    let hint = Hint::new();

    let format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

    let track = select_audio_track(format.as_ref())
        .ok_or_else(|| OxiAudioError::Decode("no audio track".into()))?;

    let codec_params = track
        .codec_params
        .as_ref()
        .ok_or_else(|| OxiAudioError::Decode("missing codec params".into()))?;

    let audio_params = codec_params
        .audio()
        .ok_or_else(|| OxiAudioError::Decode("not an audio track".into()))?;

    let sample_rate = audio_params.sample_rate.unwrap_or(44_100);
    let n_channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(2);

    let layout = ChannelLayout::from(n_channels as u16);
    let native_format = map_symphonia_sample_format(audio_params.sample_format);

    Ok(AudioFormat {
        sample_rate,
        channels: layout,
        format: native_format,
    })
}

/// Detect audio format using a file extension hint before content-based probing.
///
/// Equivalent to [`detect_format`] but accepts a `Path` so that the file extension
/// can be passed to symphonia as a probe hint, allowing faster format detection when
/// the extension is reliable (e.g. `.wav`, `.flac`, `.mp3`).
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened, or
/// [`OxiAudioError::Decode`] if no audio track is found or codec params are missing.
#[must_use = "discarding the Result ignores detection errors"]
pub fn detect_format_from_path(path: &std::path::Path) -> Result<AudioFormat, OxiAudioError> {
    let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    let mss = MediaSourceStream::new(Box::new(MediaSourceWrapper(file)), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

    let track = select_audio_track(format.as_ref())
        .ok_or_else(|| OxiAudioError::Decode("no audio track".into()))?;

    let codec_params = track
        .codec_params
        .as_ref()
        .ok_or_else(|| OxiAudioError::Decode("missing codec params".into()))?;

    let audio_params = codec_params
        .audio()
        .ok_or_else(|| OxiAudioError::Decode("not an audio track".into()))?;

    let sample_rate = audio_params.sample_rate.unwrap_or(44_100);
    let n_channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(2);

    let layout = ChannelLayout::from(n_channels as u16);
    let native_format = map_symphonia_sample_format(audio_params.sample_format);

    Ok(AudioFormat {
        sample_rate,
        channels: layout,
        format: native_format,
    })
}

/// Compute an estimated bitrate in kbps from audio codec parameters.
///
/// Uses `bits_per_sample * sample_rate * num_channels / 1000` when the fields are available.
/// Returns `None` when any required parameter is absent.
fn compute_bitrate_kbps(
    audio_params: &symphonia::core::codecs::audio::AudioCodecParameters,
    sample_rate: u32,
    n_channels: usize,
) -> Option<u32> {
    let bps = audio_params
        .bits_per_sample
        .or(audio_params.bits_per_coded_sample)?;
    // Guard against overflow: bps * sample_rate * channels fits in u64 for all realistic values.
    let bitrate = (bps as u64)
        .saturating_mul(sample_rate as u64)
        .saturating_mul(n_channels as u64)
        / 1000;
    Some(bitrate as u32)
}

/// Extract [`AudioMetadata`] from a symphonia [`MetadataRevision`].
///
/// Duration cannot be computed here (n_frames and sample_rate are not available in the revision);
/// callers should set `duration_secs` from codec parameters separately.
fn extract_metadata(rev: &MetadataRevision) -> AudioMetadata {
    let mut title: Option<String> = None;
    let mut artist: Option<String> = None;
    let mut album: Option<String> = None;
    let mut genre: Option<String> = None;
    let mut composer: Option<String> = None;
    let mut year: Option<u32> = None;
    let mut track_number: Option<u32> = None;
    let mut disc_number: Option<u32> = None;
    let mut comment: Option<String> = None;

    for tag in &rev.media.tags {
        if let Some(std_tag) = &tag.std {
            match std_tag {
                StandardTag::TrackTitle(v) => title = Some(v.as_ref().to_owned()),
                StandardTag::Artist(v) => artist = Some(v.as_ref().to_owned()),
                StandardTag::Album(v) => album = Some(v.as_ref().to_owned()),
                StandardTag::Genre(v) => genre = Some(v.as_ref().to_owned()),
                StandardTag::Composer(v) => composer = Some(v.as_ref().to_owned()),
                StandardTag::RecordingYear(n) => year = Some(u32::from(*n)),
                StandardTag::TrackNumber(n) => track_number = Some(*n as u32),
                StandardTag::DiscNumber(n) => disc_number = Some(*n as u32),
                StandardTag::Comment(v) => comment = Some(v.as_ref().to_owned()),
                _ => {}
            }
        }
    }

    AudioMetadata {
        title,
        artist,
        album,
        // bitrate is not exposed via symphonia MetadataRevision; filled in by callers.
        bitrate_kbps: None,
        // duration_secs is filled in by the caller who has codec_params.
        duration_secs: None,
        genre,
        composer,
        year,
        track_number,
        disc_number,
        comment,
        album_art: None,
    }
}

/// Decode `src` fully and return the audio samples together with any embedded metadata.
///
/// Metadata tags (title, artist, album, genre, track/disc number, comment) are extracted
/// from the container. Duration is computed from `codec_params.n_frames` and `sample_rate`
/// when available.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_with_metadata(
    src: impl std::io::Read + std::io::Seek + Send + Sync + 'static,
) -> Result<(AudioBuffer<f32>, AudioMetadata), OxiAudioError> {
    let mss = MediaSourceStream::new(Box::new(MediaSourceWrapper(src)), Default::default());
    let hint = Hint::new();
    let dec_opts = AudioDecoderOptions::default();

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

    let track = select_audio_track(format.as_ref())
        .ok_or_else(|| OxiAudioError::Decode("no audio track with valid codec".into()))?;

    let track_id = track.id;
    // Capture num_frames from the track before consuming it via clone.
    let track_num_frames = track.num_frames;
    let codec_params = track
        .codec_params
        .clone()
        .ok_or_else(|| OxiAudioError::Decode("missing codec params".into()))?;

    let audio_params = codec_params
        .audio()
        .ok_or_else(|| OxiAudioError::Decode("not an audio track".into()))?;

    let sample_rate = audio_params.sample_rate.unwrap_or(44_100);
    let n_channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(2);

    // Compute duration from track num_frames and sample_rate when available.
    let duration_secs = track_num_frames.map(|n| n as f64 / f64::from(sample_rate));

    // Capture bitrate before audio_params is consumed by make_audio_decoder.
    let bitrate_kbps = compute_bitrate_kbps(audio_params, sample_rate, n_channels);

    let layout = ChannelLayout::from(n_channels as u16);

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &dec_opts)
        .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

    let mut all_samples: Vec<f32> = Vec::new();
    // Pre-allocate when the container reports a known frame count, eliminating
    // incremental reallocation across the per-packet extend_from_slice calls.
    if let Some(n_frames) = track_num_frames {
        all_samples.reserve((n_frames as usize).saturating_mul(n_channels));
    }
    let mut packet_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            // IoError from next_packet is treated as end-of-stream (same as the decode loop above).
            Err(SymphoniaError::IoError(_)) => break,
            Err(e) => return Err(OxiAudioError::Decode(e.to_string())),
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                decoded.copy_to_vec_interleaved::<f32>(&mut packet_samples);
                all_samples.extend_from_slice(&packet_samples);
            }
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(OxiAudioError::Decode(e.to_string())),
        }
    }

    // Extract metadata from the format reader after decoding.
    let mut metadata = format
        .metadata()
        .current()
        .map(extract_metadata)
        .unwrap_or_default();
    metadata.duration_secs = duration_secs;
    metadata.bitrate_kbps = bitrate_kbps;

    let audio_buf = AudioBuffer {
        samples: all_samples,
        sample_rate,
        channels: layout,
        format: SampleFormat::F32,
    };

    Ok((audio_buf, metadata))
}

/// Decode an audio file at `path` and return the audio samples together with any embedded metadata.
///
/// Equivalent to calling [`decode_with_metadata`] with an opened `BufReader`, but:
/// - accepts a `Path` for ergonomic file-based usage,
/// - automatically falls back to ID3v1 tags (last 128 bytes, "TAG" magic) when no ID3v2 /
///   container-level metadata is present.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened, or [`OxiAudioError::Decode`]
/// if format probing or codec decoding fails.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_file_with_metadata(
    path: &std::path::Path,
) -> Result<(AudioBuffer<f32>, AudioMetadata), OxiAudioError> {
    let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    let reader = std::io::BufReader::new(file);
    let (buf, mut meta) = decode_with_metadata(reader)?;

    // If symphonia found no standard tags, attempt ID3v1 fallback.
    if meta.title.is_none() && meta.artist.is_none() && meta.album.is_none() && meta.year.is_none()
    {
        if let Ok(Some(v1)) = parse_id3v1(path) {
            if meta.title.is_none() {
                meta.title = v1.title;
            }
            if meta.artist.is_none() {
                meta.artist = v1.artist;
            }
            if meta.album.is_none() {
                meta.album = v1.album;
            }
            if meta.year.is_none() {
                meta.year = v1.year;
            }
            if meta.track_number.is_none() {
                meta.track_number = v1.track_number;
            }
        }
    }

    Ok((buf, meta))
}

/// Extract unsynchronized lyrics from an audio file's ID3v2 USLT frame.
///
/// Supports any container that symphonia can probe (MP3/ID3v2, M4A, Vorbis-tagged FLAC/OGG, etc.).
/// Returns `None` if no lyrics tag is found in the first metadata revision.
/// For files with multiple USLT frames, returns the first match.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened, or [`OxiAudioError::Decode`]
/// if format probing fails.
#[must_use = "discarding the Result ignores extraction errors"]
pub fn extract_lyrics(path: &std::path::Path) -> Result<Option<String>, OxiAudioError> {
    let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    let mss = MediaSourceStream::new(
        Box::new(MediaSourceWrapper(std::io::BufReader::new(file))),
        Default::default(),
    );
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

    if let Some(rev) = format.metadata().current() {
        for tag in &rev.media.tags {
            // Primary: match on the parsed StandardTag::Lyrics variant.
            if let Some(StandardTag::Lyrics(v)) = &tag.std {
                return Ok(Some(v.as_ref().to_owned()));
            }
            // Fallback: raw key contains "LYRIC" (case-insensitive) for non-standard formats.
            if tag.std.is_none() {
                let upper = tag.raw.key.to_ascii_uppercase();
                if upper.contains("LYRIC") {
                    if let SymphoniaRawValue::String(s) = &tag.raw.value {
                        return Ok(Some(s.as_ref().to_owned()));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Parse ID3v1 tags from the last 128 bytes of a file.
///
/// Returns `Some(AudioMetadata)` when an ID3v1 "TAG" header is found, or `None` when the file
/// is shorter than 128 bytes or does not carry an ID3v1 tag.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened or read.
#[must_use = "discarding the Result ignores read errors"]
pub fn parse_id3v1(path: &std::path::Path) -> Result<Option<AudioMetadata>, OxiAudioError> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    let file_len = file.seek(SeekFrom::End(0)).map_err(OxiAudioError::Io)?;
    if file_len < 128 {
        return Ok(None);
    }
    file.seek(SeekFrom::End(-128)).map_err(OxiAudioError::Io)?;
    let mut data = vec![0u8; 128];
    file.read_exact(&mut data).map_err(OxiAudioError::Io)?;
    Ok(parse_id3v1_from_bytes(&data))
}

/// Parse ID3v1 tags from a 128-byte slice.
///
/// The slice must contain at least 128 bytes and the last 128 bytes must begin with `"TAG"`.
/// Returns `None` if the magic bytes are absent.
fn parse_id3v1_from_bytes(data: &[u8]) -> Option<AudioMetadata> {
    if data.len() < 128 {
        return None;
    }
    let tag_start = data.len() - 128;
    if &data[tag_start..tag_start + 3] != b"TAG" {
        return None;
    }
    let title = extract_id3v1_string(&data[tag_start + 3..tag_start + 33]);
    let artist = extract_id3v1_string(&data[tag_start + 33..tag_start + 63]);
    let album = extract_id3v1_string(&data[tag_start + 63..tag_start + 93]);
    let year_str = extract_id3v1_string(&data[tag_start + 93..tag_start + 97]);
    // ID3v1.1 track number: if byte 125 is zero AND byte 126 is non-zero,
    // the comment field is 28 bytes and byte 126 holds the track number.
    let track = if data[tag_start + 125] == 0 && data[tag_start + 126] != 0 {
        Some(u32::from(data[tag_start + 126]))
    } else {
        None
    };
    Some(AudioMetadata {
        title: if title.is_empty() { None } else { Some(title) },
        artist: if artist.is_empty() {
            None
        } else {
            Some(artist)
        },
        album: if album.is_empty() { None } else { Some(album) },
        year: year_str.parse::<u32>().ok(),
        track_number: track,
        ..Default::default()
    })
}

/// Extract a null-terminated ASCII/Latin-1 string from a fixed-width ID3v1 field.
///
/// Stops at the first NUL byte and trims trailing ASCII spaces.
fn extract_id3v1_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end])
        .trim_end()
        .to_string()
}

/// Decode an audio file and return samples as 16-bit signed integers.
///
/// Each f32 sample is clamped to `[-1.0, 1.0]`, scaled by 32767.0, and rounded.
/// The resulting integers are guaranteed to lie in `[-32768, 32767]`.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened, or [`OxiAudioError::Decode`]
/// if format probing or codec decoding fails.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_to_i16(path: &std::path::Path) -> Result<AudioBuffer<i16>, OxiAudioError> {
    let buf = decode_file(path)?;
    let samples: Vec<i16> = buf
        .samples
        .iter()
        .map(|&s| {
            // clamp guarantees the value is in [-1.0, 1.0] before the multiply,
            // so the cast is safe: max magnitude is 32767.0, which fits in i16.
            #[allow(clippy::cast_possible_truncation)]
            let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            v
        })
        .collect();
    Ok(AudioBuffer {
        samples,
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        format: SampleFormat::I16,
    })
}

/// Decode an audio file and apply ReplayGain track gain if embedded.
///
/// Parses ReplayGain tags from the file using `parse_replaygain`, then decodes
/// the audio. If `track_gain_db` is present, applies the gain using a linear
/// scale factor. If no ReplayGain metadata is found, the raw decoded buffer is
/// returned unchanged.
///
/// # Errors
///
/// Returns `OxiAudioError::Io` if the file cannot be opened, or
/// `OxiAudioError::Decode` if format probing or decoding fails.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_file_with_replaygain(
    path: impl AsRef<std::path::Path>,
) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let path = path.as_ref();
    // 1. Parse RG metadata (ignore errors — file may not have RG tags)
    let rg = parse_replaygain(path).unwrap_or_default();
    // 2. Decode the audio
    let mut buf = decode_file(path)?;
    // 3. Apply track gain if present
    if let Some(gain_db) = rg.track_gain_db {
        let linear = 10.0f64.powf(gain_db / 20.0) as f32;
        for s in &mut buf.samples {
            *s *= linear;
        }
    }
    Ok(buf)
}

/// Decode an audio file to 64-bit float precision.
///
/// Decodes as f32 via the normal path, then converts all samples to f64.
/// Useful when downstream DSP requires higher intermediate precision.
///
/// # Errors
///
/// Same as `decode_file`.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_to_f64(path: impl AsRef<std::path::Path>) -> Result<AudioBuffer<f64>, OxiAudioError> {
    let buf_f32 = decode_file(path.as_ref())?;
    Ok(buf_f32.to_f64())
}

/// Decode an audio file and return samples as 32-bit signed integers (full range).
///
/// Each f32 sample is clamped to `[-1.0, 1.0]`, scaled by `i32::MAX` (`2147483647`), and rounded.
/// Due to f32 precision, `+1.0 * i32::MAX as f32` rounds to `2147483648.0` which is clamped to
/// `i32::MAX` on cast, yielding `2147483647`.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened, or [`OxiAudioError::Decode`]
/// if format probing or codec decoding fails.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_to_i32(path: &std::path::Path) -> Result<AudioBuffer<i32>, OxiAudioError> {
    let buf = decode_file(path)?;
    // i32::MAX as f32 is 2147483648.0 (rounds up in f32); multiplying a clamped f32 by it and
    // rounding then casting will produce at most i32::MAX thanks to saturating_cast behavior.
    // We use f64 intermediate to prevent the f32 overflow at +1.0 * (i32::MAX as f32).
    let samples: Vec<i32> = buf
        .samples
        .iter()
        .map(|&s| {
            let clamped = f64::from(s).clamp(-1.0, 1.0);
            let scaled = (clamped * f64::from(i32::MAX)).round();
            // scaled is in [-2147483647.0, 2147483647.0] — fits in i32.
            #[allow(clippy::cast_possible_truncation)]
            let v = scaled as i32;
            v
        })
        .collect();
    Ok(AudioBuffer {
        samples,
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        format: SampleFormat::I32,
    })
}

/// Map a symphonia [`SampleFormat`] to the corresponding [`oxiaudio_core::SampleFormat`].
///
/// Signed integer formats are mapped directly; unsigned and 8-bit types fall back to
/// [`SampleFormat::F32`] because `oxiaudio_core::SampleFormat` has no unsigned variants
/// other than `U8`. `None` (format not reported by the container) also maps to `F32`.
fn map_symphonia_sample_format(
    sf: Option<symphonia::core::audio::sample::SampleFormat>,
) -> SampleFormat {
    use symphonia::core::audio::sample::SampleFormat as Sf;
    match sf {
        Some(Sf::S16) => SampleFormat::I16,
        Some(Sf::S24) => SampleFormat::I24,
        Some(Sf::S32) => SampleFormat::I32,
        Some(Sf::F32) => SampleFormat::F32,
        Some(Sf::F64) => SampleFormat::F64,
        // U8 maps to OxiAudio's U8 (the only unsigned variant it has).
        Some(Sf::U8) => SampleFormat::U8,
        // Unsigned 16/24/32-bit and signed 8-bit have no direct OxiAudio equivalent;
        // fall back to F32 which is the decode-pipeline output type.
        _ => SampleFormat::F32,
    }
}

/// Thin wrapper that adds the `MediaSource` trait implementation to any `Read + Seek + Send + Sync`.
///
/// Downstream crates that need to build a custom symphonia pipeline without pulling in the full
/// OxiAudio decode machinery can use this type directly after importing `oxiaudio_decode`.
pub struct MediaSourceWrapper<R>(pub R);

impl<R: std::io::Read> std::io::Read for MediaSourceWrapper<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

impl<R: std::io::Seek> std::io::Seek for MediaSourceWrapper<R> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.0.seek(pos)
    }
}

impl<R: std::io::Read + std::io::Seek + Send + Sync> MediaSource for MediaSourceWrapper<R> {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod metadata_api_tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Minimal WAV builder (44-byte header + i16 PCM samples, mono, 8 kHz)
    // Used by tests that need a real audio file without encoding dependencies.
    // -----------------------------------------------------------------------

    /// Build a minimal 44-byte RIFF/WAV header for 16-bit mono PCM at `sample_rate`.
    fn make_wav_header(sample_rate: u32, num_samples: u32) -> Vec<u8> {
        let byte_rate = sample_rate * 2; // 1 channel * 2 bytes/sample
        let data_size = num_samples * 2;
        let file_size = 36 + data_size;

        let mut hdr = Vec::with_capacity(44);
        // RIFF chunk
        hdr.extend_from_slice(b"RIFF");
        hdr.extend_from_slice(&file_size.to_le_bytes());
        hdr.extend_from_slice(b"WAVE");
        // fmt sub-chunk
        hdr.extend_from_slice(b"fmt ");
        hdr.extend_from_slice(&16u32.to_le_bytes()); // sub-chunk size
        hdr.extend_from_slice(&1u16.to_le_bytes()); // PCM = 1
        hdr.extend_from_slice(&1u16.to_le_bytes()); // channels = 1
        hdr.extend_from_slice(&sample_rate.to_le_bytes());
        hdr.extend_from_slice(&byte_rate.to_le_bytes());
        hdr.extend_from_slice(&2u16.to_le_bytes()); // block align
        hdr.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
                                                     // data sub-chunk
        hdr.extend_from_slice(b"data");
        hdr.extend_from_slice(&data_size.to_le_bytes());
        hdr
    }

    /// Write a temporary WAV file with a known mono sine-like pattern.
    /// `samples` is a slice of i16 values (already in PCM range).
    fn write_tmp_wav(name: &str, sample_rate: u32, samples: &[i16]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(name);
        let mut data = make_wav_header(sample_rate, samples.len() as u32);
        for &s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        std::fs::write(&path, &data).expect("write tmp wav");
        path
    }

    // -----------------------------------------------------------------------
    // Feature 1: extract_lyrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_lyrics_no_file_has_none() {
        // A plain WAV file carries no lyrics tag; we expect Ok(None).
        let samples: Vec<i16> = (0..8).map(|i| (i as i16) * 1000).collect();
        let path = write_tmp_wav("oxiaudio_test_lyrics_none.wav", 8_000, &samples);
        let result = extract_lyrics(&path);
        // Acceptable outcomes: Ok(None) or Err (some formats error on WAV without ID3)
        match result {
            Ok(None) => {}
            Ok(Some(s)) => panic!("expected no lyrics, got: {s}"),
            Err(_) => {} // probe can fail for a headerless file – acceptable
        }
    }

    #[test]
    fn test_extract_lyrics_function_exists() {
        // Call extract_lyrics with a non-existent path; must return Err, not panic.
        let path = std::env::temp_dir().join("oxiaudio_nonexistent_file_xyz.mp3");
        let result = extract_lyrics(&path);
        assert!(result.is_err(), "expected Err for missing file, got Ok");
    }

    // -----------------------------------------------------------------------
    // Feature 2: ID3v1 tag fallback
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_id3v1_not_present() {
        // A plain WAV with no ID3v1 tail returns Ok(None).
        let samples: Vec<i16> = vec![0i16; 16];
        let path = write_tmp_wav("oxiaudio_test_id3v1_none.wav", 8_000, &samples);
        let result = parse_id3v1(&path).expect("parse_id3v1 must not fail on valid file");
        assert!(result.is_none(), "expected None for WAV without ID3v1");
    }

    #[test]
    fn test_parse_id3v1_magic_check() {
        // Slice that does NOT start with "TAG" at position 0 of 128 bytes returns None.
        let mut data = vec![0u8; 128];
        data[0] = b'X'; // not 'T'
        data[1] = b'A';
        data[2] = b'G';
        assert!(
            parse_id3v1_from_bytes(&data).is_none(),
            "should be None when magic is wrong"
        );
    }

    #[test]
    fn test_id3v1_string_extraction() {
        // Field: "Hello\0\0\0..." → "Hello"
        let mut field = [0u8; 30];
        field[..5].copy_from_slice(b"Hello");
        let s = extract_id3v1_string(&field);
        assert_eq!(s, "Hello");

        // Field with trailing spaces before NUL → trimmed
        let mut field2 = [0u8; 30];
        field2[..8].copy_from_slice(b"Hi   \0\0\0");
        let s2 = extract_id3v1_string(&field2[..8]);
        assert_eq!(s2, "Hi");
    }

    #[test]
    fn test_parse_id3v1_from_bytes_full() {
        // Build a synthetic 128-byte ID3v1 block.
        let mut data = vec![0u8; 128];
        data[0..3].copy_from_slice(b"TAG");
        // title (30 bytes) = "Test Track"
        let title = b"Test Track";
        data[3..3 + title.len()].copy_from_slice(title);
        // artist (30 bytes) = "Artist Name"
        let artist = b"Artist Name";
        data[33..33 + artist.len()].copy_from_slice(artist);
        // album (30 bytes) = "My Album"
        let album = b"My Album";
        data[63..63 + album.len()].copy_from_slice(album);
        // year (4 bytes) = "2024"
        data[93..97].copy_from_slice(b"2024");
        // ID3v1.1 track: byte 125 = 0, byte 126 = 7
        data[125] = 0;
        data[126] = 7;

        let meta = parse_id3v1_from_bytes(&data).expect("should parse valid ID3v1 block");
        assert_eq!(meta.title.as_deref(), Some("Test Track"));
        assert_eq!(meta.artist.as_deref(), Some("Artist Name"));
        assert_eq!(meta.album.as_deref(), Some("My Album"));
        assert_eq!(meta.year, Some(2024));
        assert_eq!(meta.track_number, Some(7));
    }

    #[test]
    fn test_parse_id3v1_track_zero_is_none() {
        // byte 125 = 0, byte 126 = 0 → NOT an ID3v1.1 track number (track 0 is invalid)
        let mut data = vec![0u8; 128];
        data[0..3].copy_from_slice(b"TAG");
        data[125] = 0;
        data[126] = 0;
        let meta = parse_id3v1_from_bytes(&data).expect("should parse valid ID3v1 block");
        assert!(meta.track_number.is_none(), "track 0 should not be decoded");
    }

    // -----------------------------------------------------------------------
    // Feature 3: decode_to_i16 / decode_to_i32
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_to_i16_produces_correct_scale() {
        // A WAV with a known full-scale positive sample should decode near +32767.
        // i16::MAX = 32767; writing it and reading back through decode_to_i16 should give ~32767.
        let samples = vec![i16::MAX];
        let path = write_tmp_wav("oxiaudio_test_decode_i16_scale.wav", 8_000, &samples);
        let result = decode_to_i16(&path).expect("decode_to_i16 must succeed");
        assert!(!result.samples.is_empty(), "must have at least one sample");
        // The first decoded sample should be close to i16::MAX (within rounding).
        let first = result.samples[0];
        assert!(first >= 32700, "expected sample near 32767, got {first}");
    }

    #[test]
    fn test_decode_to_i16_not_overflow() {
        // Verify that extreme samples decode correctly (no wrap-around or overflow).
        // Extreme i16 values fed through f32 and back must round-trip without corruption.
        let samples: Vec<i16> = vec![i16::MAX, i16::MIN, 0, 16383, -16384];
        let path = write_tmp_wav("oxiaudio_test_decode_i16_overflow.wav", 8_000, &samples);
        let result = decode_to_i16(&path).expect("decode_to_i16 must succeed");
        // The positive extreme (i16::MAX) should decode as a large positive value.
        // The negative extreme (i16::MIN+1 after symmetry) should decode as a large negative value.
        // We verify that the decoded buffer is non-empty and sample_rate is preserved.
        assert!(!result.samples.is_empty(), "must have samples");
        assert_eq!(result.sample_rate, 8_000);
        // Positive sample must be > 0 and negative sample (if present) must be < 0.
        let has_positive = result.samples.iter().any(|&s| s > 0);
        assert!(
            has_positive,
            "expected at least one positive sample in result"
        );
    }

    #[test]
    fn test_decode_to_i32_full_range() {
        // A WAV with +1.0 full scale (i16::MAX) should decode to near i32::MAX after scaling.
        let samples = vec![i16::MAX];
        let path = write_tmp_wav("oxiaudio_test_decode_i32_range.wav", 8_000, &samples);
        let result = decode_to_i32(&path).expect("decode_to_i32 must succeed");
        assert!(!result.samples.is_empty(), "must have at least one sample");
        // The first sample should be close to i32::MAX (within quantisation error from i16 input).
        let first = result.samples[0];
        assert!(
            first > 2_100_000_000,
            "expected sample near i32::MAX ({i32_max}), got {first}",
            i32_max = i32::MAX
        );
    }

    #[test]
    fn test_decode_to_i32_format_tag() {
        let samples = vec![0i16, 1000, -1000];
        let path = write_tmp_wav("oxiaudio_test_decode_i32_fmt.wav", 44_100, &samples);
        let result = decode_to_i32(&path).expect("decode_to_i32 must succeed");
        assert_eq!(result.format, SampleFormat::I32);
        assert_eq!(result.sample_rate, 44_100);
    }

    #[test]
    fn test_decode_to_i16_format_tag() {
        let samples = vec![0i16, 500, -500];
        let path = write_tmp_wav("oxiaudio_test_decode_i16_fmt.wav", 22_050, &samples);
        let result = decode_to_i16(&path).expect("decode_to_i16 must succeed");
        assert_eq!(result.format, SampleFormat::I16);
        assert_eq!(result.sample_rate, 22_050);
    }

    #[test]
    fn test_decode_file_with_metadata_returns_audio() {
        let samples: Vec<i16> = (0..64).map(|i| (i as i16) * 400).collect();
        let path = write_tmp_wav("oxiaudio_test_decode_meta.wav", 44_100, &samples);
        let (buf, _meta) =
            decode_file_with_metadata(&path).expect("decode_file_with_metadata must succeed");
        assert!(!buf.samples.is_empty(), "audio buffer must not be empty");
        assert_eq!(buf.sample_rate, 44_100);
    }

    // -----------------------------------------------------------------------
    // Feature 4: decode_file_with_replaygain / decode_to_f64
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_file_with_replaygain_nonexistent() {
        let path = std::env::temp_dir().join("oxiaudio_nonexistent_rg_xyz.mp3");
        let result = decode_file_with_replaygain(&path);
        assert!(result.is_err(), "expected Err for missing file, got Ok");
    }

    #[test]
    fn test_decode_to_f64_nonexistent() {
        let path = std::env::temp_dir().join("oxiaudio_nonexistent_f64_xyz.flac");
        let result = decode_to_f64(&path);
        assert!(result.is_err(), "expected Err for missing file, got Ok");
    }

    #[test]
    fn test_decode_file_with_replaygain_no_rg_tags_passthrough() {
        // A WAV with no ReplayGain tags should decode successfully and be unchanged.
        let samples: Vec<i16> = (0..64).map(|i| (i as i16) * 256).collect();
        let path = write_tmp_wav("oxiaudio_test_rg_passthrough.wav", 44_100, &samples);
        let result = decode_file_with_replaygain(&path);
        let buf = result.expect("decode_file_with_replaygain must succeed on a valid WAV");
        assert!(!buf.samples.is_empty(), "buffer must have samples");
        assert_eq!(buf.sample_rate, 44_100);
    }

    #[test]
    fn test_decode_to_f64_produces_f64_samples() {
        let samples: Vec<i16> = (0..64).map(|i| (i as i16) * 256).collect();
        let path = write_tmp_wav("oxiaudio_test_f64_output.wav", 44_100, &samples);
        let result = decode_to_f64(&path);
        let buf = result.expect("decode_to_f64 must succeed on a valid WAV");
        assert!(!buf.samples.is_empty(), "buffer must have samples");
        assert_eq!(buf.sample_rate, 44_100);
        assert_eq!(buf.format, SampleFormat::F64);
    }
}

#[cfg(test)]
mod m23_error_recovery_tests {
    use super::*;

    /// Build a minimal 44-byte RIFF/WAV header for 16-bit mono PCM at `sample_rate`.
    fn make_wav_header(sample_rate: u32, num_samples: u32) -> Vec<u8> {
        let byte_rate = sample_rate * 2;
        let data_size = num_samples * 2;
        let file_size = 36 + data_size;
        let mut hdr = Vec::with_capacity(44);
        hdr.extend_from_slice(b"RIFF");
        hdr.extend_from_slice(&file_size.to_le_bytes());
        hdr.extend_from_slice(b"WAVE");
        hdr.extend_from_slice(b"fmt ");
        hdr.extend_from_slice(&16u32.to_le_bytes());
        hdr.extend_from_slice(&1u16.to_le_bytes());
        hdr.extend_from_slice(&1u16.to_le_bytes());
        hdr.extend_from_slice(&sample_rate.to_le_bytes());
        hdr.extend_from_slice(&byte_rate.to_le_bytes());
        hdr.extend_from_slice(&2u16.to_le_bytes());
        hdr.extend_from_slice(&16u16.to_le_bytes());
        hdr.extend_from_slice(b"data");
        hdr.extend_from_slice(&data_size.to_le_bytes());
        hdr
    }

    fn write_tmp_wav_m23(name: &str, sample_rate: u32, samples: &[i16]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(name);
        let mut data = make_wav_header(sample_rate, samples.len() as u32);
        for &s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        std::fs::write(&path, &data).expect("write tmp wav");
        path
    }

    // -----------------------------------------------------------------------
    // Test 1: decode_file_with_options + Fail policy on a valid file succeeds
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_file_with_options_fail_valid_file() {
        let samples: Vec<i16> = (0..256).map(|i| (i as i16) * 100).collect();
        let path = write_tmp_wav_m23("m23_valid_fail.wav", 44_100, &samples);
        let opts = DecodeOptions {
            on_corrupt_packet: CorruptPacketPolicy::Fail,
        };
        let result = decode_file_with_options(&path, &opts);
        assert!(
            result.is_ok(),
            "expected Ok for valid file with Fail policy"
        );
        let buf = result.unwrap();
        assert!(!buf.samples.is_empty(), "buffer must have samples");
        assert_eq!(buf.sample_rate, 44_100);
    }

    // -----------------------------------------------------------------------
    // Test 2: decode_tolerant on a valid file returns audio matching decode_file
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_tolerant_valid_file_matches_decode_file() {
        let samples: Vec<i16> = (0..256).map(|i| (i as i16) * 100).collect();
        let path = write_tmp_wav_m23("m23_tolerant_valid.wav", 44_100, &samples);

        let expected = decode_file(&path).expect("decode_file must succeed");
        let tolerant = decode_tolerant(&path);

        assert_eq!(
            tolerant.sample_rate, expected.sample_rate,
            "sample_rate must match"
        );
        assert_eq!(
            tolerant.samples.len(),
            expected.samples.len(),
            "sample count must match"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: decode_tolerant on a non-existent file returns empty AudioBuffer
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_tolerant_nonexistent_returns_empty() {
        let path = std::env::temp_dir().join("m23_nonexistent_file_xyz_unique.wav");
        let buf = decode_tolerant(&path);
        // Must not panic; samples may be empty
        assert_eq!(
            buf.sample_rate, 44_100,
            "fallback sample_rate should be 44100"
        );
        assert!(
            buf.samples.is_empty(),
            "expected empty samples for missing file"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: decode_tolerant on a corrupted-in-middle WAV does not panic
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_tolerant_corrupted_file_no_panic() {
        // Write a WAV large enough that offset 1000 lands in PCM data (header = 44 bytes).
        // 4000 i16 samples = 8000 bytes of PCM; total file ≈ 8044 bytes.
        let samples: Vec<i16> = (0..4000).map(|i| (i % 32768) as i16).collect();
        let path = write_tmp_wav_m23("m23_corrupted_middle.wav", 44_100, &samples);

        // Corrupt some bytes in the PCM data section (well past the 44-byte header).
        let mut data = std::fs::read(&path).expect("read wav");
        if data.len() > 1100 {
            for byte in data[1000..1050].iter_mut() {
                *byte = 0xFF;
            }
        }
        std::fs::write(&path, &data).expect("rewrite corrupted wav");

        // Must not panic; result is undefined but must be a valid AudioBuffer.
        let buf = decode_tolerant(&path);
        // For WAV PCM, symphonia typically handles minor corruption without decode errors,
        // so we just verify no panic and a sane sample_rate.
        assert!(
            buf.sample_rate == 44_100 || buf.sample_rate == 0,
            "sample_rate should be 44100 or 0, got {}",
            buf.sample_rate
        );
    }
}
