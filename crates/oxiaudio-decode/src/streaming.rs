//! Block-by-block streaming decoder backed by symphonia.
//!
//! [`StreamingDecoder`] wraps the symphonia format reader and codec to yield
//! interleaved `f32` chunks from a FIFO. It also exposes metadata, format info,
//! seek, skip, and remaining-frame queries.

use std::collections::VecDeque;

use oxiaudio_core::{
    AudioBuffer, AudioFormat, AudioMetadata, AudioSource, ChannelLayout, OxiAudioError,
    SampleFormat,
};
use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    errors::Error as SymphoniaError,
    formats::probe::Hint,
    formats::{FormatOptions, SeekMode, SeekTo},
    io::{MediaSource, MediaSourceStream},
    meta::{MetadataOptions, StandardTag},
    units::Timestamp,
};

use crate::select_audio_track;

/// Streaming decoder that yields audio in fixed-size chunks.
///
/// Wraps symphonia's format reader and audio decoder, maintaining an internal FIFO
/// (`fifo`) of interleaved `f32` samples. Each `Iterator::next` call fills the FIFO
/// from the underlying packet stream until at least `block_size * n_channels` samples
/// are available, then drains one chunk of exactly that many samples (or fewer at EOF).
///
/// Implements both [`Iterator`] and [`AudioSource`] for flexible use in pull-based pipelines.
pub struct StreamingDecoder {
    format: Box<dyn symphonia::core::formats::FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::audio::AudioDecoder>,
    track_id: u32,
    sample_rate: u32,
    channels: ChannelLayout,
    n_channels: usize,
    block_size: usize,
    /// FIFO of interleaved f32 samples. VecDeque avoids the O(N) element-shift cost
    /// that Vec::drain(..n) incurs when consuming samples from the front.
    fifo: VecDeque<f32>,
    exhausted: bool,
    /// Total number of frames in the track, if known from codec parameters.
    total_frames: Option<u64>,
    /// Frames decoded so far (approximate, used for `remaining_frames`).
    frames_decoded: u64,
    /// Cached metadata (extracted from the container when `new` is called).
    cached_metadata: AudioMetadata,
    /// Cached audio format descriptor.
    cached_format: AudioFormat,
    /// Staging buffer reused across packet decode calls to avoid per-packet heap allocation.
    /// `copy_to_vec_interleaved` resizes this in-place each call so no extra alloc occurs.
    packet_samples: Vec<f32>,
}

/// Drain exactly `n` samples from a `VecDeque<f32>` into a new `Vec<f32>`.
///
/// Uses `as_slices()` to obtain the two contiguous segments of the deque and copies
/// them via `extend_from_slice`, avoiding the per-element index overhead of
/// `drain(..n).collect()` when the deque wraps around its ring buffer.
/// After the copy the drained region is removed with `drain(..n)`.
fn drain_fifo_to_vec(fifo: &mut std::collections::VecDeque<f32>, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let (head, tail) = fifo.as_slices();
    let head_take = head.len().min(n);
    out.extend_from_slice(&head[..head_take]);
    if out.len() < n {
        let tail_take = (n - out.len()).min(tail.len());
        out.extend_from_slice(&tail[..tail_take]);
    }
    fifo.drain(..n);
    out
}

impl StreamingDecoder {
    /// Probe `src` and prepare the decoder for block-by-block streaming.
    ///
    /// `block_size` is the number of *frames* per chunk (each frame = `n_channels` samples).
    ///
    /// Internally reuses a staging `Vec<f32>` across decoded packets to reduce allocations.
    /// Decoded samples are accumulated in a FIFO and drained in fixed-size chunks on each
    /// `next()` call, so peak memory is bounded to roughly `block_size * n_channels * 4` bytes.
    pub fn new(
        src: impl std::io::Read + std::io::Seek + Send + Sync + 'static,
        block_size: usize,
    ) -> Result<Self, OxiAudioError> {
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
        let total_frames = track.num_frames;
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

        let channels = ChannelLayout::from(n_channels as u16);

        // Capture bitrate before audio_params is consumed by make_audio_decoder.
        let bitrate_kbps = compute_bitrate_kbps_stream(audio_params, sample_rate, n_channels);

        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(audio_params, &dec_opts)
            .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

        // Extract metadata eagerly so it's available without decoding any frames.
        let cached_metadata = {
            let mut m = format
                .metadata()
                .current()
                .map(extract_metadata_rev)
                .unwrap_or_default();
            m.duration_secs = total_frames.map(|n| n as f64 / f64::from(sample_rate));
            m.bitrate_kbps = bitrate_kbps;
            m
        };

        Ok(Self {
            format,
            decoder,
            track_id,
            sample_rate,
            channels,
            n_channels,
            block_size,
            fifo: VecDeque::new(),
            exhausted: false,
            total_frames,
            frames_decoded: 0,
            cached_metadata,
            cached_format: AudioFormat {
                sample_rate,
                channels,
                format: SampleFormat::F32,
            },
            packet_samples: Vec::new(),
        })
    }

    /// Open a streaming decoder from a file path.
    ///
    /// Uses the file extension as a symphonia probe hint for faster format detection,
    /// then falls back to content-based magic-byte probing.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Io`] if the file cannot be opened, or
    /// [`OxiAudioError::Decode`] if no audio track or valid codec is found.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, OxiAudioError> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }
        let mss = MediaSourceStream::new(Box::new(MediaSourceWrapper(file)), Default::default());
        let dec_opts = AudioDecoderOptions::default();

        let mut format = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

        let track = crate::select_audio_track(format.as_ref())
            .ok_or_else(|| OxiAudioError::Decode("no audio track with valid codec".into()))?;

        let track_id = track.id;
        let total_frames = track.num_frames;
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
        let channels = ChannelLayout::from(n_channels as u16);

        // Capture bitrate before audio_params is consumed by make_audio_decoder.
        let bitrate_kbps = compute_bitrate_kbps_stream(audio_params, sample_rate, n_channels);

        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(audio_params, &dec_opts)
            .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

        let cached_metadata = {
            let mut m = format
                .metadata()
                .current()
                .map(extract_metadata_rev)
                .unwrap_or_default();
            m.duration_secs = total_frames.map(|n| n as f64 / f64::from(sample_rate));
            m.bitrate_kbps = bitrate_kbps;
            m
        };

        // Default block_size for open(); callers that need a specific block size use new().
        let block_size = 4096;

        Ok(Self {
            format,
            decoder,
            track_id,
            sample_rate,
            channels,
            n_channels,
            block_size,
            fifo: VecDeque::new(),
            exhausted: false,
            total_frames,
            frames_decoded: 0,
            cached_metadata,
            cached_format: AudioFormat {
                sample_rate,
                channels,
                format: SampleFormat::F32,
            },
            packet_samples: Vec::new(),
        })
    }

    /// Open a streaming decoder from a file path, with a custom block size.
    ///
    /// Like [`Self::open`] but also sets the number of *frames* per chunk returned
    /// by [`Self::next_block`] and [`Iterator::next`].
    ///
    /// Uses the file extension as a symphonia probe hint for faster format detection.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Io`] if the file cannot be opened, or
    /// [`OxiAudioError::Decode`] if no audio track or valid codec is found.
    pub fn open_with_block_size<P: AsRef<std::path::Path>>(
        path: P,
        block_size: usize,
    ) -> Result<Self, OxiAudioError> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }
        let mss = MediaSourceStream::new(Box::new(MediaSourceWrapper(file)), Default::default());
        let dec_opts = AudioDecoderOptions::default();

        let mut format = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

        let track = crate::select_audio_track(format.as_ref())
            .ok_or_else(|| OxiAudioError::Decode("no audio track with valid codec".into()))?;

        let track_id = track.id;
        let total_frames = track.num_frames;
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
        let channels = ChannelLayout::from(n_channels as u16);

        let bitrate_kbps = compute_bitrate_kbps_stream(audio_params, sample_rate, n_channels);

        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(audio_params, &dec_opts)
            .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

        let cached_metadata = {
            let mut m = format
                .metadata()
                .current()
                .map(extract_metadata_rev)
                .unwrap_or_default();
            m.duration_secs = total_frames.map(|n| n as f64 / f64::from(sample_rate));
            m.bitrate_kbps = bitrate_kbps;
            m
        };

        Ok(Self {
            format,
            decoder,
            track_id,
            sample_rate,
            channels,
            n_channels,
            block_size,
            fifo: VecDeque::new(),
            exhausted: false,
            total_frames,
            frames_decoded: 0,
            cached_metadata,
            cached_format: AudioFormat {
                sample_rate,
                channels,
                format: SampleFormat::F32,
            },
            packet_samples: Vec::new(),
        })
    }

    /// Returns a reference to the audio format detected from the stream.
    ///
    /// Always `Some` once the decoder has been successfully constructed.
    pub fn format(&self) -> Option<&AudioFormat> {
        Some(&self.cached_format)
    }

    /// Decode the next chunk of up to `max_frames` audio frames.
    ///
    /// Returns `None` when the stream is exhausted. Frames are decoded from the
    /// internal FIFO (refilled as needed from the packet stream).
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Decode`] on unrecoverable codec errors.
    pub fn decode_next(
        &mut self,
        max_frames: usize,
    ) -> Result<Option<AudioBuffer<f32>>, OxiAudioError> {
        let chunk_samples = max_frames * self.n_channels;
        // Refill the FIFO until we have at least chunk_samples or the stream is done.
        self.refill(chunk_samples);

        if self.fifo.is_empty() && self.exhausted {
            return Ok(None);
        }

        let n = chunk_samples.min(self.fifo.len());
        // Use as_slices() to copy from the VecDeque's contiguous head/tail segments via
        // extend_from_slice instead of iterating per-element (avoids index overhead).
        let drained = drain_fifo_to_vec(&mut self.fifo, n);
        let frames = n.checked_div(self.n_channels).unwrap_or(0);
        self.frames_decoded += frames as u64;

        Ok(Some(AudioBuffer {
            samples: drained,
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: SampleFormat::F32,
        }))
    }

    /// Seek the underlying format reader to `frame_offset` and reset the decoder state.
    ///
    /// After a seek the FIFO is cleared so the next [`Iterator::next`] call starts
    /// fresh from the seeked position.
    pub fn seek(&mut self, frame_offset: u64) -> Result<(), OxiAudioError> {
        let ts = Timestamp::try_from(frame_offset)
            .map_err(|_| OxiAudioError::Decode("frame_offset overflows Timestamp (i64)".into()))?;
        self.format
            .seek(
                SeekMode::Accurate,
                SeekTo::Timestamp {
                    ts,
                    track_id: self.track_id,
                },
            )
            .map_err(|e| OxiAudioError::Decode(e.to_string()))?;
        self.decoder.reset();
        self.fifo.clear();
        self.exhausted = false;
        self.frames_decoded = frame_offset;
        Ok(())
    }

    /// Return format info (sample rate, channel layout, sample format).
    ///
    /// Always available once construction has succeeded.
    pub fn format_info(&self) -> Option<AudioFormat> {
        Some(AudioFormat {
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: SampleFormat::F32,
        })
    }

    /// Return metadata extracted from the container (title, artist, album, etc.).
    ///
    /// May be all-`None` (`Default`) if the container carries no tag data.
    pub fn metadata(&self) -> &AudioMetadata {
        &self.cached_metadata
    }

    /// Return metadata extracted from the container as an owned clone.
    ///
    /// Convenience wrapper for callers that need an owned `AudioMetadata` value.
    pub fn metadata_owned(&self) -> AudioMetadata {
        self.cached_metadata.clone()
    }

    /// Skip forward by `frames` audio frames.
    ///
    /// Returns the actual number of frames skipped (may be less if the stream ended).
    ///
    /// Implementation: first drains as many whole frames as possible from the FIFO,
    /// then decodes-and-discards further packets until the requested count is reached
    /// or the stream is exhausted.
    pub fn skip_frames(&mut self, frames: u64) -> Result<u64, OxiAudioError> {
        let mut remaining = frames;
        let mut skipped: u64 = 0;

        // Drain from FIFO first.
        if !self.fifo.is_empty() {
            let fifo_frames = (self.fifo.len() / self.n_channels) as u64;
            let to_drop = remaining.min(fifo_frames);
            let samples_to_drop = (to_drop as usize) * self.n_channels;
            self.fifo.drain(..samples_to_drop);
            remaining -= to_drop;
            skipped += to_drop;
            self.frames_decoded += to_drop;
        }

        if remaining == 0 {
            return Ok(skipped);
        }

        // Try to seek forward; if that fails, fall back to decode-and-discard.
        let target_frame = self.frames_decoded + remaining;
        if self.seek(target_frame).is_err() {
            // Seek unsupported — decode and discard.
            // Reuse self.packet_samples (hot-path allocation optimization: avoids a fresh
            // heap allocation per packet; copy_to_vec_interleaved resizes in-place).
            while remaining > 0 && !self.exhausted {
                let packet = match self.format.next_packet() {
                    Ok(Some(p)) => p,
                    Ok(None) => {
                        self.exhausted = true;
                        break;
                    }
                    Err(SymphoniaError::ResetRequired) => {
                        self.decoder.reset();
                        continue;
                    }
                    Err(SymphoniaError::IoError(_)) => {
                        self.exhausted = true;
                        break;
                    }
                    Err(e) => return Err(OxiAudioError::Decode(e.to_string())),
                };
                if packet.track_id != self.track_id {
                    continue;
                }
                match self.decoder.decode(&packet) {
                    Ok(decoded) => {
                        decoded.copy_to_vec_interleaved::<f32>(&mut self.packet_samples);
                        let pkt_frames = self.packet_samples.len() as u64 / self.n_channels as u64;
                        if pkt_frames <= remaining {
                            remaining -= pkt_frames;
                            skipped += pkt_frames;
                            self.frames_decoded += pkt_frames;
                        } else {
                            // Keep the tail in the FIFO.
                            let consumed = (remaining as usize) * self.n_channels;
                            self.fifo.extend(&self.packet_samples[consumed..]);
                            skipped += remaining;
                            self.frames_decoded += remaining;
                            remaining = 0;
                        }
                    }
                    Err(SymphoniaError::IoError(_)) => {
                        self.exhausted = true;
                        break;
                    }
                    Err(SymphoniaError::DecodeError(_)) => continue,
                    Err(e) => return Err(OxiAudioError::Decode(e.to_string())),
                }
            }
        } else {
            // Seek succeeded — frames_decoded updated by seek().
            // Best-effort count: we asked to skip `frames` and the seek succeeded.
            skipped = frames;
        }

        Ok(skipped)
    }

    /// Return the estimated number of remaining frames, or `None` if unknown.
    pub fn remaining_frames(&self) -> Option<u64> {
        let total = self.total_frames?;
        Some(total.saturating_sub(self.frames_decoded))
    }

    /// Seek to `time_seconds` from the start of the stream.
    ///
    /// Converts the time offset to a frame offset using the known sample rate,
    /// then delegates to [`Self::seek`].
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying format does not support seeking, or if
    /// `time_seconds` is out of range.
    pub fn seek_to_time(&mut self, time_seconds: f64) -> Result<(), OxiAudioError> {
        if time_seconds < 0.0 {
            return Err(OxiAudioError::Decode(
                "seek_to_time: time_seconds must be non-negative".into(),
            ));
        }
        let frame_offset = (time_seconds * f64::from(self.sample_rate)).round() as u64;
        self.seek(frame_offset)
    }

    /// Seek to the given time position in seconds.
    ///
    /// Converts `seconds` to a frame position and calls the underlying symphonia
    /// seek API. After this call, the next [`Iterator::next`] call returns audio near
    /// `seconds`.
    ///
    /// This is a convenience alias for [`Self::seek_to_time`] using the standard name
    /// expected by callers that follow the `time_seek` naming convention.
    ///
    /// # Errors
    ///
    /// Returns `Err` if seeking fails (format doesn't support seeking, negative
    /// offset, or out-of-range position).
    #[must_use = "discarding errors ignores seek failure"]
    pub fn time_seek(&mut self, seconds: f64) -> Result<(), OxiAudioError> {
        self.seek_to_time(seconds)
    }

    /// Internal: refill the FIFO from the packet stream until we have at least
    /// `min_samples` samples or the stream is exhausted.
    ///
    /// Uses `self.packet_samples` as a reusable staging buffer (hot-path allocation
    /// optimization: `copy_to_vec_interleaved` resizes in-place; no per-packet alloc).
    fn refill(&mut self, min_samples: usize) {
        while self.fifo.len() < min_samples && !self.exhausted {
            let packet = match self.format.next_packet() {
                Ok(Some(p)) => p,
                Ok(None) => {
                    self.exhausted = true;
                    break;
                }
                Err(SymphoniaError::ResetRequired) => {
                    self.decoder.reset();
                    continue;
                }
                Err(SymphoniaError::IoError(_)) => {
                    self.exhausted = true;
                    break;
                }
                Err(_) => {
                    self.exhausted = true;
                    break;
                }
            };
            if packet.track_id != self.track_id {
                continue;
            }
            match self.decoder.decode(&packet) {
                Ok(decoded) => {
                    decoded.copy_to_vec_interleaved::<f32>(&mut self.packet_samples);
                    self.fifo.extend(&self.packet_samples);
                }
                Err(SymphoniaError::IoError(_)) => {
                    self.exhausted = true;
                    break;
                }
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(_) => {
                    self.exhausted = true;
                    break;
                }
            }
        }
    }

    /// Decode the next block of audio.
    ///
    /// Returns `None` when the stream is exhausted.
    pub fn next_block(&mut self) -> Result<Option<AudioBuffer<f32>>, OxiAudioError> {
        let chunk_samples = self.block_size * self.n_channels;
        self.refill(chunk_samples);

        if self.fifo.is_empty() && self.exhausted {
            return Ok(None);
        }

        let n = chunk_samples.min(self.fifo.len());
        // Use as_slices() to copy from the VecDeque's contiguous head/tail segments via
        // extend_from_slice instead of iterating per-element (avoids index overhead).
        let drained = drain_fifo_to_vec(&mut self.fifo, n);
        let frames = n / self.n_channels;
        self.frames_decoded += frames as u64;

        Ok(Some(AudioBuffer {
            samples: drained,
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: SampleFormat::F32,
        }))
    }
}

impl Iterator for StreamingDecoder {
    type Item = Result<AudioBuffer<f32>, OxiAudioError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_block() {
            Ok(Some(buf)) => Some(Ok(buf)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

impl AudioSource for StreamingDecoder {
    fn read_chunk(&mut self) -> Result<Option<AudioBuffer<f32>>, OxiAudioError> {
        self.next_block()
    }
}

/// Compute an estimated bitrate in kbps from audio codec parameters.
///
/// Uses `bits_per_sample * sample_rate * num_channels / 1000` when fields are available.
/// Returns `None` when any required parameter is absent.
fn compute_bitrate_kbps_stream(
    audio_params: &symphonia::core::codecs::audio::AudioCodecParameters,
    sample_rate: u32,
    n_channels: usize,
) -> Option<u32> {
    let bps = audio_params
        .bits_per_sample
        .or(audio_params.bits_per_coded_sample)?;
    let bitrate = (bps as u64)
        .saturating_mul(sample_rate as u64)
        .saturating_mul(n_channels as u64)
        / 1000;
    Some(bitrate as u32)
}

/// Extract metadata tags from a symphonia [`MetadataRevision`].
fn extract_metadata_rev(rev: &symphonia::core::meta::MetadataRevision) -> AudioMetadata {
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
        duration_secs: None, // filled in by the caller
        bitrate_kbps: None,
        genre,
        composer,
        year,
        track_number,
        disc_number,
        comment,
        album_art: None,
    }
}

/// A fluent builder for [`StreamingDecoder`] with configurable decode options.
///
/// Provides an ergonomic way to open a [`StreamingDecoder`] with non-default
/// settings such as a custom block size, track selection, or tolerant (skip-corrupt) mode.
///
/// # Example
/// ```no_run
/// use oxiaudio_decode::StreamingDecoderBuilder;
/// let decoder = StreamingDecoderBuilder::new("input.flac")
///     .block_size(8192)
///     .build()
///     .expect("open failed");
/// ```
pub struct StreamingDecoderBuilder {
    path: std::path::PathBuf,
    block_size: usize,
    /// If true, skip corrupted packets instead of returning an error.
    ///
    /// Note: symphonia's internal refill loop already skips decode errors
    /// (`DecodeError`) by continuing to the next packet, so this flag is
    /// stored for documentation/future use and does not change current behaviour.
    skip_corrupt: bool,
    /// Optional index into the list of audio tracks. When `None` (default), the first
    /// audio track is selected. When `Some(i)`, the i-th audio track (0-based) is used.
    track_index: Option<usize>,
}

impl StreamingDecoderBuilder {
    /// Create a builder targeting the given file path.
    #[must_use]
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            path: path.into(),
            block_size: 4096,
            skip_corrupt: false,
            track_index: None,
        }
    }

    /// Set the decode block size in frames (default: 4096).
    ///
    /// Each call to [`StreamingDecoder::next_block`] (or [`Iterator::next`]) returns
    /// up to this many frames of audio. Smaller values reduce latency; larger values
    /// amortise per-call overhead.
    #[must_use]
    pub fn block_size(mut self, frames: usize) -> Self {
        self.block_size = frames;
        self
    }

    /// Enable tolerant mode: skip corrupted frames instead of returning an error.
    ///
    /// When `true`, the decoder discards packets that fail to decode (e.g. due to
    /// bit-errors) and continues to the next packet rather than propagating an error.
    #[must_use]
    pub fn skip_corrupt(mut self, skip: bool) -> Self {
        self.skip_corrupt = skip;
        self
    }

    /// Select which audio track to decode by 0-based index (default: 0 = first audio track).
    ///
    /// Useful for multi-track containers such as MKV or MP4 that embed more than one
    /// audio stream. The index counts only audio tracks (tracks whose codec parameters
    /// resolve to audio codec parameters), not video or data tracks.
    ///
    /// If the requested index exceeds the number of audio tracks in the container,
    /// [`build`](Self::build) returns [`OxiAudioError::UnsupportedFormat`].
    #[must_use]
    pub fn track_index(mut self, idx: usize) -> Self {
        self.track_index = Some(idx);
        self
    }

    /// Build the [`StreamingDecoder`], opening the file at the configured path.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Io`] if the file cannot be opened,
    /// [`OxiAudioError::Decode`] if no audio track or valid codec is found, or
    /// [`OxiAudioError::UnsupportedFormat`] if a `track_index` was specified but the
    /// container has fewer audio tracks than the requested index.
    #[must_use = "discarding the Result ignores open errors"]
    pub fn build(self) -> Result<StreamingDecoder, oxiaudio_core::OxiAudioError> {
        build_streaming_decoder_with_index(&self.path, self.block_size, self.track_index)
    }
}

/// Internal helper that opens a [`StreamingDecoder`] from a path, optionally selecting an
/// audio track by index instead of always using the first one.
fn build_streaming_decoder_with_index(
    path: &std::path::Path,
    block_size: usize,
    track_index: Option<usize>,
) -> Result<StreamingDecoder, oxiaudio_core::OxiAudioError> {
    use oxiaudio_core::OxiAudioError;

    let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mss = MediaSourceStream::new(Box::new(MediaSourceWrapper(file)), Default::default());
    let dec_opts = AudioDecoderOptions::default();

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

    // Select the audio track: either by index or fall back to the first audio track.
    let track = match track_index {
        None => {
            // Default: first audio track (existing behaviour).
            crate::select_audio_track(format.as_ref())
                .ok_or_else(|| OxiAudioError::Decode("no audio track with valid codec".into()))?
        }
        Some(idx) => {
            // Collect all audio tracks in order, then index into them.
            let audio_tracks: Vec<_> = format
                .tracks()
                .iter()
                .filter(|t| t.codec_params.as_ref().and_then(|cp| cp.audio()).is_some())
                .collect();
            audio_tracks.get(idx).copied().ok_or_else(|| {
                OxiAudioError::UnsupportedFormat(format!(
                    "track index {idx} not found (container has {} audio track(s))",
                    audio_tracks.len()
                ))
            })?
        }
    };

    let track_id = track.id;
    let total_frames = track.num_frames;
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
    let channels = ChannelLayout::from(n_channels as u16);

    let bitrate_kbps = compute_bitrate_kbps_stream(audio_params, sample_rate, n_channels);

    let decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &dec_opts)
        .map_err(|e| OxiAudioError::Decode(e.to_string()))?;

    let cached_metadata = {
        let mut m = format
            .metadata()
            .current()
            .map(extract_metadata_rev)
            .unwrap_or_default();
        m.duration_secs = total_frames.map(|n| n as f64 / f64::from(sample_rate));
        m.bitrate_kbps = bitrate_kbps;
        m
    };

    Ok(StreamingDecoder {
        format,
        decoder,
        track_id,
        sample_rate,
        channels,
        n_channels,
        block_size,
        fifo: std::collections::VecDeque::new(),
        exhausted: false,
        total_frames,
        frames_decoded: 0,
        cached_metadata,
        cached_format: AudioFormat {
            sample_rate,
            channels,
            format: SampleFormat::F32,
        },
        packet_samples: Vec::new(),
    })
}

/// Thin wrapper that adds the `MediaSource` trait implementation to any `Read + Seek + Send + Sync`.
struct MediaSourceWrapper<R>(R);

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
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Write a mono 440 Hz sine-wave WAV and return it as raw bytes.
    ///
    /// Uses `hound` (already a dev-dep) so we don't need `oxiaudio-encode`.
    fn make_sine_wav(sample_rate: u32, duration_secs: f32) -> Vec<u8> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();
        let mut wav_bytes = Vec::new();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer =
            hound::WavWriter::new(Cursor::new(&mut wav_bytes), spec).expect("WavWriter::new");
        for &s in &samples {
            writer.write_sample(s).expect("write_sample");
        }
        writer.finalize().expect("finalize");
        wav_bytes
    }

    /// Write bytes to a uniquely-named temp file, returning the path.
    fn write_temp_file(bytes: &[u8], suffix: &str) -> std::path::PathBuf {
        use std::io::Write;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("oxiaudio_streaming_{ts}_{suffix}.wav"));
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(bytes).expect("write temp wav");
        path
    }

    // ── StreamingDecoderBuilder tests ──────────────────────────────────────────

    /// test_builder_default_block_size: build from a non-existent path, verify Err (not panic).
    #[test]
    fn test_builder_default_block_size() {
        let path = std::env::temp_dir().join("does_not_exist_oxiaudio_decode_builder.wav");
        let result = StreamingDecoderBuilder::new(path).build();
        assert!(
            result.is_err(),
            "expected Err when opening a non-existent file, got Ok"
        );
    }

    /// test_builder_configures_block_size: builder creates with block_size=8192, check no panic.
    #[test]
    fn test_builder_configures_block_size() {
        let wav = make_sine_wav(44_100, 0.1);
        let path = write_temp_file(&wav, "builder_block_size");
        let result = StreamingDecoderBuilder::new(&path).block_size(8192).build();
        let _ = std::fs::remove_file(&path);
        // The decoder should open successfully and use block_size=8192.
        assert!(
            result.is_ok(),
            "expected Ok when opening a valid WAV with block_size=8192"
        );
        let dec = result.expect("decoder");
        assert_eq!(
            dec.block_size, 8192,
            "block_size field should reflect the builder setting"
        );
    }

    /// test_builder_skip_corrupt_default: verify `skip_corrupt` defaults to false.
    #[test]
    fn test_builder_skip_corrupt_default() {
        let builder = StreamingDecoderBuilder::new("anything.wav");
        assert!(
            !builder.skip_corrupt,
            "skip_corrupt should default to false"
        );
    }

    /// M10-1: `skip_frames` returns a count equal to the number of frames requested
    /// (or fewer if EOF is reached before that).
    ///
    /// The existing API is `skip_frames(frames: u64) -> Result<u64, OxiAudioError>`.
    #[test]
    fn test_skip_frames_returns_frame_count() {
        // 0.5-second mono WAV at 44 100 Hz → 22 050 frames total.
        let wav = make_sine_wav(44_100, 0.5);
        let path = write_temp_file(&wav, "skip_frames");
        let mut dec = StreamingDecoder::open(&path).expect("open");
        let _ = std::fs::remove_file(&path);

        // Ask to skip 100 frames — should succeed since the file has 22 050.
        let skipped = dec.skip_frames(100).expect("skip_frames should not error");
        assert!(
            skipped > 0,
            "expected at least 1 frame skipped, got {skipped}"
        );
        // We asked for 100; a WAV with 22 050 frames can always deliver that.
        assert_eq!(skipped, 100, "skip_frames(100) returned {skipped} not 100");
    }

    /// M10-2: `remaining_frames` decreases as frames are consumed via `decode_next`.
    #[test]
    fn test_remaining_frames_decreases_on_consume() {
        let wav = make_sine_wav(44_100, 1.0); // ~44 100 frames
        let path = write_temp_file(&wav, "remaining_frames");
        let mut dec = StreamingDecoder::open(&path).expect("open");
        let _ = std::fs::remove_file(&path);

        let before = dec.remaining_frames();
        // Consume a few chunks.
        for _ in 0..3 {
            let _ = dec.decode_next(512).expect("decode_next");
        }
        let after = dec.remaining_frames();

        // Only meaningful when the WAV container reports a frame count (it should).
        if let (Some(b), Some(a)) = (before, after) {
            assert!(
                a < b,
                "remaining_frames should decrease after decoding: before={b}, after={a}"
            );
        }
        // If either is None the container didn't report frame count — acceptable.
    }

    /// M10-3: `time_seek` succeeds on a seekable WAV and the next `next()` call
    /// returns audio (i.e., the stream is still live after the seek).
    #[test]
    fn test_time_seek_succeeds() {
        // 2-second WAV so seeking to 0.5 s is well within the file.
        let wav = make_sine_wav(44_100, 2.0);
        let path = write_temp_file(&wav, "time_seek");
        let mut dec = StreamingDecoder::open(&path).expect("open");
        let _ = std::fs::remove_file(&path);

        // time_seek should succeed on a seekable WAV.
        dec.time_seek(0.5).expect("time_seek(0.5) should succeed");

        // The stream should still be live after the seek.
        let chunk = dec.next();
        assert!(
            chunk.is_some(),
            "expected Some(Ok(...)) after time_seek, got None"
        );
        let buf = chunk.unwrap().expect("chunk should not error after seek");
        assert!(
            !buf.samples.is_empty(),
            "decoded chunk after time_seek should not be empty"
        );
    }
}
