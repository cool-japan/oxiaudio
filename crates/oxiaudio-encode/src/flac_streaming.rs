/// True-streaming FLAC encoder.
///
/// Encodes FLAC frames incrementally by collecting pending samples into a
/// `pending` buffer and calling `flacenc::encode_fixed_size_frame` frame-by-frame,
/// accumulating encoded [`flacenc::component::Frame`] objects into an in-memory
/// [`flacenc::component::Stream`].  At `finalize()` the full [`Stream`] (with
/// correct `StreamInfo`) is serialised to the writer in one pass.
///
/// Unlike the buffering [`crate::FlacStreamEncoder`] (which concatenates chunks in a
/// `Vec<f32>` and encodes everything at finalize), this encoder processes each
/// `block_size`-frame block immediately — avoiding a full-audio-in-memory requirement
/// for the sample data.  The encoded frames *are* held in the `Stream` until
/// finalize, but their combined size is much smaller than the raw f32 samples.
use std::io::{Seek, Write};

use flacenc::bitsink::ByteSink;
use flacenc::component::{BitRepr, Stream, StreamInfo};
use flacenc::error::{Verified, Verify};
use flacenc::source::{Fill, FrameBuf};
use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError};

use crate::flac_core::{block_size_for_level, clamp_flac_bits, flac_full_scale};

// ─── FlacStreamingEncoder ─────────────────────────────────────────────────────

/// A FLAC encoder that encodes frames immediately as they arrive, without
/// buffering all audio in memory.
///
/// Internally, each full block of `block_size` PCM frames is encoded via
/// `flacenc::encode_fixed_size_frame` and added to the in-progress FLAC
/// [`Stream`].  Partial blocks (< `block_size` frames) are held in a small
/// `pending` buffer until the next [`Self::encode_chunk`] call fills them, or
/// until [`Self::finalize`] flushes them.
///
/// At finalize, the [`Stream`]'s STREAMINFO is updated with final sample counts
/// and fixed block-size flags, then the stream is serialised to the writer.
///
/// # Block size
///
/// The `block_size` defaults to `block_size_for_level(compression_level)` and
/// is set once at construction.  It must remain constant across all chunks for
/// the fixed-blocksize mode assertion in STREAMINFO to hold.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use oxiaudio_encode::FlacStreamingEncoder;
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let mut enc = FlacStreamingEncoder::new(
///     Cursor::new(Vec::new()),
///     44_100,
///     ChannelLayout::Stereo,
///     5,
/// ).unwrap();
///
/// let buf = AudioBuffer {
///     samples: vec![0.0f32; 2048],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Stereo,
///     format: SampleFormat::F32,
/// };
/// enc.encode_chunk(&buf).unwrap();
/// enc.finalize().unwrap();
/// ```
pub struct FlacStreamingEncoder<W: Write + Seek> {
    writer: W,
    channels: ChannelLayout,
    block_size: usize,
    bits: u8,
    /// FLAC config (verified once at construction; reused for every frame).
    cfg: Verified<flacenc::config::Encoder>,
    /// FLAC stream accumulating encoded Frame objects.
    stream: Stream,
    /// In-flight StreamInfo to track frame stats (mirrors stream.stream_info_mut()).
    // Note: Stream::add_frame already calls update_frame_info internally.
    /// Partial-block sample buffer (interleaved i32 PCM).
    pending_pcm: Vec<i32>,
    /// Monotonically increasing frame number.
    frame_number: usize,
    /// Total PCM *frames* (not samples) written so far.
    total_frames: usize,
}

impl<W: Write + Seek> FlacStreamingEncoder<W> {
    /// Create a new `FlacStreamingEncoder`.
    ///
    /// Writes no bytes at construction; all output is deferred until
    /// [`Self::finalize`].
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Encode`] if `flacenc` rejects the configuration
    /// (e.g. unsupported sample rate / channel count).
    pub fn new(
        writer: W,
        sample_rate: u32,
        channels: ChannelLayout,
        compression_level: u8,
    ) -> Result<Self, OxiAudioError> {
        let block_size = block_size_for_level(compression_level);
        let bits = clamp_flac_bits(16); // 16-bit default; full-quality path

        let n_ch = channels.channel_count();
        let stream = Stream::new(sample_rate as usize, n_ch, bits as usize)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

        let mut raw_cfg = flacenc::config::Encoder::default();
        raw_cfg.block_size = block_size;
        let cfg = raw_cfg
            .into_verified()
            .map_err(|(_, e)| OxiAudioError::Encode(e.to_string()))?;

        // Pre-allocate enough room for one full block (per-channel, then interleaved).
        let pending_pcm = Vec::with_capacity(block_size * n_ch);

        Ok(Self {
            writer,
            channels,
            block_size,
            bits,
            cfg,
            stream,
            pending_pcm,
            frame_number: 0,
            total_frames: 0,
        })
    }

    /// Create with a custom bit depth (16, 20, or 24).
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Encode`] if flacenc rejects the configuration.
    pub fn with_bits_per_sample(
        writer: W,
        sample_rate: u32,
        channels: ChannelLayout,
        compression_level: u8,
        bits_per_sample: u8,
    ) -> Result<Self, OxiAudioError> {
        let block_size = block_size_for_level(compression_level);
        let bits = clamp_flac_bits(bits_per_sample);

        let n_ch = channels.channel_count();
        let stream = Stream::new(sample_rate as usize, n_ch, bits as usize)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

        let mut raw_cfg = flacenc::config::Encoder::default();
        raw_cfg.block_size = block_size;
        let cfg = raw_cfg
            .into_verified()
            .map_err(|(_, e)| OxiAudioError::Encode(e.to_string()))?;

        let pending_pcm = Vec::with_capacity(block_size * n_ch);

        Ok(Self {
            writer,
            channels,
            block_size,
            bits,
            cfg,
            stream,
            pending_pcm,
            frame_number: 0,
            total_frames: 0,
        })
    }

    /// Append audio samples to the encoder.
    ///
    /// Each complete block of `block_size` frames is encoded immediately via
    /// `encode_fixed_size_frame` and stored in the internal `Stream`. Partial
    /// blocks are buffered in `self.pending_pcm`.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Encode`] on any encoding failure.
    pub fn encode_chunk(&mut self, buf: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
        let n_ch = self.channels.channel_count();
        let scale = flac_full_scale(self.bits);

        // Convert f32 samples to i32 interleaved PCM.
        let new_pcm: Vec<i32> = buf
            .samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * scale) as i32)
            .collect();

        // Append to the pending buffer.
        self.pending_pcm.extend_from_slice(&new_pcm);

        // Drain complete blocks.
        let samples_per_block = self.block_size * n_ch;
        while self.pending_pcm.len() >= samples_per_block {
            let block_samples: Vec<i32> = self.pending_pcm.drain(..samples_per_block).collect();
            self.encode_block(&block_samples, self.block_size)?;
        }

        Ok(())
    }

    /// Returns the total number of PCM *frames* (not samples) encoded so far.
    pub fn frames_encoded(&self) -> usize {
        self.total_frames
    }

    /// Encode one complete or partial block of interleaved i32 PCM samples.
    ///
    /// `block_samples` must be `actual_frames * n_ch` samples long.
    /// `actual_frames` must be ≤ `self.block_size`.
    fn encode_block(
        &mut self,
        block_samples: &[i32],
        actual_frames: usize,
    ) -> Result<(), OxiAudioError> {
        let n_ch = self.channels.channel_count();
        let stream_info: &StreamInfo = self.stream.stream_info();

        let mut frame_buf = FrameBuf::with_size(n_ch, actual_frames)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

        frame_buf
            .fill_interleaved(block_samples)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

        let frame =
            flacenc::encode_fixed_size_frame(&self.cfg, &frame_buf, self.frame_number, stream_info)
                .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

        self.frame_number += 1;
        self.total_frames += actual_frames;

        // add_frame also calls stream_info_mut().update_frame_info() internally.
        self.stream.add_frame(frame);

        Ok(())
    }

    /// Flush any pending samples, finalize the FLAC stream, and write to the writer.
    ///
    /// Sets STREAMINFO block sizes to signal fixed-blocksize mode (required by some
    /// decoders, including symphonia). Consumes the encoder.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError`] on encoding or I/O failure.
    pub fn finalize(mut self) -> Result<(), OxiAudioError> {
        let n_ch = self.channels.channel_count();

        // Flush any remaining samples (partial last block).
        if !self.pending_pcm.is_empty() {
            let remaining_frames = self.pending_pcm.len() / n_ch;
            if remaining_frames > 0 {
                let block: Vec<i32> = self.pending_pcm.drain(..).collect();
                self.encode_block(&block, remaining_frames)?;
            }
        }

        // Fix block sizes so the stream uses fixed-blocksize mode.
        // set_block_sizes(min, max) — after flush, min_block_size may differ from
        // max_block_size (last block is usually smaller). Use block_size for max
        // and the actual value for min is set by update_frame_info.
        // symphonia requires min == max for fixed-blocksize mode, so use block_size
        // for both (matches what FlacEncoder does).
        self.stream
            .stream_info_mut()
            .set_block_sizes(self.block_size, self.block_size)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

        // Serialise the entire stream.
        let mut sink = ByteSink::with_capacity(self.stream.count_bits());
        self.stream
            .write(&mut sink)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

        self.writer
            .write_all(sink.as_slice())
            .map_err(OxiAudioError::Io)?;

        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    use super::FlacStreamingEncoder;

    fn sine_buf_stereo(sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
        let n_frames = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..n_frames)
            .flat_map(|i| {
                let s = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
                    * 0.4;
                [s, -s]
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_flac_streaming_encoder_produces_valid_flac() {
        let buf = sine_buf_stereo(44_100, 2.0);
        let mut out = Cursor::new(Vec::new());
        let mut enc = FlacStreamingEncoder::new(&mut out, 44_100, ChannelLayout::Stereo, 5)
            .expect("FlacStreamingEncoder::new");

        // Feed in three chunks.
        let n_ch = 2usize;
        let chunk_frames = 8192usize;
        for chunk in buf.samples.chunks(chunk_frames * n_ch) {
            let chunk_buf = AudioBuffer {
                samples: chunk.to_vec(),
                sample_rate: buf.sample_rate,
                channels: buf.channels,
                format: buf.format,
            };
            enc.encode_chunk(&chunk_buf).expect("encode_chunk");
        }
        enc.finalize().expect("finalize");

        let bytes = out.into_inner();
        assert!(!bytes.is_empty(), "output must not be empty");
        assert_eq!(&bytes[..4], b"fLaC", "output must start with fLaC magic");
        assert!(
            bytes.len() > 100,
            "output must contain more than the header"
        );
    }

    #[test]
    fn test_flac_streaming_encoder_decode_roundtrip() {
        use oxiaudio_core::AudioDecoder;
        use oxiaudio_decode::SymphoniaDecoder;

        let sample_rate = 44_100u32;
        let buf = sine_buf_stereo(sample_rate, 2.0);
        let expected_frames = (sample_rate as f32 * 2.0) as usize;

        // Write to a temp file for decode.
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let path = std::env::temp_dir().join(format!("oxiaudio_flac_streaming_{ts}.flac"));

        {
            let file = std::fs::File::create(&path).expect("create temp flac");
            let bw = std::io::BufWriter::new(file);
            let mut enc = FlacStreamingEncoder::new(bw, sample_rate, ChannelLayout::Stereo, 5)
                .expect("FlacStreamingEncoder::new");

            // Feed in 4096-frame chunks.
            let n_ch = 2usize;
            for chunk in buf.samples.chunks(4096 * n_ch) {
                let chunk_buf = AudioBuffer {
                    samples: chunk.to_vec(),
                    sample_rate: buf.sample_rate,
                    channels: buf.channels,
                    format: buf.format,
                };
                enc.encode_chunk(&chunk_buf).expect("encode_chunk");
            }
            enc.finalize().expect("finalize");
        }

        // Decode and verify.
        let file = std::fs::File::open(&path).expect("open temp flac");
        let reader = std::io::BufReader::new(file);
        let decoded = SymphoniaDecoder.decode(reader).expect("decode FLAC");

        let _ = std::fs::remove_file(&path);

        // Verify sample count.
        assert_eq!(
            decoded.samples.len(),
            expected_frames * 2, // stereo
            "decoded sample count must match: expected {} got {}",
            expected_frames * 2,
            decoded.samples.len()
        );

        // Verify non-silence: at least some samples should be non-zero.
        let max_abs = decoded
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs > 0.01,
            "decoded audio must be non-silent (max_abs={max_abs})"
        );
    }

    #[test]
    fn test_flac_streaming_encoder_mono() {
        let n_frames = 44_100usize;
        let samples: Vec<f32> = (0..n_frames)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 44_100.0).sin() * 0.3)
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };

        let mut out = Cursor::new(Vec::new());
        let mut enc = FlacStreamingEncoder::new(&mut out, 44_100, ChannelLayout::Mono, 3)
            .expect("FlacStreamingEncoder::new mono");
        enc.encode_chunk(&buf).expect("encode_chunk mono");
        enc.finalize().expect("finalize mono");

        let bytes = out.into_inner();
        assert_eq!(&bytes[..4], b"fLaC", "mono output must start with fLaC");
    }
}
