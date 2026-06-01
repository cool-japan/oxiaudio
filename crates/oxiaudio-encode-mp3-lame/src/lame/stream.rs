//! Chunk-at-a-time streaming MP3 encoder backed by LAME.

use mp3lame_encoder::{max_required_buffer_size, DualPcm, Encoder, FlushNoGap, MonoPcm};
use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError};
use std::io::Write;

use super::encoder::build_lame_encoder;
use super::encoder::LameMp3Encoder;
use super::id3v2;

/// Minimum number of per-channel samples LAME processes per granule.
const LAME_FRAME_SAMPLES: usize = 1152;

/// Chunk-at-a-time MP3 encoder.
///
/// Call [`encode_chunk`](LameMp3StreamEncoder::encode_chunk) for each buffer of audio;
/// call [`finalize`](LameMp3StreamEncoder::finalize) to flush remaining frames.
///
/// If `id3_tags` are configured in the encoder config, the ID3v2.3 tag is written to
/// `dst` before any MP3 frames are produced.
///
/// Unlike WAV, MP3 requires no header backfill — `dst` needs only `impl Write`.
pub struct LameMp3StreamEncoder<W: Write> {
    encoder: Encoder,
    dst: W,
    is_mono: bool,
    sample_rate: u32,
    channels: ChannelLayout,
    frames_encoded: u64,
    bytes_written: u64,
    /// Pending interleaved f32 samples that have not yet been passed to LAME.
    /// Accumulated until at least `LAME_FRAME_SAMPLES * n_channels` are available,
    /// at which point complete LAME frames are encoded and this buffer is drained.
    accumulator: Vec<f32>,
}

impl<W: Write> LameMp3StreamEncoder<W> {
    /// Create a new streaming MP3 encoder.
    ///
    /// The LAME encoder is initialised immediately. If `config.id3_tags` is `Some`,
    /// the ID3v2.3 tag bytes are written to `dst` before any MP3 frames.
    ///
    /// # Parameters
    /// - `dst`: Output sink.
    /// - `config`: Encoder settings (bitrate, mode, optional tags).
    /// - `sample_rate`: Sample rate in Hz (e.g. 44100).
    /// - `channels`: Channel layout for all subsequent `encode_chunk` calls.
    pub fn new(
        mut dst: W,
        config: &LameMp3Encoder,
        sample_rate: u32,
        channels: ChannelLayout,
    ) -> Result<Self, OxiAudioError> {
        let is_mono = matches!(channels, ChannelLayout::Mono);
        let n_channels: u8 = if is_mono { 1 } else { 2 };

        let encoder = build_lame_encoder(config, sample_rate, n_channels, is_mono)?;

        let mut bytes_written: u64 = 0;

        // Write ID3v2.4 tag before any MP3 frames.
        if let Some(ref tags) = config.id3_tags {
            let id3_bytes = id3v2::write_id3v2_4(tags);
            bytes_written = bytes_written.saturating_add(id3_bytes.len() as u64);
            dst.write_all(&id3_bytes)?;
        }

        Ok(Self {
            encoder,
            dst,
            is_mono,
            sample_rate,
            channels,
            frames_encoded: 0,
            bytes_written,
            accumulator: Vec::new(),
        })
    }

    /// Number of audio frames (per-channel samples) encoded so far.
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
    }

    /// Bytes written to the output writer so far (including any ID3 tag header).
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Encode one chunk of audio samples and write the resulting MP3 frames to `dst`.
    ///
    /// Small chunks (fewer than 1152 per-channel samples) are accumulated in an
    /// internal buffer and only passed to LAME once a complete granule (1152
    /// per-channel samples) is available, amortising the per-frame LAME overhead.
    ///
    /// The `buf` must have the same `sample_rate` and `channels` as those passed to
    /// [`new`](LameMp3StreamEncoder::new); returns `Err` if either mismatches.
    pub fn encode_chunk(&mut self, buf: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
        if buf.sample_rate != self.sample_rate {
            return Err(OxiAudioError::Encode(format!(
                "sample rate mismatch: expected {}, got {}",
                self.sample_rate, buf.sample_rate
            )));
        }
        if buf.channels != self.channels {
            return Err(OxiAudioError::Encode(format!(
                "channel layout mismatch: stream was opened as {:?}, chunk is {:?}",
                self.channels, buf.channels
            )));
        }

        // Count frames on entry to preserve the existing semantic (total PCM
        // frames presented to encode_chunk, not total frames consumed by LAME).
        let n_input_frames = if self.is_mono {
            buf.samples.len()
        } else {
            buf.samples.len() / 2
        };
        self.frames_encoded = self.frames_encoded.saturating_add(n_input_frames as u64);

        // Accumulate incoming samples.
        self.accumulator.extend_from_slice(&buf.samples);

        let n_channels: usize = if self.is_mono { 1 } else { 2 };
        let batch_samples = LAME_FRAME_SAMPLES * n_channels;

        // Drain complete LAME granules (1152 per-channel samples) from the accumulator.
        while self.accumulator.len() >= batch_samples {
            let batch: Vec<f32> = self.accumulator.drain(..batch_samples).collect();
            self.encode_batch(&batch)?;
        }

        Ok(())
    }

    /// Encode a single fully-sized batch of interleaved f32 samples and write
    /// the resulting MP3 bytes to `dst`.
    ///
    /// `batch` must contain exactly `LAME_FRAME_SAMPLES * n_channels` samples.
    fn encode_batch(&mut self, batch: &[f32]) -> Result<(), OxiAudioError> {
        let to_i16 = |s: f32| -> i16 { (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16 };

        let n_frames = if self.is_mono {
            batch.len()
        } else {
            batch.len() / 2
        };
        let cap = max_required_buffer_size(n_frames).saturating_add(7200);
        let mut mp3_out: Vec<u8> = Vec::with_capacity(cap);

        if self.is_mono {
            // chunks_exact(8) SIMD auto-vectorisation hint.
            let mut pcm: Vec<i16> = Vec::with_capacity(batch.len());
            for chunk in batch.chunks_exact(8) {
                pcm.push(to_i16(chunk[0]));
                pcm.push(to_i16(chunk[1]));
                pcm.push(to_i16(chunk[2]));
                pcm.push(to_i16(chunk[3]));
                pcm.push(to_i16(chunk[4]));
                pcm.push(to_i16(chunk[5]));
                pcm.push(to_i16(chunk[6]));
                pcm.push(to_i16(chunk[7]));
            }
            for &s in batch.chunks_exact(8).remainder() {
                pcm.push(to_i16(s));
            }
            self.encoder
                .encode_to_vec(MonoPcm(&pcm), &mut mp3_out)
                .map_err(|e| OxiAudioError::Encode(format!("encode chunk (mono): {e}")))?;
        } else {
            // Pre-allocate both buffers upfront; use 16-sample SIMD hint.
            let mut left: Vec<i16> = Vec::with_capacity(n_frames);
            let mut right: Vec<i16> = Vec::with_capacity(n_frames);
            for chunk in batch.chunks_exact(16) {
                left.push(to_i16(chunk[0]));
                right.push(to_i16(chunk[1]));
                left.push(to_i16(chunk[2]));
                right.push(to_i16(chunk[3]));
                left.push(to_i16(chunk[4]));
                right.push(to_i16(chunk[5]));
                left.push(to_i16(chunk[6]));
                right.push(to_i16(chunk[7]));
                left.push(to_i16(chunk[8]));
                right.push(to_i16(chunk[9]));
                left.push(to_i16(chunk[10]));
                right.push(to_i16(chunk[11]));
                left.push(to_i16(chunk[12]));
                right.push(to_i16(chunk[13]));
                left.push(to_i16(chunk[14]));
                right.push(to_i16(chunk[15]));
            }
            for pair in batch.chunks_exact(16).remainder().chunks_exact(2) {
                left.push(to_i16(pair[0]));
                right.push(to_i16(pair[1]));
            }
            self.encoder
                .encode_to_vec(
                    DualPcm {
                        left: &left,
                        right: &right,
                    },
                    &mut mp3_out,
                )
                .map_err(|e| OxiAudioError::Encode(format!("encode chunk (stereo): {e}")))?;
        }

        let written = mp3_out.len() as u64;
        self.dst.write_all(&mp3_out)?;
        self.bytes_written = self.bytes_written.saturating_add(written);
        Ok(())
    }

    /// Flush remaining LAME encoder frames and close the stream.
    ///
    /// Any samples that were accumulated but not yet encoded (because they did not
    /// fill a complete 1152-sample LAME granule) are flushed to LAME here before
    /// the final `FlushNoGap` call.
    ///
    /// This must be called exactly once after all [`encode_chunk`](Self::encode_chunk)
    /// calls are complete. The encoder is consumed and cannot be reused.
    pub fn finalize(mut self) -> Result<(), OxiAudioError> {
        // Flush any samples that didn't fill a full LAME granule.
        if !self.accumulator.is_empty() {
            let remainder = std::mem::take(&mut self.accumulator);
            self.encode_batch(&remainder)?;
        }

        // LAME flush requires at least 7200 bytes of headroom in the output buffer.
        let mut out: Vec<u8> = Vec::with_capacity(7200);
        self.encoder
            .flush_to_vec::<FlushNoGap>(&mut out)
            .map_err(|e| OxiAudioError::Encode(format!("flush: {e}")))?;
        self.dst.write_all(&out)?;
        Ok(())
    }

    /// Estimated average bitrate in kbps based on bytes written and frames encoded.
    ///
    /// Returns `None` if fewer than 1 second of audio has been encoded or if the
    /// sample rate is zero.
    pub fn estimated_bitrate_kbps(&self) -> Option<u32> {
        if self.frames_encoded == 0 || self.sample_rate == 0 {
            return None;
        }
        let duration_secs = self.frames_encoded as f64 / self.sample_rate as f64;
        if duration_secs < 1.0 {
            return None;
        }
        Some((self.bytes_written as f64 * 8.0 / duration_secs / 1000.0) as u32)
    }

    /// Elapsed audio time in seconds based on frames encoded and sample rate.
    pub fn elapsed_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.frames_encoded as f64 / self.sample_rate as f64
    }
}
