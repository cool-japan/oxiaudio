//! Raw WAV writing helpers for features not supported by `hound`:
//!   - `WAVE_FORMAT_EXTENSIBLE` (0xFFFE) for multi-channel audio (>2 ch)
//!   - `LIST/INFO` chunk for embedded text metadata

use std::io::{Seek, SeekFrom, Write};

use oxiaudio_core::{AudioBuffer, AudioMetadata, ChannelLayout, OxiAudioError};

// ─── WAVE_FORMAT_EXTENSIBLE SubFormat GUIDs ───────────────────────────────────

/// SubFormat GUID for IEEE_FLOAT in WAVE_FORMAT_EXTENSIBLE (LE byte order).
const SUBFORMAT_IEEE_FLOAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

// ─── Channel masks ────────────────────────────────────────────────────────────

/// Return the WAVEFORMATEXTENSIBLE dwChannelMask for the given layout.
///
/// For any layout not explicitly handled, falls back to using `channel_count()`
/// as an approximation (consecutive speaker bits from FL).
fn channel_mask(layout: ChannelLayout) -> u32 {
    match layout {
        ChannelLayout::Mono => 0x0000_0004,       // FRONT_CENTER
        ChannelLayout::Stereo => 0x0000_0003,     // FL | FR
        ChannelLayout::Quad => 0x0000_0033,       // FL | FR | BL | BR
        ChannelLayout::Surround51 => 0x0000_003F, // FL|FR|FC|LFE|BL|BR
        ChannelLayout::Surround71 => 0x0000_00FF, // FL|FR|FC|LFE|BL|BR|SL|SR
        _ => {
            // Non-exhaustive fallback: use consecutive bits from FL.
            let n = layout.channel_count().min(32);
            (1u32 << n).wrapping_sub(1)
        }
    }
}

/// Returns `true` if the buffer should use `WAVE_FORMAT_EXTENSIBLE` instead of
/// the standard `WAVE_FORMAT_IEEE_FLOAT` (0x0003).
pub fn needs_extensible(channels: ChannelLayout) -> bool {
    channels.channel_count() > 2
}

// ─── Low-level raw WAV helpers ────────────────────────────────────────────────

/// Write a u16 in little-endian to `w`.
#[inline]
fn write_u16_le<W: Write>(w: &mut W, v: u16) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

/// Write a u32 in little-endian to `w`.
#[inline]
fn write_u32_le<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

// ─── fmt chunk writing ────────────────────────────────────────────────────────

/// Write the `fmt ` chunk for standard 32-bit IEEE float PCM (WAVE_FORMAT_IEEE_FLOAT).
///
/// Chunk layout (total 24 bytes on disk: 8-byte header + 16-byte payload):
/// ```text
/// "fmt " + 16(u32 LE) + 0x0003(u16 LE) + nChannels + nSamplesPerSec
///        + nAvgBytesPerSec + nBlockAlign + wBitsPerSample
/// ```
fn write_fmt_chunk_f32<W: Write>(
    w: &mut W,
    channels: u16,
    sample_rate: u32,
) -> std::io::Result<()> {
    let bits_per_sample: u16 = 32;
    let block_align = channels * bits_per_sample / 8;
    let avg_bytes_per_sec = sample_rate * block_align as u32;

    w.write_all(b"fmt ")?;
    write_u32_le(w, 16)?; // chunk size
    write_u16_le(w, 0x0003)?; // WAVE_FORMAT_IEEE_FLOAT
    write_u16_le(w, channels)?;
    write_u32_le(w, sample_rate)?;
    write_u32_le(w, avg_bytes_per_sec)?;
    write_u16_le(w, block_align)?;
    write_u16_le(w, bits_per_sample)?;
    Ok(())
}

/// Write the `fmt ` chunk for `WAVE_FORMAT_EXTENSIBLE` with IEEE float sub-format.
///
/// Chunk payload = 40 bytes (standard 16 + cbSize(2) + extension(22)).
fn write_fmt_chunk_extensible<W: Write>(
    w: &mut W,
    channels: u16,
    sample_rate: u32,
    layout: ChannelLayout,
) -> std::io::Result<()> {
    let bits_per_sample: u16 = 32;
    let block_align = channels * bits_per_sample / 8;
    let avg_bytes_per_sec = sample_rate * block_align as u32;

    w.write_all(b"fmt ")?;
    write_u32_le(w, 40)?; // chunk size = 40 bytes
    write_u16_le(w, 0xFFFE)?; // WAVE_FORMAT_EXTENSIBLE
    write_u16_le(w, channels)?;
    write_u32_le(w, sample_rate)?;
    write_u32_le(w, avg_bytes_per_sec)?;
    write_u16_le(w, block_align)?;
    write_u16_le(w, bits_per_sample)?;
    write_u16_le(w, 22)?; // cbSize = 22
    write_u16_le(w, bits_per_sample)?; // wValidBitsPerSample = 32
    write_u32_le(w, channel_mask(layout))?;
    w.write_all(&SUBFORMAT_IEEE_FLOAT)?;
    Ok(())
}

// ─── WAV data chunk ───────────────────────────────────────────────────────────

/// Write the `data` chunk header + all f32 samples as little-endian 32-bit IEEE float.
fn write_data_chunk_f32<W: Write>(w: &mut W, samples: &[f32]) -> std::io::Result<()> {
    let byte_count = (samples.len() * 4) as u32;
    w.write_all(b"data")?;
    write_u32_le(w, byte_count)?;
    for &s in samples {
        w.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

// ─── LIST/INFO chunk helpers ──────────────────────────────────────────────────

/// Build all INFO sub-chunk bytes from `metadata` into a `Vec<u8>`.
///
/// Returns `None` if no metadata fields are `Some`.
fn build_info_bytes(metadata: &AudioMetadata) -> Option<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();

    let write_field = |buf: &mut Vec<u8>, tag: &[u8; 4], text: &str| {
        let bytes = text.as_bytes();
        let size = (bytes.len() + 1) as u32;
        buf.extend_from_slice(tag);
        buf.extend_from_slice(&size.to_le_bytes());
        buf.extend_from_slice(bytes);
        buf.push(0u8);
        if (bytes.len() + 1) % 2 != 0 {
            buf.push(0u8);
        }
    };

    if let Some(t) = &metadata.title {
        write_field(&mut buf, b"INAM", t);
    }
    if let Some(a) = &metadata.artist {
        write_field(&mut buf, b"IART", a);
    }
    if let Some(al) = &metadata.album {
        write_field(&mut buf, b"IPRD", al);
    }
    if let Some(g) = &metadata.genre {
        write_field(&mut buf, b"IGNR", g);
    }
    if let Some(c) = &metadata.comment {
        write_field(&mut buf, b"ICMT", c);
    }
    if let Some(y) = metadata.year {
        write_field(&mut buf, b"ICRD", &y.to_string());
    }
    if let Some(composer) = &metadata.composer {
        write_field(&mut buf, b"IMUS", composer);
    }

    if buf.is_empty() {
        None
    } else {
        Some(buf)
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Write a complete WAV file with `WAVE_FORMAT_EXTENSIBLE` for multi-channel audio.
///
/// For <= 2 channels, uses standard `WAVE_FORMAT_IEEE_FLOAT`.
/// For > 2 channels, uses `WAVE_FORMAT_EXTENSIBLE` with the appropriate channel mask.
pub fn write_wav_raw<W: Write + Seek>(
    buf: &AudioBuffer<f32>,
    mut w: W,
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count() as u16;
    let use_extensible = needs_extensible(buf.channels);

    // ── RIFF header placeholder ──
    w.write_all(b"RIFF").map_err(OxiAudioError::Io)?;
    let riff_size_pos = w.stream_position().map_err(OxiAudioError::Io)?;
    write_u32_le(&mut w, 0).map_err(OxiAudioError::Io)?; // placeholder
    w.write_all(b"WAVE").map_err(OxiAudioError::Io)?;

    // ── fmt chunk ──
    if use_extensible {
        write_fmt_chunk_extensible(&mut w, channels, buf.sample_rate, buf.channels)
            .map_err(OxiAudioError::Io)?;
    } else {
        write_fmt_chunk_f32(&mut w, channels, buf.sample_rate).map_err(OxiAudioError::Io)?;
    }

    // ── data chunk ──
    write_data_chunk_f32(&mut w, &buf.samples).map_err(OxiAudioError::Io)?;

    // ── Patch RIFF size ──
    let end_pos = w.stream_position().map_err(OxiAudioError::Io)?;
    let riff_size = (end_pos - riff_size_pos - 4) as u32;
    w.seek(SeekFrom::Start(riff_size_pos))
        .map_err(OxiAudioError::Io)?;
    write_u32_le(&mut w, riff_size).map_err(OxiAudioError::Io)?;

    Ok(())
}

/// Write a complete WAV file with an embedded `LIST/INFO` chunk for metadata.
///
/// Chunk order: RIFF → fmt → LIST/INFO → data
/// Also uses `WAVE_FORMAT_EXTENSIBLE` for > 2 channel layouts.
pub fn write_wav_with_metadata<W: Write + Seek>(
    buf: &AudioBuffer<f32>,
    mut w: W,
    metadata: &AudioMetadata,
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count() as u16;
    let use_extensible = needs_extensible(buf.channels);

    // ── RIFF header placeholder ──
    w.write_all(b"RIFF").map_err(OxiAudioError::Io)?;
    let riff_size_pos = w.stream_position().map_err(OxiAudioError::Io)?;
    write_u32_le(&mut w, 0).map_err(OxiAudioError::Io)?; // placeholder
    w.write_all(b"WAVE").map_err(OxiAudioError::Io)?;

    // ── fmt chunk ──
    if use_extensible {
        write_fmt_chunk_extensible(&mut w, channels, buf.sample_rate, buf.channels)
            .map_err(OxiAudioError::Io)?;
    } else {
        write_fmt_chunk_f32(&mut w, channels, buf.sample_rate).map_err(OxiAudioError::Io)?;
    }

    // ── LIST/INFO chunk (if any metadata present) ──
    if let Some(info_bytes) = build_info_bytes(metadata) {
        // LIST chunk: tag(4) + size(4) + "INFO"(4) + sub-chunks
        let list_payload = 4u32 + info_bytes.len() as u32; // "INFO" + sub-chunks
        w.write_all(b"LIST").map_err(OxiAudioError::Io)?;
        write_u32_le(&mut w, list_payload).map_err(OxiAudioError::Io)?;
        w.write_all(b"INFO").map_err(OxiAudioError::Io)?;
        w.write_all(&info_bytes).map_err(OxiAudioError::Io)?;
    }

    // ── data chunk ──
    write_data_chunk_f32(&mut w, &buf.samples).map_err(OxiAudioError::Io)?;

    // ── Patch RIFF size ──
    let end_pos = w.stream_position().map_err(OxiAudioError::Io)?;
    let riff_size = (end_pos - riff_size_pos - 4) as u32;
    w.seek(SeekFrom::Start(riff_size_pos))
        .map_err(OxiAudioError::Io)?;
    write_u32_le(&mut w, riff_size).map_err(OxiAudioError::Io)?;

    Ok(())
}

// ─── RF64 / BW64 WAV writer ──────────────────────────────────────────────────

/// Write the `fmt ` chunk for RF64 with the given PCM format parameters.
///
/// Handles both PCM (I16/I24/I32) and IEEE float (F32) formats.
/// Chunk layout (total 24 bytes on disk: 8-byte header + 16-byte payload).
fn write_rf64_fmt_chunk<W: Write>(
    w: &mut W,
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bytes_per_sample: u16,
) -> std::io::Result<()> {
    let bits_per_sample: u16 = bytes_per_sample * 8;
    let block_align = channels * bytes_per_sample;
    let avg_bytes_per_sec = sample_rate * block_align as u32;

    w.write_all(b"fmt ")?;
    write_u32_le(w, 16)?; // chunk size (standard PCM/float fmt)
    write_u16_le(w, audio_format)?; // 1=PCM, 3=IEEE_FLOAT
    write_u16_le(w, channels)?;
    write_u32_le(w, sample_rate)?;
    write_u32_le(w, avg_bytes_per_sec)?;
    write_u16_le(w, block_align)?;
    write_u16_le(w, bits_per_sample)?;
    Ok(())
}

/// Write RF64 PCM sample data into `w` according to `bit_depth`.
///
/// Returns `Err` if `bit_depth` is `U8` (not supported for RF64).
fn write_rf64_samples<W: Write>(
    w: &mut W,
    samples: &[f32],
    bit_depth: crate::WavBitDepth,
) -> Result<(), OxiAudioError> {
    match bit_depth {
        crate::WavBitDepth::F32 => {
            for &s in samples {
                w.write_all(&s.to_le_bytes()).map_err(OxiAudioError::Io)?;
            }
        }
        crate::WavBitDepth::I16 => {
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                w.write_all(&v.to_le_bytes()).map_err(OxiAudioError::Io)?;
            }
        }
        crate::WavBitDepth::I24 => {
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) * 8_388_607.0_f32) as i32;
                // Write only the low 3 bytes in LE order.
                w.write_all(&v.to_le_bytes()[..3])
                    .map_err(OxiAudioError::Io)?;
            }
        }
        crate::WavBitDepth::I32 => {
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) * i32::MAX as f32) as i32;
                w.write_all(&v.to_le_bytes()).map_err(OxiAudioError::Io)?;
            }
        }
        crate::WavBitDepth::U8 => {
            return Err(OxiAudioError::Encode(
                "RF64 encoding with U8 bit depth is not supported".into(),
            ));
        }
    }
    Ok(())
}

/// Write an audio buffer to a writer using RF64 format (supports >4 GB files).
///
/// Unlike standard WAV, RF64 uses 64-bit chunk sizes via a `ds64` chunk, making
/// it suitable for very large files. This function always emits RF64 format
/// regardless of file size.
///
/// `U8` bit depth is not supported and returns [`OxiAudioError::Encode`].
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on any I/O failure, or [`OxiAudioError::Encode`]
/// for unsupported configurations (e.g. `U8` bit depth).
#[must_use = "discarding errors ignores write failure"]
pub fn encode_wav_rf64<W: Write + Seek>(
    buf: &AudioBuffer<f32>,
    writer: &mut W,
    bit_depth: crate::WavBitDepth,
) -> Result<(), OxiAudioError> {
    let n_channels = buf.channels.channel_count() as u64;
    let n_samples = buf.samples.len() as u64;
    let n_frames = n_samples.checked_div(n_channels).unwrap_or(0);

    let (audio_format, bytes_per_sample): (u16, u64) = match bit_depth {
        crate::WavBitDepth::I16 => (1, 2),
        crate::WavBitDepth::I24 => (1, 3),
        crate::WavBitDepth::I32 => (1, 4),
        crate::WavBitDepth::F32 => (3, 4),
        crate::WavBitDepth::U8 => {
            return Err(OxiAudioError::Encode(
                "RF64 encoding with U8 bit depth is not supported".into(),
            ));
        }
    };

    let data_bytes: u64 = n_frames * n_channels * bytes_per_sample;

    // RF64 file layout:
    //   RF64 header   :  8 bytes  ("RF64" + 0xFFFF_FFFF sentinel)
    //   "WAVE"        :  4 bytes
    //   ds64 chunk    : 36 bytes  (8-byte header + 28-byte payload)
    //   fmt  chunk    : 24 bytes  (8-byte header + 16-byte payload)
    //   data header   :  8 bytes  ("data" + size field)
    //   <sample data> : data_bytes
    //
    // total_riff_payload = everything after the initial 8-byte "RF64" + sentinel:
    //   = 4 (WAVE) + 36 (ds64) + 24 (fmt) + 8 (data hdr) + data_bytes
    let total_riff_payload: u64 = 4 + 36 + 24 + 8 + data_bytes;

    // ── RF64 RIFF header ──
    writer.write_all(b"RF64").map_err(OxiAudioError::Io)?;
    // Sentinel: real size is in ds64.
    writer
        .write_all(&0xFFFF_FFFFu32.to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    writer.write_all(b"WAVE").map_err(OxiAudioError::Io)?;

    // ── ds64 chunk ──
    // Header: "ds64" + chunk_size(4)
    writer.write_all(b"ds64").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&28u32.to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    // riff_size (64-bit LE) = total_riff_payload
    #[allow(clippy::cast_possible_truncation)]
    writer
        .write_all(&((total_riff_payload & 0xFFFF_FFFF) as u32).to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    #[allow(clippy::cast_possible_truncation)]
    writer
        .write_all(&((total_riff_payload >> 32) as u32).to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    // data_size (64-bit LE)
    #[allow(clippy::cast_possible_truncation)]
    writer
        .write_all(&((data_bytes & 0xFFFF_FFFF) as u32).to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    #[allow(clippy::cast_possible_truncation)]
    writer
        .write_all(&((data_bytes >> 32) as u32).to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    // sample_count (64-bit LE) = total PCM frames
    #[allow(clippy::cast_possible_truncation)]
    writer
        .write_all(&((n_frames & 0xFFFF_FFFF) as u32).to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    #[allow(clippy::cast_possible_truncation)]
    writer
        .write_all(&((n_frames >> 32) as u32).to_le_bytes())
        .map_err(OxiAudioError::Io)?;
    // table_length = 0 (no JUNK entries)
    writer
        .write_all(&0u32.to_le_bytes())
        .map_err(OxiAudioError::Io)?;

    // ── fmt chunk ──
    #[allow(clippy::cast_possible_truncation)]
    write_rf64_fmt_chunk(
        writer,
        audio_format,
        n_channels as u16,
        buf.sample_rate,
        bytes_per_sample as u16,
    )
    .map_err(OxiAudioError::Io)?;

    // ── data chunk header ──
    writer.write_all(b"data").map_err(OxiAudioError::Io)?;
    // Use sentinel if data exceeds 32-bit range; otherwise use real size.
    let data_size_field: u32 = if data_bytes > 0xFFFF_FFFE {
        0xFFFF_FFFF
    } else {
        #[allow(clippy::cast_possible_truncation)]
        {
            data_bytes as u32
        }
    };
    writer
        .write_all(&data_size_field.to_le_bytes())
        .map_err(OxiAudioError::Io)?;

    // ── sample data ──
    write_rf64_samples(writer, &buf.samples, bit_depth)?;

    Ok(())
}

/// Write an audio buffer to a file using RF64 format (supports >4 GB files).
///
/// Creates the file at `path`, then calls [`encode_wav_rf64`].
/// `U8` bit depth is not supported and returns [`OxiAudioError::Encode`].
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on file-creation or write failure, or
/// [`OxiAudioError::Encode`] for unsupported configurations.
#[must_use = "discarding errors ignores write failure"]
pub fn encode_wav_rf64_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    bit_depth: crate::WavBitDepth,
) -> Result<(), OxiAudioError> {
    let mut file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    encode_wav_rf64(buf, &mut file, bit_depth)
}

// ─── Progress-callback WAV encoding ──────────────────────────────────────────

/// A callback invoked during chunked WAV encoding to report progress.
///
/// Called with `(frames_done, total_frames)` — both are sample-frame counts.
/// The callback should return quickly to avoid blocking the encoder.
pub type EncodeProgressFn<'a> = &'a dyn Fn(usize, usize);

/// Encode an [`AudioBuffer<f32>`][oxiaudio_core::AudioBuffer] to WAV, reporting
/// encoding progress through a callback.
///
/// The callback is called approximately every 4096 frames (or fewer for the last
/// chunk) with `(frames_written, total_frames)`.  The final call always passes
/// `total_frames` for both arguments, signalling completion.
///
/// This function uses 32-bit IEEE float PCM (the default [`WavEncoder`][crate::WavEncoder]
/// bit depth) so that the in-memory representation is lossless.  For standard mono
/// or stereo layouts the output bytes start with `b"RIFF"`.
///
/// # Errors
///
/// Returns [`OxiAudioError::Encode`] if WAV serialisation fails, or
/// [`OxiAudioError::Io`] on any I/O error.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use std::cell::Cell;
/// use oxiaudio_encode::{encode_wav_with_progress, EncodeProgressFn};
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let buf = AudioBuffer {
///     samples: vec![0.0f32; 8192],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
/// let call_count = Cell::new(0usize);
/// let mut out = Cursor::new(Vec::new());
/// encode_wav_with_progress(&buf, &mut out, &|_done, _total| {
///     call_count.set(call_count.get() + 1);
/// })
/// .unwrap();
/// assert!(call_count.get() > 0);
/// ```
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_wav_with_progress<W: Write + Seek>(
    buf: &AudioBuffer<f32>,
    writer: W,
    progress: EncodeProgressFn<'_>,
) -> Result<(), OxiAudioError> {
    use oxiaudio_core::AudioEncoder;

    let ch = buf.channels.channel_count();
    let total_frames = buf.samples.len().checked_div(ch).unwrap_or(0);
    const CHUNK_SIZE: usize = 4096; // frames per progress report

    // Serialise to an in-memory buffer first using the default WavEncoder (F32).
    // F32 = 4 bytes per sample = 4 * ch bytes per frame.
    let bytes_per_frame = ch * 4; // 32-bit IEEE float

    let mut out_bytes: Vec<u8> = Vec::with_capacity(buf.samples.len() * 4 + 44);
    crate::WavEncoder::default()
        .encode(buf, &mut std::io::Cursor::new(&mut out_bytes))
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    // Header = everything before the PCM sample data.
    let audio_byte_len = total_frames * bytes_per_frame;
    let header_bytes = out_bytes.len().saturating_sub(audio_byte_len);

    let mut bw = std::io::BufWriter::new(writer);

    // Write the WAV header (fmt chunk, etc.)
    Write::write_all(&mut bw, &out_bytes[..header_bytes]).map_err(OxiAudioError::Io)?;

    // Write PCM audio in chunks, invoking the progress callback after each.
    let audio_bytes = &out_bytes[header_bytes..];
    let bytes_per_chunk = CHUNK_SIZE * bytes_per_frame.max(1);
    let mut frames_written: usize = 0;

    for chunk in audio_bytes.chunks(bytes_per_chunk.max(1)) {
        Write::write_all(&mut bw, chunk).map_err(OxiAudioError::Io)?;
        let chunk_frames = chunk.len() / bytes_per_frame.max(1);
        frames_written = (frames_written + chunk_frames).min(total_frames);
        progress(frames_written, total_frames);
    }

    // Ensure the final progress report signals exactly total_frames.
    if frames_written != total_frames {
        progress(total_frames, total_frames);
    }

    Ok(())
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_mask_mono() {
        assert_eq!(channel_mask(ChannelLayout::Mono), 0x0000_0004);
    }

    #[test]
    fn channel_mask_quad() {
        assert_eq!(channel_mask(ChannelLayout::Quad), 0x0000_0033);
    }
}

// ─── RF64 tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod rf64_tests {
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    use crate::WavBitDepth;

    use super::encode_wav_rf64;

    fn mono_buf(n_frames: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.5f32; n_frames],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    fn sine_buf(n_frames: usize, sr: u32) -> AudioBuffer<f32> {
        let samples = (0..n_frames)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        AudioBuffer {
            samples,
            sample_rate: sr,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_rf64_magic_bytes() {
        let buf = mono_buf(1000);
        let mut out = Cursor::new(Vec::new());
        encode_wav_rf64(&buf, &mut out, WavBitDepth::I16).expect("encode_wav_rf64");
        let bytes = out.into_inner();
        assert_eq!(&bytes[0..4], b"RF64", "first 4 bytes must be RF64 magic");
        assert_eq!(&bytes[8..12], b"WAVE", "bytes 8..12 must be WAVE form type");
    }

    #[test]
    fn test_rf64_ds64_chunk_present() {
        let buf = mono_buf(1000);
        let mut out = Cursor::new(Vec::new());
        encode_wav_rf64(&buf, &mut out, WavBitDepth::I16).expect("encode_wav_rf64");
        let bytes = out.into_inner();
        // ds64 starts immediately after the 12-byte RF64+size+WAVE header.
        assert_eq!(
            &bytes[12..16],
            b"ds64",
            "bytes 12..16 must be ds64 chunk id"
        );
    }

    #[test]
    fn test_rf64_riff_size_sentinel() {
        let buf = mono_buf(1000);
        let mut out = Cursor::new(Vec::new());
        encode_wav_rf64(&buf, &mut out, WavBitDepth::I16).expect("encode_wav_rf64");
        let bytes = out.into_inner();
        // Bytes 4..8 are the RIFF size field — must be the 0xFFFF_FFFF sentinel.
        assert_eq!(
            &bytes[4..8],
            &[0xFF, 0xFF, 0xFF, 0xFF],
            "bytes 4..8 must be the RF64 size sentinel 0xFFFF_FFFF"
        );
    }

    #[test]
    fn test_rf64_data_is_decodable() {
        // Encode a known sine wave and verify PCM bytes are non-zero.
        // For I16, each sample is 2 bytes; data starts at byte 12+36+24+8 = 80.
        let n_frames = 1000usize;
        let buf = sine_buf(n_frames, 48_000);
        let mut out = Cursor::new(Vec::new());
        encode_wav_rf64(&buf, &mut out, WavBitDepth::I16).expect("encode_wav_rf64");
        let bytes = out.into_inner();

        // Verify overall file size: 80 header bytes + 1000 frames * 1 ch * 2 bytes = 2080
        assert_eq!(bytes.len(), 80 + n_frames * 2, "unexpected file size");

        // At least some PCM values must be non-zero (sine wave, not silence).
        let pcm_bytes = &bytes[80..];
        let any_nonzero = pcm_bytes.iter().any(|&b| b != 0);
        assert!(
            any_nonzero,
            "PCM bytes should contain non-zero sine wave data"
        );

        // Spot-check: decode first I16 sample and verify it is in range.
        let sample_0 = i16::from_le_bytes([pcm_bytes[0], pcm_bytes[1]]);
        assert_eq!(
            sample_0, 0,
            "first sample of sine at t=0 should be 0 (sin(0)=0)"
        );
    }
}

// ─── Progress-callback tests ──────────────────────────────────────────────────
#[cfg(test)]
mod progress_tests {
    use std::cell::Cell;
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    use super::encode_wav_with_progress;

    fn stereo_buf(n_frames: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.1f32; n_frames * 2], // stereo
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_encode_wav_with_progress_callback_called() {
        // 10000-frame stereo buffer — callback must be called at least once.
        let buf = stereo_buf(10_000);
        let call_count = Cell::new(0usize);
        let mut out = Cursor::new(Vec::<u8>::new());
        encode_wav_with_progress(&buf, &mut out, &|_done, _total| {
            call_count.set(call_count.get() + 1);
        })
        .expect("encode_wav_with_progress should succeed");
        assert!(
            call_count.get() >= 1,
            "callback must be called at least once"
        );
    }

    #[test]
    fn test_encode_wav_with_progress_final_count() {
        // Verify the last callback invocation passes total_frames for both arguments.
        let n_frames = 10_000usize;
        let buf = stereo_buf(n_frames);
        let last_done = Cell::new(0usize);
        let last_total = Cell::new(0usize);
        let mut out = Cursor::new(Vec::<u8>::new());
        encode_wav_with_progress(&buf, &mut out, &|done, total| {
            last_done.set(done);
            last_total.set(total);
        })
        .expect("encode_wav_with_progress should succeed");
        assert_eq!(
            last_done.get(),
            n_frames,
            "final progress callback must report all frames done"
        );
        assert_eq!(
            last_total.get(),
            n_frames,
            "final progress callback must report correct total"
        );
    }

    #[test]
    fn test_encode_wav_with_progress_produces_valid_wav() {
        // Output must start with b"RIFF" (standard WAV magic).
        let buf = stereo_buf(4096);
        let mut out = Cursor::new(Vec::<u8>::new());
        encode_wav_with_progress(&buf, &mut out, &|_done, _total| {})
            .expect("encode_wav_with_progress");
        let bytes = out.into_inner();
        assert!(!bytes.is_empty(), "output must not be empty");
        assert_eq!(&bytes[..4], b"RIFF", "output must start with RIFF magic");
    }
}
