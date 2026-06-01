//! AIFF streaming encoder and metadata-aware batch encoder.
//!
//! AIFF uses big-endian byte order throughout. This module provides:
//!
//! - [`AiffStreamEncoder`] — chunk-by-chunk streaming encoder with `finalize`.
//! - [`AiffBitDepth`] — 16 or 24 bit depth selection.
//! - [`encode_aiff_with_metadata`] — batch encoder that adds NAME/AUTH/ANNO chunks.

use std::io::{Seek, SeekFrom, Write};

use oxiaudio_core::{AudioBuffer, AudioMetadata, AudioSink, ChannelLayout, OxiAudioError};

/// PCM bit depth for AIFF encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiffBitDepth {
    /// 16-bit signed integer PCM (big-endian). Standard CD-quality.
    I16,
    /// 24-bit signed integer PCM (big-endian). High-resolution audio.
    I24,
}

// ─── 80-bit IEEE 754 Extended helper ─────────────────────────────────────────

/// Encode an `f64` value as an 80-bit IEEE 754 extended-precision float (big-endian).
///
/// AIFF uses this format for the sample rate field in the `COMM` chunk.
fn f64_to_extended_bytes(value: f64) -> [u8; 10] {
    let mut bytes = [0u8; 10];
    if value == 0.0 {
        return bytes;
    }
    let bits = value.to_bits();
    let sign = (bits >> 63) as u8;
    let exp = ((bits >> 52) & 0x7FF) as i32;
    // Reconstruct the 53-bit mantissa with the implicit leading 1.
    let mantissa = (bits & 0x000F_FFFF_FFFF_FFFF) | 0x0010_0000_0000_0000;
    // Re-bias for 80-bit (bias 16383 vs double bias 1023)
    let ext_exp = (exp - 1023 + 16383) as u16;
    bytes[0] = ((sign as u16) << 7 | ext_exp >> 8) as u8;
    bytes[1] = (ext_exp & 0xFF) as u8;
    // Shift 53-bit mantissa to fill the 64-bit mantissa field (explicit integer bit).
    let mantissa_shifted = mantissa << 11;
    bytes[2..10].copy_from_slice(&mantissa_shifted.to_be_bytes());
    bytes
}

// ─── AIFF text chunk helper ───────────────────────────────────────────────────

/// Write a single AIFF text chunk (NAME, AUTH, or ANNO) with even-byte alignment padding.
///
/// Layout: `[chunk_id: 4 bytes][size: 4 bytes BE u32][text: N bytes][pad: 0-1 byte]`
fn write_text_chunk<W: Write>(
    writer: &mut W,
    chunk_id: &[u8; 4],
    text: &str,
) -> std::io::Result<()> {
    let text_bytes = text.as_bytes();
    let size = text_bytes.len() as u32;
    writer.write_all(chunk_id)?;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(text_bytes)?;
    if text_bytes.len() % 2 != 0 {
        writer.write_all(&[0u8])?; // alignment pad
    }
    Ok(())
}

/// Write an AIFF file with optional NAME, AUTH, and ANNO metadata chunks.
///
/// Chunks are written in the standard AIFF order: COMM → NAME (if present) → AUTH (if present)
/// → ANNO (if present) → SSND. The FORM chunk size is backfilled after writing all content.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on any I/O failure.
#[must_use = "discarding errors ignores encode failure"]
pub fn write_aiff_with_chunks<W: Write + Seek>(
    buf: &AudioBuffer<f32>,
    writer: &mut W,
    name: Option<&str>,
    author: Option<&str>,
    annotation: Option<&str>,
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count() as u16;
    let num_frames = (buf.samples.len() / channels as usize) as u32;
    let rate_ext = f64_to_extended_bytes(buf.sample_rate as f64);

    // ── Accumulate optional text chunks using write_text_chunk helper ──
    let mut text_bytes: Vec<u8> = Vec::new();
    if let Some(t) = name {
        write_text_chunk(&mut text_bytes, b"NAME", t).map_err(OxiAudioError::Io)?;
    }
    if let Some(a) = author {
        write_text_chunk(&mut text_bytes, b"AUTH", a).map_err(OxiAudioError::Io)?;
    }
    if let Some(c) = annotation {
        write_text_chunk(&mut text_bytes, b"ANNO", c).map_err(OxiAudioError::Io)?;
    }

    // PCM bytes (16-bit big-endian)
    let pcm_bytes: Vec<u8> = buf
        .samples
        .iter()
        .flat_map(|&s| {
            let v: i16 = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            v.to_be_bytes()
        })
        .collect();

    // SSND payload: offset(4) + blockSize(4) + data
    let ssnd_payload = 8u32 + pcm_bytes.len() as u32;

    // FORM payload = "AIFF"(4) + COMM(8+18) + text_chunks + SSND(8 + ssnd_payload)
    let form_payload = 4u32 + 26 + text_bytes.len() as u32 + 8 + ssnd_payload;

    // ── FORM header ──
    writer.write_all(b"FORM").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&form_payload.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer.write_all(b"AIFF").map_err(OxiAudioError::Io)?;

    // ── COMM chunk (18-byte payload) ──
    writer.write_all(b"COMM").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&18u32.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&channels.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&num_frames.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&16u16.to_be_bytes())
        .map_err(OxiAudioError::Io)?; // sampleSize = 16
    writer.write_all(&rate_ext).map_err(OxiAudioError::Io)?;

    // ── Optional text chunks ──
    if !text_bytes.is_empty() {
        writer.write_all(&text_bytes).map_err(OxiAudioError::Io)?;
    }

    // ── SSND chunk ──
    writer.write_all(b"SSND").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&ssnd_payload.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&0u32.to_be_bytes())
        .map_err(OxiAudioError::Io)?; // offset = 0
    writer
        .write_all(&0u32.to_be_bytes())
        .map_err(OxiAudioError::Io)?; // blockSize = 0
    writer.write_all(&pcm_bytes).map_err(OxiAudioError::Io)?;

    Ok(())
}

// ─── AiffStreamEncoder ────────────────────────────────────────────────────────

/// Streaming AIFF encoder that supports chunk-by-chunk sample writing.
///
/// Call [`AiffStreamEncoder::new`] to write the header, [`AiffStreamEncoder::encode_chunk`]
/// (or [`AudioSink::write_chunk`]) for each audio chunk, then
/// [`AiffStreamEncoder::finalize`] to patch the size fields.
///
/// # Examples
///
/// ```
/// use std::io::{BufWriter, Cursor};
/// use oxiaudio_encode::{AiffBitDepth, AiffStreamEncoder};
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let buf = AudioBuffer {
///     samples: vec![0.0f32; 512],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
///
/// let mut cursor = Cursor::new(Vec::new());
/// let mut enc = AiffStreamEncoder::new(&mut cursor, 44_100, ChannelLayout::Mono, AiffBitDepth::I16)
///     .expect("new encoder");
/// enc.encode_chunk(&buf).expect("encode_chunk");
/// enc.finalize().expect("finalize");
/// ```
pub struct AiffStreamEncoder<W: Write + Seek> {
    writer: W,
    channels: ChannelLayout,
    bit_depth: AiffBitDepth,
    frames_written: u64,
    /// Byte offset of the FORM size field (4 bytes at offset 4).
    form_size_pos: u64,
    /// Byte offset of the COMM numSampleFrames field (4 bytes).
    comm_frames_pos: u64,
    /// Byte offset of the SSND ckDataSize field (4 bytes).
    ssnd_size_pos: u64,
}

impl<W: Write + Seek> AiffStreamEncoder<W> {
    /// Create a new `AiffStreamEncoder` and write the FORM/COMM/SSND skeleton.
    ///
    /// All size fields are initially zero and will be patched by [`Self::finalize`].
    pub fn new(
        mut writer: W,
        sample_rate: u32,
        channels: ChannelLayout,
        bit_depth: AiffBitDepth,
    ) -> Result<Self, OxiAudioError> {
        let n_ch = channels.channel_count() as u16;
        let bits: u16 = match bit_depth {
            AiffBitDepth::I16 => 16,
            AiffBitDepth::I24 => 24,
        };

        // ── FORM header ──
        writer.write_all(b"FORM").map_err(OxiAudioError::Io)?;
        let form_size_pos = writer.stream_position().map_err(OxiAudioError::Io)?;
        writer
            .write_all(&0u32.to_be_bytes())
            .map_err(OxiAudioError::Io)?; // placeholder
        writer.write_all(b"AIFF").map_err(OxiAudioError::Io)?;

        // ── COMM chunk (4 ckID + 4 ckSize + 18 payload = 26 bytes) ──
        writer.write_all(b"COMM").map_err(OxiAudioError::Io)?;
        writer
            .write_all(&18u32.to_be_bytes())
            .map_err(OxiAudioError::Io)?; // always 18
        writer
            .write_all(&n_ch.to_be_bytes())
            .map_err(OxiAudioError::Io)?;
        // numSampleFrames placeholder — will be patched in finalize
        let comm_frames_pos = writer.stream_position().map_err(OxiAudioError::Io)?;
        writer
            .write_all(&0u32.to_be_bytes())
            .map_err(OxiAudioError::Io)?;
        writer
            .write_all(&bits.to_be_bytes())
            .map_err(OxiAudioError::Io)?;
        // sampleRate: 80-bit IEEE extended
        let rate_ext = f64_to_extended_bytes(sample_rate as f64);
        writer.write_all(&rate_ext).map_err(OxiAudioError::Io)?;

        // ── SSND chunk header ──
        writer.write_all(b"SSND").map_err(OxiAudioError::Io)?;
        let ssnd_size_pos = writer.stream_position().map_err(OxiAudioError::Io)?;
        writer
            .write_all(&0u32.to_be_bytes())
            .map_err(OxiAudioError::Io)?; // ckDataSize placeholder
        writer
            .write_all(&0u32.to_be_bytes())
            .map_err(OxiAudioError::Io)?; // offset = 0
        writer
            .write_all(&0u32.to_be_bytes())
            .map_err(OxiAudioError::Io)?; // blockSize = 0

        Ok(Self {
            writer,
            channels,
            bit_depth,
            frames_written: 0,
            form_size_pos,
            comm_frames_pos,
            ssnd_size_pos,
        })
    }

    /// Encode one chunk of audio samples and write them to the output stream.
    ///
    /// Samples in `buf` are clamped to `[-1.0, 1.0]`, converted to the configured
    /// integer PCM format, and written in big-endian byte order.
    pub fn encode_chunk(&mut self, buf: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
        let n_ch = self.channels.channel_count();
        let n_frames = buf.samples.len().checked_div(n_ch).unwrap_or(0);

        match self.bit_depth {
            AiffBitDepth::I16 => {
                for &s in &buf.samples {
                    let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    self.writer
                        .write_all(&v.to_be_bytes())
                        .map_err(OxiAudioError::Io)?;
                }
            }
            AiffBitDepth::I24 => {
                for &s in &buf.samples {
                    let v = (s.clamp(-1.0, 1.0) * 8_388_607.0_f32) as i32;
                    // Take the upper 3 bytes of the big-endian i32 representation
                    let b = v.to_be_bytes();
                    self.writer.write_all(&b[1..4]).map_err(OxiAudioError::Io)?;
                }
            }
        }

        self.frames_written += n_frames as u64;
        Ok(())
    }

    /// Returns the total number of PCM frames written so far.
    pub fn frames_written(&self) -> u64 {
        self.frames_written
    }

    /// Finalize the AIFF file by patching the FORM, COMM, and SSND size fields.
    ///
    /// Must be called before dropping to produce a valid AIFF file.
    pub fn finalize(mut self) -> Result<(), OxiAudioError> {
        let bytes_per_sample: u64 = match self.bit_depth {
            AiffBitDepth::I16 => 2,
            AiffBitDepth::I24 => 3,
        };
        let n_ch = self.channels.channel_count() as u64;
        let pcm_bytes = self.frames_written * n_ch * bytes_per_sample;

        // SSND ckDataSize = offset(4) + blockSize(4) + pcm_bytes
        let ssnd_data_size = pcm_bytes + 8;

        // ── Patch SSND ckDataSize ──
        self.writer
            .seek(SeekFrom::Start(self.ssnd_size_pos))
            .map_err(OxiAudioError::Io)?;
        self.writer
            .write_all(&(ssnd_data_size as u32).to_be_bytes())
            .map_err(OxiAudioError::Io)?;

        // ── Patch COMM numSampleFrames ──
        self.writer
            .seek(SeekFrom::Start(self.comm_frames_pos))
            .map_err(OxiAudioError::Io)?;
        self.writer
            .write_all(&(self.frames_written as u32).to_be_bytes())
            .map_err(OxiAudioError::Io)?;

        // ── Patch FORM ckDataSize ──
        // FORM payload = "AIFF"(4) + COMM header(8) + COMM data(18) + SSND header(8) + ssnd_data_size
        let form_size = 4u64 + 8 + 18 + 8 + ssnd_data_size;
        self.writer
            .seek(SeekFrom::Start(self.form_size_pos))
            .map_err(OxiAudioError::Io)?;
        self.writer
            .write_all(&(form_size as u32).to_be_bytes())
            .map_err(OxiAudioError::Io)?;

        Ok(())
    }
}

impl<W: Write + Seek> AudioSink for AiffStreamEncoder<W> {
    fn write_chunk(&mut self, buf: &AudioBuffer<f32>) -> Result<(), OxiAudioError> {
        self.encode_chunk(buf)
    }
}

// ─── Batch encoder with metadata ─────────────────────────────────────────────

/// Encode an [`AudioBuffer<f32>`] as AIFF with optional metadata chunks.
///
/// Writes NAME (title), AUTH (artist), and ANNO (comment) chunks before SSND.
/// Uses [`AiffBitDepth`] to select between 16-bit and 24-bit PCM.
///
/// Chunk order: FORM → COMM → NAME → AUTH → ANNO → SSND
pub fn encode_aiff_with_metadata<W: Write + Seek>(
    buf: &AudioBuffer<f32>,
    mut writer: W,
    bit_depth: AiffBitDepth,
    metadata: &AudioMetadata,
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count() as u16;
    let num_frames = (buf.samples.len() / channels as usize) as u32;
    let bits: u16 = match bit_depth {
        AiffBitDepth::I16 => 16,
        AiffBitDepth::I24 => 24,
    };
    let bytes_per_sample: u64 = match bit_depth {
        AiffBitDepth::I16 => 2,
        AiffBitDepth::I24 => 3,
    };
    let pcm_bytes = num_frames as u64 * channels as u64 * bytes_per_sample;
    let rate_ext = f64_to_extended_bytes(buf.sample_rate as f64);

    // ── Collect optional text chunks ──
    let mut text_bytes: Vec<u8> = Vec::new();

    let append_text = |buf: &mut Vec<u8>, tag: &[u8; 4], text: &str| {
        let tb = text.as_bytes();
        let size = tb.len() as u32;
        buf.extend_from_slice(tag);
        buf.extend_from_slice(&size.to_be_bytes());
        buf.extend_from_slice(tb);
        if tb.len() % 2 != 0 {
            buf.push(0u8);
        }
    };

    if let Some(t) = &metadata.title {
        append_text(&mut text_bytes, b"NAME", t);
    }
    if let Some(a) = &metadata.artist {
        append_text(&mut text_bytes, b"AUTH", a);
    }
    if let Some(c) = &metadata.comment {
        append_text(&mut text_bytes, b"ANNO", c);
    }

    let ssnd_payload = 8u64 + pcm_bytes; // offset(4) + blockSize(4) + data

    // FORM payload = "AIFF"(4) + COMM(26) + text_chunks + SSND(8 + ssnd_payload)
    let form_payload = 4u64 + 26 + text_bytes.len() as u64 + 8 + ssnd_payload;

    // ── FORM header ──
    writer.write_all(b"FORM").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&(form_payload as u32).to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer.write_all(b"AIFF").map_err(OxiAudioError::Io)?;

    // ── COMM chunk ──
    writer.write_all(b"COMM").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&18u32.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&channels.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&num_frames.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&bits.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer.write_all(&rate_ext).map_err(OxiAudioError::Io)?;

    // ── Optional text chunks ──
    if !text_bytes.is_empty() {
        writer.write_all(&text_bytes).map_err(OxiAudioError::Io)?;
    }

    // ── SSND chunk ──
    writer.write_all(b"SSND").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&(ssnd_payload as u32).to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&0u32.to_be_bytes())
        .map_err(OxiAudioError::Io)?; // offset = 0
    writer
        .write_all(&0u32.to_be_bytes())
        .map_err(OxiAudioError::Io)?; // blockSize = 0

    // ── PCM samples ──
    match bit_depth {
        AiffBitDepth::I16 => {
            for &s in &buf.samples {
                let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                writer
                    .write_all(&v.to_be_bytes())
                    .map_err(OxiAudioError::Io)?;
            }
        }
        AiffBitDepth::I24 => {
            for &s in &buf.samples {
                let v = (s.clamp(-1.0, 1.0) * 8_388_607.0_f32) as i32;
                let b = v.to_be_bytes();
                writer.write_all(&b[1..4]).map_err(OxiAudioError::Io)?;
            }
        }
    }

    Ok(())
}

// ─── AIFF-C (AIFC) ────────────────────────────────────────────────────────────

/// Codec variant for AIFF-C output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiffcCodec {
    /// Uncompressed big-endian PCM (standard AIFF-C with "NONE" codec).
    None,
    /// Uncompressed little-endian PCM ("sowt" — same data but LE byte order).
    Sowt,
}

// Pascal string for NONE: length byte (14=0x0E) + "not compressed" (14 bytes) + pad (0x00)
// Total: 16 bytes → even, no additional pad needed.
const NONE_COMPRESSION_NAME: &[u8] = b"\x0Enot compressed\x00";

// Pascal string for sowt: length byte (22=0x16) + "little-endian samples" (21 bytes) + pad (0x00)
// 0x16 = 22 but the string "little-endian samples" has 21 chars. We store 22 as the length
// (matching common implementation practice where the pad is counted), total 23 bytes → odd.
// We add one more pad byte to get 24 bytes total so the COMM chunk stays even-aligned.
const SOWT_COMPRESSION_NAME: &[u8] = b"\x16little-endian samples\x00\x00";

/// Write an AIFF-C file with the given codec.
///
/// AIFF-C is the extended container format. With `AiffcCodec::None` the output
/// is functionally identical to AIFF, just in the AIFF-C envelope. With
/// `AiffcCodec::Sowt` samples are stored as little-endian PCM.
///
/// Always writes 16-bit PCM (I16 depth). For other depths, use `AiffStreamEncoder`.
#[must_use = "discarding errors ignores encode failure"]
pub fn write_aiffc<W: std::io::Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    writer: &mut W,
    codec: AiffcCodec,
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count() as u16;
    let num_frames = (buf.samples.len() / channels as usize) as u32;
    let rate_ext = f64_to_extended_bytes(buf.sample_rate as f64);

    let compression_type: &[u8; 4] = match codec {
        AiffcCodec::None => b"NONE",
        AiffcCodec::Sowt => b"sowt",
    };
    let compression_name: &[u8] = match codec {
        AiffcCodec::None => NONE_COMPRESSION_NAME,
        AiffcCodec::Sowt => SOWT_COMPRESSION_NAME,
    };

    // COMM chunk payload for AIFC:
    //   numChannels(2) + numFrames(4) + sampleSize(2) + sampleRate(10)
    //   + compressionType(4) + compressionName(len)
    let comm_payload = 2u32 + 4 + 2 + 10 + 4 + compression_name.len() as u32;

    // PCM bytes
    let pcm_bytes: Vec<u8> = match codec {
        AiffcCodec::None => buf
            .samples
            .iter()
            .flat_map(|&s| {
                let v: i16 = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                v.to_be_bytes()
            })
            .collect(),
        AiffcCodec::Sowt => buf
            .samples
            .iter()
            .flat_map(|&s| {
                let v: i16 = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                v.to_le_bytes()
            })
            .collect(),
    };

    // SSND payload: offset(4) + blockSize(4) + data
    let ssnd_payload = 8u32 + pcm_bytes.len() as u32;

    // FORM payload = "AIFC"(4) + COMM(8 + comm_payload) + SSND(8 + ssnd_payload)
    let form_payload = 4u32 + 8 + comm_payload + 8 + ssnd_payload;

    // ── FORM header (AIFC) ──
    writer.write_all(b"FORM").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&form_payload.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer.write_all(b"AIFC").map_err(OxiAudioError::Io)?;

    // ── COMM chunk ──
    writer.write_all(b"COMM").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&comm_payload.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&channels.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&num_frames.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&16u16.to_be_bytes())
        .map_err(OxiAudioError::Io)?; // sampleSize = 16
    writer.write_all(&rate_ext).map_err(OxiAudioError::Io)?;
    writer
        .write_all(compression_type)
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(compression_name)
        .map_err(OxiAudioError::Io)?;

    // ── SSND chunk ──
    writer.write_all(b"SSND").map_err(OxiAudioError::Io)?;
    writer
        .write_all(&ssnd_payload.to_be_bytes())
        .map_err(OxiAudioError::Io)?;
    writer
        .write_all(&0u32.to_be_bytes())
        .map_err(OxiAudioError::Io)?; // offset = 0
    writer
        .write_all(&0u32.to_be_bytes())
        .map_err(OxiAudioError::Io)?; // blockSize = 0
    writer.write_all(&pcm_bytes).map_err(OxiAudioError::Io)?;

    Ok(())
}

/// Path-based convenience for AIFF-C writing.
#[must_use = "discarding errors ignores encode failure"]
pub fn write_aiffc_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    codec: AiffcCodec,
) -> Result<(), OxiAudioError> {
    let mut file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    write_aiffc(buf, &mut file, codec)
}

// ─── Internal unit tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, AudioMetadata, ChannelLayout, SampleFormat};

    use super::*;

    fn sine_mono(sample_rate: u32, duration_secs: f32) -> AudioBuffer<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        let samples = (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();
        AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn f64_to_extended_44100() {
        let bytes = f64_to_extended_bytes(44_100.0f64);
        // AIFF spec: 44100 → exp=15, mantissa = 44100 * 2^(16383+1-15-63)
        // Basic sanity: first two bytes encode the exponent (0x400E for 44100)
        assert_eq!(bytes[0], 0x40);
        assert_eq!(bytes[1], 0x0E);
    }

    #[test]
    fn aiff_stream_i16_valid_header() {
        let buf = sine_mono(44_100, 0.1);
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut enc =
                AiffStreamEncoder::new(&mut cursor, 44_100, ChannelLayout::Mono, AiffBitDepth::I16)
                    .expect("new");
            enc.encode_chunk(&buf).expect("encode_chunk");
            enc.finalize().expect("finalize");
        }
        let bytes = cursor.into_inner();
        assert!(bytes.len() > 46, "AIFF too small: {}", bytes.len());
        assert_eq!(&bytes[..4], b"FORM");
        assert_eq!(&bytes[8..12], b"AIFF");
        assert_eq!(&bytes[12..16], b"COMM");
    }

    #[test]
    fn aiff_stream_i24_roundtrip_silent() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 256],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut enc =
                AiffStreamEncoder::new(&mut cursor, 48_000, ChannelLayout::Mono, AiffBitDepth::I24)
                    .expect("new");
            enc.encode_chunk(&buf).expect("encode_chunk");
            enc.finalize().expect("finalize");
        }
        let bytes = cursor.into_inner();
        assert_eq!(&bytes[..4], b"FORM");
    }

    #[test]
    fn encode_aiff_with_metadata_title() {
        let buf = sine_mono(44_100, 0.05);
        let meta = AudioMetadata {
            title: Some("M9 Test".to_string()),
            artist: Some("OxiAudio".to_string()),
            ..Default::default()
        };
        let mut cursor = Cursor::new(Vec::new());
        encode_aiff_with_metadata(&buf, &mut cursor, AiffBitDepth::I16, &meta)
            .expect("encode_aiff_with_metadata");
        let bytes = cursor.into_inner();
        assert_eq!(&bytes[..4], b"FORM");
        let has_name = bytes.windows(4).any(|w| w == b"NAME");
        assert!(has_name, "Expected NAME chunk in AIFF output");
        let has_auth = bytes.windows(4).any(|w| w == b"AUTH");
        assert!(has_auth, "Expected AUTH chunk in AIFF output");
    }

    #[test]
    fn test_aiff_name_auth_anno_chunks() {
        let buf = sine_mono(44_100, 0.1);
        let mut cursor = Cursor::new(Vec::new());
        write_aiff_with_chunks(
            &buf,
            &mut cursor,
            Some("Test Title"),
            Some("Test Artist"),
            Some("Test Note"),
        )
        .expect("write_aiff_with_chunks");
        let bytes = cursor.into_inner();
        // Must start with FORM magic
        assert_eq!(&bytes[..4], b"FORM");
        // Verify each chunk ID is present
        let has_name = bytes.windows(4).any(|w| w == b"NAME");
        assert!(has_name, "Expected NAME chunk in output");
        let has_auth = bytes.windows(4).any(|w| w == b"AUTH");
        assert!(has_auth, "Expected AUTH chunk in output");
        let has_anno = bytes.windows(4).any(|w| w == b"ANNO");
        assert!(has_anno, "Expected ANNO chunk in output");
        // Verify the text payload follows after the 8-byte header
        let find_chunk = |id: &[u8; 4]| -> Option<usize> { bytes.windows(4).position(|w| w == id) };
        if let Some(pos) = find_chunk(b"NAME") {
            let payload_start = pos + 8; // skip 4-byte id + 4-byte size
            let title_bytes = b"Test Title";
            assert!(
                bytes[payload_start..].starts_with(title_bytes),
                "NAME payload mismatch"
            );
        }
        if let Some(pos) = find_chunk(b"AUTH") {
            let payload_start = pos + 8;
            assert!(
                bytes[payload_start..].starts_with(b"Test Artist"),
                "AUTH payload mismatch"
            );
        }
        if let Some(pos) = find_chunk(b"ANNO") {
            let payload_start = pos + 8;
            assert!(
                bytes[payload_start..].starts_with(b"Test Note"),
                "ANNO payload mismatch"
            );
        }
    }

    #[test]
    fn aiff_stream_frames_count() {
        let buf = sine_mono(44_100, 0.5);
        let mut cursor = Cursor::new(Vec::new());
        let mut enc =
            AiffStreamEncoder::new(&mut cursor, 44_100, ChannelLayout::Mono, AiffBitDepth::I16)
                .expect("new");
        enc.encode_chunk(&buf).expect("encode");
        assert_eq!(enc.frames_written(), 22_050);
        enc.finalize().expect("finalize");
    }

    // ─── AIFF-C tests ──────────────────────────────────────────────────────────

    fn small_buf() -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.25f32, -0.25f32, 0.5f32],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_aiffc_none_starts_with_form_aifc() {
        let buf = small_buf();
        let mut cursor = Cursor::new(Vec::new());
        write_aiffc(&buf, &mut cursor, AiffcCodec::None).expect("write_aiffc NONE");
        let bytes = cursor.into_inner();
        assert_eq!(&bytes[0..4], b"FORM", "Expected FORM magic");
        assert_eq!(&bytes[8..12], b"AIFC", "Expected AIFC form type");
    }

    #[test]
    fn test_aiffc_none_comm_chunk_type() {
        let buf = small_buf();
        let mut cursor = Cursor::new(Vec::new());
        write_aiffc(&buf, &mut cursor, AiffcCodec::None).expect("write_aiffc NONE");
        let bytes = cursor.into_inner();
        // NONE FourCC must appear somewhere in the COMM chunk region
        let has_none = bytes.windows(4).any(|w| w == b"NONE");
        assert!(has_none, "Expected b\"NONE\" FourCC in AIFF-C COMM chunk");
    }

    #[test]
    fn test_aiffc_sowt_little_endian_sample() {
        // Create buffer with a single sample = 0.5
        // 0.5 * 32767 = 16383 = 0x3FFF
        // BE bytes: [0x3F, 0xFF]  LE bytes: [0xFF, 0x3F]
        let buf = AudioBuffer {
            samples: vec![0.5f32],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut cursor = Cursor::new(Vec::new());
        write_aiffc(&buf, &mut cursor, AiffcCodec::Sowt).expect("write_aiffc Sowt");
        let bytes = cursor.into_inner();

        // Locate SSND chunk, then skip the 8-byte header (offset+blockSize)
        let ssnd_pos = bytes
            .windows(4)
            .position(|w| w == b"SSND")
            .expect("SSND chunk must be present");
        // SSND: 4 id + 4 size + 4 offset + 4 blockSize = 16 bytes before PCM data
        let pcm_start = ssnd_pos + 16;
        assert!(
            bytes.len() >= pcm_start + 2,
            "Output too short to contain one I16 sample"
        );
        // LE encoding of 0x3FFF: low byte first → 0xFF, then 0x3F
        assert_eq!(
            bytes[pcm_start], 0xFF,
            "First LE byte should be 0xFF (low byte of 0x3FFF)"
        );
        assert_eq!(
            bytes[pcm_start + 1],
            0x3F,
            "Second LE byte should be 0x3F (high byte of 0x3FFF)"
        );
    }
}
