//! Encode-side helpers: WAV/FLAC encoding to files, streams, and in-memory vectors.

use oxiaudio_core::{AudioBuffer, AudioEncoder, OxiAudioError};

// ─── Additional encode re-exports ────────────────────────────────────────────

/// Re-export of the WAV bit-depth selector used by [`encode_wav_with_config`].
pub use oxiaudio_encode::WavBitDepth;

/// Bit-depth selector for FLAC encoding (I16 = 16-bit, I24 = 24-bit).
pub use oxiaudio_encode::FlacBitDepth;

/// Configuration for FLAC encoding (compression level + bit depth).
pub use oxiaudio_encode::FlacConfig;

/// Write an `AudioBuffer<f32>` as AIFF (16-bit signed big-endian PCM) to the file at `path`.
pub use oxiaudio_encode::write_aiff_file;

/// Bit-depth selector for AIFF encoding (8-bit, 16-bit, 24-bit, 32-bit float).
pub use oxiaudio_encode::AiffBitDepth;

/// Sample format selector for AU/SND encoding (I16, I24, F32).
pub use oxiaudio_encode::AuEncoding;

/// Streaming AIFF encoder: encodes `AudioBuffer<f32>` chunks to an AIFF stream.
pub use oxiaudio_encode::AiffStreamEncoder;

/// Encode `buf` to WAV format and return the result as `Vec<u8>`.
pub use oxiaudio_encode::encode_wav_to_vec;

/// Encode `buf` to FLAC format and return the result as `Vec<u8>`.
pub use oxiaudio_encode::encode_flac_to_vec;

/// Encode `buf` to FLAC with the specified compression level (0–8).
pub use oxiaudio_encode::encode_flac_with_level;

/// Trait for streaming (chunk-by-chunk) audio encoders.
pub use oxiaudio_encode::StreamEncoder;

/// Streaming WAV encoder: encodes `AudioBuffer<f32>` chunks without buffering the full file.
pub use oxiaudio_encode::WavStreamEncoder;

/// Streaming FLAC encoder (accumulates chunks, encodes on finalize).
pub use oxiaudio_encode::FlacStreamEncoder;

/// True-streaming FLAC encoder (encodes frames immediately, no full-audio buffering).
pub use oxiaudio_encode::FlacStreamingEncoder;

/// Apply TPDF dithering in-place before integer quantization.
pub use oxiaudio_encode::apply_tpdf_dither;

/// Embed raw album-art bytes in a FLAC stream (convenience wrapper).
pub use oxiaudio_encode::encode_flac_with_album_art;

/// File-based convenience wrapper to embed album art in a FLAC file.
pub use oxiaudio_encode::encode_flac_with_album_art_file;

/// Builder-style configuration for WAV/FLAC encoding with optional pre-processing.
pub use oxiaudio_encode::EncoderConfig;

// ─── M20 — Vorbis + AAC + Opus streaming + SILK re-exports ───────────────────

/// Encode an `AudioBuffer<f32>` as OGG Vorbis I and return the bytes.
pub use oxiaudio_encode::encode_vorbis;

/// Encode an `AudioBuffer<f32>` as OGG Vorbis I and write to a file.
pub use oxiaudio_encode::encode_vorbis_file;

/// Encode an `AudioBuffer<f32>` as OGG Vorbis I with explicit VBR quality control.
pub use oxiaudio_encode::encode_vorbis_with_quality;

/// Encode an `AudioBuffer<f32>` as OGG Vorbis I with explicit quality, writing to a file.
pub use oxiaudio_encode::encode_vorbis_quality_file;

/// Vorbis encoder VBR quality level (q-1 through q10, or equivalently −0.1 to 1.0).
pub use oxiaudio_encode::VorbisQuality;

/// Encode an `AudioBuffer<f32>` as AAC-LC ADTS frames.
pub use oxiaudio_encode::encode_aac;

/// Encode an `AudioBuffer<f32>` as AAC-LC ADTS and write to a file.
pub use oxiaudio_encode::encode_aac_file;

/// Encode an `AudioBuffer<f32>` as an M4A/MP4 container wrapping AAC-LC audio.
pub use oxiaudio_encode::encode_m4a;

/// Encode an `AudioBuffer<f32>` as M4A and write to a file at `path`.
pub use oxiaudio_encode::encode_m4a_file;

/// Streaming Opus encoder: accepts PCM frames one at a time.
pub use oxiaudio_encode::OpusStreamEncoder;

/// Configuration for Opus encoding (bitrate, frame size).
pub use oxiaudio_encode::OpusEncodeConfig;

/// SILK bandwidth selection (NB/MB/WB/SWB) for voice encoding.
pub use oxiaudio_encode::SilkBandwidth;

/// SILK LP frame structure (NLSFs, residual, pitch, gain).
pub use oxiaudio_encode::SilkLpcFrame;

/// Analyze a PCM frame and extract SILK LP parameters (structural stub).
pub use oxiaudio_encode::analyze_silk_frame;

/// Encode a SILK LP frame to bytes (structural stub).
pub use oxiaudio_encode::encode_silk_frame;

/// Encode an `AudioBuffer<f32>` to an OGG Vorbis file at `path`.
///
/// Produces a valid Vorbis I stream with real MDCT-encoded audio (floor type-1 + residue VQ).
/// Accepts 44100 or 48000 Hz mono or stereo input.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_vorbis_to_file(
    buf: &AudioBuffer<f32>,
    path: impl AsRef<std::path::Path>,
) -> Result<(), OxiAudioError> {
    oxiaudio_encode::encode_vorbis_file(buf, path.as_ref())
}

/// Encode an `AudioBuffer<f32>` to an AAC-LC ADTS file at `path`.
///
/// Produces a valid AAC-LC ADTS stream with real MDCT spectral (CB11/ESC_HCB) Huffman encoding.
/// Accepts 1–2 channels at common sample rates.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_aac_to_file(
    buf: &AudioBuffer<f32>,
    path: impl AsRef<std::path::Path>,
) -> Result<(), OxiAudioError> {
    oxiaudio_encode::encode_aac_file(buf, path.as_ref())
}

/// Encode an `AudioBuffer<f32>` to a WAV file at the given path.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// let buf = oxiaudio::decode_file(Path::new("input.flac")).unwrap();
/// oxiaudio::encode_wav(&buf, Path::new("output.wav")).unwrap();
/// ```
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_wav(
    buf: &AudioBuffer<f32>,
    path: impl AsRef<std::path::Path>,
) -> Result<(), OxiAudioError> {
    use oxiaudio_encode::WavEncoder;
    let file = std::fs::File::create(path.as_ref()).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    WavEncoder::default().encode(buf, writer)
}

/// Encode an `AudioBuffer<f32>` to a FLAC file at the given path.
///
/// `compression_level` is taken from `FlacEncoder::default()` (level 5).
/// Use `oxiaudio_encode::FlacEncoder` directly for custom compression levels.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac(
    buf: &AudioBuffer<f32>,
    path: impl AsRef<std::path::Path>,
) -> Result<(), OxiAudioError> {
    use oxiaudio_encode::FlacEncoder;
    let file = std::fs::File::create(path.as_ref()).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    FlacEncoder::default().encode(buf, writer)
}

/// Encode an `AudioBuffer<f32>` to a WAV file with an explicit bit depth.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_wav_with_config(
    buf: &AudioBuffer<f32>,
    path: impl AsRef<std::path::Path>,
    bit_depth: WavBitDepth,
) -> Result<(), OxiAudioError> {
    use oxiaudio_encode::WavEncoder;
    let file = std::fs::File::create(path.as_ref()).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    WavEncoder { bit_depth }.encode(buf, writer)
}

/// Encode an `AudioBuffer<f32>` to a FLAC file with a custom [`FlacConfig`].
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// let buf = oxiaudio::decode_file(Path::new("input.wav")).unwrap();
/// let config = oxiaudio::FlacConfig { compression: 8, bit_depth: oxiaudio::FlacBitDepth::I24 };
/// oxiaudio::encode_flac_with_config(&buf, Path::new("output.flac"), &config).unwrap();
/// ```
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_with_config(
    buf: &AudioBuffer<f32>,
    path: impl AsRef<std::path::Path>,
    config: &FlacConfig,
) -> Result<(), OxiAudioError> {
    use oxiaudio_encode::encode_flac_with_config as encode_with_cfg;
    let file = std::fs::File::create(path.as_ref()).map_err(OxiAudioError::Io)?;
    let mut writer = std::io::BufWriter::new(file);
    encode_with_cfg(buf, &mut writer, config)
}

/// Encode an iterator of `AudioBuffer<f32>` chunks to a WAV stream using `WavStreamEncoder`.
///
/// The sample rate and channel layout are taken from the first chunk. If the iterator
/// yields no chunks, the function returns `Ok(())` immediately without writing anything.
///
/// # Errors
///
/// Returns an error if the underlying encoder fails to write any chunk or finalize the stream.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_stream<W: std::io::Write + std::io::Seek>(
    mut chunks: impl Iterator<Item = impl std::borrow::Borrow<AudioBuffer<f32>>>,
    writer: W,
) -> Result<(), OxiAudioError> {
    use oxiaudio_encode::{WavBitDepth, WavStreamEncoder};
    let first = match chunks.next() {
        Some(c) => c,
        None => return Ok(()),
    };
    let first = first.borrow();
    let mut enc =
        WavStreamEncoder::new(writer, first.sample_rate, first.channels, WavBitDepth::F32)?;
    enc.encode_chunk(first)?;
    for chunk in chunks {
        enc.encode_chunk(chunk.borrow())?;
    }
    enc.finalize()
}

/// Encode an `AudioBuffer<f64>` to a WAV file by converting samples to f32 first.
///
/// The output format is 32-bit float PCM WAV. Precision beyond f32 is lost.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_wav_f64(
    buf: &AudioBuffer<f64>,
    path: impl AsRef<std::path::Path>,
) -> Result<(), OxiAudioError> {
    use oxiaudio_encode::WavEncoder;
    let f32_buf = buf.to_f32();
    let file = std::fs::File::create(path.as_ref()).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    WavEncoder::default().encode(&f32_buf, writer)
}

/// Encode an `AudioBuffer<f32>` to an AIFF file at `path`.
///
/// Uses 16-bit signed big-endian PCM. For higher bit depth or metadata, use
/// [`oxiaudio_encode::AiffStreamEncoder`] directly.
///
/// # Errors
/// Returns `OxiAudioError` on I/O failure or encoding error.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_aiff(
    buf: &AudioBuffer<f32>,
    path: impl AsRef<std::path::Path>,
) -> Result<(), OxiAudioError> {
    use std::io::BufWriter;
    let file = std::fs::File::create(path.as_ref()).map_err(OxiAudioError::Io)?;
    let mut enc = oxiaudio_encode::AiffStreamEncoder::new(
        BufWriter::new(file),
        buf.sample_rate,
        buf.channels,
        oxiaudio_encode::AiffBitDepth::I16,
    )?;
    enc.encode_chunk(buf)?;
    enc.finalize()
}

/// Encode an `AudioBuffer<f32>` to an AU/SND file at `path` using 16-bit PCM.
///
/// For other bit depths (`I24`, `F32`), use [`oxiaudio_encode::encode_au_file`] directly
/// with an explicit [`oxiaudio_encode::AuEncoding`] variant.
///
/// # Errors
///
/// Returns `OxiAudioError` on I/O failure or encoding error.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_au(
    buf: &AudioBuffer<f32>,
    path: impl AsRef<std::path::Path>,
) -> Result<(), OxiAudioError> {
    oxiaudio_encode::encode_au_file(buf, path, oxiaudio_encode::AuEncoding::I16)
}

// ─── M18 encode additions ─────────────────────────────────────────────────────

/// AIFF-C compression codec selector (NONE = uncompressed, ULAW = μ-law, ALAW = A-law).
pub use oxiaudio_encode::AiffcCodec;

/// Write an AIFF-C file to any `Write + Seek` destination.
pub use oxiaudio_encode::write_aiffc;

/// Write an AIFF-C file at the given path.
pub use oxiaudio_encode::write_aiffc_file;

/// Encode an `AudioBuffer<f32>` as a WAV stream to any `Write` (no `Seek` required).
///
/// Writes 16-bit PCM with sentinel (0xFFFF_FFFF) RIFF/data sizes for streaming
/// to non-seekable destinations such as TCP sockets or pipes.
pub use oxiaudio_encode::encode_wav_streaming;

// ─── M17 encode additions ─────────────────────────────────────────────────────

/// Encode an `AudioBuffer<f32>` to a WAV RF64 stream (64-bit sizes for files >4 GiB).
pub use oxiaudio_encode::encode_wav_rf64;

/// Encode an `AudioBuffer<f32>` to a WAV RF64 file at the given path.
pub use oxiaudio_encode::encode_wav_rf64_file;

// ─── M16 encode additions ─────────────────────────────────────────────────────

/// ID3v2.4 tag writer with UTF-8 text encoding and syncsafe frame sizes.
///
/// Supports TIT2, TPE1, TALB, TDRC, TRCK, TCON, COMM, TCOM, APIC, and TXXX frames.
pub use oxiaudio_encode::Id3v24Tag;

/// APEv2 tag item (key + value) for embedding metadata in APE/MP3/WavPack files.
pub use oxiaudio_encode::ApeItem;

/// Write an APEv2 tag block to any writer.
pub use oxiaudio_encode::write_apev2;

/// Configuration for FLAC encoding with embedded Vorbis-comment metadata.
pub use oxiaudio_encode::FlacMetaConfig;

/// Encode an `AudioBuffer<f32>` to FLAC with embedded Vorbis-comment metadata.
pub use oxiaudio_encode::encode_flac_with_metadata;

// ─── M15 encode additions ─────────────────────────────────────────────────────

/// Write an AIFF file with optional NAME, AUTH, and ANNO metadata chunks.
pub use oxiaudio_encode::write_aiff_with_chunks;

/// Target loudness levels for two-pass normalization (Streaming, Podcast, Broadcast, Custom).
pub use oxiaudio_encode::LoudnessTarget;

/// Measure integrated loudness of a buffer and return the linear gain needed to reach `target`.
pub use oxiaudio_encode::analyze_loudness_gain;

/// Two-pass loudness normalization: measure → gain → encode as WAV to a writer.
pub use oxiaudio_encode::encode_normalized_wav;

/// Two-pass loudness normalization: measure → gain → encode as WAV to a file path.
pub use oxiaudio_encode::encode_normalized_wav_file;

/// Write an AIFF file with optional NAME, AUTH, ANNO metadata chunks, given a file path.
///
/// Convenience wrapper around [`write_aiff_with_chunks`] that creates the file for you.
#[must_use = "discarding the Result ignores write errors"]
pub fn encode_aiff_with_chunks(
    buf: &oxiaudio_core::AudioBuffer<f32>,
    path: impl AsRef<std::path::Path>,
    name: Option<&str>,
    author: Option<&str>,
    annotation: Option<&str>,
) -> Result<(), oxiaudio_core::OxiAudioError> {
    use std::io::BufWriter;
    let file = std::fs::File::create(path.as_ref()).map_err(oxiaudio_core::OxiAudioError::Io)?;
    let mut writer = BufWriter::new(file);
    oxiaudio_encode::write_aiff_with_chunks(buf, &mut writer, name, author, annotation)
}

// ─── Opus encoder ────────────────────────────────────────────────────────────

/// Encode an [`AudioBuffer<f32>`] to OGG Opus format (48 kHz, 1–2 channels only).
///
/// Writes an OGG Opus stream with `OpusHead` and `OpusTags` header pages followed
/// by CELT-mode audio frames. The `target_bitrate_kbps` parameter is accepted for
/// API compatibility but is not used by the current structural encoder.
///
/// **Note**: The current CELT implementation uses non-conformant 4-bit placeholder
/// quantization and will not produce decodable output for standard Opus decoders.
pub use oxiaudio_encode::encode_opus;

/// Encode an [`AudioBuffer<f32>`] to an OGG Opus file at `path`.
///
/// Convenience wrapper around [`encode_opus`].
pub use oxiaudio_encode::encode_opus_file;

/// Mode selector for [`encode_opus_conformant`] — chooses the conformant
/// per-frame Opus encoder (CELT, SILK, or Hybrid).
pub use oxiaudio_encode::OpusConformantMode;

/// Encode an [`AudioBuffer<f32>`] to OGG Opus using the **RFC 6716–conformant**
/// per-frame encoders (opt-in alternative to [`encode_opus`]).
///
/// Unlike [`encode_opus`] (which uses a non-conformant 4-bit placeholder CELT
/// path), this routes each 20 ms frame through a conformant SILK/CELT/Hybrid
/// writer, producing an OGG Opus stream that standard Opus decoders accept.
///
/// # Conformance level (measured against the `opus-decoder` reference crate)
/// - [`OpusConformantMode::Celt`] (default-quality): full MDCT + PVQ; decoded
///   output correlates (>0.1) with the input tone — real spectral content, but
///   this is a coarse "not-silence" gate, **not** high-SNR transparency.
/// - [`OpusConformantMode::Silk`]: **silence-only** — the conformant SILK writer
///   currently emits an inactive zero-excitation frame and ignores PCM content.
/// - [`OpusConformantMode::Hybrid`]: low band silence + CELT high-band (bands 17–20).
///
/// Audio frames are mono; stereo input is downmixed to mono per frame (the OGG
/// `OpusHead` still advertises the input channel count, matching [`encode_opus`]).
///
/// [`encode_opus`] and its byte output are unaffected by this function.
pub use oxiaudio_encode::encode_opus_conformant;

/// Encode an [`AudioBuffer<f32>`] to a conformant OGG Opus file at `path`.
///
/// File-writing convenience wrapper around [`encode_opus_conformant`]; see that
/// function for the per-mode conformance caveats (SILK is silence-only, CELT is
/// coarse-gated, not transparent).
pub use oxiaudio_encode::encode_opus_conformant_file;

// ─── M19 — FLAC MD5 verification ─────────────────────────────────────────────

/// Encode an `AudioBuffer<f32>` to FLAC with an MD5 checksum embedded in the STREAMINFO block.
pub use oxiaudio_encode::encode_flac_with_md5;

/// Encode an `AudioBuffer<f32>` to a FLAC file with an MD5 checksum embedded in STREAMINFO.
pub use oxiaudio_encode::encode_flac_with_md5_file;

/// Inject (or overwrite) the MD5 checksum in an existing FLAC file's STREAMINFO block in-place.
pub use oxiaudio_encode::inject_flac_md5;

// ─── M19 — FLAC seektable + WAV progress + FLAC picture + WAV cue ───────────

/// Encode an `AudioBuffer<f32>` to FLAC with a seektable for fast seeking.
pub use oxiaudio_encode::encode_flac_with_seektable;

/// Encode an `AudioBuffer<f32>` to a FLAC file with a seektable for fast seeking.
pub use oxiaudio_encode::encode_flac_with_seektable_file;

/// Progress callback type for encoding operations.
pub use oxiaudio_encode::EncodeProgressFn;

/// Encode an `AudioBuffer<f32>` to a WAV stream with a progress callback.
pub use oxiaudio_encode::encode_wav_with_progress;

/// Encode an `AudioBuffer<f32>` to FLAC with a progress callback.
///
/// The callback receives `(frames_accumulated, total_frames)` after each 4096-frame chunk.
/// The final invocation always passes `(total_frames, total_frames)`.
pub use oxiaudio_encode::encode_flac_with_progress;

/// Encode an `AudioBuffer<f32>` to FLAC using rayon to parallelise the f32 → i32
/// PCM sample-conversion step, then encode sequentially with flacenc.
///
/// Returns the same output as [`encode_flac`] but benefits from multi-core throughput
/// for the conversion phase on large buffers.
pub use oxiaudio_encode::encode_flac_parallel;

// ─── M19 — FLAC picture + WAV cue ────────────────────────────────────────────

/// Embed cover-art / picture metadata in a FLAC stream.
pub use oxiaudio_encode::FlacPicture;

/// Encode an `AudioBuffer<f32>` to FLAC with an embedded picture block.
pub use oxiaudio_encode::encode_flac_with_picture;

/// Encode an `AudioBuffer<f32>` to a FLAC file with an embedded picture block.
pub use oxiaudio_encode::encode_flac_with_picture_file;

/// Encode an `AudioBuffer<f32>` to FLAC with both Vorbis-comment metadata and a picture block.
pub use oxiaudio_encode::encode_flac_with_metadata_and_picture;

/// A single cue-sheet cue point for embedding in a WAV file.
pub use oxiaudio_encode::CuePoint;

/// Write an `AudioBuffer<f32>` as a WAV file with embedded cue-sheet markers to a `Write+Seek` stream.
pub use oxiaudio_encode::encode_wav_with_cues;

/// Write an `AudioBuffer<f32>` as a WAV file with embedded cue-sheet markers at the given path.
pub use oxiaudio_encode::encode_wav_with_cues_file;

/// Encode multiple `AudioBuffer<f32>` chunks as a streaming FLAC file.
///
/// Chunks must share the same sample_rate and channels. The FLAC stream
/// is written to `writer` (must implement `Write + Seek`).
///
/// `compression_level` is 0 (fastest) to 8 (best); default is 5.
///
/// # Errors
/// Returns `OxiAudioError` on I/O failure, empty chunks iterator, or mismatch.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_stream_flac<W: std::io::Write + std::io::Seek>(
    chunks: &[&AudioBuffer<f32>],
    writer: W,
    compression_level: u8,
) -> Result<(), OxiAudioError> {
    let first = chunks
        .first()
        .ok_or_else(|| OxiAudioError::Encode("encode_stream_flac: empty chunks".into()))?;
    let mut enc = oxiaudio_encode::FlacStreamEncoder::new(
        writer,
        first.sample_rate,
        first.channels,
        compression_level,
    );
    for chunk in chunks {
        enc.encode_chunk(chunk)?;
    }
    enc.finalize()
}
