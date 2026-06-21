/// FLAC Vorbis comment metadata support.
///
/// Uses `flacenc`'s `add_metadata_block` API with `MetadataBlockData::new_unknown(4, payload)`
/// to embed a VORBIS_COMMENT (type 4) metadata block in the FLAC stream.
use flacenc::bitsink::ByteSink;
use flacenc::component::{BitRepr, MetadataBlockData};
use flacenc::error::Verify;
use flacenc::source::MemSource;
use oxiaudio_core::{AudioBuffer, OxiAudioError};

use crate::flac_core::{block_size_for_level, clamp_flac_bits, flac_full_scale};

/// Configuration for FLAC encoding with optional Vorbis comment metadata.
///
/// Vorbis comment tags are written as FLAC metadata block type 4
/// (VORBIS_COMMENT) immediately after STREAMINFO.
///
/// # Examples
///
/// ```
/// use oxiaudio_encode::FlacMetaConfig;
///
/// let cfg = FlacMetaConfig {
///     compression_level: 5,
///     bits_per_sample: 16,
///     comments: vec![
///         ("TITLE".to_string(), "My Song".to_string()),
///         ("ARTIST".to_string(), "OxiAudio".to_string()),
///     ],
/// };
/// assert_eq!(cfg.compression_level, 5);
/// assert_eq!(cfg.comments.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct FlacMetaConfig {
    /// Compression level 0–8 (0 = fastest, 8 = best compression). Default: 5.
    pub compression_level: u8,
    /// PCM bit depth written to the FLAC stream (16, 20, or 24). Other values are
    /// clamped to the nearest supported depth. Default: 16.
    pub bits_per_sample: u8,
    /// Vorbis comment tags: each entry is (KEY, value), e.g. ("TITLE", "My Song").
    ///
    /// Keys are case-insensitive per the Vorbis spec; by convention they are written
    /// in UPPER_SNAKE_CASE. Keys should be ASCII 7-bit; values are UTF-8.
    pub comments: Vec<(String, String)>,
}

impl Default for FlacMetaConfig {
    fn default() -> Self {
        Self {
            compression_level: 5,
            bits_per_sample: 16,
            comments: Vec::new(),
        }
    }
}

/// Build the raw binary payload for a VORBIS_COMMENT metadata block (type 4).
///
/// The payload (without the 4-byte FLAC block header) has this layout (all LE):
/// ```text
/// vendor_length: u32 LE
/// vendor_string: [u8]   (UTF-8)
/// comment_count: u32 LE
/// For each comment:
///   length: u32 LE      (byte length of "KEY=value")
///   data:   [u8]        (UTF-8 "KEY=value", no null)
/// ```
fn build_vorbis_comment_payload(comments: &[(String, String)]) -> Vec<u8> {
    const VENDOR: &str = concat!("OxiAudio ", env!("CARGO_PKG_VERSION"));
    let vendor_bytes = VENDOR.as_bytes();

    let mut payload = Vec::new();

    // vendor_length (u32 LE) + vendor_string
    let vlen = vendor_bytes.len() as u32;
    payload.extend_from_slice(&vlen.to_le_bytes());
    payload.extend_from_slice(vendor_bytes);

    // comment_count (u32 LE)
    let count = comments.len() as u32;
    payload.extend_from_slice(&count.to_le_bytes());

    // each KEY=value entry
    for (key, value) in comments {
        let entry = format!("{key}={value}");
        let entry_bytes = entry.as_bytes();
        let elen = entry_bytes.len() as u32;
        payload.extend_from_slice(&elen.to_le_bytes());
        payload.extend_from_slice(entry_bytes);
    }

    payload
}

/// Encode `buf` to FLAC with Vorbis comment metadata.
///
/// The Vorbis comment block (type 4) is inserted immediately after the mandatory
/// STREAMINFO block using `flacenc`'s `Stream::add_metadata_block` API.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on configuration, encode, or I/O failure.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use oxiaudio_encode::{FlacMetaConfig, encode_flac_with_metadata};
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let buf = AudioBuffer {
///     samples: vec![0.0f32; 4096],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
/// let config = FlacMetaConfig {
///     compression_level: 5,
///     bits_per_sample: 16,
///     comments: vec![
///         ("TITLE".to_string(), "Test".to_string()),
///         ("ARTIST".to_string(), "OxiAudio".to_string()),
///     ],
/// };
/// let mut out = Cursor::new(Vec::new());
/// encode_flac_with_metadata(&buf, &mut out, &config).unwrap();
/// let bytes = out.into_inner();
/// assert_eq!(&bytes[..4], b"fLaC");
/// assert!(!bytes.is_empty());
/// ```
#[must_use = "discarding errors ignores encode failure"]
pub fn encode_flac_with_metadata<W: std::io::Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    mut writer: W,
    config: &FlacMetaConfig,
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count();
    let block_size = block_size_for_level(config.compression_level);
    let bits = clamp_flac_bits(config.bits_per_sample);
    let scale = flac_full_scale(bits);

    // Convert f32 samples to signed i32 interleaved PCM at the configured bit depth.
    let pcm: Vec<i32> = buf
        .samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * scale) as i32)
        .collect();

    let source = MemSource::from_samples(&pcm, channels, bits as usize, buf.sample_rate as usize);

    let mut cfg = flacenc::config::Encoder::default();
    cfg.block_size = block_size;
    let cfg = cfg
        .into_verified()
        .map_err(|(_, e)| OxiAudioError::Encode(e.to_string()))?;

    let mut stream = flacenc::encode_with_fixed_block_size(&cfg, source, block_size)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    // Fix up block sizes to ensure the STREAMINFO signals fixed-blocksize mode
    // (same workaround as FlacEncoder).
    stream
        .stream_info_mut()
        .set_block_sizes(block_size, block_size)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    // Build and inject the Vorbis comment block (type 4).
    if !config.comments.is_empty() {
        let payload = build_vorbis_comment_payload(&config.comments);
        let metadata_block = MetadataBlockData::new_unknown(4, &payload)
            .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
        stream.add_metadata_block(metadata_block);
    }

    let mut sink = ByteSink::with_capacity(stream.count_bits());
    stream
        .write(&mut sink)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    writer
        .write_all(sink.as_slice())
        .map_err(OxiAudioError::Io)?;

    Ok(())
}

// ─── SEEKTABLE support ─────────────────────────────────────────────────────────

/// Generate the raw binary payload for a FLAC SEEKTABLE metadata block (type 3).
///
/// Each seekpoint is 18 bytes in big-endian order:
/// ```text
/// sample_number: u64 BE  — 0xFFFFFFFFFFFFFFFF marks a placeholder seekpoint
/// stream_offset: u64 BE  — byte offset of the audio frame (0 for placeholders)
/// frame_samples: u16 BE  — samples per frame at this point (0 for placeholders)
/// ```
///
/// Placeholder seekpoints (sample_number == u64::MAX) are ignored by decoders
/// but reserve space in the FLAC header for future population.
fn build_seektable_payload(n_seekpoints: usize) -> Vec<u8> {
    let mut payload = Vec::with_capacity(n_seekpoints * 18);
    for _ in 0..n_seekpoints {
        payload.extend_from_slice(&u64::MAX.to_be_bytes()); // placeholder marker
        payload.extend_from_slice(&0u64.to_be_bytes()); // stream_offset
        payload.extend_from_slice(&0u16.to_be_bytes()); // frame_samples
    }
    payload
}

/// Encode `buf` to FLAC with a placeholder SEEKTABLE metadata block (type 3).
///
/// Generates one placeholder seekpoint per second of audio
/// (`ceil(total_frames / sample_rate)`, minimum 1).  Placeholder seekpoints
/// (sample_number == `u64::MAX`) are ignored by decoders but reserve space in
/// the FLAC header so tools can populate them after the file is written.
///
/// `compression_level` (0–8) maps to flacenc block sizes via the same table
/// used by [`FlacEncoder`][crate::FlacEncoder].
///
/// # Errors
///
/// Returns [`OxiAudioError`] on configuration, encode, or I/O failure.
///
/// # Examples
///
/// ```
/// use std::io::Cursor;
/// use oxiaudio_encode::encode_flac_with_seektable;
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let buf = AudioBuffer {
///     samples: vec![0.0f32; 4096],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Stereo,
///     format: SampleFormat::F32,
/// };
/// let mut out = Cursor::new(Vec::new());
/// encode_flac_with_seektable(&buf, &mut out, 5).unwrap();
/// let bytes = out.into_inner();
/// assert_eq!(&bytes[..4], b"fLaC");
/// assert!(!bytes.is_empty());
/// ```
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_with_seektable<W: std::io::Write + std::io::Seek>(
    buf: &AudioBuffer<f32>,
    mut writer: W,
    compression_level: u8,
) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count();
    let block_size = block_size_for_level(compression_level);
    let bits = clamp_flac_bits(16); // 16-bit is the standard SEEKTABLE use-case
    let scale = flac_full_scale(bits);

    let pcm: Vec<i32> = buf
        .samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * scale) as i32)
        .collect();

    let source = MemSource::from_samples(&pcm, channels, bits as usize, buf.sample_rate as usize);

    let mut cfg = flacenc::config::Encoder::default();
    cfg.block_size = block_size;
    let cfg = cfg
        .into_verified()
        .map_err(|(_, e)| OxiAudioError::Encode(e.to_string()))?;

    let mut stream = flacenc::encode_with_fixed_block_size(&cfg, source, block_size)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    stream
        .stream_info_mut()
        .set_block_sizes(block_size, block_size)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    // Build and inject the SEEKTABLE block (type 3).
    let total_frames = buf.samples.len().checked_div(channels).unwrap_or(0);
    let seek_granularity = (buf.sample_rate as usize).max(1);
    let n_seekpoints = total_frames.div_ceil(seek_granularity).max(1);
    let payload = build_seektable_payload(n_seekpoints);
    let seektable_block = MetadataBlockData::new_unknown(3, &payload)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;
    stream.add_metadata_block(seektable_block);

    let mut sink = ByteSink::with_capacity(stream.count_bits());
    stream
        .write(&mut sink)
        .map_err(|e| OxiAudioError::Encode(e.to_string()))?;

    writer
        .write_all(sink.as_slice())
        .map_err(OxiAudioError::Io)?;

    Ok(())
}

/// File-based convenience wrapper for [`encode_flac_with_seektable`].
///
/// Creates (or truncates) the file at `path` and writes a FLAC stream with a
/// placeholder SEEKTABLE metadata block.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] on file-creation failure or [`OxiAudioError::Encode`]
/// on encode failure.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_with_seektable_file(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
    compression_level: u8,
) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    let writer = std::io::BufWriter::new(file);
    encode_flac_with_seektable(buf, writer, compression_level)
}

// ─── FLAC MD5 injection ────────────────────────────────────────────────────────

/// Compute the MD5 digest (RFC 1321) of a byte slice.
///
/// This is a self-contained pure-Rust MD5 implementation. No external dependency.
pub(crate) fn md5(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    let mut a0: u32 = 0x6745_2301;
    let mut b0: u32 = 0xefcd_ab89;
    let mut c0: u32 = 0x98ba_dcfe;
    let mut d0: u32 = 0x1032_5476;

    let orig_len = data.len();
    let orig_len_bits = (orig_len as u64).wrapping_mul(8);
    let mut padded: Vec<u8> = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&orig_len_bits.to_le_bytes());

    for chunk in padded.chunks(64) {
        let mut m = [0u32; 16];
        for (i, word) in m.iter_mut().enumerate() {
            let base = i * 4;
            *word = u32::from_le_bytes([
                chunk[base],
                chunk[base + 1],
                chunk[base + 2],
                chunk[base + 3],
            ]);
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0u32..64 {
            let (f, g) = if i < 16 {
                ((b & c) | (!b & d), i)
            } else if i < 32 {
                ((d & b) | (!d & c), (5 * i + 1) % 16)
            } else if i < 48 {
                (b ^ c ^ d, (3 * i + 5) % 16)
            } else {
                (c ^ (b | !d), (7 * i) % 16)
            };
            let f = f
                .wrapping_add(a)
                .wrapping_add(K[i as usize])
                .wrapping_add(m[g as usize]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i as usize]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut digest = [0u8; 16];
    digest[0..4].copy_from_slice(&a0.to_le_bytes());
    digest[4..8].copy_from_slice(&b0.to_le_bytes());
    digest[8..12].copy_from_slice(&c0.to_le_bytes());
    digest[12..16].copy_from_slice(&d0.to_le_bytes());
    digest
}

/// Inject the correct MD5 checksum into a FLAC byte stream's STREAMINFO block.
///
/// The MD5 is computed from the original `AudioBuffer<f32>` samples converted to
/// 16-bit signed little-endian integers (the format used by our 16-bit FLAC paths).
///
/// The function modifies `flac_bytes` in-place. It validates the `fLaC` magic and
/// that the first metadata block is STREAMINFO (type 0) before patching bytes `[34..50]`.
///
/// FLAC binary layout (from the start of the file):
/// ```text
/// [0..4]   fLaC magic
/// [4..8]   block header: last(1b) | type(7b) | size(24b)
/// [8..42]  STREAMINFO payload (34 bytes):
///   [8..10]   min_block_size
///   [10..12]  max_block_size
///   [12..15]  min_frame_size (24-bit BE)
///   [15..18]  max_frame_size (24-bit BE)
///   [18..26]  packed: sample_rate(20b)|ch-1(3b)|bps-1(5b)|total_samples(36b)
///   [26..42]  MD5 signature (16 bytes) → injected here as file bytes [34..50]
/// ```
///
/// # Errors
///
/// Returns [`OxiAudioError::Encode`] if:
/// - The stream is too short (< 50 bytes).
/// - The `fLaC` magic is absent.
/// - The first metadata block is not STREAMINFO (type 0).
#[must_use = "discarding the Result ignores inject errors"]
pub fn inject_flac_md5(
    flac_bytes: &mut [u8],
    original_buf: &oxiaudio_core::AudioBuffer<f32>,
) -> Result<(), oxiaudio_core::OxiAudioError> {
    // Minimum: 4 (magic) + 4 (block header) + 34 (STREAMINFO) + 8 (MD5 field starts at 34, ends at 50)
    if flac_bytes.len() < 50 {
        return Err(oxiaudio_core::OxiAudioError::Encode(
            "FLAC stream too short to contain STREAMINFO MD5 field".to_string(),
        ));
    }
    if &flac_bytes[0..4] != b"fLaC" {
        return Err(oxiaudio_core::OxiAudioError::Encode(
            "not a FLAC stream (missing fLaC magic)".to_string(),
        ));
    }
    // Byte 4: bit 7 = last-metadata-block flag; bits 6..0 = block type.
    let block_type = flac_bytes[4] & 0x7F;
    if block_type != 0 {
        return Err(oxiaudio_core::OxiAudioError::Encode(format!(
            "first FLAC metadata block is not STREAMINFO (type={block_type})"
        )));
    }

    // Convert samples to i16 LE bytes for MD5 (16-bit encoding path).
    let sample_bytes: Vec<u8> = original_buf
        .samples
        .iter()
        .flat_map(|&s| {
            let i = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            i.to_le_bytes()
        })
        .collect();

    let digest = md5(&sample_bytes);

    // STREAMINFO payload starts at byte 8; MD5 is at payload offset 26.
    // File byte offset: 8 + 26 = 34. MD5 occupies bytes [34..50].
    flac_bytes[34..50].copy_from_slice(&digest);
    Ok(())
}

/// Encode an `AudioBuffer<f32>` to FLAC bytes with a correct MD5 in STREAMINFO.
///
/// Uses 16-bit PCM encoding at compression level 5 (the standard FLAC path). The
/// MD5 is computed from the same 16-bit signed LE representation passed to the
/// encoder, then injected into the STREAMINFO block.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on configuration, encode, or injection failure.
///
/// # Examples
///
/// ```
/// use oxiaudio_encode::encode_flac_with_md5;
/// use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
///
/// let buf = AudioBuffer {
///     samples: (0..44_100)
///         .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.5)
///         .collect(),
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
/// let bytes = encode_flac_with_md5(&buf).unwrap();
/// assert_eq!(&bytes[..4], b"fLaC");
/// assert!(bytes[34..50].iter().any(|&b| b != 0), "MD5 must be non-zero");
/// ```
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_with_md5(
    buf: &oxiaudio_core::AudioBuffer<f32>,
) -> Result<Vec<u8>, oxiaudio_core::OxiAudioError> {
    use std::io::Cursor;
    let mut out: Vec<u8> = Vec::new();
    {
        let cursor = Cursor::new(&mut out);
        encode_flac_with_seektable(buf, cursor, 5)?;
    }
    inject_flac_md5(&mut out, buf)?;
    Ok(out)
}

/// Encode an `AudioBuffer<f32>` to a FLAC file with a correct MD5 in STREAMINFO.
///
/// Creates or truncates the file at `path`. The MD5 is computed from the 16-bit
/// signed LE representation and injected after encoding.
///
/// # Errors
///
/// Returns [`OxiAudioError`] on file creation, encode, or injection failure.
#[must_use = "discarding the Result ignores encode errors"]
pub fn encode_flac_with_md5_file(
    buf: &oxiaudio_core::AudioBuffer<f32>,
    path: &std::path::Path,
) -> Result<(), oxiaudio_core::OxiAudioError> {
    let bytes = encode_flac_with_md5(buf)?;
    std::fs::write(path, &bytes).map_err(oxiaudio_core::OxiAudioError::Io)
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    use super::{
        build_seektable_payload, encode_flac_with_md5, encode_flac_with_md5_file,
        encode_flac_with_metadata, encode_flac_with_seektable, encode_flac_with_seektable_file,
        inject_flac_md5, md5, FlacMetaConfig,
    };

    fn make_buf(samples: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.0f32; samples],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_flac_metadata_vorbis_comment_roundtrip() {
        let buf = make_buf(4096);
        let config = FlacMetaConfig {
            compression_level: 5,
            bits_per_sample: 16,
            comments: vec![
                ("TITLE".to_string(), "Test".to_string()),
                ("ARTIST".to_string(), "OxiAudio".to_string()),
            ],
        };
        let mut out = Cursor::new(Vec::new());
        encode_flac_with_metadata(&buf, &mut out, &config)
            .expect("encode_flac_with_metadata should succeed");
        let bytes = out.into_inner();
        assert!(!bytes.is_empty(), "output must not be empty");
        assert_eq!(&bytes[..4], b"fLaC", "output must start with fLaC marker");
    }

    #[test]
    fn test_flac_meta_config_vorbis() {
        // Verify that the Vorbis comment payload is embedded (look for "TITLE=Test" in bytes)
        let buf = make_buf(4096);
        let config = FlacMetaConfig {
            compression_level: 3,
            bits_per_sample: 24,
            comments: vec![
                ("TITLE".to_string(), "Test".to_string()),
                ("ARTIST".to_string(), "OxiAudio".to_string()),
            ],
        };
        let mut out = Cursor::new(Vec::new());
        encode_flac_with_metadata(&buf, &mut out, &config).expect("encode_flac_with_metadata");
        let bytes = out.into_inner();
        assert_eq!(&bytes[..4], b"fLaC");

        // The Vorbis comment text "TITLE=Test" must appear somewhere in the FLAC stream.
        let needle = b"TITLE=Test";
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(
            found,
            "Vorbis comment 'TITLE=Test' must be present in the FLAC stream"
        );
    }

    #[test]
    fn test_flac_meta_config_no_comments() {
        // With no comments, encoding should still succeed and produce a valid fLaC file.
        let buf = make_buf(2048);
        let config = FlacMetaConfig {
            compression_level: 5,
            bits_per_sample: 16,
            comments: vec![],
        };
        let mut out = Cursor::new(Vec::new());
        encode_flac_with_metadata(&buf, &mut out, &config)
            .expect("encode_flac_with_metadata with no comments should succeed");
        let bytes = out.into_inner();
        assert_eq!(&bytes[..4], b"fLaC");
    }

    // ─── SEEKTABLE tests ───────────────────────────────────────────────────────

    #[test]
    fn test_seektable_payload_structure() {
        // 1 seekpoint = exactly 18 bytes; first 8 bytes must be the placeholder marker.
        let payload = build_seektable_payload(1);
        assert_eq!(payload.len(), 18, "1 seekpoint must be 18 bytes");
        let marker = u64::from_be_bytes(payload[..8].try_into().expect("slice"));
        assert_eq!(
            marker,
            u64::MAX,
            "first 8 bytes must be placeholder marker 0xFFFF…"
        );
        // stream_offset and frame_samples must both be zero
        let stream_offset = u64::from_be_bytes(payload[8..16].try_into().expect("slice"));
        let frame_samples = u16::from_be_bytes(payload[16..18].try_into().expect("slice"));
        assert_eq!(stream_offset, 0, "stream_offset must be 0 for placeholders");
        assert_eq!(frame_samples, 0, "frame_samples must be 0 for placeholders");
    }

    #[test]
    fn test_seektable_multiple_seekpoints() {
        // 3 seekpoints = 54 bytes
        let payload = build_seektable_payload(3);
        assert_eq!(payload.len(), 54, "3 seekpoints must be 54 bytes");
        // Every seekpoint must have the placeholder marker
        for i in 0..3 {
            let offset = i * 18;
            let marker = u64::from_be_bytes(payload[offset..offset + 8].try_into().expect("slice"));
            assert_eq!(
                marker,
                u64::MAX,
                "seekpoint {i} must have placeholder marker"
            );
        }
    }

    #[test]
    fn test_encode_flac_with_seektable_creates_file() {
        // 2s stereo 44100 Hz buffer → must produce a valid fLaC file
        let buf = AudioBuffer {
            samples: vec![0.0f32; 44_100 * 2 * 2], // 2 seconds, stereo
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let tmp = std::env::temp_dir().join("test_seektable.flac");
        encode_flac_with_seektable_file(&buf, &tmp, 5)
            .expect("encode_flac_with_seektable_file should succeed");
        let bytes = std::fs::read(&tmp).expect("temp file should exist");
        assert!(!bytes.is_empty(), "output file must not be empty");
        assert_eq!(&bytes[..4], b"fLaC", "output must start with fLaC marker");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_encode_flac_with_seektable_cursor() {
        // Test the in-memory writer path
        let buf = AudioBuffer {
            samples: vec![0.0f32; 44_100 * 2 * 2],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let mut out = Cursor::new(Vec::new());
        encode_flac_with_seektable(&buf, &mut out, 5)
            .expect("encode_flac_with_seektable should succeed");
        let bytes = out.into_inner();
        assert!(!bytes.is_empty(), "output must not be empty");
        assert_eq!(&bytes[..4], b"fLaC", "output must start with fLaC marker");
    }

    // ─── MD5 tests ─────────────────────────────────────────────────────────────

    /// RFC 1321 test vector: MD5("") = d41d8cd98f00b204e9800998ecf8427e
    #[test]
    fn test_md5_known_value_empty() {
        let digest = md5(b"");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex, "d41d8cd98f00b204e9800998ecf8427e",
            "MD5(\"\") must match RFC 1321"
        );
    }

    /// RFC 1321 test vector: MD5("abc") = 900150983cd24fb0d6963f7d28e17f72
    #[test]
    fn test_md5_known_value_abc() {
        let digest = md5(b"abc");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex, "900150983cd24fb0d6963f7d28e17f72",
            "MD5(\"abc\") must match RFC 1321"
        );
    }

    /// MD5("Hello, World!") must produce the well-known digest.
    #[test]
    fn test_md5_hello_world() {
        let digest = md5(b"Hello, World!");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex, "65a8e27d8879283831b664bd8b7f0ad4",
            "MD5(\"Hello, World!\") digest mismatch"
        );
    }

    /// Encoding a 0.5 s sine wave to FLAC, then injecting the MD5, must produce
    /// non-zero bytes at the STREAMINFO MD5 field ([34..50]).
    #[test]
    fn test_inject_flac_md5_non_zero() {
        let n = 44_100 / 2; // 0.5 seconds mono
        let buf = AudioBuffer {
            samples: (0..n)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.5)
                .collect(),
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let bytes = encode_flac_with_md5(&buf).expect("encode_flac_with_md5 must succeed");
        assert_eq!(&bytes[..4], b"fLaC", "must start with fLaC magic");
        // MD5 field must be non-zero for a non-silent signal
        assert!(
            bytes[34..50].iter().any(|&b| b != 0),
            "STREAMINFO MD5 field must be non-zero for a 440 Hz sine wave"
        );
    }

    /// Verify the MD5 bytes are consistent: re-computing the MD5 of the same
    /// i16-LE bytes must yield exactly the value stored in the FLAC STREAMINFO.
    #[test]
    fn test_inject_flac_md5_consistent() {
        let n = 44_100 / 4;
        let buf = AudioBuffer {
            samples: (0..n)
                .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 44_100.0).sin() * 0.3)
                .collect(),
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let bytes = encode_flac_with_md5(&buf).expect("encode_flac_with_md5 must succeed");

        // Reconstruct i16 LE bytes the same way inject_flac_md5 does.
        let sample_bytes: Vec<u8> = buf
            .samples
            .iter()
            .flat_map(|&s| {
                let i = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                i.to_le_bytes()
            })
            .collect();
        let expected = md5(&sample_bytes);
        assert_eq!(
            &bytes[34..50],
            &expected,
            "STREAMINFO MD5 must match re-computed digest"
        );
    }

    /// inject_flac_md5 must return Err on data with a wrong magic.
    #[test]
    fn test_inject_flac_md5_bad_magic() {
        let buf = make_buf(256);
        let mut bad = vec![0u8; 60];
        bad[0..4].copy_from_slice(b"JUNK");
        let result = inject_flac_md5(&mut bad, &buf);
        assert!(result.is_err(), "must fail on bad fLaC magic");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("fLaC") || msg.contains("magic"),
            "error message must mention fLaC magic, got: {msg}"
        );
    }

    /// inject_flac_md5 must return Err on streams that are too short.
    #[test]
    fn test_inject_flac_md5_too_short() {
        let buf = make_buf(256);
        let mut short = vec![b'f', b'L', b'a', b'C', 0x00, 0x00, 0x00, 0x00];
        let result = inject_flac_md5(&mut short, &buf);
        assert!(result.is_err(), "must fail on stream shorter than 50 bytes");
    }

    /// encode_flac_with_md5_file must produce a valid fLaC file on disk.
    #[test]
    fn test_encode_flac_with_md5_file() {
        let n = 44_100 / 2;
        let buf = AudioBuffer {
            samples: (0..n)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin() * 0.4)
                .collect(),
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let tmp = std::env::temp_dir().join("test_flac_md5.flac");
        encode_flac_with_md5_file(&buf, &tmp).expect("encode_flac_with_md5_file must succeed");
        let bytes = std::fs::read(&tmp).expect("temp file must exist");
        assert_eq!(&bytes[..4], b"fLaC", "file must start with fLaC magic");
        assert!(
            bytes[34..50].iter().any(|&b| b != 0),
            "STREAMINFO MD5 field in file must be non-zero"
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
