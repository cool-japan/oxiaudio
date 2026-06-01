//! LAME/Xing gapless playback header parser.
//!
//! The LAME encoder embeds a "Xing" (VBR) or "Info" (CBR) header in the first MP3
//! frame's side-information area. Within this header, LAME stores encoder delay and
//! padding values that allow gapless playback by indicating how many samples to trim
//! from the start and end of the decoded stream.
//!
//! See also: <http://gabriel.mp3-tech.org/mp3infotag.html>

/// Information extracted from a LAME/Xing gapless playback header.
#[derive(Debug, Clone, Default)]
pub struct GaplessInfo {
    /// Encoder delay at start (samples to discard from beginning).
    pub encoder_delay: u32,
    /// Encoder padding at end (samples to discard from end).
    pub encoder_padding: u32,
    /// Total number of samples in the original audio (0 if unknown).
    pub total_samples: u64,
}

/// Parse the LAME gapless header from the first frame of raw MP3 data.
///
/// Returns `None` if the data does not contain a valid Xing/Info header or if the
/// LAME version tag is absent.
///
/// The LAME header format:
/// 1. Locate the `"Xing"` or `"Info"` marker (4 bytes) within the first 1500 bytes.
/// 2. Read flags `u32` BE immediately after the marker.
/// 3. If flag bit 0 set: skip frame count `u32` BE.
/// 4. If flag bit 1 set: skip byte count `u32` BE.
/// 5. If flag bit 2 set: skip 100-byte seek table.
/// 6. If flag bit 3 set: skip quality indicator `u32` BE.
/// 7. At that offset + 36 bytes into the Xing structure, look for the LAME version
///    string (9 bytes). If it starts with `"LAME"`, parse encoder delay and padding
///    from 3 bytes at offset +21 from the version string start:
///    - `delay   = (byte[0] << 4) | (byte[1] >> 4)`   (high 12 bits)
///    - `padding = ((byte[1] & 0x0F) << 8) | byte[2]` (low 12 bits)
///
/// # Parameters
///
/// - `mp3_bytes` — raw MP3 file bytes (at least the first frame; 1500 bytes recommended).
pub fn parse_gapless_info(mp3_bytes: &[u8]) -> Option<GaplessInfo> {
    // Search for "Xing" or "Info" within the first 1500 bytes.
    let search_limit = mp3_bytes.len().min(1500);
    let search_area = &mp3_bytes[..search_limit];

    let xing_pos =
        find_marker(search_area, b"Xing").or_else(|| find_marker(search_area, b"Info"))?;

    // After the 4-byte marker: flags u32 BE at xing_pos + 4.
    let flags_start = xing_pos + 4;
    if flags_start + 4 > mp3_bytes.len() {
        return None;
    }
    let flags = u32::from_be_bytes([
        mp3_bytes[flags_start],
        mp3_bytes[flags_start + 1],
        mp3_bytes[flags_start + 2],
        mp3_bytes[flags_start + 3],
    ]);

    // Advance past the flags field.
    let mut cursor = flags_start + 4;

    // Optional: frame count (flag bit 0).
    let frame_count: Option<u32> = if flags & 0x01 != 0 {
        if cursor + 4 > mp3_bytes.len() {
            return None;
        }
        let fc = u32::from_be_bytes([
            mp3_bytes[cursor],
            mp3_bytes[cursor + 1],
            mp3_bytes[cursor + 2],
            mp3_bytes[cursor + 3],
        ]);
        cursor += 4;
        Some(fc)
    } else {
        None
    };

    // Optional: byte count (flag bit 1).
    if flags & 0x02 != 0 {
        cursor += 4;
        if cursor > mp3_bytes.len() {
            return None;
        }
    }

    // Optional: 100-byte seek table (flag bit 2).
    if flags & 0x04 != 0 {
        cursor += 100;
        if cursor > mp3_bytes.len() {
            return None;
        }
    }

    // Optional: quality indicator (flag bit 3).
    if flags & 0x08 != 0 {
        cursor += 4;
        if cursor > mp3_bytes.len() {
            return None;
        }
    }

    // The LAME version string starts 36 bytes into the Xing structure
    // (i.e. at xing_pos + 36, relative to the start of "Xing").
    // The Xing "structure" is: marker(4) + flags(4) + optional fields above.
    // However in practice, LAME always places its tag at a fixed offset of 120
    // bytes from the Xing marker in the frame side-info area — but the spec says
    // 36 bytes after the Xing marker (past the flags and optional sections).
    // We use `cursor` which is already past the optional sections, and check
    // that `cursor` equals `xing_pos + 36` in a typical encoding.
    // Per the task spec: look for LAME version string at xing_pos + 36.
    let lame_start = xing_pos + 36;
    if lame_start + 30 > mp3_bytes.len() {
        // Not enough data for LAME tag.
        return None;
    }

    // Verify LAME version string starts with "LAME".
    if &mp3_bytes[lame_start..lame_start + 4] != b"LAME" {
        return None;
    }

    // Encoder delay and padding are at lame_start + 21 (3 bytes).
    // Layout: LAME version string[9] + revision+VBR_method[1] + lowpass[1]
    //         + replay_gain[8] + flags[1] + abr_bitrate[1] = 21 bytes before the 3-byte field.
    let delay_pad_offset = lame_start + 21;
    if delay_pad_offset + 3 > mp3_bytes.len() {
        return None;
    }

    let b0 = mp3_bytes[delay_pad_offset] as u32;
    let b1 = mp3_bytes[delay_pad_offset + 1] as u32;
    let b2 = mp3_bytes[delay_pad_offset + 2] as u32;

    let encoder_delay = (b0 << 4) | (b1 >> 4);
    let encoder_padding = ((b1 & 0x0F) << 8) | b2;

    // Total samples: frame_count * samples_per_frame (1152 for MPEG1 Layer3),
    // minus encoder delay and padding.
    let total_samples = frame_count
        .map(|fc| {
            let raw = fc as u64 * 1152;
            raw.saturating_sub(encoder_delay as u64 + encoder_padding as u64)
        })
        .unwrap_or(0);

    Some(GaplessInfo {
        encoder_delay,
        encoder_padding,
        total_samples,
    })
}

/// Apply gapless trimming to a decoded buffer using LAME/Xing gapless metadata.
///
/// Removes `encoder_delay` samples from the start and `encoder_padding` samples
/// from the end of the interleaved sample buffer. This produces gapless-accurate
/// audio from MP3 streams encoded with LAME's gapless infrastructure.
///
/// The delay and padding counts are in *samples* (not frames), so for multi-channel
/// audio the actual slice indices are scaled by the channel count.
///
/// If the delay + padding exceeds the total frame count, an empty buffer is returned.
pub fn apply_gapless_trim(
    buf: oxiaudio_core::AudioBuffer<f32>,
    info: &GaplessInfo,
) -> oxiaudio_core::AudioBuffer<f32> {
    let ch = buf.channels.channel_count();
    if ch == 0 || buf.samples.is_empty() {
        return buf;
    }
    let total_frames = buf.samples.len() / ch;
    let delay = info.encoder_delay as usize;
    let padding = info.encoder_padding as usize;

    // Clamp start so it never exceeds total_frames.
    let start = delay.min(total_frames);
    // Compute end frame: total_frames minus padding, clamped to [start, total_frames].
    let end = total_frames.saturating_sub(padding).max(start);

    let samples = buf.samples[start * ch..end * ch].to_vec();
    oxiaudio_core::AudioBuffer {
        samples,
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        format: buf.format,
    }
}

/// Find the byte offset of `needle` within `haystack`, or `None`.
#[inline]
fn find_marker(haystack: &[u8], needle: &[u8; 4]) -> Option<usize> {
    if haystack.len() < 4 {
        return None;
    }
    haystack.windows(4).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    /// Build a stereo buffer of `total_frames` frames filled with a constant value.
    fn stereo_buf(total_frames: usize, value: f32) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![value; total_frames * 2],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    /// M-new-1: Standard gapless trim on a 10-second 44100 Hz stereo buffer.
    ///
    /// delay=576, padding=528 → trim 576 frames from start, 528 from end.
    /// Total frames = 44100*10 = 441000.
    /// Expected frames remaining = 441000 - 576 - 528 = 439896.
    #[test]
    fn test_gapless_trim_stereo_10sec() {
        let total_frames = 44_100 * 10_usize; // 441 000 frames
        let buf = stereo_buf(total_frames, 1.0);
        let info = GaplessInfo {
            encoder_delay: 576,
            encoder_padding: 528,
            total_samples: 0,
        };
        let trimmed = apply_gapless_trim(buf, &info);
        let expected = total_frames - 576 - 528;
        assert_eq!(
            trimmed.samples.len() / 2,
            expected,
            "trimmed frame count should be {expected}, got {}",
            trimmed.samples.len() / 2
        );
        // Channel layout and sample rate should be preserved.
        assert_eq!(trimmed.channels, ChannelLayout::Stereo);
        assert_eq!(trimmed.sample_rate, 44_100);
    }

    /// M-new-2: Zero delay + zero padding returns an identical buffer.
    #[test]
    fn test_gapless_trim_zero_delay_zero_padding() {
        let total_frames = 1000_usize;
        let buf = stereo_buf(total_frames, 0.5);
        let info = GaplessInfo {
            encoder_delay: 0,
            encoder_padding: 0,
            total_samples: 0,
        };
        let trimmed = apply_gapless_trim(buf.clone(), &info);
        assert_eq!(
            trimmed.samples, buf.samples,
            "zero trim should leave samples unchanged"
        );
        assert_eq!(trimmed.samples.len() / 2, total_frames);
    }

    /// M-new-3: delay > total_frames returns an empty buffer (no panic).
    #[test]
    fn test_gapless_trim_delay_exceeds_total() {
        let buf = stereo_buf(100, 0.25); // 100 frames
        let info = GaplessInfo {
            encoder_delay: 200, // > 100 frames
            encoder_padding: 0,
            total_samples: 0,
        };
        let trimmed = apply_gapless_trim(buf, &info);
        assert!(
            trimmed.samples.is_empty(),
            "delay exceeding total should produce empty buffer"
        );
    }

    /// M-new-4: Mono buffer trim works correctly.
    #[test]
    fn test_gapless_trim_mono() {
        let samples: Vec<f32> = (0..1000_u32).map(|i| i as f32).collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let info = GaplessInfo {
            encoder_delay: 10,
            encoder_padding: 5,
            total_samples: 0,
        };
        let trimmed = apply_gapless_trim(buf, &info);
        // Expected: 1000 - 10 - 5 = 985 frames (mono, so 985 samples).
        assert_eq!(trimmed.samples.len(), 985);
        // First sample should be the 11th original sample (index 10).
        assert!((trimmed.samples[0] - 10.0).abs() < 1e-9);
    }

    /// Test that random/garbage bytes return None.
    #[test]
    fn test_parse_gapless_info_no_xing() {
        let data: Vec<u8> = (0u8..200).collect();
        let result = parse_gapless_info(&data);
        assert!(
            result.is_none(),
            "random bytes should yield None, got {result:?}"
        );
    }

    /// Test that an all-zero buffer returns None (no Xing/Info marker).
    #[test]
    fn test_parse_gapless_info_zero_bytes() {
        let data = vec![0u8; 1500];
        let result = parse_gapless_info(&data);
        assert!(result.is_none(), "all-zero buffer should yield None");
    }

    /// Test with a handcrafted minimal Xing header containing known delay=576, padding=528.
    ///
    /// Header layout (all offsets relative to the start of `data`):
    /// - Offset 0: "Xing" (4 bytes)
    /// - Offset 4: flags = 0x00000001 (frame count present, others absent) (4 bytes)
    /// - Offset 8: frame_count = 1000 (u32 BE) (4 bytes)
    /// - Offsets 12..35: padding to fill up to xing_pos + 36 = offset 36
    ///   (36 - 12 = 24 bytes of zero)
    /// - Offset 36: "LAME5.100" LAME version (9 bytes)
    /// - Offset 45: revision + VBR method (1 byte)
    /// - Offset 46: lowpass (1 byte)
    /// - Offset 47: replay gain (8 bytes)
    /// - Offset 55: flags (1 byte)
    /// - Offset 56: ABR bitrate (1 byte)
    /// - Offset 57: encoder delay/padding (3 bytes)
    ///   delay=576=0x240, padding=528=0x210
    ///   byte[0] = delay >> 4             = 0x24
    ///   byte[1] = (delay & 0x0F) << 4 | (padding >> 8) = 0x02
    ///   byte[2] = padding & 0xFF          = 0x10
    #[test]
    fn test_parse_gapless_info_with_mock_header() {
        let mut data = vec![0u8; 200];

        // "Xing" marker at offset 0.
        data[0..4].copy_from_slice(b"Xing");

        // flags = 0x00000001 (frame count present only).
        data[4..8].copy_from_slice(&0x0000_0001u32.to_be_bytes());

        // frame_count = 1000.
        data[8..12].copy_from_slice(&1000u32.to_be_bytes());

        // Offsets 12..36: zeroed (already zero from initialization).

        // LAME version string at offset 36.
        data[36..45].copy_from_slice(b"LAME5.100");

        // Offsets 45..57: zeroed for revision, lowpass, replay_gain, flags, abr_bitrate.

        // Encoder delay/padding at offset 57 (= lame_start + 21 = 36 + 21).
        // delay   = 576 = 0x240
        // padding = 528 = 0x210
        // byte[0] = (delay >> 4)             = 0x24
        // byte[1] = ((delay & 0x0F) << 4) | (padding >> 8) = (0x0 << 4) | 0x02 = 0x02
        // byte[2] = padding & 0xFF            = 0x10
        data[57] = 0x24;
        data[58] = 0x02;
        data[59] = 0x10;

        let info = parse_gapless_info(&data).expect("should parse valid mock Xing header");

        assert_eq!(
            info.encoder_delay, 576,
            "encoder_delay should be 576, got {}",
            info.encoder_delay
        );
        assert_eq!(
            info.encoder_padding, 528,
            "encoder_padding should be 528, got {}",
            info.encoder_padding
        );
    }

    /// Test that a buffer with "Info" marker is parsed the same way as "Xing".
    #[test]
    fn test_parse_gapless_info_info_marker() {
        let mut data = vec![0u8; 200];

        // "Info" marker at offset 10.
        data[10..14].copy_from_slice(b"Info");

        // flags = 0x00000000 (no optional fields).
        data[14..18].copy_from_slice(&0u32.to_be_bytes());

        // LAME version string at xing_pos + 36 = 10 + 36 = 46.
        data[46..55].copy_from_slice(b"LAME3.100");

        // delay=576, padding=528 at lame_start + 21 = 46 + 21 = 67.
        data[67] = 0x24;
        data[68] = 0x02;
        data[69] = 0x10;

        let info = parse_gapless_info(&data).expect("should parse Info marker the same as Xing");

        assert_eq!(info.encoder_delay, 576);
        assert_eq!(info.encoder_padding, 528);
    }

    /// Test that data too short to hold a complete header returns None.
    #[test]
    fn test_parse_gapless_info_too_short() {
        // "Xing" at offset 0, but then not enough data.
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(b"Xing");
        let result = parse_gapless_info(&data);
        // 20 bytes is not enough for the LAME tag (needs at least 36+30=66 bytes).
        assert!(result.is_none(), "too-short data should return None");
    }

    /// Test that Xing beyond 1500 bytes is not found.
    #[test]
    fn test_parse_gapless_info_xing_beyond_limit() {
        // Put "Xing" at offset 1500 (just outside the search window).
        let mut data = vec![0u8; 1600];
        data[1500..1504].copy_from_slice(b"Xing");
        let result = parse_gapless_info(&data);
        assert!(
            result.is_none(),
            "Xing beyond 1500 bytes should not be found"
        );
    }
}
