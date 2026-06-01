use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
use std::path::PathBuf;

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

/// Build a minimal AIFF file in memory with optional NAME/AUTH text chunks.
fn make_aiff_with_metadata(title: &str, artist: &str) -> Vec<u8> {
    let mut out = Vec::new();

    // COMM chunk payload (18 bytes): 1 channel, 4410 frames, 16-bit, 44100 Hz
    let comm_data: Vec<u8> = {
        let mut v = Vec::new();
        v.extend_from_slice(&1u16.to_be_bytes()); // num_channels
        v.extend_from_slice(&4410u32.to_be_bytes()); // num_sample_frames
        v.extend_from_slice(&16u16.to_be_bytes()); // bit_depth
                                                   // 44100 Hz as 80-bit extended: precomputed bytes
        v.extend_from_slice(&[0x40, 0x0E, 0xAC, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        v
    };

    let pad_str = |s: &str| -> Vec<u8> {
        let mut v = s.as_bytes().to_vec();
        // even-boundary pad per AIFF spec (pad *payload* to even)
        if v.len() % 2 != 0 {
            v.push(0);
        }
        v
    };

    let title_padded = pad_str(title);
    let artist_padded = pad_str(artist);

    // PCM data: 4410 silent 16-bit samples = 8820 bytes
    let pcm: Vec<u8> = vec![0u8; 4410 * 2];

    // SSND chunk payload: 4-byte offset + 4-byte blockAlign + PCM
    let ssnd_data_size = 8u32 + pcm.len() as u32;

    // Build chunk list
    let form_data_size = 4 // "AIFF"
        + 8 + 18 // COMM header + data
        + 8 + title_padded.len() // NAME
        + 8 + artist_padded.len() // AUTH
        + 8 + ssnd_data_size as usize; // SSND

    out.extend_from_slice(b"FORM");
    out.extend_from_slice(&(form_data_size as u32).to_be_bytes());
    out.extend_from_slice(b"AIFF");

    out.extend_from_slice(b"COMM");
    out.extend_from_slice(&18u32.to_be_bytes());
    out.extend_from_slice(&comm_data);

    out.extend_from_slice(b"NAME");
    // Write the UNPADDED size in the chunk header (actual content length)
    let title_raw = title.as_bytes();
    out.extend_from_slice(&(title_raw.len() as u32).to_be_bytes());
    out.extend_from_slice(title_raw);
    // Pad to even boundary after the chunk
    if title_raw.len() % 2 != 0 {
        out.push(0);
    }

    out.extend_from_slice(b"AUTH");
    let artist_raw = artist.as_bytes();
    out.extend_from_slice(&(artist_raw.len() as u32).to_be_bytes());
    out.extend_from_slice(artist_raw);
    if artist_raw.len() % 2 != 0 {
        out.push(0);
    }

    out.extend_from_slice(b"SSND");
    out.extend_from_slice(&ssnd_data_size.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes()); // offset
    out.extend_from_slice(&0u32.to_be_bytes()); // blockAlign
    out.extend_from_slice(&pcm);

    // Recompute FORM size correctly based on actual out length
    // (the header is 8 bytes: "FORM" + size), so FORM payload = total - 8
    let actual_form_payload = out.len() - 8;
    let size_bytes = (actual_form_payload as u32).to_be_bytes();
    out[4] = size_bytes[0];
    out[5] = size_bytes[1];
    out[6] = size_bytes[2];
    out[7] = size_bytes[3];

    out
}

#[test]
fn test_aiff_metadata_name_auth() {
    let bytes = make_aiff_with_metadata("My Song", "My Artist");
    let path = temp("oxiaudio_m10_aiff_meta.aiff");
    std::fs::write(&path, &bytes).expect("write");

    let (buf, meta) =
        oxiaudio_decode::decode_aiff_with_metadata(&path).expect("decode_aiff_with_metadata");
    let _ = std::fs::remove_file(&path);

    assert_eq!(buf.sample_rate, 44_100);
    assert_eq!(
        meta.title.as_deref(),
        Some("My Song"),
        "title should be read from NAME chunk"
    );
    assert_eq!(
        meta.artist.as_deref(),
        Some("My Artist"),
        "artist should be read from AUTH chunk"
    );
}

#[test]
fn test_decode_aiff_basic_still_works() {
    // Verify the non-metadata AIFF decode path still works
    let bytes = make_aiff_with_metadata("", "");
    let path = temp("oxiaudio_m10_aiff_basic.aiff");
    std::fs::write(&path, &bytes).expect("write");
    let buf = oxiaudio_decode::decode_aiff_file(&path).expect("decode_aiff_file");
    let _ = std::fs::remove_file(&path);
    assert_eq!(buf.sample_rate, 44_100);
    assert_eq!(buf.channels.channel_count(), 1);
}

#[test]
fn test_wav_extensible_quad_decode() {
    use oxiaudio_core::AudioEncoder;
    use oxiaudio_encode::WavEncoder;
    use std::io::BufWriter;

    // Create a 4-channel WAV using the encoder
    let n = 1024 * 4; // 1024 frames × 4 channels
    let samples = vec![0.1f32; n];
    let buf_in = AudioBuffer {
        samples,
        sample_rate: 44_100,
        channels: ChannelLayout::Quad,
        format: SampleFormat::F32,
    };
    let path = temp("oxiaudio_m10_wav_quad.wav");
    {
        let file = std::fs::File::create(&path).expect("create");
        let mut enc = WavEncoder::default();
        enc.encode(&buf_in, BufWriter::new(file))
            .expect("encode quad");
    }
    // Now decode it back
    let buf_out = oxiaudio_decode::decode_file(&path).expect("decode quad wav");
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        buf_out.channels.channel_count(),
        4,
        "should decode 4 channels"
    );
    assert_eq!(buf_out.sample_rate, 44_100);
}
