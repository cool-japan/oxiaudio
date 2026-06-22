//! M4A/MP4 container writer for AAC audio.
//!
//! Implements the minimal box structure required for a valid M4A file:
//! ftyp + moov (mvhd/trak/mdia/minf/stbl) + mdat.
//!
//! References: ISO 14496-12 (base media file format), ISO 14496-3 (AAC audio in MP4).

use std::io::Write;
use std::path::Path;

use oxiaudio_core::{AudioBuffer, OxiAudioError};

use crate::aac::{encode_aac, sampling_freq_index};

// Box type tags (4 bytes each)
const BOX_FTYP: &[u8; 4] = b"ftyp";
const BOX_MOOV: &[u8; 4] = b"moov";
const BOX_MVHD: &[u8; 4] = b"mvhd";
const BOX_TRAK: &[u8; 4] = b"trak";
const BOX_TKHD: &[u8; 4] = b"tkhd";
const BOX_MDIA: &[u8; 4] = b"mdia";
const BOX_MDHD: &[u8; 4] = b"mdhd";
const BOX_HDLR: &[u8; 4] = b"hdlr";
const BOX_MINF: &[u8; 4] = b"minf";
const BOX_SMHD: &[u8; 4] = b"smhd";
const BOX_DINF: &[u8; 4] = b"dinf";
const BOX_DREF: &[u8; 4] = b"dref";
const BOX_URL: &[u8; 4] = b"url ";
const BOX_STBL: &[u8; 4] = b"stbl";
const BOX_STSD: &[u8; 4] = b"stsd";
const BOX_MP4A: &[u8; 4] = b"mp4a";
const BOX_ESDS: &[u8; 4] = b"esds";
const BOX_STTS: &[u8; 4] = b"stts";
const BOX_STSC: &[u8; 4] = b"stsc";
const BOX_STSZ: &[u8; 4] = b"stsz";
const BOX_STCO: &[u8; 4] = b"stco";
const BOX_MDAT: &[u8; 4] = b"mdat";

/// Write a box header: 4-byte size (big-endian) + 4-byte type.
fn write_box_header(w: &mut Vec<u8>, size: u32, box_type: &[u8; 4]) {
    w.extend_from_slice(&size.to_be_bytes());
    w.extend_from_slice(box_type);
}

/// Write a 32-bit big-endian integer.
fn write_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_be_bytes());
}

/// Write a 16-bit big-endian integer.
fn write_u16(w: &mut Vec<u8>, v: u16) {
    w.extend_from_slice(&v.to_be_bytes());
}

/// Return type alias for `collect_frames` to avoid complex nested generic types.
type AdtsFramePair = (Vec<Vec<u8>>, Vec<Vec<u8>>);

/// Collect raw ADTS frames from the AAC encoder, stripping the 7-byte ADTS header from each.
///
/// Returns `(adts_frames, raw_aac_payloads)` where `raw_aac_payloads` is each frame without
/// the ADTS header (needed for mdat in M4A).
fn collect_frames(buf: &AudioBuffer<f32>) -> Result<AdtsFramePair, OxiAudioError> {
    // Encode to a memory buffer, then parse ADTS frames
    let mut encoded = Vec::new();
    encode_aac(buf, &mut encoded)?;

    let bytes = encoded;
    let mut adts_frames = Vec::new();
    let mut raw_frames = Vec::new();
    let mut pos = 0usize;

    while pos + 7 <= bytes.len() {
        // ADTS sync: 0xFF 0xFx (protection_absent may vary)
        if bytes[pos] != 0xFF || (bytes[pos + 1] & 0xF0) != 0xF0 {
            break;
        }
        // Extract 13-bit frame length from bits 30-42 of the 56-bit header
        // byte[3] bits [1:0] -> len[12:11], byte[4] -> len[10:3], byte[5] bits [7:5] -> len[2:0]
        let len_hi = ((bytes[pos + 3] & 0x03) as usize) << 11;
        let len_mid = (bytes[pos + 4] as usize) << 3;
        let len_lo = ((bytes[pos + 5] >> 5) & 0x07) as usize;
        let frame_len = len_hi | len_mid | len_lo;

        if frame_len < 7 || pos + frame_len > bytes.len() {
            break;
        }

        adts_frames.push(bytes[pos..pos + frame_len].to_vec());
        // Strip the 7-byte ADTS header for the raw M4A payload
        raw_frames.push(bytes[pos + 7..pos + frame_len].to_vec());
        pos += frame_len;
    }

    Ok((adts_frames, raw_frames))
}

/// Build the ESDS (Elementary Stream Descriptor) box body for AAC-LC.
///
/// The body starts with version+flags (4 bytes), followed by the ES_Descriptor
/// hierarchy containing AudioSpecificConfig.
#[allow(clippy::cast_possible_truncation)]
fn build_esds_body(channels: u8, sfi: u8) -> Vec<u8> {
    // AudioSpecificConfig (2 bytes): object_type=2 (AAC-LC), sfi, channel_config
    // Bit layout: [5: object_type=2][4: sfi][4: channel_config][1: frameLen=0][1: dep=0][1: ext=0]
    let asc_byte0: u8 = (2u8 << 3) | (sfi >> 1);
    let asc_byte1: u8 = ((sfi & 1) << 7) | ((channels & 0x0F) << 3);

    // Sizes — all small enough to fit in a u8 (verified by structure)
    let dec_spec_len: u8 = 2; // AudioSpecificConfig = 2 bytes
    let dec_config_body_len: u8 = 13; // object_type + stream_type + bufSize + maxBR + avgBR
    let dec_config_len: u8 = dec_config_body_len + 2 + dec_spec_len; // + DecoderSpecificInfo tag+len
    let sl_config_len: u8 = 1; // predefined=2
                               // ES_Descriptor body: ES_ID(2) + priority(1) + DecoderConfigDescriptor (2+dec_config_len)
                               //                     + SLConfigDescriptor (2+sl_config_len)
    let es_desc_body_len: u8 = 3 + 2 + dec_config_len + 2 + sl_config_len;

    let mut esds = Vec::new();
    write_u32(&mut esds, 0); // version 0 + flags

    // ES_Descriptor
    esds.push(0x03); // tag
    esds.push(es_desc_body_len);
    esds.extend_from_slice(&[0, 1]); // ES_ID = 1
    esds.push(0); // stream priority = 0

    // DecoderConfigDescriptor
    esds.push(0x04); // tag
    esds.push(dec_config_len);
    esds.push(0x40); // objectTypeIndication = Audio ISO/IEC 14496-3
    esds.push(0x15); // streamType=Audio (0x05 << 2 | upstream=0 | reserved=1)
    esds.extend_from_slice(&[0, 0, 0]); // bufferSizeDB = 0
    write_u32(&mut esds, 128_000); // maxBitrate = 128 kbps
    write_u32(&mut esds, 128_000); // avgBitrate = 128 kbps

    // DecoderSpecificInfo (AudioSpecificConfig)
    esds.push(0x05); // tag
    esds.push(dec_spec_len);
    esds.push(asc_byte0);
    esds.push(asc_byte1);

    // SLConfigDescriptor
    esds.push(0x06); // tag
    esds.push(sl_config_len);
    esds.push(2); // predefined = 2 (ISO 14496-1 SL packet header)

    esds
}

/// Build the complete moov box in memory, returning `(moov_bytes, stco_offset_patch_pos)`.
///
/// `stco_offset_patch_pos` is the byte index within the returned Vec where the 4-byte
/// chunk-offset value must be written once the final mdat position is known.
fn build_moov(
    sample_rate: u32,
    channels: usize,
    n_frames: u32,
    raw_frames: &[Vec<u8>],
    sfi: u8,
) -> (Vec<u8>, usize) {
    const SAMPLES_PER_FRAME: u32 = 1024; // AAC-LC ONLY_LONG window
    let duration = n_frames * SAMPLES_PER_FRAME; // in timescale (= sample_rate) units

    // ── mvhd (version 0, 108 bytes) ──────────────────────────────────────────
    let mut mvhd = Vec::new();
    write_box_header(&mut mvhd, 108, BOX_MVHD);
    write_u32(&mut mvhd, 0); // version 0 + flags
    write_u32(&mut mvhd, 0); // creation time
    write_u32(&mut mvhd, 0); // modification time
    write_u32(&mut mvhd, sample_rate); // timescale
    write_u32(&mut mvhd, duration); // duration
    write_u32(&mut mvhd, 0x0001_0000); // rate = 1.0 (16.16 fixed)
    write_u16(&mut mvhd, 0x0100); // volume = 1.0 (8.8 fixed)
    mvhd.extend_from_slice(&[0u8; 10]); // reserved
                                        // Unity matrix (9 × 4 bytes = 36 bytes)
    #[rustfmt::skip]
    mvhd.extend_from_slice(&[
        0x00,0x01,0x00,0x00,  0x00,0x00,0x00,0x00,  0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,  0x00,0x01,0x00,0x00,  0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,  0x00,0x00,0x00,0x00,  0x40,0x00,0x00,0x00,
    ]);
    mvhd.extend_from_slice(&[0u8; 24]); // pre_defined
    write_u32(&mut mvhd, 2); // next_track_id

    // ── esds ─────────────────────────────────────────────────────────────────
    let esds_body = build_esds_body(channels as u8, sfi);
    let esds_box_size = (8 + esds_body.len()) as u32;

    // ── mp4a (SampleEntry) ────────────────────────────────────────────────────
    // size(4)+type(4)+reserved(6)+data_ref_index(2)+reserved2(8)+channel_count(2)
    // +sample_size(2)+pre_defined(2)+reserved3(2)+sample_rate_fp(4)+esds_box
    let mp4a_size = 8 + 6 + 2 + 8 + 2 + 2 + 2 + 2 + 4 + esds_box_size;
    let mut mp4a = Vec::new();
    write_box_header(&mut mp4a, mp4a_size, BOX_MP4A);
    mp4a.extend_from_slice(&[0u8; 6]); // reserved
    write_u16(&mut mp4a, 1); // data reference index = 1
    mp4a.extend_from_slice(&[0u8; 8]); // reserved2
    write_u16(&mut mp4a, channels as u16); // channel_count
    write_u16(&mut mp4a, 16); // sample_size = 16 bits
    write_u16(&mut mp4a, 0); // pre_defined
    write_u16(&mut mp4a, 0); // reserved3
                             // sample_rate as 16.16 fixed-point; sample_rate fits in u16 (max 96000 < 65536 fails
                             // for some rates, so store as-is in the upper 16 bits with explicit masking)
    let sr_fp: u32 = sample_rate.min(0xFFFF) << 16;
    write_u32(&mut mp4a, sr_fp);
    // esds box
    write_box_header(&mut mp4a, esds_box_size, BOX_ESDS);
    mp4a.extend_from_slice(&esds_body);

    // ── stsd ─────────────────────────────────────────────────────────────────
    let stsd_size = (8 + 4 + 4 + mp4a.len()) as u32;
    let mut stsd = Vec::new();
    write_box_header(&mut stsd, stsd_size, BOX_STSD);
    write_u32(&mut stsd, 0); // version + flags
    write_u32(&mut stsd, 1); // entry count = 1
    stsd.extend_from_slice(&mp4a);

    // ── stts (time-to-sample, 1 entry: all frames same delta) ────────────────
    let stts_size = 8 + 4 + 4 + 8u32; // header + version_flags + entry_count + 1 entry
    let mut stts = Vec::new();
    write_box_header(&mut stts, stts_size, BOX_STTS);
    write_u32(&mut stts, 0); // version + flags
    write_u32(&mut stts, 1); // entry count = 1
    write_u32(&mut stts, n_frames); // sample count
    write_u32(&mut stts, SAMPLES_PER_FRAME); // sample delta

    // ── stsc (sample-to-chunk, 1 entry: all samples in 1 chunk) ─────────────
    let stsc_size = 8 + 4 + 4 + 12u32;
    let mut stsc = Vec::new();
    write_box_header(&mut stsc, stsc_size, BOX_STSC);
    write_u32(&mut stsc, 0); // version + flags
    write_u32(&mut stsc, 1); // entry count
    write_u32(&mut stsc, 1); // first_chunk = 1
    write_u32(&mut stsc, n_frames); // samples_per_chunk
    write_u32(&mut stsc, 1); // sample_description_index

    // ── stsz (sample sizes) ───────────────────────────────────────────────────
    let stsz_size = 8 + 4 + 4 + 4 + n_frames * 4;
    let mut stsz = Vec::new();
    write_box_header(&mut stsz, stsz_size, BOX_STSZ);
    write_u32(&mut stsz, 0); // version + flags
    write_u32(&mut stsz, 0); // sample_size = 0 (variable)
    write_u32(&mut stsz, n_frames);
    for f in raw_frames {
        write_u32(&mut stsz, f.len() as u32);
    }

    // ── stco (chunk offset, 1 chunk) — placeholder, patched later ────────────
    let stco_size = 8 + 4 + 4 + 4u32;
    let mut stco = Vec::new();
    write_box_header(&mut stco, stco_size, BOX_STCO);
    write_u32(&mut stco, 0); // version + flags
    write_u32(&mut stco, 1); // entry count = 1
                             // placeholder for mdat offset — remember position so we can patch it
    let stco_placeholder_in_stco = stco.len(); // = 12, index of the 4-byte offset field
    write_u32(&mut stco, 0); // PLACEHOLDER

    // ── stbl ─────────────────────────────────────────────────────────────────
    let stbl_size = (8 + stsd.len() + stts.len() + stsc.len() + stsz.len() + stco.len()) as u32;
    let mut stbl = Vec::new();
    // Track where stco ends up inside stbl (for offset patching)
    let stco_offset_in_stbl =
        8 + stsd.len() + stts.len() + stsc.len() + stsz.len() + stco_placeholder_in_stco;
    write_box_header(&mut stbl, stbl_size, BOX_STBL);
    stbl.extend_from_slice(&stsd);
    stbl.extend_from_slice(&stts);
    stbl.extend_from_slice(&stsc);
    stbl.extend_from_slice(&stsz);
    stbl.extend_from_slice(&stco);

    // ── smhd (sound media header, 16 bytes) ───────────────────────────────────
    let mut smhd = Vec::new();
    write_box_header(&mut smhd, 16, BOX_SMHD);
    write_u32(&mut smhd, 0); // version + flags
    write_u16(&mut smhd, 0); // balance = 0
    write_u16(&mut smhd, 0); // reserved

    // ── url (self-contained data entry, 12 bytes) ─────────────────────────────
    let mut url_box = Vec::new();
    write_box_header(&mut url_box, 12, BOX_URL);
    write_u32(&mut url_box, 1); // version 0 + flags: self-contained = 0x000001

    // ── dref ─────────────────────────────────────────────────────────────────
    let dref_size = (8 + 4 + 4 + url_box.len()) as u32;
    let mut dref = Vec::new();
    write_box_header(&mut dref, dref_size, BOX_DREF);
    write_u32(&mut dref, 0); // version + flags
    write_u32(&mut dref, 1); // entry count = 1
    dref.extend_from_slice(&url_box);

    // ── dinf ─────────────────────────────────────────────────────────────────
    let dinf_size = (8 + dref.len()) as u32;
    let mut dinf = Vec::new();
    write_box_header(&mut dinf, dinf_size, BOX_DINF);
    dinf.extend_from_slice(&dref);

    // ── minf ─────────────────────────────────────────────────────────────────
    let minf_size = (8 + smhd.len() + dinf.len() + stbl.len()) as u32;
    let mut minf = Vec::new();
    // Track where stbl ends up inside minf (for stco patching)
    let stbl_offset_in_minf = 8 + smhd.len() + dinf.len();
    write_box_header(&mut minf, minf_size, BOX_MINF);
    minf.extend_from_slice(&smhd);
    minf.extend_from_slice(&dinf);
    minf.extend_from_slice(&stbl);

    // ── mdhd (media header, version 0, 32 bytes) ─────────────────────────────
    let mut mdhd = Vec::new();
    write_box_header(&mut mdhd, 32, BOX_MDHD);
    write_u32(&mut mdhd, 0); // version + flags
    write_u32(&mut mdhd, 0); // creation time
    write_u32(&mut mdhd, 0); // modification time
    write_u32(&mut mdhd, sample_rate); // timescale
    write_u32(&mut mdhd, duration); // duration
                                    // language = 'und' (0x55C4) packed as ISO 639-2/T 3×5-bit chars, + pre_defined=0
    write_u32(&mut mdhd, 0x55C4_0000u32);

    // ── hdlr ─────────────────────────────────────────────────────────────────
    let hdlr_name = b"SoundHandler\0";
    let hdlr_size = (8 + 4 + 4 + 4 + 12 + hdlr_name.len()) as u32;
    let mut hdlr = Vec::new();
    write_box_header(&mut hdlr, hdlr_size, BOX_HDLR);
    write_u32(&mut hdlr, 0); // version + flags
    write_u32(&mut hdlr, 0); // pre_defined
    hdlr.extend_from_slice(b"soun"); // handler type
    hdlr.extend_from_slice(&[0u8; 12]); // reserved
    hdlr.extend_from_slice(hdlr_name);

    // ── mdia ─────────────────────────────────────────────────────────────────
    let mdia_size = (8 + mdhd.len() + hdlr.len() + minf.len()) as u32;
    let mut mdia = Vec::new();
    // Track minf offset inside mdia (for stco patching)
    let minf_offset_in_mdia = 8 + mdhd.len() + hdlr.len();
    write_box_header(&mut mdia, mdia_size, BOX_MDIA);
    mdia.extend_from_slice(&mdhd);
    mdia.extend_from_slice(&hdlr);
    mdia.extend_from_slice(&minf);

    // ── tkhd (track header, version 0, 92 bytes) ─────────────────────────────
    let mut tkhd = Vec::new();
    write_box_header(&mut tkhd, 92, BOX_TKHD);
    write_u32(&mut tkhd, 3); // version 0 + flags: track-enabled(1) + in-movie(2)
    write_u32(&mut tkhd, 0); // creation time
    write_u32(&mut tkhd, 0); // modification time
    write_u32(&mut tkhd, 1); // track ID = 1
    write_u32(&mut tkhd, 0); // reserved
    write_u32(&mut tkhd, duration); // duration (in movie timescale = sample_rate)
    tkhd.extend_from_slice(&[0u8; 8]); // reserved
    write_u16(&mut tkhd, 0); // layer
    write_u16(&mut tkhd, 0); // alternate_group
    write_u16(&mut tkhd, 0x0100); // volume = 1.0 (8.8 fixed)
    write_u16(&mut tkhd, 0); // reserved
                             // Unity matrix
    #[rustfmt::skip]
    tkhd.extend_from_slice(&[
        0x00,0x01,0x00,0x00,  0x00,0x00,0x00,0x00,  0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,  0x00,0x01,0x00,0x00,  0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,  0x00,0x00,0x00,0x00,  0x40,0x00,0x00,0x00,
    ]);
    write_u32(&mut tkhd, 0); // width (audio track: 0)
    write_u32(&mut tkhd, 0); // height (audio track: 0)

    // ── trak ─────────────────────────────────────────────────────────────────
    let trak_size = (8 + tkhd.len() + mdia.len()) as u32;
    let mut trak = Vec::new();
    // Track mdia offset inside trak
    let mdia_offset_in_trak = 8 + tkhd.len();
    write_box_header(&mut trak, trak_size, BOX_TRAK);
    trak.extend_from_slice(&tkhd);
    trak.extend_from_slice(&mdia);

    // ── moov ─────────────────────────────────────────────────────────────────
    let moov_payload_size = mvhd.len() + trak.len();
    let moov_box_size = (8 + moov_payload_size) as u32;
    let mut moov = Vec::with_capacity(8 + moov_payload_size);
    // Track trak offset inside moov
    let trak_offset_in_moov = 8 + mvhd.len();
    write_box_header(&mut moov, moov_box_size, BOX_MOOV);
    moov.extend_from_slice(&mvhd);
    moov.extend_from_slice(&trak);

    // Compute the absolute byte offset of the stco chunk-offset field in moov
    let stco_field_in_moov = trak_offset_in_moov
        + mdia_offset_in_trak
        + minf_offset_in_mdia
        + stbl_offset_in_minf
        + stco_offset_in_stbl;

    (moov, stco_field_in_moov)
}

/// Encode `buf` as M4A and write to `writer`.
///
/// Builds all boxes in memory to avoid requiring `Seek` on the output writer.
///
/// # Sample rate note
///
/// The `mp4a` sample description box stores the sample rate as a 16.16 fixed-point number,
/// with the integer part occupying only the upper 16 bits.  For standard audio rates
/// (≤ 65535 Hz) the value round-trips exactly.  For rates such as 88200 or 96000 Hz the
/// integer field would overflow 16 bits, so it is clamped to 0xFFFF in that field.
/// The actual sample rate is still recorded correctly in the `mdhd` timescale field and
/// in the AudioSpecificConfig (ESDS `DecoderSpecificInfo`), so decoders that read those
/// fields will see the true rate.
///
/// # Errors
///
/// Returns `OxiAudioError::Encode` for unsupported configurations or
/// `OxiAudioError::Io` on I/O failures.
pub fn encode_m4a<W: Write>(buf: &AudioBuffer<f32>, writer: &mut W) -> Result<(), OxiAudioError> {
    let channels = buf.channels.channel_count();
    let sample_rate = buf.sample_rate;

    if channels == 0 || channels > 2 {
        return Err(OxiAudioError::Encode(format!(
            "M4A supports 1-2 channels; got {channels}"
        )));
    }
    let sfi = sampling_freq_index(sample_rate).ok_or_else(|| {
        OxiAudioError::Encode(format!("M4A unsupported sample rate: {sample_rate}"))
    })?;

    // Collect ADTS frames and raw payloads (ADTS header stripped)
    let (_adts_frames, raw_frames) = collect_frames(buf)?;
    let n_frames = raw_frames.len() as u32;

    // Build ftyp (28 bytes)
    // size(4)+type(4)+major_brand(4)+minor_version(4)+compat_brand×3(12)
    let ftyp_size: u32 = 8 + 4 + 4 + 4 * 3;
    let mut ftyp = Vec::with_capacity(ftyp_size as usize);
    write_box_header(&mut ftyp, ftyp_size, BOX_FTYP);
    ftyp.extend_from_slice(b"M4A "); // major brand
    write_u32(&mut ftyp, 0); // minor version
    ftyp.extend_from_slice(b"M4A "); // compatible brand 1
    ftyp.extend_from_slice(b"mp42"); // compatible brand 2
    ftyp.extend_from_slice(b"isom"); // compatible brand 3

    // Build moov in memory; get back the byte position of the stco offset field
    let (mut moov, stco_field_in_moov) =
        build_moov(sample_rate, channels, n_frames, &raw_frames, sfi);

    // mdat: size(4)+type(4)+raw_frames
    let mdat_payload: u32 = raw_frames.iter().map(|f| f.len() as u32).sum();
    let mdat_size = 8 + mdat_payload;

    // The stco chunk offset must point to the first byte of sample data — that is,
    // immediately after the 8-byte mdat box header (size + type).
    // mdat box position: ftyp_size + moov_size; data position: that + 8.
    let mdat_offset = ftyp_size + moov.len() as u32 + 8;

    // Patch the stco chunk-offset placeholder with the real mdat offset
    if stco_field_in_moov + 4 <= moov.len() {
        moov[stco_field_in_moov..stco_field_in_moov + 4]
            .copy_from_slice(&mdat_offset.to_be_bytes());
    }

    // Write ftyp + moov + mdat in one sequential pass (no Seek required)
    writer.write_all(&ftyp).map_err(OxiAudioError::Io)?;
    writer.write_all(&moov).map_err(OxiAudioError::Io)?;

    // mdat header
    let mut mdat_hdr = [0u8; 8];
    mdat_hdr[..4].copy_from_slice(&mdat_size.to_be_bytes());
    mdat_hdr[4..8].copy_from_slice(BOX_MDAT);
    writer.write_all(&mdat_hdr).map_err(OxiAudioError::Io)?;

    // mdat payload: raw AAC frames (no ADTS headers)
    for frame in &raw_frames {
        writer.write_all(frame).map_err(OxiAudioError::Io)?;
    }

    Ok(())
}

/// Write M4A to a file at `path`.
///
/// # Errors
///
/// Returns `OxiAudioError` on I/O failure or unsupported format.
pub fn encode_m4a_file(buf: &AudioBuffer<f32>, path: &Path) -> Result<(), OxiAudioError> {
    let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
    let mut writer = std::io::BufWriter::new(file);
    encode_m4a(buf, &mut writer)
}

// ─── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    use std::io::Cursor;

    fn silence_buf(sample_rate: u32, frames: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.0f32; frames],
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_m4a_starts_with_ftyp() {
        let buf = silence_buf(44_100, 4096);
        let mut out = Cursor::new(Vec::new());
        encode_m4a(&buf, &mut out).expect("M4A encode must succeed");
        let bytes = out.into_inner();
        assert!(bytes.len() > 16, "M4A output must be non-trivially large");
        assert_eq!(
            &bytes[4..8],
            b"ftyp",
            "M4A must have ftyp as first box type"
        );
        assert_eq!(&bytes[8..12], b"M4A ", "ftyp major brand must be M4A ");
    }

    #[test]
    fn test_m4a_contains_moov() {
        let buf = silence_buf(44_100, 4096);
        let mut out = Cursor::new(Vec::new());
        encode_m4a(&buf, &mut out).expect("M4A encode");
        let bytes = out.into_inner();
        let has_moov = bytes.windows(4).any(|w| w == b"moov");
        assert!(has_moov, "M4A must contain a moov box");
    }

    #[test]
    fn test_m4a_contains_mdat() {
        let buf = silence_buf(44_100, 4096);
        let mut out = Cursor::new(Vec::new());
        encode_m4a(&buf, &mut out).expect("M4A encode");
        let bytes = out.into_inner();
        let has_mdat = bytes.windows(4).any(|w| w == b"mdat");
        assert!(has_mdat, "M4A must contain an mdat box");
    }

    #[test]
    fn test_m4a_stereo_encode() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 4096 * 2],
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let mut out = Cursor::new(Vec::new());
        encode_m4a(&buf, &mut out).expect("M4A stereo encode");
        let bytes = out.into_inner();
        assert!(
            bytes.windows(4).any(|w| w == b"ftyp"),
            "stereo M4A must have ftyp"
        );
    }

    #[test]
    fn test_m4a_file_write() {
        let buf = silence_buf(44_100, 4096);
        let tmp = std::env::temp_dir().join("oxiaudio_test.m4a");
        encode_m4a_file(&buf, &tmp).expect("M4A file write");
        let bytes = std::fs::read(&tmp).expect("read M4A file");
        assert_eq!(&bytes[4..8], b"ftyp", "file must have ftyp box");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_m4a_rejects_unsupported_channels() {
        use oxiaudio_core::ChannelLayout;
        let buf = AudioBuffer {
            samples: vec![0.0f32; 1024 * 6],
            sample_rate: 44_100,
            channels: ChannelLayout::Surround51,
            format: SampleFormat::F32,
        };
        let mut out = Cursor::new(Vec::new());
        let result = encode_m4a(&buf, &mut out);
        assert!(
            result.is_err(),
            "M4A encoder must reject more than 2 channels"
        );
    }

    #[test]
    fn test_m4a_rejects_unsupported_sample_rate() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 1024],
            sample_rate: 22222, // not in the MPEG-4 table
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mut out = Cursor::new(Vec::new());
        let result = encode_m4a(&buf, &mut out);
        assert!(
            result.is_err(),
            "M4A encoder must reject unsupported sample rates"
        );
    }

    #[test]
    fn test_m4a_box_order_is_ftyp_moov_mdat() {
        let buf = silence_buf(44_100, 4096);
        let mut out = Cursor::new(Vec::new());
        encode_m4a(&buf, &mut out).expect("M4A encode");
        let bytes = out.into_inner();

        // ftyp must be first
        assert_eq!(&bytes[4..8], b"ftyp", "ftyp must be first box");

        // moov must come before mdat
        let moov_pos = bytes
            .windows(4)
            .position(|w| w == b"moov")
            .expect("moov must exist");
        let mdat_pos = bytes
            .windows(4)
            .position(|w| w == b"mdat")
            .expect("mdat must exist");
        assert!(moov_pos < mdat_pos, "moov must precede mdat");
    }

    #[test]
    fn test_m4a_stco_offset_points_to_mdat_data() {
        let buf = silence_buf(44_100, 4096);
        let mut out = Cursor::new(Vec::new());
        encode_m4a(&buf, &mut out).expect("M4A encode");
        let bytes = out.into_inner();

        // Find the mdat box start
        let mdat_start = bytes
            .windows(4)
            .position(|w| w == b"mdat")
            .expect("mdat must exist");
        // mdat data starts 8 bytes after the box header start (at mdat_start - 4 for size)
        // Actually mdat_start is the position of b"mdat" tag, so data is at mdat_start + 4
        let mdat_data_start = mdat_start + 4;

        // Find stco box and read its offset
        let stco_pos = bytes
            .windows(4)
            .position(|w| w == b"stco")
            .expect("stco must exist");
        // stco_pos points to the "stco" type tag (4 bytes).
        // Layout from stco_pos: type(4) + version_flags(4) + entry_count(4) + chunk_offset(4)
        let offset_pos = stco_pos + 4 + 4 + 4; // skip type, version_flags, entry_count → chunk_offset
        let chunk_offset = u32::from_be_bytes([
            bytes[offset_pos],
            bytes[offset_pos + 1],
            bytes[offset_pos + 2],
            bytes[offset_pos + 3],
        ]) as usize;

        // The chunk offset in stco points to the mdat data (not the header)
        assert_eq!(
            chunk_offset, mdat_data_start,
            "stco chunk offset must point to mdat data start"
        );
    }
}
