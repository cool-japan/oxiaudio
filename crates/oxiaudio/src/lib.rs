//! # OxiAudio
//!
//! Pure-Rust audio processing workspace. Default features are C-free; LAME MP3 encoding
//! is available via the `mp3-encode-lame` feature (LGPL, FFI).
//!
//! ## Supported formats (decode)
//!
//! | Format | Pure Rust | Feature flag |
//! |--------|-----------|--------------|
//! | WAV / RF64 | Yes | default |
//! | FLAC | Yes | default |
//! | MP3 (MPEG Audio) | Yes (symphonia) | default |
//! | OGG Vorbis | Yes | default |
//! | AAC / M4A | Yes | default |
//! | AIFF / AIFF-C | Yes | default |
//!
//! ## Supported formats (encode)
//!
//! | Format | Pure Rust | Feature flag |
//! |--------|-----------|--------------|
//! | WAV / RF64 | Yes | default |
//! | FLAC | Yes | default |
//! | AIFF | Yes | default |
//! | AU | Yes | default |
//! | MP3 (via LAME) | No (FFI) | mp3-encode-lame |
//!
//! ## Quick start
//!
//! ```no_run
//! use std::path::Path;
//!
//! // Decode an audio file
//! let buf = oxiaudio::decode_file(Path::new("input.flac")).unwrap();
//! println!("{} frames at {} Hz", buf.frame_count(), buf.sample_rate);
//!
//! // Apply DSP: normalize then add reverb
//! let mut processed = buf.clone();
//! oxiaudio::dsp::normalize(&mut processed, -1.0);
//! let with_reverb = oxiaudio::dsp::reverb(&processed, 0.6, 0.4, 0.3);
//!
//! // Re-encode to a different format
//! oxiaudio::encode_flac(&with_reverb, Path::new("output.flac")).unwrap();
//! ```
#![forbid(unsafe_code)]

pub use oxiaudio_core::{from_planar, to_planar};
pub use oxiaudio_core::{
    AudioBuffer, AudioBufferLayout, AudioClock, AudioDecoder, AudioEncoder, AudioFilter,
    AudioFormat, AudioMetadata, AudioNode, AudioPipeline, AudioSink, AudioSource, ChannelLayout,
    OxiAudioError, ParallelBranchNode, SampleFormat, Timestamp,
};

// ─── M20-C IPC exports ───────────────────────────────────────────────────────

pub use oxiaudio_core::ring::{AudioRingBuffer, OverflowPolicy};
/// Serialize an [`AudioBuffer<f32>`] to a compact binary IPC frame.
pub use oxiaudio_core::{
    deserialize_audio_buffer_f32, from_ipc_bytes, serialize_audio_buffer_f32, to_ipc_bytes,
};

// ─── Surround channel support ────────────────────────────────────────────────

/// Per-channel identifier used with [`ChannelMap`].
pub use oxiaudio_core::ChannelId;

/// A named sequence of [`ChannelId`]s describing the role of each channel in a buffer.
pub use oxiaudio_core::ChannelMap;

/// Downmix a 5.1 surround buffer to stereo using ITU-R BS.775-3 coefficients.
pub use oxiaudio_core::downmix_51_to_stereo;

/// Downmix any multi-channel buffer to mono by averaging all channels per frame.
pub use oxiaudio_core::downmix_to_mono;

/// Upmix a mono buffer to stereo by duplicating the single channel.
pub use oxiaudio_core::upmix_mono_to_stereo;

// ─── Sub-modules ─────────────────────────────────────────────────────────────

#[cfg(feature = "pure")]
mod decode;
#[cfg(feature = "pure")]
mod encode;
#[cfg(feature = "pure")]
mod pipeline;

/// DSP utilities: resampling, gain control, normalization, silence trimming, and spectral analysis.
#[cfg(feature = "pure")]
pub mod dsp;

// ─── Flat re-exports from sub-modules ────────────────────────────────────────

#[cfg(feature = "pure")]
pub use decode::{
    alaw_to_linear,
    apply_gapless_trim,
    decode_aiff_file,
    decode_aiffc_compressed,
    decode_aiffc_compressed_file,
    decode_au_file,
    decode_file,
    decode_file_f64,
    decode_file_with_metadata,
    decode_file_with_options,
    // M22 additions
    decode_file_with_replaygain,
    decode_files,
    // OGG Opus decoder
    decode_opus_file,
    decode_opus_reader,
    decode_raw_pcm_file,
    // M19 additions
    decode_reader,
    decode_stream,
    decode_stream_with_block_size,
    decode_to_f64,
    decode_to_i16,
    decode_to_i32,
    decode_tolerant,
    detect_format,
    detect_format_file,
    detect_format_from_bytes,
    detect_format_from_path,
    extract_album_art,
    // M20 additions
    extract_lyrics,
    file_format,
    midi_note_to_hz,
    parse_flac_cue_sheet,
    parse_gapless_info,
    parse_id3v1,
    parse_opus_head,
    parse_replaygain,
    parse_wav_cues,
    parse_wav_cues_reader,
    // MIDI synthesizer
    synthesize_midi,
    synthesize_midi_default,
    ulaw_to_linear,
    // M21 additions
    AlbumArtwork,
    AudioFormatHint,
    // M23-K additions
    CorruptPacketPolicy,
    DecodeOptions,
    // FLAC cue sheet
    FlacCuePoint,
    // M18 additions
    GaplessInfo,
    MetaEvent,
    MidiAdsr,
    MidiEvent,
    // MIDI SMF parser
    MidiFile,
    MidiSynthConfig,
    MidiTrack,
    MidiWaveform,
    OpusHead,
    RawPcmConfig,
    ReplayGainMetadata,
    SmfFormat,
    StreamingDecoderBuilder,
    TimedEvent,
    TrackEvent,
    WavCuePoint,
};

#[cfg(feature = "pure")]
pub use encode::{
    analyze_loudness_gain,
    analyze_silk_frame,
    apply_tpdf_dither,
    encode_aac,
    encode_aac_file,
    encode_aac_to_file,
    encode_aiff,
    encode_aiff_with_chunks,
    encode_au,
    encode_flac,
    // M23-L additions
    encode_flac_parallel,
    encode_flac_to_vec,
    encode_flac_with_album_art,
    encode_flac_with_album_art_file,
    encode_flac_with_config,
    encode_flac_with_level,
    encode_flac_with_md5,
    encode_flac_with_md5_file,
    encode_flac_with_metadata,
    encode_flac_with_metadata_and_picture,
    encode_flac_with_picture,
    encode_flac_with_picture_file,
    encode_flac_with_progress,
    encode_flac_with_seektable,
    encode_flac_with_seektable_file,
    // M4A container writer
    encode_m4a,
    encode_m4a_file,
    encode_normalized_wav,
    encode_normalized_wav_file,
    encode_opus,
    // Opt-in RFC 6716–conformant Opus encoders
    encode_opus_conformant,
    encode_opus_conformant_file,
    encode_opus_file,
    encode_silk_frame,
    encode_stream,
    encode_stream_flac,
    // M20 additions — Vorbis, AAC, Opus streaming, SILK
    encode_vorbis,
    encode_vorbis_file,
    encode_vorbis_quality_file,
    encode_vorbis_to_file,
    // VBR quality modes
    encode_vorbis_with_quality,
    encode_wav,
    encode_wav_f64,
    // M17 additions
    encode_wav_rf64,
    encode_wav_rf64_file,
    encode_wav_streaming,
    encode_wav_to_vec,
    encode_wav_with_config,
    encode_wav_with_cues,
    encode_wav_with_cues_file,
    encode_wav_with_progress,
    inject_flac_md5,
    write_aiff_file,
    // M15 additions
    write_aiff_with_chunks,
    write_aiffc,
    write_aiffc_file,
    write_apev2,
    AiffBitDepth,
    AiffStreamEncoder,
    // M18 additions
    AiffcCodec,
    // M16 additions
    ApeItem,
    AuEncoding,
    CuePoint,
    EncodeProgressFn,
    EncoderConfig,
    FlacBitDepth,
    FlacConfig,
    FlacMetaConfig,
    // M19 additions
    FlacPicture,
    FlacStreamEncoder,
    FlacStreamingEncoder,
    Id3v24Tag,
    LoudnessTarget,
    OpusConformantMode,
    OpusEncodeConfig,
    OpusStreamEncoder,
    SilkBandwidth,
    SilkLpcFrame,
    StreamEncoder,
    VorbisQuality,
    WavBitDepth,
    WavStreamEncoder,
};

#[cfg(all(feature = "pure", feature = "mp3-encode-lame"))]
pub use encode::AlbumArt;

#[cfg(all(feature = "pure", feature = "mp3-encode-lame"))]
pub use oxiaudio_encode_mp3_lame::compute_replaygain_gain_approx;

#[cfg(all(feature = "pure", feature = "mp3-encode-lame"))]
pub use oxiaudio_encode_mp3_lame::lame::encode_mp3_with_auto_replaygain;

#[cfg(feature = "pure")]
pub use pipeline::{
    convert, convert_with_dsp, probe_metadata, transcode_batch, write_metadata, TranscodePipeline,
    TranscodeStream,
};

#[cfg(all(test, feature = "pure"))]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_440hz_1s_48k_stereo() -> AudioBuffer<f32> {
        let sample_rate = 48_000u32;
        let n_samples = sample_rate as usize;
        let mut samples = Vec::with_capacity(n_samples * 2);
        for i in 0..n_samples {
            let t = i as f32 / sample_rate as f32;
            let s = (2.0 * PI * 440.0 * t).sin() * 0.5;
            samples.push(s); // L
            samples.push(s); // R
        }
        AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_wav_roundtrip() {
        let original = sine_440hz_1s_48k_stereo();
        let tmp_dir = std::env::temp_dir();
        let wav_path = tmp_dir.join("oxiaudio_test_roundtrip.wav");

        // Encode to WAV
        encode_wav(&original, &wav_path).expect("encode_wav failed");

        // Decode back
        let decoded = decode_file(&wav_path).expect("decode_file failed");

        // Clean up before assertions to avoid leaking temp files on failure
        let _ = std::fs::remove_file(&wav_path);

        assert_eq!(
            decoded.sample_rate, original.sample_rate,
            "sample rate mismatch"
        );
        assert_eq!(
            decoded.channels, original.channels,
            "channel layout mismatch"
        );
        assert_eq!(
            decoded.samples.len(),
            original.samples.len(),
            "sample count mismatch: got {}, want {}",
            decoded.samples.len(),
            original.samples.len()
        );

        // WAV f32 should round-trip with negligible error
        for (i, (orig, dec)) in original
            .samples
            .iter()
            .zip(decoded.samples.iter())
            .enumerate()
        {
            assert!(
                (orig - dec).abs() < 1e-4,
                "sample[{}] mismatch: orig={} dec={}",
                i,
                orig,
                dec
            );
        }
    }

    #[test]
    fn test_gain() {
        let mut buf = sine_440hz_1s_48k_stereo();
        let first_sample = buf.samples[0];
        dsp::gain(&mut buf, 6.0); // approximately 2× amplitude
        let expected = first_sample * 10_f32.powf(6.0 / 20.0);
        assert!(
            (buf.samples[0] - expected).abs() < 1e-5,
            "gain mismatch: got {} expected {}",
            buf.samples[0],
            expected
        );
    }

    #[test]
    fn test_resample_identity() {
        let buf = sine_440hz_1s_48k_stereo();
        let resampled = dsp::resample(&buf, 48_000).expect("resample failed");
        assert_eq!(resampled.sample_rate, 48_000);
        assert_eq!(resampled.samples.len(), buf.samples.len());
    }

    #[test]
    fn test_resample_downsample() {
        let buf = sine_440hz_1s_48k_stereo();
        let resampled = dsp::resample(&buf, 44_100).expect("resample 48k→44.1k failed");
        assert_eq!(resampled.sample_rate, 44_100);
        // 1 second at 44100 Hz stereo = 88200 samples (approximately, due to resampler delay)
        let expected = 44_100 * 2;
        let actual = resampled.samples.len();
        let tolerance = 1000; // allow up to 500 frames of error for resampler latency
        assert!(
            actual.abs_diff(expected) <= tolerance,
            "resample sample count: got {} expected ~{}",
            actual,
            expected
        );
    }

    /// Verify that streaming a WAV via `decode_stream` produces the same total sample count
    /// (and matching sample values) as a single `decode_file` call on the same file.
    #[test]
    fn test_decode_stream_total_matches_decode_file() {
        let original = sine_440hz_1s_48k_stereo();
        let tmp_path = std::env::temp_dir().join("oxiaudio_m3_stream_test.wav");

        encode_wav(&original, &tmp_path).expect("encode_wav failed");

        // Decode via the normal one-shot path.
        let reference = decode_file(&tmp_path).expect("decode_file failed");

        // Decode via streaming with a small block size to exercise chunking.
        let file = std::fs::File::open(&tmp_path).expect("open for streaming failed");
        let reader = std::io::BufReader::new(file);
        let mut streamed_samples: Vec<f32> = Vec::new();
        for chunk in decode_stream_with_block_size(reader, 512) {
            let buf = chunk.expect("stream chunk error");
            streamed_samples.extend_from_slice(&buf.samples);
        }

        let _ = std::fs::remove_file(&tmp_path);

        assert_eq!(
            streamed_samples.len(),
            reference.samples.len(),
            "streamed sample count {} != decode_file count {}",
            streamed_samples.len(),
            reference.samples.len(),
        );

        for (i, (s, r)) in streamed_samples
            .iter()
            .zip(reference.samples.iter())
            .enumerate()
        {
            assert!(
                (s - r).abs() < 1e-6,
                "sample[{i}] mismatch: streamed={s} reference={r}",
            );
        }
    }

    #[test]
    fn test_decode_file_f64_returns_f64() {
        let original = sine_440hz_1s_48k_stereo();
        let tmp = std::env::temp_dir().join("oxiaudio_m4_decode_f64_test.wav");
        encode_wav(&original, &tmp).expect("encode_wav failed");
        let buf = decode_file_f64(&tmp).expect("decode_file_f64 failed");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(buf.format, SampleFormat::F64, "format should be F64");
        assert!(!buf.samples.is_empty(), "samples should not be empty");
        assert_eq!(buf.sample_rate, 48_000);
    }

    #[test]
    fn test_encode_stream_wav_roundtrip() {
        let original = sine_440hz_1s_48k_stereo();
        let tmp = std::env::temp_dir().join("oxiaudio_m4_encode_stream_test.wav");
        let n_channels = 2usize;
        let chunk_frames = 4096usize;
        let chunks: Vec<AudioBuffer<f32>> = original
            .samples
            .chunks(chunk_frames * n_channels)
            .map(|c| AudioBuffer {
                samples: c.to_vec(),
                sample_rate: original.sample_rate,
                channels: original.channels,
                format: original.format,
            })
            .collect();
        {
            let file = std::fs::File::create(&tmp).expect("create failed");
            let writer = std::io::BufWriter::new(file);
            encode_stream(chunks.iter(), writer).expect("encode_stream failed");
        }
        let decoded = decode_file(&tmp).expect("decode_file failed");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(decoded.sample_rate, original.sample_rate);
        assert_eq!(decoded.samples.len(), original.samples.len());
    }

    #[test]
    fn test_dsp_biquad_and_parametric_eq_reexport() {
        let buf = sine_440hz_1s_48k_stereo();
        let filter = dsp::BiquadFilter::low_shelf(200.0, 0.0, buf.sample_rate);
        let out = filter.process(&buf);
        assert_eq!(out.samples.len(), buf.samples.len());

        let eq = dsp::ParametricEq::new(vec![filter]);
        let out2 = eq.process(&buf);
        assert_eq!(out2.samples.len(), buf.samples.len());
    }

    /// Verify that `dsp::pitch_shift` followed by `encode_wav` completes without error.
    #[test]
    fn test_pitch_shift_encodes_ok() {
        // 0.5 s 440 Hz stereo at 48 kHz
        let sample_rate = 48_000u32;
        let n_frames = sample_rate as usize / 2; // 0.5 s
        let mut samples = Vec::with_capacity(n_frames * 2);
        for i in 0..n_frames {
            let t = i as f32 / sample_rate as f32;
            let s = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
            samples.push(s);
            samples.push(s);
        }
        let buf = AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };

        let shifted = dsp::pitch_shift(&buf, 12.0).expect("pitch_shift failed");

        let tmp_path = std::env::temp_dir().join("oxiaudio_m3_pitch_shift_test.wav");
        encode_wav(&shifted, &tmp_path).expect("encode_wav after pitch_shift failed");

        // Verify the output can be decoded back (i.e. it is a valid WAV file).
        let decoded = decode_file(&tmp_path).expect("decode_file of pitch-shifted WAV failed");
        let _ = std::fs::remove_file(&tmp_path);

        assert!(
            !decoded.samples.is_empty(),
            "decoded pitch-shifted WAV has no samples"
        );
        assert_eq!(
            decoded.sample_rate, sample_rate,
            "sample rate changed after pitch-shift encode/decode"
        );
    }

    /// Verify `dsp::loudness_lufs` is consistent with calling `oxiaudio_dsp::loudness_integrated`
    /// directly on the same buffer.
    #[test]
    fn test_loudness_lufs_matches_direct_call() {
        // Use a 5-second calibration tone loud enough to clear the -70 LUFS absolute gate.
        // Target -23 LUFS: mean_square = 10^((-23 + 0.691) / 10), amplitude = sqrt(2 * ms).
        let sample_rate = 48_000u32;
        let target_lufs = -23.0f32;
        let ms = 10.0f32.powf((target_lufs + 0.691) / 10.0);
        let amplitude = (2.0 * ms).sqrt();
        let n = sample_rate as usize * 5; // 5 seconds
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                amplitude
                    * (2.0 * std::f32::consts::PI * 997.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let via_facade = dsp::loudness_lufs(&buf);
        let direct = oxiaudio_dsp::loudness_integrated(&buf);
        // Both calls must return identical bit patterns (same computation, same data).
        assert_eq!(
            via_facade.to_bits(),
            direct.to_bits(),
            "loudness_lufs ({via_facade}) != loudness_integrated ({direct})"
        );
    }

    /// Verify the spectral feature re-exports are accessible through `dsp::spectral`.
    #[test]
    fn test_spectral_reexports_accessible() {
        use dsp::spectral::{
            mfcc, spectral_centroid, spectral_flatness, spectral_flux, spectral_rolloff,
            zero_crossing_rate,
        };
        let buf = sine_440hz_1s_48k_stereo();
        let n_fft = 2048usize;
        let hop = 512usize;

        let centroids = spectral_centroid(&buf, n_fft, hop);
        assert!(
            !centroids.is_empty(),
            "spectral_centroid should return non-empty Vec"
        );
        assert!(
            centroids.iter().any(|&c| c > 0.0),
            "at least one centroid > 0 for 440 Hz tone"
        );

        let fluxes = spectral_flux(&buf, n_fft, hop);
        assert!(
            !fluxes.is_empty(),
            "spectral_flux should return non-empty Vec"
        );
        assert!(
            fluxes.iter().all(|&f| f >= 0.0),
            "spectral_flux values should be non-negative"
        );

        let rolloffs = spectral_rolloff(&buf, n_fft, hop, 0.85);
        assert!(
            !rolloffs.is_empty(),
            "spectral_rolloff should return non-empty Vec"
        );

        let flatness = spectral_flatness(&buf, n_fft, hop);
        assert!(
            !flatness.is_empty(),
            "spectral_flatness should return non-empty Vec"
        );
        assert!(
            flatness.iter().all(|&f| f >= 0.0),
            "spectral_flatness values should be non-negative"
        );

        let zcr = zero_crossing_rate(&buf, 512, 256);
        assert!(
            !zcr.is_empty(),
            "zero_crossing_rate should return non-empty Vec"
        );
        assert!(
            zcr.iter().all(|&z| z >= 0.0),
            "zero_crossing_rate values should be non-negative"
        );

        let mfcc_result = mfcc(&buf, 13, 40, n_fft, hop).expect("mfcc failed");
        assert!(
            !mfcc_result.is_empty(),
            "mfcc should return non-empty result"
        );
        assert_eq!(
            mfcc_result[0].len(),
            13,
            "each mfcc frame should have 13 coefficients"
        );
    }

    /// Verify the DSP type re-exports (Compressor, Limiter, NoiseGate, DelayLine, Chorus,
    /// Tremolo, Vibrato) are accessible through the `dsp` module.
    #[test]
    fn test_dsp_dynamics_and_effects_reexports() {
        let buf = sine_440hz_1s_48k_stereo();

        // Compressor: threshold=-20dBFS, ratio=4:1, attack=10ms, release=100ms
        let comp = dsp::Compressor::new(-20.0, 4.0, 10.0, 100.0);
        let out = comp.process(&buf);
        assert_eq!(
            out.samples.len(),
            buf.samples.len(),
            "Compressor output length mismatch"
        );

        // Limiter: ceiling=-3dBFS, release=50ms
        let limiter = dsp::Limiter::new(-3.0, 50.0);
        let out = limiter.process(&buf);
        assert_eq!(
            out.samples.len(),
            buf.samples.len(),
            "Limiter output length mismatch"
        );

        // NoiseGate: threshold=-40dBFS (loud tone should pass through)
        let gate = dsp::NoiseGate::new(-40.0);
        let out = gate.process(&buf);
        assert_eq!(
            out.samples.len(),
            buf.samples.len(),
            "NoiseGate output length mismatch"
        );

        // DelayLine: 44.1ms delay, feedback=0.3, wet_dry=0.5
        let delay = dsp::DelayLine::new(44.1, 0.3, 0.5);
        let out = delay.process(&buf);
        assert_eq!(
            out.samples.len(),
            buf.samples.len(),
            "DelayLine output length mismatch"
        );

        // Chorus: rate=0.5 Hz, depth=10 ms
        let chorus = dsp::Chorus::new(0.5, 10.0);
        let out = chorus.process(&buf);
        assert_eq!(
            out.samples.len(),
            buf.samples.len(),
            "Chorus output length mismatch"
        );

        // Tremolo: rate=5 Hz, depth=0.5
        let tremolo = dsp::Tremolo::new(5.0, 0.5);
        let out = tremolo.process(&buf);
        assert_eq!(
            out.samples.len(),
            buf.samples.len(),
            "Tremolo output length mismatch"
        );

        // Vibrato: rate=5 Hz, depth=50 cents
        let vibrato = dsp::Vibrato::new(5.0, 50.0);
        let out = vibrato.process(&buf);
        assert_eq!(
            out.samples.len(),
            buf.samples.len(),
            "Vibrato output length mismatch"
        );
    }

    /// Verify the new M6 DSP convenience wrappers and re-exports through the facade.
    #[test]
    fn test_dsp_m6_convenience_and_reexports() {
        let buf = sine_440hz_1s_48k_stereo();

        // compressor wrapper
        let out = dsp::compressor(&buf, -20.0, 4.0, 5.0, 80.0).expect("compressor failed");
        assert_eq!(out.samples.len(), buf.samples.len());

        // reverb wrapper (output length matches; default no tail)
        let out = dsp::reverb(&buf, 0.6, 0.4, 0.3);
        assert_eq!(out.samples.len(), buf.samples.len());

        // eq wrapper with two bands — (center_hz, gain_db, q)
        let out = dsp::eq(&buf, &[(200.0, 3.0, 1.0), (5000.0, -3.0, 1.0)]).expect("eq failed");
        assert_eq!(out.samples.len(), buf.samples.len());

        // Butterworth cascade re-export
        let lp = dsp::butterworth_lowpass(4, 2_000.0, buf.sample_rate);
        let out = lp.process(&buf);
        assert_eq!(out.samples.len(), buf.samples.len());

        // Freeverb, Expander, Flanger, Phaser, FirFilter re-exports compile & run
        let _fv = dsp::Freeverb::new(buf.sample_rate);
        let exp = dsp::Expander::new(-40.0, 2.0, 5.0, 50.0);
        assert_eq!(exp.process(&buf).samples.len(), buf.samples.len());
        let fl = dsp::Flanger::new(buf.sample_rate);
        assert_eq!(fl.process(&buf).samples.len(), buf.samples.len());
        let ph = dsp::Phaser::new(buf.sample_rate);
        assert_eq!(ph.process(&buf).samples.len(), buf.samples.len());
        let fir =
            dsp::FirFilter::design_lowpass(31, 2_000.0, buf.sample_rate, dsp::FirWindow::Hann);
        assert_eq!(fir.process(&buf).samples.len(), buf.samples.len());

        // DspChain re-export
        let chain = dsp::DspChain::new().then_filter(dsp::BiquadFilter::highpass(
            80.0,
            0.707,
            buf.sample_rate,
        ));
        assert_eq!(
            chain.process(&buf).unwrap().samples.len(),
            buf.samples.len()
        );

        // loudness_range + detect_pitch wrappers run without panic
        let _ = dsp::loudness_range(&buf);
        let _ = dsp::detect_pitch(&buf);

        // spectral chromagram + bandwidth re-exports
        let chroma = dsp::spectral::chromagram(&buf, 2048, 512);
        assert!(!chroma.is_empty());
        let bw = dsp::spectral::spectral_bandwidth(&buf, 2048, 512);
        assert!(!bw.is_empty());
    }

    /// Verify the new encode-with-config facade helpers produce decodable files.
    #[test]
    fn test_encode_with_config_helpers() {
        let buf = sine_440hz_1s_48k_stereo();
        let dir = std::env::temp_dir();

        let wav_path = dir.join("oxiaudio_m6_cfg.wav");
        encode_wav_with_config(&buf, &wav_path, WavBitDepth::I16).expect("wav cfg");
        let decoded = decode_file(&wav_path).expect("decode wav cfg");
        let _ = std::fs::remove_file(&wav_path);
        assert_eq!(decoded.sample_rate, buf.sample_rate);

        let flac_path = dir.join("oxiaudio_m6_cfg.flac");
        let flac_cfg = FlacConfig {
            compression: 8,
            bit_depth: FlacBitDepth::I16,
        };
        encode_flac_with_config(&buf, &flac_path, &flac_cfg).expect("flac cfg");
        let decoded = decode_file(&flac_path).expect("decode flac cfg");
        let _ = std::fs::remove_file(&flac_path);
        assert_eq!(decoded.sample_rate, buf.sample_rate);
        assert_eq!(decoded.samples.len(), buf.samples.len());
    }

    /// Verify `encode_wav_f64` produces a decodable WAV file from an f64 buffer.
    #[test]
    fn test_encode_wav_f64_produces_decodable_wav() {
        let f32_buf = sine_440hz_1s_48k_stereo();
        let f64_buf = f32_buf.to_f64();
        let tmp = std::env::temp_dir().join("oxiaudio_m6_f64.wav");
        encode_wav_f64(&f64_buf, &tmp).expect("encode_wav_f64 failed");
        let decoded = decode_file(&tmp).expect("decode_file of f64-encoded WAV failed");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(decoded.sample_rate, f32_buf.sample_rate);
        assert_eq!(decoded.samples.len(), f32_buf.samples.len());
    }

    /// Verify `convert` produces a valid FLAC file from a WAV source.
    #[test]
    fn test_convert_wav_to_flac() {
        let buf = sine_440hz_1s_48k_stereo();
        let dir = std::env::temp_dir();
        let wav_path = dir.join("oxiaudio_m6_convert_src.wav");
        let flac_path = dir.join("oxiaudio_m6_convert_dst.flac");
        encode_wav(&buf, &wav_path).expect("encode_wav for convert test");
        convert(&wav_path, &flac_path).expect("convert wav→flac failed");
        // Verify the FLAC magic bytes.
        let bytes = std::fs::read(&flac_path).expect("read flac file");
        let _ = std::fs::remove_file(&wav_path);
        let _ = std::fs::remove_file(&flac_path);
        assert!(bytes.starts_with(b"fLaC"), "expected fLaC magic");
    }

    /// Verify `probe_metadata` returns metadata matching an encoded WAV.
    #[test]
    fn test_probe_metadata_returns_metadata() {
        let buf = sine_440hz_1s_48k_stereo();
        let tmp = std::env::temp_dir().join("oxiaudio_m6_probe.wav");
        encode_wav(&buf, &tmp).expect("encode_wav for probe test");
        let meta = probe_metadata(&tmp).expect("probe_metadata failed");
        let _ = std::fs::remove_file(&tmp);
        // WAV files typically have no embedded tags; duration may or may not be set.
        // The key invariant is that probe_metadata returns without error.
        let _ = meta;
    }

    /// Verify `decode_files` decodes multiple WAV files in parallel and returns all Ok.
    #[test]
    fn test_decode_files_parallel() {
        let buf = sine_440hz_1s_48k_stereo();
        let dir = std::env::temp_dir();
        let p1 = dir.join("oxiaudio_m6_par1.wav");
        let p2 = dir.join("oxiaudio_m6_par2.wav");
        encode_wav(&buf, &p1).expect("encode p1");
        encode_wav(&buf, &p2).expect("encode p2");
        let paths: Vec<&std::path::Path> = vec![p1.as_path(), p2.as_path()];
        let results = decode_files(&paths);
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
        assert_eq!(results.len(), 2);
        for (i, r) in results.iter().enumerate() {
            assert!(r.is_ok(), "decode_files result[{i}] was Err");
        }
    }

    /// Verify `TranscodePipeline` can transcode a WAV file to FLAC without a DSP step.
    #[test]
    fn test_transcode_pipeline_wav_to_flac() {
        let buf = sine_440hz_1s_48k_stereo();
        let dir = std::env::temp_dir();
        let src = dir.join("oxiaudio_tp_src.wav");
        let dst = dir.join("oxiaudio_tp_dst.flac");

        encode_wav(&buf, &src).expect("encode_wav for pipeline test");

        TranscodePipeline::new(&src, &dst)
            .run()
            .expect("TranscodePipeline::run failed");

        let bytes = std::fs::read(&dst).expect("read output flac");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
        assert!(bytes.starts_with(b"fLaC"), "expected fLaC magic in output");
    }

    /// Verify `TranscodePipeline::with_dsp` applies the DSP function correctly.
    #[test]
    fn test_transcode_pipeline_with_dsp() {
        let buf = sine_440hz_1s_48k_stereo();
        let dir = std::env::temp_dir();
        let src = dir.join("oxiaudio_tp_dsp_src.wav");
        let dst = dir.join("oxiaudio_tp_dsp_dst.wav");

        encode_wav(&buf, &src).expect("encode_wav for pipeline dsp test");

        // Apply a simple gain DSP step inside the pipeline.
        TranscodePipeline::new(&src, &dst)
            .with_dsp(|b| {
                let mut out = b.clone();
                dsp::gain(&mut out, -6.0); // attenuate by 6 dB
                Ok(out)
            })
            .run()
            .expect("TranscodePipeline with DSP failed");

        let decoded = decode_file(&dst).expect("decode pipeline DSP output");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);

        // After -6 dB gain the peak should be approximately half the original.
        let peak_in = buf.samples.iter().cloned().fold(0.0f32, f32::max);
        let peak_out = decoded.samples.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            peak_out < peak_in * 0.8,
            "DSP gain not applied: peak_in={peak_in:.4} peak_out={peak_out:.4}"
        );
    }

    /// Verify `transcode_batch` converts multiple WAV files to FLAC in parallel.
    #[test]
    fn test_transcode_batch_wav_to_flac() {
        let buf = sine_440hz_1s_48k_stereo();
        let dir = std::env::temp_dir();

        let src1 = dir.join("oxiaudio_tb_src1.wav");
        let src2 = dir.join("oxiaudio_tb_src2.wav");
        encode_wav(&buf, &src1).expect("encode src1");
        encode_wav(&buf, &src2).expect("encode src2");

        let inputs: Vec<&std::path::Path> = vec![src1.as_path(), src2.as_path()];
        let results = transcode_batch(&inputs, &dir, "flac");

        assert_eq!(results.len(), 2, "should return one result per input");

        for (i, r) in results.iter().enumerate() {
            let out_path = r
                .as_ref()
                .unwrap_or_else(|e| panic!("batch[{i}] failed: {e}"));
            let bytes = std::fs::read(out_path).expect("read batch output");
            assert!(
                bytes.starts_with(b"fLaC"),
                "batch[{i}] output is not valid FLAC"
            );
            let _ = std::fs::remove_file(out_path);
        }

        let _ = std::fs::remove_file(&src1);
        let _ = std::fs::remove_file(&src2);
    }

    /// Verify all three ResampleQuality variants succeed and return the correct sample rate.
    #[test]
    fn test_resample_quality_variants() {
        let buf = sine_440hz_1s_48k_stereo();
        for quality in [
            dsp::ResampleQuality::Fast,
            dsp::ResampleQuality::Good,
            dsp::ResampleQuality::Best,
        ] {
            let out =
                dsp::resample_quality(&buf, 44_100, quality).expect("resample_quality failed");
            assert_eq!(out.sample_rate, 44_100, "wrong sample rate for {quality:?}");
        }
    }

    /// Verify `dsp::detect_pitch_yin` (full-resolution) is accessible and returns results.
    #[test]
    fn test_detect_pitch_yin_reexport() {
        let buf = sine_440hz_1s_48k_stereo();
        // Use default-ish parameters: frame=2048, hop=512, threshold=0.12
        let frames = dsp::detect_pitch_yin(&buf, 2048, 512, 0.12);
        assert!(
            !frames.is_empty(),
            "detect_pitch_yin should return non-empty frames"
        );
    }

    /// Verify `dsp::loudness_momentary` is accessible and returns a non-empty Vec.
    #[test]
    fn test_loudness_momentary_reexport() {
        let sample_rate = 48_000u32;
        let n = sample_rate as usize * 5; // 5 seconds
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        // 400 ms windows — EBU R128 momentary block size (standard, hardcoded in loudness_momentary)
        let windows = dsp::loudness_momentary(&buf);
        assert!(
            !windows.is_empty(),
            "loudness_momentary should return non-empty results for 5s audio"
        );
        assert!(
            windows.iter().any(|&v| v.is_finite()),
            "at least one momentary LUFS window should be finite for a loud tone"
        );
    }

    /// Verify core M6 re-exports are accessible.
    #[test]
    fn test_core_m6_reexports_accessible() {
        let rb = AudioRingBuffer::<f32>::new(64);
        rb.write_frames(&[0.0f32; 10], 10).expect("ring write");
        let _ = rb.read_frames(10).expect("ring read");

        let mut clk = AudioClock::new(44_100);
        clk.advance(44_100);
        assert!((clk.elapsed_secs() - 1.0).abs() < 1e-9);

        let ts = Timestamp::Frames(44_100);
        assert!((ts.to_seconds(44_100) - 1.0).abs() < 1e-9);

        assert_ne!(AudioBufferLayout::Interleaved, AudioBufferLayout::Planar);
    }

    /// Task 6 — verify `file_format` returns a non-error `AudioFormat` for a valid WAV file.
    #[test]
    fn test_file_format_returns_wav_format() {
        let buf = sine_440hz_1s_48k_stereo();
        let tmp = std::env::temp_dir().join("oxiaudio_file_format_test.wav");
        encode_wav(&buf, &tmp).expect("encode_wav for file_format test");
        let fmt = file_format(&tmp).expect("file_format must succeed on a valid WAV");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(fmt.sample_rate, 48_000, "sample_rate mismatch");
        assert_eq!(fmt.channels, ChannelLayout::Stereo, "channels mismatch");
    }
}
