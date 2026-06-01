#![forbid(unsafe_code)]

pub mod buffer;
pub mod clock;
pub mod error;
pub mod format;
pub mod ipc;
pub mod layout;
pub mod metadata;
pub mod pipeline;
pub mod ring;
pub mod sample;
pub mod traits;

pub use buffer::{
    downmix_51_to_stereo, downmix_to_mono, from_planar, from_planar_into, from_planar_unchecked,
    to_planar, upmix_mono_to_stereo, AudioBuffer,
};
pub use clock::{AudioClock, Timestamp};
pub use error::OxiAudioError;
pub use format::{AudioBufferLayout, AudioFormat, SampleFormat};
pub use ipc::{
    deserialize_audio_buffer_f32, from_ipc_bytes, serialize_audio_buffer_f32, to_ipc_bytes,
};
pub use layout::{ChannelId, ChannelLayout, ChannelMap};
pub use metadata::AudioMetadata;
pub use pipeline::{AudioNode, AudioPipeline, ParallelBranchNode};
pub use ring::AudioRingBuffer;
pub use sample::Sample;
pub use traits::{
    AudioDecoder, AudioEncoder, AudioFilter, AudioSink, AudioSource, StreamingDecoder,
};

#[cfg(all(feature = "serde", test))]
mod serde_tests {
    use super::*;

    #[test]
    fn audio_buffer_serde_roundtrip_json() {
        let buf = AudioBuffer {
            samples: vec![0.0f32, 0.5, -0.5, 1.0, -1.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let json = serde_json::to_string(&buf).expect("serialize must succeed");
        let decoded: AudioBuffer<f32> =
            serde_json::from_str(&json).expect("deserialize must succeed");
        assert_eq!(decoded.sample_rate, buf.sample_rate);
        assert_eq!(decoded.channels, buf.channels);
        assert_eq!(decoded.format, buf.format);
        for (a, b) in buf.samples.iter().zip(decoded.samples.iter()) {
            assert!((a - b).abs() < 1e-7, "sample mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn audio_metadata_serde_roundtrip_json() {
        let meta = AudioMetadata {
            title: Some("Test Track".into()),
            artist: Some("Artist".into()),
            year: Some(2026),
            bitrate_kbps: Some(320),
            ..Default::default()
        };
        let json = serde_json::to_string(&meta).expect("serialize must succeed");
        let decoded: AudioMetadata = serde_json::from_str(&json).expect("deserialize must succeed");
        assert_eq!(decoded, meta);
    }

    #[test]
    fn sample_format_serde_roundtrip() {
        for fmt in [
            SampleFormat::U8,
            SampleFormat::I16,
            SampleFormat::I24,
            SampleFormat::I32,
            SampleFormat::F32,
            SampleFormat::F64,
        ] {
            let json = serde_json::to_string(&fmt).expect("serialize must succeed");
            let decoded: SampleFormat =
                serde_json::from_str(&json).expect("deserialize must succeed");
            assert_eq!(decoded, fmt);
        }
    }

    #[test]
    fn channel_layout_serde_roundtrip() {
        for layout in [
            ChannelLayout::Mono,
            ChannelLayout::Stereo,
            ChannelLayout::Quad,
            ChannelLayout::Surround51,
            ChannelLayout::Surround71,
        ] {
            let json = serde_json::to_string(&layout).expect("serialize must succeed");
            let decoded: ChannelLayout =
                serde_json::from_str(&json).expect("deserialize must succeed");
            assert_eq!(decoded, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_io() {
        let e = OxiAudioError::Io(std::io::Error::other("test-io"));
        let s = e.to_string();
        assert!(!s.is_empty());
        assert!(s.contains("test-io"), "got: {s}");
    }

    #[test]
    fn error_display_decode() {
        let e = OxiAudioError::Decode("decode-msg".into());
        let s = e.to_string();
        assert!(s.contains("decode-msg"), "got: {s}");
    }

    #[test]
    fn error_display_encode() {
        let e = OxiAudioError::Encode("encode-msg".into());
        let s = e.to_string();
        assert!(s.contains("encode-msg"), "got: {s}");
    }

    #[test]
    fn error_display_unsupported() {
        let e = OxiAudioError::UnsupportedFormat("fmt-msg".into());
        let s = e.to_string();
        assert!(s.contains("fmt-msg"), "got: {s}");
    }

    #[test]
    fn empty_audiobuffer_is_valid() {
        let buf = AudioBuffer::<f32> {
            samples: vec![],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        assert_eq!(buf.samples.len(), 0);
        assert_eq!(buf.sample_rate, 44_100);
    }

    #[test]
    fn audioformat_audiometadata_derives() {
        let af = AudioFormat {
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let af2 = af;
        assert_eq!(af, af2);
        let af3 = AudioFormat {
            sample_rate: 44_100,
            ..af
        };
        assert_ne!(af, af3);

        let am = AudioMetadata {
            title: Some("Test".into()),
            ..Default::default()
        };
        let am2 = am.clone();
        assert_eq!(am, am2);
    }

    #[test]
    fn audiometadata_default_all_none() {
        let m = AudioMetadata::default();
        assert!(m.title.is_none());
        assert!(m.artist.is_none());
        assert!(m.album.is_none());
        assert!(m.duration_secs.is_none());
        assert!(m.bitrate_kbps.is_none());
    }

    #[test]
    fn test_audiobuffer_clone() {
        let buf = AudioBuffer {
            samples: vec![0.1f32, 0.2, 0.3],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let cloned = buf.clone();
        assert_eq!(buf.samples, cloned.samples);
        assert_eq!(buf.sample_rate, cloned.sample_rate);
        assert_eq!(buf.channels, cloned.channels);
        assert_eq!(buf.format, cloned.format);
    }

    #[test]
    fn test_to_f64_and_back() {
        let original = AudioBuffer {
            samples: vec![0.0f32, 0.5, -0.5, 1.0, -1.0],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let f64_buf = original.to_f64();
        assert_eq!(f64_buf.format, SampleFormat::F64);
        let back = f64_buf.to_f32();
        assert_eq!(back.format, SampleFormat::F32);
        for (a, b) in original.samples.iter().zip(back.samples.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_i16_conversion_roundtrip() {
        let original = AudioBuffer {
            samples: vec![0i16, 1000, -1000, i16::MAX, i16::MIN + 1],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::I16,
        };
        let f32_buf = AudioBuffer::<f32>::from(&original);
        assert_eq!(f32_buf.format, SampleFormat::F32);
        let back = AudioBuffer::<i16>::from(&f32_buf);
        assert_eq!(back.format, SampleFormat::I16);
        for (a, b) in original.samples.iter().zip(back.samples.iter()) {
            assert!(
                ((*a as i32) - (*b as i32)).abs() <= 1,
                "mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_i32_conversion_roundtrip() {
        let original = AudioBuffer {
            samples: vec![0i32, 100_000, -100_000, i32::MAX / 2],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::I32,
        };
        let f32_buf = AudioBuffer::<f32>::from(&original);
        assert_eq!(f32_buf.format, SampleFormat::F32);
        let back = AudioBuffer::<i32>::from(&f32_buf);
        assert_eq!(back.format, SampleFormat::I32);
        for (a, b) in original.samples.iter().zip(back.samples.iter()) {
            let diff = ((*a as i64) - (*b as i64)).abs();
            assert!(diff <= 1024, "mismatch: {a} vs {b} (diff {diff})");
        }
    }

    #[test]
    fn test_split_and_from_planar() {
        let samples = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let buf = AudioBuffer {
            samples: samples.clone(),
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let planes = buf.split_to_planar();
        assert_eq!(planes.len(), 2);
        assert_eq!(planes[0], vec![0.1f32, 0.3, 0.5]);
        assert_eq!(planes[1], vec![0.2f32, 0.4, 0.6]);
        let rebuilt = AudioBuffer::<f32>::from_planar(planes, 48_000, SampleFormat::F32);
        assert_eq!(rebuilt.samples, samples);
        assert_eq!(rebuilt.channels, ChannelLayout::Stereo);
    }

    #[test]
    fn test_sample_format_f64_variant() {
        assert_ne!(SampleFormat::F64, SampleFormat::F32);
        assert_ne!(SampleFormat::F64, SampleFormat::I16);
        assert_ne!(SampleFormat::F64, SampleFormat::I32);
        assert_eq!(SampleFormat::F64, SampleFormat::F64);
    }

    #[test]
    fn channel_layout_count() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
    }

    #[test]
    fn display_channel_layout() {
        assert_eq!(ChannelLayout::Mono.to_string(), "mono");
        assert_eq!(ChannelLayout::Stereo.to_string(), "stereo");
    }

    #[test]
    fn display_sample_format() {
        assert_eq!(SampleFormat::F32.to_string(), "f32");
        assert_eq!(SampleFormat::I16.to_string(), "i16");
        assert_eq!(SampleFormat::I32.to_string(), "i32");
        assert_eq!(SampleFormat::F64.to_string(), "f64");
    }

    #[test]
    fn duration_secs_stereo() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 96_000],
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let d = buf.duration_secs();
        assert!((d - 1.0).abs() < 1e-6, "expected 1.0s, got {d}");
    }

    #[test]
    fn frame_count_mono() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 1024],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        assert_eq!(buf.frame_count(), 1024);
    }

    #[test]
    fn is_empty_empty_and_non_empty() {
        let empty = AudioBuffer::<f32> {
            samples: vec![],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        assert!(empty.is_empty());
        let non = AudioBuffer {
            samples: vec![0.0f32],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        assert!(!non.is_empty());
    }

    #[test]
    fn silence_is_zeros() {
        let buf = AudioBuffer::<f32>::silence(48_000, ChannelLayout::Stereo, 512);
        assert_eq!(buf.frame_count(), 512);
        assert!(buf.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn slice_frames_sub_range() {
        let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let sliced = buf.slice_frames(2, 5);
        assert_eq!(sliced.samples, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn append_success() {
        let mut a = AudioBuffer {
            samples: vec![1.0f32, 2.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![3.0f32, 4.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        a.append(&b).unwrap();
        assert_eq!(a.samples, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn append_channel_mismatch() {
        let mut a = AudioBuffer {
            samples: vec![0.0f32],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![0.0f32, 0.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        assert!(matches!(
            a.append(&b),
            Err(OxiAudioError::InvalidChannelLayout(_))
        ));
    }

    #[test]
    fn append_sample_rate_mismatch() {
        let mut a = AudioBuffer {
            samples: vec![0.0f32],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![0.0f32],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        assert!(matches!(
            a.append(&b),
            Err(OxiAudioError::InvalidSampleRate(_))
        ));
    }

    #[test]
    fn peak_and_rms_amplitude() {
        let buf = AudioBuffer {
            samples: vec![0.0f32, 0.5, -1.0, 0.25],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        assert!((buf.peak_amplitude() - 1.0).abs() < 1e-6);
        let expected_rms = ((0.0 + 0.25 + 1.0 + 0.0625) / 4.0f32).sqrt();
        assert!((buf.rms_amplitude() - expected_rms).abs() < 1e-5);
    }

    #[test]
    fn peak_db_rms_db() {
        let buf = AudioBuffer {
            samples: vec![0.0f32, 1.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        assert!((buf.peak_db() - 0.0).abs() < 1e-5);
        let empty = AudioBuffer::<f32>::silence(44_100, ChannelLayout::Mono, 0);
        assert!(empty.peak_db().is_infinite());
    }

    #[test]
    fn fade_in_first_sample_is_zero() {
        let mut buf = AudioBuffer {
            samples: vec![1.0f32; 100],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        buf.fade_in(50);
        assert!(
            buf.samples[0].abs() < 1e-6,
            "first sample after fade_in should be ~0"
        );
        assert!(
            (buf.samples[99] - 1.0).abs() < 1e-5,
            "sample after fade region should be unchanged"
        );
    }

    #[test]
    fn fade_out_last_sample_is_near_zero() {
        let mut buf = AudioBuffer {
            samples: vec![1.0f32; 100],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        buf.fade_out(50);
        assert!(
            buf.samples[99].abs() < 0.1,
            "last sample after fade_out should be near 0, got {}",
            buf.samples[99]
        );
        assert!(
            (buf.samples[0] - 1.0).abs() < 1e-5,
            "sample before fade region should be unchanged"
        );
    }

    #[test]
    fn error_invalid_channel_layout() {
        let e = OxiAudioError::InvalidChannelLayout("test".into());
        assert!(e.to_string().contains("invalid channel layout"));
    }

    #[test]
    fn error_invalid_sample_rate() {
        let e = OxiAudioError::InvalidSampleRate("test".into());
        assert!(e.to_string().contains("invalid sample rate"));
    }

    #[test]
    fn error_buffer_over_underflow() {
        assert!(OxiAudioError::BufferOverflow("x".into())
            .to_string()
            .contains("buffer overflow"));
        assert!(OxiAudioError::BufferUnderflow("y".into())
            .to_string()
            .contains("buffer underflow"));
    }

    #[test]
    fn sample_format_metadata() {
        assert_eq!(SampleFormat::U8.bit_depth(), 8);
        assert_eq!(SampleFormat::I16.bit_depth(), 16);
        assert_eq!(SampleFormat::I24.bit_depth(), 24);
        assert_eq!(SampleFormat::I32.bit_depth(), 32);
        assert_eq!(SampleFormat::F32.bit_depth(), 32);
        assert_eq!(SampleFormat::F64.bit_depth(), 64);

        assert!(SampleFormat::F32.is_float());
        assert!(SampleFormat::F64.is_float());
        assert!(!SampleFormat::I16.is_float());
        assert!(SampleFormat::I24.is_integer());

        assert_eq!(SampleFormat::U8.byte_size(), 1);
        assert_eq!(SampleFormat::I16.byte_size(), 2);
        assert_eq!(SampleFormat::I24.byte_size(), 3);
        assert_eq!(SampleFormat::I32.byte_size(), 4);
        assert_eq!(SampleFormat::F64.byte_size(), 8);
    }

    #[test]
    fn sample_format_from_str() {
        assert_eq!("f32".parse::<SampleFormat>().unwrap(), SampleFormat::F32);
        assert_eq!("I16".parse::<SampleFormat>().unwrap(), SampleFormat::I16);
        assert_eq!(SampleFormat::try_from("i24").unwrap(), SampleFormat::I24);
        assert_eq!("double".parse::<SampleFormat>().unwrap(), SampleFormat::F64);
        assert!("bogus".parse::<SampleFormat>().is_err());
    }

    #[test]
    fn channel_layout_from_u16() {
        assert_eq!(ChannelLayout::from(1u16), ChannelLayout::Mono);
        assert_eq!(ChannelLayout::from(2u16), ChannelLayout::Stereo);
        assert_eq!(ChannelLayout::from(4u16), ChannelLayout::Quad);
        assert_eq!(ChannelLayout::from(6u16), ChannelLayout::Surround51);
        assert_eq!(ChannelLayout::from(8u16), ChannelLayout::Surround71);
        // Unknown channel counts fall back to Stereo
        assert_eq!(ChannelLayout::from(3u16), ChannelLayout::Stereo);
        assert_eq!(ChannelLayout::from(99u16), ChannelLayout::Stereo);
    }

    #[test]
    fn mix_with_additive() {
        let mut a = AudioBuffer {
            samples: vec![0.1f32, 0.2, 0.3, 0.4],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![1.0f32, 1.0, 1.0, 1.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        a.mix_with(&b, 0.5).unwrap();
        assert!((a.samples[0] - 0.6).abs() < 1e-6);
        assert!((a.samples[3] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn mix_with_mismatch_errors() {
        let mut a = AudioBuffer {
            samples: vec![0.0f32],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![0.0f32, 0.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        assert!(a.mix_with(&b, 1.0).is_err());
    }

    #[test]
    fn crossfade_length_and_endpoints() {
        // Two mono ramps; overlap 4 frames.
        let a = AudioBuffer {
            samples: vec![1.0f32; 10],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![2.0f32; 8],
            sample_rate: 48_000,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let xf = AudioBuffer::crossfade(&a, &b, 4).unwrap();
        // 10 + 8 - 4 = 14 frames.
        assert_eq!(xf.frame_count(), 14);
        // Head is pure a.
        assert!((xf.samples[0] - 1.0).abs() < 1e-6);
        // Tail is pure b.
        assert!((xf.samples[13] - 2.0).abs() < 1e-6);
        // Equal-power crossfade of two correlated DC signals peaks at
        // 1*cos(theta) + 2*sin(theta), max ~= sqrt(1^2 + 2^2) = 2.236; stay bounded.
        for &s in &xf.samples {
            assert!(
                (0.9..=2.3).contains(&s),
                "crossfade sample out of range: {s}"
            );
        }
    }

    // ------------------------------------------------------------------
    // M6 new tests
    // ------------------------------------------------------------------

    #[test]
    fn sample_trait_roundtrip_i16() {
        use crate::sample::Sample;
        let v: i16 = 16383;
        let f = v.to_f32();
        let back = i16::from_f32(f);
        assert!(
            (v as i32 - back as i32).abs() <= 1,
            "i16 roundtrip: {v} vs {back}"
        );
    }

    #[test]
    fn sample_trait_roundtrip_i32() {
        use crate::sample::Sample;
        let v: i32 = 1_000_000_000;
        let f = v.to_f32();
        let back = i32::from_f32(f);
        assert!(
            (v as i64 - back as i64).abs() <= 1024,
            "i32 roundtrip: {v} vs {back}"
        );
    }

    #[test]
    fn sample_trait_roundtrip_f64() {
        use crate::sample::Sample;
        let v: f64 = 0.75;
        let f = v.to_f32();
        let back = f64::from_f32(f);
        assert!((back - v).abs() < 1e-6, "f64 roundtrip: {v} vs {back}");
    }

    #[test]
    fn sample_u8_from_f32_boundaries() {
        use crate::sample::Sample;
        assert_eq!(u8::from_f32(0.0), 128u8);
        assert_eq!(u8::from_f32(1.0), 255u8);
        // -1.0: clamp(-1,1)*127 + 128 = 1.0 → rounds to 1
        assert_eq!(u8::from_f32(-1.0), 1u8);
        // midpoint 128 maps to 0.0
        let f = u8::EQUILIBRIUM.to_f32();
        assert!(f.abs() < 0.01, "128 should map near 0.0, got {f}");
    }

    #[test]
    fn display_u8_i24_formats() {
        assert_eq!(SampleFormat::U8.to_string(), "u8");
        assert_eq!(SampleFormat::I24.to_string(), "i24");
    }

    #[test]
    fn try_from_str_sample_format() {
        assert_eq!(SampleFormat::try_from("f32").unwrap(), SampleFormat::F32);
        assert_eq!(SampleFormat::try_from("i24").unwrap(), SampleFormat::I24);
        assert!(matches!(
            SampleFormat::try_from("xyz"),
            Err(OxiAudioError::UnsupportedFormat(_))
        ));
    }

    #[test]
    fn channel_layout_from_u16_fallback() {
        assert_eq!(ChannelLayout::from(1u16), ChannelLayout::Mono);
        assert_eq!(ChannelLayout::from(2u16), ChannelLayout::Stereo);
        assert_eq!(ChannelLayout::from(5u16), ChannelLayout::Stereo);
    }

    #[test]
    fn ring_buffer_write_read_frames() {
        let rb = AudioRingBuffer::<f32>::new(256);
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        rb.write_frames(&data, 100).unwrap();
        let out = rb.read_frames(100).unwrap();
        assert_eq!(out.len(), 100);
        for (i, v) in out.iter().enumerate() {
            assert!((v - i as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn ring_buffer_overflow_error() {
        let rb = AudioRingBuffer::<f32>::new(8); // capacity = 8
        let data = vec![0.0f32; 8];
        rb.write_frames(&data, 8).unwrap();
        let extra = vec![0.0f32; 1];
        assert!(matches!(
            rb.write_frames(&extra, 1),
            Err(OxiAudioError::BufferOverflow(_))
        ));
    }

    #[test]
    fn ring_buffer_underflow_error() {
        let rb = AudioRingBuffer::<f32>::new(16);
        assert!(matches!(
            rb.read_frames(1),
            Err(OxiAudioError::BufferUnderflow(_))
        ));
    }

    #[test]
    fn audio_clock_advance_elapsed() {
        let mut clk = AudioClock::new(48_000);
        clk.advance(48_000);
        assert_eq!(clk.elapsed_frames(), 48_000);
        let secs = clk.elapsed_secs();
        assert!((secs - 1.0).abs() < 1e-9, "expected 1.0s, got {secs}");
    }

    #[test]
    fn audio_clock_drift_ppm_zero_when_exact() {
        let mut clk = AudioClock::new(48_000);
        clk.advance(48_000);
        // 48000 frames at 48kHz = exactly 1.0 s → nominal = 1e9 ns
        let wall_ns = 1_000_000_000u64; // exactly 1 second
        let drift = clk.drift_ppm_from_ns(wall_ns);
        assert!(
            drift.abs() < 1e-6,
            "drift should be 0.0 ppm when wall = nominal, got {drift}"
        );
    }

    #[test]
    fn audio_clock_drift_ppm_positive_when_fast() {
        // Wall clock ran slower (1.001 s) → audio is "ahead" → positive drift
        let mut clk = AudioClock::new(48_000);
        clk.advance(48_000);
        let wall_ns = 1_001_000_000u64; // 1.001 seconds
        let drift = clk.drift_ppm_from_ns(wall_ns);
        // Expected: (1e9 - 1.001e9) / 1.001e9 * 1e6 ≈ −999 ppm (audio behind wall)
        // Nominal 1e9 ns, actual 1.001e9 ns → drift ≈ −999 ppm
        assert!(
            drift < -990.0 && drift > -1010.0,
            "expected ~-999 ppm drift when wall is 1.001s, got {drift}"
        );
    }

    #[test]
    fn audio_clock_drift_ppm_zero_frames() {
        let clk = AudioClock::new(48_000);
        // No frames advanced → returns 0.0
        assert_eq!(clk.drift_ppm_from_ns(1_000_000_000), 0.0);
    }

    #[test]
    fn audio_clock_drift_ppm_zero_wall_ns() {
        let mut clk = AudioClock::new(48_000);
        clk.advance(48_000);
        // Zero wall ns → returns 0.0 (guard clause)
        assert_eq!(clk.drift_ppm_from_ns(0), 0.0);
    }

    #[test]
    fn timestamp_roundtrip() {
        let sr = 44_100u32;
        let ts_frames = Timestamp::Frames(44_100);
        assert!((ts_frames.to_seconds(sr) - 1.0).abs() < 1e-9);
        let ts_secs = Timestamp::Seconds(1.0);
        assert_eq!(ts_secs.to_frames(sr), 44_100);
    }

    #[test]
    fn audio_pipeline_bypass_passes_through() {
        struct DoubleSamples;
        impl AudioNode for DoubleSamples {
            fn name(&self) -> &str {
                "double"
            }
            fn bypass(&self) -> bool {
                true
            }
            fn process(&self, input: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
                let samples = input.samples.iter().map(|&s| s * 2.0).collect();
                Ok(AudioBuffer { samples, ..*input })
            }
        }
        let input = AudioBuffer {
            samples: vec![0.5f32, 0.5],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let pipeline = AudioPipeline::new().push_node(Box::new(DoubleSamples));
        let out = pipeline.process(&input).unwrap();
        assert_eq!(out.samples, input.samples);
    }

    #[test]
    fn audio_pipeline_processing_applies() {
        struct HalfGain;
        impl HalfGain {
            fn name(&self) -> &str {
                "half"
            }
        }
        impl AudioNode for HalfGain {
            fn name(&self) -> &str {
                self.name()
            }
            fn process(&self, input: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
                let samples = input.samples.iter().map(|&s| s * 0.5).collect();
                Ok(AudioBuffer { samples, ..*input })
            }
        }
        let input = AudioBuffer {
            samples: vec![1.0f32, 0.8],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let pipeline = AudioPipeline::new().push_node(Box::new(HalfGain));
        let out = pipeline.process(&input).unwrap();
        assert!((out.samples[0] - 0.5).abs() < 1e-6);
        assert!((out.samples[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn mix_with_correct_sum() {
        let mut a = AudioBuffer {
            samples: vec![0.2f32, 0.4],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![0.1f32, 0.2],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        a.mix_with(&b, 1.0).unwrap();
        assert!((a.samples[0] - 0.3).abs() < 1e-6);
        assert!((a.samples[1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn crossfade_smooth_transition() {
        let a = AudioBuffer {
            samples: vec![1.0f32; 100],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![0.0f32; 100],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let overlap = 20;
        let out = AudioBuffer::crossfade(&a, &b, overlap).unwrap();
        // Expected length: 100 + 100 - 20 = 180
        assert_eq!(out.frame_count(), 180);
        // First sample (before overlap) should be 1.0
        assert!((out.samples[0] - 1.0).abs() < 1e-5);
        // Last sample (b's tail) should be 0.0
        assert!(out.samples[179].abs() < 1e-5);
    }

    #[test]
    fn resample_linear_frame_count() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 44_100],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let out = buf.resample_linear(22_050);
        let diff = (out.frame_count() as i64 - 22_050i64).abs();
        assert!(
            diff <= 2,
            "expected ~22050 frames, got {}",
            out.frame_count()
        );
        assert_eq!(out.sample_rate, 22_050);
    }

    #[test]
    fn to_planar_from_planar_standalone_roundtrip() {
        let samples = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let buf = AudioBuffer {
            samples: samples.clone(),
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let planar = to_planar(&buf);
        assert_eq!(planar.len(), 2);
        let rebuilt = from_planar(&planar, 48_000).unwrap();
        assert_eq!(rebuilt.samples, samples);
        assert_eq!(rebuilt.channels, ChannelLayout::Stereo);
    }

    #[test]
    fn error_buffer_overflow_format() {
        let e = OxiAudioError::BufferOverflow("test".into());
        assert!(e.to_string().contains("buffer overflow"));
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn error_buffer_underflow_format() {
        let e = OxiAudioError::BufferUnderflow("test".into());
        assert!(e.to_string().contains("buffer underflow"));
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn audio_buffer_layout_variants() {
        assert_ne!(AudioBufferLayout::Interleaved, AudioBufferLayout::Planar);
        let x = AudioBufferLayout::Interleaved;
        let y = x;
        assert_eq!(x, y);
    }

    #[test]
    fn audio_format_is_copy() {
        let af = AudioFormat {
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let af2 = af; // Copy — no clone() needed
        assert_eq!(af, af2);
    }

    // ------------------------------------------------------------------
    // Surround channel layout tests
    // ------------------------------------------------------------------

    #[test]
    fn test_surround51_channel_count() {
        assert_eq!(ChannelLayout::Surround51.channel_count(), 6);
        assert_eq!(ChannelLayout::Surround71.channel_count(), 8);
        assert_eq!(ChannelLayout::Quad.channel_count(), 4);
        assert_eq!(ChannelLayout::Atmos714.channel_count(), 12);
        // Verify AudioBuffer::silence produces the correct interleaved sample count.
        let buf = AudioBuffer::<f32>::silence(44100, ChannelLayout::Surround51, 1000);
        assert_eq!(
            buf.samples.len(),
            6000,
            "5.1 silence: 6 channels * 1000 frames"
        );
        assert_eq!(
            buf.frame_count(),
            1000,
            "5.1 silence: frame_count must be 1000"
        );
    }

    #[test]
    fn test_channel_map_for_51() {
        let map = ChannelMap::for_layout(ChannelLayout::Surround51);
        assert_eq!(map.channel_count(), 6);
        assert_eq!(map.get(0), Some(ChannelId::FrontLeft));
        assert_eq!(map.get(2), Some(ChannelId::FrontCenter));
        assert_eq!(map.get(3), Some(ChannelId::LowFrequency));
    }

    #[test]
    fn test_downmix_51_to_stereo() {
        // 5.1 buffer with only center channel active: 1 frame
        // FL=0, FR=0, FC=1, LFE=0, RL=0, RR=0
        let samples = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0];
        let buf = AudioBuffer {
            samples,
            sample_rate: 44100,
            channels: ChannelLayout::Surround51,
            format: SampleFormat::F32,
        };
        let stereo = downmix_51_to_stereo(&buf).expect("downmix should succeed");
        assert_eq!(stereo.channels, ChannelLayout::Stereo);
        // L = 0 + 0.707*1 + 0.707*0 = 0.707
        assert!(
            (stereo.samples[0] - 0.707).abs() < 0.01,
            "L={}",
            stereo.samples[0]
        );
        assert!(
            (stereo.samples[1] - 0.707).abs() < 0.01,
            "R={}",
            stereo.samples[1]
        );
    }

    #[test]
    fn test_upmix_mono_to_stereo() {
        let buf = AudioBuffer {
            samples: vec![0.5f32, -0.5],
            sample_rate: 44100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let stereo = upmix_mono_to_stereo(&buf).expect("upmix should succeed");
        assert_eq!(stereo.channels, ChannelLayout::Stereo);
        assert_eq!(stereo.samples, vec![0.5f32, 0.5, -0.5, -0.5]);
    }

    #[test]
    fn test_channel_layout_display_surround() {
        assert_eq!(ChannelLayout::Surround51.to_string(), "5.1");
        assert_eq!(ChannelLayout::Surround71.to_string(), "7.1");
        assert_eq!(ChannelLayout::Quad.to_string(), "quad");
    }

    #[test]
    fn test_from_u16_surround() {
        assert_eq!(ChannelLayout::from(6u16), ChannelLayout::Surround51);
        assert_eq!(ChannelLayout::from(8u16), ChannelLayout::Surround71);
        assert_eq!(ChannelLayout::from(4u16), ChannelLayout::Quad);
    }

    #[test]
    fn test_downmix_to_mono_surround() {
        // 4-channel Quad: average of all channels → 0.4
        let buf = AudioBuffer {
            samples: vec![0.4f32, 0.4, 0.4, 0.4],
            sample_rate: 44100,
            channels: ChannelLayout::Quad,
            format: SampleFormat::F32,
        };
        let mono = downmix_to_mono(&buf);
        assert_eq!(mono.channels, ChannelLayout::Mono);
        assert!(
            (mono.samples[0] - 0.4).abs() < 1e-6,
            "mono={}",
            mono.samples[0]
        );
    }

    #[test]
    fn test_downmix_51_to_stereo_wrong_input() {
        let buf = AudioBuffer {
            samples: vec![0.0f32, 0.0],
            sample_rate: 44100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        assert!(matches!(
            downmix_51_to_stereo(&buf),
            Err(OxiAudioError::InvalidChannelLayout(_))
        ));
    }

    #[test]
    fn test_upmix_mono_to_stereo_wrong_input() {
        let buf = AudioBuffer {
            samples: vec![0.0f32, 0.0],
            sample_rate: 44100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        assert!(matches!(
            upmix_mono_to_stereo(&buf),
            Err(OxiAudioError::InvalidChannelLayout(_))
        ));
    }

    #[test]
    fn test_channel_map_index_of() {
        let map = ChannelMap::for_layout(ChannelLayout::Surround71);
        assert_eq!(map.index_of(ChannelId::FrontLeft), Some(0));
        assert_eq!(map.index_of(ChannelId::SideLeft), Some(6));
        assert_eq!(map.index_of(ChannelId::SideRight), Some(7));
        assert_eq!(map.index_of(ChannelId::TopFrontLeft), None);
    }

    // ------------------------------------------------------------------
    // M11 new tests: SampleFormat extensions + AudioBuffer utilities
    // ------------------------------------------------------------------

    #[test]
    fn test_sample_format_byte_size() {
        assert_eq!(SampleFormat::U8.byte_size(), 1);
        assert_eq!(SampleFormat::I16.byte_size(), 2);
        assert_eq!(SampleFormat::I24.byte_size(), 3);
        assert_eq!(SampleFormat::F32.byte_size(), 4);
        assert_eq!(SampleFormat::I32.byte_size(), 4);
        assert_eq!(SampleFormat::F64.byte_size(), 8);
    }

    #[test]
    fn test_sample_format_is_float() {
        assert!(SampleFormat::F32.is_float());
        assert!(SampleFormat::F64.is_float());
        assert!(!SampleFormat::I16.is_float());
        assert!(!SampleFormat::I24.is_float());
        assert!(!SampleFormat::U8.is_float());
    }

    #[test]
    fn test_sample_format_normalize_i32_to_f32() {
        // U8: midpoint 128 → 0.0, 0 → −1.0, 255 → ~0.996
        let fmt = SampleFormat::U8;
        assert!((fmt.normalize_i32_to_f32(128) - 0.0).abs() < 1e-5);
        assert!((fmt.normalize_i32_to_f32(0) - (-1.0)).abs() < 1e-5);
        // I16: max → 1.0
        assert!((SampleFormat::I16.normalize_i32_to_f32(i16::MAX as i32) - 1.0).abs() < 1e-5);
        // I24: max → 1.0
        assert!(
            (SampleFormat::I24.normalize_i32_to_f32(8_388_607) - 1.0).abs() < 1e-5,
            "I24 max should normalize to ~1.0"
        );
        // I32: max → 1.0
        assert!((SampleFormat::I32.normalize_i32_to_f32(i32::MAX) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_audio_buffer_mixed_with() {
        let a = AudioBuffer {
            samples: vec![1.0f32, 1.0, 1.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![0.5f32, 0.5, 0.5],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let mixed = a.mixed_with(&b, 1.0).unwrap();
        assert!((mixed.samples[0] - 1.5).abs() < 1e-6);
        assert_eq!(mixed.samples.len(), 3);
    }

    #[test]
    fn test_audio_buffer_mixed_with_channel_mismatch() {
        let a = AudioBuffer {
            samples: vec![1.0f32],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![1.0f32, 1.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        assert!(a.mixed_with(&b, 1.0).is_err());
    }

    #[test]
    fn test_audio_buffer_mixed_with_length_extension() {
        // When other is longer, output extends with zero-padded self.
        let a = AudioBuffer {
            samples: vec![1.0f32; 3],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![0.5f32; 6],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let out = a.mixed_with(&b, 1.0).unwrap();
        assert_eq!(out.samples.len(), 6);
        // Tail (frames 3-5): 0 + 0.5 * 1.0 = 0.5
        assert!((out.samples[5] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_reverse() {
        let mut buf = AudioBuffer {
            samples: vec![1.0f32, 2.0, 3.0, 4.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        buf.reverse();
        assert_eq!(buf.samples, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_audio_buffer_reversed_stereo() {
        // Stereo: frames = [L0,R0, L1,R1, L2,R2]
        let buf = AudioBuffer {
            samples: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let r = buf.reversed();
        // reversed frames: [L2,R2, L1,R1, L0,R0]
        assert_eq!(r.samples, vec![5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
        // Original must be unchanged.
        assert_eq!(buf.samples, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_audio_buffer_linear_crossfade() {
        let a = AudioBuffer {
            samples: vec![1.0f32; 100],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![0.0f32; 100],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let out = a.linear_crossfade(&b, 50).unwrap();
        // Output length = 100 + 100 - 50 = 150
        assert_eq!(out.frame_count(), 150);
        // First sample = 1.0 (from a, before fade)
        assert!((out.samples[0] - 1.0).abs() < 1e-5);
        // Last sample = 0.0 (from b, after fade)
        assert!(out.samples[149].abs() < 1e-5);
    }

    #[test]
    fn test_audio_buffer_linear_crossfade_mismatch() {
        let a = AudioBuffer {
            samples: vec![1.0f32],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let b = AudioBuffer {
            samples: vec![1.0f32, 1.0],
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        assert!(a.linear_crossfade(&b, 1).is_err());
    }

    #[test]
    fn test_audio_buffer_gain_ramp() {
        let mut buf = AudioBuffer {
            samples: vec![1.0f32; 101],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        buf.gain_ramp(0.0, 1.0);
        assert!(buf.samples[0].abs() < 1e-5, "start should be 0");
        assert!((buf.samples[100] - 1.0).abs() < 1e-5, "end should be 1.0");
    }

    #[test]
    fn test_audio_buffer_gain_ramp_constant() {
        let mut buf = AudioBuffer {
            samples: vec![0.5f32; 10],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        buf.gain_ramp(2.0, 2.0);
        for &s in &buf.samples {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "constant ramp should double all samples"
            );
        }
    }
}
