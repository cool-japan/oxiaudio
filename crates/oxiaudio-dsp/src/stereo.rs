//! Mid-side stereo encode/decode.
use oxiaudio_core::{AudioBuffer, ChannelLayout, OxiAudioError};

/// Encode a stereo L/R buffer to mid-side (M/S) representation.
///
/// `mid[n]  = (L[n] + R[n]) * 0.5`
/// `side[n] = (L[n] - R[n]) * 0.5`
///
/// The returned buffer has the same layout as the input (interleaved stereo),
/// with M in the left channel slot and S in the right channel slot.
///
/// # Errors
///
/// Returns `OxiAudioError::InvalidChannelLayout` if the input is not stereo.
#[must_use = "discarding the Result ignores encode errors"]
pub fn ms_encode(buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if buf.channels != ChannelLayout::Stereo {
        return Err(OxiAudioError::InvalidChannelLayout(format!(
            "ms_encode requires stereo input, got {:?}",
            buf.channels
        )));
    }

    let n_frames = buf.samples.len() / 2;
    let mut out_samples = Vec::with_capacity(buf.samples.len());

    for frame in 0..n_frames {
        let l = buf.samples[frame * 2];
        let r = buf.samples[frame * 2 + 1];
        let mid = (l + r) * 0.5;
        let side = (l - r) * 0.5;
        out_samples.push(mid);
        out_samples.push(side);
    }

    Ok(AudioBuffer {
        samples: out_samples,
        sample_rate: buf.sample_rate,
        channels: ChannelLayout::Stereo,
        format: buf.format,
    })
}

/// Decode a mid-side (M/S) buffer back to stereo L/R.
///
/// `L[n] = M[n] + S[n]`
/// `R[n] = M[n] - S[n]`
///
/// # Errors
///
/// Returns `OxiAudioError::InvalidChannelLayout` if the input is not stereo.
#[must_use = "discarding the Result ignores decode errors"]
pub fn ms_decode(buf: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
    if buf.channels != ChannelLayout::Stereo {
        return Err(OxiAudioError::InvalidChannelLayout(format!(
            "ms_decode requires stereo input, got {:?}",
            buf.channels
        )));
    }

    let n_frames = buf.samples.len() / 2;
    let mut out_samples = Vec::with_capacity(buf.samples.len());

    for frame in 0..n_frames {
        let mid = buf.samples[frame * 2];
        let side = buf.samples[frame * 2 + 1];
        let l = mid + side;
        let r = mid - side;
        out_samples.push(l);
        out_samples.push(r);
    }

    Ok(AudioBuffer {
        samples: out_samples,
        sample_rate: buf.sample_rate,
        channels: ChannelLayout::Stereo,
        format: buf.format,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::SampleFormat;

    fn stereo_buf(samples: Vec<f32>) -> AudioBuffer<f32> {
        AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_ms_roundtrip() {
        // Arbitrary stereo signal: encode then decode should recover original within 1e-6.
        let original = stereo_buf(vec![0.8f32, -0.3, 0.5, 0.1, -0.2, 0.7, 0.0, -1.0]);
        let encoded = ms_encode(&original).expect("encode failed");
        let decoded = ms_decode(&encoded).expect("decode failed");

        assert_eq!(original.samples.len(), decoded.samples.len());
        for (i, (&orig, &dec)) in original
            .samples
            .iter()
            .zip(decoded.samples.iter())
            .enumerate()
        {
            assert!(
                (orig - dec).abs() < 1e-6,
                "sample {i}: expected {orig}, got {dec}, diff={}",
                (orig - dec).abs()
            );
        }
    }

    #[test]
    fn test_ms_encode_mono_err() {
        // Mono input should return an Err.
        let mono = AudioBuffer {
            samples: vec![0.5f32, 0.3, 0.1],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let result = ms_encode(&mono);
        assert!(result.is_err(), "mono input should return Err");
        matches!(result, Err(OxiAudioError::InvalidChannelLayout(_)));
    }

    #[test]
    fn test_ms_encode_correlated_stereo() {
        // Identical L/R channels => side channel should be all zeros.
        let samples: Vec<f32> = (0..8)
            .flat_map(|i| {
                let v = i as f32 * 0.1;
                [v, v]
            })
            .collect();
        let buf = stereo_buf(samples);
        let encoded = ms_encode(&buf).expect("encode failed");

        // Side channel is at odd indices.
        for frame in 0..(encoded.samples.len() / 2) {
            let side = encoded.samples[frame * 2 + 1];
            assert!(
                side.abs() < 1e-7,
                "frame {frame}: expected side=0.0, got {side}"
            );
        }
    }

    #[test]
    fn test_ms_decode_anti_correlated() {
        // M=0, S=1 => L = M+S = 1, R = M-S = -1.
        let ms_buf = stereo_buf(vec![0.0f32, 1.0]);
        let decoded = ms_decode(&ms_buf).expect("decode failed");

        assert_eq!(decoded.samples.len(), 2, "expected 2 samples (1 frame)");
        let l = decoded.samples[0];
        let r = decoded.samples[1];
        assert!((l - 1.0f32).abs() < 1e-7, "expected L=1.0, got {l}");
        assert!((r - (-1.0f32)).abs() < 1e-7, "expected R=-1.0, got {r}");
    }
}
