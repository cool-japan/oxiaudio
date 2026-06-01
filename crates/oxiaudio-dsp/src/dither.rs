//! TPDF (triangular probability density function) noise-shaped dithering.

use oxiaudio_core::{AudioBuffer, SampleFormat};

/// Apply TPDF dithering to `buf` in preparation for quantization to `bit_depth` bits.
///
/// Adds triangular-distributed noise with peak amplitude of ±1 LSB at the target
/// bit depth. This reduces quantization distortion to shaped noise that is
/// perceptually less audible than truncation.
///
/// The input is expected to be in the range `[-1.0, 1.0]`. The dithered output
/// remains in `[-1.0, 1.0]` with noise added before quantization.
///
/// Use before encoding to I16 (pass `bit_depth = 16`) or I24 (pass `bit_depth = 24`).
///
/// # Panics
///
/// Does not panic (uses safe PRNG with wrapping arithmetic).
pub fn apply_tpdf_dither(buf: &AudioBuffer<f32>, bit_depth: u32) -> AudioBuffer<f32> {
    let lsb = 2.0_f32 / (1u64 << bit_depth) as f32;
    let mut samples = buf.samples.clone();
    let mut state: u64 = 6_364_136_223_846_793_005; // LCG seed
    for s in &mut samples {
        // Two uniform random numbers in [0, 1) from a simple LCG PRNG
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[allow(clippy::cast_precision_loss)]
        let r1 = (state >> 33) as f32 / (1u32 << 31) as f32 - 0.5;
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[allow(clippy::cast_precision_loss)]
        let r2 = (state >> 33) as f32 / (1u32 << 31) as f32 - 0.5;
        // TPDF noise = (r1 + r2) * lsb — triangular distribution, peak ±lsb
        *s = (*s + (r1 + r2) * lsb).clamp(-1.0, 1.0);
    }
    AudioBuffer {
        samples,
        sample_rate: buf.sample_rate,
        channels: buf.channels,
        format: SampleFormat::F32,
    }
}

/// Apply ATH-weighted (Absolute Threshold of Hearing) noise-shaped dithering.
///
/// Instead of flat triangular noise, this uses a 1st-order error-feedback noise
/// shaper that pushes quantization noise toward frequencies where the ear is least
/// sensitive (high frequencies and very low frequencies).
///
/// # Algorithm
///
/// For each sample `x`:
/// 1. Generate TPDF noise `d = (r1 + r2) * lsb`.
/// 2. Apply feedback shaping: `d_shaped = d + hp_coeff * e_prev`.
/// 3. `x_dithered = (x + d_shaped).clamp(-1.0, 1.0)`.
/// 4. `x_quantized = quantize(x_dithered, bit_depth)`.
/// 5. `e_prev = x_quantized - x_dithered` (fed back on the next sample).
/// 6. Output `x_dithered` (pre-quantization float, noise-shaped).
///
/// `hp_coeff = -0.5` pushes noise energy toward higher frequencies.
///
/// Processing is per-channel: interleaved samples are split into channels,
/// each channel is processed with its own independent feedback state, then
/// re-interleaved.
///
/// # Parameters
///
/// - `buf`         — input buffer (f32, range `[-1.0, 1.0]`).
/// - `bit_depth`   — target bit depth (e.g., 16 or 24).
/// - `sample_rate` — sample rate in Hz (retained in the output buffer metadata).
#[must_use = "returns dithered buffer"]
pub fn apply_noise_shaped_dither(
    buf: &AudioBuffer<f32>,
    bit_depth: u32,
    sample_rate: u32,
) -> AudioBuffer<f32> {
    use oxiaudio_core::SampleFormat;

    let lsb = 2.0_f32 / (1u64 << bit_depth) as f32;
    // High-pass feedback coefficient: pushes noise toward high frequencies.
    let hp_coeff = -0.5_f32;

    let n_channels = buf.channels.channel_count();
    let n_frames = buf.samples.len() / n_channels;

    // Split interleaved samples into per-channel Vecs.
    let mut channels: Vec<Vec<f32>> = (0..n_channels)
        .map(|ch| {
            (0..n_frames)
                .map(|f| buf.samples[f * n_channels + ch])
                .collect()
        })
        .collect();

    // Seed the LCG from the buffer length — same approach as apply_tpdf_dither.
    let base_seed = buf.samples.len() as u64 ^ 0xDEAD_BEEF;

    // Process each channel independently.
    for (ch_idx, ch_samples) in channels.iter_mut().enumerate() {
        // Unique seed per channel so channels aren't correlated.
        let mut state: u64 = base_seed
            .wrapping_add(ch_idx as u64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);

        let mut e_prev = 0.0_f32;

        for s in ch_samples.iter_mut() {
            // Two uniform LCG samples → TPDF noise in [-lsb, +lsb].
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let r1 = (state >> 33) as f32 / (1u32 << 31) as f32 - 0.5;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let r2 = (state >> 33) as f32 / (1u32 << 31) as f32 - 0.5;

            // TPDF base noise + high-pass error feedback.
            let d_shaped = (r1 + r2) * lsb + hp_coeff * e_prev;

            let x_dithered = (*s + d_shaped).clamp(-1.0, 1.0);

            // Quantize: round to nearest LSB step, then back to float.
            let x_quantized = (x_dithered / lsb).round() * lsb;

            // Quantization error fed back to next sample.
            e_prev = x_quantized - x_dithered;

            // Output the pre-quantization dithered value (noise-shaped float).
            *s = x_dithered;
        }
    }

    // Re-interleave channels.
    let mut out_samples = vec![0.0_f32; n_frames * n_channels];
    for (ch, ch_data) in channels.iter().enumerate() {
        for (f, &v) in ch_data.iter().enumerate() {
            out_samples[f * n_channels + ch] = v;
        }
    }

    AudioBuffer {
        samples: out_samples,
        sample_rate,
        channels: buf.channels,
        format: SampleFormat::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    fn silent_buf(n: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![0.0; n],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_noise_shaped_dither_output_range() {
        // All output samples must remain in [-1.0, 1.0] regardless of input.
        let samples: Vec<f32> = (0..4410)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 44_100.0).sin() * 0.9)
            .collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let out = apply_noise_shaped_dither(&buf, 16, 44_100);
        for (i, &s) in out.samples.iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(&s),
                "sample {i} = {s} is out of [-1.0, 1.0]"
            );
        }
        // Also verify the output is non-silent (noise is added, not silenced).
        let rms: f32 =
            (out.samples.iter().map(|&x| x * x).sum::<f32>() / out.samples.len() as f32).sqrt();
        assert!(rms > 0.1, "output should not be silent: rms={rms}");
    }

    #[test]
    fn test_noise_shaped_dither_stereo() {
        // Verify stereo processing stays in range.
        let n = 2048usize;
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / 44_100.0;
            let s = (std::f32::consts::TAU * 1000.0 * t).sin() * 0.8;
            samples.push(s);
            samples.push(-s);
        }
        let buf = AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let out = apply_noise_shaped_dither(&buf, 16, 44_100);
        assert_eq!(out.samples.len(), n * 2);
        for &s in &out.samples {
            assert!((-1.0..=1.0).contains(&s), "stereo sample {s} out of range");
        }
    }

    #[test]
    fn test_tpdf_dither_16bit_noise_level() {
        let buf = silent_buf(44_100);
        let dithered = apply_tpdf_dither(&buf, 16);
        let rms: f32 = (dithered.samples.iter().map(|s| s * s).sum::<f32>()
            / dithered.samples.len() as f32)
            .sqrt();
        // RMS of TPDF noise at 16 bit: approx lsb/sqrt(6) ≈ 2/65536/sqrt(6) ≈ 1.25e-5
        let expected_rms = 2.0_f32 / 65536.0 / 6.0_f32.sqrt();
        assert!(
            (rms - expected_rms).abs() < expected_rms * 0.5,
            "TPDF RMS {rms:.2e} should be near {expected_rms:.2e}"
        );
    }

    #[test]
    fn test_tpdf_dither_clamps_output() {
        let buf = AudioBuffer {
            samples: vec![1.0f32; 1024],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let dithered = apply_tpdf_dither(&buf, 16);
        for &s in &dithered.samples {
            assert!((-1.0..=1.0).contains(&s), "sample {s} out of range [-1, 1]");
        }
    }

    #[test]
    fn test_tpdf_dither_preserves_signal() {
        // A loud signal should still be close to original after dithering
        let samples: Vec<f32> = (0..4410)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 44_100.0).sin() * 0.9)
            .collect();
        let buf = AudioBuffer {
            samples: samples.clone(),
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        };
        let dithered = apply_tpdf_dither(&buf, 16);
        let max_diff = samples
            .iter()
            .zip(dithered.samples.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let lsb = 2.0_f32 / 65536.0;
        // Max diff should be within ±1 TPDF LSB (both r1+r2 at max = ±lsb)
        assert!(
            max_diff <= lsb * 1.5,
            "max diff {max_diff:.2e} too large (lsb={lsb:.2e})"
        );
    }
}
