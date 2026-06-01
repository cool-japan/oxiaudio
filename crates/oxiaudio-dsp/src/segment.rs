//! Audio segmentation: silence-based splitting.
use oxiaudio_core::AudioBuffer;

/// Split an audio buffer into non-silent segments.
///
/// Scans the buffer frame-by-frame and collects runs of silence (RMS of all
/// channels in a frame below `threshold_db` dBFS). When a silence run of at
/// least `min_silence_frames` consecutive frames is detected, the buffer is
/// split at the midpoint of that run. Segments shorter than
/// `min_silence_frames / 2` frames are dropped.
///
/// Returns an empty `Vec` if the entire buffer is silence.
pub fn silence_split(
    buf: &AudioBuffer<f32>,
    threshold_db: f32,
    min_silence_frames: usize,
) -> Vec<AudioBuffer<f32>> {
    let n_channels = buf.channels.channel_count();
    if n_channels == 0 || buf.samples.is_empty() {
        return vec![];
    }

    let n_frames = buf.samples.len() / n_channels;
    if n_frames == 0 {
        return vec![];
    }

    // Convert threshold from dBFS to linear amplitude.
    // For RMS/peak: linear_thresh = 10^(threshold_db / 20).
    let linear_thresh = 10.0f32.powf(threshold_db / 20.0);

    // For each frame, determine if it is silent by checking peak amplitude
    // across all channels in that frame.
    let frame_is_silent: Vec<bool> = (0..n_frames)
        .map(|f| {
            let base = f * n_channels;
            let peak = buf.samples[base..base + n_channels]
                .iter()
                .map(|&s| s.abs())
                .fold(0.0f32, f32::max);
            peak < linear_thresh
        })
        .collect();

    // Walk through frames and collect split points at midpoints of qualifying
    // silence runs. A split point is recorded in terms of frame index (inclusive
    // start of the segment after the split).
    //
    // Strategy:
    //   - Track the start of the current silence run.
    //   - When a silence run ends (we hit a non-silent frame), if the run's
    //     length >= min_silence_frames, record a split at the midpoint of that run.
    //   - After the loop, handle any trailing silence run.

    let min_silence = min_silence_frames.max(1);

    // Split points are the frame indices where new segments begin.
    // We always have an implicit segment starting at frame 0.
    let mut split_starts: Vec<usize> = vec![0];

    let mut silence_run_start: Option<usize> = None;

    for (f, &is_silent) in frame_is_silent.iter().enumerate() {
        if is_silent {
            if silence_run_start.is_none() {
                silence_run_start = Some(f);
            }
        } else {
            // Non-silent frame — check if the silence run just ended.
            if let Some(run_start) = silence_run_start.take() {
                let run_len = f - run_start;
                if run_len >= min_silence {
                    // Split at the midpoint of the silence run.
                    let mid = run_start + run_len / 2;
                    split_starts.push(mid);
                }
            }
        }
    }

    // Trailing silence run: we don't split on it (there's nothing after),
    // but it will be dropped as an all-silence segment by the filter below.

    // Add a sentinel past the end.
    split_starts.push(n_frames);

    // Build segments from consecutive split_starts pairs and filter out:
    //   1. Empty segments.
    //   2. All-silence segments (handles the all-silence-input case and
    //      leading/trailing silence fragments).
    //   3. Segments shorter than min_silence / 2 frames.
    let min_segment_frames = (min_silence / 2).max(1);

    let mut result = Vec::new();

    for window in split_starts.windows(2) {
        let seg_start = window[0];
        let seg_end = window[1];

        if seg_end <= seg_start {
            continue;
        }

        let seg_frames = seg_end - seg_start;

        // Drop segments that are too short.
        if seg_frames < min_segment_frames {
            continue;
        }

        // Drop segments that are entirely silence.
        let has_signal = (seg_start..seg_end).any(|f| !frame_is_silent[f]);
        if !has_signal {
            continue;
        }

        let sample_start = seg_start * n_channels;
        let sample_end = seg_end * n_channels;
        let samples = buf.samples[sample_start..sample_end].to_vec();

        result.push(AudioBuffer {
            samples,
            sample_rate: buf.sample_rate,
            channels: buf.channels,
            format: buf.format,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiaudio_core::{ChannelLayout, SampleFormat};

    fn mono_buf(samples: Vec<f32>) -> AudioBuffer<f32> {
        AudioBuffer {
            samples,
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_silence_split_all_silence() {
        // All-zero buffer should return an empty Vec.
        let buf = mono_buf(vec![0.0f32; 1000]);
        let segments = silence_split(&buf, -40.0, 10);
        assert!(
            segments.is_empty(),
            "all-silence buffer should return empty Vec, got {} segments",
            segments.len()
        );
    }

    #[test]
    fn test_silence_split_no_silence() {
        // Loud signal throughout: should return one segment equal to input.
        let samples: Vec<f32> = (0..100).map(|_| 0.5f32).collect();
        let buf = mono_buf(samples.clone());
        let segments = silence_split(&buf, -40.0, 10);
        assert_eq!(
            segments.len(),
            1,
            "no-silence buffer should return 1 segment, got {}",
            segments.len()
        );
        assert_eq!(
            segments[0].samples, samples,
            "single segment should equal input"
        );
    }

    #[test]
    fn test_silence_split_two_segments() {
        // Layout: [50 signal frames] [30 silence frames] [50 signal frames]
        // With min_silence_frames=20, the 30-frame silence should trigger a split.
        // Split at midpoint of silence (frame 65), so:
        //   segment 0: frames 0..65  (50 signal + 15 silence = 65 frames)
        //   segment 1: frames 65..130 (15 silence + 50 signal = 65 frames)
        // Both segments contain signal, so both survive the filter.
        let signal_val = 0.5f32;
        let signal: Vec<f32> = vec![signal_val; 50];
        let silence: Vec<f32> = vec![0.0f32; 30];
        let mut samples = signal.clone();
        samples.extend_from_slice(&silence);
        samples.extend_from_slice(&signal);

        let buf = mono_buf(samples);
        let segments = silence_split(&buf, -40.0, 20);

        assert_eq!(
            segments.len(),
            2,
            "expected 2 segments, got {}",
            segments.len()
        );

        // Each segment should contain at least the 50 signal frames
        // (could also include some of the silence boundary).
        let total_signal_frames: usize = segments.iter().map(|s| s.samples.len()).sum();
        // Total frames in input = 130; frames in result should be 130 (no frames dropped).
        assert_eq!(
            total_signal_frames, 130,
            "total frames in segments should equal input frames"
        );
    }

    #[test]
    fn test_silence_split_min_silence_frames() {
        // A silence run shorter than min_silence_frames should NOT be a split point,
        // so the buffer should come back as a single segment.
        let signal: Vec<f32> = vec![0.5f32; 50];
        let short_silence: Vec<f32> = vec![0.0f32; 5]; // only 5 frames of silence
        let mut samples = signal.clone();
        samples.extend_from_slice(&short_silence);
        samples.extend_from_slice(&signal);

        let buf = mono_buf(samples);
        // Require at least 20 frames of silence to split.
        let segments = silence_split(&buf, -40.0, 20);

        assert_eq!(
            segments.len(),
            1,
            "silence shorter than min_silence_frames should not split; got {} segments",
            segments.len()
        );
    }
}
