use crate::buffer::AudioBuffer;
use crate::error::OxiAudioError;
use crate::format::SampleFormat;

#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    Mono,
    Stereo,
    Quad,
    Surround51,
    /// 5.1 with side surround channels (L, R, C, LFE, SL, SR).
    Surround51Side,
    Surround71,
    /// Dolby Atmos 7.1.4 (L, R, C, LFE, BL, BR, SL, SR + 4 height channels).
    Atmos714,
}

impl ChannelLayout {
    pub fn channel_count(&self) -> usize {
        match self {
            ChannelLayout::Mono => 1,
            ChannelLayout::Stereo => 2,
            ChannelLayout::Quad => 4,
            ChannelLayout::Surround51 => 6,
            ChannelLayout::Surround51Side => 6,
            ChannelLayout::Surround71 => 8,
            ChannelLayout::Atmos714 => 12,
        }
    }
}

impl std::fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelLayout::Mono => write!(f, "mono"),
            ChannelLayout::Stereo => write!(f, "stereo"),
            ChannelLayout::Quad => write!(f, "quad"),
            ChannelLayout::Surround51 => write!(f, "5.1"),
            ChannelLayout::Surround51Side => write!(f, "5.1side"),
            ChannelLayout::Surround71 => write!(f, "7.1"),
            ChannelLayout::Atmos714 => write!(f, "7.1.4"),
        }
    }
}

impl From<u16> for ChannelLayout {
    /// Map a raw channel count to the best-fit layout.
    ///
    /// `1` → `Mono`, `4` → `Quad`, `6` → `Surround51`, `8` → `Surround71`,
    /// `12` → `Atmos714`; everything else → `Stereo`.
    fn from(count: u16) -> Self {
        match count {
            1 => ChannelLayout::Mono,
            4 => ChannelLayout::Quad,
            6 => ChannelLayout::Surround51,
            8 => ChannelLayout::Surround71,
            12 => ChannelLayout::Atmos714,
            _ => ChannelLayout::Stereo,
        }
    }
}

/// Semantic label for a specific audio channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChannelId {
    FrontLeft,
    FrontRight,
    FrontCenter,
    LowFrequency,
    RearLeft,
    RearRight,
    SideLeft,
    SideRight,
    TopFrontLeft,
    TopFrontRight,
    TopRearLeft,
    TopRearRight,
}

/// Ordered mapping from channel index to [`ChannelId`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChannelMap {
    channels: Vec<ChannelId>,
}

impl ChannelMap {
    /// Create a new `ChannelMap` from an ordered list of channel IDs.
    pub fn new(channels: Vec<ChannelId>) -> Self {
        Self { channels }
    }

    /// Return the standard channel map for the given `layout`.
    pub fn for_layout(layout: ChannelLayout) -> Self {
        use ChannelId::*;
        match layout {
            ChannelLayout::Mono => Self::new(vec![FrontLeft]),
            ChannelLayout::Stereo => Self::new(vec![FrontLeft, FrontRight]),
            ChannelLayout::Quad => Self::new(vec![FrontLeft, FrontRight, RearLeft, RearRight]),
            ChannelLayout::Surround51 => Self::new(vec![
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                RearLeft,
                RearRight,
            ]),
            ChannelLayout::Surround51Side => Self::new(vec![
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                SideLeft,
                SideRight,
            ]),
            ChannelLayout::Surround71 => Self::new(vec![
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                RearLeft,
                RearRight,
                SideLeft,
                SideRight,
            ]),
            ChannelLayout::Atmos714 => Self::new(vec![
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                RearLeft,
                RearRight,
                SideLeft,
                SideRight,
                TopFrontLeft,
                TopFrontRight,
                TopRearLeft,
                TopRearRight,
            ]),
        }
    }

    /// Number of mapped channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Return the `ChannelId` at index `idx`, or `None` if out of range.
    pub fn get(&self, idx: usize) -> Option<ChannelId> {
        self.channels.get(idx).copied()
    }

    /// Iterate over the channel IDs in order.
    pub fn iter(&self) -> std::slice::Iter<'_, ChannelId> {
        self.channels.iter()
    }

    /// Return the first index of `id` in the map, or `None` if not present.
    pub fn index_of(&self, id: ChannelId) -> Option<usize> {
        self.channels.iter().position(|&c| c == id)
    }

    /// Remap audio channels from `src_map` ordering to `dst_map` ordering.
    ///
    /// Channels present in `dst_map` but absent in `src_map` are filled with silence.
    /// Channels present in `src_map` but absent in `dst_map` are discarded.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `buf.channels.channel_count()` does not match
    /// `src_map.channel_count()`.
    #[must_use = "discarding the result ignores remap errors"]
    pub fn remap(
        buf: &AudioBuffer<f32>,
        src_map: &ChannelMap,
        dst_map: &ChannelMap,
    ) -> Result<AudioBuffer<f32>, OxiAudioError> {
        let src_channels = src_map.channel_count();
        if buf.channels.channel_count() != src_channels {
            return Err(OxiAudioError::InvalidChannelLayout(format!(
                "remap: buffer has {} channels but src_map has {}",
                buf.channels.channel_count(),
                src_channels,
            )));
        }
        let dst_channels = dst_map.channel_count();

        // Build a lookup: for each dst channel index, find the matching src channel index.
        let mapping: Vec<Option<usize>> = dst_map
            .channels
            .iter()
            .map(|&dst_id| src_map.index_of(dst_id))
            .collect();

        let frame_count = buf.samples.len().checked_div(src_channels).unwrap_or(0);

        let mut samples = vec![0.0f32; frame_count * dst_channels];
        for f in 0..frame_count {
            for (dst_c, &src_idx) in mapping.iter().enumerate() {
                if let Some(src_c) = src_idx {
                    samples[f * dst_channels + dst_c] = buf.samples[f * src_channels + src_c];
                }
                // else: remains 0.0 (silence)
            }
        }

        let out_layout = ChannelLayout::from(dst_channels as u16);
        Ok(AudioBuffer {
            samples,
            sample_rate: buf.sample_rate,
            channels: out_layout,
            format: SampleFormat::F32,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_remap_stereo_swap() {
        // Create a stereo buffer: L=1.0, R=-1.0 across 4 frames.
        let samples: Vec<f32> = (0..4).flat_map(|_| [1.0f32, -1.0f32]).collect();
        let buf = AudioBuffer {
            samples,
            sample_rate: 44100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };

        let src_map = ChannelMap::new(vec![ChannelId::FrontLeft, ChannelId::FrontRight]);
        let dst_map = ChannelMap::new(vec![ChannelId::FrontRight, ChannelId::FrontLeft]);

        let remapped = ChannelMap::remap(&buf, &src_map, &dst_map).expect("remap should succeed");

        assert_eq!(remapped.channels.channel_count(), 2);
        // After swap: ch0 = FrontRight (was -1.0), ch1 = FrontLeft (was 1.0)
        for f in 0..4 {
            assert_eq!(
                remapped.samples[f * 2],
                -1.0,
                "frame {f}: expected ch0 (FrontRight) = -1.0"
            );
            assert_eq!(
                remapped.samples[f * 2 + 1],
                1.0,
                "frame {f}: expected ch1 (FrontLeft) = 1.0"
            );
        }
    }

    #[test]
    fn test_remap_channel_count_mismatch_returns_err() {
        let buf = AudioBuffer {
            samples: vec![0.0f32; 4],
            sample_rate: 44100,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let src_map = ChannelMap::new(vec![ChannelId::FrontLeft]); // only 1 channel
        let dst_map = ChannelMap::new(vec![ChannelId::FrontLeft]);
        assert!(ChannelMap::remap(&buf, &src_map, &dst_map).is_err());
    }
}
