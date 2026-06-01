use crate::error::OxiAudioError;
use crate::layout::ChannelLayout;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 8-bit unsigned PCM (bias at 128).
    U8,
    F32,
    I16,
    /// 24-bit packed signed integer PCM.
    I24,
    I32,
    F64,
}

impl SampleFormat {
    /// Normalize a raw integer sample value (i32) to f32 in [-1.0, 1.0].
    ///
    /// For `F32` / `F64` the value is returned as-is (cast to f32).
    pub fn normalize_i32_to_f32(&self, raw: i32) -> f32 {
        match self {
            SampleFormat::U8 => (raw as f32 - 128.0) / 128.0,
            SampleFormat::I16 => raw as f32 / i16::MAX as f32,
            SampleFormat::I24 => raw as f32 / 8_388_607.0,
            SampleFormat::I32 => raw as f32 / i32::MAX as f32,
            SampleFormat::F32 | SampleFormat::F64 => raw as f32,
        }
    }

    /// Nominal bit depth of the format: 8/16/24/32/64.
    pub fn bit_depth(&self) -> u16 {
        match self {
            SampleFormat::U8 => 8,
            SampleFormat::I16 => 16,
            SampleFormat::I24 => 24,
            SampleFormat::I32 | SampleFormat::F32 => 32,
            SampleFormat::F64 => 64,
        }
    }

    /// `true` for floating-point formats (`F32`, `F64`).
    pub fn is_float(&self) -> bool {
        matches!(self, SampleFormat::F32 | SampleFormat::F64)
    }

    /// `true` for integer formats (`U8`, `I16`, `I24`, `I32`).
    pub fn is_integer(&self) -> bool {
        !self.is_float()
    }

    /// Bytes used to store one sample of this format in memory.
    ///
    /// `I24` is reported as 3 bytes (packed on-disk size); in-memory it is
    /// typically carried in an `i32`.
    pub fn byte_size(&self) -> usize {
        match self {
            SampleFormat::U8 => 1,
            SampleFormat::I16 => 2,
            SampleFormat::I24 => 3,
            SampleFormat::I32 | SampleFormat::F32 => 4,
            SampleFormat::F64 => 8,
        }
    }
}

impl std::fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleFormat::U8 => write!(f, "u8"),
            SampleFormat::F32 => write!(f, "f32"),
            SampleFormat::I16 => write!(f, "i16"),
            SampleFormat::I24 => write!(f, "i24"),
            SampleFormat::I32 => write!(f, "i32"),
            SampleFormat::F64 => write!(f, "f64"),
        }
    }
}

impl std::str::FromStr for SampleFormat {
    type Err = OxiAudioError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "u8" | "uint8" => Ok(SampleFormat::U8),
            "i16" | "s16" | "int16" => Ok(SampleFormat::I16),
            "i24" | "s24" | "int24" => Ok(SampleFormat::I24),
            "i32" | "s32" | "int32" => Ok(SampleFormat::I32),
            "f32" | "float" | "float32" => Ok(SampleFormat::F32),
            "f64" | "double" | "float64" => Ok(SampleFormat::F64),
            other => Err(OxiAudioError::UnsupportedFormat(format!(
                "unknown sample format string: {other:?}"
            ))),
        }
    }
}

impl TryFrom<&str> for SampleFormat {
    type Error = OxiAudioError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// Lightweight format descriptor returned by the probe API (no audio decoded).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: ChannelLayout,
    pub format: SampleFormat,
}

/// Describes the memory layout of multi-channel audio in an `AudioBuffer`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioBufferLayout {
    /// Channels are interleaved: L R L R …
    Interleaved,
    /// Each channel occupies a contiguous block: L L … R R …
    Planar,
}
