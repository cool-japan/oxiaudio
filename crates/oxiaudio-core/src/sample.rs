//! Generic [`Sample`] trait abstracting over PCM sample types.
//!
//! The trait converts each native sample type to and from a normalized `f32`
//! in `[-1.0, 1.0]`, and exposes the silence value ([`Sample::EQUILIBRIUM`]) and
//! peak value ([`Sample::MAX_AMPLITUDE`]). It is implemented for `u8`, `i16`,
//! `i32`, `f32`, and `f64`.
//!
//! `AudioBuffer<T>` does **not** require `T: Sample` — the trait is provided as
//! a building block for generic DSP and conversion code that opts in explicitly.

/// A PCM sample type convertible to/from a normalized `f32`.
pub trait Sample: Copy + Send + Sync + 'static {
    /// The silence value for this sample type (0.0 for float, midpoint for `u8`).
    const EQUILIBRIUM: Self;
    /// The maximum positive amplitude representable by this type.
    const MAX_AMPLITUDE: Self;

    /// Convert this sample to a normalized `f32` in `[-1.0, 1.0]`.
    fn to_f32(self) -> f32;

    /// Construct this sample from a normalized `f32` (clamped to `[-1.0, 1.0]`
    /// for integer targets).
    fn from_f32(value: f32) -> Self;
}

impl Sample for f32 {
    const EQUILIBRIUM: Self = 0.0;
    const MAX_AMPLITUDE: Self = 1.0;

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value
    }
}

impl Sample for f64 {
    const EQUILIBRIUM: Self = 0.0;
    const MAX_AMPLITUDE: Self = 1.0;

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value as f64
    }
}

impl Sample for i16 {
    const EQUILIBRIUM: Self = 0;
    const MAX_AMPLITUDE: Self = i16::MAX;

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32 / i16::MAX as f32
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        (value.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
    }
}

impl Sample for i32 {
    const EQUILIBRIUM: Self = 0;
    const MAX_AMPLITUDE: Self = i32::MAX;

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32 / i32::MAX as f32
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        (value.clamp(-1.0, 1.0) * i32::MAX as f32) as i32
    }
}

impl Sample for u8 {
    /// 8-bit unsigned PCM is biased at 128 (the silence midpoint).
    const EQUILIBRIUM: Self = 128;
    const MAX_AMPLITUDE: Self = u8::MAX;

    #[inline]
    fn to_f32(self) -> f32 {
        (self as f32 - 128.0) / 128.0
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        let scaled = value.clamp(-1.0, 1.0) * 127.0 + 128.0;
        scaled.round().clamp(0.0, 255.0) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_roundtrip_identity() {
        for &v in &[-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            assert!((f32::from_f32(v).to_f32() - v).abs() < 1e-9);
        }
    }

    #[test]
    fn i16_roundtrip_within_lsb() {
        for &v in &[-1.0f32, -0.5, 0.0, 0.25, 1.0] {
            let s = i16::from_f32(v);
            assert!((s.to_f32() - v).abs() < 1.0 / i16::MAX as f32 + 1e-6);
        }
    }

    #[test]
    fn u8_silence_is_midpoint() {
        assert_eq!(u8::EQUILIBRIUM, 128);
        assert!((u8::EQUILIBRIUM.to_f32()).abs() < 1e-2);
        assert_eq!(u8::from_f32(0.0), 128);
        assert_eq!(u8::from_f32(1.0), 255);
        assert_eq!(u8::from_f32(-1.0), 1);
    }

    #[test]
    fn clamping_saturates() {
        assert_eq!(i16::from_f32(2.0), i16::MAX);
        assert_eq!(i16::from_f32(-2.0), -i16::MAX);
        assert_eq!(i32::from_f32(5.0), i32::MAX);
    }

    #[test]
    fn equilibrium_and_max_constants() {
        assert_eq!(f32::EQUILIBRIUM, 0.0);
        assert_eq!(f32::MAX_AMPLITUDE, 1.0);
        assert_eq!(i16::MAX_AMPLITUDE, i16::MAX);
        assert_eq!(i32::EQUILIBRIUM, 0);
    }
}
