/// A point in time expressed either as a frame count or wall-clock seconds.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Timestamp {
    /// Absolute position in frames.
    Frames(u64),
    /// Absolute position in seconds.
    Seconds(f64),
}

impl Timestamp {
    /// Convert to an absolute frame position at the given sample rate.
    pub fn to_frames(&self, sample_rate: u32) -> u64 {
        match self {
            Timestamp::Frames(f) => *f,
            Timestamp::Seconds(s) => (*s * sample_rate as f64).round() as u64,
        }
    }

    /// Convert to a wall-clock position (seconds) at the given sample rate.
    pub fn to_seconds(&self, sample_rate: u32) -> f64 {
        match self {
            Timestamp::Frames(f) => *f as f64 / sample_rate as f64,
            Timestamp::Seconds(s) => *s,
        }
    }
}

/// Monotonic audio clock that tracks elapsed frames and reports drift vs wallclock.
#[derive(Debug, Clone)]
pub struct AudioClock {
    sample_rate: u32,
    elapsed_frames: u64,
    start_time: std::time::Instant,
}

impl AudioClock {
    /// Create a new clock for the given sample rate. The wallclock reference is set now.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            elapsed_frames: 0,
            start_time: std::time::Instant::now(),
        }
    }

    /// Advance the audio clock by `frames` samples.
    pub fn advance(&mut self, frames: u64) {
        self.elapsed_frames += frames;
    }

    /// Total frames advanced since creation.
    pub fn elapsed_frames(&self) -> u64 {
        self.elapsed_frames
    }

    /// Elapsed audio time in seconds (frames / sample_rate).
    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed_frames as f64 / self.sample_rate as f64
    }

    /// Clock drift in PPM (parts per million). Positive means the audio clock is ahead.
    pub fn drift_ppm(&self) -> f64 {
        let wall_secs = self.start_time.elapsed().as_secs_f64();
        if wall_secs < 1e-9 {
            return 0.0;
        }
        let audio_secs = self.elapsed_secs();
        (audio_secs - wall_secs) / wall_secs * 1_000_000.0
    }

    /// Compute clock drift in parts-per-million given an explicit wall-clock duration
    /// in nanoseconds.
    ///
    /// This is the testable counterpart to [`drift_ppm`]: callers supply the measured
    /// wall time directly rather than relying on `std::time::Instant`. Positive values
    /// mean the audio clock is running *ahead* of wall time.
    ///
    /// Returns `0.0` if `elapsed_frames` or `elapsed_wall_ns` is zero.
    ///
    /// [`drift_ppm`]: Self::drift_ppm
    pub fn drift_ppm_from_ns(&self, elapsed_wall_ns: u64) -> f64 {
        if self.elapsed_frames == 0 || elapsed_wall_ns == 0 {
            return 0.0;
        }
        let nominal_ns = self.elapsed_frames as f64 * 1_000_000_000.0 / self.sample_rate as f64;
        let actual_ns = elapsed_wall_ns as f64;
        (nominal_ns - actual_ns) / actual_ns * 1_000_000.0
    }
}
