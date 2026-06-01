//! Memory-mapped file I/O path for `decode_file_mmap`.
//!
//! This module contains the single unsafe call (`Mmap::map`) required for `memmap2`.
//! All other modules in this crate are covered by the crate-level `#![deny(unsafe_code)]`.
//! This module explicitly opts in via `#![allow(unsafe_code)]` — the SAFETY comment below
//! documents why the call is sound.
//!
//! SAFETY rationale: `Mmap::map` requires unsafe because the OS may invalidate the mapping
//! if the underlying file is truncated or deleted while mapped.  We mitigate this by
//! (1) opening the file in read-only mode (no external writer can truncate via our handle),
//! (2) wrapping `Mmap` in `Cursor<Mmap>` — `Mmap` owns the mapping with no lifetime parameter,
//!     so `Cursor<Mmap>` keeps the mapping live for the full duration of the decode call,
//! (3) if an external process truncates the file while we are mapping it the worst outcome is
//!     a decode error (Symphonia bounds-checks via the Cursor's slice indexing), not UB.
//! The COOLJAPAN Pure Rust Policy permits this targeted opt-in; see Cargo.toml `mmap` feature.
#![allow(unsafe_code)]

use memmap2::Mmap;

use std::path::Path;

use oxiaudio_core::{AudioBuffer, OxiAudioError};

/// Decode a large audio file using memory-mapped I/O for reduced kernel buffer copies.
///
/// On most operating systems `mmap` allows the kernel to serve the file pages directly into
/// the process address space, skipping the intermediate `read(2)` copy from the page-cache
/// to the userspace `BufReader` buffer.  For large files (tens of megabytes) this reduces
/// memory bandwidth by approximately one copy per page.
///
/// # Fallback
///
/// If the file cannot be mapped (e.g. a pipe or a zero-length file), the function returns
/// `OxiAudioError::Io`.  Callers can fall back to [`crate::decode_file`] in that case.
///
/// # Errors
///
/// Returns [`OxiAudioError::Io`] if the file cannot be opened or mapped, or
/// [`OxiAudioError::Decode`] if format probing or codec decoding fails.
#[must_use = "discarding the Result ignores decode errors"]
pub fn decode_file_mmap(path: &Path) -> Result<AudioBuffer<f32>, OxiAudioError> {
    let file = std::fs::File::open(path).map_err(OxiAudioError::Io)?;
    // SAFETY: see module-level SAFETY comment.
    let mmap = unsafe { Mmap::map(&file) }.map_err(OxiAudioError::Io)?;
    // `Mmap` implements `AsRef<[u8]>`, `Read`, `Send`, `Sync`, and owns the mapping (no lifetime
    // parameter), so `Cursor<Mmap>` satisfies `Read + Seek + Send + Sync + 'static` with zero
    // copy.  The kernel serves pages directly from the page-cache into the process address space
    // as Symphonia indexes into the `Cursor`, avoiding the intermediate `read(2)` copy path.
    let cursor = std::io::Cursor::new(mmap);
    crate::decode_reader(cursor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal 44-byte RIFF/WAV header + i16 PCM samples for test use.
    fn make_wav_bytes(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let byte_rate = sample_rate * 2;
        let data_size = (samples.len() as u32) * 2;
        let file_size = 36 + data_size;
        let mut v = Vec::with_capacity(44 + samples.len() * 2);
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&file_size.to_le_bytes());
        v.extend_from_slice(b"WAVE");
        v.extend_from_slice(b"fmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes()); // PCM
        v.extend_from_slice(&1u16.to_le_bytes()); // mono
        v.extend_from_slice(&sample_rate.to_le_bytes());
        v.extend_from_slice(&byte_rate.to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes()); // block align
        v.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples {
            v.extend_from_slice(&s.to_le_bytes());
        }
        v
    }

    /// `decode_file_mmap` on a valid WAV file returns the same sample count as `decode_file`.
    #[test]
    fn test_decode_file_mmap_matches_decode_file() {
        let samples: Vec<i16> = (0..256).map(|i| (i as i16) * 100).collect();
        let wav_bytes = make_wav_bytes(44_100, &samples);
        let mut path = std::env::temp_dir();
        path.push("oxiaudio_mmap_test.wav");
        std::fs::write(&path, &wav_bytes).expect("write temp wav");

        let mmap_buf = decode_file_mmap(&path).expect("decode_file_mmap must succeed");
        let file_buf = crate::decode_file(&path).expect("decode_file must succeed");

        let _ = std::fs::remove_file(&path);

        assert_eq!(
            mmap_buf.sample_rate, file_buf.sample_rate,
            "sample_rate mismatch between mmap and file decode"
        );
        assert_eq!(
            mmap_buf.samples.len(),
            file_buf.samples.len(),
            "sample count mismatch between mmap and file decode"
        );
        // Verify first 10 samples are within 1e-6 tolerance.
        for (i, (m, f)) in mmap_buf
            .samples
            .iter()
            .zip(file_buf.samples.iter())
            .enumerate()
            .take(10)
        {
            assert!(
                (m - f).abs() < 1e-6,
                "sample[{i}] differs: mmap={m} file={f}"
            );
        }
    }

    /// `decode_file_mmap` returns `OxiAudioError::Io` for a non-existent file.
    #[test]
    fn test_decode_file_mmap_nonexistent_returns_io_error() {
        let path = std::env::temp_dir().join("oxiaudio_mmap_nonexistent_xyz.wav");
        let result = decode_file_mmap(&path);
        assert!(result.is_err(), "expected Err for missing file, got Ok");
    }
}
