//! Pipeline types and batch-processing utilities:
//! `TranscodePipeline`, `transcode_batch`, `convert`, `convert_with_dsp`,
//! `write_metadata`, `probe_metadata`.

use oxiaudio_core::{AudioBuffer, AudioMetadata, ChannelLayout, OxiAudioError};

use crate::decode::decode_file;
use crate::encode::{encode_flac, encode_wav};

/// Type alias for the optional DSP closure stored inside [`TranscodePipeline`].
type DspFn = Box<dyn Fn(&AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> + Send>;

/// Internal helper: encode `buf` to `path`, selecting the encoder from the file extension.
///
/// Supported output extensions: `wav`, `flac`, `aif`, `aiff`.
pub(crate) fn convert_buf_to_path(
    buf: &AudioBuffer<f32>,
    path: &std::path::Path,
) -> Result<(), OxiAudioError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "wav" => encode_wav(buf, path),
        "flac" => encode_flac(buf, path),
        "aif" | "aiff" => {
            use std::io::BufWriter;
            let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
            let mut w = BufWriter::new(file);
            oxiaudio_encode::write_aiff(buf, &mut w)
        }
        _ => Err(OxiAudioError::UnsupportedFormat(format!(
            "unsupported output format: .{ext}"
        ))),
    }
}

/// Auto-convert an audio file to another format, inferring formats from file extensions.
///
/// Supported output formats: `.wav`, `.flac`, `.aif`, `.aiff`.
#[must_use = "discarding the Result ignores convert errors"]
pub fn convert(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<(), OxiAudioError> {
    let buf = decode_file(input_path)?;
    convert_buf_to_path(&buf, output_path)
}

/// Decode an audio file, apply a DSP transformation function, and encode to a new file.
///
/// The output format is inferred from the file extension of `output` (`.wav` or `.flac`).
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// oxiaudio::convert_with_dsp(
///     Path::new("input.flac"),
///     Path::new("output_normalized.wav"),
///     |mut buf| { oxiaudio::dsp::normalize(&mut buf, -1.0); buf },
/// ).unwrap();
/// ```
#[must_use = "discarding the Result ignores convert errors"]
pub fn convert_with_dsp<F>(
    input: &std::path::Path,
    output: &std::path::Path,
    dsp_fn: F,
) -> Result<(), OxiAudioError>
where
    F: FnOnce(AudioBuffer<f32>) -> AudioBuffer<f32>,
{
    let buf = decode_file(input)?;
    let processed = dsp_fn(buf);
    match output.extension().and_then(|e| e.to_str()) {
        Some("wav") => encode_wav(&processed, output),
        Some("flac") => encode_flac(&processed, output),
        ext => Err(OxiAudioError::UnsupportedFormat(format!(
            "unsupported output format: {:?}",
            ext
        ))),
    }
}

/// Write metadata tags to an existing audio file.
///
/// Currently supports WAV files only. The file is decoded and re-encoded with the
/// supplied metadata embedded as a `LIST/INFO` chunk. FLAC and other formats return
/// [`OxiAudioError::UnsupportedFormat`].
#[must_use = "discarding the Result ignores write errors"]
pub fn write_metadata(
    path: &std::path::Path,
    metadata: &AudioMetadata,
) -> Result<(), OxiAudioError> {
    let buf = decode_file(path)?;
    match path.extension().and_then(|e| e.to_str()) {
        Some("wav") => {
            let file = std::fs::File::create(path).map_err(OxiAudioError::Io)?;
            let writer = std::io::BufWriter::new(file);
            oxiaudio_encode::WavEncoder::default().encode_with_metadata(&buf, writer, metadata)
        }
        ext => Err(OxiAudioError::UnsupportedFormat(format!(
            "write_metadata not supported for format: {:?}",
            ext
        ))),
    }
}

/// A composable transcode pipeline: decode from `input_path`, optionally apply a DSP
/// function, then encode to `output_path` (format inferred from extension).
///
/// Use [`TranscodePipeline::new`] to create, optionally chain [`TranscodePipeline::with_dsp`],
/// then call [`TranscodePipeline::run`] to execute.
///
/// Supported output formats: `.wav`, `.flac`, `.aif`, `.aiff`.
#[must_use = "call .run() to execute the transcode"]
pub struct TranscodePipeline {
    input_path: std::path::PathBuf,
    output_path: std::path::PathBuf,
    dsp_fn: Option<DspFn>,
}

impl TranscodePipeline {
    /// Create a new transcode pipeline that reads from `input` and writes to `output`.
    pub fn new(input: &std::path::Path, output: &std::path::Path) -> Self {
        Self {
            input_path: input.to_path_buf(),
            output_path: output.to_path_buf(),
            dsp_fn: None,
        }
    }

    /// Attach a DSP processing function applied between decode and encode.
    ///
    /// The closure receives a reference to the decoded buffer and must return a
    /// (potentially new) buffer. Errors are propagated from [`TranscodePipeline::run`].
    pub fn with_dsp<F>(mut self, f: F) -> Self
    where
        F: Fn(&AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> + Send + 'static,
    {
        self.dsp_fn = Some(Box::new(f));
        self
    }

    /// Execute the transcode pipeline.
    ///
    /// 1. Decode `input_path`.
    /// 2. Apply the optional DSP function (if set via [`with_dsp`]).
    /// 3. Encode to `output_path`, selecting the codec from the file extension.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError`] on decode failure, DSP failure, or encode failure.
    ///
    /// [`with_dsp`]: TranscodePipeline::with_dsp
    pub fn run(self) -> Result<(), OxiAudioError> {
        let buf = decode_file(&self.input_path)?;
        let processed = match self.dsp_fn {
            Some(ref f) => f(&buf)?,
            None => buf,
        };
        convert_buf_to_path(&processed, &self.output_path)
    }
}

/// A composable streaming transcode pipeline: decode in chunks → apply DSP → encode.
///
/// # Example
/// ```no_run
/// use oxiaudio::TranscodeStream;
/// let ts = TranscodeStream::new(
///     std::path::Path::new("input.wav"),
///     std::path::Path::new("output.flac"),
/// ).expect("open");
/// ts.run().expect("transcode");
/// ```
#[must_use = "call .run() to execute the transcode"]
pub struct TranscodeStream {
    input_path: std::path::PathBuf,
    output_path: std::path::PathBuf,
    filters: Vec<Box<dyn oxiaudio_core::AudioFilter>>,
    chunk_frames: usize,
}

impl TranscodeStream {
    /// Create a `TranscodeStream` from input to output.
    ///
    /// Output format is inferred from the file extension (.wav / .flac / .aiff).
    pub fn new(
        input: impl AsRef<std::path::Path>,
        output: impl AsRef<std::path::Path>,
    ) -> Result<Self, OxiAudioError> {
        Ok(Self {
            input_path: input.as_ref().to_path_buf(),
            output_path: output.as_ref().to_path_buf(),
            filters: Vec::new(),
            chunk_frames: 4096,
        })
    }

    /// Add a DSP filter applied to every chunk during transcoding.
    pub fn with_filter(mut self, filter: Box<dyn oxiaudio_core::AudioFilter>) -> Self {
        self.filters.push(filter);
        self
    }

    /// Set the decode chunk size in frames (default: 4096).
    pub fn with_chunk_frames(mut self, frames: usize) -> Self {
        self.chunk_frames = frames;
        self
    }

    /// Run the transcode pipeline: decode → apply filters → encode.
    ///
    /// Reads the input in streaming chunks of `chunk_frames` frames, applies all
    /// registered filters to the concatenated buffer, and writes to the output.
    /// The format of the output is determined by the file extension.
    pub fn run(self) -> Result<(), OxiAudioError> {
        use oxiaudio_core::SampleFormat;

        // Stream-decode the input into chunks of self.chunk_frames.
        let file = std::fs::File::open(&self.input_path).map_err(OxiAudioError::Io)?;
        let reader = std::io::BufReader::new(file);
        let stream = crate::decode::decode_stream_with_block_size(reader, self.chunk_frames);

        // Collect and concatenate all chunks.
        let mut all_samples: Vec<f32> = Vec::new();
        let mut sample_rate = 0u32;
        let mut channels = ChannelLayout::Mono;
        for chunk_result in stream {
            let chunk = chunk_result?;
            if all_samples.is_empty() {
                sample_rate = chunk.sample_rate;
                channels = chunk.channels;
            }
            all_samples.extend_from_slice(&chunk.samples);
        }

        let mut buf = AudioBuffer {
            samples: all_samples,
            sample_rate,
            channels,
            format: SampleFormat::F32,
        };

        // Apply filters in order.
        for filter in &self.filters {
            buf = filter.apply(&buf)?;
        }

        // Encode based on output extension.
        let ext = self
            .output_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let file = std::fs::File::create(&self.output_path).map_err(OxiAudioError::Io)?;
        let writer = std::io::BufWriter::new(file);

        match ext.as_str() {
            "wav" => {
                use oxiaudio_core::AudioEncoder;
                oxiaudio_encode::WavEncoder::default()
                    .encode(&buf, writer)
                    .map_err(|e| OxiAudioError::Encode(e.to_string()))
            }
            "flac" => {
                let mut enc = oxiaudio_encode::FlacStreamEncoder::new(
                    writer,
                    buf.sample_rate,
                    buf.channels,
                    5,
                );
                enc.encode_chunk(&buf)?;
                enc.finalize()
            }
            "aiff" | "aif" => {
                let mut enc = oxiaudio_encode::AiffStreamEncoder::new(
                    writer,
                    buf.sample_rate,
                    buf.channels,
                    oxiaudio_encode::AiffBitDepth::I16,
                )?;
                enc.encode_chunk(&buf)?;
                enc.finalize()
            }
            other => Err(OxiAudioError::UnsupportedFormat(format!(
                "TranscodeStream: unsupported output format '.{other}'"
            ))),
        }
    }
}

/// Extract metadata from an audio file.
///
/// Internally decodes the file to extract embedded metadata tags; the audio data is discarded.
/// For format probing without audio decode, use `detect_format` instead.
#[must_use = "discarding the Result ignores probe errors"]
pub fn probe_metadata(path: &std::path::Path) -> Result<AudioMetadata, OxiAudioError> {
    let (_buf, metadata) = crate::decode::decode_file_with_metadata(path)?;
    Ok(metadata)
}

/// Batch-transcode multiple audio files in parallel using rayon.
///
/// Each input file is decoded, then encoded to `output_dir` with the given `output_ext`.
/// Output file names are derived from the input file stems (the extension is replaced).
///
/// Returns one [`Result`] per input path, in the same order.  On success the value is the
/// path of the newly created output file.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// let inputs: Vec<&Path> = vec![Path::new("a.flac"), Path::new("b.flac")];
/// # let out_dir = std::env::temp_dir().join("out");
/// let results = oxiaudio::transcode_batch(&inputs, &out_dir, "wav");
/// for r in results { r.expect("transcode failed"); }
/// ```
pub fn transcode_batch(
    input_paths: &[&std::path::Path],
    output_dir: &std::path::Path,
    output_ext: &str,
) -> Vec<Result<std::path::PathBuf, OxiAudioError>> {
    use rayon::prelude::*;
    input_paths
        .par_iter()
        .map(|&input| {
            let stem = input.file_stem().ok_or_else(|| {
                OxiAudioError::UnsupportedFormat(format!(
                    "input path has no file stem: {}",
                    input.display()
                ))
            })?;
            let output = output_dir.join(stem).with_extension(output_ext);
            let buf = decode_file(input)?;
            convert_buf_to_path(&buf, &output)?;
            Ok(output)
        })
        .collect()
}
