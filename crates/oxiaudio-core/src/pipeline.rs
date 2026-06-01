use crate::buffer::AudioBuffer;
use crate::error::OxiAudioError;

/// A single named processing stage in an [`AudioPipeline`].
pub trait AudioNode: Send + Sync {
    /// Human-readable name for debugging and introspection.
    fn name(&self) -> &str;

    /// When `true`, [`AudioPipeline::process`] skips this node and passes the
    /// buffer through unchanged.
    fn bypass(&self) -> bool {
        false
    }

    /// Apply the node's processing to `input`, returning a new buffer.
    fn process(&self, input: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError>;

    /// Declared algorithmic latency introduced by this node, in frames.
    ///
    /// Returns `None` if the latency is unknown or not applicable.
    /// The default implementation returns `Some(0)` (no latency).
    fn latency_frames(&self) -> Option<usize> {
        Some(0)
    }
}

/// A linear chain of [`AudioNode`]s. Buffers flow from the first to the last node.
pub struct AudioPipeline {
    nodes: Vec<Box<dyn AudioNode>>,
}

impl AudioPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { nodes: vec![] }
    }

    /// Append a node and return the pipeline (builder pattern).
    pub fn push_node(mut self, node: Box<dyn AudioNode>) -> Self {
        self.nodes.push(node);
        self
    }

    /// Run `input` through every non-bypassed node in order.
    pub fn process(&self, input: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        let mut buf = input.clone();
        for node in &self.nodes {
            if !node.bypass() {
                buf = node.process(&buf)?;
            }
        }
        Ok(buf)
    }

    /// Sum of node latency declarations in frames (0 for now; individual nodes
    /// may expose this in a future API).
    pub fn latency_hint(&self) -> usize {
        0
    }

    /// Total declared latency in frames across all nodes in this pipeline.
    ///
    /// Returns `None` if the pipeline is empty. Nodes that return `None` from
    /// [`AudioNode::latency_frames`] are treated as contributing 0 frames.
    ///
    /// The returned value is the sum of all non-`None` latency declarations.
    #[must_use]
    pub fn total_latency_frames(&self) -> Option<usize> {
        if self.nodes.is_empty() {
            return None;
        }
        let total = self
            .nodes
            .iter()
            .map(|n| n.latency_frames().unwrap_or(0))
            .sum();
        Some(total)
    }
}

impl Default for AudioPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ParallelBranchNode
// ---------------------------------------------------------------------------

/// A pipeline node that splits the input across N branches, processes each branch
/// independently (sequentially), then mixes the results.
///
/// Each branch is an [`AudioPipeline`] with its own node chain. After all branches
/// process the input, their outputs are summed sample-by-sample and scaled by
/// `1.0 / branch_count` to normalize amplitude.
///
/// # Use cases
/// - Parallel EQ bands
/// - Dry/wet splits (one branch = dry pass-through, another = effects chain)
/// - Harmonic distortion synthesis (fundamental + harmonics)
///
/// # Examples
///
/// ```
/// use oxiaudio_core::{AudioBuffer, AudioNode, AudioPipeline, OxiAudioError,
///                     ChannelLayout, SampleFormat, ParallelBranchNode};
///
/// struct PassThrough;
/// impl AudioNode for PassThrough {
///     fn name(&self) -> &str { "pass" }
///     fn process(&self, input: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
///         Ok(input.clone())
///     }
/// }
///
/// let branch1 = AudioPipeline::new().push_node(Box::new(PassThrough));
/// let branch2 = AudioPipeline::new().push_node(Box::new(PassThrough));
/// let node = ParallelBranchNode::new(vec![branch1, branch2]).unwrap();
///
/// let buf = AudioBuffer {
///     samples: vec![1.0f32; 4],
///     sample_rate: 44_100,
///     channels: ChannelLayout::Mono,
///     format: SampleFormat::F32,
/// };
/// let out = node.process(&buf).unwrap();
/// for &s in &out.samples {
///     assert!((s - 1.0).abs() < 1e-6);
/// }
/// ```
pub struct ParallelBranchNode {
    branches: Vec<AudioPipeline>,
    gain_per_branch: f32,
    node_name: String,
}

impl ParallelBranchNode {
    /// Create a parallel node from the given branches.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::InvalidChannelLayout`] if `branches` is empty.
    #[must_use = "creating a ParallelBranchNode returns a Result that must be used"]
    pub fn new(branches: Vec<AudioPipeline>) -> Result<Self, OxiAudioError> {
        if branches.is_empty() {
            return Err(OxiAudioError::InvalidChannelLayout(
                "ParallelBranchNode requires at least one branch".into(),
            ));
        }
        let gain_per_branch = 1.0 / branches.len() as f32;
        let node_name = format!("parallel({})", branches.len());
        Ok(Self {
            branches,
            gain_per_branch,
            node_name,
        })
    }

    /// Builder-style method to append an additional branch.
    ///
    /// The gain normalization factor and node name are recomputed automatically.
    #[must_use]
    pub fn with_branch(mut self, branch: AudioPipeline) -> Self {
        self.branches.push(branch);
        self.gain_per_branch = 1.0 / self.branches.len() as f32;
        self.node_name = format!("parallel({})", self.branches.len());
        self
    }
}

impl AudioNode for ParallelBranchNode {
    fn process(&self, input: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
        // Process every branch and collect outputs.
        let outputs: Vec<AudioBuffer<f32>> = self
            .branches
            .iter()
            .map(|branch| branch.process(input))
            .collect::<Result<_, _>>()?;

        // Use the minimum output length across all branches for mixing.
        let mix_len = outputs.iter().map(|o| o.samples.len()).min().unwrap_or(0);

        let mut mixed = vec![0.0f32; mix_len];
        for output in &outputs {
            for (dst, &src) in mixed.iter_mut().zip(output.samples.iter()) {
                *dst += src * self.gain_per_branch;
            }
        }

        Ok(AudioBuffer {
            samples: mixed,
            ..*input
        })
    }

    fn name(&self) -> &str {
        &self.node_name
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChannelLayout, SampleFormat};

    struct PassThrough;

    impl AudioNode for PassThrough {
        fn name(&self) -> &str {
            "pass-through"
        }

        fn process(&self, input: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
            Ok(input.clone())
        }
    }

    fn make_branch() -> AudioPipeline {
        AudioPipeline::new().push_node(Box::new(PassThrough))
    }

    fn ones_buf(n: usize) -> AudioBuffer<f32> {
        AudioBuffer {
            samples: vec![1.0f32; n],
            sample_rate: 44_100,
            channels: ChannelLayout::Mono,
            format: SampleFormat::F32,
        }
    }

    #[test]
    fn test_total_latency_frames_empty_pipeline() {
        let pipeline = AudioPipeline::new();
        assert_eq!(
            pipeline.total_latency_frames(),
            None,
            "empty pipeline should return None"
        );
    }

    #[test]
    fn test_total_latency_frames_default_nodes_sum_zero() {
        // PassThrough uses the default latency_frames() → Some(0)
        let pipeline = AudioPipeline::new()
            .push_node(Box::new(PassThrough))
            .push_node(Box::new(PassThrough));
        assert_eq!(
            pipeline.total_latency_frames(),
            Some(0),
            "default latency nodes should sum to 0"
        );
    }

    #[test]
    fn test_total_latency_frames_custom_latency() {
        struct DelayNode {
            frames: usize,
        }
        impl AudioNode for DelayNode {
            fn name(&self) -> &str {
                "delay"
            }
            fn latency_frames(&self) -> Option<usize> {
                Some(self.frames)
            }
            fn process(&self, input: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
                Ok(input.clone())
            }
        }

        let pipeline = AudioPipeline::new()
            .push_node(Box::new(DelayNode { frames: 100 }))
            .push_node(Box::new(DelayNode { frames: 56 }))
            .push_node(Box::new(PassThrough)); // contributes 0
        assert_eq!(
            pipeline.total_latency_frames(),
            Some(156),
            "custom latency nodes should accumulate"
        );
    }

    #[test]
    fn test_total_latency_frames_unknown_latency_treated_as_zero() {
        struct UnknownLatency;
        impl AudioNode for UnknownLatency {
            fn name(&self) -> &str {
                "unknown"
            }
            fn latency_frames(&self) -> Option<usize> {
                None
            }
            fn process(&self, input: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
                Ok(input.clone())
            }
        }

        let pipeline = AudioPipeline::new()
            .push_node(Box::new(UnknownLatency))
            .push_node(Box::new(PassThrough));
        // Both contribute 0 (None → 0, default → 0)
        assert_eq!(pipeline.total_latency_frames(), Some(0));
    }

    #[test]
    fn test_parallel_branch_two_branches() {
        let node = ParallelBranchNode::new(vec![make_branch(), make_branch()]).unwrap();
        let buf = ones_buf(4);
        let out = node.process(&buf).unwrap();
        assert_eq!(out.samples.len(), 4);
        for &s in &out.samples {
            assert!((s - 1.0).abs() < 1e-6, "expected 1.0, got {s}");
        }
    }

    #[test]
    fn test_parallel_branch_empty_returns_err() {
        let result = ParallelBranchNode::new(vec![]);
        assert!(
            matches!(result, Err(OxiAudioError::InvalidChannelLayout(_))),
            "empty branches should return InvalidChannelLayout error"
        );
    }

    #[test]
    fn test_parallel_branch_with_branch() {
        let node = ParallelBranchNode::new(vec![make_branch()])
            .unwrap()
            .with_branch(make_branch());
        assert_eq!(node.name(), "parallel(2)");
    }

    #[test]
    fn test_parallel_branch_single_branch_passthrough() {
        let node = ParallelBranchNode::new(vec![make_branch()]).unwrap();
        let buf = ones_buf(8);
        let out = node.process(&buf).unwrap();
        assert_eq!(out.samples.len(), 8);
        for &s in &out.samples {
            assert!(
                (s - 1.0).abs() < 1e-6,
                "single branch: expected 1.0, got {s}"
            );
        }
    }

    #[test]
    fn test_parallel_branch_name() {
        let node =
            ParallelBranchNode::new(vec![make_branch(), make_branch(), make_branch()]).unwrap();
        assert_eq!(node.name(), "parallel(3)");
    }

    #[test]
    fn test_parallel_branch_gain_normalization() {
        // A branch that doubles every sample.
        struct Double;
        impl AudioNode for Double {
            fn name(&self) -> &str {
                "double"
            }
            fn process(&self, input: &AudioBuffer<f32>) -> Result<AudioBuffer<f32>, OxiAudioError> {
                let samples = input.samples.iter().map(|&s| s * 2.0).collect();
                Ok(AudioBuffer { samples, ..*input })
            }
        }

        // Branch 1: pass-through → output 1.0
        // Branch 2: double       → output 2.0
        // Mixed: (1.0 + 2.0) * 0.5 = 1.5
        let branch1 = AudioPipeline::new().push_node(Box::new(PassThrough));
        let branch2 = AudioPipeline::new().push_node(Box::new(Double));
        let node = ParallelBranchNode::new(vec![branch1, branch2]).unwrap();

        let buf = ones_buf(4);
        let out = node.process(&buf).unwrap();
        for &s in &out.samples {
            assert!((s - 1.5).abs() < 1e-6, "expected 1.5, got {s}");
        }
    }

    #[test]
    fn test_parallel_branch_preserves_metadata() {
        let node = ParallelBranchNode::new(vec![make_branch()]).unwrap();
        let buf = AudioBuffer {
            samples: vec![0.5f32; 2],
            sample_rate: 48_000,
            channels: ChannelLayout::Stereo,
            format: SampleFormat::F32,
        };
        let out = node.process(&buf).unwrap();
        assert_eq!(out.sample_rate, 48_000);
        assert_eq!(out.channels, ChannelLayout::Stereo);
        assert_eq!(out.format, SampleFormat::F32);
    }
}
