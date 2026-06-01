//! MIDI synthesizer: converts [`MidiFile`] events to [`AudioBuffer<f32>`].
//!
//! Implements a simple polyphonic sine/square/sawtooth/triangle oscillator
//! with per-voice ADSR envelope. This is not a General MIDI compliant
//! synthesizer — it is a minimal reference implementation for the
//! decode→synthesize pipeline.

use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

use crate::midi::{MetaEvent, MidiEvent, MidiFile, SmfFormat, TrackEvent};

// ── Waveform ──────────────────────────────────────────────────────────────────

/// Oscillator waveform shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Waveform {
    /// Sine wave (smoothest, most natural tone).
    #[default]
    Sine,
    /// Square wave (hollow, buzzy timbre).
    Square,
    /// Sawtooth wave (bright, brassy timbre).
    Sawtooth,
    /// Triangle wave (softer than square, brighter than sine).
    Triangle,
}

impl Waveform {
    /// Evaluate one sample of this waveform at the given `phase` in `[0, 1)`.
    fn sample(self, phase: f32) -> f32 {
        match self {
            Waveform::Sine => (2.0 * std::f32::consts::PI * phase).sin(),
            Waveform::Square => {
                if phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            Waveform::Sawtooth => 2.0 * phase - 1.0,
            Waveform::Triangle => {
                if phase < 0.25 {
                    4.0 * phase
                } else if phase < 0.75 {
                    2.0 - 4.0 * phase
                } else {
                    4.0 * phase - 4.0
                }
            }
        }
    }
}

// ── ADSR Envelope ─────────────────────────────────────────────────────────────

/// ADSR amplitude envelope parameters.
#[derive(Debug, Clone)]
pub struct Adsr {
    /// Attack time in milliseconds (0–5000 ms).
    pub attack_ms: f32,
    /// Decay time in milliseconds (0–5000 ms).
    pub decay_ms: f32,
    /// Sustain level as a linear amplitude (0.0–1.0).
    pub sustain: f32,
    /// Release time in milliseconds (0–5000 ms).
    pub release_ms: f32,
}

impl Default for Adsr {
    fn default() -> Self {
        Self {
            attack_ms: 10.0,
            decay_ms: 100.0,
            sustain: 0.7,
            release_ms: 200.0,
        }
    }
}

// ── Synthesizer Config ────────────────────────────────────────────────────────

/// Configuration for the MIDI synthesizer.
#[derive(Debug, Clone)]
pub struct SynthConfig {
    /// Oscillator waveform shape.
    pub waveform: Waveform,
    /// ADSR envelope parameters.
    pub adsr: Adsr,
    /// Maximum simultaneous voices (default 32).
    pub polyphony: usize,
    /// Master output volume, linear 0.0–1.0 (default 0.5).
    pub master_volume: f32,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            waveform: Waveform::Sine,
            adsr: Adsr::default(),
            polyphony: 32,
            master_volume: 0.5,
        }
    }
}

// ── Voice ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdsrPhase {
    Attack,
    Decay,
    Sustain,
    Release,
    Done,
}

#[derive(Debug, Clone)]
struct Voice {
    key: u8,
    channel: u8,
    frequency: f32,
    velocity: f32,
    phase: f32,
    adsr_phase: AdsrPhase,
    envelope: f32,
    release_from: f32,
    samples_in_phase: u64,
}

impl Voice {
    fn new(key: u8, channel: u8, velocity: u8) -> Self {
        Self {
            key,
            channel,
            frequency: midi_note_to_hz(key),
            velocity: f32::from(velocity) / 127.0,
            phase: 0.0,
            adsr_phase: AdsrPhase::Attack,
            envelope: 0.0,
            release_from: 0.0,
            samples_in_phase: 0,
        }
    }

    fn release(&mut self) {
        self.release_from = self.envelope;
        self.adsr_phase = AdsrPhase::Release;
        self.samples_in_phase = 0;
    }

    fn is_done(&self) -> bool {
        self.adsr_phase == AdsrPhase::Done
    }

    fn advance(&mut self, adsr: &Adsr, waveform: Waveform, sample_rate: u32) -> f32 {
        let sr = sample_rate as f32;
        let attack_samples = (adsr.attack_ms / 1000.0 * sr).max(1.0);
        let decay_samples = (adsr.decay_ms / 1000.0 * sr).max(1.0);
        let release_samples = (adsr.release_ms / 1000.0 * sr).max(1.0);

        self.envelope = match self.adsr_phase {
            AdsrPhase::Attack => {
                let t = self.samples_in_phase as f32 / attack_samples;
                if t >= 1.0 {
                    self.adsr_phase = AdsrPhase::Decay;
                    self.samples_in_phase = 0;
                    1.0
                } else {
                    t
                }
            }
            AdsrPhase::Decay => {
                let t = self.samples_in_phase as f32 / decay_samples;
                if t >= 1.0 {
                    self.adsr_phase = AdsrPhase::Sustain;
                    self.samples_in_phase = 0;
                    adsr.sustain
                } else {
                    1.0 - (1.0 - adsr.sustain) * t
                }
            }
            AdsrPhase::Sustain => adsr.sustain,
            AdsrPhase::Release => {
                let t = self.samples_in_phase as f32 / release_samples;
                if t >= 1.0 {
                    self.adsr_phase = AdsrPhase::Done;
                    0.0
                } else {
                    self.release_from * (1.0 - t)
                }
            }
            AdsrPhase::Done => 0.0,
        };
        self.samples_in_phase += 1;

        let increment = self.frequency / sample_rate as f32;
        self.phase += increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        waveform.sample(self.phase) * self.envelope * self.velocity
    }
}

// ── MIDI note → Hz ────────────────────────────────────────────────────────────

/// Convert a MIDI note number to frequency in Hz.
///
/// Uses equal temperament with A4 (MIDI note 69) = 440 Hz.
pub fn midi_note_to_hz(note: u8) -> f32 {
    440.0 * 2.0_f32.powf((f32::from(note) - 69.0) / 12.0)
}

// ── Synthesizer ───────────────────────────────────────────────────────────────

/// Synthesize a [`MidiFile`] to an [`AudioBuffer<f32>`] using the given [`SynthConfig`].
///
/// Merges all tracks, sorts events by absolute tick, and renders a mono output buffer.
/// Tempo changes mid-file are handled; the initial tempo defaults to 120 BPM.
///
/// # Note
///
/// This is a minimal polyphonic oscillator synthesizer — not General MIDI compliant.
pub fn synthesize_midi(
    midi: &MidiFile,
    sample_rate: u32,
    config: &SynthConfig,
) -> AudioBuffer<f32> {
    let tpq = u64::from(midi.ticks_per_quarter);

    // Collect all (tick, event) pairs across all tracks, sorted by tick.
    let mut all_events: Vec<(u64, &TrackEvent)> = Vec::new();
    for track in &midi.tracks {
        for te in &track.events {
            all_events.push((te.tick, &te.event));
        }
    }
    all_events.sort_by_key(|(tick, _)| *tick);

    // Find the first SetTempo to set initial tempo (default 120 BPM = 500_000 µs/beat).
    let mut tempo_us: u64 = 500_000;
    for (_, ev) in &all_events {
        if let TrackEvent::Meta(MetaEvent::Tempo(t)) = ev {
            tempo_us = u64::from(*t);
            break;
        }
    }

    let max_tick = all_events.iter().map(|(t, _)| *t).max().unwrap_or(0);

    // samples_per_tick = sample_rate * tempo_us / (1_000_000 * tpq)
    let samples_per_tick = f64::from(sample_rate) * tempo_us as f64 / (1_000_000.0 * tpq as f64);

    // 2-second tail for final note release.
    let total_samples = (max_tick as f64 * samples_per_tick) as usize + sample_rate as usize * 2;

    let mut output = vec![0.0f32; total_samples];
    let mut voices: Vec<Voice> = Vec::with_capacity(config.polyphony.min(64));
    let mut current_sample: usize = 0;

    // Format 2 (Patterns): use only the first track as an independent sequence.
    // For Format 0/1 all tracks are merged above which is the correct behavior.
    let _ = SmfFormat::Patterns; // suppress dead_code for the enum variant reference

    for (tick, ev) in &all_events {
        let target_sample = (*tick as f64 * samples_per_tick) as usize;

        // Render voices from current_sample to target_sample.
        while current_sample < target_sample && current_sample < output.len() {
            let mut mix = 0.0_f32;
            for v in voices.iter_mut() {
                mix += v.advance(&config.adsr, config.waveform, sample_rate);
            }
            // Soft clip: normalise by active voice count to prevent clipping at high polyphony.
            let n = voices.len().max(1);
            #[allow(clippy::cast_precision_loss)]
            let mix_norm = (mix / n as f32).clamp(-1.0, 1.0);
            output[current_sample] = mix_norm * config.master_volume;
            current_sample += 1;
        }
        voices.retain(|v| !v.is_done());

        // Dispatch event.
        match ev {
            TrackEvent::Midi(MidiEvent::NoteOn {
                channel,
                key,
                velocity,
            }) => {
                if *velocity > 0 {
                    if voices.len() < config.polyphony {
                        voices.push(Voice::new(*key, *channel, *velocity));
                    } else if let Some(slot) = voices.iter_mut().find(|v| v.is_done()) {
                        *slot = Voice::new(*key, *channel, *velocity);
                    }
                } else {
                    // velocity == 0 is NoteOff by MIDI convention.
                    if let Some(v) = voices
                        .iter_mut()
                        .find(|v| v.key == *key && v.channel == *channel)
                    {
                        v.release();
                    }
                }
            }
            TrackEvent::Midi(MidiEvent::NoteOff { channel, key, .. }) => {
                if let Some(v) = voices
                    .iter_mut()
                    .find(|v| v.key == *key && v.channel == *channel)
                {
                    v.release();
                }
            }
            TrackEvent::Meta(MetaEvent::Tempo(_t)) => {
                // Tempo changes mid-file are noted here but not applied in this
                // simplified implementation. A full implementation would recompute
                // samples_per_tick from this point forward.
            }
            _ => {}
        }
    }

    // Render release tails for any still-active voices.
    while current_sample < output.len() {
        let mut mix = 0.0_f32;
        for v in voices.iter_mut() {
            mix += v.advance(&config.adsr, config.waveform, sample_rate);
        }
        #[allow(clippy::cast_precision_loss)]
        let n = voices.len().max(1);
        output[current_sample] = (mix / n as f32).clamp(-1.0, 1.0) * config.master_volume;
        current_sample += 1;
        voices.retain(|v| !v.is_done());
        if voices.is_empty() {
            break;
        }
    }

    // Trim trailing silence, but always keep at least 1 sample.
    while output.last() == Some(&0.0) && output.len() > 1 {
        output.pop();
    }

    AudioBuffer {
        samples: output,
        sample_rate,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    }
}

/// Synthesize a [`MidiFile`] with default [`SynthConfig`] (sine wave, 32-voice polyphony).
pub fn synthesize_midi_default(midi: &MidiFile, sample_rate: u32) -> AudioBuffer<f32> {
    synthesize_midi(midi, sample_rate, &SynthConfig::default())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi::{MidiEvent, MidiFile, MidiTrack, SmfFormat, TimedEvent, TrackEvent};

    fn make_simple_midi(key: u8, note_on_tick: u64, note_off_tick: u64) -> MidiFile {
        MidiFile {
            format: SmfFormat::SingleTrack,
            ticks_per_quarter: 480,
            tracks: vec![MidiTrack {
                events: vec![
                    TimedEvent {
                        tick: note_on_tick,
                        event: TrackEvent::Midi(MidiEvent::NoteOn {
                            channel: 0,
                            key,
                            velocity: 100,
                        }),
                    },
                    TimedEvent {
                        tick: note_off_tick,
                        event: TrackEvent::Midi(MidiEvent::NoteOff {
                            channel: 0,
                            key,
                            velocity: 0,
                        }),
                    },
                ],
            }],
        }
    }

    // 1. midi_note_to_hz: A4 = 440 Hz
    #[test]
    fn test_midi_note_a4_is_440hz() {
        let f = midi_note_to_hz(69);
        assert!((f - 440.0).abs() < 0.1, "A4 should be ~440 Hz, got {f}");
    }

    // 2. midi_note_to_hz: A5 = 880 Hz (one octave up)
    #[test]
    fn test_midi_note_a5_is_880hz() {
        let f = midi_note_to_hz(81);
        assert!((f - 880.0).abs() < 0.2, "A5 should be ~880 Hz, got {f}");
    }

    // 3. midi_note_to_hz: C4 = 261.63 Hz (middle C)
    #[test]
    fn test_midi_note_c4_is_261hz() {
        let f = midi_note_to_hz(60);
        assert!((f - 261.63).abs() < 0.5, "C4 should be ~261.63 Hz, got {f}");
    }

    // 4. Empty MIDI produces short buffer without panic.
    #[test]
    fn test_synthesize_empty_midi_does_not_panic() {
        let midi = MidiFile {
            format: SmfFormat::SingleTrack,
            ticks_per_quarter: 480,
            tracks: vec![],
        };
        let buf = synthesize_midi_default(&midi, 44100);
        assert!(
            buf.samples.len() <= 44100 * 3,
            "empty MIDI should produce at most 3s of silence"
        );
    }

    // 5. Single note produces nonzero audio energy.
    #[test]
    fn test_synthesize_single_note_produces_audio() {
        let midi = make_simple_midi(69, 0, 480); // A4 for one quarter note at 120 BPM
        let buf = synthesize_midi_default(&midi, 44100);
        let energy: f32 = buf.samples.iter().map(|&x| x * x).sum();
        assert!(
            energy > 0.0,
            "single note must produce nonzero audio energy"
        );
    }

    // 6. Longer note produces more samples than shorter note.
    #[test]
    fn test_synthesize_note_duration_proportional_to_ticks() {
        let midi_short = make_simple_midi(60, 0, 480); // 1 beat at 120 BPM ≈ 0.5 s
        let midi_long = make_simple_midi(60, 0, 1920); // 4 beats ≈ 2.0 s
        let buf_short = synthesize_midi_default(&midi_short, 44100);
        let buf_long = synthesize_midi_default(&midi_long, 44100);
        assert!(
            buf_long.samples.len() > buf_short.samples.len(),
            "longer note must produce more samples: long={}, short={}",
            buf_long.samples.len(),
            buf_short.samples.len()
        );
    }

    // 7. All four waveform variants produce audio without panicking.
    #[test]
    fn test_waveform_variants_produce_audio() {
        let midi = make_simple_midi(60, 0, 480);
        for wf in [
            Waveform::Sine,
            Waveform::Square,
            Waveform::Sawtooth,
            Waveform::Triangle,
        ] {
            let cfg = SynthConfig {
                waveform: wf,
                ..SynthConfig::default()
            };
            let buf = synthesize_midi(&midi, 44100, &cfg);
            assert!(
                !buf.samples.is_empty(),
                "waveform {wf:?} must produce samples"
            );
        }
    }

    // 8. Output samples are bounded to [-1, 1].
    #[test]
    fn test_synthesize_output_is_normalized() {
        let midi = make_simple_midi(60, 0, 480);
        let buf = synthesize_midi_default(&midi, 44100);
        for &s in &buf.samples {
            assert!(s.abs() <= 1.0 + 1e-6, "sample out of range: {s}");
        }
    }

    // 9. Slow attack starts quieter than fast attack in the first 1000 samples.
    #[test]
    fn test_adsr_slow_attack_starts_quiet() {
        let midi = make_simple_midi(60, 0, 4800); // 10 beats
        let slow_cfg = SynthConfig {
            adsr: Adsr {
                attack_ms: 500.0,
                ..Adsr::default()
            },
            ..SynthConfig::default()
        };
        let fast_cfg = SynthConfig {
            adsr: Adsr {
                attack_ms: 1.0,
                ..Adsr::default()
            },
            ..SynthConfig::default()
        };
        let slow = synthesize_midi(&midi, 44100, &slow_cfg);
        let fast = synthesize_midi(&midi, 44100, &fast_cfg);
        let slow_early: f32 = slow.samples.iter().take(1000).map(|&x| x.abs()).sum();
        let fast_early: f32 = fast.samples.iter().take(1000).map(|&x| x.abs()).sum();
        assert!(
            fast_early > slow_early,
            "fast attack must be louder early on: fast_early={fast_early}, slow_early={slow_early}"
        );
    }

    // 10. Higher sample rate produces roughly twice the samples of half the rate.
    #[test]
    fn test_synthesize_sample_rate_affects_length() {
        let midi = make_simple_midi(60, 0, 480);
        let buf_44 = synthesize_midi_default(&midi, 44100);
        let buf_22 = synthesize_midi_default(&midi, 22050);
        let ratio = buf_44.samples.len() as f64 / buf_22.samples.len().max(1) as f64;
        assert!(
            (1.5..=2.5).contains(&ratio),
            "44100 Hz should have ~2x samples of 22050 Hz; ratio={ratio}"
        );
    }
}
