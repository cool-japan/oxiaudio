//! MIDI Standard MIDI File (SMF) parser.
//!
//! Supports Format 0 (single multi-channel track), Format 1 (multiple simultaneous tracks),
//! and Format 2 (multiple independent patterns). Parses:
//!
//! - `MThd` chunk: format, track count, ticks-per-quarter-note
//! - `MTrk` chunks: variable-length delta times (VLQ), meta events, MIDI channel messages
//!   with running status
//! - All standard channel messages: NoteOn/NoteOff, ControlChange, ProgramChange,
//!   PitchBend, ChannelPressure, PolyKeyPressure
//! - Standard meta events: Tempo, TimeSignature, KeySignature, EndOfTrack, TrackName,
//!   InstrumentName, Lyric, Marker, CuePoint, and a catch-all `Other`
//!
//! # Example
//!
//! ```no_run
//! use oxiaudio_decode::midi::MidiFile;
//!
//! let midi = MidiFile::from_path("song.mid").unwrap();
//! println!("Format {:?}, {} tracks, {} tpq", midi.format, midi.tracks.len(), midi.ticks_per_quarter);
//! ```

use oxiaudio_core::OxiAudioError;
use std::io::Read;
use std::path::Path;

// ─── Public types ─────────────────────────────────────────────────────────────

/// SMF format variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmfFormat {
    /// Format 0: a single track containing all channel events.
    SingleTrack,
    /// Format 1: one or more simultaneous tracks sharing the same tempo map.
    MultiTrack,
    /// Format 2: one or more independent pattern tracks.
    Patterns,
}

/// A complete parsed MIDI file.
#[derive(Debug, Clone)]
pub struct MidiFile {
    /// SMF format (0, 1, or 2).
    pub format: SmfFormat,
    /// Ticks per quarter-note (PPQN).  Negative (SMPTE) values are not supported.
    pub ticks_per_quarter: u16,
    /// Parsed tracks, in file order.
    pub tracks: Vec<MidiTrack>,
}

/// A single `MTrk` chunk containing a list of time-stamped events.
#[derive(Debug, Clone, Default)]
pub struct MidiTrack {
    /// Events in this track, each with an absolute tick position.
    pub events: Vec<TimedEvent>,
}

/// A time-stamped MIDI event (absolute ticks from the start of the track).
#[derive(Debug, Clone)]
pub struct TimedEvent {
    /// Absolute tick position within the track.
    pub tick: u64,
    /// The event payload.
    pub event: TrackEvent,
}

/// The payload of a single MIDI track event.
#[derive(Debug, Clone)]
pub enum TrackEvent {
    /// MIDI channel message.
    Midi(MidiEvent),
    /// SMF meta event.
    Meta(MetaEvent),
    /// SysEx (0xF0 / 0xF7) — data bytes only, length already consumed.
    SysEx(Vec<u8>),
}

/// MIDI channel message variants.
#[derive(Debug, Clone)]
pub enum MidiEvent {
    /// Note-off (`0x80`): `channel`, MIDI key (0–127), velocity (0–127).
    NoteOff { channel: u8, key: u8, velocity: u8 },
    /// Note-on (`0x90`): velocity 0 is treated as a note-off by convention.
    NoteOn { channel: u8, key: u8, velocity: u8 },
    /// Polyphonic key pressure / aftertouch (`0xA0`).
    PolyKeyPressure { channel: u8, key: u8, pressure: u8 },
    /// Control change (`0xB0`).
    ControlChange {
        channel: u8,
        controller: u8,
        value: u8,
    },
    /// Program change (`0xC0`).
    ProgramChange { channel: u8, program: u8 },
    /// Channel pressure / aftertouch (`0xD0`).
    ChannelPressure { channel: u8, pressure: u8 },
    /// Pitch-bend change (`0xE0`): `value` is 14-bit signed, centred at 0x2000.
    PitchBend { channel: u8, value: i16 },
}

/// SMF meta event variants.
#[derive(Debug, Clone)]
pub enum MetaEvent {
    /// Tempo in microseconds per quarter-note.
    Tempo(u32),
    /// Time signature: numerator, log2(denominator), clocks per metronome beat, 32nds per quarter.
    TimeSignature {
        numer: u8,
        denom_log2: u8,
        clocks: u8,
        thirtyseconds: u8,
    },
    /// Key signature: `sharps` is −7..+7 (negative = flats), `minor` = true for minor key.
    KeySignature { sharps: i8, minor: bool },
    /// Track name (`0x03`).
    TrackName(String),
    /// Instrument name (`0x04`).
    InstrumentName(String),
    /// Lyric text (`0x05`).
    Lyric(String),
    /// Marker text (`0x06`).
    Marker(String),
    /// Cue point text (`0x07`).
    CuePoint(String),
    /// End of track (`0x2F`).
    EndOfTrack,
    /// Any unrecognised meta event: (type byte, raw data).
    Other(u8, Vec<u8>),
}

// ─── Entry points ─────────────────────────────────────────────────────────────

impl MidiFile {
    /// Parse a MIDI file from a byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Decode`] on malformed data, truncated input, or unsupported
    /// SMPTE time codes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, OxiAudioError> {
        parse_midi_file(data)
    }

    /// Parse a MIDI file from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns [`OxiAudioError::Io`] if the file cannot be read, or
    /// [`OxiAudioError::Decode`] on malformed MIDI data.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OxiAudioError> {
        let mut buf = Vec::new();
        std::fs::File::open(path.as_ref())
            .map_err(OxiAudioError::Io)?
            .read_to_end(&mut buf)
            .map_err(OxiAudioError::Io)?;
        parse_midi_file(&buf)
    }
}

// ─── Internal parser ──────────────────────────────────────────────────────────

/// Cursor into a byte slice that advances as data is consumed.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    fn read_byte(&mut self) -> Result<u8, OxiAudioError> {
        self.data
            .get(self.pos)
            .copied()
            .ok_or_else(|| OxiAudioError::Decode("MIDI: unexpected end of data".into()))
            .inspect(|_| {
                self.pos += 1;
            })
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], OxiAudioError> {
        let start = self.pos;
        let end = start
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| {
                OxiAudioError::Decode(format!(
                    "MIDI: need {n} bytes but only {} remain",
                    self.remaining()
                ))
            })?;
        self.pos = end;
        Ok(&self.data[start..end])
    }

    fn read_u16_be(&mut self) -> Result<u16, OxiAudioError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn read_u32_be(&mut self) -> Result<u32, OxiAudioError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a MIDI variable-length quantity (VLQ) — up to 4 bytes, 7 bits each.
    fn read_vlq(&mut self) -> Result<u32, OxiAudioError> {
        let mut value: u32 = 0;
        for _ in 0..4 {
            let byte = self.read_byte()?;
            value = value
                .checked_shl(7)
                .ok_or_else(|| OxiAudioError::Decode("MIDI: VLQ overflow".into()))?
                | u32::from(byte & 0x7F);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(OxiAudioError::Decode(
            "MIDI: VLQ exceeds 4 bytes (malformed)".into(),
        ))
    }

    /// Read a 4-byte chunk tag.
    fn read_tag(&mut self) -> Result<[u8; 4], OxiAudioError> {
        let b = self.read_bytes(4)?;
        Ok([b[0], b[1], b[2], b[3]])
    }
}

// ─── Top-level file parser ─────────────────────────────────────────────────────

fn parse_midi_file(data: &[u8]) -> Result<MidiFile, OxiAudioError> {
    let mut cur = Cursor::new(data);

    // ── MThd chunk ──────────────────────────────────────────────────────────────
    let tag = cur.read_tag()?;
    if &tag != b"MThd" {
        return Err(OxiAudioError::Decode(format!(
            "MIDI: expected 'MThd' magic, got {:?}",
            core::str::from_utf8(&tag).unwrap_or("<non-utf8>")
        )));
    }

    let header_len = cur.read_u32_be()?;
    if header_len < 6 {
        return Err(OxiAudioError::Decode(format!(
            "MIDI: MThd length must be ≥ 6, got {header_len}"
        )));
    }

    let format_word = cur.read_u16_be()?;
    let num_tracks = cur.read_u16_be()? as usize;
    let division = cur.read_u16_be()?;

    // Skip any extra MThd bytes (extensions).
    let extra = header_len as usize - 6;
    if extra > 0 {
        cur.read_bytes(extra)?;
    }

    let format = match format_word {
        0 => SmfFormat::SingleTrack,
        1 => SmfFormat::MultiTrack,
        2 => SmfFormat::Patterns,
        other => {
            return Err(OxiAudioError::UnsupportedFormat(format!(
                "MIDI: unknown SMF format {other}"
            )));
        }
    };

    // SMPTE time codes (bit 15 = 1) are not supported.
    if division & 0x8000 != 0 {
        return Err(OxiAudioError::UnsupportedFormat(
            "MIDI: SMPTE timecode divisions are not supported".into(),
        ));
    }
    let ticks_per_quarter = division;

    // ── MTrk chunks ─────────────────────────────────────────────────────────────
    let mut tracks = Vec::with_capacity(num_tracks);

    for track_idx in 0..num_tracks {
        // Some SMF files have fewer MTrk chunks than the header declares; be lenient.
        if cur.remaining() < 8 {
            break;
        }

        let tag = cur.read_tag()?;
        let chunk_len = cur.read_u32_be()? as usize;

        if &tag != b"MTrk" {
            // Skip unknown chunk types.
            cur.read_bytes(chunk_len).map_err(|_| {
                OxiAudioError::Decode(format!(
                    "MIDI: skipping unknown chunk at track {track_idx}, but data truncated"
                ))
            })?;
            // Try again for the next expected track.
            tracks.push(MidiTrack::default());
            continue;
        }

        let track_data = cur.read_bytes(chunk_len)?;
        let track = parse_track(track_data, track_idx)?;
        tracks.push(track);
    }

    Ok(MidiFile {
        format,
        ticks_per_quarter,
        tracks,
    })
}

// ─── Track parser ──────────────────────────────────────────────────────────────

fn parse_track(data: &[u8], track_idx: usize) -> Result<MidiTrack, OxiAudioError> {
    let mut cur = Cursor::new(data);
    let mut events: Vec<TimedEvent> = Vec::new();
    let mut abs_tick: u64 = 0;
    // Running status: the last status byte seen (0 = none).
    let mut running_status: u8 = 0;

    while cur.remaining() > 0 {
        let delta = cur.read_vlq()?;
        abs_tick = abs_tick.saturating_add(u64::from(delta));

        // Determine the status byte, applying running status when needed.
        let first = cur
            .peek()
            .ok_or_else(|| OxiAudioError::Decode("MIDI: truncated track event".into()))?;

        let status = if first & 0x80 != 0 {
            // Fresh status byte — consume it.
            cur.read_byte()?;
            // SysEx (0xF0/0xF7) and meta (0xFF) cancel running status per RP-001.
            if first != 0xF0 && first != 0xF7 && first != 0xFF {
                running_status = first;
            } else {
                // Meta and SysEx: clear running status.
                running_status = 0;
            }
            first
        } else {
            // Data byte: reuse running status (no consume).
            if running_status == 0 {
                return Err(OxiAudioError::Decode(format!(
                    "MIDI: data byte 0x{first:02X} with no running status at track {track_idx}"
                )));
            }
            running_status
        };

        let event = match status {
            // ── Meta event ────────────────────────────────────────────────────
            0xFF => {
                let meta_type = cur.read_byte()?;
                let len = cur.read_vlq()? as usize;
                let payload = cur.read_bytes(len)?.to_vec();
                let meta = parse_meta_event(meta_type, &payload)?;
                TrackEvent::Meta(meta)
            }
            // ── SysEx ─────────────────────────────────────────────────────────
            0xF0 | 0xF7 => {
                let len = cur.read_vlq()? as usize;
                let payload = cur.read_bytes(len)?.to_vec();
                TrackEvent::SysEx(payload)
            }
            // ── Channel messages ──────────────────────────────────────────────
            s if s & 0xF0 == 0x80 => {
                // Note Off
                let key = cur.read_byte()?;
                let vel = cur.read_byte()?;
                TrackEvent::Midi(MidiEvent::NoteOff {
                    channel: s & 0x0F,
                    key: key & 0x7F,
                    velocity: vel & 0x7F,
                })
            }
            s if s & 0xF0 == 0x90 => {
                // Note On (vel 0 = note-off by convention, preserved as NoteOn here)
                let key = cur.read_byte()?;
                let vel = cur.read_byte()?;
                TrackEvent::Midi(MidiEvent::NoteOn {
                    channel: s & 0x0F,
                    key: key & 0x7F,
                    velocity: vel & 0x7F,
                })
            }
            s if s & 0xF0 == 0xA0 => {
                // Poly Key Pressure
                let key = cur.read_byte()?;
                let pressure = cur.read_byte()?;
                TrackEvent::Midi(MidiEvent::PolyKeyPressure {
                    channel: s & 0x0F,
                    key: key & 0x7F,
                    pressure: pressure & 0x7F,
                })
            }
            s if s & 0xF0 == 0xB0 => {
                // Control Change
                let controller = cur.read_byte()?;
                let value = cur.read_byte()?;
                TrackEvent::Midi(MidiEvent::ControlChange {
                    channel: s & 0x0F,
                    controller: controller & 0x7F,
                    value: value & 0x7F,
                })
            }
            s if s & 0xF0 == 0xC0 => {
                // Program Change (1 data byte)
                let program = cur.read_byte()?;
                TrackEvent::Midi(MidiEvent::ProgramChange {
                    channel: s & 0x0F,
                    program: program & 0x7F,
                })
            }
            s if s & 0xF0 == 0xD0 => {
                // Channel Pressure (1 data byte)
                let pressure = cur.read_byte()?;
                TrackEvent::Midi(MidiEvent::ChannelPressure {
                    channel: s & 0x0F,
                    pressure: pressure & 0x7F,
                })
            }
            s if s & 0xF0 == 0xE0 => {
                // Pitch Bend (14-bit, LSB first)
                let lsb = cur.read_byte()?;
                let msb = cur.read_byte()?;
                let raw = u16::from(lsb & 0x7F) | (u16::from(msb & 0x7F) << 7);
                // Signed: centre is 0x2000 = 8192
                let value = raw as i16 - 0x2000_i16;
                TrackEvent::Midi(MidiEvent::PitchBend {
                    channel: s & 0x0F,
                    value,
                })
            }
            other => {
                // Unknown / unsupported real-time / system status in a track.
                return Err(OxiAudioError::Decode(format!(
                    "MIDI: unhandled status byte 0x{other:02X} in track {track_idx}"
                )));
            }
        };

        events.push(TimedEvent {
            tick: abs_tick,
            event,
        });
    }

    Ok(MidiTrack { events })
}

// ─── Meta event parser ─────────────────────────────────────────────────────────

fn parse_meta_event(meta_type: u8, payload: &[u8]) -> Result<MetaEvent, OxiAudioError> {
    match meta_type {
        // Tempo: 3 bytes, microseconds per quarter note
        0x51 => {
            if payload.len() < 3 {
                return Err(OxiAudioError::Decode(
                    "MIDI: Tempo meta event must have 3 bytes".into(),
                ));
            }
            let us = (u32::from(payload[0]) << 16)
                | (u32::from(payload[1]) << 8)
                | u32::from(payload[2]);
            Ok(MetaEvent::Tempo(us))
        }
        // Time signature: 4 bytes
        0x58 => {
            if payload.len() < 4 {
                return Err(OxiAudioError::Decode(
                    "MIDI: TimeSignature meta event must have 4 bytes".into(),
                ));
            }
            Ok(MetaEvent::TimeSignature {
                numer: payload[0],
                denom_log2: payload[1],
                clocks: payload[2],
                thirtyseconds: payload[3],
            })
        }
        // Key signature: 2 bytes
        0x59 => {
            if payload.len() < 2 {
                return Err(OxiAudioError::Decode(
                    "MIDI: KeySignature meta event must have 2 bytes".into(),
                ));
            }
            let sharps = payload[0] as i8;
            let minor = payload[1] != 0;
            Ok(MetaEvent::KeySignature { sharps, minor })
        }
        // Track name
        0x03 => Ok(MetaEvent::TrackName(lossy_utf8(payload))),
        // Instrument name
        0x04 => Ok(MetaEvent::InstrumentName(lossy_utf8(payload))),
        // Lyric
        0x05 => Ok(MetaEvent::Lyric(lossy_utf8(payload))),
        // Marker
        0x06 => Ok(MetaEvent::Marker(lossy_utf8(payload))),
        // Cue point
        0x07 => Ok(MetaEvent::CuePoint(lossy_utf8(payload))),
        // End of track
        0x2F => Ok(MetaEvent::EndOfTrack),
        // Everything else
        other => Ok(MetaEvent::Other(other, payload.to_vec())),
    }
}

/// Convert bytes to `String`, replacing invalid UTF-8 sequences with U+FFFD.
fn lossy_utf8(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal Format 0 SMF byte vector in memory.
    ///
    /// MThd: format=0, ntrk=1, division=480
    /// MTrk payload (13 bytes):
    ///   delta=0x00, NoteOn ch0 key=60 vel=100  → 00 90 3C 64         (4 bytes)
    ///   delta=0x83 0x60 (= 480 ticks), NoteOff → 83 60 80 3C 00      (5 bytes)
    ///   delta=0x00, EndOfTrack meta             → 00 FF 2F 00         (4 bytes)
    ///   total payload = 13 bytes  → MTrk length = 0x0000000D
    fn minimal_smf0() -> Vec<u8> {
        let mut v = Vec::with_capacity(14 + 8 + 13);
        // MThd
        v.extend_from_slice(b"MThd");
        v.extend_from_slice(&0x00000006u32.to_be_bytes()); // length = 6
        v.extend_from_slice(&0x0000u16.to_be_bytes()); // format 0
        v.extend_from_slice(&0x0001u16.to_be_bytes()); // 1 track
        v.extend_from_slice(&0x01E0u16.to_be_bytes()); // 480 tpq
                                                       // MTrk
        v.extend_from_slice(b"MTrk");
        v.extend_from_slice(&0x0000000Du32.to_be_bytes()); // length = 13 bytes
                                                           // delta=0, NoteOn ch0 key=60 vel=100
        v.push(0x00); // delta = 0
        v.push(0x90); // NoteOn, ch 0
        v.push(0x3C); // key 60
        v.push(0x64); // vel 100
                      // delta=480 = VLQ 0x83 0x60, NoteOff ch0 key=60 vel=0
        v.push(0x83);
        v.push(0x60);
        v.push(0x80); // NoteOff, ch 0
        v.push(0x3C); // key 60
        v.push(0x00); // vel 0
                      // delta=0, EndOfTrack
        v.push(0x00); // delta = 0
        v.push(0xFF); // meta
        v.push(0x2F); // EndOfTrack
        v.push(0x00); // length = 0
        v
    }

    #[test]
    fn test_minimal_smf0_parse() {
        let data = minimal_smf0();
        let midi = MidiFile::from_bytes(&data).expect("parse minimal SMF 0");

        assert_eq!(midi.format, SmfFormat::SingleTrack);
        assert_eq!(midi.ticks_per_quarter, 480);
        assert_eq!(midi.tracks.len(), 1);

        let events = &midi.tracks[0].events;
        // Should have 3 events: NoteOn, NoteOff, EndOfTrack
        assert_eq!(events.len(), 3, "expected 3 events");

        // Event 0: NoteOn at tick 0
        assert_eq!(events[0].tick, 0);
        match &events[0].event {
            TrackEvent::Midi(MidiEvent::NoteOn {
                channel,
                key,
                velocity,
            }) => {
                assert_eq!(*channel, 0);
                assert_eq!(*key, 60);
                assert_eq!(*velocity, 100);
            }
            other => panic!("expected NoteOn, got {other:?}"),
        }

        // Event 1: NoteOff at tick 480
        assert_eq!(events[1].tick, 480);
        match &events[1].event {
            TrackEvent::Midi(MidiEvent::NoteOff {
                channel,
                key,
                velocity,
            }) => {
                assert_eq!(*channel, 0);
                assert_eq!(*key, 60);
                assert_eq!(*velocity, 0);
            }
            other => panic!("expected NoteOff, got {other:?}"),
        }

        // Event 2: EndOfTrack at tick 480
        assert_eq!(events[2].tick, 480);
        match &events[2].event {
            TrackEvent::Meta(MetaEvent::EndOfTrack) => {}
            other => panic!("expected EndOfTrack, got {other:?}"),
        }
    }

    #[test]
    fn test_invalid_magic_returns_error() {
        let bad: &[u8] = b"BADH\x00\x00\x00\x06\x00\x00\x00\x01\x00\xF0";
        let result = MidiFile::from_bytes(bad);
        assert!(result.is_err(), "expected Err for bad magic");
    }

    #[test]
    fn test_nonexistent_file_returns_io_error() {
        let path = std::env::temp_dir().join("oxiaudio_midi_nonexistent_xyz_unique.mid");
        let result = MidiFile::from_path(&path);
        assert!(result.is_err(), "expected Err for non-existent file");
        // Must be an Io variant, not a Decode variant
        match result.unwrap_err() {
            OxiAudioError::Io(_) => {}
            other => panic!("expected OxiAudioError::Io, got {other:?}"),
        }
    }

    #[test]
    fn test_smf_format1_two_tracks() {
        // Build Format 1 with 2 tracks, each with only an EndOfTrack event
        let mut v: Vec<u8> = Vec::new();
        v.extend_from_slice(b"MThd");
        v.extend_from_slice(&6u32.to_be_bytes());
        v.extend_from_slice(&1u16.to_be_bytes()); // format 1
        v.extend_from_slice(&2u16.to_be_bytes()); // 2 tracks
        v.extend_from_slice(&480u16.to_be_bytes());
        for _ in 0..2 {
            v.extend_from_slice(b"MTrk");
            v.extend_from_slice(&4u32.to_be_bytes()); // 4 bytes: 00 FF 2F 00
            v.push(0x00); // delta=0
            v.push(0xFF); // meta
            v.push(0x2F); // EndOfTrack
            v.push(0x00); // length=0
        }
        let midi = MidiFile::from_bytes(&v).expect("parse format 1");
        assert_eq!(midi.format, SmfFormat::MultiTrack);
        assert_eq!(midi.tracks.len(), 2);
        for track in &midi.tracks {
            assert_eq!(track.events.len(), 1);
            assert!(matches!(
                &track.events[0].event,
                TrackEvent::Meta(MetaEvent::EndOfTrack)
            ));
        }
    }

    #[test]
    fn test_running_status() {
        // Format 0, 1 track, 2 NoteOn using running status for the second
        //   00 90 3C 64  — NoteOn, ch0, key60, vel100
        //   00    3E 64  — running status NoteOn, key62, vel100 (no re-emitted 0x90)
        //   00 FF 2F 00  — EndOfTrack
        // MTrk payload: 11 bytes
        let mut v: Vec<u8> = Vec::new();
        v.extend_from_slice(b"MThd");
        v.extend_from_slice(&6u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes()); // format 0
        v.extend_from_slice(&1u16.to_be_bytes());
        v.extend_from_slice(&480u16.to_be_bytes());
        v.extend_from_slice(b"MTrk");
        v.extend_from_slice(&11u32.to_be_bytes());
        v.push(0x00);
        v.push(0x90);
        v.push(0x3C);
        v.push(0x64); // NoteOn ch0 key60 vel100
        v.push(0x00);
        v.push(0x3E);
        v.push(0x64); // running: key62 vel100
        v.push(0x00);
        v.push(0xFF);
        v.push(0x2F);
        v.push(0x00); // EndOfTrack

        let midi = MidiFile::from_bytes(&v).expect("parse running status SMF");
        let events = &midi.tracks[0].events;
        assert_eq!(events.len(), 3);

        match &events[0].event {
            TrackEvent::Midi(MidiEvent::NoteOn { key, .. }) => assert_eq!(*key, 60),
            other => panic!("expected NoteOn(60), got {other:?}"),
        }
        match &events[1].event {
            TrackEvent::Midi(MidiEvent::NoteOn { key, .. }) => assert_eq!(*key, 62),
            other => panic!("expected NoteOn(62) via running status, got {other:?}"),
        }
    }

    #[test]
    fn test_tempo_meta_event() {
        // 500000 µs/qn = 120 BPM; 3 bytes big-endian: 0x07 0xA1 0x20
        let mut v: Vec<u8> = Vec::new();
        v.extend_from_slice(b"MThd");
        v.extend_from_slice(&6u32.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes());
        v.extend_from_slice(&1u16.to_be_bytes());
        v.extend_from_slice(&480u16.to_be_bytes());
        // MTrk: delta=0 meta Tempo(500000) + EndOfTrack
        // Tempo payload: 00 FF 51 03 07 A1 20 (7 bytes)
        // EndOfTrack:    00 FF 2F 00           (4 bytes)
        // total = 11 bytes
        v.extend_from_slice(b"MTrk");
        v.extend_from_slice(&11u32.to_be_bytes());
        v.push(0x00);
        v.push(0xFF);
        v.push(0x51);
        v.push(0x03);
        v.push(0x07);
        v.push(0xA1);
        v.push(0x20); // 500000
        v.push(0x00);
        v.push(0xFF);
        v.push(0x2F);
        v.push(0x00);

        let midi = MidiFile::from_bytes(&v).expect("parse tempo meta");
        let events = &midi.tracks[0].events;
        assert_eq!(events.len(), 2);
        match &events[0].event {
            TrackEvent::Meta(MetaEvent::Tempo(us)) => assert_eq!(*us, 500_000),
            other => panic!("expected Tempo(500000), got {other:?}"),
        }
    }
}
