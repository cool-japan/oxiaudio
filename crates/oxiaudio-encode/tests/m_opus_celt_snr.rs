//! RFC 6716 CELT **reconstruction-quality** and **bit-exactness** tests.
//!
//! These go beyond "the packet decodes without error":
//!
//! * `celt_final_range_matches_reference_low_rates` / `..._high_rates` are the
//!   canonical libopus conformance check — the encoder's and the reference
//!   decoder's `final_range` registers can only agree if both consumed
//!   byte-for-byte the same symbol sequence. Swept over the whole supported
//!   frame-size range (16 B to the RFC's 1275 B limit) and a corpus that
//!   includes the worst cases for filling a frame: full-scale white noise, an
//!   impulse train, and pure tones from 120 Hz to 19 kHz. They also assert the
//!   *physical* stream fit, which `final_range` cannot see (dropping a raw-bit
//!   byte never touches the range register).
//! * `celt_in_crate_verifier_reads_back_every_field` re-parses the frame with
//!   this crate's own RFC-structured decoder (`opus_celt_verify`) and checks the
//!   header fields and range against what the encoder intended, and
//!   `celt_encoder_stage_tells_match_verifier` compares the two sides' bit
//!   positions stage by stage so a desync localises instead of only failing at
//!   the end.
//! * The quality tests assert a **per-band level floor** (worst
//!   `|20·log10(E_out/E_in)|` over bands carrying real energy) and a
//!   time-domain SNR floor at the fixed 540-sample CELT reconstruction delay,
//!   on speech-like, music-like, tonal and noise fixtures.

use opus_decoder::OpusDecoder;
use oxiaudio_encode::opus_celt::{
    encode_celt_frame_conformant_ranged, encode_celt_frame_conformant_sized,
    encode_celt_frame_conformant_traced, DEFAULT_CELT_FRAME_BYTES, MAX_CELT_FRAME_BYTES,
    MIN_CELT_FRAME_BYTES,
};
use oxiaudio_encode::opus_celt_tables::EBAND_5MS;
use oxiaudio_encode::opus_celt_verify::parse_celt_frame;
use oxiaudio_encode::opus_mdct::celt_mdct_960_overlap;

const FRAME: usize = 960;
/// Number of 20 ms frames per fixture.
const NF: usize = 10;
/// Fixed CELT reconstruction delay: the lapped analysis block for frame `t`
/// is centred on `t·960`, so decoded sample `j` corresponds to input `j − 540`.
const CELT_DELAY: usize = 540;

// ── Fixtures ──────────────────────────────────────────────────────────────────

fn tone(freq: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / 48_000.0).sin() * 0.5)
        .collect()
}

/// Speech-like: three harmonics under a slow amplitude envelope.
fn speech_like(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 3.0 * t).sin();
            env * 0.4
                * ((2.0 * std::f32::consts::PI * 180.0 * t).sin()
                    + 0.6 * (2.0 * std::f32::consts::PI * 360.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 900.0 * t).sin())
                / 1.9
        })
        .collect()
}

/// Music-like: a three-note chord plus a low-level broadband component.
fn music_like(n: usize) -> Vec<f32> {
    let mut st = 0xDEAD_BEEFu32;
    (0..n)
        .map(|i| {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((st >> 8) as f32 / (1 << 24) as f32) - 0.5;
            let t = i as f32 / 48_000.0;
            0.25 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 277.2 * t).sin()
                + 0.15 * (2.0 * std::f32::consts::PI * 330.0 * t).sin()
                + 0.03 * noise
        })
        .collect()
}

fn white_noise(n: usize, seed: u32, amp: f32) -> Vec<f32> {
    let mut st = seed;
    (0..n)
        .map(|_| {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((st >> 8) as f32 / (1 << 24) as f32) - 0.5) * amp
        })
        .collect()
}

/// A sparse impulse train: near-zero everywhere with a full-scale spike every
/// 5 ms. Maximally flat in the frequency domain, so every band demands pulses.
fn impulse_train(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| if i % 240 == 0 { 0.95 } else { -0.01 })
        .collect()
}

/// Low-rate half of the bit-exactness sweep: dense (step 4) over the region
/// where the pre-0.2.1 encoder desynchronised — the `lm == -1` pulse-cache bug
/// first bites around 90–130 bytes — plus every size a caller can reach through
/// the public API at low rates.
fn sweep_sizes_low() -> Vec<usize> {
    let mut sizes: Vec<usize> = (MIN_CELT_FRAME_BYTES..=160).step_by(7).collect();
    sizes.extend([
        MIN_CELT_FRAME_BYTES,
        DEFAULT_CELT_FRAME_BYTES,
        80,
        85,
        90,
        123,
        128,
        160,
    ]);
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// High-rate half of the sweep: sparse up to the RFC's 1275-byte frame limit.
fn sweep_sizes_high() -> Vec<usize> {
    let mut sizes: Vec<usize> = (161..=MAX_CELT_FRAME_BYTES).step_by(83).collect();
    // Payloads that put the *packet* (payload + TOC) on an OGG lacing boundary:
    // a segment table entry saturates at 255, so a 255-byte packet is one
    // segment and a 256-byte packet is two (255 + 1).
    sizes.extend([254, 255, 256, 509, 510, 511]);
    sizes.push(MAX_CELT_FRAME_BYTES);
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// The fixture corpus both halves of the sweep run against.
fn sweep_fixtures() -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("tone_120", tone(120.0, FRAME * 2)),
        ("tone_1000", tone(1000.0, FRAME * 2)),
        ("tone_8000", tone(8000.0, FRAME * 2)),
        ("tone_19000", tone(19_000.0, FRAME * 2)),
        ("speech", speech_like(FRAME * 2)),
        ("music", music_like(FRAME * 2)),
        ("noise_full_scale", white_noise(FRAME * 2, 0x2222_3333, 0.8)),
        ("noise_tiny", white_noise(FRAME * 2, 0x5151_7171, 0.02)),
        ("impulse", impulse_train(FRAME * 2)),
        ("silence", vec![0.0f32; FRAME * 2]),
    ]
}

/// Encode every fixture at every listed size and assert entropy-level exactness
/// against the reference decoder.
///
/// Two frames per fixture: frame 0 (no MDCT history — the stream-start path)
/// and frame 1 (real history). `encode_celt_frame_conformant_traced` is used
/// rather than the shrink-retrying public entry point so the assertion is about
/// the *requested* size, not about whatever size a retry settled on.
fn assert_bit_exact_over(sizes: &[usize]) {
    for &bytes in sizes {
        for (name, sig) in &sweep_fixtures() {
            for f in 0..2usize {
                let prev: &[f32] = if f == 0 { &[] } else { &sig[..FRAME] };
                let (packet, trace) = encode_celt_frame_conformant_traced(
                    prev,
                    &sig[f * FRAME..(f + 1) * FRAME],
                    bytes,
                );
                assert_eq!(
                    trace.payload_len, bytes,
                    "{name} @ {bytes} B frame {f}: payload must be exactly the requested size"
                );
                // `final_range` is a property of the *range* coder only, so it
                // cannot see a dropped raw-bit byte: a frame whose two stream
                // halves collided still reports a matching range while silently
                // losing fine-energy and sign bits. Assert the physical fit too.
                assert!(
                    trace.fits,
                    "{name} @ {bytes} B frame {f}: the range-coded and raw-bit halves \
                     collided, so bytes were dropped from the packet"
                );
                let mut dec = OpusDecoder::new(48_000, 1).expect("decoder init");
                let mut out = [0.0f32; FRAME];
                let n = dec
                    .decode_float(&packet, &mut out, false)
                    .expect("every packet must decode");
                assert_eq!(n, FRAME, "{name} @ {bytes} B frame {f}: 960 samples");
                assert_eq!(
                    dec.final_range(),
                    trace.final_range,
                    "{name} @ {bytes} B frame {f}: the encoder's final_range differs from the \
                     reference decoder's — the bitstreams desynchronised"
                );
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Encode a whole fixture frame by frame (carrying MDCT history) and decode it
/// with the reference decoder. Returns `(decoded, final_range_mismatches)`.
fn round_trip(signal: &[f32], bytes: usize) -> (Vec<f32>, usize) {
    let mut dec = OpusDecoder::new(48_000, 1).expect("decoder init");
    let mut out = Vec::with_capacity(signal.len());
    let mut mismatches = 0usize;
    let frames = signal.len() / FRAME;
    for f in 0..frames {
        let prev: &[f32] = if f == 0 {
            &[]
        } else {
            &signal[(f - 1) * FRAME..f * FRAME]
        };
        let (packet, enc_range) =
            encode_celt_frame_conformant_ranged(prev, &signal[f * FRAME..(f + 1) * FRAME], bytes);
        let mut frame_out = [0.0f32; FRAME];
        let n = dec
            .decode_float(&packet, &mut frame_out, false)
            .expect("decode_float");
        assert_eq!(n, FRAME, "every packet must decode to 960 samples");
        if dec.final_range() != enc_range {
            mismatches += 1;
        }
        out.extend_from_slice(&frame_out);
    }
    (out, mismatches)
}

/// Per-band L2 energies of one lapped analysis block.
fn band_energies(prev: &[f32], cur: &[f32]) -> Vec<f64> {
    let spec = celt_mdct_960_overlap(prev, cur);
    (0..21)
        .map(|i| {
            let lo = EBAND_5MS[i] as usize * 8;
            let hi = (EBAND_5MS[i + 1] as usize * 8).min(spec.len());
            spec[lo..hi].iter().map(|&x| (x as f64) * (x as f64)).sum()
        })
        .collect()
}

/// Worst per-band level error in dB over bands within `floor_db` of the loudest.
fn worst_band_level_error_db(input: &[f32], output: &[f32], floor_db: f64) -> f64 {
    let mut acc_in = [0.0f64; 21];
    let mut acc_out = [0.0f64; 21];
    let frames = input.len() / FRAME;
    for f in 3..frames - 1 {
        let ein = band_energies(
            &input[(f - 1) * FRAME..f * FRAME],
            &input[f * FRAME..(f + 1) * FRAME],
        );
        let eout = band_energies(
            &output[(f - 1) * FRAME..f * FRAME],
            &output[f * FRAME..(f + 1) * FRAME],
        );
        for i in 0..21 {
            acc_in[i] += ein[i];
            acc_out[i] += eout[i];
        }
    }
    let peak = acc_in.iter().cloned().fold(0.0f64, f64::max).max(1e-30);
    let mut worst = 0.0f64;
    for i in 0..21 {
        if 10.0 * (acc_in[i] / peak).log10() < floor_db {
            continue;
        }
        let err = 10.0 * (acc_out[i].max(1e-30) / acc_in[i].max(1e-30)).log10();
        worst = worst.max(err.abs());
    }
    worst
}

/// Time-domain SNR (dB) at the fixed CELT reconstruction delay.
fn snr_db_at_fixed_delay(input: &[f32], output: &[f32], start_frame: usize, frames: usize) -> f64 {
    let a = &input[start_frame * FRAME..(start_frame + frames) * FRAME];
    let b = &output[start_frame * FRAME + CELT_DELAY..(start_frame + frames) * FRAME + CELT_DELAY];
    let sig: f64 = a.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let err: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| ((x - y) as f64) * ((x - y) as f64))
        .sum();
    10.0 * (sig / err.max(1e-30)).log10()
}

// ── Bit-exactness ─────────────────────────────────────────────────────────────

/// The canonical libopus conformance check over low frame sizes — the region
/// that used to be the *only* verified one (the pre-0.2.1 cap was 80 bytes).
#[test]
fn celt_final_range_matches_reference_low_rates() {
    assert_bit_exact_over(&sweep_sizes_low());
}

/// The same check above the old 80-byte cap, up to the RFC's 1275-byte frame
/// limit (510 kbps mono). Split from the low-rate half so both run in parallel.
#[test]
fn celt_final_range_matches_reference_high_rates() {
    assert_bit_exact_over(&sweep_sizes_high());
}

/// The encoder's per-stage bit positions must equal the in-crate verifier's.
///
/// `final_range` equality alone says "the two sides agree at the end"; when it
/// fails it says nothing about *where*. Recording the same named stages on both
/// sides turns a desync into a pinpointed stage, and pins that the two
/// implementations walk the frame in the same order with the same conditionals.
#[test]
fn celt_encoder_stage_tells_match_verifier() {
    let fixtures: Vec<(&str, Vec<f32>)> = vec![
        ("music", music_like(FRAME * 2)),
        ("noise", white_noise(FRAME * 2, 0x2222_3333, 0.8)),
        ("tone_1000", tone(1000.0, FRAME * 2)),
        ("silence", vec![0.0f32; FRAME * 2]),
    ];
    for bytes in [
        MIN_CELT_FRAME_BYTES,
        64,
        123,
        256,
        640,
        MAX_CELT_FRAME_BYTES,
    ] {
        for (name, sig) in &fixtures {
            let (packet, trace) =
                encode_celt_frame_conformant_traced(&sig[..FRAME], &sig[FRAME..], bytes);
            let parsed = parse_celt_frame(&packet[1..], 0, 3);
            assert!(!parsed.error, "{name} @ {bytes} B: verifier overran");
            assert_eq!(
                trace.stage_tells, parsed.stage_tells,
                "{name} @ {bytes} B: encoder and verifier disagree on per-stage bit positions"
            );
            assert_eq!(
                trace.pvq_leaves, parsed.pvq_leaves,
                "{name} @ {bytes} B: PVQ leaf count must match"
            );
            assert_eq!(
                trace.theta_symbols, parsed.theta_symbols,
                "{name} @ {bytes} B: split-angle symbol count must match"
            );
            assert_eq!(
                trace.final_range, parsed.final_range,
                "{name} @ {bytes} B: final range must match"
            );
        }
    }
}

/// High-rate frames must actually reach the four-deep split (`lm == -1`).
///
/// That partition is the one whose pulse-cache row was mis-indexed before
/// oxiaudio 0.2.1 — `celt_bits2pulses` clamped `lm` to 0 before the `+1`, so a
/// `lm == -1` leaf read row 1 instead of row 0 and got a different pulse count
/// than the decoder derived. It only occurs at rates above ≈90 bytes/frame,
/// which is exactly why the old 80-byte cap hid it. If a future change stops
/// reaching `lm == -1`, the sweep above would still pass while silently losing
/// the coverage that catches this class of bug.
#[test]
fn celt_high_rate_frames_exercise_deep_splits() {
    let sig = white_noise(FRAME * 2, 0x2222_3333, 0.8);
    let (_p_low, low) = encode_celt_frame_conformant_traced(&sig[..FRAME], &sig[FRAME..], 32);
    let (_p_high, high) =
        encode_celt_frame_conformant_traced(&sig[..FRAME], &sig[FRAME..], MAX_CELT_FRAME_BYTES);
    assert!(
        low.min_partition_lm >= 0,
        "a 32-byte frame should not split four deep, got lm={}",
        low.min_partition_lm
    );
    assert_eq!(
        high.min_partition_lm, -1,
        "a {MAX_CELT_FRAME_BYTES}-byte frame must reach the lm = -1 partition"
    );
    assert!(
        high.max_leaf_pulses > low.max_leaf_pulses,
        "a larger frame must fund more pulses per leaf ({} vs {})",
        high.max_leaf_pulses,
        low.max_leaf_pulses
    );
}

/// The in-crate RFC-structured decoder must read every header field back with
/// the value the encoder intended, and end on the same range register.
#[test]
fn celt_in_crate_verifier_reads_back_every_field() {
    let sig = music_like(FRAME * 3);
    for bytes in [
        MIN_CELT_FRAME_BYTES,
        DEFAULT_CELT_FRAME_BYTES,
        MAX_CELT_FRAME_BYTES,
    ] {
        let (packet, enc_range) =
            encode_celt_frame_conformant_ranged(&sig[..FRAME], &sig[FRAME..2 * FRAME], bytes);
        assert_eq!(packet[0], 0xF8, "TOC must be CELT-only FB 20 ms mono");
        let parsed = parse_celt_frame(&packet[1..], 0, 3);
        assert!(
            !parsed.error,
            "verifier must not run off the end @ {bytes} B"
        );
        assert!(!parsed.silence, "encoder never sets the silence flag");
        assert!(!parsed.post_filter, "encoder never enables the post-filter");
        assert!(
            !parsed.transient,
            "encoder always codes non-transient frames"
        );
        assert!(parsed.intra, "encoder always codes intra energy");
        assert_eq!(parsed.spread, 2, "spread must be SPREAD_NORMAL");
        assert_eq!(parsed.alloc_trim, 5, "trim must be neutral");
        assert!(
            parsed.tf_res.iter().all(|&t| t == 0),
            "TF decisions must all be neutral"
        );
        assert_eq!(
            parsed.final_range, enc_range,
            "in-crate verifier range must equal the encoder's @ {bytes} B"
        );
        if bytes >= DEFAULT_CELT_FRAME_BYTES {
            assert!(
                parsed.theta_symbols > 0,
                "a music fixture at {bytes} B must exercise real split-band coding"
            );
        }
    }
}

/// Split-band coding must actually vary `itheta`.
///
/// Two broadband fixtures that differ only in how energy divides *inside* the
/// wide high bands must produce different bitstreams, and the encoder must emit
/// real theta symbols for them. Before oxiaudio 0.2.1 `itheta` was pinned to 0,
/// so the upper half of every split band decoded as zeros and the two signals
/// coded identically above the split point.
#[test]
fn celt_split_bands_code_real_theta() {
    let base = music_like(FRAME * 2);
    let tilt_lo: Vec<f32> = base
        .iter()
        .zip(tone(16_000.0, FRAME * 2))
        .map(|(&a, b)| a + 0.15 * b)
        .collect();
    let tilt_hi: Vec<f32> = base
        .iter()
        .zip(tone(19_000.0, FRAME * 2))
        .map(|(&a, b)| a + 0.15 * b)
        .collect();

    let (p_lo, _) = encode_celt_frame_conformant_ranged(
        &tilt_lo[..FRAME],
        &tilt_lo[FRAME..],
        MAX_CELT_FRAME_BYTES,
    );
    let (p_hi, _) = encode_celt_frame_conformant_ranged(
        &tilt_hi[..FRAME],
        &tilt_hi[FRAME..],
        MAX_CELT_FRAME_BYTES,
    );
    assert_ne!(
        p_lo, p_hi,
        "shifting energy inside a split band must change the bitstream"
    );
    let a = parse_celt_frame(&p_lo[1..], 0, 3);
    let b = parse_celt_frame(&p_hi[1..], 0, 3);
    assert!(
        a.theta_symbols > 0 && b.theta_symbols > 0,
        "both frames must carry split-band theta symbols ({} / {})",
        a.theta_symbols,
        b.theta_symbols
    );
}

/// Fine-energy bits must carry information: the same allocation with different
/// coarse residuals must produce different fine values.
#[test]
fn celt_fine_energy_bits_are_not_constant() {
    let mut seen = std::collections::HashSet::new();
    for amp in [0.05f32, 0.11, 0.23, 0.47] {
        let sig: Vec<f32> = tone(700.0, FRAME * 2).iter().map(|&x| x * amp).collect();
        let (packet, _) = encode_celt_frame_conformant_ranged(&sig[..FRAME], &sig[FRAME..], 64);
        let parsed = parse_celt_frame(&packet[1..], 0, 3);
        assert!(
            parsed.fine_quant.iter().any(|&e| e > 0),
            "the allocator must fund fine-energy bits"
        );
        seen.insert(parsed.fine_bits.clone());
    }
    assert!(
        seen.len() > 1,
        "fine-energy bits must depend on the coarse residual, not be constant"
    );
}

// ── Reconstruction quality ────────────────────────────────────────────────────

/// Per-band level accuracy: for every band carrying energy within 40 dB of the
/// loudest band, the decoded level must be within the stated floor.
///
/// This is the stable "SNR floor per band" metric — it is insensitive to the
/// reconstruction delay and to PVQ phase error, and it directly measures that
/// the coarse+fine energy envelope survives the round trip.
#[test]
fn celt_per_band_level_accuracy() {
    let cases: Vec<(&str, Vec<f32>, f64)> = vec![
        ("music", music_like(FRAME * NF), 6.0),
        ("noise", white_noise(FRAME * NF, 0x1234_5678, 0.4), 4.0),
        ("speech", speech_like(FRAME * NF), 8.0),
        ("tone_1000", tone(1000.0, FRAME * NF), 8.0),
    ];
    for (name, sig, floor) in cases {
        let (out, mismatches) = round_trip(&sig, DEFAULT_CELT_FRAME_BYTES);
        assert_eq!(mismatches, 0, "{name}: bitstream desync");
        let worst = worst_band_level_error_db(&sig, &out, -40.0);
        assert!(
            worst < floor,
            "{name}: worst per-band level error {worst:.2} dB exceeds the {floor:.1} dB floor"
        );
    }
}

/// Time-domain SNR at the fixed reconstruction delay, on signals whose energy
/// the neutral (non-dynalloc) allocation actually funds.
///
/// The floors are set well below measured values so the test pins a real
/// property rather than a specific quantiser tuning.
#[test]
fn celt_time_domain_snr_floor() {
    let music = music_like(FRAME * NF);
    let (out, mismatches) = round_trip(&music, DEFAULT_CELT_FRAME_BYTES);
    assert_eq!(mismatches, 0, "music: bitstream desync");
    let snr = snr_db_at_fixed_delay(&music, &out, 4, 2);
    assert!(
        snr > 8.0,
        "music-like SNR {snr:.2} dB at the fixed {CELT_DELAY}-sample delay must exceed 8 dB"
    );
}

/// Overall level must be preserved: the decoded RMS must track the input RMS.
///
/// This is what the missing CELT pre-emphasis + analysis gain used to break —
/// before the fix the decode came back ~11–19× quiet and heavily low-passed.
#[test]
fn celt_output_level_tracks_input() {
    for (name, sig) in [
        ("music", music_like(FRAME * NF)),
        ("noise", white_noise(FRAME * NF, 0x9999_1111, 0.4)),
        ("tone_300", tone(300.0, FRAME * NF)),
    ] {
        let (out, _) = round_trip(&sig, DEFAULT_CELT_FRAME_BYTES);
        let rms = |x: &[f32]| -> f64 {
            (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64).sqrt()
        };
        let ri = rms(&sig[3 * FRAME..(NF - 1) * FRAME]);
        let ro = rms(&out[3 * FRAME..(NF - 1) * FRAME]);
        let db = 20.0 * (ro / ri.max(1e-30)).log10();
        assert!(
            db.abs() < 3.0,
            "{name}: decoded level is {db:.2} dB off the input (must be within ±3 dB)"
        );
    }
}

/// A silent frame must stay silent and still decode.
#[test]
fn celt_silence_decodes_to_near_silence() {
    let sig = vec![0.0f32; FRAME * 4];
    let (out, mismatches) = round_trip(&sig, DEFAULT_CELT_FRAME_BYTES);
    assert_eq!(mismatches, 0, "silence: bitstream desync");
    let peak = out.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(peak < 0.05, "silence must decode near-silent, peak {peak}");
}

/// Larger frames must not be *worse*: the encoder clamps to the verified range
/// and the level stays correct at both ends of it.
#[test]
fn celt_frame_size_range_is_usable() {
    let sig = music_like(FRAME * NF);
    for bytes in [
        MIN_CELT_FRAME_BYTES,
        DEFAULT_CELT_FRAME_BYTES,
        320,
        MAX_CELT_FRAME_BYTES,
    ] {
        let (out, mismatches) = round_trip(&sig, bytes);
        assert_eq!(mismatches, 0, "{bytes} B: bitstream desync");
        let worst = worst_band_level_error_db(&sig, &out, -40.0);
        assert!(worst < 6.0, "{bytes} B: per-band level error {worst:.2} dB");
    }
    // Requests above the RFC's 1275-byte frame limit are clamped, not silently
    // corrupted. The packet carries one extra TOC byte on top of the payload.
    let packet = encode_celt_frame_conformant_sized(&sig[..FRAME], &sig[FRAME..2 * FRAME], 4000);
    assert_eq!(
        packet.len(),
        MAX_CELT_FRAME_BYTES + 1,
        "oversized requests must clamp to MAX_CELT_FRAME_BYTES + TOC, got {}",
        packet.len()
    );
}

/// Raising the frame-size cap must buy real quality, not just a bigger packet.
///
/// Until oxiaudio 0.2.1 the encoder clamped every request to 80 bytes/frame
/// (≈32 kbps) because bit-exactness had not been established above that. With
/// the `lm == -1` pulse-cache bug fixed, the cap is the RFC's own 1275-byte
/// frame limit, so this test pins that the extra rate actually reaches the
/// reconstruction: time-domain SNR at 320 B/frame (128 kbps) must beat the old
/// ceiling by a wide margin on both a tonal and a noise-like fixture.
#[test]
fn celt_higher_bitrate_improves_snr() {
    for (name, sig, floor_gain_db) in [
        ("music", music_like(FRAME * NF), 6.0f64),
        ("noise", white_noise(FRAME * NF, 0x1234_5678, 0.4), 6.0),
    ] {
        let (out_low, m_low) = round_trip(&sig, 80);
        assert_eq!(m_low, 0, "{name} @ 80 B: bitstream desync");
        let (out_high, m_high) = round_trip(&sig, 320);
        assert_eq!(m_high, 0, "{name} @ 320 B: bitstream desync");
        let snr_low = snr_db_at_fixed_delay(&sig, &out_low, 4, 4);
        let snr_high = snr_db_at_fixed_delay(&sig, &out_high, 4, 4);
        assert!(
            snr_high - snr_low > floor_gain_db,
            "{name}: 320 B/frame SNR {snr_high:.2} dB must beat the old 80 B ceiling \
             ({snr_low:.2} dB) by more than {floor_gain_db:.1} dB"
        );
    }
}

// ── End-to-end through the public OGG entry points ────────────────────────────

/// Decode an OGG Opus stream produced by this crate with the reference decoder.
///
/// Returns the concatenated PCM plus the number of audio packets seen.
fn decode_ogg_opus(bytes: &[u8]) -> (Vec<f32>, usize) {
    let mut dec = OpusDecoder::new(48_000, 1).expect("decoder init");
    let mut out = Vec::new();
    let mut packets = 0usize;
    let mut cursor = std::io::Cursor::new(bytes);
    let mut reader = oxiaudio_decode::ogg_reader::OggReader::new(&mut cursor);
    while let Some(packet) = reader.read_packet().expect("ogg packet") {
        // Skip the two header packets and any zero-length EOS filler page.
        if packet.is_empty() || packet.starts_with(b"OpusHead") || packet.starts_with(b"OpusTags") {
            continue;
        }
        let mut frame = vec![0.0f32; 960];
        let n = dec
            .decode_float(&packet, &mut frame, false)
            .expect("stream packet must decode");
        out.extend_from_slice(&frame[..n]);
        packets += 1;
    }
    (out, packets)
}

/// The default public entry point must produce an OGG stream a standard decoder
/// reads back at the right level — including across SILK↔CELT mode transitions,
/// which `select_conformant_mode` inserts whenever a passage goes quiet.
///
/// This is the flagship path: `encode_opus` → OGG → reference decoder.
#[test]
fn encode_opus_ogg_stream_round_trips_with_mode_transitions() {
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};

    // Three silent frames (routed to SILK), then music (routed to CELT).
    let mut samples = vec![0.0f32; FRAME * 3];
    samples.extend_from_slice(&music_like(FRAME * 7));

    let buf = AudioBuffer {
        samples: samples.clone(),
        sample_rate: 48_000,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };
    let mut ogg = Vec::new();
    oxiaudio_encode::encode_opus(&buf, &mut ogg, 32).expect("encode_opus");
    assert_eq!(&ogg[..4], b"OggS", "output must be an OGG stream");

    let (decoded, packets) = decode_ogg_opus(&ogg);
    assert_eq!(packets, 10, "one audio packet per 20 ms frame");
    assert_eq!(decoded.len(), FRAME * 10);

    // Level tracking over the CELT-coded (music) region, skipping the two frames
    // straddling the SILK→CELT transition where the decoder's overlap tail still
    // holds SILK output that the encoder's history model does not predict.
    let rms = |x: &[f32]| -> f64 {
        (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64).sqrt()
    };
    let ri = rms(&samples[5 * FRAME..9 * FRAME]);
    let ro = rms(&decoded[5 * FRAME..9 * FRAME]);
    let db = 20.0 * (ro / ri.max(1e-30)).log10();
    assert!(
        db.abs() < 4.0,
        "encode_opus stream level is {db:.2} dB off the input over the CELT region"
    );

    // The silent lead-in must stay quiet through the SILK frames.
    let lead_peak = decoded[..2 * FRAME]
        .iter()
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(
        lead_peak < 0.05,
        "silent lead-in must decode quiet, peak {lead_peak}"
    );
}

/// The **public** bitrate API must actually reach the rates the raised
/// frame-size cap unlocked, end to end through OGG and the reference decoder.
///
/// This is the gap the per-frame sweeps do not cover: `MAX_CELT_FRAME_BYTES`
/// could be 1275 while `celt_frame_bytes_for_bitrate`, `encode_opus_auto` or
/// `OpusStreamEncoder` still clamped somewhere. Requests of 64/128/256 kbps must
/// produce packets of the corresponding size (`kbps · 2.5` bytes payload plus a
/// TOC byte), decode without error, and grow monotonically with the request.
#[test]
fn public_bitrate_api_reaches_high_rates_end_to_end() {
    use oxiaudio_core::{AudioBuffer, ChannelLayout, SampleFormat};
    use oxiaudio_encode::opus_celt::celt_frame_bytes_for_bitrate;

    // Pure conversion first: the old cap silently pinned every request ≥32 kbps
    // to 80 bytes.
    assert_eq!(celt_frame_bytes_for_bitrate(32), 80);
    assert_eq!(celt_frame_bytes_for_bitrate(64), 160);
    assert_eq!(celt_frame_bytes_for_bitrate(128), 320);
    assert_eq!(celt_frame_bytes_for_bitrate(256), 640);
    assert_eq!(
        celt_frame_bytes_for_bitrate(510),
        MAX_CELT_FRAME_BYTES,
        "510 kbps is exactly the RFC frame limit"
    );
    assert_eq!(
        celt_frame_bytes_for_bitrate(2000),
        MAX_CELT_FRAME_BYTES,
        "requests beyond the RFC limit clamp, they do not wrap"
    );

    let samples = music_like(FRAME * 6);
    let buf = AudioBuffer {
        samples: samples.clone(),
        sample_rate: 48_000,
        channels: ChannelLayout::Mono,
        format: SampleFormat::F32,
    };

    let mut prev_len = 0usize;
    for kbps in [32u32, 64, 128, 256, 510] {
        let expected_payload = celt_frame_bytes_for_bitrate(kbps);

        // `encode_opus_auto` — the explicit full-path entry point.
        let mut ogg = Vec::new();
        oxiaudio_encode::encode_opus_auto(&buf, &mut ogg, kbps).expect("encode_opus_auto");
        let (decoded, packets) = decode_ogg_opus(&ogg);
        assert_eq!(packets, 6, "{kbps} kbps: one packet per 20 ms frame");
        assert_eq!(decoded.len(), FRAME * 6);
        assert!(
            ogg.len() > prev_len,
            "{kbps} kbps must produce a larger stream than the previous rate \
             ({} vs {prev_len} bytes)",
            ogg.len()
        );
        prev_len = ogg.len();

        // Per-frame packet size must equal the requested CBR payload + TOC.
        let (packet, trace) = encode_celt_frame_conformant_traced(
            &samples[FRAME..2 * FRAME],
            &samples[2 * FRAME..3 * FRAME],
            expected_payload,
        );
        assert_eq!(
            packet.len(),
            expected_payload + 1,
            "{kbps} kbps: packet must be the CBR payload plus one TOC byte"
        );
        assert!(trace.fits, "{kbps} kbps: stream halves must not collide");

        // `OpusStreamEncoder::with_bitrate` must honour the same rate.
        let mut stream_ogg = Vec::new();
        {
            let mut enc =
                oxiaudio_encode::OpusStreamEncoder::with_bitrate(&mut stream_ogg, 1, 0x5150, kbps)
                    .expect("with_bitrate");
            for f in 0..6 {
                enc.encode_frame(&samples[f * FRAME..(f + 1) * FRAME])
                    .expect("encode_frame");
            }
            enc.finalize().expect("finalize");
        }
        let (stream_decoded, stream_packets) = decode_ogg_opus(&stream_ogg);
        assert_eq!(stream_packets, 6, "{kbps} kbps: streaming packet count");
        assert_eq!(stream_decoded.len(), FRAME * 6);
        // The streaming encoder is CELT-only, so every packet is exactly the
        // CBR payload plus a TOC byte; 6 of them plus OGG framing overhead.
        assert!(
            stream_ogg.len() >= 6 * (expected_payload + 1),
            "{kbps} kbps: streaming output ({} B) must carry 6 × {} B payloads",
            stream_ogg.len(),
            expected_payload + 1
        );
    }
}

/// `OpusStreamEncoder` must carry MDCT history across `encode_frame` calls and
/// honour `with_bitrate`.
#[test]
fn opus_stream_encoder_carries_history_and_bitrate() {
    use oxiaudio_encode::OpusStreamEncoder;

    let sig = music_like(FRAME * 8);
    let mut ogg = Vec::new();
    {
        let mut enc = OpusStreamEncoder::with_bitrate(&mut ogg, 1, 0x4321, 32).expect("new");
        for f in 0..8 {
            enc.encode_frame(&sig[f * FRAME..(f + 1) * FRAME])
                .expect("encode_frame");
        }
        enc.finalize().expect("finalize");
    }
    let (decoded, packets) = decode_ogg_opus(&ogg);
    assert_eq!(packets, 8, "one audio packet per encode_frame call");

    let rms = |x: &[f32]| -> f64 {
        (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64).sqrt()
    };
    let db = 20.0 * (rms(&decoded[3 * FRAME..7 * FRAME]) / rms(&sig[3 * FRAME..7 * FRAME])).log10();
    assert!(
        db.abs() < 4.0,
        "streaming encoder level is {db:.2} dB off the input"
    );

    // A lower bitrate must produce a materially smaller stream.
    let mut small = Vec::new();
    {
        let mut enc = OpusStreamEncoder::with_bitrate(&mut small, 1, 0x4321, 8).expect("new");
        for f in 0..8 {
            enc.encode_frame(&sig[f * FRAME..(f + 1) * FRAME])
                .expect("encode_frame");
        }
        enc.finalize().expect("finalize");
    }
    assert!(
        small.len() < ogg.len(),
        "8 kbps stream ({}) must be smaller than 32 kbps ({})",
        small.len(),
        ogg.len()
    );
}

/// The hybrid packet's CELT layer bit budget must equal the budget a decoder
/// derives from the emitted packet length, and the whole two-layer stream must
/// be entropy-exact against the reference decoder.
///
/// Before oxiaudio 0.2.1 this was two separate defects:
///
/// 1. `encode_celt_hybrid_layer_into` hardcoded a 512-bit (64-byte) budget while
///    `encode_hybrid_frame_conformant` assembled the packet with a
///    variable-length `finish()`, so the decoder's `total_bits = len·8` only
///    coincidentally matched.
/// 2. The encoder never wrote the hybrid **redundancy flag** (`logp = 12`) that
///    a decoder reads between the SILK and CELT layers, so every hybrid packet
///    was off by one bit from the CELT layer onward. It still "decoded to 960
///    samples" — which is exactly why the previous version of this test passed.
///
/// The bar here is therefore `final_range` equality, not decodability.
#[test]
fn hybrid_packet_layer_budget_is_consistent() {
    use oxiaudio_encode::opus_hybrid_conform::{
        encode_hybrid_frame_conformant_sized, DEFAULT_HYBRID_FRAME_BYTES, MAX_HYBRID_FRAME_BYTES,
        MIN_HYBRID_FRAME_BYTES,
    };

    let fixtures: Vec<(&str, Vec<f32>)> = vec![
        ("music", music_like(FRAME)),
        ("noise", white_noise(FRAME, 0x2222_3333, 0.8)),
        ("tone_12000", tone(12_000.0, FRAME)),
        ("silence", vec![0.0f32; FRAME]),
    ];
    let sizes = [
        MIN_HYBRID_FRAME_BYTES,
        DEFAULT_HYBRID_FRAME_BYTES,
        128,
        320,
        MAX_HYBRID_FRAME_BYTES,
    ];
    for bytes in sizes {
        for (name, sig) in &fixtures {
            let (packet, enc_range) = encode_hybrid_frame_conformant_sized(sig, 1, bytes);
            assert_eq!(
                packet[0], 0x78,
                "TOC must be config 15 (Hybrid FB 20 ms mono)"
            );
            assert_eq!(
                packet.len() - 1,
                bytes,
                "{name} @ {bytes} B: the emitted payload must be exactly the CBR target, \
                 otherwise the decoder's total_bits differs from the CELT layer's budget"
            );

            let mut dec = OpusDecoder::new(48_000, 1).expect("decoder init");
            let mut out = vec![0.0f32; FRAME];
            let n = dec
                .decode_float(&packet, &mut out, false)
                .expect("hybrid packet must decode");
            assert_eq!(n, FRAME, "{name} @ {bytes} B: must decode to 960 samples");
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{name} @ {bytes} B: hybrid decode must be finite"
            );
            assert_eq!(
                dec.final_range(),
                enc_range,
                "{name} @ {bytes} B: hybrid final_range must match the reference decoder's"
            );
        }
    }

    // Oversized requests clamp to the RFC frame limit rather than being emitted.
    let (packet, _) = encode_hybrid_frame_conformant_sized(&music_like(FRAME), 1, 9_000);
    assert_eq!(
        packet.len(),
        MAX_HYBRID_FRAME_BYTES + 1,
        "oversized hybrid requests must clamp to MAX_HYBRID_FRAME_BYTES + TOC"
    );
}
