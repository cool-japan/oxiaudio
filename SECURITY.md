# Security Policy

OxiAudio is a pure-Rust audio codec and DSP layer maintained by
**COOLJAPAN OU (Team Kitasan)**. Because it parses untrusted, attacker-controlled
audio files, we take memory-safety and robustness seriously.

## Supported Versions

Only the **latest published `0.x` release** receives security fixes. There is no
back-porting to older `0.x` lines; please upgrade to the newest release before
reporting an issue.

| Version        | Supported          |
| -------------- | ------------------ |
| latest `0.x`   | :white_check_mark: |
| older releases | :x:                |

## Reporting a Vulnerability

Please report suspected vulnerabilities **privately** so we can triage and ship a
fix before public disclosure. **Do not open a public GitHub issue for a security
problem.**

- Email: **info@kitasan.io**
- Include: affected version, a minimal reproducer (ideally a small malformed input
  file or a `&[u8]` corpus), the observed behavior (panic, abort, hang, OOM,
  incorrect output, memory-safety issue), and the expected behavior.

We aim to acknowledge reports promptly and will coordinate a disclosure timeline
with you. Credit is given to reporters unless anonymity is requested.

## Threat Model

The decoders in `oxiaudio-decode` are the primary attack surface: they consume
untrusted bytes from arbitrary files. Our robustness guarantees for decoding
malformed or adversarial input are:

- **No panics on untrusted input.** Malformed audio must return a typed
  [`oxiaudio_core::OxiAudioError`], never `panic!`, `unwrap`/`expect` failure,
  arithmetic overflow abort, or unbounded hang. This is enforced by the
  property-based fuzz harnesses in
  `crates/oxiaudio-integration-tests/tests/fuzz_decoders.rs`, which drive every
  byte-oriented decoder with random and magic-prefixed adversarial corpora.
- **Memory safety.** The crates build under `#![deny(unsafe_code)]`. The only
  exception is the opt-in, non-default `mmap` feature, whose single `unsafe`
  block (`memmap2::Mmap::map`) is documented in the source.
- **Pure Rust by default.** The default feature set pulls in no C/C++/Fortran
  code, eliminating an entire class of native-memory vulnerabilities. Optional
  FFI backends (e.g. LAME MP3 encoding) are feature-gated and off by default.

## Non-Goals

- Decoding correctness for *well-formed* but exotic streams is a functional
  concern, not a security one, unless it leads to a safety/robustness violation.
- Resource-exhaustion from inputs that are individually valid but supplied at
  scale (e.g. a legitimately huge file) is the caller's responsibility to bound.
  Decoders still reject headers whose declared sizes are inconsistent with the
  available data.
