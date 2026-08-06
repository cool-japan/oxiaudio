# Contributing to OxiAudio

Thanks for your interest in OxiAudio, the COOLJAPAN pure-Rust audio codec + DSP
layer maintained by **COOLJAPAN OU (Team Kitasan)**. This document describes the
engineering rules that every change must follow. They are enforced in review and
by CI, so please check them locally before opening a pull request.

## Contact & Triage

- General questions and contributions: open a GitHub issue or pull request.
- **Security issues: do not open a public issue.** Follow [`SECURITY.md`](./SECURITY.md)
  and email **info@kitasan.io** for private triage.

## Supported Versions

Development targets the **latest `0.x` release only**. New work lands on the
current version branch; older `0.x` lines are not maintained.

## Getting Started

```bash
# Build the whole workspace (default, 100% pure-Rust features)
cargo build

# Run the tests (nextest is the canonical runner)
cargo nextest run

# Run the decoder fuzz harnesses specifically
cargo nextest run -p oxiaudio-integration-tests --test fuzz_decoders
```

## Required Checks (must pass before review)

1. **No warnings.** Clippy must be completely silent under the default feature
   set:

   ```bash
   cargo clippy --all-targets -- -D warnings
   ```

2. **Tests green.** All tests must pass:

   ```bash
   cargo nextest run
   ```

3. **Formatting.**

   ```bash
   cargo fmt --all
   ```

## Engineering Policies

These are hard requirements, not suggestions:

- **Pure Rust by default.** The default feature set must contain **no
  C/C++/Fortran dependencies**. Any native/FFI backend (e.g. LAME MP3 encoding)
  must be optional and feature-gated, and the crate must build and pass tests
  without it.
- **No `unwrap()` / `expect()` on untrusted input.** Any code reachable from
  decoding or encoding attacker-controlled audio must return a typed
  [`oxiaudio_core::OxiAudioError`] instead of panicking. Malformed input must
  **never** cause a panic, abort, hang, or unbounded allocation. New decoders
  must be accompanied by a fuzz property in
  `crates/oxiaudio-integration-tests/tests/fuzz_decoders.rs`.
- **No `unsafe`.** Crates build under `#![deny(unsafe_code)]`. The sole
  exception is the opt-in, non-default `mmap` feature; do not introduce new
  `unsafe` without prior discussion, and document any that is unavoidable.
- **Workspace-managed dependencies.** Declare dependencies in the root
  `Cargo.toml` `[workspace.dependencies]` and reference them from member crates
  with `<dep>.workspace = true`. Do not pin per-crate versions. Prefer the
  latest release available on crates.io, and keep new dependencies pure Rust.
- **File size limit.** Keep every source file **under 2000 lines**. Split larger
  files into focused modules.
- **Naming & style.** Follow standard Rust conventions (`snake_case` items,
  `CamelCase` types). All code, comments, and documentation are in **English**.
- **Tests use temporary directories.** Never hardcode absolute paths in tests;
  use `std::env::temp_dir()` for any filesystem fixtures.
- **Honest documentation.** Do not label partial functionality as complete.
  Document real conformance limitations where they exist rather than papering
  over them.

## Pull Requests

- Keep PRs focused; one logical change per PR.
- Include tests for new behavior and regression tests for bug fixes.
- Describe what changed and why, and note any measured conformance limitations.
- Do not bump versions or publish; releases are handled by the maintainers.

By contributing, you agree that your contributions are licensed under the
project's Apache-2.0 license (see [`LICENSE`](./LICENSE)).
