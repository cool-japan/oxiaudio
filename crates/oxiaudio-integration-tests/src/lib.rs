//! Integration tests for the OxiAudio workspace.
//!
//! This crate exists solely to host cross-crate round-trip tests and benches that
//! exercise both `oxiaudio-encode` and `oxiaudio-decode` together. Keeping them here
//! (instead of in either codec crate's dev-dependencies) breaks the circular
//! dev-dependency that otherwise blocks `cargo publish`. This crate is never published
//! (`publish = false`).
