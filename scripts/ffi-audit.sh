#!/usr/bin/env bash
set -euo pipefail
RESULT=$(cargo tree --edges normal 2>/dev/null | grep -E '(minimp3-sys v|mp3lame-sys v|lame-sys v|flac-sys v|flac-bound v|vorbis-sys v|opus-sys v|mad-sys v|id3-sys v)' || true)
if [ -n "$RESULT" ]; then
  echo "FFI AUDIT FAILED: forbidden crates found in default dependency tree:"
  echo "$RESULT"
  exit 1
fi
echo "FFI audit PASSED"
