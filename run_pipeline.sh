#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ $# -eq 0 ]]; then
  printf '%s\n' 'Usage: bash run_pipeline.sh "/path/to/catalog.pdf" [pipeline options]'
  exit 2
fi
if [[ ! -f .env ]]; then
  printf '%s\n' 'Missing .env. Keep your existing DGX settings, or create it from .env.example once.' >&2
  exit 2
fi

# Docker already serves Qwen; this starts only the PDF pipeline client.
# Explicit options after these defaults can override concurrency for a test.
exec uv run --locked --no-sync --env-file .env as-ocr pipeline \
  --async-inference --page-workers 32 --max-in-flight 32 \
  --markdown-engine compact --markdown-max-requests 20 "$@"
