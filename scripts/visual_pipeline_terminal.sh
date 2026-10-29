#!/usr/bin/env bash
# Start inside the named tmux socket; all inference runs in this terminal process.
set -euo pipefail
cd /home/domapp/Z_AiWork/As_ocr
OCR_VISUAL_ROOT="${1:?Pass the isolated run root}"
OCR_VISUAL_ACTION="${2:?Pass smoke, run, or resume}"
case "$OCR_VISUAL_ACTION" in smoke|smoke-resume|smoke-recovery|run|resume) ;; *) exit 2 ;; esac
OCR_VISUAL_LOG="$OCR_VISUAL_ROOT/$OCR_VISUAL_ACTION-terminal.log"
if [[ -e "$OCR_VISUAL_LOG" ]]; then
    OCR_VISUAL_LOG="$OCR_VISUAL_ROOT/$OCR_VISUAL_ACTION-terminal-$(date -u +%Y%m%dT%H%M%SZ).log"
fi
exec uv run --env-file .env --frozen --no-sync python -u -B scripts/visual_pipeline_run.py \
    "$OCR_VISUAL_ACTION" --root "$OCR_VISUAL_ROOT" > "$OCR_VISUAL_LOG" 2>&1
