#!/usr/bin/env bash
set -euo pipefail

echo "Starting NLLB inference server..."
echo "  NLLB_MODEL_SIZE=${NLLB_MODEL_SIZE:-}"
echo "  NLLB_MODEL_ID=${NLLB_MODEL_ID:-}"
echo "  TP_SIZE=${TP_SIZE:-}"
echo "  DTYPE=${DTYPE:-}"
echo "  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-}"
echo "  MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-}"
echo "  HOST=${HOST:-0.0.0.0}"
echo "  PORT=${PORT:-8000}"

exec python -m uvicorn server:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --workers 1
