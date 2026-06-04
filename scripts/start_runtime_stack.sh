#!/usr/bin/env bash
# Start FunASR Runtime then gateway (Linux / WSL)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/deploy"

if [[ ! -f runtime.env ]]; then
  echo "Copy runtime.env.example to runtime.env and configure model paths."
  cp -n runtime.env.example runtime.env || true
fi

docker compose --env-file runtime.env -f docker-compose.yml up -d funasr-runtime
echo "Waiting for Runtime on port ${RUNTIME_PORT:-10095}..."
sleep 10

export VOICETOTEXT_CONFIG="$ROOT/config.enterprise.yaml"
python "$ROOT/scripts/run_server.py" --config "$ROOT/config.enterprise.yaml"
