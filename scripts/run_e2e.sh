#!/usr/bin/env bash
# run_e2e.sh — Boot the full OP5 LLM RAG + OCR stack end-to-end.
#
# Starts (in this order):
#   1. FastAPI backend   (uvicorn scripts.api.serve)  → http://localhost:8000
#   2. Streamlit UI      (streamlit run Home.py)      → http://localhost:8501
#
# Then runs the 6-step smoke test:
#   - GET  /healthz
#   - POST /track1/extract
#   - POST /track2/ask
#   - GET  /inspect/{case_id}
#   - verify JSONL cost_usd > 0
#   - invoke pareto_plot.py
#
# Usage:
#   bash scripts/run_e2e.sh                  # start fresh (kills old PIDs first)
#   bash scripts/run_e2e.sh --no-ui          # skip Streamlit (backend only)
#   bash scripts/run_e2e.sh --no-smoke       # keep services running, no smoke test
#
# Stop everything:
#   bash scripts/run_e2e.sh --stop           # kills both services
#
# Prereqs:
#   - conda env 'vsf' is set up (conda activate vsf)
#   - GOOGLE_API_KEY (or GOOGLE_API_KEYS) is set in .env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# ── Args ─────────────────────────────────────────────────────────────────────
WITH_UI="true"
WITH_SMOKE="true"
ACTION="start"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-ui)    WITH_UI="false";  shift ;;
    --no-smoke) WITH_SMOKE="false"; shift ;;
    --stop)     ACTION="stop";     shift ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Conda env ─────────────────────────────────────────────────────────────────
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate vsf
fi

# ── Load .env if present ──────────────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

mkdir -p "$REPO_ROOT/results" "$REPO_ROOT/logs"

API_PID_FILE="$REPO_ROOT/logs/api.pid"
UI_PID_FILE="$REPO_ROOT/logs/ui.pid"

stop_service() {
  local name="$1"
  local pid_file="$2"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[$name] stopping PID $pid"
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

start_api() {
  echo "[API] starting FastAPI on :8000 ..."
  nohup python -m uvicorn scripts.api.serve:app \
    --host 0.0.0.0 --port 8000 \
    --no-access-log \
    >"$REPO_ROOT/logs/api.log" 2>&1 &
  echo $! >"$API_PID_FILE"
  echo "[API] PID $(cat "$API_PID_FILE")"
}

start_ui() {
  echo "[UI]  starting Streamlit on :8501 ..."
  nohup streamlit run "$REPO_ROOT/src/op5/ui/Home.py" \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --browser.gatherUsageStats false \
    >"$REPO_ROOT/logs/ui.log" 2>&1 &
  echo $! >"$UI_PID_FILE"
  echo "[UI]  PID $(cat "$UI_PID_FILE")"
}

wait_for_api() {
  local tries=30
  while (( tries > 0 )); do
    if curl -s --max-time 2 http://localhost:8000/healthz >/dev/null 2>&1; then
      echo "[API] /healthz OK"
      return 0
    fi
    (( tries-- ))
    sleep 1
  done
  echo "[API] ERROR: /healthz did not respond in 30s" >&2
  echo "[API] last log lines:" >&2
  tail -30 "$REPO_ROOT/logs/api.log" >&2 || true
  return 1
}

# ── Action dispatch ───────────────────────────────────────────────────────────
if [[ "$ACTION" == "stop" ]]; then
  stop_service "UI"  "$UI_PID_FILE"
  stop_service "API" "$API_PID_FILE"
  echo "stopped."
  exit 0
fi

# Kill stale processes from previous runs (best-effort).
stop_service "UI"  "$UI_PID_FILE" || true
stop_service "API" "$API_PID_FILE" || true
pkill -f "uvicorn scripts.api.serve" 2>/dev/null || true
pkill -f "streamlit run.*op5/ui/Home.py" 2>/dev/null || true
sleep 1

start_api
wait_for_api

if [[ "$WITH_UI" == "true" ]]; then
  start_ui
  # Give Streamlit a moment to bind (no /healthz there; just a sleep).
  sleep 4
fi

cat <<EOF

──────────────────────────────────────────────────────────────────────
  OP5 LLM RAG + OCR stack is up.
    API  :  http://localhost:8000  (FastAPI, /docs for OpenAPI)
    UI   :  http://localhost:8501 (Streamlit)
    Logs :  $(realpath "$REPO_ROOT/logs")
    Stop :  bash scripts/run_e2e.sh --stop
──────────────────────────────────────────────────────────────────────
EOF

if [[ "$WITH_SMOKE" == "true" ]]; then
  echo
  echo "[SMOKE] running 6-step end-to-end smoke test ..."
  python "$REPO_ROOT/scripts/phase-03/smoke_e2e.py"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[SMOKE] FAILED at step $rc — see logs/api.log" >&2
    exit $rc
  fi
  echo "[SMOKE] all 6 steps green. Services still running."
fi
