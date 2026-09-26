#!/usr/bin/env bash
# Start the OP5 Streamlit UI. Prereq: FastAPI service already running on $OP5_API_URL
# (default http://localhost:8000). To start the API: uvicorn scripts.api.serve:app --port 8000
set -e

conda activate vsf 2>/dev/null || true

export OP5_API_URL="${OP5_API_URL:-http://localhost:8000}"
echo "OP5_API_URL=$OP5_API_URL"
echo "Starting Streamlit at http://localhost:8501"

exec streamlit run "$(dirname "$0")/../../src/op5/ui/Home.py" \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --browser.gatherUsageStats false
