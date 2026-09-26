#!/usr/bin/env bash
# run_baseline_judge.sh — Phase 02 batch: baseline + LongLLMLingua + judge.
#
# Runs, in order:
#   1. python scripts/phase-02/run_baseline.py          (no compression)
#   2. python scripts/phase-02/run_compressed.py --rate 0.4
#   3. python scripts/phase-02/run_compressed.py --rate 0.5
#   4. python scripts/phase-02/run_compressed.py --rate 0.6
#   5. python scripts/phase-02/run_compressed.py --rate 0.7
#   6. python scripts/phase-02/judge_196_concurrent.py --profile deepseek
#
# Each step is resumable (JSONL append-mode); running twice picks up where
# it left off instead of redoing finished cases.
#
# Usage:
#   bash scripts/run_baseline_judge.sh                  # full pipeline
#   bash scripts/run_baseline_judge.sh --skip-baseline  # only compression runs + judge
#   bash scripts/run_baseline_judge.sh --only-judge     # just re-run judge
#   bash scripts/run_baseline_judge.sh --rate 0.4,0.6   # subset of rates
#   bash scripts/run_baseline_judge.sh --workers 16     # override judge concurrency
#
# Prereqs:
#   - conda env 'vsf' is set up
#   - GOOGLE_API_KEY (or GOOGLE_API_KEYS) for Gemini calls
#   - DEEPSEEK_API_KEY for LLM-as-judge

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# ── Args ─────────────────────────────────────────────────────────────────────
DO_BASELINE="true"
DO_COMPRESSED="true"
DO_JUDGE="true"
RATES=(0.4 0.5 0.6 0.7)
WORKERS=12
JUDGE_PROFILE="deepseek"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-baseline)
      DO_BASELINE="false"
      shift
      ;;
    --only-judge)
      DO_BASELINE="false"
      DO_COMPRESSED="false"
      DO_JUDGE="true"
      shift
      ;;
    --no-judge)
      DO_JUDGE="false"
      shift
      ;;
    --rate)
      IFS=',' read -r -a RATES <<<"$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --profile)
      JUDGE_PROFILE="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,30p' "$0"
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

mkdir -p "$REPO_ROOT/results/runs/phase-02"

# ── Helpers ───────────────────────────────────────────────────────────────────
banner() {
  echo
  echo "════════════════════════════════════════════════════════════════════"
  echo "  $*"
  echo "════════════════════════════════════════════════════════════════════"
}

require_key() {
  local var="$1"
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: env var $var is not set. Add it to .env and retry." >&2
    exit 1
  fi
}

# ── 1. Baseline (no compression) ─────────────────────────────────────────────
if [[ "$DO_BASELINE" == "true" ]]; then
  banner "STEP 1: BASELINE (no compression)  |  gemini-3.5-flash-lite"
  require_key GOOGLE_API_KEY
  python "$REPO_ROOT/scripts/phase-02/run_baseline.py" \
    --out-tag baseline_n196 \
    --model gemini-3.5-flash-lite \
    --sleep 3
fi

# ── 2-5. LongLLMLingua compressed runs ────────────────────────────────────────
if [[ "$DO_COMPRESSED" == "true" ]]; then
  for rate in "${RATES[@]}"; do
    pct="$(awk -v r="$rate" 'BEGIN { printf "%d", r*100 }')"
    tag="rate${pct}"
    banner "STEP: COMPRESSED  |  rate=$rate  |  tag=$tag"
    require_key GOOGLE_API_KEY
    python "$REPO_ROOT/scripts/phase-02/run_compressed.py" \
      --rate "$rate" \
      --model gemini-3.5-flash-lite \
      --sleep 3 \
      --out-tag "$tag"
  done
fi

# ── 6. LLM-as-judge ──────────────────────────────────────────────────────────
if [[ "$DO_JUDGE" == "true" ]]; then
  banner "STEP: JUDGE  |  profile=$JUDGE_PROFILE  workers=$WORKERS"
  require_key DEEPSEEK_API_KEY
  python "$REPO_ROOT/scripts/phase-02/judge_196_concurrent.py" \
    --profile "$JUDGE_PROFILE" \
    --workers "$WORKERS"
fi

banner "DONE  |  outputs in results/judges/${JUDGE_PROFILE}/phase-02/"
echo "Inspect:"
echo "  - results/judges/${JUDGE_PROFILE}/phase-02/phase-02-judged-*.jsonl"
echo "  - results/summaries/phase-02/phase-02-${JUDGE_PROFILE}-196-summary.json"
