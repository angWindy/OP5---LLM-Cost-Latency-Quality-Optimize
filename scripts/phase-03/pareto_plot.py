"""Pareto plot for Phase 03: cost vs quality scatter, highlight Pareto frontier.

Reads results/phase-03-track{1,2}-scores.jsonl, produces two PNGs in doc/figs/:
  doc/figs/phase-03-track1.png  (extraction, T1)
  doc/figs/phase-03-track2.png  (RAG, T2)

For T1: x-axis = avg_cost_usd, y-axis = avg_field_f1, color = avg_pii_f1.
For T2: x-axis = avg_cost_usd, y-axis = avg_token_f1, color = refusal_accuracy.
A point on the Pareto frontier is one where no other config has both lower
cost AND higher quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUTPUT_T1 = Path("doc/figs/phase-03-track1.png")
OUTPUT_T2 = Path("doc/figs/phase-03-track2.png")


def _load_track_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "SUMMARY" in rec:
                return rec["SUMMARY"]
    return {}


def _pareto_frontier(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    frontier: list[tuple[float, float, str]] = []
    for x, y, lbl in points:
        dominated = any(
            (cx <= x and cy >= y and (cx < x or cy > y))
            for cx, cy, _ in points
            if (cx, cy) != (x, y)
        )
        if not dominated:
            frontier.append((x, y, lbl))
    frontier.sort(key=lambda p: p[0])
    seen = set()
    pruned: list[tuple[float, float, str]] = []
    for x, y, lbl in frontier:
        key = (round(x, 6), round(y, 6))
        if key in seen:
            continue
        seen.add(key)
        pruned.append((x, y, lbl))
    return pruned


def _plot_track1(summary: dict, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"  matplotlib unavailable; skipping {out_path}", file=sys.stderr)
        return
    configs = summary.get("configs", {})
    if not configs:
        print(f"  no T1 configs in summary; skipping", file=sys.stderr)
        return
    costs: list[float] = []
    f1s: list[float] = []
    labels: list[str] = []
    piis: list[float] = []
    for cfg, s in configs.items():
        costs.append(s.get("avg_cost_usd", 0.0))
        f1s.append(s.get("avg_field_f1", 0.0))
        labels.append(cfg)
        piis.append(s.get("avg_pii_f1", 0.0))

    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(costs, f1s, c=piis, cmap="viridis", s=120, edgecolors="black")
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (costs[i], f1s[i]), textcoords="offset points", xytext=(8, 6), fontsize=10)
    pareto = _pareto_frontier(list(zip(costs, f1s, labels)))
    if pareto:
        ax.plot([p[0] for p in pareto], [p[1] for p in pareto], "r--", alpha=0.5, label="Pareto frontier")
    ax.set_xscale("log")
    ax.set_xlabel("Avg cost per case (USD, log scale)")
    ax.set_ylabel("Avg field F1")
    ax.set_title("Phase 03 — Track 1 (extraction): cost vs quality")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.colorbar(sc, ax=ax, label="avg PII F1")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  wrote {out_path}")


def _plot_track2(summary: dict, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"  matplotlib unavailable; skipping {out_path}", file=sys.stderr)
        return
    configs = summary.get("configs", {})
    if not configs:
        print(f"  no T2 configs in summary; skipping", file=sys.stderr)
        return
    costs: list[float] = []
    f1s: list[float] = []
    labels: list[str] = []
    refusals: list[float] = []
    for cfg, s in configs.items():
        costs.append(s.get("avg_cost_usd", 0.0))
        f1s.append(s.get("avg_token_f1", 0.0))
        labels.append(cfg)
        refusals.append(s.get("refusal_accuracy", 0.0))

    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(costs, f1s, c=refusals, cmap="plasma", s=120, edgecolors="black")
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (costs[i], f1s[i]), textcoords="offset points", xytext=(8, 6), fontsize=10)
    pareto = _pareto_frontier(list(zip(costs, f1s, labels)))
    if pareto:
        ax.plot([p[0] for p in pareto], [p[1] for p in pareto], "r--", alpha=0.5, label="Pareto frontier")
    ax.set_xscale("log")
    ax.set_xlabel("Avg cost per QA (USD, log scale)")
    ax.set_ylabel("Avg token F1")
    ax.set_title("Phase 03 — Track 2 (RAG): cost vs quality")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.colorbar(sc, ax=ax, label="refusal accuracy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track1-scores", default="results/phase-03-track1-scores.jsonl")
    parser.add_argument("--track2-scores", default="results/phase-03-track2-scores.jsonl")
    parser.add_argument("--out-t1", default=str(OUTPUT_T1))
    parser.add_argument("--out-t2", default=str(OUTPUT_T2))
    args = parser.parse_args()

    s1 = _load_track_summary(Path(args.track1_scores))
    s2 = _load_track_summary(Path(args.track2_scores))

    if s1:
        _plot_track1(s1, Path(args.out_t1))
    if s2:
        _plot_track2(s2, Path(args.out_t2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
