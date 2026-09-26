"""OP5 FastAPI launcher.

Run with:
    conda activate vsf
    uvicorn scripts.api.serve:app --host 0.0.0.0 --port 8000 --reload

Or directly via:
    python -m scripts.api.serve
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure `src/` is on sys.path so `op5.api` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from op5.api import create_app  # noqa: E402

app = create_app()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Reload on code changes (dev only).")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn not installed. Install with:\n  pip install fastapi uvicorn pydantic httpx",
            file=sys.stderr,
        )
        return 1

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
