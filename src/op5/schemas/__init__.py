"""JSON Schema registry for Phase 03."""

from pathlib import Path

_SCHEMAS_DIR = Path(__file__).parent


def load_schema(name: str) -> dict:
    """Load a schema JSON file by basename (no extension)."""
    p = _SCHEMAS_DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Schema not found: {p}")
    import json

    return json.loads(p.read_text(encoding="utf-8"))


__all__ = ["load_schema"]
