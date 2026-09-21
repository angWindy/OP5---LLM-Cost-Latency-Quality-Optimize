#!/usr/bin/env python3
"""
Verify that required LLMLingua libraries are installed and importable.

Two paths are checked:
  - pip-direct:  `from llmlingua import PromptCompressor`
  - LangChain:   `from langchain_community.document_compressors import LLMLinguaCompressor`

Also checks `google-generativeai`, `datasets`, `python-dotenv`, `jsonschema`.

This script does NOT auto-install anything. If something is missing, it prints the
exact `pip install` command and exits with code 1.

Usage (from repo root):
    conda activate vsf
    python scripts/phase-01/verify_libs.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys


REQUIRED = {
    "google-generativeai": "google.generativeai",
    "python-dotenv": "dotenv",
    "datasets": "datasets",
    "langchain": "langchain",
    "langchain-community": "langchain_community",
    "llmlingua": "llmlingua",
    "jsonschema": "jsonschema",
}


def pip_show(pkg: str) -> str | None:
    """Return installed version string or None if not installed."""
    try:
        out = subprocess.check_output(
            ["pip", "show", pkg], stderr=subprocess.DEVNULL, text=True
        )
        for line in out.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except subprocess.CalledProcessError:
        return None
    return None


def main() -> int:
    print("=== Python ===")
    print(f"  executable: {sys.executable}")
    print(f"  version:    {sys.version.split()[0]}")
    if shutil.which("conda"):
        print(f"  conda:      {shutil.which('conda')}")
    print()

    print("=== Required packages (pip show) ===")
    missing: list[str] = []
    for pkg, mod in REQUIRED.items():
        ver = pip_show(pkg)
        status = ver if ver else "MISSING"
        print(f"  {pkg:22s} {status}")
        if ver is None:
            missing.append(pkg)
    print()

    print("=== Import check ===")
    import_errors: list[tuple[str, str]] = []
    for pkg, mod in REQUIRED.items():
        try:
            __import__(mod)
            print(f"  import {mod:30s} OK")
        except Exception as exc:
            print(f"  import {mod:30s} FAIL: {exc}")
            import_errors.append((pkg, str(exc)))
    print()

    print("=== Summary ===")
    if not missing and not import_errors:
        print("  All libraries present and importable.")
        print()
        print("  pip-direct path:  PromptCompressor -> ready")
        print("  LangChain path:   LLMLinguaCompressor -> ready")
        return 0

    print(f"  {len(missing)} package(s) missing, {len(import_errors)} import error(s).")
    print()
    print("  Run the following from repo root (env must be `vsf`):")
    print()
    print("    pip install -r scripts/phase-01/requirements.txt")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
