"""Lightweight HTTP client wrapping the OP5 FastAPI service for the Streamlit UI."""

from __future__ import annotations

import os
from typing import Any

import httpx


class OP5Api:
    """Client for the FastAPI service. Base URL via `OP5_API_URL` env (default http://localhost:8000)."""

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = (base_url or os.getenv("OP5_API_URL", "http://localhost:8000")).rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return httpx.get(f"{self.base_url}/healthz", timeout=self.timeout).json()

    def track1_configs(self) -> list[str]:
        return httpx.get(f"{self.base_url}/track1/configs", timeout=self.timeout).json().get("configs", [])

    def track2_configs(self) -> list[str]:
        return httpx.get(f"{self.base_url}/track2/configs", timeout=self.timeout).json().get("configs", [])

    def track2_cases(self) -> list[str]:
        return httpx.get(f"{self.base_url}/track2/cases", timeout=self.timeout).json().get("cases", [])

    def track1_extract(self, pdf_path: str, config: str, use_ocr: bool = True, expected_fields: list[str] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"pdf_path": pdf_path, "config": config, "use_ocr": use_ocr}
        if expected_fields:
            body["expected_fields"] = expected_fields
        return httpx.post(f"{self.base_url}/track1/extract", json=body, timeout=self.timeout).json()

    def track2_ask(self, case_id: str, question: str, config: str, top_k: int = 8) -> dict[str, Any]:
        body = {"case_id": case_id, "question": question, "config": config, "top_k": top_k}
        return httpx.post(f"{self.base_url}/track2/ask", json=body, timeout=self.timeout).json()

    def inspect(self, case_id: str) -> dict[str, Any]:
        return httpx.get(f"{self.base_url}/inspect/{case_id}", timeout=self.timeout).json()
