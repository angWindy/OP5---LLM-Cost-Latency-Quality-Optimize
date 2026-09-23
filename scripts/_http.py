"""
Shared HTTP utilities for Phase 02 eval scripts.

Provides:
  - `get_session()`: a process-wide requests.Session with connection pooling,
    keep-alive, and retry adapter. Reusing a session avoids the TCP / TLS
    handshake cost on every Gemini call (~50-150ms saved per call).
  - `call_with_retry()`: a small retry/backoff wrapper around requests.

Why this module exists:
  - The previous scripts created a new `requests.post(...)` on every call,
    forcing a fresh TCP/TLS handshake each time. Over 200 cases that adds up
    to 10-30s of pure network overhead.
  - Retry backoffs were hard-coded to `[30, 60, 120, 180, 300]` which is
    far too long when the API returns a soft 429. We cap at 15s and use
    jittered exponential backoff instead.

This module must NOT change behavior on the happy path (single 200 OK) —
the only observable difference should be lower latency and fewer retries
that time out.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger("op5.http")

# Cap backoff so we don't lose 5+ minutes per retry when quota is exhausted.
_BACKOFF_CAP_SEC = 15.0
_DEFAULT_MAX_RETRIES = 3
_RETRY_STATUSES = (429, 500, 502, 503, 504)

_session: requests.Session | None = None
_session_lock = threading.Lock()


def get_session() -> requests.Session:
    """Return a process-wide Session with connection pooling + retries.

    Thread-safe; created on first call. Safe to call from anywhere.
    """
    global _session
    if _session is not None:
        return _session
    with _session_lock:
        if _session is not None:
            return _session
        sess = requests.Session()
        retry = Retry(
            total=0,  # we handle retries ourselves for finer control
            connect=2,
            read=2,
            status=0,
            allowed_methods=frozenset(["GET", "POST"]),
            backoff_factor=0.3,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=10,
            pool_maxsize=20,
        )
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _session = sess
        return sess


def call_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    timeout: float = 120.0,
    backoff_cap: float = _BACKOFF_CAP_SEC,
    session: requests.Session | None = None,
    **kwargs: Any,
) -> requests.Response:
    """HTTP call with capped exponential backoff on retryable status codes.

    Args:
        method: "GET" or "POST".
        url: target URL.
        max_retries: total attempts (incl. the first one).
        timeout: per-request timeout in seconds.
        backoff_cap: max sleep between retries (seconds).
        session: optional Session; defaults to `get_session()`.
        **kwargs: passed to requests (e.g. json=, headers=, params=).

    Returns:
        The final requests.Response (caller inspects status_code / json()).
    """
    sess = session or get_session()
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = sess.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code not in _RETRY_STATUSES:
                return resp
            if attempt < max_retries - 1:
                wait = min(backoff_cap, 2 ** attempt) + random.uniform(0, 1.0)
                LOG.debug(
                    "retry %d for %s (HTTP %d), sleeping %.1fs",
                    attempt + 1, url, resp.status_code, wait,
                )
                time.sleep(wait)
                continue
            return resp
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = min(backoff_cap, 2 ** attempt) + random.uniform(0, 1.0)
                LOG.debug(
                    "retry %d for %s (%s), sleeping %.1fs",
                    attempt + 1, url, type(exc).__name__, wait,
                )
                time.sleep(wait)
                continue
            raise
    if last_exc:
        raise last_exc
    return sess.request(method, url, timeout=timeout, **kwargs)
