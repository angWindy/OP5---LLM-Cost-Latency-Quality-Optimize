"""
Unified LLM-as-Judge client with profile-based routing.

Chain: NIM -> OpenRouter -> Gemini (configurable via profiles/).

Usage:
    from op5.llm import LLMJudge

    judge = LLMJudge()  # auto-chain: nim -> openrouter -> gemini
    result = judge.judge(question="...", gold="...", pred="...", context="...")

    judge = LLMJudge(profile="nim")  # single profile
    result = judge.judge(question="...", gold="...", pred="...", context="...")

Result (JudgeResult):
    verdict    : "correct"|"incorrect"|"ambiguous"|"error"|"timeout"
    correct    : True|False|None
    reason     : str
    confidence : 0.0-1.0
    model_used : str
    raw        : str (truncated 500 chars)
    latency_ms : float
"""
from __future__ import annotations

import json, os, re, time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import requests

# ---------------------------------------------------------------
# Shared prompt template (canonical, lives in judge_prompt.py)
# ---------------------------------------------------------------
from op5.llm.judge_prompt import JUDGE_PROMPT_TEMPLATE as JUDGE_PROMPT  # re-export for backward compat

@dataclass
class JudgeResult:
    verdict: Literal["correct","incorrect","ambiguous","error","timeout"]
    correct: bool | None
    reason: str
    confidence: float
    model_used: str
    raw: str
    latency_ms: float

    def to_dict(self) -> dict:
        d = {
            "verdict": self.verdict, "correct": self.correct,
            "reason": self.reason[:200], "confidence": self.confidence,
            "model_used": self.model_used, "raw": self.raw[:500],
            "latency_ms": round(self.latency_ms, 1),
        }
        return d

class _RetryableError(Exception): pass
class _FatalError(Exception): pass

# ---------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------
def _interpolate_env(value: str) -> str:
    pattern = re.compile(r'\$\{([^}]+)\}')
    while True:
        m = pattern.search(value)
        if not m: break
        value = value[:m.start()] + os.environ.get(m.group(1), "") + value[m.end():]
    return value

def _walk(obj):
    if isinstance(obj, dict): return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_walk(v) for v in obj]
    if isinstance(obj, str): return _interpolate_env(obj)
    return obj

def _load_profile(path: Path) -> dict:
    import yaml
    return _walk(yaml.safe_load(path.read_text(encoding="utf-8")))

def _profiles_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "llm" / "profiles"

def _resolve_profile(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return _load_profile(p)
    if (Path.cwd() / name_or_path).exists():
        return _load_profile(Path.cwd() / name_or_path)
    for cp in [_profiles_dir() / f"{name_or_path}.yaml",
               _profiles_dir() / name_or_path]:
        if cp.exists():
            return _load_profile(cp)
    raise FileNotFoundError(f"Profile not found: {name_or_path!r}")

def load_chain(config_path: Path | None = None) -> list[dict]:
    if config_path is None:
        config_path = _profiles_dir() / "config.yaml"
    import yaml
    names = yaml.safe_load(config_path.read_text(encoding="utf-8")) or []
    return [_resolve_profile(name) for name in names]

# ---------------------------------------------------------------
# Gemini REST call
# ---------------------------------------------------------------
def _call_gemini(model, prompt, api_key, temperature=0.1, max_tokens=800, timeout=60) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code == 401: raise _FatalError("401 — check GOOGLE_API_KEY")
    if resp.status_code == 403: raise _FatalError(f"403 — {model!r} unavailable")
    if resp.status_code == 429: raise _RetryableError(f"429 rate limit")
    if resp.status_code == 503: raise _RetryableError(f"503 unavailable")
    if resp.status_code >= 500: raise _RetryableError(f"HTTP {resp.status_code}")
    if resp.status_code != 200: raise _RetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
        return "".join(p.get("text","") for p in data["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError) as exc:
        raise _RetryableError(f"unexpected Gemini response: {exc}") from None

# ---------------------------------------------------------------
# OpenAI-compatible call
# ---------------------------------------------------------------
def _call_openai_compat(base_url, api_type, api_key, model_id, prompt,
                         temperature, max_tokens, timeout, extra_headers=None) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra_headers:
        if extra_headers.get("http_referer"): headers["HTTP-Referer"] = extra_headers["http_referer"]
        if extra_headers.get("x_title"):     headers["X-Title"]     = extra_headers["x_title"]
    payload = {
        "model": model_id, "messages": [{"role":"user","content":prompt}],
        "temperature": temperature, "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.Timeout:
        raise _RetryableError(f"timeout after {timeout}s") from None
    except requests.ConnectionError as exc:
        raise _RetryableError(f"connection error: {exc}") from None

    sc = resp.status_code
    if sc == 401: raise _FatalError(f"401 auth for {api_type}")
    if sc == 403: raise _FatalError(f"403 — {model_id!r} unavailable")
    if sc == 404: raise _FatalError(f"404 — {model_id!r} not found")
    if sc == 429: raise _RetryableError(f"429 rate limit on {model_id!r}")
    if sc == 503: raise _RetryableError(f"503 unavailable for {model_id!r}")
    if sc >= 500: raise _RetryableError(f"HTTP {sc} from {api_type}")
    if sc != 200:
        try: err_body = resp.json()
        except: err_body = {"error": resp.text[:200]}
        raise _RetryableError(f"HTTP {sc}: {err_body.get('detail', err_body.get('error', resp.text[:100]))}")
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise _RetryableError(f"unexpected response shape: {exc}") from None
    return content

def _detect_api_type(url: str) -> str:
    u = url.lower()
    if "nvidia" in u or "ngc" in u: return "nvidia"
    if "gemini" in u or "generativeai" in u or "googleapis" in u: return "gemini"
    return "openai"

# ---------------------------------------------------------------
# Unified Judge
# ---------------------------------------------------------------
class LLMJudge:
    DEFAULT_TIMEOUT = 60
    MAX_RETRIES = 2
    BACKOFF = 2.0

    def __init__(self,
                 profile: str | None = None,
                 profiles: list[str] | None = None,
                 profile_files: list[Path | str] | None = None,
                 log_path: Path | str | None = None):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)
        self.log_path = Path(log_path) if log_path else None
        if profile_files:
            self._chain = [dict(_load_profile(Path(p))) for p in profile_files]
        elif profiles:
            self._chain = [_resolve_profile(n) for n in profiles]
        elif profile:
            self._chain = [_resolve_profile(profile)]
        else:
            self._chain = load_chain()
        self._log_event("init", {
            "chain": [p.get("label", p.get("base_url","?")) for p in self._chain]})

    def judge(self, question="", gold="", pred="", context="") -> JudgeResult:
        prompt = JUDGE_PROMPT.format(
            context=(context or "")[:2000],
            question=question or "(no question — automatic extraction task)",
            gold=(gold or "")[:500], pred=(pred or "")[:500])
        for prof in self._chain:
            result = self._try_profile(prompt, prof)
            if result is not None:
                return result
        return JudgeResult(verdict="error", correct=None, reason="all profiles exhausted",
                           confidence=0.0, model_used="none", raw="", latency_ms=0.0)

    def _try_profile(self, prompt: str, prof: dict) -> JudgeResult | None:
        base_url = prof.get("base_url","")
        api_type = prof.get("api_type", _detect_api_type(base_url))
        api_key  = prof.get("api_key","")
        models   = prof.get("models", [])
        temp     = float(prof.get("temperature", 0.1))
        maxtok   = int(prof.get("max_tokens", 800))
        timeout  = int(prof.get("timeout", self.DEFAULT_TIMEOUT))
        label    = prof.get("label", base_url)
        if not api_key:
            self._log_event("profile_skip", {"label": label, "reason": "no api_key"})
            return None
        for model_id in models:
            for attempt in range(self.MAX_RETRIES + 1):
                t0 = time.perf_counter()
                try:
                    raw = self._call(base_url, api_type, api_key, model_id, prompt,
                                     temp, maxtok, timeout, prof)
                    ms = (time.perf_counter() - t0) * 1000.0
                    result = self._parse(raw, model_id, ms)
                    self._log_event("success", {"model": model_id, "profile": label,
                                               "ms": round(ms,1), "verdict": result.verdict})
                    return result
                except _RetryableError as exc:
                    wait = self.BACKOFF ** attempt * 2.0
                    self._log_event("retry", {"model": model_id, "attempt": attempt+1,
                                              "error": str(exc), "wait_s": round(wait,1)})
                    time.sleep(wait); continue
                except _FatalError as exc:
                    self._log_event("model_skip", {"model": model_id, "reason": str(exc)})
                    break
            self._log_event("model_done", {"model": model_id, "profile": label})
        self._log_event("profile_exhausted", {"label": label})
        return None

    def _call(self, base_url, api_type, api_key, model_id, prompt,
              temperature, max_tokens, timeout_sec, extra):
        if api_type == "gemini":
            return _call_gemini(model_id, prompt, api_key, temperature, max_tokens, timeout_sec)
        return _call_openai_compat(base_url, api_type, api_key, model_id, prompt,
                                   temperature, max_tokens, timeout_sec, extra)

    @staticmethod
    def _parse(raw: str, model: str, ms: float) -> JudgeResult:
        s = (raw or "").strip()
        v, conf, reason = LLMJudge._extract_json(s)
        if v is not None:
            return JudgeResult(verdict=v, correct=(v=="correct") if v!="ambiguous" else None,
                               reason=reason[:200], confidence=conf, model_used=model,
                               raw=s[:500], latency_ms=ms)
        return JudgeResult(verdict="ambiguous", correct=None,
                           reason=f"no JSON verdict (raw: {s[:80]})",
                           confidence=0.5, model_used=model, raw=s[:500], latency_ms=ms)

    @staticmethod
    def _extract_json(text: str):
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                depth += 1 if text[i]=="{" else -1 if text[i]=="}" else 0
                if depth == 0:
                    try: obj = json.loads(text[start:i+1]); return LLMJudge._parse_obj(obj)
                    except: break
        for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))):
            try: obj = json.loads(m.group()); return LLMJudge._parse_obj(obj)
            except: continue
        return None, 0.5, ""

    @staticmethod
    def _parse_obj(obj: dict):
        v = str(obj.get("verdict","")).lower().strip()
        if v not in ("correct","incorrect","ambiguous"): return None, 0.5, ""
        try: conf = float(obj.get("confidence", 0.5))
        except: conf = 0.5
        conf = max(0.0, min(1.0, conf))
        return v, conf, str(obj.get("reason",""))[:200]

    def _log_event(self, event: str, data: dict):
        if self.log_path is None: return
        try:
            line = json.dumps({"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "event": event, **data}, ensure_ascii=False)
            with self.log_path.open("a", encoding="utf-8") as fh: fh.write(line+"\n")
        except: pass

def make_judge(log_path=None, profile=None) -> LLMJudge:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)
    return LLMJudge(profile=profile, log_path=Path(log_path) if log_path else None)
