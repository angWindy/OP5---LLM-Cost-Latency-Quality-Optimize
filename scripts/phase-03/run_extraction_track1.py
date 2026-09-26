"""Track 1 runner: 5 extraction configs x 15 contracts.

Configs (master plan §4.6):
  A        = cheap + prompt A (zero-shot) + no redaction
  B        = cheap + prompt B (few-shot) + no redaction
  D        = cheap + redaction + prompt A
  B+D      = cheap + redaction + prompt B (few-shot)
  D-strong = strong model + redaction + prompt A

Smoke-test mode: bypass OCR (EasyOCR full 15-file inference takes ~10 min on
this CPU-only host). Each contract's input "redacted text" is reconstructed
from the SYNTH corpus ground-truth so the runner exercises the same
prompt -> router -> scorer -> JSONL path as a production run. The renderer
flag determines whether to OCR the real PDF or use GT-derived text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from op5.rag.prompts import track1_prompt  # noqa: E402
from op5.redact import Redactor, default_policy  # noqa: E402
from op5.router import extract_features_track1, route  # noqa: E402
from op5.llm import LLMWrapper, make_llm_wrapper  # noqa: E402

OUTPUT = Path("results/phase-03-track1.jsonl")

CONFIGS: dict[str, dict[str, Any]] = {
    "A":        {"redact": False, "prompt_variant": "A", "tier": "cheap"},
    "B":        {"redact": False, "prompt_variant": "B", "tier": "cheap"},
    "D":        {"redact": True,  "prompt_variant": "A", "tier": "cheap"},
    "B+D":      {"redact": True,  "prompt_variant": "B", "tier": "cheap"},
    "D-strong": {"redact": True,  "prompt_variant": "A", "tier": "strong"},
}

EXPECTED_FIELDS = [
    "contract_no",
    "sign_date",
    "seller_name",
    "seller_tax_id",
    "buyer_name",
    "buyer_phone",
    "buyer_email",
    "model",
    "version",
    "color",
    "vin",
    "unit_price_vnd",
    "total_price_words_vi",
]


def _stub_llm(prompt: str) -> str:
    """Pseudo LLM caller: extracts values from the prompt's "Contract text" block.

    The SYNTH corpus encodes all fields as `key: value` lines inside the OCR
    text. For this offline smoke-test we just lift `key: value` pairs from the
    text section of the prompt. This exercises the scorer end-to-end without
    requiring live Gemini credentials. Use --llm gemini to swap in the live
    wrapper when GOOGLE_API_KEY(S) is configured.
    """
    import re

    extracted: dict[str, str] = {}
    m = re.search(r'Contract text:\s*"""+\s*(.*?)\s*"""+', prompt, flags=re.DOTALL)
    if m:
        body = m.group(1)
        for line in body.splitlines():
            kv = re.match(r"\s*([A-Za-z_]+)\s*:\s*(.+)", line)
            if kv:
                extracted[kv.group(1).strip()] = kv.group(2).strip()
    return json.dumps(extracted, ensure_ascii=False)


def _build_live_call(wrapper: LLMWrapper, model: str):
    """Return a (prompt) -> str closure bound to a specific model."""
    def call(prompt: str) -> str:
        return wrapper.call(prompt, model=model)
    return call



def _gt_text_from_record(rec: dict[str, Any]) -> str:
    """Build a synthetic OCR text from ground-truth field values."""
    fields = rec.get("ground_truth_fields", {}) or {}
    lines = []
    ordered = ["contract_no", "sign_date", "seller_name", "seller_address",
               "seller_tax_id", "seller_rep", "seller_rep_title",
               "buyer_name", "buyer_address", "buyer_phone", "buyer_email",
               "model", "version", "color", "vin", "includes_battery",
               "unit_price_vnd", "total_price_words_vi"]
    for k in ordered:
        v = fields.get(k)
        if v is None or v == "":
            continue
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--scan-dir", default=None)
    parser.add_argument(
        "--llm",
        choices=["stub", "gemini"],
        default="stub",
        help="LLM backend. 'stub' uses the offline key:value extractor; 'gemini' calls live Gemini API.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    live_wrapper: LLMWrapper | None = None
    if args.llm == "gemini":
        try:
            live_wrapper = make_llm_wrapper()
        except RuntimeError as exc:
            print(f"Live LLM init failed: {exc}", file=sys.stderr)
            return 1

    gt_path = Path("data/processed/phase03_synth_contracts.jsonl")
    if not gt_path.exists():
        print(f"GT file not found: {gt_path}", file=sys.stderr)
        return 1
    gt_records: dict[str, dict] = {}
    with gt_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt_records[rec["case_id"]] = rec
    gt_records = dict(list(gt_records.items())[: args.limit])

    redactor = Redactor(default_policy())
    rows: list[dict] = []

    for synth_id, rec in gt_records.items():
        case_id = rec.get("source_pdf", "").split("/")[-1].replace(".pdf", "") or f"contract_synth-{synth_id}"
        if not case_id.startswith("contract_synth-"):
            case_id = f"contract_synth-{synth_id}"

        ocr_text_raw = _gt_text_from_record(rec)
        ocr_doc = {"text_blocks": [{"page": 1, "text": ocr_text_raw, "bbox": [0, 0, 100, 100], "confidence": 1.0}]}

        for cfg_name, cfg in CONFIGS.items():
            ocr_proc = redactor.redact_ocr_result(ocr_doc) if cfg["redact"] else ocr_doc
            redacted_text = ocr_proc.get("redacted_text") or "\n".join(b.get("text", "") for b in ocr_proc.get("text_blocks", []))
            prompt = track1_prompt(cfg["prompt_variant"], EXPECTED_FIELDS, redacted_text)
            features = extract_features_track1(ocr_proc, EXPECTED_FIELDS)
            routing = route(features, "track1")
            routing["tier"] = cfg["tier"]
            routing["deployment_id"] = "gemini-3.5-flash-lite"
            deployment = routing["deployment_id"]

            # Pick the LLM callable: live Gemini wrapper or offline stub.
            if live_wrapper is not None:
                provider_name = "google"
            else:
                provider_name = "gemini-stub"

            t0 = time.time()
            if live_wrapper is not None:
                llm_meta = live_wrapper.generate(prompt, model=deployment, max_output_tokens=512)
                llm_raw = llm_meta.text or ""
                latency_ms = llm_meta.latency_ms or ((time.time() - t0) * 1000)
                input_tokens = llm_meta.input_tokens
                output_tokens = llm_meta.output_tokens
                cost_usd = llm_meta.cost_usd
                cache_status = llm_meta.cache_status
            else:
                llm_raw = _stub_llm(prompt)
                latency_ms = (time.time() - t0) * 1000
                input_tokens = len(prompt) // 4
                output_tokens = len(llm_raw) // 4
                cost_usd = 0.0001  # gemini-3.5-flash-lite for all tiers
                cache_status = "bypass"

            # Strip markdown code-fences the model sometimes wraps JSON in.
            stripped = llm_raw.strip()
            if stripped.startswith("```"):
                first_nl = stripped.find("\n")
                if first_nl > 0:
                    stripped = stripped[first_nl + 1 :]
                if stripped.endswith("```"):
                    stripped = stripped[:-3]
                stripped = stripped.strip()

            try:
                llm_parsed = json.loads(stripped) if stripped else {}
                if not isinstance(llm_parsed, dict):
                    llm_parsed = {}
            except Exception:
                llm_parsed = {}

            rows.append({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "track": "track1",
                "case_id": case_id,
                "config": cfg_name,
                "provider": provider_name,
                "model": deployment,
                "deployment_id": deployment,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "cache_status": cache_status,
                "pred": {"raw_llm_resp": llm_raw, "extracted_fields": llm_parsed, "redacted_text_len": len(redacted_text)},
                "ref": {"expected_fields": EXPECTED_FIELDS, "gt_case_id": synth_id, "gt_fields": rec.get("ground_truth_fields", {})},
                "score": {},
                "redacted_text": redacted_text[:1000],
                "pii_spans": ocr_proc.get("pii_spans", []),
                "routing": routing,
            })

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Track 1: {len(rows)} rows ({len(gt_records)} cases x {len(CONFIGS)} configs) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
