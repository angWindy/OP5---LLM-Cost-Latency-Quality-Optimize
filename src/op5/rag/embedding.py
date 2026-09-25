"""Multilingual ONNX embedding wrapper.

Default model: intfloat/multilingual-e5-small (ONNX, no torch at inference).
Uses onnxruntime + transformers AutoTokenizer.
Falls back to deterministic hash-based pseudo-embeddings if model download fails.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
EMBED_DIM = 384


def _pseudo_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    h = hashlib.sha512(text.encode("utf-8")).digest()
    out: list[float] = []
    for i in range(dim):
        byte = h[i % len(h)]
        out.append(((byte / 255.0) - 0.5) * 0.1)
    norm = sum(x * x for x in out) ** 0.5 or 1.0
    return [x / norm for x in out]


class MultilingualEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, use_onnx: bool = True) -> None:
        self.model_name = model_name
        self.use_onnx = use_onnx
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    def _load(self) -> tuple[Any, Any] | None:
        if self._model is not None:
            return self._model, self._tokenizer
        with self._lock:
            if self._model is not None:
                return self._model, self._tokenizer
            try:
                os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.use_onnx:
                    try:
                        from optimum.onnxruntime import ORTModelForFeatureExtraction

                        self._model = ORTModelForFeatureExtraction.from_pretrained(self.model_name, export=True)
                    except Exception:
                        try:
                            from sentence_transformers import SentenceTransformer

                            self._model = SentenceTransformer(self.model_name)
                            self.use_onnx = False
                        except Exception as e:
                            logger.warning("Embedding model load failed (%s); using pseudo embeddings", e)
                            return None
                else:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning("Embedding init failed (%s); pseudo embeddings", e)
                return None
        return self._model, self._tokenizer

    def embed(self, texts: list[str]) -> list[list[float]]:
        loaded = self._load()
        if loaded is None:
            return [_pseudo_embed(t) for t in texts]
        model, tokenizer = loaded
        try:
            if hasattr(model, "encode"):
                return [list(v) for v in model.encode(texts, normalize_embeddings=True).tolist()]
            else:
                inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                outputs = model(**inputs)
                import numpy as np
                attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
                token_embeddings = outputs.last_hidden_state * attention_mask
                summed = token_embeddings.sum(dim=1)
                counts = attention_mask.sum(dim=1)
                embeddings = (summed / counts).detach().cpu().numpy()
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                normed = embeddings / np.clip(norms, 1e-9, None)
                return normed.tolist()
        except Exception as e:
            logger.warning("Embedding call failed (%s); pseudo", e)
            return [_pseudo_embed(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        if "e5" in self.model_name.lower():
            return self.embed([f"query: {query}"])[0]
        return self.embed([query])[0]

    def embed_passage(self, passage: str) -> list[float]:
        if "e5" in self.model_name.lower():
            return self.embed([f"passage: {passage}"])[0]
        return self.embed([passage])[0]
