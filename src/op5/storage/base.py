"""Base classes and factory for storage clients.

Two backends are supported:
- docker: connect to localhost ChromaDB, Postgres, localstack S3, Redis.
- inprocess: use Chroma PersistentClient + SQLite + local FS + dict.

Selection is controlled by the STORAGE_BACKEND env var (default: docker).
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Blob:
    """A binary object in object storage."""

    bucket: str
    key: str
    data: bytes
    content_type: str = "application/octet-stream"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class AuditRecord:
    """A redaction audit record."""

    case_id: str
    policy_version: str
    span_count: dict[str, int]
    total_redactions: int
    source: str
    ts: str = ""

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class EmbeddingItem:
    """A single embedding + metadata for ChromaDB."""

    id: str
    embedding: list[float]
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)


class StorageClient(ABC):
    """Abstract storage facade. Each backend implements all 4 sub-clients."""

    @abstractmethod
    def chroma(self) -> "ChromaLike": ...

    @abstractmethod
    def postgres(self) -> "PostgresLike": ...

    @abstractmethod
    def s3(self) -> "S3Like": ...

    @abstractmethod
    def redis(self) -> "RedisLike": ...

    @abstractmethod
    def backend_name(self) -> str: ...


class ChromaLike(ABC):
    @abstractmethod
    def add(self, collection: str, items: list[EmbeddingItem]) -> None: ...

    @abstractmethod
    def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def count(self, collection: str) -> int: ...

    @abstractmethod
    def delete_collection(self, collection: str) -> None: ...


class PostgresLike(ABC):
    @abstractmethod
    def insert_audit(self, record: AuditRecord) -> None: ...

    @abstractmethod
    def query_audit(self, case_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def insert_run(self, phase: str, track: str, config: str, provider: str | None, args: dict[str, Any]) -> int: ...

    @abstractmethod
    def close(self) -> None: ...


class S3Like(ABC):
    @abstractmethod
    def put(self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream", metadata: dict[str, str] | None = None) -> None: ...

    @abstractmethod
    def get(self, bucket: str, key: str) -> bytes | None: ...

    @abstractmethod
    def list_objects(self, bucket: str, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def ensure_bucket(self, bucket: str) -> None: ...


class RedisLike(ABC):
    @abstractmethod
    def get(self, key: str) -> bytes | None: ...

    @abstractmethod
    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def rate_limit_check(self, key: str, max_per_minute: int) -> bool: ...


class StorageFactory:
    """Selects Docker or in-process backend based on STORAGE_BACKEND env var."""

    @staticmethod
    def from_env(env_path: str | None = None) -> StorageClient:
        if env_path:
            try:
                from dotenv import load_dotenv

                load_dotenv(env_path)
            except ImportError:
                pass
        backend = os.getenv("STORAGE_BACKEND", "docker").lower().strip()
        if backend == "docker":
            from op5.storage.docker_backend import DockerStorage

            return DockerStorage()
        elif backend == "inprocess":
            from op5.storage.inprocess_backend import InProcessStorage

            return InProcessStorage()
        else:
            raise ValueError(f"Unknown STORAGE_BACKEND={backend!r}. Use 'docker' or 'inprocess'.")


_STORAGE: StorageClient | None = None


def STORAGE() -> StorageClient:
    """Lazy singleton accessor."""
    global _STORAGE
    if _STORAGE is None:
        _STORAGE = StorageFactory.from_env()
    return _STORAGE
