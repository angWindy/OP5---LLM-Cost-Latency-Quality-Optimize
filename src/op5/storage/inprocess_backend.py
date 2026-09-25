"""In-process storage backend (no Docker required).

Used when STORAGE_BACKEND=inprocess. Maps the 4 sub-clients to:
- ChromaDB PersistentClient (local SQLite) for vector store
- sqlite3 for Postgres-equivalent audit log
- local filesystem for object storage
- in-memory dict for Redis cache
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from op5.storage.base import (
    AuditRecord,
    ChromaLike,
    EmbeddingItem,
    PostgresLike,
    RedisLike,
    S3Like,
    StorageClient,
)


DEFAULT_CACHE_ROOT = Path(os.getenv("OP5_CACHE_DIR", "data/cache"))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class InProcessChromaClient(ChromaLike):
    def __init__(self, path: Path | None = None) -> None:
        import chromadb

        persist_dir = (path or DEFAULT_CACHE_ROOT / "chroma").resolve()
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collections: dict[str, Any] = {}

    def _get_collection(self, name: str) -> Any:
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(name=name)
        return self._collections[name]

    def add(self, collection: str, items: list[EmbeddingItem]) -> None:
        coll = self._get_collection(collection)
        coll.add(
            ids=[it.id for it in items],
            embeddings=[it.embedding for it in items],
            documents=[it.document for it in items],
            metadatas=[it.metadata for it in items],
        )

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        coll = self._get_collection(collection)
        res = coll.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
        out: list[dict[str, Any]] = []
        for i, doc_id in enumerate(res["ids"][0]):
            out.append(
                {
                    "id": doc_id,
                    "document": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i] if res.get("metadatas") else {},
                    "distance": res["distances"][0][i] if res.get("distances") else None,
                }
            )
        return out

    def count(self, collection: str) -> int:
        return self._get_collection(collection).count()

    def delete_collection(self, collection: str) -> None:
        try:
            self._client.delete_collection(name=collection)
        except Exception:
            pass
        self._collections.pop(collection, None)


class InProcessPostgresClient(PostgresLike):
    """SQLite-backed audit + run metadata."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (DEFAULT_CACHE_ROOT / "audit.sqlite")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS redaction_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    span_count TEXT NOT NULL,
                    total_redactions INTEGER NOT NULL,
                    source TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_redaction_audit_case ON redaction_audit(case_id);
                CREATE TABLE IF NOT EXISTS run_metadata (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    track TEXT,
                    config TEXT,
                    provider TEXT,
                    args TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def insert_audit(self, record: AuditRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO redaction_audit (ts, case_id, policy_version, span_count, total_redactions, source) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.ts or _now_iso(),
                    record.case_id,
                    record.policy_version,
                    json.dumps(record.span_count),
                    record.total_redactions,
                    record.source,
                ),
            )
            self._conn.commit()

    def query_audit(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, ts, case_id, policy_version, span_count, total_redactions, source FROM redaction_audit WHERE case_id = ? ORDER BY ts DESC",
                (case_id,),
            )
            return [
                {"id": r[0], "ts": r[1], "case_id": r[2], "policy_version": r[3], "span_count": json.loads(r[4]), "total_redactions": r[5], "source": r[6]}
                for r in cur.fetchall()
            ]

    def insert_run(self, phase: str, track: str, config: str, provider: str | None, args: dict[str, Any]) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO run_metadata (ts, phase, track, config, provider, args) VALUES (?, ?, ?, ?, ?, ?)",
                (_now_iso(), phase, track, config, provider, json.dumps(args)),
            )
            self._conn.commit()
            return cur.lastrowid or 0

    def close(self) -> None:
        self._conn.close()


class InProcessS3Client(S3Like):
    """Local filesystem object storage (mirror of S3 bucket/key)."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = (root or DEFAULT_CACHE_ROOT / "minio").resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, bucket: str, key: str) -> Path:
        safe_key = key.replace("..", "_")
        return self._root / bucket / safe_key

    def put(self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream", metadata: dict[str, str] | None = None) -> None:
        path = self._path(bucket, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            path.write_bytes(data)
            if metadata:
                meta_path = path.with_suffix(path.suffix + ".meta.json")
                meta_path.write_text(json.dumps({"content_type": content_type, "metadata": metadata}))

    def get(self, bucket: str, key: str) -> bytes | None:
        path = self._path(bucket, key)
        if not path.exists():
            return None
        return path.read_bytes()

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        root = self._root / bucket
        if not root.exists():
            return []
        return sorted(
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and not p.name.endswith(".meta.json") and str(p.relative_to(root)).startswith(prefix)
        )

    def ensure_bucket(self, bucket: str) -> None:
        (self._root / bucket).mkdir(parents=True, exist_ok=True)


class InProcessRedisClient(RedisLike):
    """Thread-safe in-memory dict (process-local, no TTL persistence)."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[bytes, float | None]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expires = item
            if expires is not None and expires < time.time():
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        with self._lock:
            expires = (time.time() + ttl_seconds) if ttl_seconds else None
            self._data[key] = (value, expires)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def rate_limit_check(self, key: str, max_per_minute: int) -> bool:
        bucket_key = f"ratelimit:{key}:{int(time.time() // 60)}"
        with self._lock:
            count_raw = self._data.get(bucket_key)
            count = int.from_bytes(count_raw[0], "big") if count_raw else 0
            count += 1
            self._data[bucket_key] = (count.to_bytes(4, "big"), time.time() + 70)
            return count <= max_per_minute


class InProcessStorage(StorageClient):
    def __init__(self) -> None:
        self._chroma: ChromaLike | None = None
        self._postgres: PostgresLike | None = None
        self._s3: S3Like | None = None
        self._redis: RedisLike | None = None

    def chroma(self) -> ChromaLike:
        if self._chroma is None:
            self._chroma = InProcessChromaClient()
        return self._chroma

    def postgres(self) -> PostgresLike:
        if self._postgres is None:
            self._postgres = InProcessPostgresClient()
        return self._postgres

    def s3(self) -> S3Like:
        if self._s3 is None:
            self._s3 = InProcessS3Client()
        return self._s3

    def redis(self) -> RedisLike:
        if self._redis is None:
            self._redis = InProcessRedisClient()
        return self._redis

    def backend_name(self) -> str:
        return "inprocess"
