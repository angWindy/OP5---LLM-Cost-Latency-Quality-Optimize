"""Docker-mode storage backend.

Connects to the 4 services launched by docker/docker-compose.yml:
- ChromaDB at localhost:8000
- Postgres at localhost:5432
- localstack S3 at localhost:4566 (MinIO-compatible via boto3)
- Redis at localhost:6379

Each sub-client is lazily instantiated.
"""

from __future__ import annotations

import json
import logging
import os
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

logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


class DockerChromaClient(ChromaLike):
    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        import chromadb

        self._client = chromadb.HttpClient(host=host, port=port)
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


class DockerPostgresClient(PostgresLike):
    def __init__(self, host: str = "localhost", port: int = 5432, user: str = "op5", password: str = "op5", dbname: str = "op5") -> None:
        import psycopg2
        import psycopg2.extras

        self._psycopg2 = psycopg2
        self._extras = psycopg2.extras
        self._conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname=dbname)
        self._conn.autocommit = True

    def insert_audit(self, record: AuditRecord) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO redaction_audit (case_id, policy_version, span_count, total_redactions, source, ts) VALUES (%s, %s, %s::jsonb, %s, %s, %s)",
                (
                    record.case_id,
                    record.policy_version,
                    json.dumps(record.span_count),
                    record.total_redactions,
                    record.source,
                    record.ts,
                ),
            )

    def query_audit(self, case_id: str) -> list[dict[str, Any]]:
        with self._conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM redaction_audit WHERE case_id = %s ORDER BY ts DESC", (case_id,))
            return [dict(r) for r in cur.fetchall()]

    def insert_run(self, phase: str, track: str, config: str, provider: str | None, args: dict[str, Any]) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO run_metadata (phase, track, config, provider, args) VALUES (%s, %s, %s, %s, %s::jsonb) RETURNING run_id",
                (phase, track, config, provider, json.dumps(args)),
            )
            return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()


class DockerS3Client(S3Like):
    """boto3 S3 client pointed at localstack (MinIO-compatible)."""

    def __init__(
        self,
        endpoint_url: str = "http://localhost:4566",
        access_key: str = "op5",
        secret_key: str = "op5",
        region: str = "us-east-1",
    ) -> None:
        import boto3

        self._boto3 = boto3
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def put(self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream", metadata: dict[str, str] | None = None) -> None:
        self.ensure_bucket(bucket)
        kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key, "Body": data, "ContentType": content_type}
        if metadata:
            kwargs["Metadata"] = metadata
        self._s3.put_object(**kwargs)

    def get(self, bucket: str, key: str) -> bytes | None:
        try:
            resp = self._s3.get_object(Bucket=bucket, Key=key)
            return resp["Body"].read()
        except Exception as e:
            logger.debug("S3 get miss: %s/%s: %s", bucket, key, e)
            return None

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        try:
            resp = self._s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            return [o["Key"] for o in resp.get("Contents", [])]
        except Exception:
            return []

    def ensure_bucket(self, bucket: str) -> None:
        try:
            self._s3.head_bucket(Bucket=bucket)
        except Exception:
            self._s3.create_bucket(Bucket=bucket)


class DockerRedisClient(RedisLike):
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0) -> None:
        import redis as redis_lib

        self._r = redis_lib.Redis(host=host, port=port, db=db, decode_responses=False)

    def get(self, key: str) -> bytes | None:
        return self._r.get(key)

    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        if ttl_seconds:
            self._r.setex(key, ttl_seconds, value)
        else:
            self._r.set(key, value)

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def rate_limit_check(self, key: str, max_per_minute: int) -> bool:
        bucket_key = f"ratelimit:{key}"
        count = self._r.incr(bucket_key)
        if count == 1:
            self._r.expire(bucket_key, 60)
        return count <= max_per_minute


class DockerStorage(StorageClient):
    def __init__(self) -> None:
        self._chroma: ChromaLike | None = None
        self._postgres: PostgresLike | None = None
        self._s3: S3Like | None = None
        self._redis: RedisLike | None = None

    def chroma(self) -> ChromaLike:
        if self._chroma is None:
            self._chroma = DockerChromaClient(host=_env("CHROMA_HOST", "localhost"), port=int(_env("CHROMA_PORT", "8000")))
        return self._chroma

    def postgres(self) -> PostgresLike:
        if self._postgres is None:
            self._postgres = DockerPostgresClient(
                host=_env("POSTGRES_HOST", "localhost"),
                port=int(_env("POSTGRES_PORT", "5432")),
                user=_env("POSTGRES_USER", "op5"),
                password=_env("POSTGRES_PASSWORD", "op5"),
                dbname=_env("POSTGRES_DB", "op5"),
            )
        return self._postgres

    def s3(self) -> S3Like:
        if self._s3 is None:
            self._s3 = DockerS3Client(
                endpoint_url=_env("S3_ENDPOINT", "http://localhost:4566"),
                access_key=_env("AWS_ACCESS_KEY_ID", "op5"),
                secret_key=_env("AWS_SECRET_ACCESS_KEY", "op5"),
                region=_env("AWS_DEFAULT_REGION", "us-east-1"),
            )
        return self._s3

    def redis(self) -> RedisLike:
        if self._redis is None:
            self._redis = DockerRedisClient(host=_env("REDIS_HOST", "localhost"), port=int(_env("REDIS_PORT", "6379")))
        return self._redis

    def backend_name(self) -> str:
        return "docker"
