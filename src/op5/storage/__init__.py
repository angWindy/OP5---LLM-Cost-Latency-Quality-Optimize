"""Storage clients for Phase 03.

Wraps ChromaDB (vector store), Postgres (audit), S3-compatible object storage
(localstack in dev, MinIO/AWS in prod), and Redis (cache). Each client exposes
the same interface for both Docker and in-process backends so the rest of the
codebase can swap freely via STORAGE_BACKEND env var.
"""

from op5.storage.base import (
    AuditRecord,
    Blob,
    EmbeddingItem,
    StorageClient,
    StorageFactory,
    STORAGE,
)

__all__ = [
    "AuditRecord",
    "Blob",
    "EmbeddingItem",
    "StorageClient",
    "StorageFactory",
    "STORAGE",
]
