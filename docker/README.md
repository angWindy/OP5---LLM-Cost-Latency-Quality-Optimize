# Phase 03 — Docker Storage Stack

> English-only README per AGENTS.md. Vietnamese docs live in `doc/`.

The stack runs four services via `docker/docker-compose.yml`:

| Service   | Image                                       | Host port        | Purpose                                                         |
| --------- | ------------------------------------------- | ---------------- | --------------------------------------------------------------- |
| chromadb  | `chromadb/chroma:latest`                    | 8000             | Vector store for RAG embeddings (Docker or fallback Persistent) |
| postgres  | `postgres:16-alpine`                        | 5432             | Metadata + redaction audit log                                  |
| localstack| `localstack/localstack:3.0`                 | 4566 (S3 + Dynamo + SQS) | S3-compatible object storage for OCR scans + redacted dumps |
| redis     | `redis:7-alpine`                            | 6379             | Intermediate cache (OCR result, embedding cache, rate-limit)    |

We use **LocalStack** instead of MinIO on this host because the MinIO image
is currently restricted from the Docker registry. LocalStack's S3 emulation
is wire-compatible with `boto3`; the only difference is the endpoint URL.

## Up / down

```bash
# Bring all four services up.
docker compose -f docker/docker-compose.yml up -d

# Verify each one.
docker compose -f docker/docker-compose.yml ps
curl http://localhost:8000/api/v1/heartbeat        # ChromaDB
psql postgresql://op5:op5@localhost:5432/op5        # Postgres
awslocal s3 ls --endpoint-url http://localhost:4566 # LocalStack S3
redis-cli -h localhost ping                        # Redis

# Tear down.
docker compose -f docker/docker-compose.yml down
```

## Volumes

- `chroma_data`   → `/chroma/chroma` (ChromaDB persistence)
- `pg_data`       → `/var/lib/postgresql/data` (Postgres data)
- `localstack_data` → `/var/lib/localstack` (S3 bucket state)
- `redis_data`    → `/data` (Redis AOF + RDB)

## Environment (docker/.env.example)

```bash
CHROMA_PORT=8000
POSTGRES_USER=op5
POSTGRES_PASSWORD=op5
POSTGRES_DB=op5
LOCALSTACK_HOST=localhost
LOCALSTACK_PORT=4566
REDIS_PORT=6379
```

## Buckets and tables provisioned

- **S3 buckets (LocalStack):**
  - `ocr-scanned/`    — mirrors `data/Scan/scan_phase03/` PDFs.
  - `ocr-results/`    — 1 JSON per provider per case.
  - `redacted-text/`  — redacted text dumps.

- **Postgres tables:**
  - `redaction_audit(case_id, policy_version, span_count_per_type JSONB, total_redactions INT, ts TIMESTAMPTZ)`.
  - `run_metadata(run_id, phase, track, config, ts, args JSONB)`.

- **Redis keys:**
  - `ocr:{case_id}:{provider}` → cached OCRResult JSON (TTL 1h).
  - `embedding:{model}:{chunk_hash}` → cached embedding vector (TTL 24h).
  - `ratelimit:gemini:{key_id}` → token bucket count (TTL 60s).

## In-process fallback (no Docker on dev)

```bash
export STORAGE_BACKEND=inprocess
# Chroma   → ./data/cache/chroma
# Postgres → ./data/cache/audit.sqlite
# S3       → ./data/cache/minio/
# Redis    → in-memory dict (process-local, không persist)
```

The `src/op5/storage/` abstraction (`StorageFactory.from_env()`) reads this env
var and dispatches to the correct backend — no code changes needed in the
runners or RAG layer.

## Security note

Default credentials in `docker/.env.example` are dev-only. For any non-local
deployment, rotate credentials and put them in a secret manager. Buckets are
not world-readable; LocalStack binds to localhost by default.
