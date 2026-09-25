-- Phase 03 Postgres initial schema
-- Runs on first container start via /docker-entrypoint-initdb.d/

CREATE TABLE IF NOT EXISTS redaction_audit (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    case_id         TEXT NOT NULL,
    policy_version  TEXT NOT NULL,
    span_count      JSONB NOT NULL,
    total_redactions INTEGER NOT NULL,
    source          TEXT NOT NULL  -- 'tesseract' | 'paddleocr' | 'easyocr' | 'manual'
);

CREATE INDEX IF NOT EXISTS idx_redaction_audit_case ON redaction_audit(case_id);
CREATE INDEX IF NOT EXISTS idx_redaction_audit_ts ON redaction_audit(ts DESC);

CREATE TABLE IF NOT EXISTS run_metadata (
    run_id      BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    phase       TEXT NOT NULL,
    track       TEXT,
    config      TEXT,
    provider    TEXT,
    args        JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_metadata_ts ON run_metadata(ts DESC);
CREATE INDEX IF NOT EXISTS idx_run_metadata_track ON run_metadata(track);
