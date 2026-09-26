# OP5 Streamlit UI

A 4-page Streamlit UI that calls the OP5 FastAPI service for end-to-end demos.

## Pages

| Page                       | Source file                            | What it does                                              |
| -------------------------- | -------------------------------------- | --------------------------------------------------------- |
| Home                       | `src/op5/ui/Home.py`                   | Service banner, health sidebar, run instructions          |
| Track 1 Demo               | `pages/1_Track1_Demo.py`               | Upload PDF + pick config + render extracted fields        |
| Track 2 RAG                | `pages/2_Track2_RAG.py`                | Pick case + ask question, render answer + citations       |
| Pareto / Inspect           | `pages/3_Pareto.py`                    | Inspect endpoint + Phase 03 PNG Pareto plots              |

## Prereqs

1. The FastAPI service must be running. See `scripts/api/README.md`.
2. The OP5 SYNTH corpus must be present at `data/processed/phase03_synth_contracts.jsonl`.

## Quickstart

```bash
conda activate vsf
pip install streamlit httpx

# (optional) override if the API runs on a different host
export OP5_API_URL=http://localhost:8000

bash scripts/ui/run.sh          # uses scripts/ui/run.sh
# or directly:
streamlit run src/op5/ui/Home.py --server.address 0.0.0.0 --server.port 8501
```

Open <http://localhost:8501>.

## Environment variables

| Var            | Default                | Notes                                  |
| -------------- | ---------------------- | -------------------------------------- |
| `OP5_API_URL`  | `http://localhost:8000`| Base URL of the FastAPI service        |
| `STREAMLIT_*`  | (Streamlit defaults)   | e.g. `STREAMLIT_SERVER_PORT=8502`      |

## No business logic here

The UI is intentionally thin: every widget calls the FastAPI service via
`OP5Api` in `src/op5/ui/lib/api_client.py`. There is no database access
or model code inside the Streamlit package — that lives in `src/op5/`.

## Tip: sidebar health check

If the sidebar shows "Cannot reach http://localhost:8000", the FastAPI
service isn't running or is on a different port. Start it with:
```bash
uvicorn scripts.api.serve:app --host 0.0.0.0 --port 8000 --reload
```
