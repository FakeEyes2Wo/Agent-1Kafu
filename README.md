# Kefu Agent

Minimal LangChain + LangGraph customer-service agent for the competition workflow.

## Setup

```powershell
uv sync
Copy-Item .env.example .env
```

Edit `.env` with your OpenAI-compatible chat API settings and `KAFU_API_TOKEN`.
Retrieval now uses Hugging Face `Qwen/Qwen3-Embedding-0.6B` through
`sentence-transformers` by default:

```env
EMBEDDING_BACKEND=sentence_transformers
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_QUERY_PROMPT_NAME=query
EMBEDDING_MODEL_DIR=./storage/models
```

The first index build will download/load the 0.6B embedding model into
`EMBEDDING_MODEL_DIR`; later runs reuse the local cache. If you change the
embedding backend, model, or query prompt, the vector index metadata will no
longer match and the index is rebuilt automatically.

## Build Index

```powershell
uv run python scripts/build_index.py
```

## Run API

```powershell
uv run python scripts/run_api.py
```

Endpoint:

```http
POST /chat
Authorization: Bearer change-me
Content-Type: application/json
```

Request body:

```json
{
  "question": "I want to replace the band. Are other sizes available?",
  "images": [],
  "session_id": "kf_session_001",
  "stream": false
}
```

## Generate Submission

```powershell
uv run python scripts/generate_submission.py
```

The script reads every row in `data/question_public.csv` and writes
`submission.csv` in the project root using the required `id,ret` format.

## Test

```powershell
uv run pytest -q
```
