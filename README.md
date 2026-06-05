# Kefu Agent

Minimal LangChain + LangGraph customer-service agent for the competition workflow.

## Setup

```powershell
uv sync
Copy-Item .env.example .env
```

Edit `.env` with your Bailian/DashScope API key and `KAFU_API_TOKEN`.
The default stack is Bailian-API-first for model calls, with local BM25/sparse
retrieval kept in the application:

```env
BAILIAN_API_KEY=your-dashscope-api-key
DASHSCOPE_API_KEY=
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen3.7-plus-2026-05-26
VISION_MODEL=qwen3.7-plus-2026-05-26
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-v4
RAG_BACKEND=hybrid
RERANK_ENABLED=false
```

Index building always creates a local BM25/sparse index over manual chunks.
If `BAILIAN_API_KEY` or `DASHSCOPE_API_KEY` is set, dense vectors are added
with `text-embedding-v4`; without an API key the code still builds a BM25-only
index for local checks.
If you change retrieval or embedding settings, index metadata no longer matches
and the index is rebuilt automatically.
Use `EMBEDDING_BATCH_SIZE` to reduce Bailian embedding batch size if the server
reports request-size limits.
If server-side ablation shows reranking helps, enable Bailian rerank with
`RERANK_ENABLED=true`, `RERANK_BACKEND=bailian`, and
`RERANK_MODEL=qwen3-vl-rerank`.

vLLM is not required for this default path. It is only needed if you choose to
self-host open-source or fine-tuned weights instead of calling Bailian-hosted
models. See `docs/bailian_api_first_plan.md` and
`docs/bailian_model_market_notes.md` for the deployment boundary.

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
Each row is answered independently; no persistent conversation history is read
or saved during API calls or submission generation. Retrieval contexts are
cached in `storage/vectorstore/contexts_cache.json`, and model-call traces
(draft answer, checked answer, final answer, token usage when returned by the
provider, and cache signatures) are appended to
`storage/api_cache/answers_cache.jsonl` for resumable server runs.

## Test

```powershell
uv run pytest -q
```
