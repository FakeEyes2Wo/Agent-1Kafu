# Kefu Agent

Minimal LangChain + LangGraph customer-service agent for the competition workflow.

## Setup

```powershell
uv sync
Copy-Item .env.example .env
```

For a conda/pip environment, install the default Bailian API-first stack with:

```powershell
python -m pip install -e .
```

Install extras only when needed:

```powershell
python -m pip install -e ".[dev]"
python -m pip install -e ".[local]"
```

`local` installs the optional local-model compatibility path
(`sentence-transformers` and `llama-index`). It is not needed for the default
Bailian API + local BM25/sparse workflow.

Edit `.env` with your Bailian/DashScope API key and `KAFU_API_TOKEN`.
The default stack is Bailian-API-first for model calls, with local BM25/sparse
retrieval kept in the application:

```env
BAILIAN_API_KEY=your-dashscope-api-key
DASHSCOPE_API_KEY=
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_API_BACKEND=langchain
CHAT_MODEL=qwen3.7-plus-2026-05-26
CHAT_ENABLE_THINKING=false
CHAT_MAX_TOKENS=450
CHAT_THINKING_BUDGET=0
OPENAI_RESPONSES_MODEL=gpt-5.5
OPENAI_RESPONSES_REASONING_EFFORT=none
VISION_MODEL=qwen3.7-plus-2026-05-26
VISION_MAX_IMAGES=6
VISION_MAX_TOKENS=800
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-v4
RAG_BACKEND=hybrid
RERANK_ENABLED=false
VISUAL_CONTEXT_WINDOW=360
```

`qwen3.7-plus` supports thinking mode through Bailian/DashScope. The default
keeps `CHAT_ENABLE_THINKING=false` for faster, more predictable batch
submission generation. Set it to `true` when you want higher-reasoning answers
and can accept extra tokens and latency. For thinking ablations, increase
`CHAT_MAX_TOKENS` and optionally set `CHAT_THINKING_BUDGET`; otherwise the
thinking trace can consume the output budget and leave the final answer empty.

To run OpenAI Responses API ablations instead of the default Bailian chat path,
set `CHAT_API_BACKEND=openai_responses`, `OPENAI_API_KEY`, and
`OPENAI_RESPONSES_MODEL=gpt-5.5`. Reasoning is off by default with
`OPENAI_RESPONSES_REASONING_EFFORT=none`; set it to `low`, `medium`, or `high`
only for explicit reasoning ablations.

Multimodal budgets are configurable. `VISION_MAX_IMAGES` controls how many
uploaded user images are summarized per request, `VISION_MAX_TOKENS` controls
the vision-summary output budget, and `VISUAL_CONTEXT_WINDOW` controls how much
manual text around each `<PIC>` is used for image-oriented retrieval chunks.
Changing `VISUAL_CONTEXT_WINDOW` invalidates retrieval context caches.

Index building always creates a local BM25/sparse index over manual chunks.
If `BAILIAN_API_KEY` or `DASHSCOPE_API_KEY` is set, dense vectors are added
with `text-embedding-v4`; without an API key the code still builds a BM25-only
index for local checks.
If you change retrieval or embedding settings, index metadata no longer matches
and the index is rebuilt automatically.
Keep `EMBEDDING_BATCH_SIZE=10` for Bailian `text-embedding-v4`; the service
rejects larger embedding batches.
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
uv sync --extra dev
uv run pytest -q
```
