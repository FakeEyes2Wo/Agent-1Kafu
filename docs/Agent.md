# Agent 设计

核心流程位于 `src/kefu_agent/graph.py`，由 LangGraph 编排。

## State

```python
class AgentState(TypedDict, total=False):
    question: str
    images: list[str]
    session_id: str
    persist_history: bool
    history: list[dict[str, str]]
    image_summary: str
    contexts: str
    answer: str
```

`persist_history` 控制是否读取和保存会话历史。API 默认开启；`generate_submission.py` 会关闭它，避免 `question_public.csv` 的不同题目互相污染历史上下文。

## Flow

```text
START
  -> load_context
  -> summarize_images
  -> retrieve_context
  -> generate_answer
  -> check_answer
  -> rewrite_answer
  -> save_memory
  -> END
```

## Nodes

`load_context`

- `persist_history=True` 时读取 `storage/sessions/{session_id}.json`。
- `persist_history=False` 时使用空历史。

`summarize_images`

- 有图片时调用 `VISION_MODEL` 提取订单、物流、故障或商品信息。
- vision 接口地址由 `VISION_MODEL_URL` 指定，未配置时回退到 `OPENAI_BASE_URL`。

`retrieve_context`

- 使用当前 embedding 后端检索 RAG 证据。
- 默认 embedding 是 Hugging Face `Qwen/Qwen3-Embedding-0.6B`。

`generate_answer`

- 调用 `CHAT_MODEL` 基于历史、图片摘要、检索证据和通用客服政策生成候选回答。
- 不再使用模板 fallback；模型失败或空回答会直接报错。

`check_answer`

- 调用同一个 `CHAT_MODEL` 做 reflection。
- 检查候选回答是否完整、礼貌、无编造、是否保留 `<PIC>` 标记。
- 合格则原样返回；不合格则输出修正后的最终客服回复。

`rewrite_answer`

- 调用同一个 `CHAT_MODEL` 对已核验回答做语气改写。
- 只让回复更自然、更有人情味，不新增事实、承诺、金额、时效或商品参数。

`save_memory`

- `persist_history=True` 时保存本轮问答。
- `persist_history=False` 时跳过保存。

## Submission

`scripts/generate_submission.py` 调用：

```python
answer_question(question, session_id=f"submission_{qid}", persist_history=False)
```

因此公开题生成不会读取旧的 `submission_*` 历史，也不会再写入新的 `submission_*` 会话文件。断点续传只依赖 `submission.csv` 中已有的非空 `ret`。
