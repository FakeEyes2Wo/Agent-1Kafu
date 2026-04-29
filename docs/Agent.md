# Agent 设计

核心流程位于 `src/kefu_agent/graph.py`，由 LangGraph 编排。

## State

```python
class AgentState(TypedDict, total=False):
    question: str
    images: list[str]
    session_id: str
    image_summary: str
    contexts: str
    answer: str
```

当前链路不读取、不拼接、不保存持久化历史。`session_id` 仅保留为接口返回和请求追踪标识，不参与上下文记忆。

## Flow

```text
START
  -> summarize_images
  -> retrieve_context
  -> generate_answer
  -> check_answer
  -> END
```

## Nodes

`summarize_images`

- 有图片时调用 `VISION_MODEL` 提取订单、物流、故障或商品信息。
- vision 接口地址由 `VISION_MODEL_URL` 指定，未配置时回退到 `OPENAI_BASE_URL`。

`retrieve_context`

- 使用当前 embedding 后端检索 RAG 证据。
- 默认 embedding 是 Hugging Face `Qwen/Qwen3-Embedding-0.6B`。

`generate_answer`

- 调用 `CHAT_MODEL` 基于图片摘要、检索证据和通用客服政策生成候选回答。
- prompt 明确要求中英双语响应规则：英文提问用英文回答，中文提问用中文回答。
- 一行输入包含多个子问题时，视为当前请求内部的多轮对话：先回答第一个子问题，再回答第二个子问题，后续回答承接前面已回答的上下文。
- 回答在清晰、完整、不漏答的基础上尽可能简洁。
- 不再使用模板 fallback；模型失败或空回答会直接报错。

`check_answer`

- 调用同一个 `CHAT_MODEL` 一次性完成 reflection 和 rewrite。
- 检查候选回答是否完整、无编造、是否保留 `<PIC>` 标记。
- 同步执行语言规则检查：英文问题保持英文回复，中文问题保持中文回复。
- 同步检查多子问题是否按原始顺序逐个回应，且表达是否足够简洁。
- 不新增事实、承诺、金额、时效或商品参数。

## Submission

`scripts/generate_submission.py` 对 `data/question_public.csv` 的每一行单独生成答案：

```python
answer_question(question, session_id=f"submission_{qid}", contexts=contexts_by_id[qid])
```

因此公开题生成不会读取旧的 `submission_*` 历史，也不会写入新的 `submission_*` 会话文件。断点续传只依赖 `submission.csv` 中已有的非空 `ret`。
