# 百炼模型广场调用边界

## 浏览器确认结果

基于当前浏览器中百炼模型广场页面：

| 能力 | 页面模型 | 模型 ID | 是否可直接 API |
| --- | --- | --- | --- |
| 文本 + 视觉理解 + 智能体生成 | Qwen3.7-Plus | `qwen3.7-plus-2026-05-26` | 是，页面有 `API 参考` |
| 纯文本强模型 | Qwen3.7-Max | `qwen3.7-max-2026-05-20` | 是，页面有 `API 参考` |
| 文本向量 | Qwen-Embedding | `text-embedding-v4` | 是，页面有 `API 参考` |
| RAG 重排 | Qwen-Rerank | `qwen3-vl-rerank` | 是，页面有 `API 参考` |
| 其他百炼模型广场模型 | 例如 DeepSeek、Kimi、MiniMax、Qwen 开源模型托管版 | 以页面模型 ID 为准 | 只要页面提供 `API 参考`，就按托管 API 调用 |

## 对本赛题的选择

- 必选：`qwen3.7-plus-2026-05-26` 用于回答生成和用户上传图片理解。
- 必选：本地 BM25 / sparse 检索，负责型号、按钮、故障灯、配件名等精确召回。
- 推荐：`text-embedding-v4` 作为 dense 召回补充；无 API key 时降级为 BM25-only。
- 可选：`qwen3-vl-rerank` 可以直接调 API；当前代码支持 `RERANK_BACKEND=bailian`，但默认关闭，等服务器做消融后再启用。

## 必须本地部署的情况

百炼模型广场中有 `API 参考` 的模型不需要本地 vLLM。需要本地部署的只有：

- 使用未在百炼托管 API 中开放的开源权重。
- 使用自己微调后的私有权重，并且没有通过百炼“模型部署”发布为 API。
- 要求离线推理，或成本/限流策略决定不走百炼托管 API。

因此当前应用方案不准备 vLLM；只保留可替换的 OpenAI-compatible `base_url`，以后如果
改成本地 vLLM，也只需要把 `BAILIAN_BASE_URL` / `OPENAI_BASE_URL` 指向本地服务并替换模型名。
