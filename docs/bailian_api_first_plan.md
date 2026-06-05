# 百炼 API 优先应用方案

## 目标

在不把大模型本地化作为前提的情况下，完成赛题 `/chat` 客服智能体：

- 大模型生成、视觉理解、文本向量优先走百炼 OpenAI-compatible API。
- BM25 / sparse lexical 检索在应用本地完成，兜住型号、部件名、按钮名、指示灯等精确 token。
- 本地只保留知识库解析、索引、RAG 编排、图片引用后处理和 FastAPI 服务。

## 默认模型与服务

| 模块 | 默认选择 | 部署方式 |
| --- | --- | --- |
| 文本/多模态回答 | `qwen3.7-plus-2026-05-26` | 百炼 API |
| 用户上传图片理解 | `qwen3.7-plus-2026-05-26` | 百炼 API |
| Dense embedding | `text-embedding-v4` | 百炼 API |
| BM25 / sparse | 内置 BM25 | 本地应用 |
| Rerank | 默认关闭；可选 `qwen3-vl-rerank` | 百炼 API |

## 运行链路

```text
POST /chat
  -> summarize_images: 有图时调用 qwen3.7-plus
  -> retrieve_context:
       1. 本地解析后的 manual chunks
       2. BM25 精确召回
       3. 中英混合 query 同时查中文和英文手册
       4. 如有 BAILIAN_API_KEY / DASHSCOPE_API_KEY，叠加 text-embedding-v4 dense 召回
       5. image-neighbor pseudo chunks 本地 lexical 召回
       6. RRF 融合、去重、截断
  -> generate_answer: qwen3.7-plus evidence-grounded 生成
  -> check_answer: qwen3.7-plus 质检改写
  -> PIC formatter: 只允许输出 context 中出现过的图片 ID
```

## 服务器准备

必须准备：

1. `BAILIAN_API_KEY` / `DASHSCOPE_API_KEY`，并确认 `https://dashscope.aliyuncs.com/compatible-mode/v1` 可访问。
2. `data/question_public.csv`、`data/submission_example.csv`、`data/KownledgeBase/手册`。
3. 构建索引：`python scripts/build_index.py`。未配置 API key 时只构建 BM25-only 索引；配置 API key 后会同时写入 `text-embedding-v4` dense 向量。
4. 启动服务：`python scripts/run_api.py`。

可选准备：

- 如果要 rerank，可设置 `RERANK_ENABLED=true`、`RERANK_BACKEND=bailian`、`RERANK_MODEL=qwen3-vl-rerank`，走百炼排序模型 API。
- 如果要降低视觉预处理成本，可离线批量生成图片 caption/OCR 缓存。
- 如果要做消融实验，可比较 `BM25 only`、`BM25 + text-embedding-v4`、`BM25 + dense + visual`。
- 百炼 `text-embedding-v4` 单次最多 10 条文本，保持 `EMBEDDING_BATCH_SIZE=10`。

## GPU / 服务器迁移清单

1. 安装依赖：`uv sync`，或按 `pyproject.toml` 创建 Python 3.11+ 环境。
2. 复制 `.env.example` 为 `.env`，至少填写 `BAILIAN_API_KEY` / `DASHSCOPE_API_KEY` 和 `KAFU_API_TOKEN`。
3. 确认数据路径与 `.env` 一致：`data/KownledgeBase/手册`、`data/KownledgeBase/手册/插图`、`data/question_public.csv`、`data/submission_example.csv`。
4. 先跑 `python scripts/build_index.py`。如果已配置 `BAILIAN_API_KEY` / `DASHSCOPE_API_KEY`，会在 BM25 索引之外补 dense 向量。
5. 需要 API 服务时跑 `python scripts/run_api.py`；需要生成提交时跑 `python scripts/generate_submission.py`。
6. 若启用 rerank，先用少量问题确认延迟和成本，再切到全量生成。

## vLLM / 本地部署边界

赛题要求提交可调用的 `/chat` 智能体 API，并未要求使用 vLLM。当前方案把
`qwen3.7-plus`、`text-embedding-v4` 和可选 `qwen3-vl-rerank` 都视为百炼
托管 API 能力；本地只运行检索与应用编排。只有以下情况才需要 vLLM 或类似
本地推理框架：

- 下载开源权重并在自己的 GPU 上提供 OpenAI-compatible 推理接口。
- 部署自己微调后的模型权重。
- 出于成本、限流或离线要求，不使用百炼托管模型。

如果继续采用百炼 API-first，GPU 服务器只需要跑 Python 应用、索引构建和
可选实验脚本，不需要准备 vLLM。

## 当前代码落点

- `src/kefu_agent/config.py`：默认切到百炼兼容 API、`text-embedding-v4`、`RAG_BACKEND=hybrid`。
- `src/kefu_agent/rag/`：重建 RAG 包，包含解析、BM25、dense 召回、RRF、visual pseudo chunk、PIC 后处理。
- `.env.example`：提供百炼 API 优先配置。
- `README.md`：说明无 API key 时可构建 BM25-only 索引用于本地轻量检查。
