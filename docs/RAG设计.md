# RAG 设计

> 当前实现已切换为“百炼 API + 本地 BM25/sparse”的 hybrid RAG：默认用百炼
> `text-embedding-v4` 做可选 dense 向量，BM25/sparse、image-neighbor pseudo
> chunks、RRF 融合和 PIC 后处理都在本地完成。最新落地说明见
> `docs/bailian_api_first_plan.md`。

## 目标

系统从 `data/KownledgeBase/手册/*.txt` 构建本地检索索引，在回答用户问题前检索相关手册片段和配图证据，并把证据交给客服大模型生成最终回复。

## Embedding 方案

默认使用百炼 OpenAI-compatible embedding API；无 API key 时自动降级为 BM25-only 本地索引：

```env
BAILIAN_API_KEY=your-dashscope-api-key
DASHSCOPE_API_KEY=
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_BATCH_SIZE=64
```

实现位于 `src/kefu_agent/rag/`：

- `parsing.py` 解析手册、分块并绑定 `<PIC>` 图片 ID。
- `retrieval.py` 负责 BM25/sparse、可选 dense、RRF 融合和去重。
- `visual.py` 将图片邻近文本构造成 image pseudo chunks，补充配图召回。
- `formatting.py` 和 `images.py` 负责上下文与最终答案的 PIC 后处理。

可选后端：

- `EMBEDDING_BACKEND=openai` / `bailian` / `dashscope`：百炼或其他 OpenAI-compatible embedding 接口。
- `EMBEDDING_BACKEND=none`：仅本地 BM25/sparse。
- `EMBEDDING_BACKEND=hash`：轻量本地哈希向量，只用于离线快速测试。
- `EMBEDDING_BACKEND=sentence_transformers`：仅在明确要本地开源 embedding 时使用。

## 索引存储

索引构建脚本：

```powershell
uv run python scripts/build_index.py
```

输出文件：

```text
storage/vectorstore/index.jsonl
storage/vectorstore/index_meta.json
```

`index.jsonl` 每行是一个 chunk，包含文本、来源、图片 ID、语言标签和可选向量。`index_meta.json` 记录当前 embedding backend、model、chunk 参数、RAG 版本、PIC 版本和是否启用 dense vectors。配置变化时，检索流程会自动重建索引，避免复用旧索引。

## 检索流程

1. 如果索引不存在或 metadata 不匹配，自动构建索引。
2. 根据问题语言选择中文、英文或中英双路手册召回。
3. 本地 BM25/sparse 召回型号、按钮、故障灯等精确 token。
4. 如果配置了 `BAILIAN_API_KEY` / `DASHSCOPE_API_KEY`，调用 `text-embedding-v4` 做 dense 召回。
5. 如果 `VISUAL_RETRIEVER` 未关闭，补充 image-neighbor pseudo chunks。
6. 使用 RRF 融合、去重、截断，返回 `.env` 中 `TOP_K` 指定数量的 chunk。

## 图片引用

手册文本中的 `<PIC>` 会与图片 ID 顺序关联。检索命中 chunk 后，模型上下文会把占位符展开成 `<PIC> 图片ID </PIC>`，并额外提供 `可用图片：["..."]` 供模型选择。模型回答需要引用图片时输出同样的带名标记，例如 `<PIC> drill0_04 </PIC>`；提交前系统会后处理为正文中的裸 `<PIC>`，并按出现顺序在末尾追加图片列表，例如：`回答正文 <PIC>,["drill0_04"]`。

## 后续优化

- 服务器阶段启用 `RERANK_BACKEND=bailian` + `qwen3-vl-rerank` 并做消融。
- 对插图增加离线 caption/OCR 缓存。
- 如果数据规模扩大，再考虑 FAISS、Chroma 或 Milvus；当前数据量优先保持本地 JSONL/BM25 简洁实现。
