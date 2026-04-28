# RAG 设计

## 目标

系统从 `data/手册/*.txt` 构建文本向量索引，在回答用户问题前检索相关手册片段，并把检索证据交给客服大模型生成最终回复。

## Embedding 方案

默认使用 Hugging Face `Qwen/Qwen3-Embedding-0.6B`：

```env
EMBEDDING_BACKEND=sentence_transformers
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_QUERY_PROMPT_NAME=query
EMBEDDING_MODEL_DIR=./storage/models
```

实现位于 `src/kefu_agent/rag.py`：

- `SentenceTransformerEmbeddings` 通过 `sentence-transformers` 加载模型。
- 模型缓存目录由 `EMBEDDING_MODEL_DIR` 指定，本地已有缓存时由 Hugging Face/sentence-transformers 复用，不重复下载。
- query 侧使用 `prompt_name="query"`，符合 Qwen3 Embedding 的推荐用法。
- document 侧不加 prompt。
- 向量会归一化后写入索引。
- 文本切块使用 LangChain 的 `RecursiveCharacterTextSplitter`。
- 检索打分使用 `numpy` 向量化 cosine similarity。
- 首次构建索引会下载并加载 0.6B 模型。

可选后端：

- `EMBEDDING_BACKEND=sentence_transformers`：默认 Hugging Face 本地模型。
- `EMBEDDING_BACKEND=openai`：OpenAI 兼容 embedding 接口。
- `EMBEDDING_BACKEND=hash`：轻量本地哈希向量，只用于离线快速测试。

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

`index.jsonl` 每行是一个 chunk，包含文本、来源、图片 ID 和向量。
`index_meta.json` 记录当前 embedding backend、model 和 query prompt。配置变化时，检索流程会自动重建索引，避免复用旧模型生成的向量。

## 检索流程

1. 如果索引不存在或 metadata 不匹配，自动构建索引。
2. 使用当前 embedding 后端将用户问题转为 query vector。
3. 读取 `index.jsonl` 中的 chunk vectors。
4. 计算 cosine similarity。
5. 按分数从高到低排序。
6. 返回 `.env` 中 `TOP_K` 指定数量的 chunk。

## 图片引用

手册文本中的 `<PIC>` 会与图片 ID 顺序关联。检索命中 chunk 后，`image_ids` 会进入模型上下文；生成回答时，模型会在相关位置保留 `<PIC>` 标记。

## 后续优化

- 引入 FAISS、Chroma 或 Milvus 替代 JSONL 全量扫描。
- 加入 reranker。
- 对插图增加 CLIP 或多模态 embedding。
