# 配置参考

所有配置通过 `.env` 文件设置，字段名大小写不敏感。

## API / 认证

| 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `KAFU_API_TOKEN` | `str` | `change-me` | FastAPI `/chat` 接口鉴权 token |
| `OPENAI_API_KEY` | `str` | `""` | LLM / DashScope embedding / DashScope rerank 共用 API key |
| `OPENAI_BASE_URL` | `str` | `https://api.openai.com/v1` | LLM API 地址（DashScope 用兼容模式时为 `https://dashscope.aliyuncs.com/compatible-mode/v1`） |

## LLM

| 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `CHAT_MODEL` | `str` | `gpt-4o-mini` | 对话模型名 |
| `VISION_MODEL_URL` | `str` | `""` | 视觉模型 API 地址，空则复用 `OPENAI_BASE_URL` |
| `MODEL_TIMEOUT_SECONDS` | `float` | `60` | 模型调用超时（秒） |

## Embedding

| 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `EMBEDDING_MODEL` | `str` | `text-embedding-v3` | 模型名。`text-embedding-v*` 走 DashScope API；其他走 HuggingFace 本地加载 |
| `EMBEDDING_BACKEND` | `str` | `openai` | `"openai"` → DashScope(OpenAI 兼容)；`"sentence_transformers"` / `"huggingface"` / `"hf"` → 本地模型 |
| `EMBEDDING_QUERY_PROMPT_NAME` | `str` | `query` | HuggingFace 模型的 query prompt 名（DashScope 忽略） |
| `EMBEDDING_MODEL_DIR` | `path` | `./storage/models` | HuggingFace 模型缓存目录 |

### 支持的 DashScope Embedding 模型

| 模型 | 维度 |
|---|---|
| `text-embedding-v1` | 1536 |
| `text-embedding-v2` | 1536 |
| `text-embedding-v3` | 1024 |
| `text-embedding-v4` | 1024 |

通过环境变量 `DASHSCOPE_API_KEY` 也可单独设置 key（优先级低于 `OPENAI_API_KEY` 显式传入）。

## Rerank

| 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `RERANK_ENABLED` | `bool` | `False` | 是否启用重排序 |
| `RERANK_MODEL` | `str` | `qwen3-rerank` | 模型名 |
| `RERANK_BACKEND` | `str` | `local` | `"local"` → 本地 CrossEncoder；`"dashscope"` → 云端 API |
| `RERANK_TOP_N` | `int` | `8` | 重排序后保留数量 |

### Rerank 后端说明

| 后端 | 实现 | 适用模型 |
|---|---|---|
| `local` | `sentence_transformers.CrossEncoder` 本地加载 | `BAAI/bge-reranker-v2-m3` 等 HuggingFace 模型 |
| `dashscope` | `dashscope.TextReRank.call()` SDK 调用 | `qwen3-rerank`（100+ 语言、500 文档、4000 token/doc）、`gte-rerank-v2` |

DashScope rerank 的 API key 复用 `OPENAI_API_KEY`。后端选择由 `RERANK_BACKEND` 控制，模型名由 `RERANK_MODEL` 指定。

## 服务

| 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `APP_HOST` | `str` | `0.0.0.0` | FastAPI 监听地址 |
| `APP_PORT` | `int` | `8000` | FastAPI 监听端口 |

## 数据路径

| 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `DATA_DIR` | `path` | `./data` | 数据根目录 |
| `MANUAL_DIR` | `path` | `./data/手册` | 产品手册目录 |
| `IMAGE_DIR` | `path` | `./data/手册/插图` | 手册插图目录 |
| `VECTORSTORE_DIR` | `path` | `./storage/vectorstore` | 向量库存储目录（旧） |
| `LLAMAINDEX_DIR` | `path` | `./storage/llamaindex` | LlamaIndex 索引目录（旧） |
| `RAG_DATA_DIR` | `path` | `./storage/rag_check` | RAG 索引及缓存目录 |

`RAG_DATA_DIR` 下结构：
```
storage/rag_check/
├── index/
│   ├── faiss.index        # 向量索引
│   ├── bm25_index.pkl     # BM25 索引
│   ├── chunks.parquet     # 子块数据
│   └── sections.parquet   # 段落数据
├── image_specs.csv        # 图片描述
└── cache/                 # 构建缓存
```

## RAG 检索参数

| 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `RETRIEVAL_TOP_K` | `int` | `20` | 混合检索（Dense + BM25）各自召回数量 |
| `TOKEN_BUDGET` | `int` | `1024` | 检索注入 LLM 的最大 token 数 |
| `CHUNK_SIZE` | `int` | `256` | 索引构建时分块大小 |
| `CHUNK_OVERLAP` | `int` | `26` | 分块重叠量 |
| `TOP_K` | `int` | `8` | （旧参数，LlamaIndex 用） |
| `RAG_BACKEND` | `str` | `llamaindex` | （旧参数） |

## 视觉检索参数

| 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `VISUAL_RETRIEVER` | `str` | `lexical` | 视觉检索方式 |
| `VISUAL_TOP_K` | `int` | `8` | 视觉检索返回数量 |
