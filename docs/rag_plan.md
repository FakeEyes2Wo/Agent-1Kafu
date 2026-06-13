# RAG 检索管线设计

## 设计概要

两阶段架构：**离线构建索引**（chunk → embed → FAISS + BM25 双索引）→ **在线查询**（路由 → 混合召回 → 子块重排 → 按预算扩展父块 → 证据校验 → 结构化生成）。

核心原则：**小块搜索、大块补充**（Nainwani & Baban, 2025）。但具体参数——search chunk 大小、overlap、parent 预算、reranker top-k——全部作为实验项，不在设计阶段写死。

---

## 一、Chunk 切分

### 1.1 层级标题切分

手册文本天然具有 `#` ~ `####` 标题结构。切分利用这一结构，不做语义嵌入切分。

```python
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")

@dataclass
class HeadingPath:
    levels: list[tuple[int, str]]   # [(1, "安装电池"), (2, "装入/取出电池")]

    def to_text(self) -> str:
        return " > ".join(title for _, title in self.levels)
```

每个 block 记录完整 `heading_path`（从根标题到当前的路径），存入元数据并拼入 `embed_text`。

```python
def split_by_headings(text: str) -> list[tuple[HeadingPath, str]]:
    """按层级标题切分。首段（第一个 # 之前）heading_path 为空。"""

    blocks: list[tuple[HeadingPath, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def _flush() -> None:
        if buf:
            blocks.append((HeadingPath(list(stack)), "".join(buf)))
            buf.clear()

    for line in text.splitlines(keepends=True):
        m = HEADING_RE.match(line.lstrip())
        if m:
            _flush()
            level = len(m.group(1))
            title = m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        buf.append(line)

    _flush()
    return blocks
```

### 1.2 原子块保护

以下结构在切分时视为不可拆分的原子单元，不跨越 block 边界拆分：

- **图片及其上下文说明**：`<image id='N'>...</image>` 及紧随其前后 1 行文本
- **编号步骤组**：连续的 `1. 2. 3.` 或 `• • •` 或 `①②③` 等列表
- **安全警告块**：以 "⚠" / "警告" / "WARNING" / "注意" 起始，到下一标题或空行结束
- **表格/规格参数**：连续的 key-value 行或 `<table>` 标记块

```python
ATOMIC_PATTERNS = [
    ("image",   re.compile(r"(?:^.*\n)?<image[^>]*>.*?</image>(?:\n.*)?", re.DOTALL)),
    ("steps",   re.compile(r"(?:^(?:\d+[\.\、\)]|[•\-\*]|\$\\textcircled).*\n?){2,}", re.MULTILINE)),
    ("warning", re.compile(r"(?:⚠|警告|WARNING|注意|Caution).*?(?:\n\n|\n#|$)", re.DOTALL)),
]

def protect_atomic_blocks(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """将原子块替换为 __ATOMIC_{tag}_{N}__ 占位符。"""
    mapping: dict[str, tuple[str, str]] = {}
    ctr = itertools.count()

    for tag, pattern in ATOMIC_PATTERNS:
        text = pattern.sub(lambda m, t=tag: _stash(m, t, mapping, ctr), text)
    return text, mapping

def _stash(m: re.Match, tag: str, mapping: dict, ctr) -> str:
    pid = f"__ATOMIC_{tag}_{next(ctr)}__"
    mapping[pid] = (tag, m.group())
    return pid
```

### 1.3 Search Chunk 生成

```python
# 实验参数，不定值
SEARCH_CHUNK_SIZES = [128, 256, 384, 512]   # tokens
OVERLAP_RATIOS = [0, 0.1, 0.2]

def split_block_into_search_chunks(
    block_text: str,
    heading_path: HeadingPath,
    manual_name: str,
    block_index: int,
    chunk_size: int,
    overlap_ratio: float,
    tokenizer,
) -> list[SearchChunk]:
    """
    1. 保护原子块
    2. 基于 tokenizer 的 RecursiveCharacterTextSplitter 切分
    3. 还原原子块
    4. 为每个 search chunk 生成嵌入文本：manual_name + heading_path + chunk_text
    """
    protected, atom_map = protect_atomic_blocks(block_text)

    overlap_tokens = int(chunk_size * overlap_ratio)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=overlap_tokens,
        separators=["\n\n", "\n", "。", ".", "，", ",", " ", ""],
        keep_separator=True,
    )
    raw_chunks = splitter.split_text(protected)

    chunks = []
    for i, raw in enumerate(raw_chunks):
        text = restore_atomic_blocks(raw, atom_map)
        chunks.append(SearchChunk(
            chunk_id=f"{sha8(manual_name)}:{block_index}:{i}",
            manual_name=manual_name,
            heading_path=heading_path,
            block_index=block_index,
            chunk_index=i,
            content=text,
            embed_text=f"{manual_name}\n{heading_path.to_text()}\n{text}",
            image_ids=extract_image_ids(text),
            has_images=bool(IMAGE_TAG_RE.search(text)),
            is_atomic_block=(len(atom_map) == 1 and i == 0),
        ))
    return chunks
```

### 1.4 实现要点

- **Token 级切分**：`RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer)`，按 token 数而非字符数切分。
- **占位符膨胀补偿**：`__ATOMIC_N__` 占位符比原始原子块短，还原后 chunk 会膨胀。在 `chunk_size` 上预留 20% 余量（目标 256 tokens → 切分窗口设为 ~205 tokens）。
- **稳定 ID**：`chunk_id = sha8(manual_name) + block_index + chunk_index`。索引升级时 ID 保持不变。

---

## 二、索引构建

### 2.1 双索引架构

```python
@dataclass
class IndexBuildConfig:
    embed_model_name: str = "BAAI/bge-m3"  # 候选 1；须评测对比 text-embedding-v4
    embed_dim: int | None = None            # 由模型输出推导，不硬编码
    chunk_sizes: list[int] = field(default_factory=lambda: [256, 384])
    overlap_ratio: float = 0.1
    dense_top_k: int = 40
    bm25_top_k: int = 40
    rerank_top_k: int = 30
    parent_budget_tokens: int = 1024
    manifest_version: str = "0.1.0"

def build_indices(
    cfg: IndexBuildConfig,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """产出：chunks.parquet, sections.parquet, faiss.index, bm25_index.pkl, manifest.json。"""

    # 1. 切分所有手册
    all_sections: list[Section] = []
    all_chunks: list[SearchChunk] = []
    for txt_path in sorted(cache_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        lang = "en" if "英文" in txt_path.name else "zh"
        sections, chunks = process_manual(text, txt_path.name, lang, cfg)
        all_sections.extend(sections)
        all_chunks.extend(chunks)

    # 2. 批量嵌入 search chunks 的 embed_text
    model = load_embedding_model(cfg.embed_model_name)
    embed_texts = [c.embed_text for c in all_chunks]
    embeddings = batch_embed(model, embed_texts)
    cfg.embed_dim = embeddings.shape[1]  # 从结果推导

    # 3. FAISS Dense Index
    faiss.normalize_L2(embeddings)
    dense_index = faiss.IndexFlatIP(cfg.embed_dim)
    dense_index.add(embeddings)
    faiss.write_index(dense_index, str(output_dir / "faiss.index"))

    # 4. BM25 Sparse Index
    tokenized = tokenize_for_bm25(embed_texts)
    bm25_index = BM25Okapi(tokenized)
    with open(output_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    # 5. 存储
    chunks_df = pd.DataFrame([asdict(c) for c in all_chunks])
    chunks_df["embedding"] = list(embeddings)
    chunks_df.to_parquet(output_dir / "chunks.parquet")

    sections_df = pd.DataFrame([asdict(s) for s in all_sections])
    sections_df.to_parquet(output_dir / "sections.parquet")

    # 6. Manifest
    manifest = {
        "version": cfg.manifest_version,
        "built_at": datetime.now().isoformat(),
        "embed_model": cfg.embed_model_name,
        "embed_dim": cfg.embed_dim,
        "chunk_size": cfg.chunk_sizes,
        "overlap_ratio": cfg.overlap_ratio,
        "num_chunks": len(all_chunks),
        "num_sections": len(all_sections),
        "source_file_hashes": {p.name: sha256(p) for p in cache_dir.glob("*.txt")},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
```

### 2.2 索引目录结构

```
rag_data/index/
├── faiss.index           # IndexFlatIP, 维度由 embedding 推导
├── bm25_index.pkl        # rank_bm25
├── sections.parquet      # section_id, heading_path, full_text, chunk_ids
├── chunks.parquet        # chunk_id, section_id, embed_text, content, metadata, embedding
└── manifest.json         # 模型名、维度、chunk参数、构建时间、文档哈希
```

---

## 三、在线查询管线

```
User Query
  │
  ▼
┌─ Query Router ─────────────────────────────────────┐
│  · 语言检测 (zh/en)                                 │
│  · 产品识别 (相机/冰箱/...)                           │
│  · 问题类型 (事实/步骤/故障/图片依赖/多步骤)            │
│  · 是否拆解为子问题 (仅多步骤)                        │
└──────────┬─────────────────────────────────────────┘
           │
           ▼
┌─ Hybrid Retrieval ────────────────────────────────┐
│  Dense top-40  +  BM25 top-40                     │
│       │                │                           │
│       └── RRF fusion ──┘                           │
│  → 融合后 top-N 个 search chunks                    │
└──────────┬─────────────────────────────────────────┘
           │
           ▼
┌─ Child-level Rerank ──────────────────────────────┐
│  CrossEncoder(query, child_chunk_text)             │
│  → rerank top-30                                   │
│  → 加上前后相邻 chunk (每个命中 ±1)                   │
└──────────┬─────────────────────────────────────────┘
           │
           ▼
┌─ Parent Expansion (token-budget aware) ───────────┐
│  按 rerank 分数降序，逐个扩展 parent section          │
│  直到累积 tokens 达到 budget (1024/2048)            │
│  同一 section 被多个 child 命中时不重复计入            │
└──────────┬─────────────────────────────────────────┘
           │
           ▼
┌─ Evidence Sufficiency Check ──────────────────────┐
│  LLM 判断：当前上下文中证据是否足以回答？              │
│  · 充分 → 继续生成                                  │
│  · 不充分 → 扩大检索 (query rewriting → 再召一轮)     │
│  · 仍不充分 → 拒答                                  │
└──────────┬─────────────────────────────────────────┘
           │
           ▼
┌─ Structured Generation ───────────────────────────┐
│  LLM 输出:                                         │
│  { answer, cited_evidence_ids, image_ids }         │
│  程序校验 image_ids 白名单后渲染 <PIC>               │
└────────────────────────────────────────────────────┘
```

### 3.1 Query Router

```python
class QueryRouter:
    """用轻量模型 (<3B) 或规则分类器，不在检索关键路径上消耗主力 LLM。"""

    ROUTER_PROMPT = """\
分析用户问题，输出 JSON:
{
  "language": "zh" | "en",
  "product": "相机" | "冰箱" | ... | "unknown",
  "question_type": "fact" | "procedure" | "troubleshoot" | "image_dependent" | "multi_step",
  "sub_questions": ["..."] | null,
  "needs_image_retrieval": true | false
}
仅 multi_step 时拆解为 sub_questions；image_dependent 指涉及按钮位置/指示灯/图中内容。"""

    def route(self, query: str) -> RouteResult:
        ...
```

参考：Adaptive-RAG 按复杂度选择策略；RQ-RAG 拆解后对各子问题独立检索；CRAG 在检索不足时触发纠正。

### 3.2 Hybrid Retrieval

```python
def hybrid_retrieve(
    query: str,
    dense_index: faiss.Index,
    bm25_index: BM25Okapi,
    embed_model,
    metadata: pd.DataFrame,
    dense_k: int = 40,
    bm25_k: int = 40,
    rrf_k: int = 60,
) -> list[dict]:
    """Dense + BM25 → RRF 融合"""

    # Dense 路
    q_vec = np.array([embed_model.embed_query(query)], dtype="float32")
    faiss.normalize_L2(q_vec)
    d_scores, d_indices = dense_index.search(q_vec, dense_k)

    # BM25 路
    tokenized_query = query.split()  # 或 jieba 分词
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[-bm25_k:][::-1]

    # RRF 融合
    rrf_scores: dict[int, float] = {}
    for rank, idx in enumerate(d_indices[0]):
        if idx != -1:
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(bm25_top_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

    sorted_chunk_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    return [
        {
            "chunk_id": cid,
            "rrf_score": rrf_scores[cid],
            **metadata.iloc[cid].to_dict(),
        }
        for cid in sorted_chunk_ids
    ]
```

手册中包含型号名（"NP-50"）、按钮名（"MENU/OK"）、错误码和专有名词，稀疏检索在精确匹配上天然优于 dense。混合检索在系统实验中已被验证为投产比最优的组合之一。

备选方案：用 BGE-M3（BAAI, 2024）单模型同时输出 dense + sparse + multi-vector，无需单独维护 BM25 索引。需在评测集上与 DashScope `text-embedding-v4` 对比延迟和费用。

### 3.3 Child-level Rerank

```python
def rerank_children(
    query: str,
    candidates: list[dict],
    reranker: CrossEncoder,
    top_k: int = 30,
    add_neighbors: bool = True,
) -> list[dict]:
    """子块级重排：CrossEncoder 打分 → top_k → 补入前后相邻 chunk（同 block 内）。"""
    pairs = [(query, c["content"]) for c in candidates]
    scores = reranker.compute_score(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    top = candidates[:top_k]

    if add_neighbors:
        expanded = {}
        for c in top:
            expanded[c["chunk_id"]] = c
            for neighbor_id in get_neighbor_chunk_ids(c):
                if neighbor_id not in expanded:
                    expanded[neighbor_id] = lookup_chunk(neighbor_id)
        return sorted(expanded.values(), key=lambda c: c.get("rerank_score", 0), reverse=True)

    return top
```

先映射完整 section 再 rerank 的问题：超长章节被截断，中部关键信息被稀释（"Lost in the Middle", Liu et al., 2023），且密集步骤和图片常集中在章节局部。因此先子块重排，再按预算扩展父块。

### 3.4 Token-budget Parent Expansion

```python
def expand_to_parents(
    reranked_children: list[dict],
    sections_df: pd.DataFrame,
    token_budget: int = 1024,
    tokenizer=None,
) -> list[dict]:
    """按 rerank 分数降序，逐个映射 child → parent section，累计 tokens 不超过 budget。同一 section 去重。"""
    selected_sections: dict[str, dict] = {}
    tokens_used = 0

    for child in reranked_children:
        section_id = child["section_id"]
        if section_id in selected_sections:
            continue

        section = sections_df.loc[section_id]
        section_tokens = len(tokenizer.encode(section["full_text"]))

        if tokens_used + section_tokens <= token_budget:
            selected_sections[section_id] = {
                "section_id": section_id,
                "heading_path": section["heading_path"],
                "full_text": section["full_text"],
                "trigger_chunk": child["chunk_id"],      # 哪个子块触发了该 section 的入选
                "rerank_score": child["rerank_score"],
            }
            tokens_used += section_tokens
        elif tokens_used == 0:
            # 第一个 section 就超预算 → 截断到 budget
            selected_sections[section_id] = {**section, "truncated": True}
            break
        else:
            continue

    return sorted(selected_sections.values(), key=lambda s: s["rerank_score"], reverse=True)
```

候选 parent budget：`[512, 1024, 2048]`，需在评测集上对比。

手册章节长度差异大（200 ~ 5000+ tokens），LongRAG 那种直接检索长段的策略不适合跨章节查询。这里先子块定位再按预算扩展，兼顾精确性和上下文完整性。

### 3.5 Evidence Sufficiency Check

```python
SUFFICIENCY_PROMPT = """\
根据以下检索到的手册内容，判断是否能充分回答用户问题。

手册内容:
{context}

用户问题: {query}

回答 JSON:
{
  "sufficient": true | false,
  "missing_info": "..." | null,   // 不充分时，说明缺少什么信息
  "suggested_query": "..." | null // 不充分时，建议改写查询以弥补缺失
}
"""
```

不充分时触发第二路：`suggested_query` → 混合检索 → 子块重排 → 合并到上下文，再次判断。仍不充分则拒答。

参考 CRAG（Yan et al., 2024）的检索质量评估和纠正机制。

### 3.6 Structured Generation

```python
GENERATION_PROMPT = """\
你是一个产品使用助手。根据手册参考内容回答用户问题。

手册参考内容:
{context}

用户问题: {query}

请输出 JSON:
{
  "answer": "你的回答（Markdown 格式）",
  "cited_evidence_ids": ["section_id_1", ...],   // 引用来源
  "image_ids": [1, 17, ...]                      // 回答中应展示的图片 ID
}

规则:
1. 仅依据提供的手册内容作答，不要编造
2. 内容不足以回答时，明确告知用户
3. 操作步骤按顺序列出，安全警告着重强调
4. image_ids 只包含回答中确实需要展示的图片
5. cited_evidence_ids 标明回答信息的出处
"""

def generate(self, state: AgentState) -> Command:
    response = self.llm.invoke(messages)
    result = parse_json_response(response.content)

    # 白名单校验：仅保留 context_docs 中真实存在的 image_ids
    valid_ids = {i for doc in state["context_docs"] for i in doc.get("image_ids", [])}
    result["image_ids"] = [i for i in result["image_ids"] if i in valid_ids]

    final_answer, file_list = render_pics(result["answer"], result["image_ids"], self.img_pairs)

    return Command(update={
        "answer": final_answer,
        "file_list": file_list,
        "citations": result["cited_evidence_ids"],
        "messages": [AIMessage(content=final_answer)],
    })
```

模型输出结构化 `{answer, cited_evidence_ids, image_ids}`，程序端对 `image_ids` 做白名单校验后确定性渲染，避免 LLM 遗漏或编造图片引用。

---

## 四、图片双通道管线

### 4.1 设计理由

vLLM 生成的图片描述无法完整保留：按钮空间关系、箭头指向、图中 OCR 文字（屏幕菜单文本）、指示灯状态和颜色。ColPali（Faysse et al., ICLR 2025）表明视觉检索在视觉丰富文档上可显著优于纯文本管线。因此保留视觉通道作为按需激活的补充路径。

### 4.2 双通道设计

```
Query
  │
  ├── 默认通道（文本检索）
  │   query → hybrid retrieval → 搜索正文 chunk + 图片 description
  │   图片描述作为正文的一部分参与检索
  │
  └── 可选通道（视觉检索）
      仅在 query 被 Router 标记为 image_dependent 时激活
      → 图片 IndexFlatIP + 图片 description 检索
      → 或 CLIP/BGE-VL 多模态嵌入检索
```

### 4.3 ImageSpec 扩展

```python
class ImageSpec(BaseModel):
    imgname: str
    description: str

    # 新增字段
    visible_text: str | None = None       # 图中 OCR 文字
    actions: list[str] | None = None      # 图中展示的操作动作 ["按下电源键", "滑动解锁"]
    parts: list[str] | None = None        # 图中可见的部件名称
    context_relation: str | None = None   # 图片在步骤中的角色: "step_illustration" | "reference" | "warning"
    confidence: float | None = None       # 描述置信度 (vLLM 输出可附带)
```

新增字段由 vLLM 在 `build_image_specs` 阶段生成（修改 `ZH_IMAGE_PARSE_PROMPT` / `EN_IMAGE_PARSE_PROMPT` 要求输出更多细节）。

### 4.4 图片参与检索的两条路径

**路径 A（文本通道，默认）**：图片的 `description + visible_text + parts` 作为其对应 chunk 的一部分嵌入。当用户问"电源按钮在哪"，命中的是包含该图片描述的 search chunk。

**路径 B（视觉通道，按需）**：将图片单独向量化（可用 CLIP 或 BGE-VL），建立图片向量库。当 Router 判定 `image_dependent=true` 时，用 query 直接检索图片向量。图片向量和文本 chunk 的 RRF 融合分数共同决定最终上下文。

路径 B 的具体实现推迟到评测验证了图片相关查询的 recall 瓶颈后再铺开。

---

## 五、评测框架

### 5.1 分层测试集

每组 20–30 条，附带 gold evidence（正确答案所在的 section_id 和 image_ids）：

| 查询类型 | 示例 | gold evidence |
|----------|------|---------------|
| 事实查询 | "NP-50 电池充满需要多久？" | section_id: "相机:电池充电" |
| 操作步骤 | "如何装入 instax 相纸？" | section_id: "相机:装入相纸盒" |
| 故障排除 | "相机无法开机怎么办？" | section_id: "相机:故障排除" |
| 多章节 | "如何从拍摄到打印完成一张照片？" | section_ids: ["拍摄", "打印"] |
| 图片依赖 | "哪个是快门按钮？" | section_id + image_ids: [17] |
| 无答案 | "这款相机支持 8K 视频吗？" | evidence: null, 应拒答 |
| 型号/错误码 | "NP-50 电池兼容哪些机型？" | section_id |

### 5.2 指标矩阵

**检索阶段**：

| 指标 | 定义 |
|------|------|
| Recall@k (dense) | gold section 是否在 dense top-k 中 |
| Recall@k (hybrid) | gold section 是否在 RRF 融合后 top-k 中 |
| MRR | gold section 排名的倒数均值 |
| nDCG@10 | gold section 的折损累计增益 |
| Evidence-span recall | gold 证据片段在检索结果中的覆盖率 |

**重排阶段**：

| 指标 | 定义 |
|------|------|
| Reranker Recall@5 | gold section 是否在 reranker top-5 中 |
| Δ vs Dense | Reranker 相对纯 Dense 的 Recall 增益 |
| Δ vs BM25 | Reranker 相对纯 BM25 的 Recall 增益 |

**生成阶段**（参考 RAGChecker, NeurIPS 2024；RAGTruth, ACL 2024）：

| 指标 | 定义 |
|------|------|
| Correctness | 回答声明与 gold answer 的事实一致性 |
| Completeness | gold answer 中的关键信息点被覆盖的比例 |
| Faithfulness | 回答中每个声明是否能被 context_docs 中的证据支撑 |
| Citation Accuracy | cited_evidence_ids 是否准确指向了使用其信息的位置 |
| Refusal Accuracy | 无答案查询的拒答率 |
| Hallucination Rate | 不支持声明的比例（RAGChecker: claim-level entailment） |
| Self-Knowledge | 模型是否在未引用上下文时使用了自身知识 |

**图片阶段**：

| 指标 | 定义 |
|------|------|
| Image Precision | 输出 image_ids 中正确的比例 |
| Image Recall | gold image_ids 中被输出的比例 |
| Image Order | 图片在回答中的出现顺序是否正确 |
| Invalid ID Rate | 输出中非法/白名单外 image_id 的比例（目标 = 0） |

**工程指标**：

| 指标 | 定义 |
|------|------|
| P50/P95 Latency | 端到端延迟的中位数和 95 分位 |
| Total Tokens | query + context + completion 的总 token 消耗 |
| API Cost | 每次查询的 API 费用 |
| Index Size | 索引文件的磁盘占用 |

### 5.3 为什么只测 Recall 不够

RAGChecker 的实验表明高 recall 不一定带来高 faithfulness——检索结果越完整，generator 的噪声敏感度越高。ARES（Saad-Falcon et al., 2024）进一步指出需要区分检索噪声、信息整合失败和矛盾证据三种根本不同的错误模式。仅测整体 faithfulness 无法定位问题所在。

---

## 六、实验路线图

按优先级顺序推进，每步记录指标：

1. **Fixed-chunk Dense baseline**：512-token 固定窗 + FAISS + top-5 直送 → 记录所有指标
2. **层级标题 + 原子块切分**：替换固定窗为 heading-based，其余不变 → Δ Recall/faithfulness
3. **Parent-child 检索**（SINR 范式）：子块搜索 → 父块扩展，token budget=1024 → Δ
4. **Dense + BM25 混合**：RRF 融合 → Δ 型号/错误码/专有名词类查询
5. **子块 Rerank 后按预算扩展**：替换"先映射 section 再 rerank" → Δ rerank recall + 上下文利用率
6. **查询路由 + 多问题拆解**：Adaptive-RAG 风格 → Δ 多步骤查询正确率
7. **检索充分性判断与拒答**：CRAG 风格 → Δ faithfulness + refusal accuracy
8. **图片文本双通道**：视觉通道接入 → Δ 图片依赖查询 recall
9. **高成本对照组**：Late Chunking（Jina AI, 2024）和 RAPTOR（Sarthi et al., 2024）作为上层检索质量的天花板参照，判断当前管线还差多远

每步单独 commit，指标写入实验日志。不做预判，让数据说话。

---

## 七、依赖项

```
# 核心
langchain >= 1.0.0
langgraph >= 1.0.0
langchain-openai >= 1.0.0
langchain-text-splitters >= 1.0.0

# 嵌入 (二选一，需评测)
# 方案 A: DashScope API
# 方案 B: BAAI/bge-m3 (本地部署)
FlagEmbedding >= 1.3.0     # BGE-M3

# 检索
faiss-cpu >= 1.8.0
rank-bm25 >= 0.2.2

# 重排
sentence-transformers >= 2.7.0   # bge-reranker-v2-m3

# 数据
pandas >= 2.2.0
numpy >= 2.0.0
pyarrow >= 15.0.0               # parquet 读写

# API
fastapi >= 0.115.0
uvicorn[standard] >= 0.30.0

# 评估
ragchecker >= 0.1.0             # 细粒度诊断 (NeurIPS 2024)
```

---

## 参考论文

| 论文 | 会议/年份 | 核心贡献 |
|------|----------|----------|
| Search Is Not Retrieval | arXiv 2025.11 | 搜索与检索解耦，小块匹配+大块推理 |
| Rethinking Chunk Size | arXiv 2025.05 | 最优 chunk 大小依赖任务和嵌入模型 |
| Searching for Best Practices in RAG | EMNLP 2024 | 混合检索、重排、查询改写的系统实验 |
| Adaptive-RAG | NAACL 2024 | 按问题复杂度自适应选择检索策略 |
| CRAG | ACL 2024 | 检索质量评估与纠正机制 |
| BGE-M3 | arXiv 2024 | 单模型 dense+sparse+multi-vector |
| ColPali | ICLR 2025 | 视觉文档检索，VLM+Late Interaction |
| Col-Bandit | arXiv 2026 | 自适应剪枝 late-interaction，~5× 加速 |
| RAGChecker | NeurIPS 2024 | 声明级忠实度诊断 |
| RAGTruth | ACL 2024 | 幻觉标注数据集，span 级标注 |
| Lost in the Middle | ACL 2023 | 长上下文中间信息被忽视 |
| RAPTOR | ICLR 2024 | 层级摘要树增强长文档检索 |
| Late Chunking | 2024 | 嵌入先行、切分后行，保留跨块上下文 |
