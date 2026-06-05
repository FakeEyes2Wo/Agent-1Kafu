# 数据集与 RAG 方案研究

> 目标：基于赛题数据结构和近年 RAG / 多模态 RAG 研究，选择最适合本赛题的 RAG 方案。本文不在本地进行提交得分测试；训练、索引压测和榜单验证迁移到专用 GPU 服务器完成。
>
> 2026-06-05 更新：当前应用落地优先使用百炼 API（`qwen3.7-plus`、
> `text-embedding-v4`、可选 `qwen3-vl-rerank`），BM25/sparse 和 RAG
> 编排保留在本地；详见 `docs/bailian_api_first_plan.md`。

## 1. 数据集结论

### 1.1 下载与校验

已下载并校验 DataFountain 赛题数据：

| 文件 | 本地路径 | MD5 | 结论 |
| --- | --- | --- | --- |
| 知识数据手册 | `data/KownledgeBase.zip` | `85c76f071b5c4ddb1c69ececc275ca66` | 与页面一致 |
| 知识问答的问题 | `data/question_public.csv` | `bb49f8520a24e51c67895a2944d4fefc` | 与页面一致 |
| 提交样例 | `data/submission_example.csv` | `6bcf6865a736c84666ee646b0565bc09` | 与页面一致 |

解压后目录：

```text
data/
  question_public.csv
  submission_example.csv
  KownledgeBase/
    手册/
      *.txt
      插图/
        *.png
        *.jpg
```

### 1.2 问题集结构

`question_public.csv`：

- 行数：400
- 字段：`id,question`
- `id` 范围：1-436，中间有缺号，因此提交必须按 CSV 原始顺序输出，不应假设连续 ID。
- 问题长度：最短 10 字符，最长 161 字符，均值约 47.8，中位数 39，P90 约 85。
- 语言分布：
  - 中文：205
  - 英文：187
  - 中英混合：8
- 显式换行 / 多子问题：16 条。
- 粗略中文关键词匹配：
  - 通用客服/售后类：约 61 条。
  - 手册操作/故障/安装类：至少 169 条；英文题多数也是手册类问题。

典型问题形态：

```text
"请问你们家的商品支持7天无理由退换货吗？",
"需要自己承担运费吗？"

"我的DCB101型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？"

"How do I install a bimini top on my boat if I want to use the canopy?"
```

关键含义：

- 这是一个“通用客服政策 + 产品手册问答 + 中英双语”的混合数据集。
- 英文题不是翻译噪声，而是集中覆盖英文产品说明书。
- 很多题是短查询，不能只依赖 dense embedding；型号、部件、按钮、指示灯这类 token 需要 lexical / sparse 检索兜底。
- 16 条显式多子问题必须做子问题拆解，否则容易漏答。

### 1.3 手册结构

解压后手册数据：

- 手册 txt 文件：21 个。
- 可解析为手册单元：40 个。
  - 其中 `汇总英文手册.txt` 包含 20 行，每行基本是一个英文手册 JSON 单元。
  - 大部分中文手册是单个 JSON-like 单元。
- 手册总文本量：约 1,224,685 字符。
- `<PIC>` 标记：约 2,606 个。
- 声明图片 ID：2,577 个，去重后 2,569 个。
- 插图文件：2,608 个。
- 声明的图片 ID 均能在插图目录找到对应文件。
- 约 39 个图片文件没有被手册声明引用，仍建议纳入视觉索引，避免遗漏。

大文件分布：

| 手册 | 单元数 | 文本字符 | 图片 ID | 特点 |
| --- | ---: | ---: | ---: | --- |
| `汇总英文手册.txt` | 20 | 1,079,881 | 1,783 | 英文题主要来源，需按行拆成独立手册 |
| `相机手册.txt` | 1 | 15,839 | 87 | 目录、步骤、LCD/按钮/拍摄模式等 |
| `洗碗机手册.txt` | 1 | 14,114 左右 | 30 | JSON 转义不规范，需 `ast.literal_eval` 兜底 |
| `健身追踪器手册.txt` | 1 | 12,982 | 57 | 样例题来源，表带/充电/环境条件等 |
| `电钻手册.txt` | 1 | 10,895 | 26 | 型号、充电器指示灯、警示表格密集 |
| `健身单车手册.txt` | 1 | 10,824 | 43 | 装配步骤与安全警告 |
| `发电机手册.txt` | 1 | 9,166 | 101 | 标识、机器部件、步骤图多 |

### 1.4 数据噪声与工程风险

1. **解析格式不完全一致**：多数文件可 `json.loads`，但 `洗碗机手册.txt` 存在无效转义，需要 `ast.literal_eval` 兜底；`汇总英文手册.txt` 是多行 JSON 单元，不能当单个 JSON 文件解析。
2. **中英混合明显**：检索需要语言路由；中文 query 检中文手册，英文 query 检英文手册，中英混合 query 需要双路召回。
3. **图片 ID 与 `<PIC>` 强绑定**：评分样例要求正文 `<PIC>` 加末尾图片列表，错误图片会直接影响“图文结合”评分。
4. **通用客服题没有外部知识库证据**：需要单独维护政策 KB，不能强行检索商品手册。
5. **手册标题质量不均**：有些手册标题层级正常，有些整段挤在一个标题中；切块不能只依赖 Markdown 标题。

## 2. 近年研究启发

### 2.1 基础 RAG：非参数知识库仍是核心

Lewis 等人的 RAG 工作证明，生成模型结合外部可检索知识库后，可以在知识密集任务上生成更具体、更事实化的回答。对本赛题而言，手册和插图就是外部非参数知识库，不能只依赖大模型记忆。

适用点：

- 所有产品操作、型号、指示灯、步骤、表带尺寸等必须从手册取证。
- 回答必须保留证据来源对应的图片 ID。

### 2.2 多语种、多粒度 Embedding

BGE-M3 强调 multilingual、multi-functionality、multi-granularity：支持 100+ 语言，同时支持 dense、sparse、multi-vector，多粒度从短句到长文档。该特点非常贴合本数据集的中文、英文、短查询、长手册混合场景。

Qwen3 系列强调多语言和推理能力；当前应用默认选择百炼 `text-embedding-v4` 作为 dense embedding，若改为本地开源权重时可考虑 `Qwen/Qwen3-Embedding-0.6B` 或同级模型。但考虑到本赛题短查询和型号检索较多，单 dense 不够，建议与 sparse/lexical 组合。

适用点：

- 中文与英文 query 共存。
- query 短、手册长，存在粒度不匹配。
- 需要 dense 语义召回 + sparse 精确词召回 + reranker 精排。

### 2.3 Late Interaction / Multi-vector 检索

ColBERTv2 表明 token 级 late interaction 能提升检索细粒度匹配，尤其适合短查询对长文档片段的检索。BGE-M3 也提供 multi-vector 方向。对本赛题，按钮名、型号、状态灯、部件名等 token 级匹配非常关键。

适用点：

- “DCB107/DCB112 指示灯含义”这类 query 需要型号和状态词同时命中。
- “bimini top / anchor light / jet wash” 等英文部件名需要精确匹配。

### 2.4 查询改写与 HyDE

HyDE 的核心思路是先生成假想相关文档，再用该文档 embedding 检索真实语料。它对无标注、零样本 dense retrieval 有帮助。本赛题公开问题没有标准答案标签，HyDE 可用于：

- 将短 query 扩展为“可能出现于手册中的操作描述”。
- 英文题生成更像 manual 的段落，用于检索英文说明书。
- 中文题补充同义词：如“灯闪烁”扩展为“指示灯工作状态、充电中、已充满、过热/过冷延迟”。

注意：HyDE 生成内容可能有幻觉，因此只用于召回，不可直接进入最终答案。

### 2.5 RAPTOR / 层次化检索

RAPTOR 通过递归聚类和摘要构建树形索引，用于长文档不同抽象层级的检索。本数据集的 `汇总英文手册.txt` 很长，且英文题覆盖 20 个产品手册；对“安全注意事项、维护、故障排除”这类宽泛问题，层次摘要能先定位产品/章节，再进入细粒度 chunk。

适用点：

- 英文大文件先拆成 20 个 manual units，再为每个 manual 生成 section summary。
- 宽泛 query 先召回 manual/section，细节 query 再召回具体步骤 chunk。

### 2.6 Self-RAG / CRAG：检索质量自检

Self-RAG 强调自适应检索与反思，CRAG 强调检索评价器和纠错路径。赛题评分中“幻觉抑制”占技术实现核心，因此应增加轻量检索评价：

- 如果 top contexts 与 query 相关性低，触发 query rewrite / HyDE / 更大 top-k。
- 如果检索到多个产品冲突，要求 reranker 再判别，不直接生成。
- 如果证据不足，回答“需要补充型号/图片/订单信息”，不要编造。

### 2.7 多模态 RAG

MuRAG 证明多模态 memory 对图文 QA 有价值；ColPali 进一步从视觉文档检索角度证明直接对页面图像建模可利用 layout/table/figure 等视觉线索；近年的 Multimodal RAG survey 也强调跨模态检索、融合和评估是 MRAG 的关键。

本赛题虽然不是 PDF 页面检索，而是“手册文本 + 插图 ID”，但评价明确要求相关配图。最合适的做法不是把图片交给生成模型临时猜，而是把每个图片 ID 作为可检索对象：

```text
image_id
  -> 所属手册
  -> 所在章节
  -> 前后文本窗口
  -> OCR / caption
  -> 文件名前缀与序号
```

然后在 text chunk 检索之外并行做 image chunk 检索。

## 3. 推荐方案：Router + Hybrid Multimodal RAG + Rerank + Evidence-grounded Generation

### 3.1 总体架构

```text
用户问题 / question_public row
        |
        v
Query Normalizer
  - 清理引号、换行、异常逗号
  - 语言识别 zh/en/mixed
  - 子问题拆解
  - 产品/型号/部件/动作抽取
        |
        v
Query Router
  - policy: 通用客服政策
  - manual: 产品手册
  - mixed: 政策 + 手册
        |
        v
Retrieval per sub-question
  - BM25 / lexical
  - Dense embedding
  - Sparse embedding
  - Visual pseudo chunk
  - Optional HyDE query
        |
        v
RRF Fusion -> Cross-encoder Rerank -> Evidence Pack
        |
        v
Answer Generator
  - 逐子问题回答
  - 只使用 evidence
  - 图片只输出裸 <PIC>
        |
        v
Verifier / Formatter
  - 覆盖检查
  - 幻觉检查
  - 图片 ID 合法性检查
  - 输出 ret 格式
```

### 3.2 解析与切块

必须先修复当前仓库缺失的 `kefu_agent.rag`，并在其中实现健壮解析。

推荐解析策略：

1. 对每个 `.txt` 先尝试 `json.loads(raw)`。
2. 失败时逐行尝试 `json.loads(line)`，用于 `汇总英文手册.txt`。
3. 仍失败时尝试 `ast.literal_eval(raw)`，用于 `洗碗机手册.txt` 这类无效转义。
4. 仍失败则当纯文本处理，并记录 warning。
5. 每个解析单元分配稳定 `manual_unit_id`，例如：
   - `zh_drill`
   - `zh_fitness_tracker`
   - `en_manual_09_boat`
   - `en_manual_10_camera`

推荐切块：

- `section chunk`：按标题、目录项、步骤号、警告词、表格边界切。
- `sliding chunk`：对标题不可靠的手册，用 300-600 token 窗口 + 80 overlap。
- `image-neighbor chunk`：围绕每个 `<PIC>` 抽取前后 200-400 字符，绑定单个图片 ID。
- `summary chunk`：每个 manual unit / section 生成摘要，用于宽泛问题的 first-stage routing。

chunk metadata：

```json
{
  "chunk_id": "...",
  "manual_id": "...",
  "manual_name": "...",
  "language": "zh|en",
  "section_path": ["...", "..."],
  "text": "...",
  "image_ids": ["..."],
  "chunk_type": "section|sliding|image_neighbor|summary",
  "keywords": ["型号", "部件", "动作"]
}
```

### 3.3 图片索引

对 2,608 张图片建立 `image pseudo chunks`：

```json
{
  "image_id": "Manual16_51",
  "file": "Manual16_51.jpg",
  "manual_id": "zh_fitness_tracker",
  "nearby_text": "表带尺寸如下所示...",
  "caption": "可选：由 VLM 生成",
  "ocr": "可选：由 OCR/VLM 提取",
  "section_path": ["表带尺寸"],
  "prefix": "Manual"
}
```

GPU 服务器建议执行：

1. 使用 Qwen2.5-VL / Qwen3-VL / InternVL 对图片生成短 caption。
2. 对包含文字的图片做 OCR，优先识别型号、按钮、灯态、表格项。
3. caption/OCR 不直接作为事实答案，只作为检索字段；最终答案仍以手册文本和图片上下文为证据。

### 3.4 检索组合

推荐 first-stage recall 使用 4 路：

| 路径 | 目的 | 推荐实现 |
| --- | --- | --- |
| Lexical / BM25 | 型号、部件名、按钮、英文短语精确匹配 | `rank-bm25` 或 Lucene/Elasticsearch |
| Dense | 语义召回和同义表达 | 百炼 `text-embedding-v4`；本地备选 `Qwen3-Embedding-0.6B` 或 `bge-m3` |
| Sparse / multi-vector | 短 query 与长 chunk 细粒度匹配 | `bge-m3` sparse/multivector 或 ColBERTv2 |
| Visual pseudo chunk | 图片 ID、图示步骤、表格/部件图召回 | caption/OCR/nearby text embedding |

融合：

- 使用 Reciprocal Rank Fusion（RRF）合并多路候选。
- 每个子问题保留 top 30-50 候选。
- 同一 `manual_id` 和 `section_path` 做去重与 MMR，避免 context 全被重复片段占满。

### 3.5 Reranker

推荐把 reranker 作为服务器消融项，而不是默认必开：

- 首选直接调用百炼 `qwen3-vl-rerank` API，代码中对应 `RERANK_BACKEND=bailian`。
- 如果改成本地开源权重，再考虑 `BAAI/bge-reranker-v2-m3` 或同级多语种 reranker。
- rerank 输入采用结构化格式：

```text
Query: ...
Sub-question: ...
Manual: ...
Section: ...
Text: ...
Images: [...]
```

精排输出 top 6-10 个 evidence，最多 2-4 个图片 evidence，避免把过长 context 塞给生成模型。

### 3.6 Policy / Manual / Mixed 路由

不要只用关键词把问题直接判成 policy。推荐三路：

| 类型 | 示例 | 处理 |
| --- | --- | --- |
| `policy` | 7天无理由、发票、物流、投诉、退款 | 使用固定客服政策 KB，不检索产品手册 |
| `manual` | 电钻灯态、表带尺寸、空调滤网清洁 | 只检索产品手册与图片 |
| `mixed` | 商品故障 + 保修/维修 + 型号 | 同时检索手册和政策，生成时分开回答 |

路由可以先用规则，再用轻量 LLM 分类器修正。

### 3.7 生成与后处理

生成 prompt 应强制 evidence-grounded：

1. 每个子问题必须单独回答。
2. 每个事实必须能在 evidence 中找到。
3. 没证据时说明需要补充，不编造。
4. 需要配图时正文只输出裸 `<PIC>`。
5. 系统后处理根据 evidence 中图片 ID 顺序追加 `,["..."]`。
6. 如果回答没有 `<PIC>`，不要追加图片列表。

推荐后处理检查：

- `PIC count <= evidence image count`
- 图片 ID 必须存在于 `data/KownledgeBase/手册/插图`
- 删除模型输出的文件名、图片 ID、`<PIC>id</PIC>`，统一转换为裸 `<PIC>`
- 如果 context 无图片但答案有 `<PIC>`，删除 `<PIC>`

## 4. 训练与服务器实施计划

### 4.1 不建议直接微调整个生成大模型

当前公开问题没有答案标签，直接 SFT 生成模型风险较高：

- 容易学到提交样例中的无效客服话术。
- 无法保证手册事实准确。
- 成本高，收益不如先优化检索与 rerank。

更合适的训练对象：

1. embedding / retriever。
2. reranker。
3. 图像 caption/OCR 模块。
4. 答案 verifier / router。

### 4.2 合成训练数据

可在 GPU 服务器生成训练数据：

#### Retrieval pair

从每个 section / image-neighbor chunk 生成 3-5 个 query：

```text
positive: (synthetic_query, source_chunk)
hard negative:
  - 同一产品不同章节
  - 不同产品但动作相似，如“清洁滤网”
  - 同一型号附近但错误灯态/按钮
```

#### Image pair

对每张图片生成 caption/OCR 后构造：

```text
query: "如何查看 DCB107 过热/过冷延迟的指示灯？"
positive image chunk: drill0_06 + nearby text
hard negative: drill0_04 / drill0_05
```

#### Policy pair

构造通用客服问法：

```text
query: "退货运费谁承担？"
positive: policy_七天无理由_运费
hard negative: product manual warranty / unrelated manual
```

### 4.3 推荐服务器实验顺序

不做本地得分测试，但建议服务器按以下实验执行：

| 阶段 | 实验 | 目的 |
| --- | --- | --- |
| E0 | 规则解析 + section chunk + BM25 | 建最小可用 baseline |
| E1 | BM25 + dense embedding | 验证语义召回提升 |
| E2 | BM25 + dense + visual pseudo chunk | 验证图片命中 |
| E3 | 加 reranker | 提高 top context 精度 |
| E4 | 加 query rewrite / HyDE | 提高短 query / 英文 query 召回 |
| E5 | 子问题拆解 + per-subquery retrieval | 提高多问题覆盖 |
| E6 | 训练 reranker / embedding | 用 synthetic labels 提升稳健性 |

### 4.4 离线验证指标

即使不在本地得分，也应在服务器做离线诊断：

| 指标 | 说明 |
| --- | --- |
| `manual_recall@k` | 目标产品/手册是否出现在 top-k |
| `image_recall@k` | 样例或人工标注图片 ID 是否出现在 top-k |
| `context_precision` | top context 中无关片段比例 |
| `subquestion_coverage` | 多子问题是否逐项回答 |
| `faithfulness` | 回答事实是否都有 evidence |
| `pic_validity` | `<PIC>` 数量和末尾图片列表是否合法 |
| `latency_text` | 文本题是否稳定在 20s 内 |
| `latency_image` | 含图题是否稳定在 30s 内 |

建议至少人工标注 50 条 golden set：

- 20 条中文手册题。
- 15 条英文手册题。
- 10 条通用客服题。
- 5 条多子问题 / mixed 题。

### 4.5 本地与 GPU 服务器分工

本地只保留轻量分析和代码检查，不做榜单得分或大模型训练：

| 环境 | 执行内容 | 不执行内容 |
| --- | --- | --- |
| 本地 | 数据结构统计、MD5 校验、解析器单元测试、方案文档 | A 榜/B 榜提交评分、大模型训练、全量 VLM caption |
| GPU 服务器 | 全量索引构建、图片 caption/OCR、embedding/reranker 训练、消融实验、生成提交 | 无 |

迁移到服务器时建议按以下目录组织：

```text
project/
  data/
    question_public.csv
    submission_example.csv
    KownledgeBase/
      手册/
      手册/插图/
  storage/
    parsed/
    indexes/
    image_captions/
    experiments/
    submissions/
  configs/
    rag_baseline.yaml
    rag_hybrid_rerank.yaml
    rag_multimodal.yaml
```

推荐服务器运行顺序：

```text
1. parse_manuals
   -> 输出 parsed/manual_units.jsonl, parsed/text_chunks.jsonl, parsed/image_chunks.jsonl

2. caption_images
   -> 输出 image_captions/*.jsonl

3. build_indexes
   -> BM25 index, dense index, sparse/multivector index, image pseudo-chunk index

4. retrieval_ablation
   -> 比较 bm25 / dense / hybrid / hybrid+rerank / hybrid+rerank+visual

5. generate_submission
   -> 输出 submissions/submission_*.csv

6. error_analysis
   -> 抽样失败题，回看 query routing、top contexts、图片 ID、最终回答
```

服务器消融结果建议统一记录：

```json
{
  "experiment": "hybrid_rerank_visual_v1",
  "embedding": "bge-m3",
  "reranker": "bge-reranker-v2-m3",
  "caption_model": "qwen-vl",
  "top_k_recall": 50,
  "top_k_rerank": 8,
  "manual_recall_at_10": "...",
  "image_recall_at_10": "...",
  "pic_validity": "...",
  "avg_latency": "...",
  "submission_score": "..."
}
```

## 5. 结论：最适合本赛题的 RAG 方案

最适合本赛题的不是单纯向量库 RAG，而是：

```text
Policy/Manual/Mixed Router
  + Section-aware Text Chunking
  + Image-neighbor Pseudo Chunks
  + BM25/Sparse + Dense + Visual Hybrid Recall
  + RRF Fusion
  + Multilingual Cross-encoder Rerank
  + Evidence-grounded Generation
  + PIC-safe Formatter
```

推荐模型组合：

| 模块 | 首选 | 备选 |
| --- | --- | --- |
| Dense embedding | 百炼 `text-embedding-v4`；本地备选 `Qwen/Qwen3-Embedding-0.6B` 或 `BAAI/bge-m3` | `bge-large-zh-v1.5` + English embedding 分语言 |
| Sparse / multi-vector | 本地 BM25/sparse lexical | `bge-m3` sparse、ColBERTv2 |
| Reranker | 百炼 `qwen3-vl-rerank` API（默认关闭，消融后启用） | 本地 `BAAI/bge-reranker-v2-m3` |
| Image caption/OCR | Qwen2.5-VL / Qwen3-VL / InternVL | PaddleOCR + BLIP/CLIP caption |
| Generator | 强中文/英文客服能力模型 | 可按服务器资源选择 7B/14B/32B |

优先级：

1. 先修复 RAG 包和解析链路。
2. 再做 hybrid recall + rerank。
3. 再做 image pseudo chunk 和 PIC-safe formatter。
4. 最后做 synthetic data 训练与 verifier。

这条路线最符合数据集的实际结构：中英双语、短 query、长手册、强图片绑定、通用政策与产品手册混合，并且能直接服务赛题的评分维度：RAG 检索效果、多模态理解、多轮/多子问题覆盖和幻觉抑制。

## 参考资料

- Lewis et al., 2020, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks: https://arxiv.org/abs/2005.11401
- Gao et al., 2022, HyDE / Precise Zero-Shot Dense Retrieval without Relevance Labels: https://arxiv.org/abs/2212.10496
- Santhanam et al., 2021, ColBERTv2: https://arxiv.org/abs/2112.01488
- Chen et al., 2024, BGE M3-Embedding: https://arxiv.org/abs/2402.03216
- Asai et al., 2023, Self-RAG: https://arxiv.org/abs/2310.11511
- Yan et al., 2024, Corrective Retrieval Augmented Generation: https://arxiv.org/abs/2401.15884
- Sarthi et al., 2024, RAPTOR: https://arxiv.org/abs/2401.18059
- Chen et al., 2022, MuRAG: https://arxiv.org/abs/2210.02928
- Faysse et al., 2024, ColPali: https://arxiv.org/abs/2407.01449
- Zhu et al., 2024, MuRAR: https://arxiv.org/abs/2408.08521
- Mei et al., 2025, A Survey of Multimodal Retrieval-Augmented Generation: https://arxiv.org/abs/2504.08748
- Ru et al., 2024, RAGChecker: https://arxiv.org/abs/2408.08067
