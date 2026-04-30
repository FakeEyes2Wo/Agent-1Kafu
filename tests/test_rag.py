import sys
from pathlib import Path
from types import SimpleNamespace

from kefu_agent.rag import (
    Chunk,
    HashEmbeddings,
    SentenceTransformerEmbeddings,
    build_index,
    cosine,
    format_answer_with_image_list,
    format_contexts,
    parse_manual,
    retrieve,
    visual_retrieve,
    _image_context_text,
    _metadata_image_ids,
    _node_to_chunk,
    _rank_chunks,
    _visual_chunks,
    _manual_language,
    _manual_language_filters,
    _query_manual_language,
    _reciprocal_rank_fusion,
    _rerank_nodes,
    _resolve_hf_model_name,
    _write_index_metadata,
    split_manual,
)


def _rag_settings(tmp_path: Path, **overrides):
    vectorstore_dir = tmp_path / "vectorstore"
    settings = SimpleNamespace(
        rag_backend="llamaindex",
        vectorstore_dir=vectorstore_dir,
        llamaindex_dir=tmp_path / "llamaindex",
        index_path=vectorstore_dir / "index.jsonl",
        index_meta_path=vectorstore_dir / "index_meta.json",
        embedding_backend="hash",
        embedding_model="hash",
        embedding_query_prompt_name="",
        embedding_model_dir=tmp_path / "models",
        retrieval_top_k=20,
        rerank_enabled=False,
        rerank_model="fake-reranker",
        rerank_top_n=8,
        visual_retriever="lexical",
        visual_top_k=8,
        top_k=8,
        manual_dir=tmp_path / "manuals",
        chunk_size=700,
        chunk_overlap=120,
    )
    for name, value in overrides.items():
        setattr(settings, name, value)
    return settings


def test_hash_embeddings_have_similarity_signal():
    emb = HashEmbeddings()
    a = emb.embed_query("fitness tracker band size")
    b = emb.embed_query("band size options")
    c = emb.embed_query("invoice title and tax id")
    assert cosine(a, b) > cosine(a, c)


def test_parse_manual_json_shape(tmp_path):
    path = tmp_path / "manual.txt"
    path.write_text('["# Title\\nBody<PIC>", ["img_1"]]', encoding="utf-8")
    text, images = parse_manual(path)
    assert "Body" in text
    assert "<PIC>img_1</PIC>" in text
    assert images == ["img_1"]


def test_split_manual_keeps_image_refs(monkeypatch):
    chunks = list(split_manual("manual", "# Title Body<PIC> more body", ["img_1"]))
    assert chunks
    assert chunks[0]["image_ids"] == ["img_1"]


def test_manual_chunks_tag_english_and_chinese_sources():
    english_chunks = list(split_manual("汇总英文手册", "# Title Body", []))
    chinese_chunks = list(split_manual("健身追踪器手册", "# 标题 正文", []))

    assert english_chunks[0]["manual_language"] == "en"
    assert chinese_chunks[0]["manual_language"] == "zh"
    assert _manual_language("English Manual") == "en"


def test_query_language_uses_english_only_for_plain_english_questions():
    assert _query_manual_language("How do I replace the watch band?") == "en"
    assert _query_manual_language("VR 头显怎么开机？") == "zh"
    assert _query_manual_language("如何 replace the watch band?") == "zh"
    filters = _manual_language_filters("en")
    assert filters.filters[0].key == "manual_language"
    assert filters.filters[0].value == "en"


def test_build_index_uses_llamaindex_backend(monkeypatch, tmp_path):
    settings = _rag_settings(tmp_path)
    calls = {}

    def persist(persist_dir):
        calls["persist_dir"] = persist_dir

    class FakeVectorStoreIndex:
        def __init__(self, nodes, embed_model):
            calls["nodes"] = nodes
            calls["embed_model"] = embed_model
            self.storage_context = SimpleNamespace(persist=persist)

    import llama_index.core

    monkeypatch.setattr(llama_index.core, "VectorStoreIndex", FakeVectorStoreIndex)
    monkeypatch.setattr("kefu_agent.rag.retrieval.get_settings", lambda: settings)
    monkeypatch.setattr(
        "kefu_agent.rag.retrieval.load_manual_chunks",
        lambda: iter(
            [
                {
                    "id": "manual-1",
                    "manual": "manual",
                    "title": "title",
                    "text": "body",
                    "image_ids": [],
                }
            ]
        ),
    )
    monkeypatch.setattr("kefu_agent.rag.retrieval._manual_chunks_to_nodes", lambda chunks: chunks)
    monkeypatch.setattr("kefu_agent.rag.retrieval._llama_embed_model_cached", lambda *args: "embed")

    assert build_index() == 1
    assert calls["nodes"][0]["id"] == "manual-1"
    assert calls["embed_model"] == "embed"
    assert calls["persist_dir"] == str(settings.llamaindex_dir)


def test_retrieve_returns_chunks_from_llamaindex_nodes(monkeypatch, tmp_path):
    settings = _rag_settings(tmp_path)
    settings.llamaindex_dir.mkdir(parents=True)
    (settings.llamaindex_dir / "docstore.json").write_text("{}", encoding="utf-8")
    settings.vectorstore_dir.mkdir(parents=True)
    _write_index_metadata(settings, [])
    calls = {}

    class FakeNode:
        node_id = "manual-1"
        metadata = {
            "manual": "manual",
            "title": "title",
            "image_ids": '["img_1"]',
        }

        def get_content(self, **kwargs):
            calls["metadata_mode"] = kwargs.get("metadata_mode")
            return "body <PIC>"

    class FakeRetriever:
        def retrieve(self, query):
            calls["query"] = query
            return [SimpleNamespace(node=FakeNode())]

    class FakeIndex:
        def as_retriever(self, similarity_top_k, filters=None):
            calls["similarity_top_k"] = similarity_top_k
            calls["filters"] = filters
            return FakeRetriever()

    monkeypatch.setattr("kefu_agent.rag.retrieval.get_settings", lambda: settings)
    monkeypatch.setattr(
        "kefu_agent.rag.retrieval._load_llama_index",
        lambda persist_dir, embedding_model, model_dir: FakeIndex(),
    )

    chunks = retrieve("问题", top_k=1)

    assert calls["query"] == "问题"
    assert calls["similarity_top_k"] == 1
    assert calls["filters"].filters[0].key == "manual_language"
    assert calls["filters"].filters[0].value == "zh"
    assert chunks == [
        Chunk("manual-1", "manual", "title", "body <PIC>", ["img_1"], [])
    ]


def test_retrieve_filters_llamaindex_to_english_manuals(monkeypatch, tmp_path):
    settings = _rag_settings(tmp_path)
    settings.llamaindex_dir.mkdir(parents=True)
    (settings.llamaindex_dir / "docstore.json").write_text("{}", encoding="utf-8")
    settings.vectorstore_dir.mkdir(parents=True)
    _write_index_metadata(settings, [])
    calls = {}

    class FakeRetriever:
        def retrieve(self, query):
            calls["query"] = query
            return []

    class FakeIndex:
        def as_retriever(self, similarity_top_k, filters=None):
            calls["filters"] = filters
            return FakeRetriever()

    monkeypatch.setattr("kefu_agent.rag.retrieval.get_settings", lambda: settings)
    monkeypatch.setattr(
        "kefu_agent.rag.retrieval._load_llama_index",
        lambda persist_dir, embedding_model, model_dir: FakeIndex(),
    )
    monkeypatch.setattr(
        "kefu_agent.rag.retrieval._lexical_retrieve",
        lambda query, manual_language, top_k: [],
    )
    monkeypatch.setattr(
        "kefu_agent.rag.retrieval._visual_retrieve",
        lambda query, manual_language, top_k: [],
    )

    assert retrieve("How do I charge it?", top_k=1) == []
    assert calls["filters"].filters[0].key == "manual_language"
    assert calls["filters"].filters[0].value == "en"


def test_retrieve_merges_lexical_candidates_with_vector_results(monkeypatch, tmp_path):
    settings = _rag_settings(tmp_path)
    settings.llamaindex_dir.mkdir(parents=True)
    (settings.llamaindex_dir / "docstore.json").write_text("{}", encoding="utf-8")
    settings.vectorstore_dir.mkdir(parents=True)
    _write_index_metadata(settings, [])

    vector_node = SimpleNamespace(node_id="vector", metadata={}, text="vector")
    lexical_chunk = Chunk("lexical", "manual", "title", "lexical text", [], [])

    class FakeRetriever:
        def retrieve(self, query):
            return [vector_node]

    class FakeIndex:
        def as_retriever(self, similarity_top_k, filters=None):
            return FakeRetriever()

    monkeypatch.setattr("kefu_agent.rag.retrieval.get_settings", lambda: settings)
    monkeypatch.setattr(
        "kefu_agent.rag.retrieval._load_llama_index",
        lambda persist_dir, embedding_model, model_dir: FakeIndex(),
    )
    monkeypatch.setattr(
        "kefu_agent.rag.retrieval._lexical_retrieve",
        lambda query, manual_language, top_k: [lexical_chunk],
    )
    monkeypatch.setattr(
        "kefu_agent.rag.retrieval._visual_retrieve",
        lambda query, manual_language, top_k: [],
    )

    chunks = retrieve("普通问题", top_k=2)

    assert [chunk.id for chunk in chunks] == ["vector", "lexical"]


def test_rerank_orders_nodes(monkeypatch, tmp_path):
    settings = _rag_settings(tmp_path, rerank_model="reranker")

    class FakeReranker:
        def predict(self, pairs):
            assert pairs == [("query", "low"), ("query", "high")]
            return [0.1, 0.9]

    low = SimpleNamespace(text="low")
    high = SimpleNamespace(text="high")
    monkeypatch.setattr("kefu_agent.rag.retrieval.get_settings", lambda: settings)
    monkeypatch.setattr("kefu_agent.rag.retrieval._get_reranker", lambda *args: FakeReranker())

    assert _rerank_nodes("query", [low, high], top_n=2) == [high, low]


def test_reciprocal_rank_fusion_promotes_overlap():
    vector_a = SimpleNamespace(node_id="a", metadata={}, text="a")
    vector_b = SimpleNamespace(node_id="b", metadata={}, text="b")
    lexical_b = Chunk("b", "manual", "title", "b", [], [])
    lexical_c = Chunk("c", "manual", "title", "c", [], [])

    fused = _reciprocal_rank_fusion([[vector_a, vector_b], [lexical_b, lexical_c]])

    assert [getattr(item, "node_id", getattr(item, "id", "")) for item in fused] == [
        "b",
        "a",
        "c",
    ]


def test_visual_retrieve_builds_single_image_context(monkeypatch, tmp_path):
    settings = _rag_settings(tmp_path, visual_top_k=2)
    manual_chunks = [
        {
            "id": "manual-1",
            "manual": "manual",
            "title": "glass cleaning",
            "text": "wipe first <PIC>img_glass</PIC> then store <PIC>img_other</PIC>",
            "image_ids": ["img_glass", "img_other"],
            "manual_language": "en",
        }
    ]
    monkeypatch.setattr("kefu_agent.rag.visual.get_settings", lambda: settings)
    monkeypatch.setattr(
        "kefu_agent.rag.visual.load_manual_chunks",
        lambda: iter(manual_chunks),
    )
    _visual_chunks.cache_clear()

    chunks = visual_retrieve("glass cleaning", "en", top_k=2)

    assert chunks[0].image_ids == ["img_glass"]
    assert "<PIC>img_glass</PIC>" in chunks[0].text
    assert "img_other" not in chunks[0].text


def test_image_context_text_keeps_only_target_pic():
    text = "before <PIC>target</PIC> between <PIC>other</PIC> after"

    assert _image_context_text(text, "target") == "before <PIC>target</PIC> between after"


def test_metadata_preserves_image_ids():
    node = SimpleNamespace(
        node_id="manual-1",
        text="body",
        metadata={"manual": "manual", "title": "title", "image_ids": '["img_1"]'},
    )

    chunk = _node_to_chunk(SimpleNamespace(node=node))

    assert chunk.image_ids == ["img_1"]
    assert _metadata_image_ids(["img_2"]) == ["img_2"]


def test_format_contexts_keeps_pic_placeholders_and_lists_image_ids():
    chunks = [
        Chunk(
            "a",
            "manual",
            "title",
            "first step <PIC> second step <PIC> third step <PIC>",
            ["img_1", "img_2"],
            [1.0],
        )
    ]

    text = format_contexts(chunks)

    assert '可用图片：["img_1", "img_2"]' in text
    assert "first step <PIC>img_1</PIC>" in text
    assert "second step <PIC>img_2</PIC>" in text
    assert "third step <PIC>" in text


def test_format_answer_with_image_list_uses_pic_order_from_contexts(monkeypatch):
    monkeypatch.setattr("kefu_agent.rag.images._valid_image_ids", lambda image_dir: frozenset())
    contexts = 'text <PIC> img_1 </PIC> and <PIC> img_2 </PIC>\n可用图片：["img_1", "img_2"]'

    answer = format_answer_with_image_list(
        "第一步 <PIC> img_1 </PIC> 第二步 <PIC> img_2 </PIC>", contexts
    )

    assert answer == '第一步 <PIC> 第二步 <PIC>,["img_1", "img_2"]'


def test_format_answer_with_image_list_falls_back_to_context_order_for_bare_pics(
    monkeypatch,
):
    monkeypatch.setattr("kefu_agent.rag.images._valid_image_ids", lambda image_dir: frozenset())
    contexts = 'text <PIC> img_1 </PIC> and <PIC> img_2 </PIC>\n可用图片：["img_1", "img_2"]'

    answer = format_answer_with_image_list("第一步 <PIC> 第二步 <PIC>", contexts)

    assert answer == '第一步 <PIC> 第二步 <PIC>,["img_1", "img_2"]'


def test_format_answer_with_image_list_filters_invalid_image_ids(monkeypatch):
    monkeypatch.setattr(
        "kefu_agent.rag.images._valid_image_ids",
        lambda image_dir: frozenset({"img_1", "img_2"}),
    )
    contexts = 'text <PIC> img_1 </PIC> and <PIC> img_2 </PIC>\n可用图片：["img_1", "img_2"]'

    answer = format_answer_with_image_list(
        "第一步 <PIC 图片ID：img_9> 第二步 <PIC>", contexts
    )

    assert answer == '第一步 <PIC> 第二步 <PIC>,["img_1", "img_2"]'


def test_format_answer_with_image_list_ignores_model_image_ids(monkeypatch):
    monkeypatch.setattr(
        "kefu_agent.rag.images._valid_image_ids",
        lambda image_dir: frozenset({"ctx_1", "ctx_2", "wrong_1"}),
    )
    contexts = 'evidence <PIC>ctx_1</PIC> more <PIC>ctx_2</PIC>\n可用图片：["ctx_1", "ctx_2"]'

    answer = format_answer_with_image_list(
        '第一步 <PIC>wrong_1</PIC> 第二步 <PIC 图片ID：wrong_2>,["wrong_3"]',
        contexts,
    )

    assert answer == '第一步 <PIC> 第二步 <PIC>,["ctx_1", "ctx_2"]'


def test_format_answer_with_image_list_removes_pics_without_context_images(monkeypatch):
    monkeypatch.setattr("kefu_agent.rag.images._valid_image_ids", lambda image_dir: frozenset())

    answer = format_answer_with_image_list(
        "press the lights <PIC> then brush <PIC>",
        "evidence without images",
    )

    assert answer == "press the lights then brush"


def test_format_answer_with_image_list_trims_extra_pics(monkeypatch):
    monkeypatch.setattr(
        "kefu_agent.rag.images._valid_image_ids",
        lambda image_dir: frozenset({"img_1"}),
    )

    answer = format_answer_with_image_list(
        "first <PIC> second <PIC> third <PIC>",
        'evidence <PIC>img_1</PIC>\n可用图片：["img_1"]',
    )

    assert answer == 'first <PIC> second third,["img_1"]'


def test_format_answer_with_image_list_preserves_answers_without_pics():
    assert format_answer_with_image_list("不需要图片", "可用图片：[\"img_1\"]") == "不需要图片"


def test_sentence_transformer_query_uses_prompt_name():
    class FakeArray:
        def __init__(self, rows):
            self.rows = rows

        def tolist(self):
            return self.rows

    class FakeModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            return FakeArray([[1.0, 0.0] for _ in texts])

    fake_model = FakeModel()
    embeddings = SentenceTransformerEmbeddings(
        "Qwen/Qwen3-Embedding-0.6B", query_prompt_name="query"
    )
    embeddings._model = fake_model

    assert embeddings.embed_query("query text") == [1.0, 0.0]
    assert fake_model.calls[-1][1]["prompt_name"] == "query"

    assert embeddings.embed_documents(["doc text"]) == [[1.0, 0.0]]
    assert "prompt_name" not in fake_model.calls[-1][1]


def test_sentence_transformer_uses_project_local_cache_dir(monkeypatch, tmp_path):
    calls = []

    class FakeArray:
        def tolist(self):
            return [[1.0, 0.0]]

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

        def encode(self, texts, **kwargs):
            return FakeArray()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    model_dir = tmp_path / "models"
    embeddings = SentenceTransformerEmbeddings(
        "Qwen/Qwen3-Embedding-0.6B",
        query_prompt_name="query",
        model_dir=model_dir,
    )

    assert embeddings.embed_query("query text") == [1.0, 0.0]
    assert model_dir.exists()
    assert calls == [
        (
            "Qwen/Qwen3-Embedding-0.6B",
            {"trust_remote_code": True, "cache_folder": str(model_dir)},
        )
    ]


def test_sentence_transformer_prefers_local_snapshot_when_cache_exists(
    monkeypatch, tmp_path
):
    calls = []

    class FakeArray:
        def tolist(self):
            return [[1.0, 0.0]]

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

        def encode(self, texts, **kwargs):
            return FakeArray()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    model_dir = tmp_path / "models"
    snapshot_dir = (
        model_dir
        / "models--Qwen--Qwen3-Embedding-0.6B"
        / "snapshots"
        / "revision"
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    (snapshot_dir / "model.safetensors").write_text("", encoding="utf-8")
    embeddings = SentenceTransformerEmbeddings(
        "Qwen/Qwen3-Embedding-0.6B",
        query_prompt_name="query",
        model_dir=model_dir,
    )

    assert embeddings.embed_query("query text") == [1.0, 0.0]
    assert calls == [
        (
            str(snapshot_dir.resolve()),
            {
                "trust_remote_code": True,
                "cache_folder": str(model_dir),
            },
        )
    ]


def test_resolve_hf_model_name_uses_local_path_or_repo_id(tmp_path):
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    (local_model / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (local_model / "model.safetensors").write_text("", encoding="utf-8")
    model_dir = tmp_path / "models"
    snapshot_dir = model_dir / "models--BAAI--bge-reranker-v2-m3" / "snapshots" / "rev"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text('{"model_type": "xlm-roberta"}', encoding="utf-8")
    (snapshot_dir / "model.safetensors").write_text("", encoding="utf-8")

    assert _resolve_hf_model_name(str(local_model), model_dir) == str(local_model.resolve())
    assert _resolve_hf_model_name(str(snapshot_dir.parents[1]), model_dir) == str(
        snapshot_dir.resolve()
    )
    assert _resolve_hf_model_name("BAAI/bge-reranker-v2-m3", model_dir) == str(
        snapshot_dir.resolve()
    )
    assert _resolve_hf_model_name("missing/model", model_dir) == "missing/model"


def test_resolve_hf_model_name_never_returns_cache_repo_parent(tmp_path):
    model_dir = tmp_path / "models"
    cache_dir = model_dir / "models--Qwen--Qwen3-Embedding-0.6B"
    cache_dir.mkdir(parents=True)
    (cache_dir / "blobs").mkdir()
    (cache_dir / "snapshots").mkdir()
    (cache_dir / "config.json").write_text("{}", encoding="utf-8")
    (cache_dir / "model.safetensors").write_text("", encoding="utf-8")

    assert _resolve_hf_model_name(str(cache_dir), model_dir) == (
        "Qwen/Qwen3-Embedding-0.6B"
    )


def test_sentence_transformer_does_not_fallback_to_cache_repo_parent(
    monkeypatch, tmp_path
):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls.append(model_name)
            raise RuntimeError("load failed")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    cache_dir = tmp_path / "models--Qwen--Qwen3-Embedding-0.6B"
    cache_dir.mkdir()
    (cache_dir / "blobs").mkdir()

    embeddings = SentenceTransformerEmbeddings(str(cache_dir), model_dir=tmp_path)

    try:
        embeddings.embed_query("query")
    except RuntimeError:
        pass

    assert calls == ["Qwen/Qwen3-Embedding-0.6B"]


def test_rank_chunks_uses_cosine_order():
    chunks = [
        Chunk("a", "manual", "title", "low", [], [0.0, 1.0]),
        Chunk("b", "manual", "title", "high", [], [1.0, 0.0]),
        Chunk("c", "manual", "title", "middle", [], [0.5, 0.5]),
    ]

    ranked = _rank_chunks([1.0, 0.0], chunks, top_k=2)

    assert [chunk.id for chunk in ranked] == ["b", "c"]
