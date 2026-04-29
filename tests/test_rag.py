import sys
from pathlib import Path
from types import SimpleNamespace

from kefu_agent.rag import (
    Chunk,
    HashEmbeddings,
    SentenceTransformerEmbeddings,
    build_index,
    cosine,
    format_contexts,
    parse_manual,
    retrieve,
    _metadata_image_ids,
    _node_to_chunk,
    _rank_chunks,
    _rerank_nodes,
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
        top_k=8,
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
    assert images == ["img_1"]


def test_split_manual_keeps_image_refs(monkeypatch):
    chunks = list(split_manual("manual", "# Title Body<PIC> more body", ["img_1"]))
    assert chunks
    assert chunks[0]["image_ids"] == ["img_1"]


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
    monkeypatch.setattr("kefu_agent.rag.get_settings", lambda: settings)
    monkeypatch.setattr(
        "kefu_agent.rag.load_manual_chunks",
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
    monkeypatch.setattr("kefu_agent.rag._manual_chunks_to_nodes", lambda chunks: chunks)
    monkeypatch.setattr("kefu_agent.rag._llama_embed_model_cached", lambda *args: "embed")

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
        def as_retriever(self, similarity_top_k):
            calls["similarity_top_k"] = similarity_top_k
            return FakeRetriever()

    monkeypatch.setattr("kefu_agent.rag.get_settings", lambda: settings)
    monkeypatch.setattr(
        "kefu_agent.rag._load_llama_index",
        lambda persist_dir, embedding_model, model_dir: FakeIndex(),
    )

    chunks = retrieve("question", top_k=1)

    assert calls["query"] == "question"
    assert calls["similarity_top_k"] == 1
    assert chunks == [
        Chunk("manual-1", "manual", "title", "body <PIC>", ["img_1"], [])
    ]


def test_rerank_orders_nodes(monkeypatch, tmp_path):
    settings = _rag_settings(tmp_path, rerank_model="reranker")

    class FakeReranker:
        def predict(self, pairs):
            assert pairs == [("query", "low"), ("query", "high")]
            return [0.1, 0.9]

    low = SimpleNamespace(text="low")
    high = SimpleNamespace(text="high")
    monkeypatch.setattr("kefu_agent.rag.get_settings", lambda: settings)
    monkeypatch.setattr("kefu_agent.rag._get_reranker", lambda model_name: FakeReranker())

    assert _rerank_nodes("query", [low, high], top_n=2) == [high, low]


def test_metadata_preserves_image_ids():
    node = SimpleNamespace(
        node_id="manual-1",
        text="body",
        metadata={"manual": "manual", "title": "title", "image_ids": '["img_1"]'},
    )

    chunk = _node_to_chunk(SimpleNamespace(node=node))

    assert chunk.image_ids == ["img_1"]
    assert _metadata_image_ids(["img_2"]) == ["img_2"]


def test_format_contexts_inlines_pic_image_ids():
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

    assert "图片ID：img_1, img_2" in text
    assert "first step <PIC 图片ID：img_1>" in text
    assert "second step <PIC 图片ID：img_2>" in text
    assert "third step <PIC>" in text


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


def test_sentence_transformer_prefers_local_files_when_cache_snapshot_exists(
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
    embeddings = SentenceTransformerEmbeddings(
        "Qwen/Qwen3-Embedding-0.6B",
        query_prompt_name="query",
        model_dir=model_dir,
    )

    assert embeddings.embed_query("query text") == [1.0, 0.0]
    assert calls == [
        (
            "Qwen/Qwen3-Embedding-0.6B",
            {
                "trust_remote_code": True,
                "cache_folder": str(model_dir),
                "local_files_only": True,
            },
        )
    ]


def test_rank_chunks_uses_cosine_order():
    chunks = [
        Chunk("a", "manual", "title", "low", [], [0.0, 1.0]),
        Chunk("b", "manual", "title", "high", [], [1.0, 0.0]),
        Chunk("c", "manual", "title", "middle", [], [0.5, 0.5]),
    ]

    ranked = _rank_chunks([1.0, 0.0], chunks, top_k=2)

    assert [chunk.id for chunk in ranked] == ["b", "c"]
