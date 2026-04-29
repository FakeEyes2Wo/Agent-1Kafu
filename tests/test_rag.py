import sys
from types import SimpleNamespace

from kefu_agent.rag import (
    Chunk,
    HashEmbeddings,
    SentenceTransformerEmbeddings,
    cosine,
    format_contexts,
    parse_manual,
    _rank_chunks,
    split_manual,
)


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
