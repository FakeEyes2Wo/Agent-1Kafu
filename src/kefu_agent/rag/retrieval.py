import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from kefu_agent.config import Settings, get_settings

from .embeddings import get_embeddings
from .formatting import format_contexts
from .images import RAG_CONTEXT_FORMAT_VERSION, format_answer_with_image_list
from .parsing import (
    MANUAL_LANGUAGE_FILTER_VERSION,
    MANUAL_LANGUAGE_METADATA_KEY,
    MANUAL_PIC_TAG_VERSION,
    _manual_language,
    _manual_language_filters,
    _query_manual_language,
    load_manual_chunks,
    parse_manual,
    split_manual,
)
from .schema import Chunk
from .text import cosine, ordered_unique, term_counts
from .visual import VISUAL_RETRIEVER_VERSION, _visual_retrieve, visual_retrieve


HYBRID_SEARCH_VERSION = "3"


def build_index() -> int:
    settings = get_settings()
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(load_manual_chunks())
    if _rag_backend(settings) == "llamaindex":
        settings.llamaindex_dir.mkdir(parents=True, exist_ok=True)
        nodes = _manual_chunks_to_nodes(chunks)
        if nodes:
            from llama_index.core import VectorStoreIndex

            embed_model = _llama_embed_model_cached(
                settings.embedding_model, str(settings.embedding_model_dir)
            )
            index = VectorStoreIndex(nodes, embed_model=embed_model)
            index.storage_context.persist(persist_dir=str(settings.llamaindex_dir))
        _write_index_metadata(settings, [])
        return len(chunks)

    vectors = _embed_documents_if_enabled([chunk["text"] for chunk in chunks])
    with settings.index_path.open("w", encoding="utf-8") as f:
        for chunk, vector in zip(chunks, vectors, strict=False):
            item = Chunk(vector=vector, **chunk)
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    _write_index_metadata(settings, vectors[0] if vectors else [])
    _load_index.cache_clear()
    _bm25_state.cache_clear()
    return len(chunks)


def retrieve(query: str, top_k: int | None = None) -> list[Chunk]:
    settings = get_settings()
    final_limit = top_k or settings.top_k
    if final_limit <= 0:
        return []

    manual_languages = _query_manual_languages(query)
    recall_k = max(settings.retrieval_top_k, final_limit)
    if _rag_backend(settings) == "llamaindex":
        if (
            not settings.llamaindex_dir.exists()
            or not any(settings.llamaindex_dir.iterdir())
            or not _index_metadata_matches(settings)
        ):
            if build_index() == 0:
                return []
        index = _load_llama_index(
            str(settings.llamaindex_dir),
            settings.embedding_model,
            str(settings.embedding_model_dir),
        )
        candidates = []
        for manual_language in manual_languages:
            nodes = index.as_retriever(
                similarity_top_k=recall_k,
                filters=_manual_language_filters(manual_language),
            ).retrieve(query)
            candidates.extend(
                [
                    [_node_to_chunk(node) for node in nodes],
                    _lexical_retrieve(query, manual_language, recall_k),
                ]
            )
            if settings.visual_retriever.strip().lower() != "off":
                candidates.append(
                    _visual_retrieve(query, manual_language, settings.visual_top_k)
                )
        fused_chunks = [_as_chunk(item) for item in _reciprocal_rank_fusion(candidates)]
        if settings.rerank_enabled:
            fused_chunks = _rerank_nodes(query, fused_chunks, settings.rerank_top_n)
        return _dedupe_chunks(fused_chunks)[:final_limit]

    if not settings.index_path.exists() or not _index_metadata_matches(settings):
        if build_index() == 0:
            return []

    candidates = []
    for manual_language in manual_languages:
        candidates.extend(
            [
                _lexical_retrieve(query, manual_language, recall_k),
                _dense_retrieve(query, manual_language, recall_k),
            ]
        )
        if settings.visual_retriever.strip().lower() != "off":
            candidates.append(_visual_retrieve(query, manual_language, settings.visual_top_k))
    fused = _reciprocal_rank_fusion(candidates)
    fused_chunks = [_as_chunk(item) for item in fused]
    if settings.rerank_enabled:
        fused_chunks = _rerank_nodes(query, fused_chunks, settings.rerank_top_n)
    return _dedupe_chunks(fused_chunks)[:final_limit]


def _embed_documents_if_enabled(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    backend = settings.embedding_backend.strip().lower()
    if not texts or backend in {"none", "disabled", "off"}:
        return [[] for _ in texts]
    if backend in {"openai", "bailian", "dashscope"} and not settings.has_openai_key:
        return [[] for _ in texts]
    embeddings = get_embeddings()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), settings.embedding_batch_size):
        vectors.extend(embeddings.embed_documents(texts[start : start + settings.embedding_batch_size]))
    return vectors


def _dense_retrieve(query: str, manual_language: str, top_k: int) -> list[Chunk]:
    settings = get_settings()
    backend = settings.embedding_backend.strip().lower()
    if backend in {"none", "disabled", "off"} or top_k <= 0:
        return []
    if backend in {"openai", "bailian", "dashscope"} and not settings.has_openai_key:
        return []
    chunks = [
        chunk for chunk in _load_index(str(settings.index_path))
        if chunk.manual_language == manual_language and chunk.vector
    ]
    if not chunks:
        return []
    query_vector = get_embeddings().embed_query(query)
    ranked = sorted(
        (
            Chunk(**{**chunk.__dict__, "score": cosine(query_vector, chunk.vector)})
            for chunk in chunks
        ),
        key=lambda chunk: chunk.score,
        reverse=True,
    )
    return ranked[:top_k]


def _lexical_retrieve(query: str, manual_language: str, top_k: int) -> list[Chunk]:
    chunks = [
        chunk for chunk in _load_index(str(get_settings().index_path))
        if chunk.manual_language == manual_language
    ]
    if not chunks or top_k <= 0:
        return []
    state = _bm25_state(str(get_settings().index_path), manual_language)
    query_terms = term_counts(query)
    if not query_terms:
        return []
    scored: list[Chunk] = []
    for chunk in chunks:
        tf = state["doc_tf"].get(chunk.id, Counter())
        score = 0.0
        doc_len = state["doc_len"].get(chunk.id, 0)
        for term, query_count in query_terms.items():
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = state["idf"].get(term, 0.0)
            denom = freq + state["k1"] * (
                1 - state["b"] + state["b"] * doc_len / state["avgdl"]
            )
            score += query_count * idf * (freq * (state["k1"] + 1)) / denom
        if score > 0:
            scored.append(Chunk(**{**chunk.__dict__, "score": score}))
    scored.sort(key=lambda chunk: chunk.score, reverse=True)
    return scored[:top_k]


@lru_cache(maxsize=4)
def _bm25_state(index_path: str, manual_language: str) -> dict[str, Any]:
    chunks = [
        chunk for chunk in _load_index(index_path)
        if chunk.manual_language == manual_language
    ]
    doc_tf: dict[str, Counter[str]] = {}
    doc_len: dict[str, int] = {}
    df: Counter[str] = Counter()
    for chunk in chunks:
        counts = term_counts(" ".join([chunk.manual, chunk.title, chunk.text]))
        doc_tf[chunk.id] = counts
        doc_len[chunk.id] = sum(counts.values())
        for term in counts:
            df[term] += 1
    doc_count = max(1, len(chunks))
    avgdl = sum(doc_len.values()) / doc_count or 1.0
    idf = {
        term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }
    return {
        "doc_tf": doc_tf,
        "doc_len": doc_len,
        "idf": idf,
        "avgdl": avgdl,
        "k1": 1.5,
        "b": 0.75,
    }


def _reciprocal_rank_fusion(rank_lists: Iterable[Iterable[Any]], k: int = 60) -> list[Any]:
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, Any] = {}
    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list, start=1):
            item_id = _item_id(item)
            if not item_id:
                continue
            scores[item_id] += 1.0 / (k + rank)
            best.setdefault(item_id, item)
    ordered_ids = sorted(scores, key=lambda item_id: scores[item_id], reverse=True)
    fused = []
    for item_id in ordered_ids:
        item = _as_chunk(best[item_id])
        fused.append(Chunk(**{**item.__dict__, "score": scores[item_id]}))
    return fused


def _rerank_nodes(query: str, nodes: list[Any], top_n: int) -> list[Any]:
    if not nodes or top_n <= 0:
        return []
    settings = get_settings()
    backend = getattr(settings, "rerank_backend", "local").strip().lower()
    if backend in {"none", "off", "disabled"}:
        return nodes[:top_n]
    if backend in {"bailian", "dashscope", "api"}:
        return _rerank_with_bailian(query, nodes, top_n)
    pairs = [(query, _node_text(node)) for node in nodes]
    scores = _get_reranker(settings.rerank_model, str(settings.embedding_model_dir)).predict(pairs)
    ranked = sorted(zip(nodes, scores, strict=False), key=lambda item: item[1], reverse=True)
    return [node for node, _score in ranked[:top_n]]


def _rerank_with_bailian(query: str, nodes: list[Any], top_n: int) -> list[Any]:
    settings = get_settings()
    if not settings.has_openai_key:
        raise RuntimeError(
            "Set BAILIAN_API_KEY or DASHSCOPE_API_KEY before using RERANK_BACKEND=bailian."
        )
    model = settings.rerank_model.strip()
    texts = [_node_text(node) for node in nodes]
    if model == "qwen3-rerank":
        url = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        payload = {
            "model": model,
            "query": query,
            "documents": texts,
            "top_n": top_n,
            "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
        }
    else:
        url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        payload = {
            "model": model,
            "input": {
                "query": {"text": query},
                "documents": [{"text": text} for text in texts],
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": False,
                "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
            },
        }
    response = _post_json(url, payload, settings.model_api_key, settings.model_timeout_seconds)
    output = response.get("output") or {}
    results = response.get("results") or output.get("results") or []
    reranked = []
    for result in results:
        try:
            index = int(result["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < len(nodes):
            reranked.append(nodes[index])
    return reranked[:top_n] or nodes[:top_n]


def _post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Bailian rerank API failed: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Bailian rerank API request failed: {exc.reason}") from exc


@lru_cache(maxsize=2)
def _get_reranker(model_name: str, cache_folder: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, cache_folder=cache_folder, trust_remote_code=True)


def _dedupe_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for chunk in chunks:
        key = chunk.id
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


@lru_cache(maxsize=1)
def _load_index(path: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    index_path = Path(path)
    if not index_path.exists():
        return chunks
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def _node_to_chunk(node_with_score: Any) -> Chunk:
    node = getattr(node_with_score, "node", node_with_score)
    metadata = dict(getattr(node, "metadata", {}) or {})
    return Chunk(
        id=str(getattr(node, "node_id", None) or getattr(node, "id_", "") or metadata.get("id", "")),
        manual=str(metadata.get("manual", "")),
        title=str(metadata.get("title", "")),
        text=_node_text(node),
        image_ids=_metadata_image_ids(metadata.get("image_ids")),
        vector=[],
        manual_language=str(metadata.get(MANUAL_LANGUAGE_METADATA_KEY, "zh")),
    )


def _node_text(node_with_score: Any) -> str:
    node = getattr(node_with_score, "node", node_with_score)
    if hasattr(node, "get_content"):
        try:
            return str(node.get_content(metadata_mode="none"))
        except TypeError:
            return str(node.get_content())
    return str(getattr(node, "text", ""))


def _metadata_image_ids(value: Any) -> list[str]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _as_chunk(item: Any) -> Chunk:
    if isinstance(item, Chunk):
        return item
    return _node_to_chunk(item)


def _item_id(item: Any) -> str:
    if isinstance(item, Chunk):
        return item.id
    return str(getattr(item, "node_id", None) or getattr(item, "id", "") or getattr(item, "id_", ""))


def _index_metadata_matches(settings: Settings) -> bool:
    if not settings.index_meta_path.exists():
        return False
    try:
        metadata = json.loads(settings.index_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = _index_signature(settings)
    return all(metadata.get(key) == value for key, value in expected.items())


def _write_index_metadata(settings: Settings, sample_vector: list[float]) -> None:
    metadata = _index_signature(settings)
    metadata["vector_dim"] = len(sample_vector)
    settings.index_meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _index_signature(settings: Settings) -> dict[str, str]:
    return {
        "embedding_backend": settings.embedding_backend.strip().lower(),
        "embedding_model": settings.embedding_model,
        "embedding_query_prompt_name": settings.embedding_query_prompt_name,
        "rag_backend": settings.rag_backend.strip().lower(),
        "chunk_size": str(settings.chunk_size),
        "chunk_overlap": str(settings.chunk_overlap),
        "manual_language_filter_version": MANUAL_LANGUAGE_FILTER_VERSION,
        "manual_pic_tag_version": MANUAL_PIC_TAG_VERSION,
        "hybrid_search_version": HYBRID_SEARCH_VERSION,
        "visual_retriever_version": VISUAL_RETRIEVER_VERSION,
        "dense_vectors_enabled": str(_dense_vectors_enabled(settings)),
    }


def _rank_chunks(query_vector: list[float], chunks: list[Chunk], top_k: int) -> list[Chunk]:
    ranked = sorted(
        (Chunk(**{**chunk.__dict__, "score": cosine(query_vector, chunk.vector)}) for chunk in chunks),
        key=lambda chunk: chunk.score,
        reverse=True,
    )
    return ranked[:top_k]


def _rag_backend(settings: Settings) -> str:
    backend = settings.rag_backend.strip().lower()
    if backend in {"hybrid", "bm25", "local_bm25"}:
        return "hybrid"
    if backend == "llamaindex":
        return "llamaindex"
    raise RuntimeError(f"Unsupported RAG_BACKEND: {settings.rag_backend}")


def _dense_vectors_enabled(settings: Settings) -> bool:
    backend = settings.embedding_backend.strip().lower()
    if backend in {"none", "disabled", "off"}:
        return False
    if backend in {"openai", "bailian", "dashscope"}:
        return settings.has_openai_key
    return True


def _query_manual_languages(query: str) -> list[str]:
    primary = _query_manual_language(query)
    if primary == "zh" and re.search(r"[A-Za-z]{3,}", query):
        return ["zh", "en"]
    return [primary]


def _manual_chunks_to_nodes(chunks: list[dict]) -> list[Any]:
    try:
        from llama_index.core.schema import TextNode
    except Exception:
        return chunks
    return [
        TextNode(
            id_=chunk["id"],
            text=chunk["text"],
            metadata={
                "manual": chunk["manual"],
                "title": chunk["title"],
                "image_ids": json.dumps(chunk["image_ids"], ensure_ascii=False),
                MANUAL_LANGUAGE_METADATA_KEY: chunk.get("manual_language", "zh"),
            },
        )
        for chunk in chunks
    ]


@lru_cache(maxsize=2)
def _llama_embed_model_cached(model_name: str, cache_folder: str) -> Any:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=model_name,
        cache_folder=cache_folder,
        trust_remote_code=True,
    )


@lru_cache(maxsize=2)
def _load_llama_index(persist_dir: str, embedding_model: str, model_dir: str) -> Any:
    from llama_index.core import StorageContext, load_index_from_storage

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    embed_model = _llama_embed_model_cached(embedding_model, model_dir)
    return load_index_from_storage(storage_context, embed_model=embed_model)
