import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from .core import (
    HYBRID_SEARCH_VERSION,
    MANUAL_LANGUAGE_FILTER_VERSION,
    MANUAL_LANGUAGE_METADATA_KEY,
    MANUAL_PIC_TAG_VERSION,
    VISUAL_RETRIEVER_VERSION,
    Chunk,
    _resolve_hf_model_name,
    filter_chunks_by_manual_language as _filter_chunks_by_manual_language,
    lexical_score as _score_text,
    manual_language as _manual_language,
    manual_language_filters as _manual_language_filters,
    query_manual_language as _query_manual_language,
    rank_by_score,
    read_json_file,
)
from .embeddings import get_embeddings
from .manuals import cached_manual_chunks, load_manual_chunks
from .ranking import (
    _fine_rank_nodes,
    _node_identity,
    _node_text,
    _node_to_chunk,
    _rank_chunks,
    _reciprocal_rank_fusion,
)
from .visual import visual_retrieve as _visual_retrieve


def build_index() -> int:
    settings = get_settings()
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(load_manual_chunks())
    if _rag_backend(settings) != "legacy":
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

    embeddings = get_embeddings()
    texts = [chunk["text"] for chunk in chunks]
    vectors = embeddings.embed_documents(texts) if texts else []

    with settings.index_path.open("w", encoding="utf-8") as f:
        for chunk, vector in zip(chunks, vectors, strict=False):
            item = Chunk(vector=vector, **chunk)
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    _write_index_metadata(settings, vectors[0] if vectors else [])
    return len(chunks)


def retrieve(query: str, top_k: int | None = None) -> list[Chunk]:
    settings = get_settings()
    final_limit = top_k or settings.top_k
    if final_limit <= 0:
        return []
    candidate_limit = _candidate_limit(settings, final_limit)
    manual_language = _query_manual_language(query)

    if _rag_backend(settings) == "legacy":
        return _retrieve_legacy(
            settings, query, manual_language, final_limit, candidate_limit
        )
    return _retrieve_llamaindex(
        settings, query, manual_language, final_limit, candidate_limit
    )


def _retrieve_legacy(
    settings: Settings,
    query: str,
    manual_language: str,
    final_limit: int,
    candidate_limit: int,
) -> list[Chunk]:
    if not settings.index_path.exists() or not _index_metadata_matches(settings):
        build_index()
        _load_index.cache_clear()
    chunks = _filter_chunks_by_manual_language(
        _load_index(str(settings.index_path)),
        manual_language,
    )
    if not chunks:
        return []
    vector_chunks = _rank_chunks(
        get_embeddings().embed_query(query), chunks, candidate_limit
    )
    candidates = _hybrid_candidates(
        vector_chunks,
        _lexical_rank_chunks(query, chunks, candidate_limit),
        _visual_retrieve(query, manual_language, candidate_limit),
    )
    return _fine_rank_nodes(query, candidates, candidate_limit)[:final_limit]


def _retrieve_llamaindex(
    settings: Settings,
    query: str,
    manual_language: str,
    final_limit: int,
    candidate_limit: int,
) -> list[Chunk]:
    if (
        not settings.llamaindex_dir.exists()
        or not any(settings.llamaindex_dir.iterdir())
        or not _index_metadata_matches(settings)
    ):
        if build_index() == 0:
            return []
        _load_llama_index.cache_clear()

    index = _load_llama_index(
        str(settings.llamaindex_dir),
        settings.embedding_model,
        str(settings.embedding_model_dir),
    )
    vector_nodes = index.as_retriever(
        similarity_top_k=candidate_limit,
        filters=_manual_language_filters(manual_language),
    ).retrieve(query)
    candidates = _hybrid_candidates(
        vector_nodes,
        _lexical_retrieve(query, manual_language, candidate_limit),
        _visual_retrieve(query, manual_language, candidate_limit),
    )
    nodes = _final_ranked_nodes(settings, query, candidates, final_limit, candidate_limit)
    return [_node_to_chunk(node) for node in nodes[:final_limit]]


def _final_ranked_nodes(
    settings: Settings,
    query: str,
    candidates: list[Any],
    final_limit: int,
    candidate_limit: int,
) -> list[Any]:
    if not settings.rerank_enabled:
        return _fine_rank_nodes(query, candidates, candidate_limit)
    return _rerank_nodes(query, candidates, max(final_limit, settings.rerank_top_n))


def _candidate_limit(settings: Settings, final_limit: int) -> int:
    return max(
        final_limit,
        final_limit * 3,
        int(getattr(settings, "retrieval_top_k", final_limit)),
    )


def _hybrid_candidates(
    vector_results: list[Any],
    lexical_results: list[Any],
    visual_results: list[Any],
) -> list[Any]:
    return _reciprocal_rank_fusion([vector_results, lexical_results, visual_results])


def _manual_chunks_to_nodes(chunks: list[dict]) -> list[Any]:
    from llama_index.core.schema import TextNode

    return [
        TextNode(
            id_=chunk["id"],
            text=chunk["text"],
            metadata={
                "manual": chunk["manual"],
                "title": chunk["title"],
                "image_ids": json.dumps(chunk["image_ids"], ensure_ascii=False),
                MANUAL_LANGUAGE_METADATA_KEY: chunk.get(
                    "manual_language", _manual_language(chunk["manual"])
                ),
            },
        )
        for chunk in chunks
    ]


@lru_cache(maxsize=4)
def _llama_embed_model_cached(model_name: str, cache_folder: str) -> Any:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=_resolve_hf_model_name(model_name, cache_folder),
        cache_folder=cache_folder,
        trust_remote_code=True,
    )


@lru_cache(maxsize=2)
def _load_llama_index(persist_dir: str, embedding_model: str, model_dir: str) -> Any:
    from llama_index.core import StorageContext, load_index_from_storage

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    embed_model = _llama_embed_model_cached(embedding_model, model_dir)
    return load_index_from_storage(storage_context, embed_model=embed_model)


def _rerank_nodes(query: str, nodes: list[Any], top_n: int) -> list[Any]:
    if not nodes or top_n <= 0:
        return []

    pairs = [(query, _node_text(node)) for node in nodes]
    settings = get_settings()
    scores = _get_reranker(
        settings.rerank_model, str(settings.embedding_model_dir)
    ).predict(pairs)
    ranked = sorted(
        zip(nodes, scores, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    return [node for node, _score in ranked[:top_n]]


@lru_cache(maxsize=2)
def _get_reranker(model_name: str, cache_folder: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        _resolve_hf_model_name(model_name, cache_folder),
        cache_folder=cache_folder,
        trust_remote_code=True,
    )


def _lexical_retrieve(query: str, manual_language: str, top_k: int) -> list[Chunk]:
    settings = get_settings()
    chunks = list(
        _hybrid_chunks(
            str(settings.manual_dir),
            settings.chunk_size,
            settings.chunk_overlap,
            MANUAL_PIC_TAG_VERSION,
        )
    )
    chunks = _filter_chunks_by_manual_language(chunks, manual_language)
    return _lexical_rank_chunks(query, chunks, top_k)


def _hybrid_chunks(
    manual_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    pic_tag_version: str,
) -> tuple[Chunk, ...]:
    return cached_manual_chunks(manual_dir, chunk_size, chunk_overlap, pic_tag_version)


def _lexical_rank_chunks(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    return rank_by_score(chunks, lambda chunk: _lexical_score(query, chunk), top_k)


def _lexical_score(query: str, chunk: Chunk) -> float:
    return _score_text(query, f"{chunk.manual} {chunk.title} {chunk.text}")


def _index_metadata_matches(settings: Settings) -> bool:
    if not settings.index_meta_path.exists():
        return False
    metadata = read_json_file(settings.index_meta_path)
    if not isinstance(metadata, dict):
        return False
    expected = _embedding_signature(settings)
    return all(metadata.get(key) == value for key, value in expected.items())


def _write_index_metadata(settings: Settings, sample_vector: list[float]) -> None:
    metadata = _embedding_signature(settings)
    metadata["vector_dim"] = len(sample_vector)
    settings.index_meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _embedding_signature(settings: Settings) -> dict[str, str]:
    return {
        "embedding_backend": settings.embedding_backend.strip().lower(),
        "embedding_model": settings.embedding_model,
        "embedding_query_prompt_name": settings.embedding_query_prompt_name,
        "rag_backend": _rag_backend(settings),
        "retrieval_top_k": str(settings.retrieval_top_k),
        "rerank_enabled": str(settings.rerank_enabled),
        "rerank_model": settings.rerank_model,
        "rerank_top_n": str(settings.rerank_top_n),
        "manual_language_filter_version": MANUAL_LANGUAGE_FILTER_VERSION,
        "manual_pic_tag_version": MANUAL_PIC_TAG_VERSION,
        "hybrid_search_version": HYBRID_SEARCH_VERSION,
        "visual_retriever_version": VISUAL_RETRIEVER_VERSION,
        "visual_retriever": getattr(settings, "visual_retriever", "lexical"),
        "visual_top_k": str(getattr(settings, "visual_top_k", 8)),
    }


def _rag_backend(settings: Settings) -> str:
    backend = settings.rag_backend.strip().lower()
    if backend not in {"llamaindex", "legacy"}:
        raise RuntimeError(f"Unsupported RAG_BACKEND: {settings.rag_backend}")
    return backend


@lru_cache(maxsize=1)
def _load_index(path: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks
