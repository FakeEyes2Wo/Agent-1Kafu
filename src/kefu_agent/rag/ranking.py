from typing import Any, Iterable

import numpy as np

from .core import (
    MANUAL_LANGUAGE_METADATA_KEY,
    Chunk,
    coerce_string_list,
    lexical_score as _score_text,
)


def _node_to_chunk(node_with_score: Any) -> Chunk:
    if isinstance(node_with_score, Chunk):
        return node_with_score
    node = getattr(node_with_score, "node", node_with_score)
    metadata = dict(getattr(node, "metadata", {}) or {})
    return Chunk(
        id=_node_id(node, metadata, fallback=""),
        manual=str(metadata.get("manual", "")),
        title=str(metadata.get("title", "")),
        text=_node_text(node),
        image_ids=_metadata_image_ids(metadata.get("image_ids")),
        vector=[],
        manual_language=str(metadata.get(MANUAL_LANGUAGE_METADATA_KEY, "zh")),
    )


def _node_text(node_with_score: Any) -> str:
    node = getattr(node_with_score, "node", node_with_score)
    if not hasattr(node, "get_content"):
        return str(getattr(node, "text", ""))
    try:
        return str(node.get_content(metadata_mode="none"))
    except TypeError:
        return str(node.get_content())


def _metadata_image_ids(value: Any) -> list[str]:
    return coerce_string_list(value)


def _rank_chunks(
    query_vector: list[float], chunks: list[Chunk], top_k: int
) -> list[Chunk]:
    if not query_vector or not chunks or top_k <= 0:
        return []

    corpus = np.asarray([chunk.vector for chunk in chunks], dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    if corpus.ndim != 2 or query.ndim != 1 or corpus.shape[1] != query.shape[0]:
        raise RuntimeError(
            "Query vector dimension does not match the index. Rebuild the vector index."
        )

    query_norm = np.linalg.norm(query) or 1.0
    corpus_norms = np.linalg.norm(corpus, axis=1)
    corpus_norms[corpus_norms == 0] = 1.0
    scores = (corpus @ query) / (corpus_norms * query_norm)

    top_indices = np.argsort(scores)[-min(top_k, len(chunks)) :][::-1]
    return [chunks[int(index)] for index in top_indices]


def _fine_rank_nodes(query: str, nodes: list[Any], top_n: int) -> list[Any]:
    if top_n <= 0:
        return []
    return [
        node
        for rank, node in sorted(
            enumerate(nodes),
            key=lambda item: (-_fine_score(query, item[1], item[0]), item[0]),
        )
    ][:top_n]


def _fine_score(query: str, node: Any, rank: int) -> float:
    chunk = _node_to_chunk(node)
    text = f"{chunk.manual} {chunk.title} {' '.join(chunk.image_ids)} {chunk.text}"
    return _score_text(query, text) + 1.0 / (60 + rank + 1)


def _reciprocal_rank_fusion(rankings: Iterable[list[Any]], *, k: int = 60) -> list[Any]:
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    nodes: dict[str, Any] = {}
    order = 0
    for ranking in rankings:
        for rank, node in enumerate(ranking, start=1):
            identity = _node_identity(node)
            if identity not in nodes:
                nodes[identity] = node
                first_seen[identity] = order
                order += 1
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
    return [
        nodes[identity]
        for identity in sorted(
            scores,
            key=lambda identity: (-scores[identity], first_seen[identity]),
        )
    ]


def _merge_nodes(primary: list[Any], fallback: list[Any]) -> list[Any]:
    return _reciprocal_rank_fusion([primary, fallback])


def _node_identity(node_with_score: Any) -> str:
    node = getattr(node_with_score, "node", node_with_score)
    metadata = getattr(node, "metadata", {}) or {}
    return _node_id(node, metadata, fallback=id(node))


def _node_id(node: Any, metadata: dict, fallback: Any) -> str:
    return str(
        getattr(node, "node_id", None)
        or getattr(node, "id_", None)
        or getattr(node, "id", None)
        or metadata.get("id", None)
        or fallback
    )
