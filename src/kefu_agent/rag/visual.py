import re
from functools import lru_cache

from kefu_agent.config import get_settings

from .formatting import _inline_pic_image_tags
from .parsing import _query_manual_language, load_manual_chunks
from .schema import Chunk
from .text import term_counts


VISUAL_RETRIEVER_VERSION = "2"


def visual_retrieve(query: str, manual_language: str | None = None, top_k: int | None = None) -> list[Chunk]:
    return _visual_retrieve(
        query,
        manual_language or _query_manual_language(query),
        top_k or get_settings().visual_top_k,
    )


def _visual_retrieve(query: str, manual_language: str, top_k: int) -> list[Chunk]:
    query_terms = term_counts(query)
    if not query_terms or top_k <= 0:
        return []
    scored = []
    for chunk in _visual_chunks():
        if chunk.manual_language != manual_language:
            continue
        doc_terms = term_counts(" ".join([chunk.manual, chunk.title, chunk.text]))
        score = sum(query_terms[token] * doc_terms.get(token, 0) for token in query_terms)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        Chunk(**{**chunk.__dict__, "score": float(score)})
        for score, chunk in scored[:top_k]
    ]


@lru_cache(maxsize=1)
def _visual_chunks() -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for item in load_manual_chunks():
        image_ids = item.get("image_ids") or []
        if not image_ids:
            continue
        tagged_text = _inline_pic_image_tags(item["text"], image_ids)
        for image_id in image_ids:
            chunks.append(
                Chunk(
                    id=f"{item['id']}::{image_id}",
                    manual=item["manual"],
                    title=item["title"],
                    text=_image_context_text(tagged_text, image_id),
                    image_ids=[image_id],
                    vector=[],
                    manual_language=item.get("manual_language", "zh"),
                    chunk_type="image",
                )
            )
    return tuple(chunks)


def _image_context_text(text: str, image_id: str, window: int = 240) -> str:
    pattern = re.compile(r"<\s*PIC\s*>\s*([^<>]+?)\s*<\s*/\s*PIC\s*>", re.I)
    matches = list(pattern.finditer(text))
    target = next((match for match in matches if match.group(1).strip() == image_id), None)
    if target is None:
        return text[: window * 2]
    start = max(0, target.start() - window)
    end = min(len(text), target.end() + window)
    snippet = text[start:end]

    def replace(match: re.Match[str]) -> str:
        current = match.group(1).strip()
        return f"<PIC>{current}</PIC>" if current == image_id else ""

    return re.sub(r"\s+", " ", pattern.sub(replace, snippet)).strip()
