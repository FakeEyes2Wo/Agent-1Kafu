import re
from functools import lru_cache

from ..config import get_settings
from .core import (
    MANUAL_PIC_TAG_VERSION,
    Chunk,
    filter_chunks_by_manual_language as _filter_chunks_by_manual_language,
    lexical_score as _score_text,
    rank_by_score,
)
from .manuals import cached_manual_chunks


def visual_retrieve(query: str, manual_language: str, top_k: int) -> list[Chunk]:
    settings = get_settings()
    if getattr(settings, "visual_retriever", "lexical").strip().lower() in {
        "off",
        "none",
        "disabled",
    }:
        return []
    limit = min(top_k, max(0, int(getattr(settings, "visual_top_k", top_k))))
    if limit <= 0:
        return []
    chunks = _filter_chunks_by_manual_language(
        list(
            _visual_chunks(
                str(settings.manual_dir),
                settings.chunk_size,
                settings.chunk_overlap,
                MANUAL_PIC_TAG_VERSION,
            )
        ),
        manual_language,
    )
    return _visual_rank_chunks(query, chunks, limit)


@lru_cache(maxsize=2)
def _visual_chunks(
    manual_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    pic_tag_version: str,
) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for chunk in cached_manual_chunks(
        manual_dir, chunk_size, chunk_overlap, pic_tag_version
    ):
        for image_id in chunk.image_ids:
            image_id = str(image_id)
            chunks.append(
                Chunk(
                    id=f"{chunk.id}#image:{image_id}",
                    manual=chunk.manual,
                    title=chunk.title,
                    text=_image_context_text(chunk.text, image_id),
                    image_ids=[image_id],
                    vector=[],
                    manual_language=chunk.manual_language,
                )
            )
    return tuple(chunks)


def _visual_rank_chunks(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    return rank_by_score(chunks, lambda chunk: _visual_score(query, chunk), top_k)


def _visual_score(query: str, chunk: Chunk) -> float:
    text = f"{chunk.manual} {chunk.title} {' '.join(chunk.image_ids)} {chunk.text}"
    query_lower = query.lower()
    image_bonus = any(
        image_id.lower() in query_lower for image_id in chunk.image_ids
    )
    return _score_text(query, text) + float(image_bonus)


def _image_context_text(text: str, image_id: str, window: int = 260) -> str:
    match = re.search(
        rf"<\s*PIC\s*>\s*{re.escape(image_id)}\s*<\s*/\s*PIC\s*>",
        text,
        flags=re.I,
    )
    if not match:
        return f"{_strip_pic_tags(text)[: window * 2].strip()} <PIC>{image_id}</PIC>"

    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    snippet = text[start:end]

    def replace_pic(pic_match: re.Match[str]) -> str:
        current_id = pic_match.group(1).strip()
        return f"<PIC>{image_id}</PIC>" if current_id == image_id else ""

    snippet = re.sub(
        r"<\s*PIC\s*>\s*([^<>]+?)\s*<\s*/\s*PIC\s*>",
        replace_pic,
        snippet,
        flags=re.I,
    )
    if f"<PIC>{image_id}</PIC>" not in snippet:
        snippet = f"{snippet} <PIC>{image_id}</PIC>"
    return re.sub(r"\s{2,}", " ", snippet).strip()


def _strip_pic_tags(text: str) -> str:
    text = re.sub(r"<\s*PIC\s*>\s*([^<>]+?)\s*<\s*/\s*PIC\s*>", " ", text, flags=re.I)
    return re.sub(r"<\s*PIC\s*>", " ", text, flags=re.I)
