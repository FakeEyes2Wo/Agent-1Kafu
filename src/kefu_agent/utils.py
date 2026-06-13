from __future__ import annotations

import hashlib
import itertools
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from .schemas import ManualEntry


PIC_RE = re.compile(r"<PIC>", re.IGNORECASE)
EmbeddingModelName = str

HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")

ATOMIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("image", re.compile(r"(?:^.*\n)?<image[^>]*>.*?</image>(?:\n.*)?", re.DOTALL)),
    ("steps", re.compile(r"(?:^(?:\d+[.\、\)]|[•\-*]|\$\\textcircled).*\n?){2,}", re.MULTILINE)),
    ("warning", re.compile(r"(?:⚠|警告|WARNING|注意|Caution).*?(?:\n\n|\n#|$)", re.DOTALL)),
]


class ImageContext(BaseModel):
    imgname: str
    pic_indices: list[int]
    context: str


# ── embedding ────────────────────────────────────────────────────────

def get_embedding_model(
    model_name: EmbeddingModelName,
    *,
    api_key: str | None = None,
    dimensions: int | None = 1024,
    device: str = "cpu",
    model_dir: str | Path | None = None,
    query_prompt_name: str | None = None,
) -> "Embeddings":
    if model_name.startswith("text-embedding-v"):
        return _DashScopeEmbeddings(
            model=model_name,
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY", ""),
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    encode_kwargs = {"normalize_embeddings": True}
    if query_prompt_name:
        encode_kwargs["prompt_name"] = query_prompt_name
    return HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=str(model_dir) if model_dir else None,
        model_kwargs={"device": device},
        encode_kwargs=encode_kwargs,
    )


class _DashScopeEmbeddings:
    """DashScope TextEmbedding 封装，实现 LangChain Embeddings 接口。"""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import dashscope
        result: list[list[float]] = []
        for i in range(0, len(texts), 10):
            batch = texts[i : i + 10]
            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=batch,
                api_key=self.api_key,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"DashScope embedding failed: code={resp.status_code} message={resp.message}")
            result.extend(e["embedding"] for e in resp.output["embeddings"])
        return result

    def embed_query(self, text: str) -> list[float]:
        import dashscope
        resp = dashscope.TextEmbedding.call(
            model=self.model,
            input=text,
            api_key=self.api_key,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope embedding failed: code={resp.status_code} message={resp.message}")
        return resp.output["embeddings"][0]["embedding"]


# ── reranker ──────────────────────────────────────────────────────────


def get_rerank_model(
    backend: str,
    model_name: str,
    *,
    api_key: str | None = None,
) -> object:
    """返回 reranker 对象，统一接口 ``compute_score(pairs) -> list[float]``。

    backend:
        - ``"dashscope"``: 调用 DashScope TextReRank API
        - ``"local"``:   本地 ``sentence_transformers.CrossEncoder``
    """
    if not model_name:
        return None
    if backend == "dashscope":
        return _DashScopeReranker(model_name, api_key or "")
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


class _DashScopeReranker:
    """DashScope TextReRank 薄封装，接口对齐 CrossEncoder.compute_score。"""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")

    def compute_score(self, pairs: list[tuple[str, str]]) -> list[float]:
        import dashscope
        query = pairs[0][0]
        documents = [doc for _, doc in pairs]
        resp = dashscope.TextReRank.call(
            model=self.model,
            query=query,
            documents=documents,
            top_n=len(pairs),
            return_documents=False,
            api_key=self.api_key,
        )
        scores = [0.0] * len(pairs)
        for r in resp.output.results:
            idx = r.index
            if 0 <= idx < len(scores):
                scores[idx] = r.relevance_score
        return scores


# ── image context extraction (init_rag phase) ────────────────────────

def get_line_range(text: str, pos: int, line_window: int = 2) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    spans = []
    cursor = 0
    for line in lines:
        start, end = cursor, cursor + len(line)
        spans.append((start, end))
        cursor = end

    for index, (start, end) in enumerate(spans):
        if start <= pos < end:
            left = max(0, index - line_window)
            right = min(len(spans) - 1, index + line_window)
            return spans[left][0], spans[right][1]
    return 0, len(text)


def get_context_range(
    text: str, pos: int, line_window: int = 2, char_window: int = 200,
) -> tuple[int, int]:
    line_start, line_end = get_line_range(text, pos, line_window)
    start = max(line_start, pos - char_window)
    end = min(line_end, pos + len("<PIC>") + char_window)
    return start, end


def render_pic_context(
    text: str, img_list: list[str], target_imgname: str, start: int, end: int,
) -> str:
    matches = list(PIC_RE.finditer(text))
    parts = []
    cursor = start
    for pic_idx, match in enumerate(matches):
        m_start, m_end = match.span()
        if m_end <= start or m_start >= end:
            continue
        parts.append(text[cursor:m_start])
        imgname = img_list[pic_idx]
        tag = "TARGET_IMAGE" if imgname == target_imgname else "OTHER_IMAGE"
        parts.append(f"[{tag}: {imgname}, pic_idx={pic_idx}]")
        cursor = m_end
    parts.append(text[cursor:end])
    return "".join(parts).strip()


def build_image_context_index(
    entry: "ManualEntry",
    mode: Literal["zh", "en"] = "zh",
    line_window: int = 2,
    char_window: int = 200,
) -> dict[str, ImageContext]:
    text = entry.content
    img_list = entry.img_list
    matches = list(PIC_RE.finditer(text))
    if len(matches) != len(img_list):
        raise ValueError(
            f"<PIC> 数量和 img_list 不一致: pic={len(matches)}, img={len(img_list)}"
        )

    img_to_indices: dict[str, list[int]] = {}
    for pic_idx, imgname in enumerate(img_list):
        img_to_indices.setdefault(imgname, []).append(pic_idx)

    header = (
        "以下是目标图片在手册中的上下文。[TARGET_IMAGE] 表示该图片的所有出现位置。"
        if mode == "zh"
        else "Below is the target image context. [TARGET_IMAGE] marks all occurrences of this image."
    )
    result: dict[str, ImageContext] = {}
    for imgname, pic_indices in img_to_indices.items():
        chunks: list[str] = []
        seen: set[str] = set()
        for pic_idx in pic_indices:
            start, end = get_context_range(
                text, matches[pic_idx].start(),
                line_window=line_window, char_window=char_window,
            )
            chunk = render_pic_context(text, img_list, imgname, start, end)
            key = re.sub(r"\s+", " ", chunk).strip()
            if key and key not in seen:
                seen.add(key)
                chunks.append(chunk)
        result[imgname] = ImageContext(
            imgname=imgname,
            pic_indices=pic_indices,
            context=header + "\n\n" + "\n\n---\n\n".join(chunks),
        )
    return result


# ── shared helpers ───────────────────────────────────────────────────


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def estimate_tokens(text: str) -> int:
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4.0)


def protect_atomic_blocks(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    mapping: dict[str, tuple[str, str]] = {}
    ctr = itertools.count()

    def _stash(m: re.Match[str], tag: str) -> str:
        pid = f"__ATOMIC_{tag}_{next(ctr)}__"
        mapping[pid] = (tag, m.group())
        return pid

    for tag, pattern in ATOMIC_PATTERNS:
        text = pattern.sub(lambda m, t=tag: _stash(m, t), text)
    return text, mapping


def restore_atomic_blocks(text: str, mapping: dict[str, tuple[str, str]]) -> str:
    for placeholder, (_, original) in mapping.items():
        text = text.replace(placeholder, original)
    return text


__all__ = [
    "EmbeddingModelName",
    "HEADING_RE",
    "ImageContext",
    "PIC_RE",
    "build_image_context_index",
    "estimate_tokens",
    "get_embedding_model",
    "get_rerank_model",
    "protect_atomic_blocks",
    "restore_atomic_blocks",
    "sha8",
]
