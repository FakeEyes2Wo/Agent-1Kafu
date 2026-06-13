"""Chunk 切分：原子块保护 + 层级标题切分 + token 窗口滑动。"""

from __future__ import annotations

import itertools
import re

from .schemas import HEADING_RE, HeadingPath, SearchChunk, extract_image_ids, sha8


ATOMIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("image", re.compile(r"(?:^.*\n)?<image[^>]*>.*?</image>(?:\n.*)?", re.DOTALL)),
    ("steps", re.compile(r"(?:^(?:\d+[.\、\)]|[•\-*]|\$\\textcircled).*\n?){2,}", re.MULTILINE)),
    ("warning", re.compile(r"(?:⚠|警告|WARNING|注意|Caution).*?(?:\n\n|\n#|$)", re.DOTALL)),
]


def estimate_tokens(text: str) -> int:
    """中文 ~1.5 字/token，英文 ~4 字/token。"""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4.0)


def protect_atomic_blocks(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """将原子块替换为 __ATOMIC_{tag}_{N}__ 占位符。"""
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


# ── heading split ────────────────────────────────────────────────────


def split_by_headings(text: str) -> list[tuple[HeadingPath, str]]:
    """按层级标题切分，首段（第一个 # 之前）heading_path 为空。"""
    blocks: list[tuple[HeadingPath, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def _flush() -> None:
        if buf:
            blocks.append((HeadingPath(list(stack)), "".join(buf)))
            buf.clear()

    for line in text.splitlines(keepends=True):
        m = HEADING_RE.match(line.lstrip())
        if m:
            _flush()
            level = len(m.group(1))
            title = m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        buf.append(line)
    _flush()
    return blocks


# ── search chunk generation ──────────────────────────────────────────


def make_search_chunks(
    section_text: str,
    heading_path: HeadingPath,
    manual_name: str,
    section_id: str,
    chunk_size: int = 256,
    overlap_ratio: float = 0.1,
) -> list[SearchChunk]:
    """保护原子块 → token 切分 → 还原 → 生成 SearchChunk 列表。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    protected, atom_map = protect_atomic_blocks(section_text)
    effective = int(chunk_size * 0.8)
    overlap = int(effective * overlap_ratio)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective, chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", ".", "，", ",", " ", ""],
        keep_separator=True, length_function=estimate_tokens,
    )
    raw_chunks = splitter.split_text(protected)

    chunks = []
    for i, raw in enumerate(raw_chunks):
        text = restore_atomic_blocks(raw, atom_map)
        prefix = f"{manual_name}\n{heading_path.to_text()}\n"
        embed_text = prefix + text
        if estimate_tokens(embed_text) > 8000:
            head_len = max(8000 - len(prefix), 1000)
            embed_text = prefix + text[:head_len]
        chunks.append(SearchChunk(
            chunk_id=f"{sha8(manual_name)}:{sha8(section_id)}:{i}",
            manual_name=manual_name, heading_path=heading_path,
            section_id=section_id, chunk_index=i,
            content=text,
            embed_text=embed_text,
            image_ids=extract_image_ids(text),
            has_images=bool(extract_image_ids(text)),
            is_atomic_block=(len(atom_map) == 1 and i == 0),
        ))
    return chunks


__all__ = [
    "estimate_tokens",
    "make_search_chunks",
    "protect_atomic_blocks",
    "restore_atomic_blocks",
    "split_by_headings",
]
