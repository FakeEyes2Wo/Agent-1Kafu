"""RAG 管道数据模型。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


IMAGE_TAG_RE = re.compile(
    r"<image\s+(?:id\s*=\s*['\"]?)?(\d+)['\"]?\s*>(.*?)</image>",
    re.IGNORECASE | re.DOTALL,
)
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def extract_image_ids(text: str) -> set[int]:
    return {int(m.group(1)) for m in IMAGE_TAG_RE.finditer(text)}


# ── chunk / section ──────────────────────────────────────────────────


@dataclass
class HeadingPath:
    levels: list[tuple[int, str]]  # [(1, "安装"), (2, "装入电池")]

    def to_text(self) -> str:
        return " > ".join(title for _, title in self.levels)


@dataclass
class SearchChunk:
    chunk_id: str
    manual_name: str
    heading_path: HeadingPath
    section_id: str
    chunk_index: int
    content: str
    embed_text: str
    image_ids: set[int] = field(default_factory=set)
    has_images: bool = False
    is_atomic_block: bool = False

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "manual_name": self.manual_name,
            "heading_path": self.heading_path.to_text(),
            "section_id": self.section_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "embed_text": self.embed_text,
            "image_ids": sorted(self.image_ids),
            "has_images": self.has_images,
            "is_atomic_block": self.is_atomic_block,
        }


@dataclass
class Section:
    section_id: str
    manual_name: str
    heading_path: HeadingPath
    section_index: int
    full_text: str
    chunk_ids: list[str] = field(default_factory=list)
    image_ids: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "manual_name": self.manual_name,
            "heading_path": self.heading_path.to_text(),
            "section_index": self.section_index,
            "full_text": self.full_text,
            "chunk_ids": self.chunk_ids,
            "image_ids": sorted(self.image_ids),
        }


__all__ = [
    "HEADING_RE",
    "HeadingPath",
    "IMAGE_TAG_RE",
    "SearchChunk",
    "Section",
    "extract_image_ids",
    "sha8",
]
