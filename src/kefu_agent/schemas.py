from __future__ import annotations

import ast
import hashlib
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field


PIC_RE = re.compile(r"<PIC>")
IMAGE_TAG_RE = re.compile(
    r"<image\s+(?:id\s*=\s*['\"]?)?(\d+)['\"]?\s*>(.*?)</image>",
    re.IGNORECASE | re.DOTALL,
)
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def extract_image_ids(text: str) -> set[int]:
    return {int(m.group(1)) for m in IMAGE_TAG_RE.finditer(text)}


# ── legacy ImageSpec (used by init_rag / build_image_specs) ──────────

class ImageSpec(BaseModel):
    imgname: str = Field(description="图片文件名")
    description: str = Field(description="这个图片的具体内容")

    visible_text: str | None = None
    actions: list[str] | None = None
    parts: list[str] | None = None
    context_relation: str | None = None
    confidence: float | None = None

    def get_prompt(self, image_id: int) -> str:
        return f"<image id='{image_id}'>{self.description}</image>"


class ManualEntry(BaseModel):
    content: str = Field(description="当前表项的内容")
    img_list: list[str]

    def __init__(self, doc_str: str):
        content, img_list = ast.literal_eval(doc_str)
        super().__init__(content=content, img_list=img_list)

    def change_PIC(self, img_pairs: dict[str, ImageSpec]) -> str:
        images = {}
        for image_id, (name, spec) in enumerate(img_pairs.items(), start=1):
            for alias in (name, Path(name).stem, spec.imgname, Path(spec.imgname).stem):
                images[alias] = (image_id, spec)
        image_names = iter(self.img_list)

        def replace(_: re.Match[str]) -> str:
            imgname = next(image_names)
            image = images.get(imgname) or images.get(Path(imgname).stem)
            if image is None:
                raise KeyError(f"缺少图片描述: {imgname}")
            image_id, spec = image
            return spec.get_prompt(image_id)

        return PIC_RE.sub(replace, self.content)


class ManualDocument(BaseModel):
    file_path: str = Field(description="这个文档的文档路径")
    file_name: str = Field(description="这个文档的文件名")
    content_list: list[ManualEntry]

    def __init__(self, file_path: str | Path):
        path = Path(file_path)
        content_list = [
            ManualEntry(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        super().__init__(
            file_path=str(path),
            file_name=path.name,
            content_list=content_list,
        )


def turn_description_PIC(
    content: str,
    img_pairs: dict[str, ImageSpec],
) -> tuple[str, list[str]]:
    img_label_pairs = {
        image_id: Path(name).stem
        for image_id, name in enumerate(img_pairs, start=1)
    }
    file_list: list[str] = []

    def replace(match: re.Match[str]) -> str:
        image_id = int(match.group(1))
        if image_id not in img_label_pairs:
            raise KeyError(f"缺少图片 id: {image_id}")
        file_list.append(img_label_pairs[image_id])
        return "<PIC>"

    return IMAGE_TAG_RE.sub(replace, content), file_list


# ── RAG pipeline structures ──────────────────────────────────────────


@dataclass
class HeadingPath:
    levels: list[tuple[int, str]]  # [(1, "安装电池"), (2, "装入/取出电池")]

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


# ── exports ──────────────────────────────────────────────────────────

__all__ = [
    "HEADING_RE",
    "HeadingPath",
    "IMAGE_TAG_RE",
    "ImageSpec",
    "ManualDocument",
    "ManualEntry",
    "PIC_RE",
    "SearchChunk",
    "Section",
    "extract_image_ids",
    "sha8",
    "turn_description_PIC",
]
