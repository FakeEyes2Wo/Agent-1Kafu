import ast
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings
from .core import Chunk, manual_language as _manual_language, parse_json_value


def load_manual_chunks() -> Iterable[dict]:
    settings = get_settings()
    for path in sorted(settings.manual_dir.glob("*.txt")):
        text, image_ids = parse_manual(path)
        yield from split_manual(path.stem, text, image_ids)


def parse_manual(path: Path) -> tuple[str, list[str]]:
    raw = path.read_text(encoding="utf-8").strip()
    parsed = _parse_manual_payload(raw)
    if parsed:
        body, image_ids = parsed
        return _apply_image_ids_to_pic_placeholders(body, image_ids), image_ids
    return raw, []


@lru_cache(maxsize=2)
def cached_manual_chunks(
    manual_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    pic_tag_version: str,
) -> tuple[Chunk, ...]:
    del manual_dir, chunk_size, chunk_overlap, pic_tag_version
    return tuple(Chunk(vector=[], **chunk) for chunk in load_manual_chunks())


def _parse_manual_payload(raw: str) -> tuple[str, list[str]] | None:
    payload = _manual_payload(parse_json_value(raw))
    return payload or _manual_payload(_literal_eval(raw))


def _literal_eval(raw: str) -> Any | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None


def _manual_payload(value: Any) -> tuple[str, list[str]] | None:
    if not isinstance(value, list) or not value:
        return None
    images = value[1] if len(value) > 1 and isinstance(value[1], list) else []
    return str(value[0]), [str(item) for item in images]


def _apply_image_ids_to_pic_placeholders(text: str, image_ids: list[str]) -> str:
    image_iter = iter(image_ids)

    def replace_pic(match: re.Match[str]) -> str:
        if re.match(r"\s*[^<>]+?</\s*PIC\s*>", text[match.end() :], re.I):
            return match.group(0)
        image_id = next(image_iter, None)
        if not image_id:
            return match.group(0)
        return f"<PIC>{image_id}</PIC>"

    return re.sub(r"<\s*PIC\s*>", replace_pic, text)


def _named_pic_image_ids(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"<\s*PIC\s*>\s*([^<>]+?)\s*<\s*/\s*PIC\s*>", text, re.I)
        if match.group(1).strip()
    ]


def split_manual(manual: str, text: str, image_ids: list[str]) -> Iterable[dict]:
    settings = get_settings()
    text = _apply_image_ids_to_pic_placeholders(text, image_ids)
    text = re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()
    image_cursor = 0
    chunk_no = 0
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    for title, body in _sections(text):
        for document in splitter.create_documents(
            [body],
            metadatas=[{"manual": manual, "title": title}],
        ):
            chunk_text = document.page_content.strip()
            if not chunk_text:
                continue
            refs = _named_pic_image_ids(chunk_text)
            pic_count = len(refs) or chunk_text.count("<PIC>")
            if not refs:
                refs = image_ids[image_cursor : image_cursor + pic_count]
            image_cursor += pic_count
            chunk_no += 1
            metadata = document.metadata
            yield {
                "id": f"{manual}-{chunk_no}",
                "manual": metadata["manual"],
                "title": metadata["title"],
                "text": chunk_text,
                "image_ids": refs,
                "manual_language": _manual_language(metadata["manual"]),
            }


def _sections(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?=#\s*)", text)
    sections: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"#\s*([^#]{1,80})", part)
        title = match.group(1).strip() if match else "\u6b63\u6587"
        sections.append((title, part))
    return sections or [("\u6b63\u6587", text)]
