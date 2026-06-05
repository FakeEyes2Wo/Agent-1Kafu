import ast
import json
import re
import warnings
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from kefu_agent.config import get_settings


MANUAL_LANGUAGE_METADATA_KEY = "manual_language"
MANUAL_LANGUAGE_FILTER_VERSION = "2"
MANUAL_PIC_TAG_VERSION = "2"


def load_manual_chunks() -> Iterable[dict]:
    settings = get_settings()
    manual_dir = settings.manual_dir
    if not manual_dir.exists():
        raise RuntimeError(f"manual_dir does not exist: {manual_dir}")
    for path in sorted(manual_dir.glob("*.txt")):
        for unit_no, (text, image_ids) in enumerate(_parse_manual_units(path), start=1):
            manual = path.stem if unit_no == 1 else f"{path.stem}_{unit_no:02d}"
            yield from split_manual(manual, text, image_ids)


def parse_manual(path: Path) -> tuple[str, list[str]]:
    units = list(_parse_manual_units(path))
    if not units:
        return "", []
    return units[0]


def _parse_manual_units(path: Path) -> Iterable[tuple[str, list[str]]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return

    parsed = _parse_value(raw)
    if parsed is not None:
        unit = _manual_unit_from_value(parsed)
        if unit:
            yield unit
            return

    yielded = False
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed_line = _parse_value(line)
        unit = _manual_unit_from_value(parsed_line) if parsed_line is not None else None
        if unit:
            yielded = True
            yield unit
    if not yielded:
        yield raw, []


def _parse_value(raw: str):
    for parser in (json.loads, ast.literal_eval):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return parser(raw)
        except Exception:
            continue
    return None


def _manual_unit_from_value(value) -> tuple[str, list[str]] | None:
    if isinstance(value, dict):
        body = value.get("text") or value.get("content") or value.get("manual") or ""
        images = value.get("images") or value.get("image_ids") or []
        if body:
            return _inject_image_ids(str(body), _string_list(images)), _string_list(images)
    if isinstance(value, list) and value:
        body = str(value[0])
        images = value[1] if len(value) > 1 and isinstance(value[1], list) else []
        image_ids = _string_list(images)
        return _inject_image_ids(body, image_ids), image_ids
    return None


def split_manual(manual: str, text: str, image_ids: list[str]) -> Iterable[dict]:
    settings = get_settings()
    normalized_text, pic_ids = _normalize_named_pic_tags(text)
    image_ids = pic_ids or image_ids
    normalized_text = re.sub(r"\s+", " ", normalized_text.replace("\r", "\n")).strip()
    normalized_text = _tag_pic_placeholders(normalized_text, image_ids)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", ". ", "；", "; ", " ", ""],
    )
    chunk_no = 0
    for title, body in _sections(normalized_text):
        title = re.sub(r"<PIC_REF_\d+>", "<PIC>", title)
        for chunk_text in splitter.split_text(body):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            refs = _chunk_image_refs(chunk_text, image_ids)
            chunk_text = re.sub(r"<PIC_REF_\d+>", "<PIC>", chunk_text)
            chunk_no += 1
            yield {
                "id": f"{manual}-{chunk_no}",
                "manual": manual,
                "title": title,
                "text": chunk_text,
                "image_ids": refs,
                "manual_language": _manual_language(manual),
                "chunk_type": "text",
            }


def _inject_image_ids(text: str, image_ids: list[str]) -> str:
    ids = iter(image_ids)

    def replace(match: re.Match[str]) -> str:
        current = next(ids, None)
        return f"<PIC>{current}</PIC>" if current else match.group(0)

    return re.sub(r"<\s*PIC\s*>", replace, text)


def _tag_pic_placeholders(text: str, image_ids: list[str]) -> str:
    cursor = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal cursor
        if cursor >= len(image_ids):
            return match.group(0)
        tag = f"<PIC_REF_{cursor}>"
        cursor += 1
        return tag

    return re.sub(r"<PIC>", replace, text)


def _chunk_image_refs(text: str, image_ids: list[str]) -> list[str]:
    refs = []
    for match in re.finditer(r"<PIC_REF_(\d+)>", text):
        index = int(match.group(1))
        if index < len(image_ids) and image_ids[index] not in refs:
            refs.append(image_ids[index])
    return refs


def _normalize_named_pic_tags(text: str) -> tuple[str, list[str]]:
    image_ids: list[str] = []

    def replace(match: re.Match[str]) -> str:
        image_id = match.group(1).strip()
        if image_id:
            image_ids.append(image_id)
        return "<PIC>"

    text = re.sub(
        r"<\s*PIC\s*>\s*([^<>]+?)\s*<\s*/\s*PIC\s*>",
        replace,
        text,
        flags=re.I,
    )
    text = re.sub(r"<\s*PIC\s*/?\s*>", "<PIC>", text, flags=re.I)
    return text, image_ids


def _sections(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?=#\s*)", text)
    sections: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"#\s*([^#]{1,100})", part)
        title = match.group(1).strip() if match else "正文"
        sections.append((title, part))
    return sections or [("正文", text)]


def _manual_language(manual: str) -> str:
    normalized = manual.lower()
    return "en" if "英文" in normalized or "english" in normalized else "zh"


def _query_manual_language(query: str) -> str:
    if any("\u4e00" <= ch <= "\u9fff" for ch in query):
        return "zh"
    return "en" if len(re.findall(r"[A-Za-z]", query)) >= 3 else "zh"


def _manual_language_filters(manual_language: str):
    try:
        from llama_index.core.vector_stores import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )

        return MetadataFilters(
            filters=[
                MetadataFilter(
                    key=MANUAL_LANGUAGE_METADATA_KEY,
                    operator=FilterOperator.EQ,
                    value=manual_language,
                )
            ]
        )
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(
            filters=[
                SimpleNamespace(
                    key=MANUAL_LANGUAGE_METADATA_KEY,
                    operator="==",
                    value=manual_language,
                )
            ]
        )


def _string_list(value) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
