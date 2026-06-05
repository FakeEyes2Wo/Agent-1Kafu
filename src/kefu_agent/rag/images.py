import json
import re
from functools import lru_cache
from pathlib import Path

from kefu_agent.config import get_settings

from .text import ordered_unique


RAG_CONTEXT_FORMAT_VERSION = "4"


def format_answer_with_image_list(answer: str, contexts: str) -> str:
    body, explicit_ids = _strip_trailing_image_list(answer.strip())
    body, named_ids = _strip_named_pic_tags(body)
    body, inline_ids = _strip_inline_pic_ids(body)
    body = _normalize_pic_placeholders(body).strip()
    pic_count = body.count("<PIC>")
    if pic_count <= 0:
        return _remove_orphan_pic_spacing(body)

    context_ids = _context_image_ids(contexts)
    valid_ids = _valid_image_ids(get_settings().image_dir)
    if valid_ids:
        context_ids = [image_id for image_id in context_ids if image_id in valid_ids]

    chosen = []
    for image_id in ordered_unique(named_ids + inline_ids + explicit_ids):
        if image_id in context_ids and image_id not in chosen:
            chosen.append(image_id)
    for image_id in context_ids:
        if image_id not in chosen:
            chosen.append(image_id)
        if len(chosen) >= pic_count:
            break

    if not chosen:
        return _remove_orphan_pic_spacing(body.replace("<PIC>", ""))

    keep = min(pic_count, len(chosen))
    body = _keep_first_pic_placeholders(body, keep)
    return f"{body},{json.dumps(chosen[:keep], ensure_ascii=False)}"


@lru_cache(maxsize=4)
def _valid_image_ids(image_dir: Path) -> frozenset[str]:
    if not image_dir.exists():
        return frozenset()
    ids = {path.stem for path in image_dir.iterdir() if path.is_file()}
    return frozenset(ids)


def _strip_trailing_image_list(answer: str) -> tuple[str, list[str]]:
    match = re.search(r"\s*[,，]\s*(\[[^\[\]]*\])\s*$", answer, flags=re.S)
    if not match:
        return answer, []
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return answer, []
    image_ids = [str(item) for item in value] if isinstance(value, list) else []
    return answer[: match.start()].rstrip(), image_ids


def _strip_named_pic_tags(text: str) -> tuple[str, list[str]]:
    image_ids: list[str] = []

    def replace(match: re.Match[str]) -> str:
        image_id = match.group(1).strip()
        if image_id:
            image_ids.append(image_id)
        return "<PIC>"

    normalized = re.sub(
        r"<\s*PIC\s*>\s*([^<>]+?)\s*<\s*/\s*PIC\s*>",
        replace,
        text,
        flags=re.I,
    )
    return normalized, image_ids


def _strip_inline_pic_ids(text: str) -> tuple[str, list[str]]:
    image_ids: list[str] = []

    def replace(match: re.Match[str]) -> str:
        image_id = match.group(1).strip()
        if image_id:
            image_ids.append(image_id)
        return "<PIC>"

    normalized = re.sub(r"<PIC\s+图片ID[:：]\s*([^>]+)>", replace, text)
    return normalized, image_ids


def _normalize_pic_placeholders(text: str) -> str:
    return re.sub(r"<\s*PIC\s*>", "<PIC>", text, flags=re.I)


def _context_image_ids(contexts: str) -> list[str]:
    image_ids: list[str] = []
    for match in re.finditer(r"<\s*PIC\s*>\s*([^<>]+?)\s*<\s*/\s*PIC\s*>", contexts, re.I):
        image_id = match.group(1).strip()
        if image_id:
            image_ids.append(image_id)
    for match in re.finditer(r"(?:可用图片|image_ids|images)[:：]\s*(\[[^\[\]]*\])", contexts):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            image_ids.extend(str(item) for item in value)
    return ordered_unique(image_ids)


def _keep_first_pic_placeholders(text: str, keep: int) -> str:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "<PIC>" if count <= keep else ""

    return _remove_orphan_pic_spacing(re.sub(r"<PIC>", replace, text))


def _remove_orphan_pic_spacing(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", text)
    return text.strip()
