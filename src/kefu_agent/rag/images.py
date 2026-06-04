import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..config import get_settings
from .core import Chunk, coerce_string_list, parse_json_value
from .manuals import _apply_image_ids_to_pic_placeholders, _named_pic_image_ids


AVAILABLE_IMAGES_LABEL = "\u53ef\u7528\u56fe\u7247"
IMAGE_ORDER_LABEL = "\u914d\u56fe\u987a\u5e8f"
SOURCE_LABEL = "\u6765\u6e90"


def format_contexts(chunks: Iterable[Chunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        text = _apply_image_ids_to_pic_placeholders(
            _normalize_pic_placeholders(chunk.text), chunk.image_ids
        )
        image_text = ""
        if chunk.image_ids:
            image_json = json.dumps(chunk.image_ids, ensure_ascii=False)
            image_text = (
                f"\n{AVAILABLE_IMAGES_LABEL}\uff1a{image_json}"
                f"\n{IMAGE_ORDER_LABEL}\uff1a{image_json}"
            )
        blocks.append(f"[{i}] {SOURCE_LABEL}\uff1a{chunk.manual} / {chunk.title}\n{text}{image_text}")
    return "\n\n".join(blocks)


def format_answer_with_image_list(answer: str, contexts: str) -> str:
    image_ids = _valid_submission_image_ids(_context_image_ids(contexts))
    body = _strip_trailing_image_list(answer.strip())
    body = _strip_named_pic_tags(body)
    body = _strip_inline_pic_image_ids(body)
    body = _strip_context_image_mentions(body, image_ids)
    body = _normalize_pic_placeholders(body).strip()
    pic_count = body.count("<PIC>")
    if pic_count <= 0:
        return body

    keep_count = min(pic_count, len(image_ids))
    body = _keep_first_pic_placeholders(body, keep_count).strip()
    if keep_count <= 0:
        return body
    return f"{body},{json.dumps(image_ids[:keep_count], ensure_ascii=False)}"


def _strip_trailing_image_list(answer: str) -> str:
    match = re.search(r"\s*[,\uff0c]\s*(\[[^\[\]]*\])\s*$", answer, flags=re.S)
    if not match:
        return answer
    value = parse_json_value(match.group(1))
    return answer[: match.start()].rstrip() if isinstance(value, list) else answer


def _strip_inline_pic_image_ids(text: str) -> str:
    return re.sub(r"<PIC\s+\u56fe\u7247ID[:\uff1a]\s*([^>]+)>", "<PIC>", text)


def _strip_named_pic_tags(text: str) -> str:
    return re.sub(
        r"<\s*PIC\s*>\s*([^<>]+?)\s*<\s*/\s*PIC\s*>",
        "<PIC>",
        text,
        flags=re.I,
    )


def _normalize_pic_placeholders(text: str) -> str:
    return re.sub(r"<\s*PIC\s*>", "<PIC>", text)


def _strip_context_image_mentions(text: str, image_ids: Iterable[str]) -> str:
    cleaned = text
    for image_id in image_ids:
        pattern = rf"(?<![\w-]){re.escape(image_id)}(?![\w-])"
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = re.sub(r"[（(]\s*[）)]", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?\uff0c\u3002\uff1b\uff1a\uff01\uff1f])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned)


def _keep_first_pic_placeholders(text: str, keep_count: int) -> str:
    seen = 0

    def replace_pic(match: re.Match[str]) -> str:
        nonlocal seen
        seen += 1
        return "<PIC>" if seen <= keep_count else ""

    text = re.sub(r"<PIC>", replace_pic, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?\uff0c\u3002\uff1b\uff1a\uff01\uff1f])", r"\1", text)
    return text


def _context_image_ids(contexts: str) -> list[str]:
    image_ids = _named_pic_image_ids(contexts)
    labels = rf"(?:{AVAILABLE_IMAGES_LABEL}|{IMAGE_ORDER_LABEL})[:\uff1a]?"
    for match in re.finditer(labels + r"(\[[^\[\]]*\])", contexts):
        image_ids.extend(coerce_string_list(match.group(1)))
    return image_ids


def _valid_submission_image_ids(candidate_ids: Iterable[str]) -> list[str]:
    valid_ids = _valid_image_ids(str(get_settings().image_dir))
    return list(
        dict.fromkeys(
            normalized
            for image_id in candidate_ids
            if (normalized := _normalize_image_id(image_id))
            and (not valid_ids or normalized in valid_ids)
        )
    )


@lru_cache(maxsize=4)
def _valid_image_ids(image_dir: str) -> frozenset[str]:
    path = Path(image_dir)
    if not path.exists():
        return frozenset()
    return frozenset(
        item.stem
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )


def _normalize_image_id(image_id: str) -> str:
    value = str(image_id).strip().strip("\"'\uff0c,;\uff1b\u3002")
    if not value:
        return ""
    return Path(value).stem if re.search(r"\.(png|jpe?g|webp)$", value, re.I) else value
