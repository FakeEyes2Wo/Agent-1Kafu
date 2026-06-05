import json
import re
from typing import Iterable

from .schema import Chunk


def format_contexts(chunks: Iterable[Chunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        text = _inline_pic_image_tags(chunk.text, chunk.image_ids)
        image_text = (
            "\n可用图片：" + json.dumps(chunk.image_ids, ensure_ascii=False)
            if chunk.image_ids
            else ""
        )
        score_text = f" score={chunk.score:.4f}" if chunk.score else ""
        blocks.append(
            f"[{i}] 来源：{chunk.manual} / {chunk.title}"
            f" type={chunk.chunk_type}{score_text}\n{text}{image_text}"
        )
    return "\n\n".join(blocks)


def _inline_pic_image_tags(text: str, image_ids: list[str]) -> str:
    if re.search(r"<\s*PIC\s*>.*?<\s*/\s*PIC\s*>", text, flags=re.I):
        return text
    ids = iter(image_ids)
    text = re.sub(r"<\s*PIC\s*>", "<PIC>", text, flags=re.I)

    def replace(match: re.Match[str]) -> str:
        image_id = next(ids, None)
        if not image_id:
            return match.group(0)
        return f"<PIC>{image_id}</PIC>"

    return re.sub(r"<PIC>", replace, text)
