from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter


from .prompts import EN_IMAGE_PARSE_PROMPT, ZH_IMAGE_PARSE_PROMPT
from .schemas import (
    IMAGE_TAG_RE,
    HeadingPath,
    ImageSpec,
    ManualDocument,
    SearchChunk,
    Section,
    extract_image_ids,
    sha8,
)
from .utils import (
    HEADING_RE,
    ImageContext,
    build_image_context_index,
    estimate_tokens,
    protect_atomic_blocks,
    restore_atomic_blocks,
)


UNKNOWN_IMGS = [
    "Camera_20.png", "Camera_66.png", "Camera_67.png",
    "Dish_washer_04.png", "Dish_washer_05.png", "Dish_washer_06.png",
    "drill0_07.png", "drill0_13.png", "generator_02.png",
]

# ══════════════════════════════════════════════════════════════════════
#  Phase 1: init_rag — image description via vLLM
# ══════════════════════════════════════════════════════════════════════


def build_image_specs(
    vllm,
    doc_list: list[ManualDocument],
    img_context_dict: dict[str, ImageContext],
    image_dir: str | Path = Path("data") / "手册" / "插图",
    image_counts_path: str | Path = "image_name_counts.csv",
) -> dict[str, ImageSpec]:
    image_dir = Path(image_dir)
    image_paths = {
        name: path
        for path in image_dir.iterdir()
        if path.is_file()
        for name in (path.name, path.stem)
    }
    image_modes = {
        imgname: "en" if document.file_name == "汇总英文手册.txt" else "zh"
        for document in doc_list
        for entry in document.content_list
        for imgname in entry.img_list
    }
    structured_vllm = vllm.with_structured_output(ImageSpec)
    img_pairs: dict[str, ImageSpec] = {}

    for imgname, image_context in img_context_dict.items():
        image_path = image_paths[imgname]
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        mode = image_modes[imgname]
        img_spec = structured_vllm.invoke(
            [
                SystemMessage(content=(
                    ZH_IMAGE_PARSE_PROMPT if mode == "zh" else EN_IMAGE_PARSE_PROMPT
                )),
                HumanMessage(content=[
                    {"type": "text", "text": f"图片文件名: {image_path.name}\n\n{image_context.context}"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                ]),
            ]
        )
        img_pairs[imgname] = img_spec.model_copy(update={"imgname": image_path.name})

    image_counts = pd.read_csv(image_counts_path)
    manual_by_prefix = {}
    for row in image_counts.itertuples(index=False):
        if pd.notna(row.manual_names):
            prefix = re.sub(r"_\d+$", "", Path(row.image_name).stem).lower()
            manual_by_prefix.setdefault(prefix, row.manual_names)

    for imgname in UNKNOWN_IMGS:
        image_path = image_paths[imgname]
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        prefix = re.sub(r"_\d+$", "", image_path.stem).lower()
        mode = "en" if "汇总英文手册" in manual_by_prefix.get(prefix, "") else "zh"
        img_spec = structured_vllm.invoke(
            [
                SystemMessage(content=(
                    ZH_IMAGE_PARSE_PROMPT if mode == "zh" else EN_IMAGE_PARSE_PROMPT
                )),
                HumanMessage(content=[
                    {"type": "text", "text": f"图片文件名: {image_path.name}\n\n"
                        + ("该图片未在手册中出现，请直接描述图片内容。"
                           if mode == "zh"
                           else "This image does not occur in the manual. Describe the image directly.")},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                ]),
            ]
        )
        img_pairs[imgname] = img_spec.model_copy(update={"imgname": image_path.name})

    return img_pairs


def init_rag(
    vllm,
    manual_dir: str | Path = Path("data") / "手册",
    image_dir: str | Path = Path("data") / "手册" / "插图",
    rag_data_dir: str | Path = "rag_data",
    image_counts_path: str | Path = "image_name_counts.csv",
) -> dict[str, ImageSpec]:
    manual_dir = Path(manual_dir)
    rag_data_dir = Path(rag_data_dir)
    cache_dir = rag_data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    doc_list = [ManualDocument(path) for path in sorted(manual_dir.glob("*.txt"))]
    img_context_dict: dict[str, ImageContext] = {}
    for document in doc_list:
        mode = "en" if document.file_name == "汇总英文手册.txt" else "zh"
        for entry in document.content_list:
            for imgname, image_context in build_image_context_index(entry, mode).items():
                if imgname in img_context_dict:
                    current = img_context_dict[imgname]
                    current.pic_indices.extend(image_context.pic_indices)
                    current.context += "\n\n---\n\n" + image_context.context
                else:
                    img_context_dict[imgname] = image_context

    pd.DataFrame(
        [ic.model_dump() for ic in img_context_dict.values()],
        columns=["imgname", "pic_indices", "context"],
    ).to_csv(cache_dir / "img_context.csv", index=False, encoding="utf-8")

    img_pairs = build_image_specs(vllm, doc_list, img_context_dict, image_dir, image_counts_path)
    pd.DataFrame(
        [{"id": iid, "imgname": spec.imgname, "description": spec.description}
         for iid, spec in enumerate(img_pairs.values(), start=1)]
    ).to_csv(rag_data_dir / "image_specs.csv", index=False, encoding="utf-8")

    for document in doc_list:
        content = "\n".join(entry.change_PIC(img_pairs) for entry in document.content_list)
        (cache_dir / document.file_name).write_text(content, encoding="utf-8")

    return img_pairs


# ══════════════════════════════════════════════════════════════════════
#  Phase 2: build_rag_index — chunk + embed + FAISS + BM25
# ══════════════════════════════════════════════════════════════════════


def _split_by_headings(text: str) -> list[tuple[HeadingPath, str]]:
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


def _split_into_search_chunks(
    section_text: str,
    heading_path: HeadingPath,
    manual_name: str,
    section_id: str,
    chunk_size: int,
    overlap_ratio: float,
) -> list[SearchChunk]:
    

    protected, atom_map = protect_atomic_blocks(section_text)
    effective_size = int(chunk_size * 0.8)
    overlap = int(effective_size * overlap_ratio)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", ".", "，", ",", " ", ""],
        keep_separator=True,
        length_function=estimate_tokens,
    )
    raw_chunks = splitter.split_text(protected)

    chunks = []
    for i, raw in enumerate(raw_chunks):
        text = restore_atomic_blocks(raw, atom_map)
        chunks.append(SearchChunk(
            chunk_id=f"{sha8(manual_name)}:{sha8(section_id)}:{i}",
            manual_name=manual_name,
            heading_path=heading_path,
            section_id=section_id,
            chunk_index=i,
            content=text,
            embed_text=f"{manual_name}\n{heading_path.to_text()}\n{text}",
            image_ids=extract_image_ids(text),
            has_images=bool(extract_image_ids(text)),
            is_atomic_block=(len(atom_map) == 1 and i == 0),
        ))
    return chunks


def _tokenize_for_bm25(texts: list[str]) -> list[list[str]]:
    tokens_list: list[list[str]] = []
    for text in texts:
        tokens: list[str] = []
        cjk_buf: list[str] = []
        for ch in text:
            if "一" <= ch <= "鿿":
                cjk_buf.append(ch)
            else:
                if cjk_buf:
                    tokens.extend(cjk_buf)
                    tokens.extend(a + b for a, b in zip(cjk_buf, cjk_buf[1:]))
                    cjk_buf.clear()
                if not ch.isspace():
                    tokens.append(ch.lower())
        if cjk_buf:
            tokens.extend(cjk_buf)
            tokens.extend(a + b for a, b in zip(cjk_buf, cjk_buf[1:]))
        tokens_list.append(tokens)
    return tokens_list


def build_rag_index(
    rag_data_dir: str | Path = "rag_data",
    embed_model=None,
    chunk_size: int = 256,
    overlap_ratio: float = 0.1,
) -> tuple:
    """
    从 rag_data/cache/*.txt 构建 FAISS + BM25 双索引。

    产出 rag_data/index/{faiss.index, bm25_index.pkl, chunks.parquet, sections.parquet, manifest.json}
    返回 (dense_index, bm25_index, chunks_df, sections_df)。
    """
    import faiss
    import numpy as np
    from rank_bm25 import BM25Okapi

    rag_data_dir = Path(rag_data_dir)
    cache_dir = rag_data_dir / "cache"
    output_dir = rag_data_dir / "index"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 切分
    all_sections: list[Section] = []
    all_chunks: list[SearchChunk] = []

    for txt_path in sorted(cache_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        manual_name = txt_path.name
        blocks = _split_by_headings(text)

        for si, (hp, block_text) in enumerate(blocks):
            sid = f"{sha8(manual_name)}:s{si}"
            scs = _split_into_search_chunks(block_text, hp, manual_name, sid, chunk_size, overlap_ratio)

            all_sections.append(Section(
                section_id=sid, manual_name=manual_name, heading_path=hp,
                section_index=si, full_text=block_text,
                chunk_ids=[c.chunk_id for c in scs],
                image_ids=extract_image_ids(block_text),
            ))
            all_chunks.extend(scs)

    # 2. 嵌入
    embed_texts = [c.embed_text for c in all_chunks]
    emb_list: list[list[float]] = []
    for i in range(0, len(embed_texts), 32):
        emb_list.extend(embed_model.embed_documents(embed_texts[i : i + 32]))

    embeddings = np.array(emb_list, dtype="float32")
    dim = embeddings.shape[1]

    # 3. FAISS
    faiss.normalize_L2(embeddings)
    dense_index = faiss.IndexFlatIP(dim)
    dense_index.add(embeddings)
    faiss.write_index(dense_index, str(output_dir / "faiss.index"))

    # 4. BM25
    tokenized = _tokenize_for_bm25(embed_texts)
    bm25_index = BM25Okapi(tokenized)
    with open(output_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    # 5. Parquet
    chunks_df = pd.DataFrame([c.to_dict() for c in all_chunks])
    chunks_df["embedding"] = [e.tolist() for e in embeddings]
    chunks_df.to_parquet(output_dir / "chunks.parquet")

    sections_df = pd.DataFrame([s.to_dict() for s in all_sections])
    sections_df.to_parquet(output_dir / "sections.parquet")

    # 6. Manifest
    (output_dir / "manifest.json").write_text(json.dumps({
        "version": "0.1.0",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "embed_dim": dim,
        "chunk_size": chunk_size,
        "overlap_ratio": overlap_ratio,
        "num_chunks": len(all_chunks),
        "num_sections": len(all_sections),
        "source_file_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(cache_dir.glob("*.txt"))
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    return dense_index, bm25_index, chunks_df, sections_df


__all__ = ["build_image_specs", "build_rag_index", "init_rag"]
