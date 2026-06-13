"""离线索引构建：chunk → embed → FAISS + BM25。"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .chunk import make_search_chunks, split_by_headings
from .schemas import Section, extract_image_ids, sha8


def _tokenize_for_bm25(texts: list[str]) -> list[list[str]]:
    """BM25 分词：中文单字+2-gram，英文按空白/标点。"""
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
    cache_dir: str | Path,
    embed_model,
    output_dir: str | Path,
    chunk_size: int = 256,
    overlap_ratio: float = 0.1,
) -> tuple:
    """
    从 cache_dir/*.txt 构建 FAISS + BM25 双索引。

    返回 (dense_index, bm25_index, chunks_df, sections_df)。
    """
    import faiss
    import numpy as np
    from rank_bm25 import BM25Okapi

    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 切分
    all_sections: list[Section] = []
    all_chunks: list = []

    for txt_path in sorted(cache_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        name = txt_path.name
        for si, (hp, block_text) in enumerate(split_by_headings(text)):
            sid = f"{sha8(name)}:s{si}"
            scs = make_search_chunks(block_text, hp, name, sid, chunk_size, overlap_ratio)
            all_sections.append(Section(
                section_id=sid, manual_name=name, heading_path=hp,
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
        "embed_dim": dim, "chunk_size": chunk_size,
        "overlap_ratio": overlap_ratio,
        "num_chunks": len(all_chunks),
        "num_sections": len(all_sections),
        "source_file_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(cache_dir.glob("*.txt"))
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    return dense_index, bm25_index, chunks_df, sections_df


__all__ = ["build_rag_index"]
