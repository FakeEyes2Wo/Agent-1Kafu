"""在线 RAG 管线：混合检索 → 子块重排 → 父块扩展 → 证据校验 → 生成。"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

import numpy as np
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command

from ..schemas import IMAGE_TAG_RE, ImageSpec, turn_description_PIC
from ..utils import EmbeddingModelName, estimate_tokens, get_embedding_model, get_rerank_model
from .prompts import GENERATION_PROMPT, SUFFICIENCY_PROMPT


# ── config ───────────────────────────────────────────────────────────


@dataclass
class ServiceConfig:
    llm_model: str
    llm_url: str = ""
    llm_api_key: str = ""              # LLM 的 key（DeepSeek）
    dashscope_api_key: str = ""        # DashScope 的 key（embedding + rerank）
    model_timeout_seconds: float = 60
    embed_model: EmbeddingModelName = "text-embedding-v4"
    embedding_model_dir: str = ""
    embedding_query_prompt_name: str = ""
    # rerank
    reranker_model: str = ""           # "qwen3-rerank" / "BAAI/bge-reranker-v2-m3" / "" to skip
    reranker_backend: str = "local"    # "dashscope" | "local"
    # index
    index_dir: str = "rag_data/index"
    image_specs_path: str = "rag_data/image_specs.csv"
    dense_k: int = 40
    bm25_k: int = 40
    rerank_k: int = 30
    token_budget: int = 1024
    max_retries: int = 2


# ── state ────────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    query: str
    messages: Annotated[list[AnyMessage], add_messages]
    context_docs: list[dict]
    answer: str
    file_list: list[str]
    citations: list[str]


# ── Workflow ─────────────────────────────────────────────────────────


class Workflow:
    """两节点管线：retrieve（混合+重排+扩展）→ generate（校验+生成）。"""

    def __init__(self, cfg: ServiceConfig) -> None:
        self.cfg = cfg
        self.llm = init_chat_model(
            model=cfg.llm_model,
            model_provider="openai",
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_url or None,
            timeout=cfg.model_timeout_seconds,
        )
        self.embed_model = get_embedding_model(
            cfg.embed_model,
            api_key=cfg.dashscope_api_key or None,
            model_dir=cfg.embedding_model_dir or None,
            query_prompt_name=cfg.embedding_query_prompt_name or None,
        )
        self.reranker = self._init_reranker(cfg)
        self._load_indices()
        self._load_image_specs()
        self.graph = self._build_graph()

    # ── init helpers ─────────────────────────────────────────────────

    def _init_reranker(self, cfg: ServiceConfig):
        return get_rerank_model(
            cfg.reranker_backend,
            cfg.reranker_model,
            api_key=cfg.dashscope_api_key,
        )

    def _load_indices(self) -> None:
        import faiss
        idx = Path(self.cfg.index_dir)
        self.dense_index = faiss.read_index(str(idx / "faiss.index"))
        with open(idx / "bm25_index.pkl", "rb") as f:
            self.bm25_index = pickle.load(f)
        self.chunks_df = pd.read_parquet(idx / "chunks.parquet")
        self.sections_df = pd.read_parquet(idx / "sections.parquet")

    def _load_image_specs(self) -> None:
        specs = pd.read_csv(self.cfg.image_specs_path)
        self.img_pairs: dict[str, ImageSpec] = {}
        self.img_label: dict[int, str] = {}
        for row in specs.itertuples():
            self.img_pairs[row.imgname] = ImageSpec(
                imgname=row.imgname, description=row.description,
            )
            self.img_label[row.uniqueID] = Path(row.imgname).stem

    # ── graph ────────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        return g.compile()

    # ── retrieve ─────────────────────────────────────────────────────

    def _retrieve(self, state: AgentState) -> Command:
        q = state["query"]
        candidates = self._hybrid_search(q)
        if self.reranker is not None:
            candidates = self._rerank(q, candidates)
        return Command(update={"context_docs": self._expand(candidates)})

    def _hybrid_search(self, query: str) -> list[dict]:
        import faiss as _faiss
        from .index import _tokenize_for_bm25

        q_vec = np.array([self.embed_model.embed_query(query)], dtype="float32")
        _faiss.normalize_L2(q_vec)
        _, d_idx = self.dense_index.search(q_vec, self.cfg.dense_k)

        bm25_scores = self.bm25_index.get_scores(_tokenize_for_bm25([query])[0])
        bm25_top = list(reversed(bm25_scores.argsort()[-self.cfg.bm25_k:]))

        rrf: dict[int, float] = {}
        for rank, idx in enumerate(d_idx[0]):
            if idx != -1:
                rrf[int(idx)] = rrf.get(int(idx), 0) + 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(bm25_top):
            rrf[idx] = rrf.get(idx, 0) + 1.0 / (60 + rank + 1)

        results = []
        for cid in sorted(rrf, key=rrf.get, reverse=True):
            row = self.chunks_df.iloc[cid].to_dict()
            row["rrf_score"] = rrf[cid]
            results.append(row)
        return results

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        pairs = [(query, c["content"]) for c in candidates]
        scores = self.reranker.compute_score(pairs)  # type: ignore[union-attr]
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
        candidates.sort(key=lambda c: c["score"], reverse=True)
        top = candidates[:self.cfg.rerank_k]

        by_section: dict[str, list[dict]] = {}
        for c in candidates:
            by_section.setdefault(c["section_id"], []).append(c)

        expanded: dict[str, dict] = {}
        for c in top:
            expanded[c["chunk_id"]] = c
            nb_list = by_section.get(c["section_id"], [])
            for i, nb in enumerate(nb_list):
                if nb["chunk_id"] == c["chunk_id"]:
                    if i > 0:
                        expanded[nb_list[i - 1]["chunk_id"]] = nb_list[i - 1]
                    if i < len(nb_list) - 1:
                        expanded[nb_list[i + 1]["chunk_id"]] = nb_list[i + 1]
                    break
        return sorted(expanded.values(), key=lambda c: c.get("score", 0), reverse=True)

    def _expand(self, children: list[dict]) -> list[dict]:
        selected: dict[str, dict] = {}
        tokens_used = 0
        for child in children:
            sid = child.get("section_id")
            if not sid or sid in selected:
                continue
            rows = self.sections_df[self.sections_df["section_id"] == sid]
            if rows.empty:
                continue
            sec = rows.iloc[0].to_dict()
            cost = estimate_tokens(sec["full_text"])
            if tokens_used + cost > self.cfg.token_budget:
                if not selected:
                    selected[sid] = {**sec, "truncated": True, "score": child.get("score", child.get("rrf_score", 0))}
                continue
            selected[sid] = {
                "section_id": sec["section_id"],
                "manual_name": sec["manual_name"],
                "heading_path": sec["heading_path"],
                "full_text": sec["full_text"],
                "image_ids": sec["image_ids"],
                "score": child.get("score", child.get("rrf_score", 0)),
            }
            tokens_used += cost
        return sorted(selected.values(), key=lambda s: s["score"], reverse=True)

    # ── generate ─────────────────────────────────────────────────────

    def _generate(self, state: AgentState) -> Command:
        q = state["query"]
        docs = state.get("context_docs", [])

        if not docs:
            return Command(update={
                "answer": "未找到相关手册内容，请提供更多产品信息。",
                "file_list": [], "citations": [],
                "messages": [AIMessage(content="未找到相关手册内容。")],
            })

        check = self._check_sufficiency(q, docs[:3])
        if not check["sufficient"]:
            docs = _merge_unique(docs, self._retry_search(check.get("rewrite", q)))

        return self._build_answer(q, docs)

    def _check_sufficiency(self, query: str, docs: list[dict]) -> dict:
        ctx_text = "\n\n---\n\n".join(
            f"[{d['manual_name']}] {d['heading_path']}\n{d['full_text'][:1500]}"
            for d in docs
        )
        resp = self.llm.invoke([
            SystemMessage(content="Output JSON only."),
            HumanMessage(content=SUFFICIENCY_PROMPT.format(context=ctx_text, query=query)),
        ])
        try:
            return json.loads(_extract_json(resp.content))  # type: ignore[arg-type]
        except (json.JSONDecodeError, KeyError):
            return {"sufficient": True, "missing": None, "rewrite": None}

    def _retry_search(self, query: str) -> list[dict]:
        candidates = self._hybrid_search(query)
        if self.reranker is not None:
            candidates = self._rerank(query, candidates)
        return self._expand(candidates)

    def _build_answer(self, query: str, docs: list[dict]) -> Command:
        ctx_text = "\n\n---\n\n".join(
            f"[{d['manual_name']} — {d['heading_path']}]\n{d['full_text']}"
            for i, d in enumerate(docs)
        )
        resp = self.llm.invoke([
            SystemMessage(content="Output JSON only."),
            HumanMessage(content=GENERATION_PROMPT.format(context=ctx_text, query=query)),
        ])
        try:
            gen = json.loads(_extract_json(resp.content))  # type: ignore[arg-type]
        except (json.JSONDecodeError, KeyError):
            gen = {"answer": resp.content, "evidence_ids": []}  # type: ignore[assignment]

        answer = gen.get("answer", resp.content)

        # 从上下文文本中提取有效 image id（比元数据更可靠，兼容旧索引）
        valid_ids = {int(m.group(1)) for m in IMAGE_TAG_RE.finditer(ctx_text)}
        used_ids = {int(m.group(1)) for m in IMAGE_TAG_RE.finditer(answer)}
        invalid_ids = used_ids - valid_ids
        if invalid_ids:
            for iid in invalid_ids:
                answer = re.sub(
                    rf"<image\s+(?:id\s*=\s*['\"]?)?{iid}['\"]?\s*>.*?</image>",
                    "", answer, flags=re.IGNORECASE | re.DOTALL,
                )

        # Image → PIC 后处理
        try:
            answer, file_list = turn_description_PIC(answer, self.img_pairs)
        except KeyError:
            file_list = []

        return Command(update={
            "answer": answer, "file_list": file_list,
            "citations": gen.get("evidence_ids", []),
            "messages": [AIMessage(content=answer)],
        })

    def _render(self, answer: str, image_ids: list[int]) -> tuple[str, list[str]]:
        fl = [self.img_label[i] for i in image_ids if i in self.img_label]
        if fl:
            answer = answer.rstrip() + "\n\n" + " ".join("<PIC>" for _ in fl)
        return answer, fl

    def run(self, query: str) -> dict[str, Any]:
        return self.graph.invoke({"query": query, "messages": []})


# ── helpers ──────────────────────────────────────────────────────────


def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group() if m else text


def _merge_unique(a: list[dict], b: list[dict]) -> list[dict]:
    seen = {d["section_id"] for d in a}
    return a + [d for d in b if d["section_id"] not in seen]


__all__ = ["ServiceConfig", "Workflow"]
