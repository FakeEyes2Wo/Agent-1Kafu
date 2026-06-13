"""编排层：配置构建、单例管理、对外接口。"""

from __future__ import annotations

import asyncio
import time
import uuid

from .config import get_settings
from .rag.retrieval import ServiceConfig, Workflow


# ── 配置构建 ──────────────────────────────────────────────────────────


def _service_config() -> ServiceConfig:
    settings = get_settings()
    return ServiceConfig(
        llm_model=settings.chat_model,
        llm_url=settings.openai_base_url,
        llm_api_key=settings.openai_api_key,
        dashscope_api_key=settings.dashscope_api_key,
        model_timeout_seconds=settings.model_timeout_seconds,
        embed_model=settings.embedding_model,
        embedding_model_dir=str(settings.embedding_model_dir),
        embedding_query_prompt_name=settings.embedding_query_prompt_name,
        reranker_model=settings.rerank_model if settings.rerank_enabled else "",
        reranker_backend=settings.rerank_backend,
        index_dir=str(settings.rag_data_dir / "index"),
        image_specs_path=str(settings.rag_data_dir / "image_specs.csv"),
        dense_k=settings.retrieval_top_k,
        bm25_k=settings.retrieval_top_k,
        rerank_k=settings.rerank_top_n,
        token_budget=settings.token_budget,
    )


_workflow: Workflow | None = None


def get_workflow() -> Workflow:
    global _workflow
    if _workflow is None:
        _workflow = Workflow(_service_config())
    return _workflow


# ── 对外接口 ──────────────────────────────────────────────────────────


def answer_question(
    question: str,
    images: list[str] | None = None,
    session_id: str | None = None,
    contexts: str | None = None,
) -> tuple[str, str, list[str]]:
    del images, contexts
    result = get_workflow().run(question)
    sid = session_id or f"kf_session_{uuid.uuid4().hex}"
    return str(result["answer"]), sid, result.get("file_list", [])


async def answer_question_async(
    question: str,
    images: list[str] | None = None,
    session_id: str | None = None,
    contexts: str | None = None,
) -> tuple[str, str, list[str]]:
    return await asyncio.to_thread(
        answer_question,
        question,
        images=images,
        session_id=session_id,
        contexts=contexts,
    )


def response_payload(answer: str, session_id: str, file_list: list[str] | None = None) -> dict:
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "answer": answer,
            "session_id": session_id,
            "timestamp": int(time.time()),
            "file_list": file_list or [],
        },
    }


__all__ = [
    "answer_question", "answer_question_async", "get_workflow", "response_payload",
]
