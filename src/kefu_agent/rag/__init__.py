from ..config import get_settings
from ..utils import get_embedding_model
from .index import build_rag_index
from .retrieval import ServiceConfig, Workflow
from .schemas import HeadingPath, SearchChunk, Section


def build_index() -> int:
    settings = get_settings()
    embed_model = get_embedding_model(
        settings.embedding_model,
        api_key=settings.dashscope_api_key or None,
        model_dir=settings.embedding_model_dir,
        query_prompt_name=settings.embedding_query_prompt_name,
    )
    _, _, chunks, _ = build_rag_index(
        settings.rag_data_dir / "cache",
        embed_model,
        settings.rag_data_dir / "index",
        chunk_size=settings.chunk_size,
        overlap_ratio=settings.chunk_overlap / settings.chunk_size,
    )
    return len(chunks)


__all__ = [
    "HeadingPath", "SearchChunk", "Section", "ServiceConfig", "Workflow",
    "build_index", "build_rag_index",
]
