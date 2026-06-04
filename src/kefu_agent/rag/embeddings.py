import hashlib
import math
from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from ..config import get_settings
from .core import _repo_id_from_hf_cache_dir, _resolve_hf_model_name, _tokens


class HashEmbeddings(Embeddings):
    """Small deterministic embedding fallback for offline tests and demos."""

    dim: int = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str,
        query_prompt_name: str = "query",
        model_dir: Path | str | None = None,
    ) -> None:
        self.model_name = model_name
        self.query_prompt_name = query_prompt_name
        self.model_dir = Path(model_dir) if model_dir else None

    @property
    def model(self):
        if not hasattr(self, "_model"):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install sentence-transformers to use "
                    "EMBEDDING_BACKEND=sentence_transformers."
                ) from exc
            kwargs = {"trust_remote_code": True}
            model_name = self.model_name
            if self.model_dir:
                self.model_dir.mkdir(parents=True, exist_ok=True)
                kwargs["cache_folder"] = str(self.model_dir)
                model_name = _resolve_hf_model_name(self.model_name, self.model_dir)
            try:
                self._model = SentenceTransformer(model_name, **kwargs)
            except Exception:
                if model_name == self.model_name or _repo_id_from_hf_cache_dir(
                    Path(self.model_name).expanduser()
                ):
                    raise
                self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        kwargs = {}
        if self.query_prompt_name:
            kwargs["prompt_name"] = self.query_prompt_name
        return self._encode([text], **kwargs)[0]

    def _encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            **kwargs,
        )
        return embeddings.tolist()


def get_embeddings() -> Embeddings:
    settings = get_settings()
    return _get_embeddings(
        settings.embedding_backend,
        settings.embedding_model,
        settings.embedding_query_prompt_name,
        str(settings.embedding_model_dir),
        settings.openai_api_key,
        settings.openai_base_url,
        settings.has_openai_key,
    )


@lru_cache(maxsize=4)
def _get_embeddings(
    backend: str,
    model_name: str,
    query_prompt_name: str,
    model_dir: str,
    openai_api_key: str,
    openai_base_url: str,
    has_openai_key: bool,
) -> Embeddings:
    backend = backend.strip().lower()
    if backend in {"sentence_transformer", "sentence_transformers", "huggingface", "hf"}:
        return SentenceTransformerEmbeddings(model_name, query_prompt_name, model_dir)
    if backend == "openai":
        if not has_openai_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be configured when EMBEDDING_BACKEND=openai."
            )
        return OpenAIEmbeddings(
            model=model_name,
            api_key=openai_api_key,
            base_url=openai_base_url,
        )
    if backend != "hash":
        raise RuntimeError(f"Unsupported EMBEDDING_BACKEND: {backend}")
    return HashEmbeddings()
