import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from kefu_agent.config import PROJECT_ROOT, get_settings

from .text import tokenize


class HashEmbeddings(Embeddings):
    """Small deterministic fallback for offline checks; not a production retriever."""

    dim: int = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
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
        settings.model_api_key,
        settings.model_base_url,
        settings.has_openai_key,
    )


@lru_cache(maxsize=8)
def _get_embeddings(
    backend: str,
    model_name: str,
    query_prompt_name: str,
    model_dir: str,
    api_key: str,
    base_url: str,
    has_api_key: bool,
) -> Embeddings:
    backend = backend.strip().lower()
    if backend in {"none", "disabled", "off"}:
        raise RuntimeError("Embedding backend is disabled.")
    if backend in {"sentence_transformer", "sentence_transformers", "huggingface", "hf"}:
        return SentenceTransformerEmbeddings(model_name, query_prompt_name, model_dir)
    if backend in {"openai", "bailian", "dashscope"}:
        if not has_api_key:
            raise RuntimeError(
                "Set BAILIAN_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY before "
                "using API embeddings."
            )
        return OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            check_embedding_ctx_length=False,
        )
    if backend != "hash":
        raise RuntimeError(f"Unsupported EMBEDDING_BACKEND: {backend}")
    return HashEmbeddings()


def _resolve_hf_model_name(model_name: str, model_dir: Path | str) -> str:
    local_path = _local_hf_model_path(model_name, Path(model_dir))
    if local_path:
        return str(local_path)
    return _repo_id_from_hf_cache_dir(Path(model_name).expanduser()) or model_name


def _local_hf_model_path(model_name: str, model_dir: Path) -> Path | None:
    configured_path = Path(model_name).expanduser()
    candidates = [configured_path]
    if not configured_path.is_absolute():
        candidates.append(PROJECT_ROOT / configured_path)
    for path in candidates:
        snapshot_path = _snapshot_model_path(path)
        if snapshot_path:
            return snapshot_path.resolve()
        if _is_hf_cache_repo_dir(path):
            continue
        if path.exists() and (path.is_file() or _is_hf_model_dir(path)):
            return path.resolve()

    repo_dir = model_dir / f"models--{model_name.replace('/', '--')}"
    return _snapshot_model_path(repo_dir)


def _snapshot_model_path(repo_dir: Path) -> Path | None:
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = [path for path in snapshots_dir.iterdir() if _is_hf_model_dir(path)]
    return max(snapshots, key=lambda path: path.stat().st_mtime) if snapshots else None


def _is_hf_model_dir(path: Path) -> bool:
    config_path = path / "config.json"
    return (
        path.is_dir()
        and not _is_hf_cache_repo_dir(path)
        and _config_has_model_type(config_path)
        and (
            (path / "modules.json").exists()
            or any(path.glob("*.safetensors"))
            or any(path.glob("pytorch_model*.bin"))
        )
    )


def _is_hf_cache_repo_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("models--") and (
        (path / "snapshots").exists() or (path / "blobs").exists()
    )


def _repo_id_from_hf_cache_dir(path: Path) -> str | None:
    if not path.name.startswith("models--"):
        return None
    parts = path.name.removeprefix("models--").split("--")
    return "/".join(parts) if len(parts) >= 2 and all(parts) else None


def _config_has_model_type(config_path: Path) -> bool:
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(config.get("model_type"))
