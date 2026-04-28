import ast
import hashlib
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from .config import Settings, get_settings


@dataclass
class Chunk:
    id: str
    manual: str
    title: str
    text: str
    image_ids: list[str]
    vector: list[float]


class HashEmbeddings(Embeddings):
    """Small deterministic embedding fallback for offline tests and demos."""

    dim: int = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _tokens(text)
        for token in tokens:
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
            if self.model_dir:
                self.model_dir.mkdir(parents=True, exist_ok=True)
                kwargs["cache_folder"] = str(self.model_dir)
                if _has_hf_cache_snapshot(self.model_dir, self.model_name):
                    kwargs["local_files_only"] = True
            try:
                self._model = SentenceTransformer(self.model_name, **kwargs)
            except Exception:
                if not kwargs.get("local_files_only"):
                    raise
                kwargs.pop("local_files_only")
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


def build_index() -> int:
    settings = get_settings()
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(load_manual_chunks())
    embeddings = get_embeddings()
    texts = [chunk["text"] for chunk in chunks]
    vectors = embeddings.embed_documents(texts) if texts else []

    with settings.index_path.open("w", encoding="utf-8") as f:
        for chunk, vector in zip(chunks, vectors, strict=False):
            item = Chunk(vector=vector, **chunk)
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    _write_index_metadata(settings, vectors[0] if vectors else [])
    return len(chunks)


def retrieve(query: str, top_k: int | None = None) -> list[Chunk]:
    settings = get_settings()
    if not settings.index_path.exists() or not _index_metadata_matches(settings):
        build_index()
        _load_index.cache_clear()
    chunks = _load_index(str(settings.index_path))
    if not chunks:
        return []

    query_vector = get_embeddings().embed_query(query)
    return _rank_chunks(query_vector, chunks, top_k or settings.top_k)


def _rank_chunks(query_vector: list[float], chunks: list[Chunk], top_k: int) -> list[Chunk]:
    if not query_vector or not chunks or top_k <= 0:
        return []

    corpus = np.asarray([chunk.vector for chunk in chunks], dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    if corpus.ndim != 2 or query.ndim != 1 or corpus.shape[1] != query.shape[0]:
        raise RuntimeError(
            "Query vector dimension does not match the index. Rebuild the vector index."
        )

    query_norm = np.linalg.norm(query) or 1.0
    corpus_norms = np.linalg.norm(corpus, axis=1)
    corpus_norms[corpus_norms == 0] = 1.0
    scores = (corpus @ query) / (corpus_norms * query_norm)

    limit = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[-limit:][::-1]
    return [chunks[int(index)] for index in top_indices]


def format_contexts(chunks: Iterable[Chunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        image_text = f" 图片ID：{', '.join(chunk.image_ids)}" if chunk.image_ids else ""
        blocks.append(
            f"[{i}] 来源：{chunk.manual} / {chunk.title}{image_text}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def _index_metadata_matches(settings: Settings) -> bool:
    if not settings.index_meta_path.exists():
        return False
    try:
        metadata = json.loads(settings.index_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = _embedding_signature(settings)
    return all(metadata.get(key) == value for key, value in expected.items())


def _write_index_metadata(settings: Settings, sample_vector: list[float]) -> None:
    metadata = _embedding_signature(settings)
    metadata["vector_dim"] = len(sample_vector)
    settings.index_meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _embedding_signature(settings: Settings) -> dict[str, str]:
    return {
        "embedding_backend": settings.embedding_backend.strip().lower(),
        "embedding_model": settings.embedding_model,
        "embedding_query_prompt_name": settings.embedding_query_prompt_name,
    }


def _has_hf_cache_snapshot(cache_dir: Path, model_name: str) -> bool:
    repo_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return False
    return any(path.is_dir() for path in snapshots_dir.iterdir())


def load_manual_chunks() -> Iterable[dict]:
    settings = get_settings()
    for path in sorted(settings.manual_dir.glob("*.txt")):
        text, image_ids = parse_manual(path)
        yield from split_manual(path.stem, text, image_ids)


def parse_manual(path: Path) -> tuple[str, list[str]]:
    raw = path.read_text(encoding="utf-8").strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                value = parser(raw)
            if isinstance(value, list) and value:
                body = str(value[0])
                images = value[1] if len(value) > 1 and isinstance(value[1], list) else []
                return body, [str(item) for item in images]
        except Exception:
            pass
    return raw, []


def split_manual(manual: str, text: str, image_ids: list[str]) -> Iterable[dict]:
    settings = get_settings()
    text = re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()
    sections = _sections(text)
    image_cursor = 0
    chunk_no = 0
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    for title, body in sections:
        for chunk_text in splitter.split_text(body):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            pic_count = chunk_text.count("<PIC>")
            refs = image_ids[image_cursor : image_cursor + pic_count]
            image_cursor += pic_count
            chunk_no += 1
            yield {
                "id": f"{manual}-{chunk_no}",
                "manual": manual,
                "title": title,
                "text": chunk_text,
                "image_ids": refs,
            }


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    length = min(left.shape[0], right.shape[0])
    left = left[:length]
    right = right[:length]
    return float(
        np.dot(left, right)
        / ((np.linalg.norm(left) or 1.0) * (np.linalg.norm(right) or 1.0))
    )


@lru_cache(maxsize=1)
def _load_index(path: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?=#\s*)", text)
    sections: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"#\s*([^#]{1,80})", part)
        title = match.group(1).strip() if match else "正文"
        sections.append((title, part))
    return sections or [("正文", text)]


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+", lowered)
    chars = [ch for ch in lowered if "\u4e00" <= ch <= "\u9fff"]
    grams = chars + ["".join(chars[i : i + 2]) for i in range(max(0, len(chars) - 1))]
    return words + grams
