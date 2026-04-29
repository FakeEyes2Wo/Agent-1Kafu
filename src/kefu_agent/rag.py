import ast
import hashlib
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from .config import PROJECT_ROOT, Settings, get_settings


MANUAL_LANGUAGE_METADATA_KEY = "manual_language"
MANUAL_LANGUAGE_FILTER_VERSION = "1"
RAG_CONTEXT_FORMAT_VERSION = "3"


@dataclass
class Chunk:
    id: str
    manual: str
    title: str
    text: str
    image_ids: list[str]
    vector: list[float]
    manual_language: str = "zh"


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


def build_index() -> int:
    settings = get_settings()
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(load_manual_chunks())
    if _rag_backend(settings) != "legacy":
        settings.llamaindex_dir.mkdir(parents=True, exist_ok=True)
        nodes = _manual_chunks_to_nodes(chunks)
        if nodes:
            from llama_index.core import VectorStoreIndex

            embed_model = _llama_embed_model_cached(
                settings.embedding_model, str(settings.embedding_model_dir)
            )
            index = VectorStoreIndex(nodes, embed_model=embed_model)
            index.storage_context.persist(persist_dir=str(settings.llamaindex_dir))
        _write_index_metadata(settings, [])
        return len(chunks)

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
    final_limit = top_k or settings.top_k
    if final_limit <= 0:
        return []
    manual_language = _query_manual_language(query)

    if _rag_backend(settings) == "legacy":
        if not settings.index_path.exists() or not _index_metadata_matches(settings):
            build_index()
            _load_index.cache_clear()
        chunks = _load_index(str(settings.index_path))
        chunks = _filter_chunks_by_manual_language(chunks, manual_language)
        if not chunks:
            return []
        return _rank_chunks(get_embeddings().embed_query(query), chunks, final_limit)

    if (
        not settings.llamaindex_dir.exists()
        or not any(settings.llamaindex_dir.iterdir())
        or not _index_metadata_matches(settings)
    ):
        if build_index() == 0:
            return []
        _load_llama_index.cache_clear()

    index = _load_llama_index(
        str(settings.llamaindex_dir),
        settings.embedding_model,
        str(settings.embedding_model_dir),
    )
    initial_k = final_limit
    if settings.rerank_enabled:
        initial_k = max(settings.retrieval_top_k, final_limit)
    nodes = index.as_retriever(
        similarity_top_k=initial_k,
        filters=_manual_language_filters(manual_language),
    ).retrieve(query)

    if settings.rerank_enabled:
        nodes = _rerank_nodes(query, nodes, top_k or settings.rerank_top_n)

    return [_node_to_chunk(node) for node in nodes[:final_limit]]


def _manual_chunks_to_nodes(chunks: list[dict]) -> list[Any]:
    from llama_index.core.schema import TextNode

    return [
        TextNode(
            id_=chunk["id"],
            text=chunk["text"],
            metadata={
                "manual": chunk["manual"],
                "title": chunk["title"],
                "image_ids": json.dumps(chunk["image_ids"], ensure_ascii=False),
                MANUAL_LANGUAGE_METADATA_KEY: chunk.get(
                    "manual_language", _manual_language(chunk["manual"])
                ),
            },
        )
        for chunk in chunks
    ]


@lru_cache(maxsize=4)
def _llama_embed_model_cached(model_name: str, cache_folder: str) -> Any:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=_resolve_hf_model_name(model_name, cache_folder),
        cache_folder=cache_folder,
        trust_remote_code=True,
    )


@lru_cache(maxsize=2)
def _load_llama_index(persist_dir: str, embedding_model: str, model_dir: str) -> Any:
    from llama_index.core import StorageContext, load_index_from_storage

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    embed_model = _llama_embed_model_cached(embedding_model, model_dir)
    return load_index_from_storage(storage_context, embed_model=embed_model)


def _rerank_nodes(query: str, nodes: list[Any], top_n: int) -> list[Any]:
    if not nodes or top_n <= 0:
        return []

    pairs = [(query, _node_text(node)) for node in nodes]
    settings = get_settings()
    scores = _get_reranker(
        settings.rerank_model, str(settings.embedding_model_dir)
    ).predict(pairs)
    ranked = sorted(zip(nodes, scores, strict=False), key=lambda item: item[1], reverse=True)
    return [node for node, _score in ranked[:top_n]]


@lru_cache(maxsize=2)
def _get_reranker(model_name: str, cache_folder: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        _resolve_hf_model_name(model_name, cache_folder),
        cache_folder=cache_folder,
        trust_remote_code=True,
    )


def _node_to_chunk(node_with_score: Any) -> Chunk:
    node = getattr(node_with_score, "node", node_with_score)
    metadata = dict(getattr(node, "metadata", {}) or {})
    return Chunk(
        id=str(
            getattr(node, "node_id", None)
            or getattr(node, "id_", "")
            or metadata.get("id", "")
        ),
        manual=str(metadata.get("manual", "")),
        title=str(metadata.get("title", "")),
        text=_node_text(node),
        image_ids=_metadata_image_ids(metadata.get("image_ids")),
        vector=[],
        manual_language=str(metadata.get(MANUAL_LANGUAGE_METADATA_KEY, "zh")),
    )


def _node_text(node_with_score: Any) -> str:
    node = getattr(node_with_score, "node", node_with_score)
    if hasattr(node, "get_content"):
        try:
            return str(node.get_content(metadata_mode="none"))
        except TypeError:
            return str(node.get_content())
    return str(getattr(node, "text", ""))


def _metadata_image_ids(value: Any) -> list[str]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


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
        text = _inline_pic_image_tags(chunk.text, chunk.image_ids)
        image_text = (
            "\n可用图片：" + json.dumps(chunk.image_ids, ensure_ascii=False)
            if chunk.image_ids
            else ""
        )
        blocks.append(f"[{i}] 来源：{chunk.manual} / {chunk.title}\n{text}{image_text}")
    return "\n\n".join(blocks)


def format_answer_with_image_list(answer: str, contexts: str) -> str:
    body, explicit_image_ids = _strip_trailing_image_list(answer.strip())
    body, tagged_image_ids = _strip_named_pic_tags(body)
    body, inline_image_ids = _strip_inline_pic_image_ids(body)
    body = _normalize_pic_placeholders(body).strip()
    pic_count = body.count("<PIC>")
    if pic_count <= 0:
        return body

    image_ids = _ordered_unique(tagged_image_ids + inline_image_ids + explicit_image_ids)
    for image_id in _context_image_ids(contexts):
        if image_id not in image_ids:
            image_ids.append(image_id)
        if len(image_ids) >= pic_count:
            break
    if not image_ids:
        return body
    return f"{body},{json.dumps(image_ids[:pic_count], ensure_ascii=False)}"


def _inline_pic_image_tags(text: str, image_ids: list[str]) -> str:
    image_iter = iter(image_ids)
    text = _normalize_pic_placeholders(text)

    def replace_pic(match: re.Match[str]) -> str:
        image_id = next(image_iter, None)
        if image_id is None:
            return match.group(0)
        return f"<PIC> {image_id} </PIC>"

    return re.sub(r"<PIC>", replace_pic, text)


def _strip_trailing_image_list(answer: str) -> tuple[str, list[str]]:
    match = re.search(r"\s*[,，]\s*(\[[^\[\]]*\])\s*$", answer, flags=re.S)
    if not match:
        return answer, []
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return answer, []
    image_ids = [str(item) for item in value] if isinstance(value, list) else []
    return answer[: match.start()].rstrip(), image_ids


def _strip_inline_pic_image_ids(text: str) -> tuple[str, list[str]]:
    image_ids: list[str] = []

    def replace_pic(match: re.Match[str]) -> str:
        image_id = match.group(1).strip()
        if image_id:
            image_ids.append(image_id)
        return "<PIC>"

    normalized = re.sub(r"<PIC\s+图片ID[:：]\s*([^>]+)>", replace_pic, text)
    return normalized, image_ids


def _strip_named_pic_tags(text: str) -> tuple[str, list[str]]:
    image_ids: list[str] = []

    def replace_pic(match: re.Match[str]) -> str:
        image_id = match.group(1).strip()
        if image_id:
            image_ids.append(image_id)
        return "<PIC>"

    normalized = re.sub(
        r"<\s*PIC\s*>\s*([^<>]+?)\s*<\s*/\s*PIC\s*>",
        replace_pic,
        text,
        flags=re.I,
    )
    return normalized, image_ids


def _normalize_pic_placeholders(text: str) -> str:
    return re.sub(r"<\s*PIC\s*>", "<PIC>", text)


def _context_image_ids(contexts: str) -> list[str]:
    image_ids: list[str] = []
    for match in re.finditer(r"<\s*PIC\s*>\s*([^<>]+?)\s*<\s*/\s*PIC\s*>", contexts, re.I):
        image_id = match.group(1).strip()
        if image_id:
            image_ids.append(image_id)
    for match in re.finditer(r"(?:可用图片|配图顺序)：(\[[^\[\]]*\])", contexts):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            image_ids.extend(str(item) for item in value)
    return image_ids


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


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
        "rag_backend": _rag_backend(settings),
        "retrieval_top_k": str(settings.retrieval_top_k),
        "rerank_enabled": str(settings.rerank_enabled),
        "rerank_model": settings.rerank_model,
        "rerank_top_n": str(settings.rerank_top_n),
        "manual_language_filter_version": MANUAL_LANGUAGE_FILTER_VERSION,
    }


def _rag_backend(settings: Settings) -> str:
    backend = settings.rag_backend.strip().lower()
    if backend not in {"llamaindex", "legacy"}:
        raise RuntimeError(f"Unsupported RAG_BACKEND: {settings.rag_backend}")
    return backend


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
                "manual_language": _manual_language(manual),
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


def _manual_language(manual: str) -> str:
    normalized = manual.lower()
    return "en" if "英文" in normalized or "english" in normalized else "zh"


def _query_manual_language(query: str) -> str:
    return "en" if _is_english_query(query) else "zh"


def _is_english_query(text: str) -> bool:
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return False
    return len(re.findall(r"[A-Za-z]", text)) >= 3


def _filter_chunks_by_manual_language(
    chunks: list[Chunk], manual_language: str
) -> list[Chunk]:
    return [
        chunk
        for chunk in chunks
        if getattr(chunk, MANUAL_LANGUAGE_METADATA_KEY, "zh") == manual_language
    ]


def _manual_language_filters(manual_language: str) -> Any:
    from llama_index.core.vector_stores import (
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )

    return MetadataFilters(
        filters=[
            MetadataFilter(
                key=MANUAL_LANGUAGE_METADATA_KEY,
                operator=FilterOperator.EQ,
                value=manual_language,
            )
        ]
    )
