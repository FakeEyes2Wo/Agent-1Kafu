import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from ..config import PROJECT_ROOT


MANUAL_LANGUAGE_METADATA_KEY = "manual_language"
MANUAL_LANGUAGE_FILTER_VERSION = "1"
MANUAL_PIC_TAG_VERSION = "1"
HYBRID_SEARCH_VERSION = "2"
VISUAL_RETRIEVER_VERSION = "1"
RAG_CONTEXT_FORMAT_VERSION = "7"
DEFAULT_MANUAL_LANGUAGE = "zh"
MANUAL_LANGUAGE_RULES = (("en", ("\u82f1\u6587", "english")),)
T = TypeVar("T")


@dataclass
class Chunk:
    id: str
    manual: str
    title: str
    text: str
    image_ids: list[str]
    vector: list[float]
    manual_language: str = DEFAULT_MANUAL_LANGUAGE


def manual_language(manual: str) -> str:
    normalized = manual.lower()
    for language, patterns in MANUAL_LANGUAGE_RULES:
        if any(pattern.lower() in normalized for pattern in patterns):
            return language
    return DEFAULT_MANUAL_LANGUAGE


def query_manual_language(query: str) -> str:
    if any("\u4e00" <= ch <= "\u9fff" for ch in query):
        return "zh"
    return "en" if len(re.findall(r"[A-Za-z]", query)) >= 3 else "zh"


def filter_chunks_by_manual_language(chunks: list[Any], language: str) -> list[Any]:
    return [
        chunk for chunk in chunks
        if getattr(chunk, MANUAL_LANGUAGE_METADATA_KEY, DEFAULT_MANUAL_LANGUAGE) == language
    ]


def manual_language_filters(language: str) -> Any:
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
                value=language,
            )
        ]
    )


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+", lowered)
    chars = [ch for ch in lowered if "\u4e00" <= ch <= "\u9fff"]
    grams = chars + ["".join(chars[i : i + 2]) for i in range(max(0, len(chars) - 1))]
    return words + grams


def lexical_score(query: str, text: str) -> float:
    query_terms = set(_tokens(query))
    if not query_terms:
        return 0.0
    text_terms = set(_tokens(text))
    overlap = query_terms & text_terms
    if not overlap:
        return 0.0
    phrase_bonus = sum(
        2.0
        for term in query_terms
        if len(term) >= 4 and term in text.lower()
    )
    return len(overlap) / math.sqrt(len(text_terms) or 1) + phrase_bonus


def rank_by_score(items: list[T], score_fn: Callable[[T], float], top_k: int) -> list[T]:
    if top_k <= 0:
        return []
    return [
        item for item, _score in sorted(
            ((item, score_fn(item)) for item in items),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if _score > 0
    ][:top_k]


def parse_json_value(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def read_json_file(path: Path) -> Any | None:
    try:
        return parse_json_value(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def coerce_string_list(value: Any) -> list[str]:
    parsed = parse_json_value(value) if isinstance(value, str) else value
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


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
    config = read_json_file(config_path)
    if not isinstance(config, dict):
        return False
    return bool(config.get("model_type"))
