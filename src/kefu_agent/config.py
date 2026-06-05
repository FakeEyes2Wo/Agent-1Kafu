from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER_API_KEYS = {
    "your-api-key",
    "your-dashscope-api-key",
    "your-bailian-api-key",
    "sk-...",
}


def _configured_api_key(*keys: str) -> str:
    for key in keys:
        value = key.strip()
        if value and value.lower() not in PLACEHOLDER_API_KEYS:
            return value
    return ""


def _configured_api_key_source(named_keys: tuple[tuple[str, str], ...]) -> str:
    for name, key in named_keys:
        if _configured_api_key(key):
            return name
    return ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    kafu_api_token: str = "change-me"

    bailian_api_key: str = ""
    dashscope_api_key: str = ""
    bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_api_key: str = ""
    openai_base_url: str = ""
    chat_model: str = "qwen3.7-plus-2026-05-26"
    chat_enable_thinking: bool = False
    chat_max_tokens: int = 450
    chat_thinking_budget: int = 0
    vision_model: str = "qwen3.7-plus-2026-05-26"
    vision_model_url: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_backend: str = "openai"
    embedding_query_prompt_name: str = ""
    embedding_model_dir: Path = Field(default=Path("./storage/models"))
    embedding_batch_size: int = 10

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    data_dir: Path = Field(default=Path("./data"))
    manual_dir: Path = Field(default=Path("./data/KownledgeBase/手册"))
    image_dir: Path = Field(default=Path("./data/KownledgeBase/手册/插图"))
    vectorstore_dir: Path = Field(default=Path("./storage/vectorstore"))
    llamaindex_dir: Path = Field(default=Path("./storage/llamaindex"))

    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 8
    rag_backend: str = "hybrid"
    retrieval_top_k: int = 20
    rerank_enabled: bool = False
    rerank_backend: str = "none"
    rerank_model: str = "qwen3-vl-rerank"
    rerank_top_n: int = 8
    visual_retriever: str = "lexical"
    visual_top_k: int = 8
    model_timeout_seconds: float = 60

    def model_post_init(self, __context: object) -> None:
        for name in (
            "data_dir",
            "manual_dir",
            "image_dir",
            "vectorstore_dir",
            "llamaindex_dir",
            "embedding_model_dir",
        ):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                path = (PROJECT_ROOT / path).resolve()
            setattr(self, name, path)

    @property
    def index_path(self) -> Path:
        return self.vectorstore_dir / "index.jsonl"

    @property
    def index_meta_path(self) -> Path:
        return self.vectorstore_dir / "index_meta.json"

    @property
    def has_model_api_key(self) -> bool:
        return bool(self.model_api_key)

    @property
    def has_openai_key(self) -> bool:
        return self.has_model_api_key

    @property
    def model_api_key(self) -> str:
        return _configured_api_key(
            self.bailian_api_key,
            self.dashscope_api_key,
            self.openai_api_key,
        )

    @property
    def model_api_key_source(self) -> str:
        return _configured_api_key_source(
            (
                ("bailian", self.bailian_api_key),
                ("dashscope", self.dashscope_api_key),
                ("openai", self.openai_api_key),
            )
        )

    @property
    def model_base_url(self) -> str:
        if self.openai_base_url.strip():
            return self.openai_base_url.strip()
        if self.model_api_key_source == "openai":
            return "https://api.openai.com/v1"
        return (
            self.bailian_base_url.strip()
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    @property
    def vision_base_url(self) -> str:
        return self.vision_model_url.strip() or self.model_base_url

    @property
    def use_openai_embeddings(self) -> bool:
        return self.embedding_backend.strip().lower() in {
            "openai",
            "bailian",
            "dashscope",
        }

    @property
    def use_sentence_transformer_embeddings(self) -> bool:
        return self.embedding_backend.strip().lower() in {
            "sentence_transformer",
            "sentence_transformers",
            "huggingface",
            "hf",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
