from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    kafu_api_token: str = "change-me"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o-mini"
    vision_model_url: str = ""
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_backend: str = "sentence_transformers"
    embedding_query_prompt_name: str = "query"
    embedding_model_dir: Path = Field(default=Path("./storage/models"))

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    data_dir: Path = Field(default=Path("./data"))
    manual_dir: Path = Field(default=Path("./data/手册"))
    image_dir: Path = Field(default=Path("./data/手册/插图"))
    vectorstore_dir: Path = Field(default=Path("./storage/vectorstore"))
    llamaindex_dir: Path = Field(default=Path("./storage/llamaindex"))

    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 8
    rag_backend: str = "llamaindex"
    retrieval_top_k: int = 20
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
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
    def has_openai_key(self) -> bool:
        key = self.openai_api_key.strip()
        return bool(key and key != "your-api-key")

    @property
    def vision_base_url(self) -> str:
        return self.vision_model_url.strip() or self.openai_base_url

    @property
    def use_openai_embeddings(self) -> bool:
        return self.embedding_backend.strip().lower() == "openai"

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
