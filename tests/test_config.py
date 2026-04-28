from kefu_agent.config import PROJECT_ROOT, Settings


def test_vision_base_url_falls_back_to_openai_base_url():
    settings = Settings(
        _env_file=None,
        openai_base_url="https://chat.example/v1",
        vision_model_url="",
    )

    assert settings.vision_base_url == "https://chat.example/v1"


def test_vision_base_url_uses_dedicated_url():
    settings = Settings(
        _env_file=None,
        openai_base_url="https://chat.example/v1",
        vision_model_url="https://vision.example/v1",
    )

    assert settings.vision_base_url == "https://vision.example/v1"


def test_embedding_backend_defaults_to_sentence_transformers_qwen():
    settings = Settings(_env_file=None)

    assert settings.embedding_backend == "sentence_transformers"
    assert settings.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.embedding_query_prompt_name == "query"
    assert settings.embedding_model_dir == (PROJECT_ROOT / "storage" / "models").resolve()
    assert settings.use_sentence_transformer_embeddings
    assert not settings.use_openai_embeddings


def test_embedding_backend_can_enable_openai():
    settings = Settings(_env_file=None, embedding_backend="openai")

    assert settings.use_openai_embeddings
    assert not settings.use_sentence_transformer_embeddings
