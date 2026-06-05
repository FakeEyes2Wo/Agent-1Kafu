from kefu_agent.config import PROJECT_ROOT, Settings


def test_vision_base_url_falls_back_to_openai_base_url():
    settings = Settings(
        _env_file=None,
        openai_base_url="https://chat.example/v1",
        vision_model_url="",
    )

    assert settings.vision_base_url == "https://chat.example/v1"


def test_vision_base_url_defaults_to_bailian_base_url():
    settings = Settings(_env_file=None)

    assert settings.vision_base_url == settings.bailian_base_url


def test_vision_base_url_uses_dedicated_url():
    settings = Settings(
        _env_file=None,
        openai_base_url="https://chat.example/v1",
        vision_model_url="https://vision.example/v1",
    )

    assert settings.vision_base_url == "https://vision.example/v1"


def test_defaults_to_bailian_api_with_local_hybrid_rag():
    settings = Settings(_env_file=None)

    assert settings.chat_api_backend == "langchain"
    assert settings.chat_model == "qwen3.7-plus-2026-05-26"
    assert not settings.chat_enable_thinking
    assert settings.chat_max_tokens == 450
    assert settings.chat_thinking_budget == 0
    assert settings.openai_responses_model == "gpt-5.5"
    assert settings.openai_responses_reasoning_effort == "none"
    assert not settings.use_openai_responses
    assert settings.vision_model == "qwen3.7-plus-2026-05-26"
    assert settings.vision_max_images == 6
    assert settings.vision_max_tokens == 800
    assert settings.embedding_backend == "openai"
    assert settings.embedding_model == "text-embedding-v4"
    assert settings.embedding_query_prompt_name == ""
    assert settings.embedding_model_dir == (PROJECT_ROOT / "storage" / "models").resolve()
    assert settings.embedding_batch_size == 10
    assert settings.rag_backend == "hybrid"
    assert settings.retrieval_top_k == 20
    assert not settings.rerank_enabled
    assert settings.rerank_backend == "none"
    assert settings.rerank_top_n == 8
    assert settings.visual_context_window == 360
    assert settings.use_openai_embeddings
    assert not settings.use_sentence_transformer_embeddings


def test_placeholder_bailian_key_is_not_treated_as_configured():
    settings = Settings(_env_file=None, bailian_api_key="your-dashscope-api-key")

    assert not settings.has_model_api_key
    assert not settings.has_openai_key


def test_dashscope_key_is_used_when_bailian_placeholder_is_present():
    settings = Settings(
        _env_file=None,
        bailian_api_key="your-dashscope-api-key",
        dashscope_api_key="sk-dashscope",
    )

    assert settings.model_api_key == "sk-dashscope"
    assert settings.model_api_key_source == "dashscope"
    assert settings.has_model_api_key


def test_openai_key_without_base_url_uses_openai_default_base_url():
    settings = Settings(_env_file=None, openai_api_key="sk-openai")

    assert settings.model_api_key == "sk-openai"
    assert settings.model_api_key_source == "openai"
    assert settings.model_base_url == "https://api.openai.com/v1"


def test_openai_responses_backend_uses_only_openai_key():
    settings = Settings(
        _env_file=None,
        chat_api_backend="openai_responses",
        bailian_api_key="sk-bailian",
        openai_api_key="sk-openai",
    )

    assert settings.use_openai_responses
    assert settings.openai_configured_api_key == "sk-openai"


def test_embedding_backend_can_enable_openai():
    settings = Settings(_env_file=None, embedding_backend="openai")

    assert settings.use_openai_embeddings
    assert not settings.use_sentence_transformer_embeddings
