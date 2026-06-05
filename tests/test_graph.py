import asyncio

from langchain.messages import AIMessage

from kefu_agent import graph


def test_answer_question_runs_without_persistent_history(monkeypatch):
    monkeypatch.setattr(graph, "retrieve", lambda query: [])
    monkeypatch.setattr(graph, "_require_chat_model", lambda: None)
    monkeypatch.setattr(
        graph,
        "_invoke_chat",
        lambda prompt, error_context: "answer" if error_context == "generate answer" else "reflected",
    )

    answer, session_id = graph.answer_question(
        "question",
        session_id="submission_1",
    )

    assert answer == "reflected"
    assert session_id == "submission_1"


def test_answer_question_can_use_precomputed_contexts(monkeypatch):
    monkeypatch.setattr(graph, "_require_chat_model", lambda: None)
    monkeypatch.setattr(graph, "retrieve", lambda query: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(
        graph,
        "_invoke_chat",
        lambda prompt, error_context: "answer" if error_context == "generate answer" else "checked",
    )

    answer, _ = graph.answer_question(
        "question",
        session_id="submission_1",
        contexts="cached evidence",
    )

    assert answer == "checked"


def test_answer_question_with_trace_returns_draft_final_and_usage(monkeypatch):
    monkeypatch.setattr(graph, "_require_chat_model", lambda: None)
    monkeypatch.setattr(graph, "retrieve", lambda query: [])

    def fake_invoke(prompt, error_context):
        if error_context == "generate answer":
            return graph.ChatText("draft", {"input_tokens": 3, "output_tokens": 1})
        return graph.ChatText("final", {"input_tokens": 4, "output_tokens": 2})

    monkeypatch.setattr(graph, "_invoke_chat", fake_invoke)

    answer, session_id, trace = graph.answer_question_with_trace(
        "question",
        session_id="submission_1",
        contexts="cached evidence",
    )

    assert answer == "final"
    assert session_id == "submission_1"
    assert trace["draft_answer"] == "draft"
    assert trace["checked_answer"] == "final"
    assert trace["final_answer"] == "final"
    assert trace["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert [call["phase"] for call in trace["api_calls"]] == [
        "generate_answer",
        "check_and_rewrite_answer",
    ]


def test_answer_question_async_uses_sync_entrypoint(monkeypatch):
    def fake_answer_question(question, **kwargs):
        return f"answer for {question}", kwargs["session_id"]

    monkeypatch.setattr(graph, "answer_question", fake_answer_question)

    answer, session_id = asyncio.run(
        graph.answer_question_async(
            "question",
            session_id="sid",
            contexts="cached evidence",
        )
    )

    assert answer == "answer for question"
    assert session_id == "sid"


def test_general_policy_question_overrides_retrieval_context(monkeypatch):
    monkeypatch.setattr(graph, "retrieve", lambda query: (_ for _ in ()).throw(AssertionError))

    state = graph.retrieve_context(
        {
            "question": "请问支持7天无理由退换货吗？运费谁承担？",
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": "unrelated manual evidence",
        }
    )

    assert state["contexts"] == graph.GENERAL_POLICY_CONTEXT


def test_expanded_general_policy_keywords_skip_manual_retrieval():
    assert graph._is_general_policy_question("商品保质期还有1个月就过期了怎么办？")
    assert graph._is_general_policy_question("纸质版说明书和电子版在哪里找？")
    assert graph._is_general_policy_question("上门安装人员额外收取配件费怎么办？")
    assert graph._is_general_policy_question("收到二手商品还有污渍，我要投诉")


def test_manual_question_still_uses_retrieval(monkeypatch):
    calls = []

    def fake_retrieve(query):
        calls.append(query)
        return []

    monkeypatch.setattr(graph, "retrieve", fake_retrieve)

    state = graph.retrieve_context(
        {
            "question": "如何更换健身追踪器表带？",
            "image_summary": graph.NO_IMAGE_SUMMARY,
        }
    )

    assert calls == ["如何更换健身追踪器表带？"]
    assert state["contexts"] == ""


def test_check_answer_checks_and_rewrites_once(monkeypatch):
    calls = []

    def fake_invoke(prompt, error_context):
        calls.append((prompt, error_context))
        return "better warmer answer"

    monkeypatch.setattr(graph, "_invoke_chat", fake_invoke)

    state = graph.check_answer(
        {
            "question": "question",
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": "evidence",
            "answer": "draft answer",
        }
    )

    assert state["answer"] == "better warmer answer"
    assert calls[-1][1] == "check and rewrite answer"
    assert "draft answer" in calls[-1][0]
    assert "按原始顺序逐个回应" in calls[-1][0]
    assert "图片映射检查" in calls[-1][0]
    assert "简洁性检查" in calls[-1][0]
    assert "80-180 中文字" in calls[-1][0]
    assert "最多列 2-3 个必要项" in calls[-1][0]
    assert "只保留裸 `<PIC>`" in calls[-1][0]
    assert "禁止输出任何图片 ID" in calls[-1][0]
    assert "不使用 Markdown 粗体" in calls[-1][0]


def test_check_answer_falls_back_to_draft_when_rewrite_is_empty(monkeypatch):
    monkeypatch.setattr(graph, "_invoke_chat", lambda prompt, error_context: "")

    state = graph.check_answer(
        {
            "question": "question",
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": "evidence",
            "answer": "draft answer",
        }
    )

    assert state["answer"] == "draft answer"
    assert state["checked_answer"] == "draft answer"
    assert state["draft_answer"] == "draft answer"
    assert state["api_calls"][-1]["text_chars"] == 0


def test_generate_answer_uses_reflection_when_initial_answer_is_empty(monkeypatch):
    calls = []

    monkeypatch.setattr(graph, "_require_chat_model", lambda: None)

    def fake_invoke(prompt, error_context):
        calls.append(error_context)
        if error_context == "generate answer":
            return ""
        return "rescued answer"

    monkeypatch.setattr(graph, "_invoke_chat", fake_invoke)

    state = graph.generate_answer(
        {
            "question": "question",
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": "evidence",
        }
    )

    assert state["answer"] == "rescued answer"
    assert calls == ["generate answer", "check and rewrite answer"]


def test_generate_answer_uses_context_fallback_when_model_returns_empty(monkeypatch):
    calls = []

    monkeypatch.setattr(graph, "_require_chat_model", lambda: None)

    def fake_invoke(prompt, error_context):
        calls.append(error_context)
        return ""

    monkeypatch.setattr(graph, "_invoke_chat", fake_invoke)

    state = graph.generate_answer(
        {
            "question": "如何检查电池？",
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": (
                "[1] 来源：manual / title type=text\n"
                "先按住电源键 <PIC>img_1</PIC>，再观察指示灯。\n"
                "可用图片：[\"img_1\"]"
            ),
        }
    )

    assert state["answer"] == "根据说明，先按住电源键 <PIC>，再观察指示灯。"
    assert state["draft_answer"] == state["answer"]
    assert "img_1" not in state["answer"]
    assert calls == ["generate answer", "check and rewrite answer"]


def test_generate_answer_prompt_has_language_rule_and_no_history(monkeypatch):
    prompts = []

    monkeypatch.setattr(graph, "_require_chat_model", lambda: None)

    def fake_invoke(prompt, error_context):
        prompts.append(prompt)
        return "answer"

    monkeypatch.setattr(graph, "_invoke_chat", fake_invoke)

    graph.generate_answer(
        {
            "question": "Can I return it?",
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": "evidence",
        }
    )

    assert "If the customer asks in English, answer in English" in prompts[-1]
    assert "客户用英文提问时，使用英文回答" in prompts[-1]
    assert "当前请求内部的多轮对话" in prompts[-1]
    assert "RAG 检索证据使用规则" in prompts[-1]
    assert "图片信息映射规则" in prompts[-1]
    assert "简洁优先" in prompts[-1]
    assert "80-180 中文字" in prompts[-1]
    assert "不使用 Markdown 粗体" in prompts[-1]
    assert "只有证据不足" in prompts[-1]
    assert "禁止引入无关商品手册" in prompts[-1]
    assert "只在对应步骤、部件、状态或操作旁边输出裸 `<PIC>`" in prompts[-1]
    assert "禁止输出任何图片 ID" in prompts[-1]
    assert "历史对话" not in prompts[-1]


def test_invoke_chat_uses_langchain_v1_init_chat_model(monkeypatch):
    calls = {}

    class Settings:
        chat_model = "chat-model"
        model_api_key = "test-key"
        model_base_url = "https://example.test/v1"
        model_timeout_seconds = 12
        chat_max_tokens = 900

    class FakeModel:
        def invoke(self, messages):
            calls["messages"] = messages
            return AIMessage(content=" model answer ")

    def fake_init_chat_model(**kwargs):
        calls["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(graph, "get_settings", lambda: Settings())
    monkeypatch.setattr(graph, "init_chat_model", fake_init_chat_model)

    answer = graph._invoke_chat("prompt text", "generate answer")

    assert answer == "model answer"
    assert calls["kwargs"] == {
        "model": "chat-model",
        "model_provider": "openai",
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "temperature": 0.2,
        "max_tokens": 900,
        "timeout": 12,
        "max_retries": 1,
        "extra_body": None,
    }
    assert calls["messages"][0].content == "prompt text"
    assert not hasattr(graph, "_chat_llm")


def test_invoke_chat_can_enable_bailian_thinking(monkeypatch):
    calls = {}

    class Settings:
        chat_model = "qwen3.7-plus-2026-05-26"
        model_api_key = "test-key"
        model_api_key_source = "bailian"
        model_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model_timeout_seconds = 12
        chat_enable_thinking = True
        chat_max_tokens = 1800
        chat_thinking_budget = 1024

    class FakeModel:
        def invoke(self, messages):
            return AIMessage(content=" model answer ")

    def fake_init_chat_model(**kwargs):
        calls["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(graph, "get_settings", lambda: Settings())
    monkeypatch.setattr(graph, "init_chat_model", fake_init_chat_model)

    assert graph._invoke_chat("prompt text", "generate answer") == "model answer"
    assert calls["kwargs"]["max_tokens"] == 1800
    assert calls["kwargs"]["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": 1024,
    }


def test_invoke_chat_can_use_openai_responses(monkeypatch):
    calls = {}

    class Settings:
        use_openai_responses = True
        openai_configured_api_key = "sk-openai"
        openai_base_url = ""
        openai_responses_model = "gpt-5.5"
        openai_responses_reasoning_effort = "none"
        model_timeout_seconds = 30
        chat_max_tokens = 1200

    class FakeResponse:
        output_text = " responses answer "
        usage = {"input_tokens": 11, "output_tokens": 3}

    class FakeResponses:
        def create(self, **kwargs):
            calls["request"] = kwargs
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setattr(graph, "get_settings", lambda: Settings())
    monkeypatch.setattr(graph, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        graph,
        "init_chat_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError),
    )

    answer = graph._invoke_chat("prompt text", "generate answer")

    assert answer == "responses answer"
    assert answer.usage == {"input_tokens": 11, "output_tokens": 3}
    assert calls["client"] == {
        "api_key": "sk-openai",
        "timeout": 30,
        "max_retries": 1,
    }
    assert calls["request"] == {
        "model": "gpt-5.5",
        "input": "prompt text",
        "max_output_tokens": 1200,
        "reasoning": {"effort": "none"},
    }


def test_require_chat_model_checks_openai_responses_key(monkeypatch):
    class Settings:
        use_openai_responses = True
        openai_configured_api_key = ""

    monkeypatch.setattr(graph, "get_settings", lambda: Settings())

    try:
        graph._require_chat_model()
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("expected missing OpenAI key to fail")


def test_check_answer_formats_pic_list_from_contexts(monkeypatch):
    monkeypatch.setattr(
        "kefu_agent.rag.images._valid_image_ids",
        lambda image_dir: frozenset(),
    )
    monkeypatch.setattr(
        graph,
        "_invoke_chat",
        lambda prompt, error_context: (
            "电池组充电中 <PIC> img_1 </PIC> 已充满 <PIC> img_2 </PIC>"
        ),
    )

    state = graph.check_answer(
        {
            "question": "question",
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": '证据 <PIC> img_1 </PIC> 已充满 <PIC> img_2 </PIC>\n可用图片：["img_1", "img_2"]',
            "answer": "draft",
        }
    )

    assert state["answer"] == '电池组充电中 <PIC> 已充满 <PIC>,["img_1", "img_2"]'
