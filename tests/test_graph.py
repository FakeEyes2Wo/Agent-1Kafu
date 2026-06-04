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


def test_manual_product_questions_do_not_skip_retrieval():
    assert not graph._is_general_policy_question("电钻的三年有限保修包含哪些内容？")
    assert not graph._is_general_policy_question("功能键盘的保修政策通常包含哪些内容？")
    assert not graph._is_general_policy_question("How can I replace the fuse of the boat?")
    assert graph._is_general_policy_question("请问你们的商品能提供上门安装服务吗？")


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


def test_plan_answer_describes_policy_and_image_rules():
    state = graph.plan_answer(
        {
            "question": "如何安装空调遥控器支架？",
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": '证据 <PIC> img_1 </PIC>\n可用图片：["img_1"]',
        }
    )

    assert "问题类型：手册问答" in state["response_plan"]
    assert "图片策略：证据含配图" in state["response_plan"]
    assert "必须按原始顺序覆盖" in state["response_plan"]


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
    assert calls[0][1] == "verify answer support"
    assert calls[-1][1] == "check and rewrite answer"
    assert "draft answer" in calls[-1][0]
    assert "证据核验反馈" in calls[-1][0]
    assert "回答规划" in calls[-1][0]
    assert "按回答规划和用户原始顺序覆盖问题" in calls[-1][0]
    assert "删除或改写证据核验反馈中指出的不可靠内容" in calls[-1][0]
    assert "图片只保留裸 `<PIC>`" in calls[-1][0]
    assert "保持简洁自然" in calls[-1][0]


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


def test_verify_answer_support_uses_evidence_prompt(monkeypatch):
    prompts = []

    def fake_invoke(prompt, error_context):
        prompts.append((prompt, error_context))
        return "UNSUPPORTED: unsupported claim"

    monkeypatch.setattr(graph, "_invoke_chat", fake_invoke)

    feedback = graph._verify_answer_support(
        {
            "question": "question",
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": "evidence",
        },
        "candidate answer",
    )

    assert feedback == "UNSUPPORTED: unsupported claim"
    assert prompts[0][1] == "verify answer support"
    assert "RAG evidence verifier" in prompts[0][0]
    assert "candidate answer" in prompts[0][0]


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

    assert "按用户提问语言回复" in prompts[-1]
    assert "多个子问题按原始顺序逐一回应" in prompts[-1]
    assert "优先使用检索证据" in prompts[-1]
    assert "证据不足时只询问最少必要信息" in prompts[-1]
    assert "不要输出图片 ID 或末尾图片列表" in prompts[-1]
    assert "不要输出标题、JSON、评分或内部说明" in prompts[-1]
    assert "历史对话" not in prompts[-1]


def test_invoke_chat_uses_langchain_v1_init_chat_model(monkeypatch):
    calls = {}

    class Settings:
        chat_model = "chat-model"
        openai_api_key = "test-key"
        openai_base_url = "https://example.test/v1"
        model_timeout_seconds = 12

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
        "max_tokens": 700,
        "timeout": 12,
        "max_retries": 1,
    }
    assert calls["messages"][0].content == "prompt text"
    assert not hasattr(graph, "_chat_llm")


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


def test_check_answer_sanitizes_internal_heading(monkeypatch):
    monkeypatch.setattr(
        graph,
        "_invoke_chat",
        lambda prompt, error_context: "最终回复：根据检索证据，按下电源键即可。",
    )

    state = graph.check_answer(
        {
            "question": "question",
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": "evidence",
            "answer": "draft",
        }
    )

    assert state["answer"] == "按下电源键即可。"
