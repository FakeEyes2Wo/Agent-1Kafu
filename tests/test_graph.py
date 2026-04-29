import asyncio

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
    assert "尽可能简洁" in calls[-1][0]


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
    assert "尽可能简洁" in prompts[-1]
    assert "历史对话" not in prompts[-1]
