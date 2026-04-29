import asyncio

from kefu_agent import graph


def test_answer_question_can_skip_history(monkeypatch):
    calls = {"load": 0, "save": 0}

    monkeypatch.setattr(graph, "retrieve", lambda query: [])
    monkeypatch.setattr(graph, "_require_chat_model", lambda: None)
    monkeypatch.setattr(
        graph,
        "_invoke_chat",
        lambda prompt, error_context: "answer" if error_context == "generate answer" else "reflected",
    )

    def fake_load_history(session_id):
        calls["load"] += 1
        return [{"question": "old", "answer": "old answer"}]

    def fake_save_turn(session_id, question, answer):
        calls["save"] += 1

    monkeypatch.setattr(graph, "load_history", fake_load_history)
    monkeypatch.setattr(graph, "save_turn", fake_save_turn)

    answer, session_id = graph.answer_question(
        "question",
        session_id="submission_1",
        persist_history=False,
    )

    assert answer == "reflected"
    assert session_id == "submission_1"
    assert calls == {"load": 0, "save": 0}


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
        persist_history=False,
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
            persist_history=False,
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
            "history": [],
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": "evidence",
            "answer": "draft answer",
        }
    )

    assert state["answer"] == "better warmer answer"
    assert calls[-1][1] == "check and rewrite answer"
    assert "draft answer" in calls[-1][0]


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
            "history": [],
            "image_summary": graph.NO_IMAGE_SUMMARY,
            "contexts": "evidence",
        }
    )

    assert state["answer"] == "rescued answer"
    assert calls == ["generate answer", "check and rewrite answer"]
