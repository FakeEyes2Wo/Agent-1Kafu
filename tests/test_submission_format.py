from scripts.generate_submission import (
    CONTEXT_CACHE_VERSION,
    _context_cache_signature,
    _load_completed_rows,
    _question_rows,
    clean_question,
    main,
    prepare_context_cache,
    validate_submission,
    write_submission,
)


def test_clean_question_multiline_csv_cell():
    raw = '"first line",\n"second line"'
    cleaned = clean_question(raw)
    assert cleaned == "first line\nsecond line"


def test_clean_question_handles_windows_newline_and_chinese_comma():
    raw = '"first line"，\r\n"second line"'
    cleaned = clean_question(raw)
    assert cleaned == "first line\nsecond line"


def test_question_rows_keep_all_public_questions(tmp_path):
    question_path = tmp_path / "question_public.csv"
    question_path.write_text(
        'id,question\n1,"""first"",\n""second"""\n2,"""single"""\n',
        encoding="utf-8",
    )

    rows = _question_rows(question_path)

    assert [row["id"] for row in rows] == ["1", "2"]


def test_load_completed_rows_skips_empty_answers(tmp_path):
    output = tmp_path / "submission.csv"
    output.write_text("id,ret\n1,done\n2,\n3,  \n", encoding="utf-8")

    assert _load_completed_rows(output) == {"1": "done"}


def test_write_submission_preserves_question_order_and_blanks_missing_answers(tmp_path):
    question_path = tmp_path / "question_public.csv"
    output_path = tmp_path / "submission.csv"
    question_path.write_text(
        'id,question\n1,"""first"",\n""second"""\n2,"""single"""\n3,"""third"",\n""more"""\n',
        encoding="utf-8",
    )

    write_submission(
        question_path,
        output_path,
        ["id", "ret"],
        {"3": "answer 3", "2": "answer 2", "1": "answer 1"},
    )

    assert output_path.read_text(encoding="utf-8-sig") == (
        "id,ret\n1,answer 1\n2,answer 2\n3,answer 3\n"
    )

    write_submission(question_path, output_path, ["id", "ret"], {"3": "answer 3"})

    assert output_path.read_text(encoding="utf-8-sig") == (
        "id,ret\n1,\n2,\n3,answer 3\n"
    )


def test_write_submission_uses_utf8_with_bom(tmp_path):
    question_path = tmp_path / "question_public.csv"
    output_path = tmp_path / "submission.csv"
    question_path.write_text(
        'id,question\n1,"""问题"""\n',
        encoding="utf-8",
    )

    write_submission(
        question_path,
        output_path,
        ["id", "ret"],
        {"1": "您好，问题已收到"},
    )

    raw = output_path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8-sig").replace("\r\n", "\n") == "id,ret\n1,您好，问题已收到\n"


def test_validate_submission_requires_complete_rows(tmp_path):
    question_path = tmp_path / "question_public.csv"
    output_path = tmp_path / "submission.csv"
    question_path.write_text(
        'id,question\n1,"""first"",\n""more"""\n2,"""second"",\n""more"""\n',
        encoding="utf-8",
    )
    output_path.write_text("id,ret\n1,answer 1\n", encoding="utf-8")

    try:
        validate_submission(question_path, output_path)
    except RuntimeError as exc:
        assert "row count" in str(exc)
    else:
        raise AssertionError("validate_submission should reject partial output")


def test_validate_submission_allows_blank_answers_when_ids_match(tmp_path):
    question_path = tmp_path / "question_public.csv"
    output_path = tmp_path / "submission.csv"
    question_path.write_text(
        'id,question\n1,"""first"""\n2,"""second"""\n',
        encoding="utf-8",
    )
    output_path.write_text("id,ret\n1,\n2,answer 2\n", encoding="utf-8")

    validate_submission(question_path, output_path)


def test_prepare_context_cache_reuses_existing_contexts(monkeypatch, tmp_path):
    cache_path = tmp_path / "contexts_cache.json"
    questions = [
        {"id": "1", "question": '"first"'},
        {"id": "2", "question": '"second"'},
    ]

    class Settings:
        embedding_backend = "hash"
        embedding_model = "hash"
        embedding_query_prompt_name = ""
        top_k = 2
        rag_backend = "llamaindex"
        retrieval_top_k = 20
        rerank_enabled = False
        rerank_model = "BAAI/bge-reranker-v2-m3"
        rerank_top_n = 8

    calls = []

    def fake_retrieve(question):
        calls.append(question)
        return [question]

    monkeypatch.setattr("scripts.generate_submission.retrieve", fake_retrieve)
    monkeypatch.setattr(
        "scripts.generate_submission.format_contexts",
        lambda chunks: f"context for {chunks[0]}",
    )

    first = prepare_context_cache(questions, cache_path, Settings())
    second = prepare_context_cache(questions, cache_path, Settings())

    assert first == {"1": "context for first", "2": "context for second"}
    assert second == first
    assert calls == ["first", "second"]


def test_context_cache_signature_includes_rag_fields():
    class Settings:
        embedding_backend = "HASH"
        embedding_model = "hash"
        embedding_query_prompt_name = ""
        top_k = 2
        rag_backend = "llamaindex"
        retrieval_top_k = 20
        rerank_enabled = True
        rerank_model = "reranker"
        rerank_top_n = 6
        visual_retriever = "lexical"
        visual_top_k = 5

    signature = _context_cache_signature(Settings())

    assert CONTEXT_CACHE_VERSION == 7
    assert signature == {
        "version": 7,
        "embedding_backend": "hash",
        "embedding_model": "hash",
        "embedding_query_prompt_name": "",
        "top_k": 2,
        "rag_backend": "llamaindex",
        "retrieval_top_k": 20,
        "rerank_enabled": True,
        "rerank_model": "reranker",
        "rerank_top_n": 6,
        "manual_language_filter_version": "1",
        "manual_pic_tag_version": "1",
        "hybrid_search_version": "1",
        "visual_retriever_version": "1",
        "visual_retriever": "lexical",
        "visual_top_k": 5,
        "rag_context_format_version": "6",
    }


def test_main_generates_submission_without_history_argument(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "question_public.csv").write_text(
        'id,question\n1,"""hello"",\n""again"""\n2,"""single"""\n',
        encoding="utf-8",
    )
    (data_dir / "submission_example.csv").write_text("id,ret\n1,example\n", encoding="utf-8")

    class Settings:
        pass

    Settings.data_dir = data_dir
    Settings.vectorstore_dir = tmp_path / "storage"

    monkeypatch.setattr("scripts.generate_submission.get_settings", lambda: Settings())
    monkeypatch.setattr("scripts.generate_submission.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.generate_submission.prepare_context_cache",
        lambda questions, cache_path, settings: {
            row["id"]: f"context {row['id']}" for row in questions
        },
    )

    calls = []

    async def fake_answer_question_async(
        question,
        session_id=None,
        contexts=None,
    ):
        calls.append((question, session_id, contexts))
        return "answer", session_id

    monkeypatch.setattr(
        "scripts.generate_submission.answer_question_async",
        fake_answer_question_async,
    )

    main(["--workers", "2"])

    assert calls == [
        ("hello\nagain", "submission_1", "context 1"),
        ("single", "submission_2", "context 2"),
    ]


def test_main_resumes_and_writes_blank_rows_for_missing_answers(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "question_public.csv").write_text(
        'id,question\n1,"""done"""\n2,"""missing"""\n3,"""also missing"""\n',
        encoding="utf-8",
    )
    (data_dir / "submission_example.csv").write_text("id,ret\n1,example\n", encoding="utf-8")
    (tmp_path / "submission.csv").write_text("id,ret\n1,old answer\n", encoding="utf-8")

    class Settings:
        pass

    Settings.data_dir = data_dir
    Settings.vectorstore_dir = tmp_path / "storage"

    monkeypatch.setattr("scripts.generate_submission.get_settings", lambda: Settings())
    monkeypatch.setattr("scripts.generate_submission.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.generate_submission.prepare_context_cache",
        lambda questions, cache_path, settings: {
            row["id"]: f"context {row['id']}" for row in questions
        },
    )

    calls = []

    async def fake_answer_question_async(
        question,
        session_id=None,
        contexts=None,
    ):
        calls.append((question, session_id, contexts))
        return f"new answer for {question}", session_id

    monkeypatch.setattr(
        "scripts.generate_submission.answer_question_async",
        fake_answer_question_async,
    )

    main(["--workers", "2"])

    assert calls == [
        ("missing", "submission_2", "context 2"),
        ("also missing", "submission_3", "context 3"),
    ]
    assert (tmp_path / "submission.csv").read_text(encoding="utf-8-sig") == (
        "id,ret\n"
        "1,old answer\n"
        "2,new answer for missing\n"
        "3,new answer for also missing\n"
    )


def test_main_force_regenerates_existing_answers(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "question_public.csv").write_text(
        'id,question\n1,"""done"""\n2,"""also done"""\n',
        encoding="utf-8",
    )
    (data_dir / "submission_example.csv").write_text("id,ret\n1,example\n", encoding="utf-8")
    (tmp_path / "submission.csv").write_text(
        "id,ret\n1,old answer\n2,old answer 2\n",
        encoding="utf-8",
    )

    class Settings:
        pass

    Settings.data_dir = data_dir
    Settings.vectorstore_dir = tmp_path / "storage"

    monkeypatch.setattr("scripts.generate_submission.get_settings", lambda: Settings())
    monkeypatch.setattr("scripts.generate_submission.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.generate_submission.prepare_context_cache",
        lambda questions, cache_path, settings: {
            row["id"]: f"context {row['id']}" for row in questions
        },
    )

    calls = []

    async def fake_answer_question_async(
        question,
        session_id=None,
        contexts=None,
    ):
        calls.append((question, session_id, contexts))
        return f"new answer for {question}", session_id

    monkeypatch.setattr(
        "scripts.generate_submission.answer_question_async",
        fake_answer_question_async,
    )

    main(["--workers", "2", "--force"])

    assert calls == [
        ("done", "submission_1", "context 1"),
        ("also done", "submission_2", "context 2"),
    ]
    assert (tmp_path / "submission.csv").read_text(encoding="utf-8-sig") == (
        "id,ret\n"
        "1,new answer for done\n"
        "2,new answer for also done\n"
    )
