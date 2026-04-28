from scripts.generate_submission import (
    _load_completed_rows,
    _question_rows,
    clean_question,
    main,
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


def test_write_submission_preserves_question_order_for_completed_rows(tmp_path):
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

    assert output_path.read_text(encoding="utf-8") == (
        "id,ret\n1,answer 1\n2,answer 2\n3,answer 3\n"
    )


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


def test_main_disables_history_for_submission(monkeypatch, tmp_path):
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

    monkeypatch.setattr("scripts.generate_submission.get_settings", lambda: Settings())
    monkeypatch.setattr("scripts.generate_submission.PROJECT_ROOT", tmp_path)

    calls = []

    def fake_answer_question(question, session_id=None, persist_history=True):
        calls.append((question, session_id, persist_history))
        return "answer", session_id

    monkeypatch.setattr("scripts.generate_submission.answer_question", fake_answer_question)

    main()

    assert calls == [
        ("hello\nagain", "submission_1", False),
        ("single", "submission_2", False),
    ]
