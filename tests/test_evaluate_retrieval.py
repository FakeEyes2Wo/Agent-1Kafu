import csv

from kefu_agent.rag import Chunk
from scripts import evaluate_retrieval


def test_evaluate_retrieval_writes_csv_and_skips_general_policy(
    monkeypatch, tmp_path
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    question_path = data_dir / "question_public.csv"
    output_path = tmp_path / "storage" / "eval" / "retrieval_eval.csv"
    question_path.write_text(
        "id,question\n1,Can I return it?\n2,How do I replace the band?\n",
        encoding="utf-8",
    )

    class Settings:
        pass

    Settings.data_dir = data_dir

    calls = []

    def fake_retrieve(question):
        calls.append(question)
        return [
            Chunk(
                id="manual-1",
                manual="manual",
                title="title",
                text="body <PIC>",
                image_ids=["img_1"],
                vector=[],
            )
        ]

    monkeypatch.setattr(evaluate_retrieval, "get_settings", lambda: Settings())
    monkeypatch.setattr(evaluate_retrieval, "retrieve", fake_retrieve)

    evaluate_retrieval.main(
        [
            "--limit",
            "2",
            "--questions",
            str(question_path),
            "--output",
            str(output_path),
        ]
    )

    with output_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    assert calls == ["How do I replace the band?"]
    assert rows[0]["is_general_policy"] == "True"
    assert rows[0]["context_count"] == "0"
    assert rows[1]["is_general_policy"] == "False"
    assert rows[1]["context_count"] == "1"
    assert rows[1]["has_pic"] == "True"
    assert rows[1]["avg_context_chars"] == "10.0"
    assert rows[1]["top_manuals"] == "manual"
    assert rows[1]["top_titles"] == "title"
