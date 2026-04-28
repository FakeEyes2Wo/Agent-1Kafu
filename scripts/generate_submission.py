from pathlib import Path
import csv
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from kefu_agent.config import PROJECT_ROOT, get_settings
from kefu_agent.graph import answer_question


def clean_question(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = []
    for line in text.splitlines():
        line = line.strip().replace('""', '"')
        line = line.rstrip(",，").strip()
        line = line.strip('"').strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def main() -> None:
    settings = get_settings()
    question_path = settings.data_dir / "question_public.csv"
    sample_path = settings.data_dir / "submission_example.csv"
    output_path = PROJECT_ROOT / "submission.csv"

    questions = _question_rows(question_path)
    fieldnames = _submission_fieldnames(sample_path)
    question_ids = {row["id"] for row in questions}
    completed = {
        qid: answer
        for qid, answer in _load_completed_rows(output_path).items()
        if qid in question_ids
    }
    rows_by_id = dict(completed)

    with tqdm(
        total=len(questions),
        initial=len(completed),
        desc="Generating submission",
        unit="question",
        dynamic_ncols=True,
    ) as progress:
        for row in questions:
            qid = row["id"]
            progress.set_postfix_str(f"id={qid}", refresh=False)
            if qid in completed:
                continue

            question = clean_question(row["question"])
            try:
                answer, _ = answer_question(
                    question,
                    session_id=f"submission_{qid}",
                    persist_history=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"failed to generate id={qid}; saved progress is in {output_path}"
                ) from exc
            answer = answer.strip()
            if not answer:
                raise RuntimeError(
                    f"empty ret for id={qid}; saved progress is in {output_path}"
                )

            rows_by_id[qid] = answer
            write_submission(question_path, output_path, fieldnames, rows_by_id)
            progress.update(1)

    write_submission(question_path, output_path, fieldnames, rows_by_id)
    validate_submission(question_path, output_path)
    print(f"submission_path={output_path}")
    print(f"rows={len(rows_by_id)}")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _question_rows(question_path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(question_path)


def _submission_fieldnames(sample_path: Path) -> list[str]:
    with sample_path.open("r", encoding="utf-8", newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames or ["id", "ret"]
    return fieldnames if fieldnames == ["id", "ret"] else ["id", "ret"]


def _load_completed_rows(output_path: Path) -> dict[str, str]:
    if not output_path.exists():
        return {}

    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "ret"]:
            return {}
        return {
            str(row["id"]): row["ret"].strip()
            for row in reader
            if row.get("id") and row.get("ret", "").strip()
        }


def write_submission(
    question_path: Path,
    output_path: Path,
    fieldnames: list[str],
    rows_by_id: dict[str, str],
) -> None:
    questions = _question_rows(question_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in questions:
            qid = row["id"]
            if qid in rows_by_id:
                writer.writerow({"id": qid, "ret": rows_by_id[qid]})
    os.replace(tmp_path, output_path)


def validate_submission(question_path: Path, output_path: Path) -> None:
    questions = _question_rows(question_path)
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames != ["id", "ret"]:
        raise RuntimeError("submission.csv must have columns: id,ret")
    if len(rows) != len(questions):
        raise RuntimeError(
            "submission.csv row count does not match question_public.csv"
        )
    for source, result in zip(questions, rows, strict=True):
        if source["id"] != result["id"]:
            raise RuntimeError("submission.csv id order does not match question_public.csv")
        if not result["ret"].strip():
            raise RuntimeError(f"empty ret for id={result['id']}")


if __name__ == "__main__":
    main()
