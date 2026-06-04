from pathlib import Path
import argparse
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kefu_agent.config import get_settings
from scripts.generate_submission import CSV_ENCODING, clean_question


DEFAULT_OUTPUT = PROJECT_ROOT / "question_answer.csv"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args([] if argv is None else argv)
    rows = join_question_answer(
        question_path=args.questions,
        submission_path=args.submission,
        require_complete=not args.allow_missing,
    )
    write_question_answer(args.output, rows)
    print(f"output_path={args.output}")
    print(f"rows={len(rows)}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Join question_public.csv questions with answers from submission.csv."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=settings.data_dir / "question_public.csv",
        help="Path to question_public.csv.",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=PROJECT_ROOT / "submission.csv",
        help="Path to submission.csv; answers are read from the ret column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Keep rows with missing answers instead of raising an error.",
    )
    return parser.parse_args(argv)


def join_question_answer(
    question_path: Path,
    submission_path: Path,
    require_complete: bool = True,
) -> list[dict[str, str]]:
    questions = _read_csv_rows(question_path)
    answers = _submission_answers(submission_path)
    rows = []
    missing_ids = []

    for row in questions:
        qid = str(row["id"])
        answer = answers.get(qid, "")
        if require_complete and not answer:
            missing_ids.append(qid)
        question = clean_question(row["question"])
        rows.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "qa_text": f"Question:\n{question}\n\nAnswer:\n{answer}",
            }
        )

    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        suffix = "..." if len(missing_ids) > 10 else ""
        raise RuntimeError(f"missing answers for id(s): {preview}{suffix}")
    return rows


def write_question_answer(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "qa_text"])
        writer.writeheader()
        writer.writerows(rows)


def _submission_answers(submission_path: Path) -> dict[str, str]:
    with submission_path.open("r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "ret"]:
            raise RuntimeError("submission.csv must have columns: id,ret")
        return {
            str(row["id"]): row.get("ret", "").strip()
            for row in reader
            if row.get("id")
        }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main(sys.argv[1:])
