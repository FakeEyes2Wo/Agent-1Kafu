from pathlib import Path
import argparse
import csv
import sys
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kefu_agent.config import get_settings
from kefu_agent.graph import _is_general_policy_question
from kefu_agent.rag import retrieve
from scripts.generate_submission import CSV_ENCODING, clean_question


FIELDNAMES = "id question is_general_policy context_count has_pic avg_context_chars top_manuals top_titles".split()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args([] if argv is None else argv)
    settings = get_settings()
    question_path = args.questions or settings.data_dir / "question_public.csv"
    output_path = args.output or PROJECT_ROOT / "storage" / "eval" / "retrieval_eval.csv"

    with question_path.open("r", encoding=CSV_ENCODING, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    output = [
        _evaluate_row(row)
        for row in tqdm(rows, desc="Evaluating retrieval", unit="question", dynamic_ncols=True)
    ]
    _write_eval(output_path, output)
    print(f"eval_path={output_path}")
    print(f"rows={len(output)}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _evaluate_row(row: dict[str, str]) -> dict[str, str]:
    question = clean_question(row["question"])
    result = {"id": row["id"], "question": question}
    if _is_general_policy_question(question):
        return result | {
            "is_general_policy": "True",
            "context_count": "0",
            "has_pic": "False",
            "avg_context_chars": "0.0",
            "top_manuals": "",
            "top_titles": "",
        }

    chunks = retrieve(question)
    avg_chars = sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0.0
    return result | {
        "is_general_policy": "False",
        "context_count": str(len(chunks)),
        "has_pic": str(any("<PIC>" in chunk.text or chunk.image_ids for chunk in chunks)),
        "avg_context_chars": f"{avg_chars:.1f}",
        "top_manuals": " | ".join(_ordered_unique(chunk.manual for chunk in chunks)),
        "top_titles": " | ".join(_ordered_unique(chunk.title for chunk in chunks)),
    }


def _ordered_unique(values) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _write_eval(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
