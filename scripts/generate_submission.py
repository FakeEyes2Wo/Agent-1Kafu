from pathlib import Path
import argparse
import asyncio
import csv
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from kefu_agent.config import PROJECT_ROOT, get_settings
from kefu_agent.graph import answer_question_async
from kefu_agent.rag import format_contexts, retrieve


CONTEXT_CACHE_VERSION = 1


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


def main(argv: list[str] | None = None) -> None:
    asyncio.run(main_async(_parse_args([] if argv is None else argv)))


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()
    question_path = settings.data_dir / "question_public.csv"
    sample_path = settings.data_dir / "submission_example.csv"
    output_path = PROJECT_ROOT / "submission.csv"
    context_cache_path = args.contexts_cache or settings.vectorstore_dir / "contexts_cache.json"

    questions = _question_rows(question_path)
    fieldnames = _submission_fieldnames(sample_path)
    question_ids = {row["id"] for row in questions}
    completed = {
        qid: answer
        for qid, answer in _load_completed_rows(output_path).items()
        if qid in question_ids
    }
    rows_by_id = dict(completed)
    write_submission(question_path, output_path, fieldnames, rows_by_id)
    contexts_by_id = prepare_context_cache(questions, context_cache_path, settings)

    with tqdm(
        total=len(questions),
        initial=len(completed),
        desc="Generating submission",
        unit="question",
        dynamic_ncols=True,
    ) as progress:
        missing_rows = [row for row in questions if row["id"] not in completed]
        await _generate_missing_answers(
            missing_rows,
            contexts_by_id,
            question_path,
            output_path,
            fieldnames,
            rows_by_id,
            max(1, args.workers),
            progress,
        )

    write_submission(question_path, output_path, fieldnames, rows_by_id)
    validate_submission(question_path, output_path)
    print(f"submission_path={output_path}")
    print(f"rows={len(questions)}")
    print(f"filled={len(rows_by_id)}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("SUBMISSION_WORKERS", "4")),
        help="Number of concurrent answer generation workers.",
    )
    parser.add_argument(
        "--contexts-cache",
        type=Path,
        default=None,
        help="Path to the retrieval contexts cache JSON file.",
    )
    return parser.parse_args(argv)


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


def prepare_context_cache(
    questions: list[dict[str, str]],
    cache_path: Path,
    settings,
) -> dict[str, str]:
    signature = _context_cache_signature(settings)
    cache = _load_context_cache(cache_path, signature)
    items = cache["items"]

    valid_count = sum(
        1
        for row in questions
        if _valid_context_cache_item(items.get(row["id"]), clean_question(row["question"]))
    )
    with tqdm(
        total=len(questions),
        initial=valid_count,
        desc="Caching retrieval",
        unit="question",
        dynamic_ncols=True,
    ) as progress:
        for row in questions:
            qid = row["id"]
            question = clean_question(row["question"])
            if _valid_context_cache_item(items.get(qid), question):
                continue

            progress.set_postfix_str(f"id={qid}", refresh=False)
            items[qid] = {
                "question": question,
                "contexts": format_contexts(retrieve(question)),
            }
            _write_context_cache(cache_path, cache)
            progress.update(1)

    _write_context_cache(cache_path, cache)
    return {
        row["id"]: items[row["id"]]["contexts"]
        for row in questions
        if _valid_context_cache_item(items.get(row["id"]), clean_question(row["question"]))
    }


def _load_context_cache(cache_path: Path, signature: dict) -> dict:
    if not cache_path.exists():
        return {"signature": signature, "items": {}}
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {"signature": signature, "items": {}}
    if cache.get("signature") != signature or not isinstance(cache.get("items"), dict):
        return {"signature": signature, "items": {}}
    return cache


def _write_context_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, cache_path)


def _context_cache_signature(settings) -> dict:
    return {
        "version": CONTEXT_CACHE_VERSION,
        "embedding_backend": settings.embedding_backend.strip().lower(),
        "embedding_model": settings.embedding_model,
        "embedding_query_prompt_name": settings.embedding_query_prompt_name,
        "top_k": settings.top_k,
    }


def _valid_context_cache_item(item: dict | None, question: str) -> bool:
    return (
        isinstance(item, dict)
        and item.get("question") == question
        and isinstance(item.get("contexts"), str)
    )


async def _generate_missing_answers(
    missing_rows: list[dict[str, str]],
    contexts_by_id: dict[str, str],
    question_path: Path,
    output_path: Path,
    fieldnames: list[str],
    rows_by_id: dict[str, str],
    workers: int,
    progress: tqdm,
) -> None:
    queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
    for row in missing_rows:
        queue.put_nowait(row)

    write_lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            qid = row["id"]
            progress.set_postfix_str(f"id={qid}", refresh=False)
            question = clean_question(row["question"])
            try:
                answer, _ = await answer_question_async(
                    question,
                    session_id=f"submission_{qid}",
                    contexts=contexts_by_id[qid],
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

            async with write_lock:
                rows_by_id[qid] = answer
                write_submission(question_path, output_path, fieldnames, rows_by_id)
                progress.update(1)
            queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(min(workers, len(missing_rows)))]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


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
            writer.writerow({"id": qid, "ret": rows_by_id.get(qid, "")})
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


if __name__ == "__main__":
    main(sys.argv[1:])
