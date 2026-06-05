from pathlib import Path
import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from kefu_agent.config import PROJECT_ROOT, get_settings
from kefu_agent.graph import CHAT_MAX_TOKENS, answer_question_with_trace_async
from kefu_agent.prompts import (
    ANSWER_PROMPT,
    CHECK_AND_REWRITE_PROMPT,
    COMMON_POLICY,
    IMAGE_SUMMARY_PROMPT,
)
from kefu_agent.rag import (
    HYBRID_SEARCH_VERSION,
    MANUAL_LANGUAGE_FILTER_VERSION,
    MANUAL_PIC_TAG_VERSION,
    RAG_CONTEXT_FORMAT_VERSION,
    VISUAL_RETRIEVER_VERSION,
    format_contexts,
    retrieve,
)


CONTEXT_CACHE_VERSION = 7
ANSWER_CACHE_VERSION = 1
# utf-8-sig writes a UTF-8 BOM and reads both BOM and non-BOM UTF-8 CSV files.
CSV_ENCODING = "utf-8-sig"


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
    answer_cache_path = (
        args.answers_cache
        or PROJECT_ROOT / "storage" / "api_cache" / "answers_cache.jsonl"
    )
    answer_cache_signature = _answer_cache_signature(settings)

    questions = _question_rows(question_path)
    fieldnames = _submission_fieldnames(sample_path)
    question_ids = {row["id"] for row in questions}
    completed = {
        qid: answer
        for qid, answer in _load_completed_rows(output_path).items()
        if qid in question_ids
    } if not args.force else {}
    rows_by_id = dict(completed)
    write_submission(question_path, output_path, fieldnames, rows_by_id)
    contexts_by_id = prepare_context_cache(questions, context_cache_path, settings)
    if not args.force:
        answer_cache = _load_answer_cache(answer_cache_path)
        for row in questions:
            qid = row["id"]
            if qid in rows_by_id:
                continue
            question = clean_question(row["question"])
            cached = answer_cache.get(qid)
            if _valid_answer_cache_item(
                cached,
                answer_cache_signature,
                question,
                contexts_by_id.get(qid, ""),
            ):
                rows_by_id[qid] = cached["final_answer"].strip()
        write_submission(question_path, output_path, fieldnames, rows_by_id)

    with tqdm(
        total=len(questions),
        initial=len(rows_by_id),
        desc="Generating submission",
        unit="question",
        dynamic_ncols=True,
    ) as progress:
        missing_rows = [row for row in questions if row["id"] not in rows_by_id]
        await _generate_missing_answers(
            missing_rows,
            contexts_by_id,
            question_path,
            output_path,
            fieldnames,
            rows_by_id,
            answer_cache_path,
            answer_cache_signature,
            max(1, args.workers),
            max(0, args.retries),
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
        "--retries",
        type=int,
        default=int(os.getenv("SUBMISSION_RETRIES", "2")),
        help="Retries per question after transient API failures or empty answers.",
    )
    parser.add_argument(
        "--contexts-cache",
        type=Path,
        default=None,
        help="Path to the retrieval contexts cache JSON file.",
    )
    parser.add_argument(
        "--answers-cache",
        type=Path,
        default=None,
        help="Path to the API answer trace cache JSONL file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all answers instead of resuming completed rows in submission.csv.",
    )
    return parser.parse_args(argv)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        return list(csv.DictReader(f))


def _question_rows(question_path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(question_path)


def _submission_fieldnames(sample_path: Path) -> list[str]:
    with sample_path.open("r", encoding=CSV_ENCODING, newline="") as f:
        fieldnames = csv.DictReader(f).fieldnames or ["id", "ret"]
    return fieldnames if fieldnames == ["id", "ret"] else ["id", "ret"]


def _load_completed_rows(output_path: Path) -> dict[str, str]:
    if not output_path.exists():
        return {}

    with output_path.open("r", encoding=CSV_ENCODING, newline="") as f:
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
        "embedding_batch_size": getattr(settings, "embedding_batch_size", 64),
        "top_k": settings.top_k,
        "rag_backend": settings.rag_backend,
        "retrieval_top_k": settings.retrieval_top_k,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_backend": getattr(settings, "rerank_backend", "none"),
        "rerank_model": settings.rerank_model,
        "rerank_top_n": settings.rerank_top_n,
        "manual_language_filter_version": MANUAL_LANGUAGE_FILTER_VERSION,
        "manual_pic_tag_version": MANUAL_PIC_TAG_VERSION,
        "hybrid_search_version": HYBRID_SEARCH_VERSION,
        "visual_retriever_version": VISUAL_RETRIEVER_VERSION,
        "visual_retriever": getattr(settings, "visual_retriever", "lexical"),
        "visual_top_k": getattr(settings, "visual_top_k", 8),
        "rag_context_format_version": RAG_CONTEXT_FORMAT_VERSION,
    }


def _answer_cache_signature(settings) -> dict:
    prompt_fingerprint = "\n".join(
        [
            ANSWER_PROMPT,
            CHECK_AND_REWRITE_PROMPT,
            COMMON_POLICY,
            IMAGE_SUMMARY_PROMPT,
        ]
    )
    return {
        "version": ANSWER_CACHE_VERSION,
        "chat_model": getattr(settings, "chat_model", ""),
        "chat_enable_thinking": bool(getattr(settings, "chat_enable_thinking", False)),
        "vision_model": getattr(settings, "vision_model", ""),
        "chat_max_tokens": CHAT_MAX_TOKENS,
        "prompt_hash": _stable_hash(prompt_fingerprint),
        "rag_context_format_version": RAG_CONTEXT_FORMAT_VERSION,
    }


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_answer_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    items: dict[str, dict] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("id") and item.get("final_answer"):
                items[str(item["id"])] = item
    return items


def _valid_answer_cache_item(
    item: dict | None,
    signature: dict,
    question: str,
    contexts: str,
) -> bool:
    return (
        isinstance(item, dict)
        and item.get("signature") == signature
        and item.get("question_hash") == _stable_hash(question)
        and item.get("contexts_hash") == _stable_hash(contexts)
        and isinstance(item.get("final_answer"), str)
        and bool(item["final_answer"].strip())
    )


def _answer_cache_item(
    qid: str,
    question: str,
    contexts: str,
    answer: str,
    session_id: str,
    trace: dict,
    signature: dict,
) -> dict:
    return {
        "signature": signature,
        "id": qid,
        "session_id": session_id,
        "question": question,
        "question_hash": _stable_hash(question),
        "contexts_hash": _stable_hash(contexts),
        "chat_model": signature.get("chat_model", ""),
        "vision_model": signature.get("vision_model", ""),
        "draft_answer": trace.get("draft_answer", ""),
        "checked_answer": trace.get("checked_answer", ""),
        "final_answer": answer,
        "image_summary": trace.get("image_summary", ""),
        "api_calls": trace.get("api_calls", []),
        "usage": trace.get("usage", {}),
        "created_at": int(time.time()),
    }


def _append_answer_cache(cache_path: Path, item: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


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
    answer_cache_path: Path,
    answer_cache_signature: dict,
    workers: int,
    retries: int,
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
            contexts = contexts_by_id[qid]
            try:
                answer, session_id, trace = await _answer_with_retries(
                    question,
                    session_id=f"submission_{qid}",
                    contexts=contexts,
                    retries=retries,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"failed to generate id={qid} after {retries + 1} attempt(s); "
                    f"saved progress is in {output_path}"
                ) from exc

            async with write_lock:
                rows_by_id[qid] = answer
                _append_answer_cache(
                    answer_cache_path,
                    _answer_cache_item(
                        qid,
                        question,
                        contexts,
                        answer,
                        session_id,
                        trace,
                        answer_cache_signature,
                    ),
                )
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


async def _answer_with_retries(
    question: str,
    session_id: str,
    contexts: str,
    retries: int,
) -> tuple[str, str, dict]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            answer, resolved_session_id, trace = await answer_question_with_trace_async(
                question,
                session_id=session_id,
                contexts=contexts,
            )
            answer = answer.strip()
            if not answer:
                raise RuntimeError("empty answer")
            return answer, resolved_session_id, trace
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            await asyncio.sleep(min(30, 2**attempt))

    if last_error is None:
        raise RuntimeError("answer generation failed")
    raise last_error


def write_submission(
    question_path: Path,
    output_path: Path,
    fieldnames: list[str],
    rows_by_id: dict[str, str],
) -> None:
    questions = _question_rows(question_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in questions:
            qid = row["id"]
            writer.writerow({"id": qid, "ret": rows_by_id.get(qid, "")})
    os.replace(tmp_path, output_path)


def validate_submission(question_path: Path, output_path: Path) -> None:
    questions = _question_rows(question_path)
    with output_path.open("r", encoding=CSV_ENCODING, newline="") as f:
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
