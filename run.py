"""批量推理脚本：读取 question_public.csv，并发调用 agent 生成 submission.csv。"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import time
import traceback
from pathlib import Path

from src.kefu_agent.graph import answer_question_async

INPUT = Path("data/question_public.csv")
OUTPUT = Path("submission.csv")
DEFAULT_WORKERS = 8


def read_questions(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row["question"].strip()
            if q:
                rows.append((row["id"], q))
    return rows


async def process_one(
    qid: str,
    question: str,
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
) -> tuple[str, str, float]:
    start = time.perf_counter()
    async with sem:
        try:
            answer, _, file_list = await answer_question_async(question)
        except Exception:
            answer = f"[ERROR] {traceback.format_exc()}"
            file_list = []
    elapsed = time.perf_counter() - start
    full = f'"{answer}", {json.dumps(file_list)}' if file_list else f'"{answer}"'
    print(f"[{idx}/{total}] id={qid} 耗时 {elapsed:.1f}s")
    return qid, full, elapsed


async def main_async(workers: int) -> None:
    if not INPUT.exists():
        print(f"[ERROR] 输入文件不存在: {INPUT}")
        sys.exit(1)

    questions = read_questions(INPUT)
    total = len(questions)
    print(f"共 {total} 条问题，{workers} 并发，开始推理...")

    sem = asyncio.Semaphore(workers)
    tasks = [
        process_one(qid, question, sem, idx, total)
        for idx, (qid, question) in enumerate(questions, 1)
    ]
    results = await asyncio.gather(*tasks)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        for qid, full, _ in results:
            writer.writerow({"id": qid, "answer": full})

    total_time = sum(r[2] for r in results)
    print(f"完成，{total} 条结果已写入 {OUTPUT}，总耗时 {total_time:.1f}s（并发）")


def main() -> None:
    workers = DEFAULT_WORKERS
    if len(sys.argv) > 1:
        workers = int(sys.argv[1])
    asyncio.run(main_async(workers))


if __name__ == "__main__":
    main()
