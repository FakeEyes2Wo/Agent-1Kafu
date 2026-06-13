"""单条问题测试脚本：按 ID 从 question_public.csv 读取问题，输出完整结果。"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from src.kefu_agent.graph import get_workflow

INPUT = Path("data/question_public.csv")


def get_question(qid: str) -> str | None:
    with INPUT.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] == str(qid):
                return row["question"].strip()
    return None


def main() -> None:
    qid = sys.argv[1] if len(sys.argv) > 1 else "52"
    print(f"查询 ID: {qid}")

    question = get_question(qid)
    if question is None:
        print(f"[ERROR] 未找到 ID={qid} 的问题")
        sys.exit(1)

    print(f"\n问题:\n{question}\n")
    print("=" * 60)

    start = time.perf_counter()
    try:
        result = get_workflow().run(question)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    elapsed = time.perf_counter() - start

    answer = result["answer"]
    file_list = result.get("file_list", [])
    print(f"\n回答:\n{answer}")
    print(f"\nfile_list: {file_list}")
    print(f"citations: {result.get('citations', [])}")
    print(f"耗时: {elapsed:.1f}s")
    print(f"\n--- 提交格式 ---")
    if file_list:
        print(f'"{answer}", {file_list}')
    else:
        print(f'"{answer}"')


if __name__ == "__main__":
    main()
