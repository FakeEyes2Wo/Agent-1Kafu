from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUMMARY_ENGLISH_MANUAL = "\u6c47\u603b\u82f1\u6587\u624b\u518c"


def _image_name_lookup(image_dir: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lookup[path.name] = path.name
        lookup.setdefault(path.stem, path.name)
    return lookup


def _source_manual_name(manual_name: str, line_number: int) -> str:
    if manual_name == SUMMARY_ENGLISH_MANUAL:
        return f"{manual_name}:\u884c{line_number}"
    return manual_name


def count_image_names(
    manual_dir: Path,
    image_dir: Path,
) -> list[dict[str, str | int]]:
    image_lookup = _image_name_lookup(image_dir)
    counts: Counter[str] = Counter()
    manual_names: defaultdict[str, set[str]] = defaultdict(set)

    for image_name in image_lookup.values():
        counts.setdefault(image_name, 0)

    for manual_path in sorted(manual_dir.glob("*.txt")):
        manual_name = manual_path.stem
        for line_number, line in enumerate(
            manual_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            data = ast.literal_eval(line)
            if (
                not isinstance(data, (list, tuple))
                or len(data) != 2
                or not isinstance(data[1], list)
            ):
                raise ValueError(
                    f"manual entry must contain a file list: {manual_path}:{line_number}"
                )

            source_manual_name = _source_manual_name(manual_name, line_number)
            for raw_image_name in data[1]:
                image_name = image_lookup.get(str(raw_image_name), str(raw_image_name))
                counts[image_name] += 1
                manual_names[image_name].add(source_manual_name)

    return [
        {
            "image_name": image_name,
            "count": count,
            "manual_names": " | ".join(sorted(manual_names[image_name])),
        }
        for image_name, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )
    ]


def write_csv(rows: list[dict[str, str | int]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_name", "count", "manual_names"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count manual image references and record every source manual name."
    )
    parser.add_argument(
        "--manual-dir",
        type=Path,
        default=Path("data") / "\u624b\u518c",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data") / "\u624b\u518c" / "\u63d2\u56fe",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("image_name_counts.csv"),
    )
    args = parser.parse_args()

    rows = count_image_names(args.manual_dir, args.image_dir)
    write_csv(rows, args.output)
    multi_manual_count = sum(
        1 for row in rows if " | " in str(row["manual_names"])
    )
    print(f"wrote {args.output} ({len(rows)} images)")
    print(f"images appearing in multiple manuals: {multi_manual_count}")


if __name__ == "__main__":
    main()
