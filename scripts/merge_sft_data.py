import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import (
    INTERIM_CLEANED_DIR,
    INTERIM_CONVERTED_DIR,
    PROCESSED_SFT_DIR,
    clean_records,
    ensure_parent,
    read_jsonl,
    write_jsonl,
)


SOURCE_TO_TASK = {
    "finqa": "qa",
    "tatqa": "qa",
    "ectsum": "summary",
    "finer": "ner",
    "finred": "re",
    "fiqa": "sentiment",
    "finsen": "sentiment",
}

TASK_WEIGHTS = {
    "qa": 0.30,
    "summary": 0.15,
    "ner": 0.20,
    "re": 0.20,
    "sentiment": 0.15,
}


def verify_json_format(task: str, text: str) -> bool:
    if task not in ["ner", "re"]:
        return True
    try:
        parsed = json.loads(text)
        return isinstance(parsed, list)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_SFT_DIR / "finance_sft_data.jsonl",
    )
    args = parser.parse_args()

    rnd = random.Random(args.seed)
    source_files = {
        key: INTERIM_CONVERTED_DIR / f"{key}_sft.jsonl"
        for key in SOURCE_TO_TASK
    }

    missing = [key for key, path in source_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing converted datasets: {missing}")

    cleaned_by_source: dict[str, list[dict]] = {}
    raw_counts: dict[str, int] = {}
    clean_counts: dict[str, int] = {}
    format_drop_counts: dict[str, int] = defaultdict(int)

    for source, path in source_files.items():
        raw = read_jsonl(path)
        cleaned = clean_records(raw)
        raw_counts[source] = len(raw)

        task = SOURCE_TO_TASK[source]
        valid_records = []
        for record in cleaned:
            if verify_json_format(task, record["assistant"]):
                valid_records.append(record)
            else:
                format_drop_counts[source] += 1

        clean_counts[source] = len(valid_records)
        cleaned_by_source[source] = valid_records

        cleaned_path = INTERIM_CLEANED_DIR / f"{source}_sft.jsonl"
        write_jsonl(valid_records, cleaned_path)

    grouped: dict[str, list[dict]] = {key: [] for key in TASK_WEIGHTS}
    for source, rows in cleaned_by_source.items():
        task = SOURCE_TO_TASK[source]
        for record in rows:
            grouped[task].append(
                {
                    "user": record["user"],
                    "assistant": record["assistant"],
                    "_task": task,
                    "_source": source,
                }
            )

    for task, rows in grouped.items():
        if not rows:
            raise ValueError(f"Task '{task}' has no available samples after cleaning")

    max_total = min(
        int(len(grouped[task]) / TASK_WEIGHTS[task]) for task in TASK_WEIGHTS
    )
    if max_total <= 0:
        raise ValueError("No data available to merge after balancing")

    target_per_task: dict[str, int] = {}
    used = 0
    task_names = list(TASK_WEIGHTS)
    for task in task_names[:-1]:
        count = int(max_total * TASK_WEIGHTS[task])
        count = max(1, min(count, len(grouped[task])))
        target_per_task[task] = count
        used += count

    last_task = task_names[-1]
    target_per_task[last_task] = max(
        1,
        min(max_total - used, len(grouped[last_task])),
    )

    merged: list[dict] = []
    for task, rows in grouped.items():
        sources_in_task: defaultdict[str, list[dict]] = defaultdict(list)
        for record in rows:
            sources_in_task[record["_source"]].append(record)

        num_sources = len(sources_in_task)
        target_n = target_per_task[task]
        picked: list[dict] = []
        remaining_target = target_n
        sorted_sources = sorted(
            sources_in_task,
            key=lambda source: len(sources_in_task[source]),
        )
        for index, source in enumerate(sorted_sources):
            avg_needed = remaining_target // (num_sources - index)
            take_n = min(avg_needed, len(sources_in_task[source]))
            rnd.shuffle(sources_in_task[source])
            picked.extend(sources_in_task[source][:take_n])
            remaining_target -= take_n

        rnd.shuffle(picked)
        merged.extend(picked)

    rnd.shuffle(merged)

    main_records = [
        {
            "user": record["user"],
            "assistant": record["assistant"],
            "source_dataset": record["_source"],
            "task_type": record["_task"],
        }
        for record in merged
    ]

    task_index = [
        {"idx": index, "task": record["task_type"], "source": record["source_dataset"]}
        for index, record in enumerate(main_records)
    ]

    ensure_parent(args.output)
    write_jsonl(main_records, args.output)
    index_path = PROCESSED_SFT_DIR / "finance_sft_index_tasks.jsonl"
    write_jsonl(task_index, index_path)

    metadata = {
        "raw_counts": raw_counts,
        "format_drop_counts": dict(format_drop_counts),
        "clean_counts": clean_counts,
        "group_counts_after_clean": {key: len(value) for key, value in grouped.items()},
        "target_per_task": target_per_task,
        "merged_total": len(main_records),
        "seed": args.seed,
    }
    metadata_path = PROCESSED_SFT_DIR / "metadata.json"
    ensure_parent(metadata_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[merge] wrote {len(main_records)} samples -> {args.output}")
    print(f"[merge] wrote task index -> {index_path}")
    print(f"[merge] wrote metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
