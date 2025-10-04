#!/usr/bin/env python3
"""Identify enum-like fields in the tickets dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find fields with limited unique values (potential enums)."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/raw/tickets.json"),
        help="Path to the tickets JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=25,
        help="Maximum number of unique values to consider as enum",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}")
    return data


def to_hashable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return tuple(to_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, to_hashable(val)) for key, val in value.items()))
    return str(value)


def format_value(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)


def collect_enum_candidates(
    records: Iterable[Dict[str, Any]],
    threshold: int,
) -> Dict[str, List[str]]:
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")

    unique_values: Dict[str, set[Any]] = defaultdict(set)
    samples: Dict[str, Dict[Any, Any]] = defaultdict(dict)

    for record in records:
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            canonical = to_hashable(value)
            unique_values[key].add(canonical)
            samples[key].setdefault(canonical, value)

    candidates: Dict[str, List[str]] = {}
    for field, values in unique_values.items():
        unique_count = len(values)
        if 0 < unique_count < threshold:
            formatted = sorted(format_value(samples[field][item]) for item in values)
            candidates[field] = formatted

    return candidates


def main() -> None:
    args = parse_args()
    records = load_records(args.path)
    candidates = collect_enum_candidates(records, args.threshold)

    if not candidates:
        print("No enum-like fields found.")
        return

    for field in sorted(candidates):
        options = candidates[field]
        print(f"{field} ({len(options)} unique values):")
        for option in options:
            print(f"  - {option}")


if __name__ == "__main__":
    main()
