"""Utility: IO helpers for JSONL, JSON, and CSV."""
import json
import csv
import os
from pathlib import Path
from typing import Any, Iterable, List


def read_jsonl(path: str) -> List[dict]:
    """Read a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict], path: str) -> None:
    """Write an iterable of dicts to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[io] Wrote {path}")


def read_json(path: str) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str, indent: int = 2) -> None:
    """Write an object to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    print(f"[io] Wrote {path}")


def write_csv(rows: List[dict], path: str) -> None:
    """Write a list of dicts to a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[io] Wrote {path}")


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
