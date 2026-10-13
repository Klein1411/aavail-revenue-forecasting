"""Append-only JSONL prediction activity, with test-specific paths."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path


def write_event(log_file: str | Path, event: dict) -> None:
    target = Path(log_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    enriched = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), **event}
    with target.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(enriched, ensure_ascii=False, default=str) + '\n')


def read_events(log_file: str | Path, limit: int = 100) -> list[dict]:
    if not 1 <= limit <= 1000:
        raise ValueError('limit must be 1..1000')
    path = Path(log_file)
    if not path.exists():
        return []
    results = []
    for line in path.read_text(encoding='utf-8').splitlines()[-limit:]:
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return results
