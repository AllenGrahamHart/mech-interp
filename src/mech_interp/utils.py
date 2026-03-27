from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def batched(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return vector / norm


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_run_dir(root: str | Path, experiment_name: str) -> Path:
    path = Path(root) / f"{experiment_name}_{timestamp_slug()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(payload):
        payload = asdict(payload)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_nonempty_lines(path: str | Path) -> list[str]:
    lines = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
    return lines


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def to_serializable(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def write_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    import pandas as pd

    frame = pd.DataFrame(list(rows))
    ensure_parent_dir(path)
    frame.to_csv(path, index=False)

