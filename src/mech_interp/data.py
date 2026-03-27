from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from mech_interp.config import DatasetConfig


@dataclass
class TextExample:
    text: str
    label: int
    split: str
    metadata: dict[str, Any]


def _apply_common_filters(rows: list[dict[str, Any]], config: DatasetConfig) -> list[dict[str, Any]]:
    filtered = rows
    if config.require_short:
        filtered = [row for row in filtered if bool(row.get("is_short", True))]
    if config.require_edited is not None:
        filtered = [row for row in filtered if bool(row.get("edited")) == config.require_edited]
    return filtered


def _sample_per_class(
    rows: list[TextExample],
    limit_per_class: int | None,
    seed: int,
) -> list[TextExample]:
    if limit_per_class is None:
        return rows

    grouped: dict[int, list[TextExample]] = {}
    for row in rows:
        grouped.setdefault(row.label, []).append(row)

    rng = random.Random(seed)
    sampled: list[TextExample] = []
    for label in sorted(grouped):
        bucket = grouped[label][:]
        rng.shuffle(bucket)
        sampled.extend(bucket[:limit_per_class])

    rng.shuffle(sampled)
    return sampled


def load_ethics_split(config: DatasetConfig, split: str, limit_per_class: int | None) -> list[TextExample]:
    from datasets import load_dataset

    dataset = load_dataset(config.dataset_id, config.subset, split=split)
    rows = dataset.to_list()
    rows = _apply_common_filters(rows, config)

    examples = [
        TextExample(
            text=str(row[config.text_field]).strip(),
            label=int(row[config.label_field]),
            split=split,
            metadata={
                key: value
                for key, value in row.items()
                if key not in {config.text_field, config.label_field}
            },
        )
        for row in rows
    ]

    return _sample_per_class(examples, limit_per_class=limit_per_class, seed=config.shuffle_seed)

