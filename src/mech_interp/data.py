from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
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
    import pandas as pd
    from huggingface_hub import hf_hub_download

    filename = f"data/{config.subset}/{split}.csv"
    csv_path = hf_hub_download(config.dataset_id, repo_type="dataset", filename=filename)
    rows = pd.read_csv(csv_path).to_dict(orient="records")
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


def load_local_csv_split(
    config: DatasetConfig,
    split: str,
    limit_per_class: int | None,
) -> list[TextExample]:
    if not config.local_path:
        raise ValueError("DatasetConfig.local_path must be set for dataset_id='local_csv'.")

    csv_path = Path(config.local_path)
    if not csv_path.is_absolute() and not csv_path.exists():
        for candidate in (
            Path("/root") / csv_path,
            Path(__file__).resolve().parents[2] / csv_path,
        ):
            if candidate.exists():
                csv_path = candidate
                break
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    rows = [row for row in rows if str(row.get(config.split_field, "")).strip() == split]
    rows = _apply_common_filters(rows, config)
    examples = [
        TextExample(
            text=str(row[config.text_field]).strip(),
            label=int(row[config.label_field]),
            split=split,
            metadata={
                key: value
                for key, value in row.items()
                if key not in {config.text_field, config.label_field, config.split_field}
            },
        )
        for row in rows
    ]
    return _sample_per_class(examples, limit_per_class=limit_per_class, seed=config.shuffle_seed)


def load_text_split(config: DatasetConfig, split: str, limit_per_class: int | None) -> list[TextExample]:
    if config.dataset_id == "hendrycks/ethics":
        return load_ethics_split(config=config, split=split, limit_per_class=limit_per_class)

    if config.dataset_id == "local_csv":
        return load_local_csv_split(config=config, split=split, limit_per_class=limit_per_class)

    if config.dataset_id == "glue":
        from datasets import load_dataset

        dataset = load_dataset(config.dataset_id, config.subset, split=split)
        rows = [dict(row) for row in dataset]
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

    raise NotImplementedError(
        f"Dataset loader not implemented for dataset_id='{config.dataset_id}'."
    )
