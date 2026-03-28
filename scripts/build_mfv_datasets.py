from __future__ import annotations

import csv
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

SOURCE_PDF_URL = "https://scottaclifford.com/wp-content/uploads/2012/02/CICSA_MoralVignettes_BRM.pdf"
OUTPUT_DIR = Path("data/moral_foundations")


def _normalize_text(text: str) -> str:
    normalized = " ".join(text.split())
    if normalized and normalized[-1] not in ".!?":
        normalized += "."
    return normalized


def _base_foundation(label: str) -> str:
    base = label.split("(", 1)[0].strip().lower()
    return base.replace(" ", "_")


def _read_pdf_lines(pdf_url: str) -> list[str]:
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        pdf_path = tmpdir / "mfv.pdf"
        txt_path = tmpdir / "mfv.txt"
        subprocess.run(["curl", "-sSL", pdf_url, "-o", str(pdf_path)], check=True)
        subprocess.run(["pdftotext", str(pdf_path), str(txt_path)], check=True)
        return txt_path.read_text(encoding="utf-8").split("\n")


def _slice_lines(lines: list[str], start: int, end: int) -> list[str]:
    # input line numbers in the paper notes are 1-based and inclusive
    return [line.strip() for line in lines[start - 1 : end] if line.strip()]


def _parse_full_mfv(lines: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    segments = [
        ((934, 965), (968, 999)),
        ((1266, 1298), (1301, 1333)),
        ((1608, 1642), (1644, 1678)),
    ]
    for scenario_range, foundation_range in segments:
        scenarios = _slice_lines(lines, *scenario_range)
        foundations = _slice_lines(lines, *foundation_range)
        if len(scenarios) != len(foundations):
            raise ValueError(
                f"MFV segment length mismatch: {scenario_range=} {foundation_range=} "
                f"({len(scenarios)} scenarios vs {len(foundations)} foundations)"
            )
        for text, foundation_label in zip(scenarios, foundations, strict=True):
            rows.append(
                {
                    "text": _normalize_text(text),
                    "foundation_raw": foundation_label,
                    "foundation": _base_foundation(foundation_label),
                    "is_control": 0,
                }
            )

    liberty_scenarios = _slice_lines(lines, 1960, 1976)
    control_scenarios = _slice_lines(lines, 1977, 1992)
    rows.extend(
        {
            "text": _normalize_text(text),
            "foundation_raw": "Liberty",
            "foundation": "liberty",
            "is_control": 0,
        }
        for text in liberty_scenarios
    )
    rows.extend(
        {
            "text": _normalize_text(text),
            "foundation_raw": "Social Norms",
            "foundation": "social_norms",
            "is_control": 1,
        }
        for text in control_scenarios
    )

    counts = Counter(row["foundation"] for row in rows)
    expected = {
        "care": 32,
        "fairness": 17,
        "loyalty": 16,
        "authority": 17,
        "sanctity": 17,
        "liberty": 17,
        "social_norms": 16,
    }
    if counts != expected:
        raise ValueError(f"Unexpected MFV counts: {counts} != {expected}")

    return rows


def _assign_split(rows: list[dict[str, object]], eval_count: int, seed: int) -> list[dict[str, object]]:
    import random

    shuffled = [dict(row) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    for idx, row in enumerate(shuffled):
        row["split"] = "test" if idx < eval_count else "train"
    return shuffled


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_outputs() -> None:
    lines = _read_pdf_lines(SOURCE_PDF_URL)
    rows = _parse_full_mfv(lines)

    all_rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["foundation"]), []).append(row)

    for foundation, bucket in grouped.items():
        eval_count = max(4, round(len(bucket) * 0.25))
        seed = 42 + sum(ord(ch) for ch in foundation)
        split_rows = _assign_split(bucket, eval_count=eval_count, seed=seed)
        all_rows.extend(split_rows)

    all_rows.sort(key=lambda row: (str(row["foundation"]), str(row["split"]), str(row["text"])))
    _write_csv(
        OUTPUT_DIR / "mfv_full.csv",
        all_rows,
        fieldnames=["text", "foundation_raw", "foundation", "is_control", "split"],
    )

    controls = grouped["social_norms"]
    target_foundations = ["care", "fairness", "loyalty", "authority", "sanctity", "liberty"]
    for foundation in target_foundations:
        positive_rows = [dict(row) for row in grouped[foundation]]
        negative_rows = [dict(row) for row in controls]

        pos_eval = max(4, round(len(positive_rows) * 0.25))
        neg_eval = max(4, round(len(negative_rows) * 0.25))
        pos_rows = _assign_split(positive_rows, eval_count=pos_eval, seed=100 + len(foundation))
        neg_rows = _assign_split(negative_rows, eval_count=neg_eval, seed=200 + len(foundation))

        binary_rows = []
        for row in pos_rows:
            binary_rows.append(
                {
                    "text": row["text"],
                    "label": 1,
                    "split": row["split"],
                    "source_foundation": row["foundation"],
                    "target_foundation": foundation,
                    "is_control": 0,
                }
            )
        for row in neg_rows:
            binary_rows.append(
                {
                    "text": row["text"],
                    "label": 0,
                    "split": row["split"],
                    "source_foundation": row["foundation"],
                    "target_foundation": foundation,
                    "is_control": 1,
                }
            )
        binary_rows.sort(key=lambda row: (row["split"], row["label"], row["text"]))
        _write_csv(
            OUTPUT_DIR / f"mfv_{foundation}_vs_social_norms.csv",
            binary_rows,
            fieldnames=[
                "text",
                "label",
                "split",
                "source_foundation",
                "target_foundation",
                "is_control",
            ],
        )


if __name__ == "__main__":
    build_outputs()
