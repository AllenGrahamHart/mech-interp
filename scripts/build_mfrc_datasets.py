from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

FOUNDATION_LABELS = ["Care", "Equality", "Proportionality", "Purity", "Authority", "Loyalty"]
SPECIAL_LABELS = ["Thin Morality", "Non-Moral"]
ALL_LABELS = FOUNDATION_LABELS + SPECIAL_LABELS
OUTPUT_DIR = Path("data/mfrc")
TEST_FRACTION = 0.2
SPLIT_SEED = 42


def _key_for_row(row: dict[str, object]) -> tuple[str, str, str]:
    return (str(row["text"]), str(row["subreddit"]), str(row["bucket"]))


def _stable_split(key: tuple[str, str, str], seed: int = SPLIT_SEED) -> str:
    digest = hashlib.sha256(("||".join(key) + f"||{seed}").encode("utf-8")).hexdigest()
    score = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return "test" if score < TEST_FRACTION else "train"


def _parse_labels(annotation: str) -> list[str]:
    return [label.strip() for label in annotation.split(",") if label.strip()]


def _bool_str(value: bool) -> str:
    return "1" if value else "0"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_outputs() -> None:
    dataset = load_dataset("USC-MOLA-Lab/MFRC", split="train")
    by_comment: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for raw_row in dataset:
        row = dict(raw_row)
        by_comment[_key_for_row(row)].append(row)

    aggregated_rows: list[dict[str, object]] = []
    comment_records: list[dict[str, object]] = []
    strict_nonmoral_records: list[dict[str, object]] = []
    positive_records: dict[str, list[dict[str, object]]] = {label: [] for label in FOUNDATION_LABELS}
    pure_positive_records: dict[str, list[dict[str, object]]] = {
        label: [] for label in FOUNDATION_LABELS
    }

    for key, rows in by_comment.items():
        text, subreddit, bucket = key
        kept_rows = [row for row in rows if row["confidence"] != "Not Confident"]
        votes = Counter()
        for row in kept_rows:
            for label in _parse_labels(str(row["annotation"])):
                votes[label] += 1

        present_foundations = [label for label in FOUNDATION_LABELS if votes[label] >= 2]
        has_nonmoral = votes["Non-Moral"] >= 2
        has_thin = votes["Thin Morality"] >= 2
        is_strict_nonmoral = has_nonmoral and not present_foundations and not has_thin
        is_inclusive_nonmoral = has_nonmoral and not present_foundations
        is_pure_foundation = (
            len(present_foundations) == 1 and not has_nonmoral and not has_thin
        )
        split = _stable_split(key)

        record = {
            "text": text,
            "subreddit": subreddit,
            "bucket": bucket,
            "split": split,
            "original_annotation_count": len(rows),
            "retained_annotation_count": len(kept_rows),
            "present_foundations": "|".join(present_foundations),
            "present_label_count": len(
                [label for label in FOUNDATION_LABELS + SPECIAL_LABELS if votes[label] >= 2]
            ),
            "is_strict_nonmoral": _bool_str(is_strict_nonmoral),
            "is_inclusive_nonmoral": _bool_str(is_inclusive_nonmoral),
            "has_thin_morality": _bool_str(has_thin),
            "has_nonmoral": _bool_str(has_nonmoral),
            "is_pure_foundation": _bool_str(is_pure_foundation),
        }
        for label in FOUNDATION_LABELS + SPECIAL_LABELS:
            slug = label.lower().replace(" ", "_")
            record[f"votes_{slug}"] = votes[label]
            record[f"present_{slug}"] = _bool_str(votes[label] >= 2)

        aggregated_rows.append(record)
        comment_records.append(
            {
                "text": text,
                "subreddit": subreddit,
                "bucket": bucket,
                "split": split,
                "retained_annotation_count": len(kept_rows),
                "original_annotation_count": len(rows),
                "present_foundations": "|".join(present_foundations),
                "has_thin_morality": _bool_str(has_thin),
                "has_nonmoral": _bool_str(has_nonmoral),
            }
        )
        if is_strict_nonmoral:
            strict_nonmoral_records.append(comment_records[-1])

        for foundation in present_foundations:
            if not has_nonmoral and not has_thin:
                positive_records[foundation].append(comment_records[-1])
                if is_pure_foundation:
                    pure_positive_records[foundation].append(comment_records[-1])

    aggregated_rows.sort(key=lambda row: (row["split"], row["bucket"], row["subreddit"], row["text"]))
    _write_csv(
        OUTPUT_DIR / "mfrc_aggregated_comments.csv",
        aggregated_rows,
        fieldnames=[
            "text",
            "subreddit",
            "bucket",
            "split",
            "original_annotation_count",
            "retained_annotation_count",
            "present_foundations",
            "present_label_count",
            "is_strict_nonmoral",
            "is_inclusive_nonmoral",
            "has_thin_morality",
            "has_nonmoral",
            "is_pure_foundation",
            *[f"votes_{label.lower().replace(' ', '_')}" for label in FOUNDATION_LABELS + SPECIAL_LABELS],
            *[f"present_{label.lower().replace(' ', '_')}" for label in FOUNDATION_LABELS + SPECIAL_LABELS],
        ],
    )

    for foundation in FOUNDATION_LABELS:
        slug = foundation.lower()
        binary_rows: list[dict[str, object]] = []
        for record in positive_records[foundation]:
            binary_rows.append(
                {
                    "text": record["text"],
                    "label": 1,
                    "split": record["split"],
                    "subreddit": record["subreddit"],
                    "bucket": record["bucket"],
                    "retained_annotation_count": record["retained_annotation_count"],
                    "original_annotation_count": record["original_annotation_count"],
                    "present_foundations": record["present_foundations"],
                    "target_foundation": foundation,
                    "positive_variant": "high_precision",
                }
            )
        for record in strict_nonmoral_records:
            binary_rows.append(
                {
                    "text": record["text"],
                    "label": 0,
                    "split": record["split"],
                    "subreddit": record["subreddit"],
                    "bucket": record["bucket"],
                    "retained_annotation_count": record["retained_annotation_count"],
                    "original_annotation_count": record["original_annotation_count"],
                    "present_foundations": record["present_foundations"],
                    "target_foundation": foundation,
                    "positive_variant": "high_precision",
                }
            )
        binary_rows.sort(key=lambda row: (row["split"], row["label"], row["bucket"], row["subreddit"], row["text"]))
        _write_csv(
            OUTPUT_DIR / f"mfrc_{slug}_vs_strict_nonmoral.csv",
            binary_rows,
            fieldnames=[
                "text",
                "label",
                "split",
                "subreddit",
                "bucket",
                "retained_annotation_count",
                "original_annotation_count",
                "present_foundations",
                "target_foundation",
                "positive_variant",
            ],
        )

        pure_rows: list[dict[str, object]] = []
        for record in pure_positive_records[foundation]:
            pure_rows.append(
                {
                    "text": record["text"],
                    "label": 1,
                    "split": record["split"],
                    "subreddit": record["subreddit"],
                    "bucket": record["bucket"],
                    "retained_annotation_count": record["retained_annotation_count"],
                    "original_annotation_count": record["original_annotation_count"],
                    "present_foundations": record["present_foundations"],
                    "target_foundation": foundation,
                    "positive_variant": "pure",
                }
            )
        for record in strict_nonmoral_records:
            pure_rows.append(
                {
                    "text": record["text"],
                    "label": 0,
                    "split": record["split"],
                    "subreddit": record["subreddit"],
                    "bucket": record["bucket"],
                    "retained_annotation_count": record["retained_annotation_count"],
                    "original_annotation_count": record["original_annotation_count"],
                    "present_foundations": record["present_foundations"],
                    "target_foundation": foundation,
                    "positive_variant": "pure",
                }
            )
        pure_rows.sort(key=lambda row: (row["split"], row["label"], row["bucket"], row["subreddit"], row["text"]))
        _write_csv(
            OUTPUT_DIR / f"mfrc_{slug}_pure_vs_strict_nonmoral.csv",
            pure_rows,
            fieldnames=[
                "text",
                "label",
                "split",
                "subreddit",
                "bucket",
                "retained_annotation_count",
                "original_annotation_count",
                "present_foundations",
                "target_foundation",
                "positive_variant",
            ],
        )

    strict_nonmoral_count = len(strict_nonmoral_records)
    lines = [
        "# MFRC Aggregation",
        "",
        "This directory contains comment-level MFRC datasets derived from the Hugging Face dataset",
        "`USC-MOLA-Lab/MFRC` using a high-precision aggregation rule.",
        "",
        "Aggregation rule:",
        "- Group by `(text, subreddit, bucket)`.",
        "- Drop `Not Confident` annotations.",
        "- Split multi-label annotations on commas.",
        "- A foundation is present if it receives at least 2 retained votes.",
        "- `strict_nonmoral` means `Non-Moral >= 2`, no foundation has 2 votes, and `Thin Morality < 2`.",
        "- High-precision positive examples require the target foundation to have 2 votes and both `Non-Moral < 2` and `Thin Morality < 2`.",
        "- Pure positives additionally require no other foundation to have 2 votes.",
        "",
        f"Aggregated comments: {len(aggregated_rows)}",
        f"Strict non-moral comments: {strict_nonmoral_count}",
        "",
        "High-precision foundation counts:",
    ]
    for foundation in FOUNDATION_LABELS:
        lines.append(f"- {foundation}: {len(positive_records[foundation])}")
    lines.append("")
    lines.append("Pure foundation counts:")
    for foundation in FOUNDATION_LABELS:
        lines.append(f"- {foundation}: {len(pure_positive_records[foundation])}")
    lines.append("")
    lines.append("Files:")
    lines.append("- `mfrc_aggregated_comments.csv`: comment-level votes and derived flags.")
    lines.append("- `mfrc_<foundation>_vs_strict_nonmoral.csv`: main one-vs-non-moral datasets.")
    lines.append("- `mfrc_<foundation>_pure_vs_strict_nonmoral.csv`: cleaner pure-positive variants.")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_outputs()
