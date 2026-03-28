from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mech_interp.config import DirectionConfig


FOUNDATIONS = ["care", "loyalty", "proportionality", "authority", "equality", "purity"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_run_key(run_dir: Path) -> tuple[str, int]:
    stem = run_dir.name
    prefix, _, suffix = stem.partition("_same_layer_")
    foundation = prefix.replace("mfrc_", "")
    layer = int(suffix.split("_", 1)[0])
    return foundation, layer


def _load_run_metadata(run_dir: Path) -> dict[str, object]:
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    labels = np.asarray(
        [int(row["label"]) for row in _read_csv(run_dir / "train_projections.csv")], dtype=np.int64
    )
    activations = np.load(run_dir / "train_activations.npy").astype(np.float64)
    direction = np.load(run_dir / "direction.npy").astype(np.float64)
    direction /= np.linalg.norm(direction)
    return {
        "run_dir": str(run_dir),
        "config": DirectionConfig(**config["direction"]),
        "summary": summary,
        "labels": labels,
        "activations": activations,
        "direction": direction,
    }


def _bootstrap_directions(
    activations: np.ndarray,
    labels: np.ndarray,
    direction_config: DirectionConfig,
    repeats: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(labels == direction_config.positive_label)
    neg_idx = np.flatnonzero(labels != direction_config.positive_label)
    out = np.zeros((repeats, activations.shape[1]), dtype=np.float64)
    for repeat in range(repeats):
        sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        positive_mean = activations[sampled_pos].mean(axis=0)
        negative_mean = activations[sampled_neg].mean(axis=0)
        direction = positive_mean - negative_mean
        out[repeat] = direction / np.linalg.norm(direction)
    return out


def _angle_from_cosine(cosine: np.ndarray | float) -> np.ndarray | float:
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _format_angle(observed: float, lower: float, upper: float) -> str:
    return f"{observed:.1f} [{lower:.1f}, {upper:.1f}]"


def _format_auroc(value: float) -> str:
    return f"{value:.3f}"


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="tmp/modal_results/mfrc_same_layer_runs")
    parser.add_argument(
        "--best-layer-summary",
        default="tmp/modal_results/mfrc_layerwise_structure/mfrc_best_layer_summary.csv",
    )
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="tmp/modal_results/mfrc_same_layer_tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_layer_rows = _read_csv(Path(args.best_layer_summary))
    best_layer_map = {row["foundation"]: int(row["best_layer"]) for row in best_layer_rows}
    row_order = sorted(FOUNDATIONS, key=lambda foundation: (best_layer_map[foundation], foundation))

    runs_dir = Path(args.runs_dir)
    run_map: dict[tuple[str, int], Path] = {}
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        foundation, layer = _parse_run_key(run_dir)
        run_map[(foundation, layer)] = run_dir

    metadata_map = {
        key: _load_run_metadata(run_dir)
        for key, run_dir in run_map.items()
    }

    bootstrap_map = {}
    for index, (key, metadata) in enumerate(sorted(metadata_map.items())):
        bootstrap_map[key] = _bootstrap_directions(
            activations=np.asarray(metadata["activations"], dtype=np.float64),
            labels=np.asarray(metadata["labels"], dtype=np.int64),
            direction_config=metadata["config"],
            repeats=args.repeats,
            seed=args.seed + 1000 * (index + 1),
        )

    angle_rows_numeric: list[dict[str, object]] = []
    angle_rows_formatted: list[dict[str, object]] = []
    auroc_rows_numeric: list[dict[str, object]] = []
    auroc_rows_formatted: list[dict[str, object]] = []

    for focal in row_order:
        focal_layer = best_layer_map[focal]
        row_label = f"{focal.capitalize()} (L{focal_layer})"
        focal_key = (focal, focal_layer)
        focal_direction = np.asarray(metadata_map[focal_key]["direction"], dtype=np.float64)
        focal_boot = np.asarray(bootstrap_map[focal_key], dtype=np.float64)

        angle_numeric = {"row": row_label, "focal_foundation": focal, "focal_layer": focal_layer}
        angle_formatted = {"row": row_label}
        auroc_numeric = {"row": row_label, "focal_foundation": focal, "focal_layer": focal_layer}
        auroc_formatted = {"row": row_label}

        for other in FOUNDATIONS:
            other_key = (other, focal_layer)
            other_direction = np.asarray(metadata_map[other_key]["direction"], dtype=np.float64)
            other_boot = np.asarray(bootstrap_map[other_key], dtype=np.float64)

            if other == focal:
                observed_angle = 0.0
                lower = 0.0
                upper = 0.0
                angle_formatted[other.capitalize()] = "--"
            else:
                observed_cosine = float(np.dot(focal_direction, other_direction))
                observed_angle = float(_angle_from_cosine(observed_cosine))
                bootstrap_cosines = np.sum(focal_boot * other_boot, axis=1)
                bootstrap_angles = _angle_from_cosine(bootstrap_cosines)
                lower = float(np.percentile(bootstrap_angles, 2.5))
                upper = float(np.percentile(bootstrap_angles, 97.5))
                angle_formatted[other.capitalize()] = _format_angle(observed_angle, lower, upper)

            angle_numeric[f"{other}_angle_deg"] = observed_angle
            angle_numeric[f"{other}_angle_ci_lower"] = lower
            angle_numeric[f"{other}_angle_ci_upper"] = upper

            eval_auroc = float(metadata_map[other_key]["summary"]["eval_metrics"]["auroc"])
            auroc_numeric[f"{other}_eval_auroc"] = eval_auroc
            auroc_formatted[other.capitalize()] = _format_auroc(eval_auroc)

        angle_rows_numeric.append(angle_numeric)
        angle_rows_formatted.append(angle_formatted)
        auroc_rows_numeric.append(auroc_numeric)
        auroc_rows_formatted.append(auroc_formatted)

    foundation_headers = [foundation.capitalize() for foundation in FOUNDATIONS]
    _write_csv(
        output_dir / "angle_table_numeric.csv",
        angle_rows_numeric,
        fieldnames=[
            "row",
            "focal_foundation",
            "focal_layer",
            *[
                field
                for foundation in FOUNDATIONS
                for field in (
                    f"{foundation}_angle_deg",
                    f"{foundation}_angle_ci_lower",
                    f"{foundation}_angle_ci_upper",
                )
            ],
        ],
    )
    _write_csv(
        output_dir / "angle_table_formatted.csv",
        angle_rows_formatted,
        fieldnames=["row", *foundation_headers],
    )
    _write_csv(
        output_dir / "auroc_table_numeric.csv",
        auroc_rows_numeric,
        fieldnames=[
            "row",
            "focal_foundation",
            "focal_layer",
            *[f"{foundation}_eval_auroc" for foundation in FOUNDATIONS],
        ],
    )
    _write_csv(
        output_dir / "auroc_table_formatted.csv",
        auroc_rows_formatted,
        fieldnames=["row", *foundation_headers],
    )

    angle_lines = [
        "\\begin{tabular}{l" + "c" * len(FOUNDATIONS) + "}",
        "\\toprule",
        "Row & " + " & ".join(foundation_headers) + " \\\\",
        "\\midrule",
    ]
    for row in angle_rows_formatted:
        angle_lines.append(
            _latex_escape(str(row["row"]))
            + " & "
            + " & ".join(_latex_escape(str(row[header])) for header in foundation_headers)
            + " \\\\"
        )
    angle_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output_dir / "angle_table.tex").write_text("\n".join(angle_lines), encoding="utf-8")

    auroc_lines = [
        "\\begin{tabular}{l" + "c" * len(FOUNDATIONS) + "}",
        "\\toprule",
        "Row & " + " & ".join(foundation_headers) + " \\\\",
        "\\midrule",
    ]
    for row in auroc_rows_formatted:
        auroc_lines.append(
            _latex_escape(str(row["row"]))
            + " & "
            + " & ".join(_latex_escape(str(row[header])) for header in foundation_headers)
            + " \\\\"
        )
    auroc_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output_dir / "auroc_table.tex").write_text("\n".join(auroc_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
