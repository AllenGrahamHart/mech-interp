#!/usr/bin/env python3
"""Project the ETHICS wrongness direction onto the MFRC foundation span."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_FOUNDATIONS = {
    "care": "mfrc_care_vs_strict_nonmoral_layers_20260328_141458",
    "equality": "mfrc_equality_vs_strict_nonmoral_layers_20260328_141937",
    "proportionality": "mfrc_proportionality_vs_strict_nonmoral_layers_20260328_141936",
    "purity": "mfrc_purity_vs_strict_nonmoral_layers_20260328_141939",
    "authority": "mfrc_authority_vs_strict_nonmoral_layers_20260328_141938",
    "loyalty": "mfrc_loyalty_vs_strict_nonmoral_layers_20260328_141936",
}


def load_unit_vector(path: Path) -> np.ndarray:
    vector = np.load(path).astype(np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(f"zero vector at {path}")
    return vector / norm


def layer_summary(
    wrongness_dir: Path,
    foundation_dirs: dict[str, Path],
    layer: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    wrongness = load_unit_vector(wrongness_dir / f"layer_{layer:02d}_direction.npy")
    names = list(foundation_dirs)
    foundation_vectors = [
        load_unit_vector(foundation_dirs[name] / f"layer_{layer:02d}_direction.npy")
        for name in names
    ]
    basis_matrix = np.column_stack(foundation_vectors)
    rank = int(np.linalg.matrix_rank(basis_matrix))
    q, _ = np.linalg.qr(basis_matrix)
    q = q[:, :rank]
    projection = q @ (q.T @ wrongness)
    residual = wrongness - projection
    explained_fraction = float(np.linalg.norm(projection) ** 2)
    subspace_angle_deg = float(
        np.degrees(np.arcsin(np.clip(np.linalg.norm(residual), -1.0, 1.0)))
    )
    ls_coef, *_ = np.linalg.lstsq(basis_matrix, wrongness, rcond=None)
    individual_cosines = {
        name: float(np.dot(wrongness, vector))
        for name, vector in zip(names, foundation_vectors)
    }
    ls_coefficients = {
        name: float(coef) for name, coef in zip(names, ls_coef, strict=True)
    }
    summary = {
        "layer": layer,
        "rank": rank,
        "explained_fraction": explained_fraction,
        "subspace_angle_deg": subspace_angle_deg,
    }
    return summary, individual_cosines, ls_coefficients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrongness-dir",
        type=Path,
        default=Path(
            "tmp/modal_results/morality_full_directions/"
            "morality_direction_open_full_layers_20260327_190213"
        ),
    )
    parser.add_argument(
        "--foundations-root",
        type=Path,
        default=Path("tmp/modal_results/mfrc_foundations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/modal_results/foundation_subspace_projection"),
    )
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--focus-layer", type=int, default=15)
    args = parser.parse_args()

    foundation_dirs = {
        name: args.foundations_root / run for name, run in DEFAULT_FOUNDATIONS.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_rows: list[dict[str, float]] = []
    layer_details: dict[int, dict[str, object]] = {}

    for layer in range(args.num_layers):
        summary, cosines, coeffs = layer_summary(
            args.wrongness_dir, foundation_dirs, layer
        )
        layer_rows.append(summary)
        layer_details[layer] = {
            "summary": summary,
            "individual_cosines": cosines,
            "least_squares_coefficients": coeffs,
        }

    best_row = max(layer_rows, key=lambda row: row["explained_fraction"])
    focus_detail = layer_details[args.focus_layer]
    best_detail = layer_details[int(best_row["layer"])]

    with (args.output_dir / "layerwise_projection_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "rank", "explained_fraction", "subspace_angle_deg"],
        )
        writer.writeheader()
        writer.writerows(layer_rows)

    summary = {
        "focus_layer": args.focus_layer,
        "focus_layer_explained_fraction": focus_detail["summary"]["explained_fraction"],
        "focus_layer_subspace_angle_deg": focus_detail["summary"]["subspace_angle_deg"],
        "best_layer": int(best_row["layer"]),
        "best_layer_explained_fraction": best_row["explained_fraction"],
        "best_layer_subspace_angle_deg": best_row["subspace_angle_deg"],
        "foundations": list(DEFAULT_FOUNDATIONS),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "focus_layer_detail.json").write_text(
        json.dumps(focus_detail, indent=2)
    )
    (args.output_dir / "best_layer_detail.json").write_text(
        json.dumps(best_detail, indent=2)
    )


if __name__ == "__main__":
    main()
