from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mech_interp.config import DirectionConfig
from mech_interp.direction import fit_direction


def _load_labels(run_dir: Path, split_name: str = "train") -> np.ndarray:
    path = run_dir / f"{split_name}_projections.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def _load_direction_config(run_dir: Path) -> DirectionConfig:
    raw = json.loads((run_dir / "config.json").read_text())
    return DirectionConfig(**raw["direction"])


def _bootstrap_direction(
    activations: np.ndarray,
    labels: np.ndarray,
    direction_config: DirectionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    pos_idx = np.flatnonzero(labels == direction_config.positive_label)
    neg_idx = np.flatnonzero(labels != direction_config.positive_label)
    sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    indices = np.concatenate([sampled_pos, sampled_neg])
    sampled_activations = activations[indices]
    sampled_labels = labels[indices]
    return fit_direction(sampled_activations, sampled_labels, direction_config).direction.astype(
        np.float64
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activations_a = np.load(run_a / "train_activations.npy").astype(np.float64)
    activations_b = np.load(run_b / "train_activations.npy").astype(np.float64)
    labels_a = _load_labels(run_a)
    labels_b = _load_labels(run_b)
    config_a = _load_direction_config(run_a)
    config_b = _load_direction_config(run_b)

    full_dir_a = np.load(run_a / "direction.npy").astype(np.float64)
    full_dir_b = np.load(run_b / "direction.npy").astype(np.float64)
    full_dir_a /= np.linalg.norm(full_dir_a)
    full_dir_b /= np.linalg.norm(full_dir_b)

    observed_cosine = float(np.dot(full_dir_a, full_dir_b))
    observed_angle_deg = float(np.degrees(np.arccos(np.clip(observed_cosine, -1.0, 1.0))))

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int]] = []
    cosines = []
    angles = []
    cos_to_full_a = []
    cos_to_full_b = []
    for repeat in range(args.repeats):
        dir_a = _bootstrap_direction(activations_a, labels_a, config_a, rng)
        dir_b = _bootstrap_direction(activations_b, labels_b, config_b, rng)
        dir_a /= np.linalg.norm(dir_a)
        dir_b /= np.linalg.norm(dir_b)
        cosine = float(np.dot(dir_a, dir_b))
        angle_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        rows.append(
            {
                "repeat": repeat,
                "cosine_between_directions": cosine,
                "angle_deg_between_directions": angle_deg,
                "cosine_to_full_a": float(np.dot(dir_a, full_dir_a)),
                "cosine_to_full_b": float(np.dot(dir_b, full_dir_b)),
            }
        )
        cosines.append(cosine)
        angles.append(angle_deg)
        cos_to_full_a.append(float(np.dot(dir_a, full_dir_a)))
        cos_to_full_b.append(float(np.dot(dir_b, full_dir_b)))

    summary = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "repeats": args.repeats,
        "observed_cosine": observed_cosine,
        "observed_angle_deg": observed_angle_deg,
        "cosine_mean": float(np.mean(cosines)),
        "cosine_ci_lower": float(np.percentile(cosines, 2.5)),
        "cosine_ci_upper": float(np.percentile(cosines, 97.5)),
        "angle_deg_mean": float(np.mean(angles)),
        "angle_deg_ci_lower": float(np.percentile(angles, 2.5)),
        "angle_deg_ci_upper": float(np.percentile(angles, 97.5)),
        "cosine_to_full_a_mean": float(np.mean(cos_to_full_a)),
        "cosine_to_full_b_mean": float(np.mean(cos_to_full_b)),
        "cosine_to_full_a_ci_lower": float(np.percentile(cos_to_full_a, 2.5)),
        "cosine_to_full_a_ci_upper": float(np.percentile(cos_to_full_a, 97.5)),
        "cosine_to_full_b_ci_lower": float(np.percentile(cos_to_full_b, 2.5)),
        "cosine_to_full_b_ci_upper": float(np.percentile(cos_to_full_b, 97.5)),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (output_dir / "bootstrap_replicates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "repeat",
                "cosine_between_directions",
                "angle_deg_between_directions",
                "cosine_to_full_a",
                "cosine_to_full_b",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
