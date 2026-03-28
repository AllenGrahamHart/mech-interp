from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_activation_split(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    activations = np.load(run_dir / "train_activations.npy").astype(np.float32)
    frame = pd.read_csv(run_dir / "train_projections.csv")
    if activations.shape[0] != len(frame):
        raise ValueError(
            f"Mismatched rows in {run_dir}: {activations.shape[0]} activations vs {len(frame)} labels."
        )
    labels = frame["label"].to_numpy(dtype=np.int64)
    return activations, labels


def _fit_direction(activations: np.ndarray, labels: np.ndarray, target_label: int) -> np.ndarray:
    target_mask = labels == target_label
    other_mask = ~target_mask
    direction = activations[target_mask].mean(axis=0) - activations[other_mask].mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm <= 0:
        raise ValueError("Degenerate direction with zero norm.")
    return (direction / norm).astype(np.float32)


def _bootstrap_directions(
    activations: np.ndarray,
    labels: np.ndarray,
    target_label: int,
    num_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_idx = np.flatnonzero(labels == target_label)
    other_idx = np.flatnonzero(labels != target_label)
    rng = np.random.default_rng(seed)

    full_direction = _fit_direction(activations, labels, target_label)
    boot_directions = np.empty((num_bootstrap, activations.shape[1]), dtype=np.float32)

    for bootstrap_id in range(num_bootstrap):
        sampled_target = rng.choice(target_idx, size=len(target_idx), replace=True)
        sampled_other = rng.choice(other_idx, size=len(other_idx), replace=True)
        sampled_indices = np.concatenate([sampled_target, sampled_other])
        sampled_labels = np.concatenate(
            [
                np.full(len(sampled_target), target_label, dtype=np.int64),
                np.full(len(sampled_other), 1 - target_label, dtype=np.int64),
            ]
        )
        boot_directions[bootstrap_id] = _fit_direction(
            activations[sampled_indices],
            sampled_labels,
            target_label,
        )

    cosines = np.clip(boot_directions @ full_direction, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cosines))
    return full_direction, boot_directions, cosines, angles_deg


def _summarize_distribution(values: np.ndarray) -> dict[str, float | list[float]]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "p02_5": float(np.percentile(values, 2.5)),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p97_5": float(np.percentile(values, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap angular uncertainty for immorality and negative-sentiment directions."
    )
    parser.add_argument("--morality-run-dir", required=True)
    parser.add_argument("--sentiment-run-dir", required=True)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    morality_dir = Path(args.morality_run_dir)
    sentiment_dir = Path(args.sentiment_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    morality_acts, morality_labels = _load_activation_split(morality_dir)
    sentiment_acts, sentiment_labels = _load_activation_split(sentiment_dir)

    seed_seq = np.random.SeedSequence(args.seed)
    morality_seed, sentiment_seed = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(2)]

    immorality_direction, immorality_boot, immorality_cos, immorality_angle = _bootstrap_directions(
        morality_acts,
        morality_labels,
        target_label=1,
        num_bootstrap=args.num_bootstrap,
        seed=morality_seed,
    )
    (
        negative_sentiment_direction,
        negative_sentiment_boot,
        negative_sentiment_cos,
        negative_sentiment_angle,
    ) = _bootstrap_directions(
        sentiment_acts,
        sentiment_labels,
        target_label=0,
        num_bootstrap=args.num_bootstrap,
        seed=sentiment_seed,
    )

    observed_overlap_cos = float(
        np.clip(np.dot(immorality_direction, negative_sentiment_direction), -1.0, 1.0)
    )
    observed_overlap_angle = float(np.degrees(np.arccos(observed_overlap_cos)))

    overlap_cos = np.clip(
        np.sum(immorality_boot * negative_sentiment_boot, axis=1),
        -1.0,
        1.0,
    )
    overlap_angle = np.degrees(np.arccos(overlap_cos))

    replicates = pd.DataFrame(
        {
            "bootstrap_id": np.arange(args.num_bootstrap, dtype=np.int64),
            "immorality_cosine_to_full": immorality_cos,
            "immorality_angle_deg": immorality_angle,
            "negative_sentiment_cosine_to_full": negative_sentiment_cos,
            "negative_sentiment_angle_deg": negative_sentiment_angle,
            "overlap_cosine": overlap_cos,
            "overlap_angle_deg": overlap_angle,
        }
    )
    replicates.to_csv(output_dir / "bootstrap_replicates.csv", index=False)

    summary = {
        "num_bootstrap": args.num_bootstrap,
        "seed": args.seed,
        "immorality_direction": {
            "train_examples": int(len(morality_labels)),
            "target_examples": int(np.sum(morality_labels == 1)),
            "other_examples": int(np.sum(morality_labels != 1)),
            "cosine_to_full": _summarize_distribution(immorality_cos),
            "angle_deg_to_full": _summarize_distribution(immorality_angle),
        },
        "negative_sentiment_direction": {
            "train_examples": int(len(sentiment_labels)),
            "target_examples": int(np.sum(sentiment_labels == 0)),
            "other_examples": int(np.sum(sentiment_labels != 0)),
            "cosine_to_full": _summarize_distribution(negative_sentiment_cos),
            "angle_deg_to_full": _summarize_distribution(negative_sentiment_angle),
        },
        "direction_overlap": {
            "observed_cosine": observed_overlap_cos,
            "observed_angle_deg": observed_overlap_angle,
            "bootstrap_cosine": _summarize_distribution(overlap_cos),
            "bootstrap_angle_deg": _summarize_distribution(overlap_angle),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
