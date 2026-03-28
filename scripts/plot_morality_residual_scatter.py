from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _load_split(activations_path: Path, projections_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    activations = np.load(activations_path)
    frame = pd.read_csv(projections_path)
    if activations.shape[0] != len(frame):
        raise ValueError(
            f"Mismatched rows: {activations_path} has {activations.shape[0]} rows but "
            f"{projections_path} has {len(frame)}."
        )
    return activations, frame


def _add_immoral_arrow(axis, split_df: pd.DataFrame, x_col: str, y_col: str) -> None:
    immoral_df = split_df[split_df["label"] == 1]
    if immoral_df.empty:
        return
    tip_x = float(immoral_df[x_col].mean())
    tip_y = float(immoral_df[y_col].mean())
    axis.annotate(
        "",
        xy=(tip_x, tip_y),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "linewidth": 2.0, "color": "black"},
        zorder=5,
    )
    axis.scatter([tip_x], [tip_y], c="black", s=30, marker="x", linewidths=1.5, zorder=6)
    axis.annotate(
        "immoral centroid",
        xy=(tip_x, tip_y),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9,
        color="black",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot learned morality projection against a residual orthogonal PCA axis."
    )
    parser.add_argument("--train-activations", required=True)
    parser.add_argument("--train-projections", required=True)
    parser.add_argument("--eval-activations", required=True)
    parser.add_argument("--eval-projections", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    train_acts, train_df = _load_split(Path(args.train_activations), Path(args.train_projections))
    eval_acts, eval_df = _load_split(Path(args.eval_activations), Path(args.eval_projections))
    direction = np.load(Path(args.direction)).astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    combined = np.concatenate([train_acts, eval_acts], axis=0)
    morality_projection = combined @ direction
    residual = combined - np.outer(morality_projection, direction)

    residual_pca = PCA(n_components=1, random_state=0)
    residual_pc1 = residual_pca.fit_transform(residual)[:, 0]

    train_count = len(train_df)
    train_plot_df = train_df.copy()
    train_plot_df["morality_projection"] = morality_projection[:train_count]
    train_plot_df["residual_pc1"] = residual_pc1[:train_count]
    train_plot_df["dataset_split"] = "train"

    eval_plot_df = eval_df.copy()
    eval_plot_df["morality_projection"] = morality_projection[train_count:]
    eval_plot_df["residual_pc1"] = residual_pc1[train_count:]
    eval_plot_df["dataset_split"] = "eval"

    plot_df = pd.concat([train_plot_df, eval_plot_df], ignore_index=True)
    plot_df["class_name"] = plot_df["label"].map({0: "acceptable", 1: "immoral"}).fillna("unknown")

    import matplotlib.pyplot as plt

    colors = {"acceptable": "#1f77b4", "immoral": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, split_name, split_df in [
        (axes[0], "Train", train_plot_df),
        (axes[1], "Eval", eval_plot_df),
    ]:
        split_df = split_df.copy()
        split_df["class_name"] = split_df["label"].map({0: "acceptable", 1: "immoral"}).fillna("unknown")
        for class_name in ["acceptable", "immoral"]:
            class_df = split_df[split_df["class_name"] == class_name]
            axis.scatter(
                class_df["morality_projection"],
                class_df["residual_pc1"],
                s=10,
                alpha=0.35,
                c=colors[class_name],
                label=class_name,
                edgecolors="none",
            )
        axis.set_title(split_name)
        axis.set_xlabel("Projection onto learned morality direction")
        axis.set_ylabel("Residual PC1 (orthogonal to morality)")
        _add_immoral_arrow(axis, split_df, "morality_projection", "residual_pc1")
        axis.legend(frameon=False)

    explained = residual_pca.explained_variance_ratio_[0]
    fig.suptitle(
        f"ETHICS Layer-15 Activations: Morality Axis vs Residual PC1 "
        f"(Residual PC1 {explained:.1%} variance)"
    )
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
