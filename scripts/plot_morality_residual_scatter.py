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


def _centroid_tip(split_df: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float] | None:
    immoral_df = split_df[split_df["label"] == 1]
    if immoral_df.empty:
        return None
    return float(immoral_df[x_col].mean()), float(immoral_df[y_col].mean())


def _add_arrow(axis, tip_x: float, tip_y: float, color: str = "black", linestyle: str = "-") -> None:
    axis.annotate(
        "",
        xy=(tip_x, tip_y),
        xytext=(0.0, 0.0),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 2.0,
            "color": color,
            "linestyle": linestyle,
        },
        zorder=5,
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
    parser.add_argument("--comparison-direction")
    parser.add_argument("--comparison-sign", type=float, default=1.0)
    parser.add_argument("--comparison-label", default="negative sentiment direction")
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
    from matplotlib.legend_handler import HandlerPatch
    from matplotlib.patches import FancyArrowPatch

    colors = {"acceptable": "#1f77b4", "immoral": "#d62728"}
    immorality_direction_handle = FancyArrowPatch(
        (0.0, 0.0),
        (1.0, 0.0),
        arrowstyle="->",
        linewidth=2.0,
        color="black",
        label="immorality direction",
    )
    comparison_direction_handle = FancyArrowPatch(
        (0.0, 0.0),
        (1.0, 0.0),
        arrowstyle="->",
        linewidth=2.0,
        linestyle="--",
        color="#2ca02c",
        label=args.comparison_label,
    )
    arrow_handler = HandlerPatch(
        patch_func=lambda legend, orig_handle, xdescent, ydescent, width, height, fontsize: FancyArrowPatch(
            (xdescent, ydescent + 0.5 * height),
            (xdescent + width, ydescent + 0.5 * height),
            arrowstyle="->",
            mutation_scale=max(fontsize * 1.1, 12.0),
            linewidth=2.0,
            color=orig_handle.get_edgecolor(),
            linestyle=orig_handle.get_linestyle(),
        )
    )
    comparison_xy = None
    if args.comparison_direction:
        comparison_direction = np.load(Path(args.comparison_direction)).astype(np.float32)
        comparison_direction = args.comparison_sign * comparison_direction
        comparison_direction = comparison_direction / np.linalg.norm(comparison_direction)
        comparison_projection = float(comparison_direction @ direction)
        comparison_residual = comparison_direction - comparison_projection * direction
        comparison_residual_pc1 = float(residual_pca.transform(comparison_residual.reshape(1, -1))[0, 0])
        comparison_xy = np.array([comparison_projection, comparison_residual_pc1], dtype=np.float32)
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
        axis.set_xlabel("Projection onto learned immorality direction")
        axis.set_ylabel("Residual PC1 (orthogonal to immorality)")
        centroid_tip = _centroid_tip(split_df, "morality_projection", "residual_pc1")
        if centroid_tip is not None:
            tip_x, tip_y = centroid_tip
            _add_arrow(axis, tip_x, tip_y, color="black", linestyle="-")
            if comparison_xy is not None:
                comp_norm = float(np.linalg.norm(comparison_xy))
                if comp_norm > 0:
                    target_norm = float(np.hypot(tip_x, tip_y))
                    scaled = comparison_xy * (target_norm / comp_norm)
                    _add_arrow(
                        axis,
                        float(scaled[0]),
                        float(scaled[1]),
                        color="#2ca02c",
                        linestyle="--",
                    )
        handles, labels = axis.get_legend_handles_labels()
        handles.append(immorality_direction_handle)
        labels.append("immorality direction")
        if comparison_xy is not None:
            handles.append(comparison_direction_handle)
            labels.append(args.comparison_label)
        axis.legend(handles, labels, frameon=False, handler_map={FancyArrowPatch: arrow_handler})

    explained = residual_pca.explained_variance_ratio_[0]
    fig.suptitle(
        f"ETHICS Layer-15 Activations: Immorality Axis vs Residual PC1 "
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
