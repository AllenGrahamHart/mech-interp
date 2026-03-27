from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.metrics import roc_auc_score


def load_aurocs(path: Path) -> dict[tuple[str, float], float]:
    rows = list(csv.DictReader(path.open()))
    by_key: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["mode"], float(row["scale"]))
        by_key.setdefault(key, []).append(row)

    aurocs: dict[tuple[str, float], float] = {}
    for key, group_rows in by_key.items():
        labels = [int(row["label"]) for row in group_rows]
        gaps = [float(row["logprob_gap"]) for row in group_rows]
        aurocs[key] = float(roc_auc_score(labels, gaps))
    return aurocs


def plot_aurocs(
    full_morality: dict[tuple[str, float], float],
    pure_sentiment: dict[tuple[str, float], float],
    morality_residual: dict[tuple[str, float], float],
    output_path: Path,
    csv_path: Path,
    baseline_remove_only: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if baseline_remove_only:
        order = [
            ("baseline", 0.0, "Baseline"),
            ("remove", 0.0, "Remove"),
        ]
    else:
        order = [
            ("baseline", 0.0, "Baseline"),
            ("remove", 0.0, "Remove"),
            ("add", -1.0, "-1"),
            ("add", 1.0, "+1"),
            ("add", -3.0, "-3"),
            ("add", 3.0, "+3"),
        ]

    series = [
        ("Full morality", full_morality, "#1f77b4"),
        ("Pure sentiment", pure_sentiment, "#d62728"),
        ("Morality residual", morality_residual, "#2ca02c"),
    ]

    rows = []
    x = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, values, color in series:
        y = [values[(mode, scale)] for mode, scale, _ in order]
        ax.plot(x, y, marker="o", linewidth=2, markersize=6, label=label, color=color)
        for idx, score in enumerate(y):
            rows.append(
                {
                    "series": label,
                    "mode": order[idx][0],
                    "scale": order[idx][1],
                    "display_label": order[idx][2],
                    "auroc": score,
                }
            )

    ax.set_xticks(x, [display for _, _, display in order])
    ax.set_xlabel("Intervention")
    ax.set_ylabel("AUROC on ETHICS")
    ax.set_title("ETHICS AUROC Under Direction Interventions at Layer 15")
    ax.set_ylim(0.6, 0.82)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["series", "mode", "scale", "display_label", "auroc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AUROC comparisons across intervention runs.")
    parser.add_argument("--full-morality", required=True, help="Path to intervention_results.csv for full morality.")
    parser.add_argument("--pure-sentiment", required=True, help="Path to intervention_results.csv for pure sentiment.")
    parser.add_argument("--morality-residual", required=True, help="Path to intervention_results.csv for orthogonalized morality.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path for plotted AUROCs.")
    parser.add_argument(
        "--baseline-remove-only",
        action="store_true",
        help="Plot only baseline and remove conditions.",
    )
    args = parser.parse_args()

    plot_aurocs(
        full_morality=load_aurocs(Path(args.full_morality)),
        pure_sentiment=load_aurocs(Path(args.pure_sentiment)),
        morality_residual=load_aurocs(Path(args.morality_residual)),
        output_path=Path(args.output),
        csv_path=Path(args.output_csv),
        baseline_remove_only=args.baseline_remove_only,
    )


if __name__ == "__main__":
    main()
