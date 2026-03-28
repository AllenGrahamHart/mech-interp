from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_best_directions(base_dir: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for run_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        summary = json.loads((run_dir / "summary.json").read_text())
        best_layer = int(summary["best_layer_by_eval_auroc"])
        layer_rows = list(csv.DictReader((run_dir / "layer_summary.csv").open()))
        best_row = next(row for row in layer_rows if int(row["layer"]) == best_layer)
        direction = np.load(run_dir / f"layer_{best_layer:02d}_direction.npy").astype(np.float64)
        direction /= np.linalg.norm(direction)
        name = run_dir.name.split("_layers_")[0].replace("mfrc_", "").replace(
            "_vs_strict_nonmoral", ""
        )
        items.append(
            {
                "name": name,
                "run_dir": str(run_dir),
                "best_layer": best_layer,
                "eval_auroc": float(best_row["eval_auroc"]),
                "train_examples": int(summary["train_examples"]),
                "eval_examples": int(summary["eval_examples"]),
                "direction": direction,
            }
        )
    return items


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base_dir = Path("tmp/modal_results/mfrc_foundations")
    output_dir = Path("tmp/modal_results/mfrc_foundation_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    items = _load_best_directions(base_dir)
    ethics_direction = np.load("tmp/modal_results/layer_15_direction_full.npy").astype(np.float64)
    ethics_direction /= np.linalg.norm(ethics_direction)
    sentiment_direction = np.load(
        "tmp/modal_results/sentiment_matched_directions/"
        "sentiment_direction_open_matched_layers_20260327_223456/layer_14_direction.npy"
    ).astype(np.float64)
    sentiment_direction = -sentiment_direction / np.linalg.norm(sentiment_direction)

    summary_rows: list[dict[str, object]] = []
    cosine_rows: list[dict[str, object]] = []

    names = [str(item["name"]) for item in items]
    matrix = np.zeros((len(items), len(items)), dtype=np.float64)
    for i, item_i in enumerate(items):
        u = np.asarray(item_i["direction"], dtype=np.float64)
        summary_rows.append(
            {
                "foundation": item_i["name"],
                "best_layer": item_i["best_layer"],
                "eval_auroc": item_i["eval_auroc"],
                "train_examples": item_i["train_examples"],
                "eval_examples": item_i["eval_examples"],
                "cosine_to_ethics_wrongness_layer15": float(np.dot(u, ethics_direction)),
                "cosine_to_negative_sentiment_layer14": float(np.dot(u, sentiment_direction)),
            }
        )
        row = {"foundation": item_i["name"]}
        for j, item_j in enumerate(items):
            v = np.asarray(item_j["direction"], dtype=np.float64)
            cosine = float(np.dot(u, v))
            matrix[i, j] = cosine
            row[str(item_j["name"])] = cosine
        cosine_rows.append(row)

    _write_csv(
        output_dir / "mfrc_foundation_summary.csv",
        summary_rows,
        fieldnames=[
            "foundation",
            "best_layer",
            "eval_auroc",
            "train_examples",
            "eval_examples",
            "cosine_to_ethics_wrongness_layer15",
            "cosine_to_negative_sentiment_layer14",
        ],
    )
    _write_csv(
        output_dir / "mfrc_foundation_cosine_matrix.csv",
        cosine_rows,
        fieldnames=["foundation", *names],
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("MFRC Foundation Direction Cosine Similarity")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(output_dir / "mfrc_foundation_cosine_heatmap.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
