from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_run_payloads(base_dir: Path) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for run_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        name = run_dir.name.split("_layers_")[0].replace("mfrc_", "").replace(
            "_vs_strict_nonmoral", ""
        )
        summary = json.loads((run_dir / "summary.json").read_text())
        layer_rows = list(csv.DictReader((run_dir / "layer_summary.csv").open()))
        directions = {
            int(path.stem.split("_")[1]): np.load(path).astype(np.float64)
            for path in run_dir.glob("layer_*_direction.npy")
        }
        for layer, vector in directions.items():
            directions[layer] = vector / np.linalg.norm(vector)
        payloads[name] = {
            "run_dir": str(run_dir),
            "summary": summary,
            "layer_rows": layer_rows,
            "directions": directions,
        }
    return payloads


def _plot_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    base_dir = Path("tmp/modal_results/mfrc_foundations")
    output_dir = Path("tmp/modal_results/mfrc_layerwise_structure")
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = _load_run_payloads(base_dir)
    foundations = sorted(payloads.keys())
    all_layers = sorted(
        {
            int(row["layer"])
            for payload in payloads.values()
            for row in payload["layer_rows"]
        }
    )

    best_rows: list[dict[str, object]] = []
    layer_auroc_rows: list[dict[str, object]] = []
    auroc_matrix = np.zeros((len(foundations), len(all_layers)), dtype=np.float64)

    for i, foundation in enumerate(foundations):
        payload = payloads[foundation]
        summary = payload["summary"]
        best_layer = int(summary["best_layer_by_eval_auroc"])
        layer_rows = {
            int(row["layer"]): row
            for row in payload["layer_rows"]
        }
        best_row = layer_rows[best_layer]
        best_rows.append(
            {
                "foundation": foundation,
                "best_layer": best_layer,
                "best_eval_auroc": float(best_row["eval_auroc"]),
                "train_examples": int(summary["train_examples"]),
                "eval_examples": int(summary["eval_examples"]),
                "run_dir": payload["run_dir"],
            }
        )
        for j, layer in enumerate(all_layers):
            eval_auroc = float(layer_rows[layer]["eval_auroc"])
            auroc_matrix[i, j] = eval_auroc
            layer_auroc_rows.append(
                {
                    "foundation": foundation,
                    "layer": layer,
                    "eval_auroc": eval_auroc,
                    "eval_accuracy": float(layer_rows[layer]["eval_accuracy"]),
                    "eval_balanced_accuracy": float(layer_rows[layer]["eval_balanced_accuracy"]),
                    "train_auroc": float(layer_rows[layer]["train_auroc"]),
                }
            )

    _write_csv(
        output_dir / "mfrc_best_layer_summary.csv",
        best_rows,
        fieldnames=[
            "foundation",
            "best_layer",
            "best_eval_auroc",
            "train_examples",
            "eval_examples",
            "run_dir",
        ],
    )
    _write_csv(
        output_dir / "mfrc_layer_auroc_long.csv",
        layer_auroc_rows,
        fieldnames=[
            "foundation",
            "layer",
            "eval_auroc",
            "eval_accuracy",
            "eval_balanced_accuracy",
            "train_auroc",
        ],
    )

    auroc_wide_rows: list[dict[str, object]] = []
    for foundation, row_values in zip(foundations, auroc_matrix, strict=True):
        row = {"foundation": foundation}
        for layer, value in zip(all_layers, row_values, strict=True):
            row[f"layer_{layer:02d}"] = float(value)
        auroc_wide_rows.append(row)
    _write_csv(
        output_dir / "mfrc_layer_auroc_matrix.csv",
        auroc_wide_rows,
        fieldnames=["foundation", *[f"layer_{layer:02d}" for layer in all_layers]],
    )
    _plot_heatmap(
        matrix=auroc_matrix,
        row_labels=foundations,
        col_labels=[str(layer) for layer in all_layers],
        title="MFRC Foundation Decoding AUROC by Layer",
        colorbar_label="Eval AUROC",
        output_path=output_dir / "mfrc_layer_auroc_heatmap.png",
        vmin=0.5,
        vmax=1.0,
    )

    cosine_rows: list[dict[str, object]] = []
    cosine_matrix = np.zeros((len(foundations), len(foundations)), dtype=np.float64)
    auroc_at_focal_rows: list[dict[str, object]] = []
    auroc_at_focal_matrix = np.zeros((len(foundations), len(foundations)), dtype=np.float64)

    for i, focal in enumerate(foundations):
        focal_best_layer = next(
            row["best_layer"] for row in best_rows if row["foundation"] == focal
        )
        focal_best_layer = int(focal_best_layer)
        focal_direction = payloads[focal]["directions"][focal_best_layer]
        cosine_row = {"focal_foundation": focal, "focal_best_layer": focal_best_layer}
        auroc_row = {"focal_foundation": focal, "focal_best_layer": focal_best_layer}
        for j, other in enumerate(foundations):
            other_direction = payloads[other]["directions"][focal_best_layer]
            cosine = float(np.dot(focal_direction, other_direction))
            cosine_matrix[i, j] = cosine
            cosine_row[other] = cosine

            layer_row = next(
                row for row in payloads[other]["layer_rows"] if int(row["layer"]) == focal_best_layer
            )
            eval_auroc = float(layer_row["eval_auroc"])
            auroc_at_focal_matrix[i, j] = eval_auroc
            auroc_row[other] = eval_auroc

        cosine_rows.append(cosine_row)
        auroc_at_focal_rows.append(auroc_row)

    _write_csv(
        output_dir / "mfrc_same_layer_cosines_at_best_layers.csv",
        cosine_rows,
        fieldnames=["focal_foundation", "focal_best_layer", *foundations],
    )
    _write_csv(
        output_dir / "mfrc_eval_auroc_at_focal_best_layers.csv",
        auroc_at_focal_rows,
        fieldnames=["focal_foundation", "focal_best_layer", *foundations],
    )
    _plot_heatmap(
        matrix=cosine_matrix,
        row_labels=foundations,
        col_labels=foundations,
        title="Cosine Similarity at Each Foundation's Best Layer",
        colorbar_label="Cosine similarity",
        output_path=output_dir / "mfrc_same_layer_cosine_heatmap.png",
        vmin=-1.0,
        vmax=1.0,
    )
    _plot_heatmap(
        matrix=auroc_at_focal_matrix,
        row_labels=foundations,
        col_labels=foundations,
        title="Foundation Eval AUROC at Each Focal Best Layer",
        colorbar_label="Eval AUROC",
        output_path=output_dir / "mfrc_eval_auroc_at_focal_best_layers_heatmap.png",
        vmin=0.5,
        vmax=1.0,
    )


if __name__ == "__main__":
    main()
