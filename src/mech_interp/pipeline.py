from __future__ import annotations

from pathlib import Path

import numpy as np

from mech_interp.config import ExperimentConfig, load_experiment_config
from mech_interp.data import load_ethics_split
from mech_interp.direction import DirectionFit, evaluate_direction, fit_direction, project_onto_direction
from mech_interp.judge import judge_statements
from mech_interp.modeling import collect_pooled_activations, load_model_bundle
from mech_interp.sae_analysis import analyze_sae_features
from mech_interp.utils import make_run_dir, read_nonempty_lines, to_serializable, write_csv, write_json


def _projection_rows(examples, prompts, scores):
    for example, prompt, score in zip(examples, prompts, scores, strict=True):
        yield {
            "split": example.split,
            "label": example.label,
            "text": example.text,
            "prompt": prompt,
            "projection": float(score),
        }


def _plot_projection_histogram(train_scores, train_labels, eval_scores, eval_labels, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, scores, labels, title in [
        (axes[0], train_scores, train_labels, "Train"),
        (axes[1], eval_scores, eval_labels, "Eval"),
    ]:
        axis.hist(scores[labels == 0], bins=30, alpha=0.7, label="acceptable")
        axis.hist(scores[labels == 1], bins=30, alpha=0.7, label="immoral")
        axis.set_title(title)
        axis.set_xlabel("Projection onto morality direction")
        axis.set_ylabel("Count")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _score_custom_statements(
    statements_path: str | Path,
    config: ExperimentConfig,
    model_bundle,
    direction_fit: DirectionFit,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    statements = read_nonempty_lines(statements_path)
    activations, prompts = collect_pooled_activations(
        texts=statements,
        prompt_template=config.prompts.representation_template,
        model_bundle=model_bundle,
        activation_config=config.activations,
    )
    projections = project_onto_direction(activations, direction_fit.direction)
    judgment_rows = judge_statements(
        statements=statements,
        prompt_template=config.prompts.judgment_template,
        positive_completion=config.prompts.positive_completion,
        negative_completion=config.prompts.negative_completion,
        model_bundle=model_bundle,
        max_length=config.activations.max_length,
    )

    merged_rows = []
    logprob_gaps = []
    for statement, prompt, projection, judgment_row in zip(
        statements,
        prompts,
        projections,
        judgment_rows,
        strict=True,
    ):
        row = {
            "statement": statement,
            "prompt": prompt,
            "projection": float(projection),
            "projection_label": (
                config.direction.positive_name
                if projection >= direction_fit.threshold
                else config.direction.negative_name
            ),
        }
        row.update(judgment_row)
        merged_rows.append(row)
        logprob_gaps.append(judgment_row["logprob_gap"])

    metrics = {}
    if len(merged_rows) >= 2:
        metrics["projection_logprob_gap_corr"] = float(
            np.corrcoef(projections, np.asarray(logprob_gaps, dtype=np.float32))[0, 1]
        )
    else:
        metrics["projection_logprob_gap_corr"] = float("nan")
    return merged_rows, metrics


def run_morality_direction(
    config_path: str | Path,
    statements_path: str | Path | None = None,
) -> dict[str, object]:
    config = load_experiment_config(config_path)
    run_dir = make_run_dir(config.output_root, config.experiment_name)

    write_json(run_dir / "config.json", config.to_dict())

    train_examples = load_ethics_split(
        config=config.dataset,
        split=config.dataset.train_split,
        limit_per_class=config.dataset.limit_train_per_class,
    )
    eval_examples = load_ethics_split(
        config=config.dataset,
        split=config.dataset.eval_split,
        limit_per_class=config.dataset.limit_eval_per_class,
    )

    model_bundle = load_model_bundle(config.model)

    train_activations, train_prompts = collect_pooled_activations(
        texts=[example.text for example in train_examples],
        prompt_template=config.prompts.representation_template,
        model_bundle=model_bundle,
        activation_config=config.activations,
    )
    eval_activations, eval_prompts = collect_pooled_activations(
        texts=[example.text for example in eval_examples],
        prompt_template=config.prompts.representation_template,
        model_bundle=model_bundle,
        activation_config=config.activations,
    )

    train_labels = np.asarray([example.label for example in train_examples], dtype=np.int64)
    eval_labels = np.asarray([example.label for example in eval_examples], dtype=np.int64)

    direction_fit = fit_direction(
        activations=train_activations,
        labels=train_labels,
        config=config.direction,
    )

    train_scores = project_onto_direction(train_activations, direction_fit.direction)
    eval_scores = project_onto_direction(eval_activations, direction_fit.direction)
    eval_metrics = evaluate_direction(
        activations=eval_activations,
        labels=eval_labels,
        direction=direction_fit.direction,
        threshold=direction_fit.threshold,
    )

    np.save(run_dir / "direction.npy", direction_fit.direction.astype(np.float32))
    np.save(run_dir / "train_activations.npy", train_activations.astype(np.float32))
    np.save(run_dir / "eval_activations.npy", eval_activations.astype(np.float32))

    write_csv(run_dir / "train_projections.csv", _projection_rows(train_examples, train_prompts, train_scores))
    write_csv(run_dir / "eval_projections.csv", _projection_rows(eval_examples, eval_prompts, eval_scores))

    _plot_projection_histogram(
        train_scores=train_scores,
        train_labels=train_labels,
        eval_scores=eval_scores,
        eval_labels=eval_labels,
        output_path=run_dir / "projection_histogram.png",
    )

    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "direction_threshold": direction_fit.threshold,
        "train_metrics": direction_fit.train_metrics,
        "eval_metrics": eval_metrics,
        "layer": config.activations.layer,
        "site": config.activations.site,
        "positive_name": config.direction.positive_name,
        "negative_name": config.direction.negative_name,
    }

    if config.sae.enabled:
        feature_rows, sae_metrics = analyze_sae_features(
            activations=train_activations,
            labels=train_labels,
            direction=direction_fit.direction,
            config=config.sae,
            positive_label=config.direction.positive_label,
        )
        write_csv(run_dir / "sae_feature_summary.csv", feature_rows)
        summary["sae"] = {
            "release": config.sae.release,
            "sae_id": config.sae.sae_id,
            "metrics": sae_metrics,
            "top_features_file": "sae_feature_summary.csv",
        }

    if statements_path is not None:
        custom_rows, custom_metrics = _score_custom_statements(
            statements_path=statements_path,
            config=config,
            model_bundle=model_bundle,
            direction_fit=direction_fit,
        )
        write_csv(run_dir / "custom_statement_scores.csv", custom_rows)
        summary["custom_statements"] = {
            "count": len(custom_rows),
            "metrics": custom_metrics,
            "file": "custom_statement_scores.csv",
        }

    write_json(run_dir / "summary.json", to_serializable(summary))
    return to_serializable(summary)
