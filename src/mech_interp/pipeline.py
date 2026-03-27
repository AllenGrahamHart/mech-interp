from __future__ import annotations

from pathlib import Path

import numpy as np

from mech_interp.config import ExperimentConfig, load_experiment_config
from mech_interp.data import load_ethics_split
from mech_interp.direction import (
    DirectionFit,
    cosine_similarity,
    evaluate_direction,
    fit_direction,
    project_onto_direction,
)
from mech_interp.judge import batched_completion_logprob_gap, completion_logprob_gap, judge_statements
from mech_interp.interventions import residual_direction_intervention
from mech_interp.judge import next_token_logit_gap
from mech_interp.modeling import (
    collect_layerwise_pooled_activations,
    collect_pooled_activations,
    load_model_bundle,
)
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


def _default_sample_sizes(max_per_class: int) -> list[int]:
    sizes = []
    candidate = 4
    while candidate <= max_per_class:
        sizes.append(candidate)
        candidate *= 2
    if max_per_class >= 2 and max_per_class not in sizes:
        sizes.append(max_per_class)
    return sizes


def _stability_rows(
    activations: np.ndarray,
    labels: np.ndarray,
    direction_fit: DirectionFit,
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    positive_label = config.direction.positive_label
    positive_indices = np.where(labels == positive_label)[0]
    negative_indices = np.where(labels != positive_label)[0]
    max_per_class = min(len(positive_indices), len(negative_indices))
    sample_sizes = config.stability.sample_sizes or _default_sample_sizes(max_per_class)

    rng = np.random.default_rng(config.stability.random_seed)
    rows = []
    for sample_size in sample_sizes:
        if sample_size > max_per_class:
            continue
        for repeat in range(config.stability.repeats):
            sampled_positive = rng.choice(positive_indices, size=sample_size, replace=False)
            sampled_negative = rng.choice(negative_indices, size=sample_size, replace=False)
            sampled_indices = np.concatenate([sampled_positive, sampled_negative])
            sampled_fit = fit_direction(
                activations=activations[sampled_indices],
                labels=labels[sampled_indices],
                config=config.direction,
            )
            rows.append(
                {
                    "sample_size_per_class": int(sample_size),
                    "repeat": int(repeat),
                    "cosine_to_full_direction": cosine_similarity(
                        sampled_fit.direction,
                        direction_fit.direction,
                    ),
                    "threshold": float(sampled_fit.threshold),
                    "train_accuracy": float(sampled_fit.train_metrics["accuracy"]),
                    "train_balanced_accuracy": float(sampled_fit.train_metrics["balanced_accuracy"]),
                    "train_auroc": float(sampled_fit.train_metrics["auroc"]),
                }
            )
    return rows


def run_layer_sweep(
    config_path: str | Path,
) -> dict[str, object]:
    config = load_experiment_config(config_path)
    run_dir = make_run_dir(config.output_root, f"{config.experiment_name}_layers")
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
    train_layerwise, train_prompts = collect_layerwise_pooled_activations(
        texts=[example.text for example in train_examples],
        prompt_template=config.prompts.representation_template,
        model_bundle=model_bundle,
        activation_config=config.activations,
    )
    eval_layerwise, eval_prompts = collect_layerwise_pooled_activations(
        texts=[example.text for example in eval_examples],
        prompt_template=config.prompts.representation_template,
        model_bundle=model_bundle,
        activation_config=config.activations,
    )

    train_labels = np.asarray([example.label for example in train_examples], dtype=np.int64)
    eval_labels = np.asarray([example.label for example in eval_examples], dtype=np.int64)

    layer_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    best_layer = None
    best_eval_auroc = float("-inf")

    for layer in sorted(train_layerwise):
        train_activations = train_layerwise[layer]
        eval_activations = eval_layerwise[layer]
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

        np.save(run_dir / f"layer_{layer:02d}_direction.npy", direction_fit.direction.astype(np.float32))
        write_csv(
            run_dir / f"layer_{layer:02d}_train_projections.csv",
            _projection_rows(train_examples, train_prompts, train_scores),
        )
        write_csv(
            run_dir / f"layer_{layer:02d}_eval_projections.csv",
            _projection_rows(eval_examples, eval_prompts, eval_scores),
        )

        layer_row = {
            "layer": int(layer),
            "threshold": float(direction_fit.threshold),
            "train_accuracy": float(direction_fit.train_metrics["accuracy"]),
            "train_balanced_accuracy": float(direction_fit.train_metrics["balanced_accuracy"]),
            "train_auroc": float(direction_fit.train_metrics["auroc"]),
            "eval_accuracy": float(eval_metrics["accuracy"]),
            "eval_balanced_accuracy": float(eval_metrics["balanced_accuracy"]),
            "eval_auroc": float(eval_metrics["auroc"]),
        }
        layer_rows.append(layer_row)

        if config.stability.enabled:
            for row in _stability_rows(train_activations, train_labels, direction_fit, config):
                row["layer"] = int(layer)
                stability_rows.append(row)

        if eval_metrics["auroc"] > best_eval_auroc:
            best_eval_auroc = eval_metrics["auroc"]
            best_layer = int(layer)

    write_csv(run_dir / "layer_summary.csv", layer_rows)
    if stability_rows:
        write_csv(run_dir / "direction_stability.csv", stability_rows)

    summary = {
        "run_dir": str(run_dir),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "num_layers": len(layer_rows),
        "best_layer_by_eval_auroc": best_layer,
        "layer_summary_file": "layer_summary.csv",
        "stability_file": "direction_stability.csv" if stability_rows else None,
    }
    write_json(run_dir / "summary.json", to_serializable(summary))
    return to_serializable(summary)


def _limit_examples_per_class(examples, max_examples_per_class: int | None, positive_label: int):
    if max_examples_per_class is None:
        return list(examples)

    limited = []
    positive_count = 0
    negative_count = 0
    for example in examples:
        if example.label == positive_label and positive_count < max_examples_per_class:
            limited.append(example)
            positive_count += 1
        elif example.label != positive_label and negative_count < max_examples_per_class:
            limited.append(example)
            negative_count += 1
        if positive_count >= max_examples_per_class and negative_count >= max_examples_per_class:
            break
    return limited


def _intervention_metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["mode"]), float(row["scale"]))
        grouped.setdefault(key, []).append(row)

    metric_rows = []
    for (mode, scale), group_rows in sorted(grouped.items()):
        labels = np.asarray([int(row["label"]) for row in group_rows], dtype=np.int64)
        gaps = np.asarray([float(row["logprob_gap"]) for row in group_rows], dtype=np.float32)
        predictions = (gaps >= 0).astype(np.int64)
        metric_rows.append(
            {
                "mode": mode,
                "scale": scale,
                "count": int(len(group_rows)),
                "accuracy": float((predictions == labels).mean()),
                "mean_gap": float(gaps.mean()),
                "mean_gap_immoral": float(gaps[labels == 1].mean()) if np.any(labels == 1) else float("nan"),
                "mean_gap_acceptable": float(gaps[labels == 0].mean()) if np.any(labels == 0) else float("nan"),
                "mean_gap_delta_from_baseline": float(
                    np.mean([float(row["logprob_gap_delta"]) for row in group_rows])
                ),
            }
        )
    return metric_rows


def run_causal_benchmark(
    config_path: str | Path,
    layer: int | None = None,
) -> dict[str, object]:
    config = load_experiment_config(config_path)
    chosen_layer = config.activations.layer if layer is None else layer
    run_dir = make_run_dir(config.output_root, f"{config.experiment_name}_causal_layer_{chosen_layer:02d}")
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
    eval_examples = _limit_examples_per_class(
        eval_examples,
        max_examples_per_class=config.intervention.max_examples_per_class,
        positive_label=config.direction.positive_label,
    )

    model_bundle = load_model_bundle(config.model)
    tokenizer = model_bundle.tokenizer
    train_activations, _ = collect_pooled_activations(
        texts=[example.text for example in train_examples],
        prompt_template=config.prompts.representation_template,
        model_bundle=model_bundle,
        activation_config=config.activations,
    )
    train_labels = np.asarray([example.label for example in train_examples], dtype=np.int64)
    direction_fit = fit_direction(
        activations=train_activations,
        labels=train_labels,
        config=config.direction,
    )
    np.save(run_dir / "direction.npy", direction_fit.direction.astype(np.float32))

    prompts = [
        config.intervention.judgment_template.format(text=example.text)
        for example in eval_examples
    ]

    use_completion_scoring = (
        config.intervention.positive_completion is not None
        and config.intervention.negative_completion is not None
    )

    def score_prompt(prompt: str) -> dict[str, object]:
        if use_completion_scoring:
            return completion_logprob_gap(
                prompt=prompt,
                positive_completion=str(config.intervention.positive_completion),
                negative_completion=str(config.intervention.negative_completion),
                model_bundle=model_bundle,
                max_length=config.activations.max_length,
            )
        return next_token_logit_gap(
            prompt=prompt,
            positive_token=config.intervention.positive_token,
            negative_token=config.intervention.negative_token,
            model_bundle=model_bundle,
            max_length=config.activations.max_length,
        )

    result_rows: list[dict[str, object]] = []
    if use_completion_scoring:
        baseline_scores = batched_completion_logprob_gap(
            prompts=prompts,
            positive_completion=str(config.intervention.positive_completion),
            negative_completion=str(config.intervention.negative_completion),
            model_bundle=model_bundle,
            max_length=config.activations.max_length,
            batch_size=config.activations.batch_size,
        )

        for example, prompt, baseline in zip(eval_examples, prompts, baseline_scores, strict=True):
            result_rows.append(
                {
                    "layer": int(chosen_layer),
                    "mode": "baseline",
                    "scale": 0.0,
                    "label": int(example.label),
                    "text": example.text,
                    "prompt": prompt,
                    "logprob_gap": float(baseline["logprob_gap"]),
                    "preferred_completion": str(baseline["preferred_completion"]),
                    "logprob_gap_delta": 0.0,
                }
            )

        if config.intervention.remove_component:
            removed_scores = batched_completion_logprob_gap(
                prompts=prompts,
                positive_completion=str(config.intervention.positive_completion),
                negative_completion=str(config.intervention.negative_completion),
                model_bundle=model_bundle,
                max_length=config.activations.max_length,
                batch_size=config.activations.batch_size,
                intervention_factory=lambda token_indices: residual_direction_intervention(
                    model=model_bundle.model,
                    layer_index=chosen_layer,
                    direction=direction_fit.direction,
                    remove_component=True,
                    token_index=token_indices,
                ),
            )
            for example, prompt, baseline, removed in zip(
                eval_examples, prompts, baseline_scores, removed_scores, strict=True
            ):
                result_rows.append(
                    {
                        "layer": int(chosen_layer),
                        "mode": "remove",
                        "scale": 0.0,
                        "label": int(example.label),
                        "text": example.text,
                        "prompt": prompt,
                        "logprob_gap": float(removed["logprob_gap"]),
                        "preferred_completion": str(removed["preferred_completion"]),
                        "logprob_gap_delta": float(removed["logprob_gap"] - baseline["logprob_gap"]),
                    }
                )

        for scale in config.intervention.add_scales:
            steered_scores = batched_completion_logprob_gap(
                prompts=prompts,
                positive_completion=str(config.intervention.positive_completion),
                negative_completion=str(config.intervention.negative_completion),
                model_bundle=model_bundle,
                max_length=config.activations.max_length,
                batch_size=config.activations.batch_size,
                intervention_factory=lambda token_indices, scale=scale: residual_direction_intervention(
                    model=model_bundle.model,
                    layer_index=chosen_layer,
                    direction=direction_fit.direction,
                    scale=float(scale),
                    remove_component=False,
                    token_index=token_indices,
                ),
            )
            for example, prompt, baseline, steered in zip(
                eval_examples, prompts, baseline_scores, steered_scores, strict=True
            ):
                result_rows.append(
                    {
                        "layer": int(chosen_layer),
                        "mode": "add",
                        "scale": float(scale),
                        "label": int(example.label),
                        "text": example.text,
                        "prompt": prompt,
                        "logprob_gap": float(steered["logprob_gap"]),
                        "preferred_completion": str(steered["preferred_completion"]),
                        "logprob_gap_delta": float(steered["logprob_gap"] - baseline["logprob_gap"]),
                    }
                )
    else:
        for example, prompt in zip(eval_examples, prompts, strict=True):
            prompt_token_index = (
                tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=config.activations.max_length,
                )["input_ids"].shape[1]
                - 1
            )
            baseline = score_prompt(prompt)

            baseline_row = {
                "layer": int(chosen_layer),
                "mode": "baseline",
                "scale": 0.0,
                "label": int(example.label),
                "text": example.text,
                "prompt": prompt,
                "logprob_gap": float(baseline["logprob_gap"]),
                "preferred_completion": str(baseline["preferred_completion"]),
                "logprob_gap_delta": 0.0,
            }
            result_rows.append(baseline_row)

            if config.intervention.remove_component:
                with residual_direction_intervention(
                    model=model_bundle.model,
                    layer_index=chosen_layer,
                    direction=direction_fit.direction,
                    remove_component=True,
                    token_index=prompt_token_index,
                ):
                    removed = score_prompt(prompt)
                result_rows.append(
                    {
                        "layer": int(chosen_layer),
                        "mode": "remove",
                        "scale": 0.0,
                        "label": int(example.label),
                        "text": example.text,
                        "prompt": prompt,
                        "logprob_gap": float(removed["logprob_gap"]),
                        "preferred_completion": str(removed["preferred_completion"]),
                        "logprob_gap_delta": float(removed["logprob_gap"] - baseline["logprob_gap"]),
                    }
                )

            for scale in config.intervention.add_scales:
                with residual_direction_intervention(
                    model=model_bundle.model,
                    layer_index=chosen_layer,
                    direction=direction_fit.direction,
                    scale=float(scale),
                    remove_component=False,
                    token_index=prompt_token_index,
                ):
                    steered = score_prompt(prompt)
                result_rows.append(
                    {
                        "layer": int(chosen_layer),
                        "mode": "add",
                        "scale": float(scale),
                        "label": int(example.label),
                        "text": example.text,
                        "prompt": prompt,
                        "logprob_gap": float(steered["logprob_gap"]),
                        "preferred_completion": str(steered["preferred_completion"]),
                        "logprob_gap_delta": float(steered["logprob_gap"] - baseline["logprob_gap"]),
                    }
                )

    metric_rows = _intervention_metric_rows(result_rows)
    write_csv(run_dir / "intervention_results.csv", result_rows)
    write_csv(run_dir / "intervention_summary.csv", metric_rows)

    summary = {
        "run_dir": str(run_dir),
        "layer": int(chosen_layer),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "direction_threshold": float(direction_fit.threshold),
        "train_metrics": direction_fit.train_metrics,
        "intervention_results_file": "intervention_results.csv",
        "intervention_summary_file": "intervention_summary.csv",
    }
    write_json(run_dir / "summary.json", to_serializable(summary))
    return to_serializable(summary)
