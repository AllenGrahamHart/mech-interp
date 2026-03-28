from __future__ import annotations

import csv
import json
from pathlib import Path

import tomllib


def _toml_str(value: str) -> str:
    return json.dumps(value)


def _render_config(raw: dict, experiment_name: str, layer: int) -> str:
    dataset = raw["dataset"]
    prompts = raw["prompts"]
    model = raw["model"]
    direction = raw["direction"]
    activations = raw["activations"]

    lines = [
        f"experiment_name = {_toml_str(experiment_name)}",
        'output_root = "runs"',
        "",
        "[model]",
        f'model_id = {_toml_str(model["model_id"])}',
        f'hf_token_env = {_toml_str(model["hf_token_env"])}',
        f'dtype = {_toml_str(model["dtype"])}',
        f'attn_implementation = {_toml_str(model["attn_implementation"])}',
        f'device_map = {_toml_str(model["device_map"])}',
        f'trust_remote_code = {"true" if model["trust_remote_code"] else "false"}',
        "",
        "[dataset]",
        f'dataset_id = {_toml_str(dataset["dataset_id"])}',
        f'local_path = {_toml_str(dataset["local_path"])}',
        f'train_split = {_toml_str(dataset["train_split"])}',
        f'eval_split = {_toml_str(dataset["eval_split"])}',
        f'split_field = {_toml_str(dataset["split_field"])}',
        f'text_field = {_toml_str(dataset["text_field"])}',
        f'label_field = {_toml_str(dataset["label_field"])}',
        f'require_short = {"true" if dataset["require_short"] else "false"}',
        f'limit_train_per_class = {dataset["limit_train_per_class"]}',
        f'limit_eval_per_class = {dataset["limit_eval_per_class"]}',
        f'shuffle_seed = {dataset["shuffle_seed"]}',
        "",
        "[prompts]",
        f'representation_template = {_toml_str(prompts["representation_template"])}',
        f'judgment_template = {_toml_str(prompts["judgment_template"])}',
        f'positive_completion = {_toml_str(prompts["positive_completion"])}',
        f'negative_completion = {_toml_str(prompts["negative_completion"])}',
        "",
        "[activations]",
        f'layer = {layer}',
        f'site = {_toml_str(activations["site"])}',
        f'pooling = {_toml_str(activations["pooling"])}',
        f'batch_size = {activations["batch_size"]}',
        f'max_length = {activations["max_length"]}',
        "",
        "[stability]",
        "enabled = false",
        "repeats = 0",
        "sample_sizes = []",
        "random_seed = 42",
        "",
        "[intervention]",
        "enabled = false",
        f'judgment_template = {_toml_str(prompts["judgment_template"])}',
        f'positive_completion = {_toml_str(prompts["positive_completion"])}',
        f'negative_completion = {_toml_str(prompts["negative_completion"])}',
        "add_scales = [-3.0, -1.0, 1.0, 3.0]",
        "remove_component = true",
        "max_examples_per_class = 32",
        "",
        "[direction]",
        f'method = {_toml_str(direction["method"])}',
        f'positive_label = {direction["positive_label"]}',
        f'positive_name = {_toml_str(direction["positive_name"])}',
        f'negative_name = {_toml_str(direction["negative_name"])}',
        f'normalize = {"true" if direction["normalize"] else "false"}',
        "",
        "[sae]",
        "enabled = false",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    summary_path = Path("tmp/modal_results/mfrc_layerwise_structure/mfrc_best_layer_summary.csv")
    output_dir = Path("configs/mfrc_same_layer")
    output_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    unique_layers = sorted({int(row["best_layer"]) for row in summary_rows})

    for config_path in sorted(Path("configs").glob("mfrc_*_vs_strict_nonmoral.toml")):
        if "layer" in config_path.stem:
            continue
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        foundation_slug = config_path.stem.replace("_vs_strict_nonmoral", "")
        for layer in unique_layers:
            experiment_name = f"{foundation_slug}_same_layer_{layer:02d}"
            rendered = _render_config(raw=raw, experiment_name=experiment_name, layer=layer)
            target = output_dir / f"{experiment_name}.toml"
            target.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
