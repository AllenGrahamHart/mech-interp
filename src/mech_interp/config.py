from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    model_id: str
    hf_token_env: str = "HF_TOKEN"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device_map: str = "auto"
    trust_remote_code: bool = False


@dataclass
class DatasetConfig:
    dataset_id: str = "hendrycks/ethics"
    subset: str = "commonsense"
    train_split: str = "train"
    eval_split: str = "test"
    text_field: str = "input"
    label_field: str = "label"
    require_short: bool = True
    require_edited: bool | None = None
    limit_train_per_class: int | None = 256
    limit_eval_per_class: int | None = 128
    shuffle_seed: int = 42


@dataclass
class PromptConfig:
    representation_template: str = "Scenario: {text}\nMoral evaluation:"
    judgment_template: str = (
        "Scenario: {text}\nIs this morally acceptable? Answer with one word:"
    )
    positive_completion: str = " unacceptable"
    negative_completion: str = " acceptable"


@dataclass
class ActivationConfig:
    layer: int = 12
    site: str = "resid_post"
    pooling: str = "last_non_pad"
    batch_size: int = 16
    max_length: int = 128


@dataclass
class StabilityConfig:
    enabled: bool = True
    repeats: int = 8
    sample_sizes: list[int] = field(default_factory=list)
    random_seed: int = 42


@dataclass
class InterventionConfig:
    enabled: bool = True
    judgment_template: str = "Scenario: {text}\nIs this morally wrong? Answer:"
    positive_token: str = " yes"
    negative_token: str = " no"
    add_scales: list[float] = field(default_factory=lambda: [-3.0, -1.0, 1.0, 3.0])
    remove_component: bool = True
    max_examples_per_class: int | None = 16


@dataclass
class DirectionConfig:
    method: str = "mean_diff"
    positive_label: int = 1
    positive_name: str = "immoral"
    negative_name: str = "acceptable"
    normalize: bool = True


@dataclass
class SAEConfig:
    enabled: bool = True
    release: str | None = None
    sae_id: str | None = None
    batch_size: int = 64
    top_k: int = 30


@dataclass
class ExperimentConfig:
    experiment_name: str = "morality_direction"
    output_root: str = "runs"
    model: ModelConfig = field(default_factory=lambda: ModelConfig(model_id="google/gemma-3-1b-it"))
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    activations: ActivationConfig = field(default_factory=ActivationConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _coerce_section(section_cls: type[Any], raw: dict[str, Any] | None) -> Any:
    return section_cls(**(raw or {}))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "morality_direction"),
        output_root=raw.get("output_root", "runs"),
        model=_coerce_section(ModelConfig, raw.get("model")),
        dataset=_coerce_section(DatasetConfig, raw.get("dataset")),
        prompts=_coerce_section(PromptConfig, raw.get("prompts")),
        activations=_coerce_section(ActivationConfig, raw.get("activations")),
        stability=_coerce_section(StabilityConfig, raw.get("stability")),
        intervention=_coerce_section(InterventionConfig, raw.get("intervention")),
        direction=_coerce_section(DirectionConfig, raw.get("direction")),
        sae=_coerce_section(SAEConfig, raw.get("sae")),
    )
