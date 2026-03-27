from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mech_interp.config import ActivationConfig, ModelConfig
from mech_interp.utils import batched


@dataclass
class ModelBundle:
    tokenizer: object
    model: object


def _torch_dtype_from_name(name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype '{name}'.") from exc


def load_model_bundle(config: ModelConfig) -> ModelBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get(config.hf_token_env)
    if token is not None and not token.strip():
        token = None
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        token=token,
        torch_dtype=_torch_dtype_from_name(config.dtype),
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
    )
    model.eval()
    return ModelBundle(tokenizer=tokenizer, model=model)


def format_prompts(texts: Sequence[str], template: str) -> list[str]:
    return [template.format(text=text) for text in texts]


def _select_hidden_state(hidden_states: tuple, layer: int, site: str):
    if site != "resid_post":
        raise NotImplementedError(
            "Only resid_post is implemented in the baseline extractor. "
            "Other hook sites should use explicit forward hooks."
        )

    hidden_state_index = layer + 1
    if hidden_state_index >= len(hidden_states):
        raise IndexError(
            f"Requested layer {layer}, but hidden_states has only {len(hidden_states) - 1} transformer layers."
        )
    return hidden_states[hidden_state_index]


def _pool_hidden_states(hidden_states, attention_mask, pooling: str):
    import torch

    if pooling == "last_non_pad":
        positions = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, positions]

    if pooling == "mean_non_pad":
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    raise ValueError(f"Unsupported pooling mode '{pooling}'.")


def collect_pooled_activations(
    texts: Sequence[str],
    prompt_template: str,
    model_bundle: ModelBundle,
    activation_config: ActivationConfig,
) -> tuple[np.ndarray, list[str]]:
    import torch

    prompts = format_prompts(texts, prompt_template)
    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    device = next(model.parameters()).device

    pooled_batches = []
    for prompt_batch in batched(prompts, activation_config.batch_size):
        encoded = tokenizer(
            list(prompt_batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=activation_config.max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden_state = _select_hidden_state(
                outputs.hidden_states,
                layer=activation_config.layer,
                site=activation_config.site,
            )
            pooled = _pool_hidden_states(
                hidden_state,
                attention_mask=encoded["attention_mask"],
                pooling=activation_config.pooling,
            )
            pooled_batches.append(pooled.detach().float().cpu())

    activations = torch.cat(pooled_batches, dim=0).numpy()
    return activations, prompts


def collect_layerwise_pooled_activations(
    texts: Sequence[str],
    prompt_template: str,
    model_bundle: ModelBundle,
    activation_config: ActivationConfig,
    layers: Sequence[int] | None = None,
) -> tuple[dict[int, np.ndarray], list[str]]:
    import torch

    prompts = format_prompts(texts, prompt_template)
    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    device = next(model.parameters()).device

    layer_batches: dict[int, list] = {}
    resolved_layers: list[int] | None = list(layers) if layers is not None else None

    for prompt_batch in batched(prompts, activation_config.batch_size):
        encoded = tokenizer(
            list(prompt_batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=activation_config.max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states

            if resolved_layers is None:
                resolved_layers = list(range(len(hidden_states) - 1))

            for layer in resolved_layers:
                hidden_state = _select_hidden_state(
                    hidden_states,
                    layer=layer,
                    site=activation_config.site,
                )
                pooled = _pool_hidden_states(
                    hidden_state,
                    attention_mask=encoded["attention_mask"],
                    pooling=activation_config.pooling,
                )
                layer_batches.setdefault(layer, []).append(pooled.detach().float().cpu())

    if resolved_layers is None:
        raise RuntimeError("Could not resolve layer indices from model outputs.")

    layerwise = {
        layer: torch.cat(layer_batches[layer], dim=0).numpy()
        for layer in resolved_layers
    }
    return layerwise, prompts
