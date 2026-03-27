from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence


def _get_transformer_layers(model):
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(getattr(model, "language_model", None), "model", None), "layers", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise AttributeError("Could not find a transformer block list on the loaded model.")


@contextmanager
def residual_direction_intervention(
    model,
    layer_index: int,
    direction,
    scale: float = 0.0,
    remove_component: bool = False,
    token_index: int | Sequence[int] = -1,
):
    import torch

    layers = _get_transformer_layers(model)
    layer = layers[layer_index]
    direction = torch.as_tensor(direction)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        direction_device = direction.to(device=modified.device, dtype=modified.dtype)
        if isinstance(token_index, int):
            target_index = token_index if token_index >= 0 else modified.shape[1] + token_index
            selected = modified[:, target_index, :]
            batch_indices = None
        else:
            target_indices = torch.as_tensor(token_index, device=modified.device, dtype=torch.long)
            target_indices = torch.where(target_indices >= 0, target_indices, modified.shape[1] + target_indices)
            if target_indices.shape[0] != modified.shape[0]:
                raise ValueError(
                    "Per-example token indices must match the batch dimension of the hooked activation."
                )
            batch_indices = torch.arange(modified.shape[0], device=modified.device)
            selected = modified[batch_indices, target_indices, :]

        if remove_component:
            coeff = (selected * direction_device).sum(dim=-1, keepdim=True)
            selected = selected - coeff * direction_device

        if scale != 0.0:
            selected = selected + scale * direction_device

        if batch_indices is None:
            modified[:, target_index, :] = selected
        else:
            modified[batch_indices, target_indices, :] = selected

        if isinstance(output, tuple):
            return (modified,) + tuple(output[1:])
        return modified

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
