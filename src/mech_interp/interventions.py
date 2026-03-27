from __future__ import annotations

from contextlib import contextmanager


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
):
    import torch

    layers = _get_transformer_layers(model)
    layer = layers[layer_index]
    direction = torch.as_tensor(direction)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        direction_device = direction.to(device=modified.device, dtype=modified.dtype)
        delta = direction_device.view(1, 1, -1)

        if remove_component:
            coeff = (modified[:, -1, :] * direction_device).sum(dim=-1, keepdim=True)
            modified[:, -1, :] = modified[:, -1, :] - coeff * direction_device

        if scale != 0.0:
            modified[:, -1, :] = modified[:, -1, :] + scale * delta.squeeze(1)

        if isinstance(output, tuple):
            return (modified,) + tuple(output[1:])
        return modified

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

