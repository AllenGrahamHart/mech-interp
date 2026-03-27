from __future__ import annotations

import numpy as np

from mech_interp.config import SAEConfig


def _resolve_feature_directions(decoder, direction_dim: int):
    if decoder.ndim != 2:
        raise ValueError("Expected a rank-2 decoder matrix.")
    if decoder.shape[1] == direction_dim:
        return decoder
    if decoder.shape[0] == direction_dim:
        return decoder.T
    raise ValueError(
        f"Decoder shape {tuple(decoder.shape)} is incompatible with direction dimension {direction_dim}."
    )


def analyze_sae_features(
    activations: np.ndarray,
    labels: np.ndarray,
    direction: np.ndarray,
    config: SAEConfig,
    positive_label: int = 1,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    import torch
    from sae_lens import SAE

    if not config.release or not config.sae_id:
        raise ValueError("SAE analysis is enabled but release / sae_id is missing.")

    sae, _, _ = SAE.from_pretrained(release=config.release, sae_id=config.sae_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device)
    sae.eval()

    activation_tensor = torch.tensor(activations, dtype=torch.float32, device=device)
    feature_batches = []
    for start in range(0, activation_tensor.shape[0], config.batch_size):
        batch = activation_tensor[start : start + config.batch_size]
        with torch.no_grad():
            feature_batches.append(sae.encode(batch).detach().float().cpu())
    features = torch.cat(feature_batches, dim=0).numpy()

    positive_mask = labels == positive_label
    negative_mask = ~positive_mask

    positive_mean = features[positive_mask].mean(axis=0)
    negative_mean = features[negative_mask].mean(axis=0)
    activation_diff = positive_mean - negative_mean

    decoder = getattr(sae, "W_dec", None)
    if decoder is None:
        raise AttributeError("Loaded SAE has no W_dec attribute.")
    decoder = decoder.detach().float().cpu()
    feature_directions = _resolve_feature_directions(decoder, direction_dim=direction.shape[0]).numpy()
    alignment = feature_directions @ direction

    metrics = {
        "mean_active_features": float((features > 0).sum(axis=1).mean()),
        "max_feature_activation": float(features.max()),
    }

    feature_rows = []
    top_indices = np.argsort(np.abs(activation_diff))[::-1][: config.top_k]
    for index in top_indices:
        feature_rows.append(
            {
                "feature_index": int(index),
                "activation_diff": float(activation_diff[index]),
                "positive_mean_activation": float(positive_mean[index]),
                "negative_mean_activation": float(negative_mean[index]),
                "direction_alignment": float(alignment[index]),
            }
        )

    return feature_rows, metrics
