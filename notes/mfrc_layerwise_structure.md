# MFRC layerwise structure

We analyzed the six MFRC foundation directions in two complementary ways:

1. For each foundation, find the layer with the highest held-out AUROC.
2. At each foundation's own best layer, compare its direction to the other foundation directions extracted in that same layer.

The corresponding artifacts are:

- `tmp/modal_results/mfrc_layerwise_structure/mfrc_best_layer_summary.csv`
- `tmp/modal_results/mfrc_layerwise_structure/mfrc_layer_auroc_long.csv`
- `tmp/modal_results/mfrc_layerwise_structure/mfrc_layer_auroc_heatmap.png`
- `tmp/modal_results/mfrc_layerwise_structure/mfrc_same_layer_cosines_at_best_layers.csv`
- `tmp/modal_results/mfrc_layerwise_structure/mfrc_same_layer_cosine_heatmap.png`
- `tmp/modal_results/mfrc_layerwise_structure/mfrc_eval_auroc_at_focal_best_layers.csv`

## Best-decoding layers

Best held-out AUROC layers:

- `Care`: layer `4`, AUROC `0.871`
- `Loyalty`: layer `6`, AUROC `0.784`
- `Proportionality`: layer `6`, AUROC `0.779`
- `Authority`: layer `19`, AUROC `0.777`
- `Equality`: layer `22`, AUROC `0.886`
- `Purity`: layer `22`, AUROC `0.800`

This layered pattern is itself evidence against a single collapsed moral-foundation representation.
Different foundations are most linearly recoverable at different depths.

## Same-layer comparison at each foundation's best layer

For each focal foundation `f`, we took its best layer `l_f` and compared the direction for `f` to all other foundation directions computed in that same layer.

Two observations stand out.

### 1. Some pairs are genuinely close

- `Loyalty` vs `Proportionality` at layer `6`: cosine `0.855`
- `Equality` vs `Purity` at layer `22`: cosine `0.699`

These are the pairs we followed up with bootstrap analysis.

### 2. The whole family is more aligned than expected

At several focal best layers, many other foundations also decode strongly and their directions are highly aligned.

Examples:

- at `Care`'s best layer `4`
  - `Proportionality` cosine: `0.922`
  - `Purity` cosine: `0.916`
  - `Loyalty` cosine: `0.872`

- at `Authority`'s best layer `19`
  - `Equality` cosine: `0.868`
  - `Loyalty` cosine: `0.776`
  - `Proportionality` cosine: `0.706`

This means the current one-vs-`Non-Moral` setup is not extracting six nearly orthogonal foundation axes.

## Interpretation

The strongest defensible interpretation is:

- foundation-relevant directions are distributed across layers
- different foundations peak at different depths
- but a large shared component is present across many foundations

This shared component is likely at least partly induced by the task design:

- every foundation is contrasted against the same `strict_nonmoral` baseline
- so the learned directions can all pick up a generic `moralized/foundation-laden vs non-moral` axis

That explains why:

- the best layers differ across foundations
- yet same-layer cosine similarities can still be very high

## Consequence for next experiments

The MFRC result should not be framed as evidence for six independent moral-foundation directions.
Instead it supports a layered and partially overlapping geometry:

- some foundations have different preferred layers
- some pairs are tightly aligned
- and there is probably a broad shared moral-foundation subspace on top of the pair-specific structure

If we want sharper disentanglement later, the next methodological move would be to compare foundations directly against each other, or to explicitly factor out the shared `foundation vs non-moral` component before measuring pairwise similarity.
