# MFRC same-layer tables

We built two aligned tables for the MFRC foundation analysis.

## Table 1

Angles between foundation directions at each focal foundation's best-decoding layer, with `1000`-replicate bootstrap confidence intervals.

Artifact:

- `tmp/modal_results/mfrc_same_layer_tables/angle_table_formatted.csv`
- `tmp/modal_results/mfrc_same_layer_tables/angle_table_numeric.csv`
- `tmp/modal_results/mfrc_same_layer_tables/angle_table.tex`

## Table 2

Held-out AUROC for each foundation measured at that same focal layer.

Artifact:

- `tmp/modal_results/mfrc_same_layer_tables/auroc_table_formatted.csv`
- `tmp/modal_results/mfrc_same_layer_tables/auroc_table_numeric.csv`
- `tmp/modal_results/mfrc_same_layer_tables/auroc_table.tex`

## Main interpretation

The tables make two points simultaneously:

1. Different foundations peak at different layers:
   - `Care`: layer `4`
   - `Loyalty` / `Proportionality`: layer `6`
   - `Authority`: layer `19`
   - `Equality` / `Purity`: layer `22`

2. Even within those focal layers, many foundations remain strongly decodable and often have fairly small angular separation.

This supports a layered but overlapping organization rather than six independent foundation axes.

Examples:

- At `Care (L4)`, `Proportionality` is only about `22.8°` away and still has AUROC `0.765`.
- At `Authority (L19)`, `Equality` is about `29.7°` away and has AUROC `0.805`.
- At `Equality (L22)`, `Purity` is about `45.6°` away and has AUROC `0.800`.

So the stronger claim is:

- foundation-relevant directions are distributed across depth
- some pairs are clearly distinct but still aligned
- and the one-vs-`Non-Moral` setup appears to induce a broad shared moralized subspace across foundations
