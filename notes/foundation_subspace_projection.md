# ETHICS Wrongness Projected onto the MFRC Foundation Span

This note asks a simple same-layer subspace question:

- Fit the ETHICS wrongness direction `w` from acceptable vs immoral ETHICS scenarios.
- Fit six MFRC foundation directions in the **same residual-stream layer**:
  `care`, `equality`, `proportionality`, `purity`, `authority`, `loyalty`.
- Form the six-dimensional foundation span in that layer.
- Project `w` onto that span and measure the explained fraction:

\[
\text{explained fraction} = \lVert \mathrm{proj}_{\mathcal{S}}(w) \rVert^2
\]

where `w` is unit norm and `\mathcal{S}` is the MFRC foundation subspace.

## Main Result

At the main ETHICS layer (`15`), the six-foundation MFRC span explains only about **19.1%** of the ETHICS wrongness direction.

- Focus layer: `15`
- Explained fraction: `0.1911`
- Angle from wrongness to the foundation span: `64.1 degrees`

So at the layer where ETHICS wrongness is strongest, most of the wrongness direction lies **outside** the same-layer MFRC foundation span.

## Layerwise Result

This overlap is not constant across depth. Using the same projection analysis for all layers:

- The best overlap occurs at layer `23`
- Explained fraction there: `0.4075`
- Angle from wrongness to the foundation span: `50.3 degrees`

So the foundation span captures more of wrongness in the late layers, but still not a majority of it.

## Interpretation

The clean reading is:

- ETHICS wrongness is **not** well described as just ``the sum of the six MFRC foundations'' in a single mid-layer subspace.
- Same-layer foundation directions become more relevant in the late layers, suggesting some convergence between foundation expression and overall wrongness as the model approaches the output.
- Even in the best late layer, more than half of the wrongness direction remains outside the six-foundation span.

This is consistent with the broader story so far:

- wrongness is partly shared with other moral concepts,
- but not reducible to them,
- and the relationship is strongly layer-dependent.

## Which Foundations Align Most?

At layer `15`, the strongest individual cosine overlaps with wrongness are:

- `authority`: `0.251`
- `equality`: `0.199`
- `purity`: `0.176`

At layer `23`, the strongest individual cosine overlaps are:

- `authority`: `0.410`
- `purity`: `0.348`
- `equality`: `0.347`
- `loyalty`: `0.316`

These are descriptive overlaps only. Because the foundation directions are themselves correlated, the least-squares decomposition coefficients should not be interpreted as unique causal contributions.

## Artifacts

- [summary.json](/home/u2470931/mech_interp/tmp/modal_results/foundation_subspace_projection/summary.json)
- [layerwise_projection_summary.csv](/home/u2470931/mech_interp/tmp/modal_results/foundation_subspace_projection/layerwise_projection_summary.csv)
- [focus_layer_detail.json](/home/u2470931/mech_interp/tmp/modal_results/foundation_subspace_projection/focus_layer_detail.json)
- [best_layer_detail.json](/home/u2470931/mech_interp/tmp/modal_results/foundation_subspace_projection/best_layer_detail.json)

The analysis script is:

- [analyze_wrongness_foundation_subspace.py](/home/u2470931/mech_interp/scripts/analyze_wrongness_foundation_subspace.py)
