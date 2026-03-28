# MFRC same-layer pairwise comparison

We compared the most overlapping foundation pairs at the same layer rather than using each foundation's own best layer:

- `Loyalty` vs `Proportionality` at layer `6`
- `Equality` vs `Purity` at layer `22`

This is a cleaner comparison than mixing layers, because both directions in each pair are extracted from the same residual-stream space at the same depth.

## Point estimates

Using the directions from the full layer sweeps:

- `Loyalty` vs `Proportionality` at layer `6`
  - cosine similarity: `0.8550`
  - angle: `31.24°`

- `Equality` vs `Purity` at layer `22`
  - cosine similarity: `0.6993`
  - angle: `45.63°`

So both pairs are meaningfully aligned, but the loyalty/proportionality pair is much closer.

## Bootstrap distinctness

We then ran single-layer extraction jobs that save train activations and bootstrapped each pair with `1000` stratified resamples.

### Loyalty vs Proportionality at layer 6

- observed cosine: `0.8550`
- bootstrap cosine 95% interval: `[0.7047, 0.8844]`
- observed angle: `31.24°`
- bootstrap angle 95% interval: `[27.82°, 45.20°]`

Direction stability within each dataset:

- loyalty cosine-to-full 95% interval: `[0.9390, 0.9809]`
- proportionality cosine-to-full 95% interval: `[0.9536, 0.9848]`

Interpretation:

- the two directions are clearly not identical, since the bootstrap cosine interval is far below `1`
- but they are strongly aligned and may reflect a partially shared latent subspace

### Equality vs Purity at layer 22

- observed cosine: `0.6993`
- bootstrap cosine 95% interval: `[0.5546, 0.7814]`
- observed angle: `45.63°`
- bootstrap angle 95% interval: `[38.61°, 56.31°]`

Direction stability within each dataset:

- equality cosine-to-full 95% interval: `[0.9716, 0.9914]`
- purity cosine-to-full 95% interval: `[0.9748, 0.9917]`

Interpretation:

- these directions are also clearly distinct
- they are aligned, but much less tightly than loyalty and proportionality

## Takeaway

The MFRC extension does not produce six mutually orthogonal foundation axes.
Instead, it produces a mixed picture:

- some foundations are quite distinct, such as `Care`
- some foundations appear to form closely aligned pairs, especially `Loyalty` and `Proportionality`
- others are moderately aligned, such as `Equality` and `Purity`

So the more defensible claim is not that Qwen contains six clean, independent moral-foundation axes, but that multiple foundation-relevant directions are linearly recoverable and some of them occupy overlapping subspaces.

## Artifacts

- broad foundation comparison:
  - `tmp/modal_results/mfrc_foundation_comparison/mfrc_foundation_summary.csv`
  - `tmp/modal_results/mfrc_foundation_comparison/mfrc_foundation_cosine_matrix.csv`
  - `tmp/modal_results/mfrc_foundation_comparison/mfrc_foundation_cosine_heatmap.png`
- bootstrap pairwise results:
  - `tmp/modal_results/foundation_pair_bootstrap/loyalty_vs_proportionality_layer6/summary.json`
  - `tmp/modal_results/foundation_pair_bootstrap/equality_vs_purity_layer22/summary.json`
