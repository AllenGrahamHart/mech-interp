# Moral Foundations Extension

Date: 2026-03-28

## Motivation

The current project identifies a single linear direction associated with ETHICS commonsense wrongness in Qwen 0.5B. A natural extension is to ask whether this one-dimensional wrongness signal decomposes into more specific foundation-level directions.

Using Moral Foundations Theory (MFT), the concrete question becomes:

> Is the ETHICS wrongness direction partly spanned by distinct foundation-specific directions such as care, fairness, loyalty, authority, sanctity, and liberty?

This is a more interesting and more distinctive result than simply reporting that one linear morality-like feature exists.

## Prior Art

Two papers are especially relevant:

- Abdulhai et al. (EMNLP 2024), *Moral Foundations of Large Language Models*.
  This is a behavioral MFT paper. It studies whether LLMs exhibit different moral foundations and how prompting can shift those expressed values.

- Yu et al. (arXiv 2026), *Tracing Moral Foundations in Large Language Models*.
  This is the closest mechanistic prior art. It studies foundation-level moral structure in hidden states and reports that moral foundations are distributed, layered, and only partly disentangled.

So the broad claim that moral foundations can be probed in LLMs is not novel.

What still looks worthwhile here is a narrower and cleaner small-model study:

- raw residual activations rather than SAEs as the primary object,
- a small open model,
- simple mean-difference directions,
- explicit comparison to the existing ETHICS wrongness axis,
- and a decomposition question:
  does wrongness factor into a small set of foundation-level directions?

## Dataset Choice

The cleanest stimulus source is the Moral Foundations Vignettes (MFV) line, because the examples are:

- short,
- concrete,
- intentionally foundation-specific,
- and paired with non-moral social-norm controls.

### What is currently in the repo

The repository now contains a local MFV-derived dataset extracted from the original Clifford et al. paper PDF:

- `data/moral_foundations/mfv_full.csv`
- `data/moral_foundations/mfv_<foundation>_vs_social_norms.csv` for each of:
  - care
  - fairness
  - loyalty
  - authority
  - sanctity
  - liberty

Counts in the extracted full set:

- care: 32
- fairness: 17
- loyalty: 16
- authority: 17
- sanctity: 17
- liberty: 17
- social norms: 16

### Important caveat

The original paper discusses a recommended 90-item subset (Table 6) that satisfies both:

- the Study 1 classification criterion, and
- the Study 2 factor-loading criterion.

However, the accessible PDF text does not include the actual Table 6 contents, only the rule used to create it. For now, the local dataset uses the full labeled vignette set from Table 1 rather than the hidden 90-item subset.

This is acceptable for an exploratory extension, but if the foundation experiments become central to the paper, it would be worth sourcing the exact 90-item subset later.

## Immediate Experiment Plan

### 1. Fit one direction per foundation

For each of the six foundations, fit a direction from:

- positive class: the target foundation vignettes
- negative class: social-norm controls

This is cleaner than foundation-vs-all-other-foundations, because the contrast is against non-moral rather than differently moral stimuli.

### 2. Compare the six directions

Measure:

- pairwise cosine similarity,
- layerwise strength,
- and bootstrap angular stability.

Questions:

- Are some foundations much cleaner than others?
- Are some pairs strongly aligned, e.g. care and fairness?
- Is liberty weak or noisy in a 0.5B model?

### 3. Relate them back to ETHICS wrongness

Given the six foundation directions, test:

- cosine between each foundation direction and the ETHICS wrongness direction,
- and whether the ETHICS wrongness direction lies substantially in the span of the six foundation directions.

This would turn the extension into a real decomposition result rather than a disconnected side experiment.

### 4. Compare each foundation to sentiment

Repeat the sentiment control analysis foundation by foundation:

- Which foundations align most strongly with negative sentiment?
- Which foundations remain most distinct after orthogonalizing against sentiment?

I would expect care/harm to align more strongly with negative sentiment than, say, loyalty or authority.

### 5. Only then consider causal interventions

If one or more foundation directions are clean enough, run causal removal on foundation-specific prompts such as:

> Does this scenario violate the care/harm foundation?

Because the per-foundation datasets are small, causal claims here should be more modest than for ETHICS.

## Recommended Framing

If this extension works, the paper can make a stronger claim:

> The model contains a stable linear wrongness direction, and this direction is partly decomposable into foundation-specific residual directions corresponding to distinct domains of moral judgment.

That is a more interesting result than a single morality axis on its own.

## Suggested Next Step

The clean first pass is:

1. run the layer sweep for each of the six MFV foundation-vs-control datasets,
2. identify the best layer for each,
3. compute the pairwise cosine matrix,
4. and compare each foundation direction to both:
   - the ETHICS wrongness direction, and
   - the negative-sentiment direction.

Only after that should we decide whether the extension is strong enough to include in the main paper.
