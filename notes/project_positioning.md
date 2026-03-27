# Project Positioning: Consensus Morality in Raw Residual Streams

Date: 2026-03-27

## Working Claim

This project is not about discovering whether morality exists anywhere inside language models.

That broad claim is already covered by prior work on:

- moral direction in embedding space,
- linear hidden-state representations of related safety concepts,
- and recent papers on internal moral or ethical subspaces.

Instead, the project asks a narrower and cleaner question:

> Can a simple layerwise linear direction, fit from high-consensus ETHICS commonsense scenarios using final-token raw residual activations, provide a stable and causally meaningful representation of moral acceptability in a small open language model?

## Why This Is Still Worth Doing

The contribution is methodological clarity rather than maximal novelty.

The design is intentionally simple:

- small open model,
- raw activations rather than SAE features,
- one fixed prompt template,
- one fixed readout position,
- one simple linear estimator,
- explicit sample-size stability,
- explicit causal intervention.

This makes the result easier to interpret than richer but more confounded probing pipelines.

## What Counts as Success

The main result is not "high classification accuracy."

The main result is a chain of evidence:

1. A layerwise mean-difference direction separates acceptable from immoral ETHICS commonsense scenarios on held-out data.
2. The estimated direction stabilizes as the number of examples increases.
3. The strongest signal appears in a coherent band of middle or later layers.
4. Intervening on that direction changes the model's moral-judgment behavior or judgment-related logits.
5. The effect is not reducible to simple sentiment, toxicity, or prompt-template artifacts.

If all five hold, we have evidence for a genuine candidate morality feature in the residual stream.

## What We Should Avoid Claiming

We should avoid saying:

- "we discovered morality in LLMs,"
- "we found the one true moral direction,"
- "the model has human morality,"
- or "the model's internal morality is fully captured by a single axis."

The strongest defensible claim is narrower:

- the model contains a stable linear representation correlated with consensus commonsense moral acceptability,
- and this representation has some causal significance for downstream moral judgments.

## Minimal Experimental Scope

### Phase 1: Descriptive

- Fit a layerwise morality direction from ETHICS commonsense short scenarios.
- Evaluate held-out separation.
- Quantify stability under increasing sample size.

### Phase 2: Causal

- Remove the direction at the best layer and measure degradation in moral judgment.
- Add the direction and measure the shift in judgment-related logits or outputs.

### Phase 3: Controls

- Compare against sentiment.
- Compare against toxicity.
- Check prompt-template robustness.
- Check whether obvious lexical shortcuts explain the effect.

## Extensions

Natural extensions, if the basic result is solid:

- compare pretrained versus instruct-tuned models,
- compare Qwen and Gemma,
- compare multiple languages,
- test whether the learned morality direction is language-dependent by translating matched scenarios across languages,
- project contentious or controversial items onto the learned direction,
- compare latent morality with explicit verbal judgments.

These are secondary. The core paper should stand without them.

## Current Best Framing for Writing

Short version:

> We study whether consensus commonsense moral acceptability is encoded as a simple, stable, and causally meaningful linear feature in the raw residual stream of a small open LLM.

Longer version:

> Using high-consensus ETHICS commonsense scenarios, we fit layerwise linear morality directions from final-token residual activations, test their stability as the number of examples increases, and evaluate whether interventions on these directions causally influence the model's moral judgments. Our goal is not to maximize predictive performance, but to identify whether a simple and interpretable moral representation exists in the model's raw internal states.
