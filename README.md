# Mechanistic Interpretability for Gemma 3 Morality

This repository bootstraps a practical workflow for studying a "morality direction" in `google/gemma-3-1b-*` using:

- `hendrycks/ethics` (`commonsense`) as the high-consensus moral / immoral dataset
- Gemma Scope 2 sparse autoencoders for feature-level analysis
- Modal for remote GPU execution and persistent Hugging Face caching

The baseline implemented here is:

1. Load ETHICS commonsense examples.
2. Extract residual-stream activations from a chosen Gemma 3 layer.
3. Fit a linear morality direction from morally wrong vs acceptable scenarios.
4. Evaluate separation on held-out ETHICS examples.
5. Optionally encode those activations with a Gemma Scope SAE and rank aligned features.
6. Score custom statements and compare latent projection against Gemma's stated moral judgment.

This is intentionally a strong baseline rather than a finished paper pipeline. It gets the mechanics in place so later work can add causal interventions, multi-foundation decompositions, controversial examples, or full circuit tracing.

## Why This Design

The main practical choices are grounded in current primary sources:

- Hugging Face's ETHICS dataset card exposes `hendrycks/ethics`, with `commonsense` examples containing `input`, `label`, `is_short`, and `edited` fields.
- Hugging Face's Gemma 3 docs document `google/gemma-3-1b-pt` as the 1B text-only checkpoint and note Gemma 3 support in Transformers.
- Google's Gemma Scope 2 model cards expose 1B pretrained and instruction-tuned SAE releases, with SAELens loading examples such as:
  - `release="gemma-scope-2-1b-pt-resid_post"`
  - `sae_id="layer_12_width_16k_l0_small"`
- Modal documents the recommended pattern of storing downloaded model assets inside a persistent Volume, then committing the Volume so later runs reuse the cache.

One important practical note: for a `transformers`-based workflow there is not really a "never download weights" path. The workable approach is to keep downloads off your laptop and cache them remotely inside a Modal Volume. That is what this codebase does.

## Repository Layout

```text
configs/
  morality_direction_it.toml
  morality_direction_pt.toml
data/
  statements/example_statements.txt
src/mech_interp/
  cli.py
  config.py
  data.py
  direction.py
  interventions.py
  judge.py
  modeling.py
  pipeline.py
  sae_analysis.py
  utils.py
modal_app.py
```

## Local Setup

Gemma models are gated. Before running anything, accept the Gemma license on Hugging Face and create an access token with model access.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
export HF_TOKEN=...
```

Run the baseline locally:

```bash
python3 -m mech_interp.cli run \
  --config configs/morality_direction_it.toml \
  --statements data/statements/example_statements.txt
```

Outputs are written under `runs/<experiment>_<timestamp>/`, including:

- `summary.json`
- `direction.npy`
- `train_projections.csv`
- `eval_projections.csv`
- `projection_histogram.png`
- `sae_feature_summary.csv` if SAE analysis is enabled
- `custom_statement_scores.csv` if a statements file is provided

## Modal Setup

Install Modal locally and authenticate once:

```bash
pip install modal
modal setup
modal secret create huggingface HF_TOKEN=...
```

Then launch the remote run:

```bash
modal run modal_app.py --config configs/morality_direction_it.toml --statements data/statements/example_statements.txt
```

This mounts two persistent Modal Volumes:

- `/cache/hf` for Hugging Face downloads and tokenizer/model/SAE cache
- `/root/project/runs` for experiment outputs

## Choosing PT vs IT

- `configs/morality_direction_it.toml`
  - Best for comparing internal projections to explicit model judgments.
  - Uses `google/gemma-3-1b-it` and `gemma-scope-2-1b-it-resid_post`.
- `configs/morality_direction_pt.toml`
  - Better if you want to focus on pretrained internal structure before instruction tuning.
  - Uses `google/gemma-3-1b-pt` and `gemma-scope-2-1b-pt-resid_post`.

## Current Limitations

- The implemented baseline uses residual-stream activations at one layer at a time.
- The default direction fit is a normalized difference-of-means vector, not a full causal proof.
- SAE analysis is feature-ranking and alignment-oriented; it does not yet trace end-to-end circuits.
- The intervention helpers are included, but not yet wired into a full automated causal benchmark.

## Good Next Steps

Once the baseline is running, the natural extensions are:

1. Compare morality directions across layers.
2. Compare PT vs IT models.
3. Fit separate directions for different moral foundations or controversy subsets.
4. Add activation ablation / steering experiments using `interventions.py`.
5. Compare latent morality scores with explicit Gemma answers on contentious items.

