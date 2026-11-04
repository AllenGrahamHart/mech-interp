# Moral Representations in Language Models

Research code for studying linear representations of moral acceptability and moral
foundations in language-model activations, led by **Allen Hart, Research Associate
at the University of Warwick**.

The repository supports Allen's stream in the
[Mercial Research Fellowship](https://mercial.uk/fellowship.html), which asks how
models evaluate weight exfiltration, successionism, self-sustaining AI
civilisations, and other potentially catastrophic misalignment scenarios.
**The experiments recorded here concern ordinary moral scenarios and Reddit
comments. Applying these methods to the fellowship's AI-specific scenarios remains
an open research task.**

## Two experimental programmes

| | 1. Single moral-acceptability feature | 2. Six moral foundations |
| --- | --- | --- |
| Question | Is there a stable linear direction associated with judging an action wrong? | How are different kinds of moral language represented? |
| Dataset | Hendrycks et al.'s ETHICS, short commonsense scenarios | Moral Foundations Reddit Corpus (MFRC) |
| Contrast | Wrong/immoral vs acceptable/not wrong | Each foundation vs comments annotated Non-Moral |
| Analyses performed | Layer sweeps, ablation, sentiment controls, random-direction controls, bootstrap stability | Layer sweeps, comparisons within the same layer, bootstrap angle tables |
| Status | Baseline with evidence from interventions | Exploratory geometry and decoding analysis |

**All results below use `Qwen/Qwen2.5-0.5B-Instruct` on Modal GPUs**, with raw
hidden states and normalized differences of class means. These experiments use
neither sparse autoencoders nor nonlinear classifiers. The model has 24 transformer
layers; layer numbers in this repository are zero-based.

## Programme 1: A single moral-acceptability feature

### Dataset and direction extraction

We use the short scenarios from the commonsense subset of
[ETHICS (Hendrycks et al.)](https://arxiv.org/abs/2008.02275), loaded from
[the Hugging Face dataset](https://huggingface.co/datasets/hendrycks/ethics).

| Split | Acceptable / not wrong (0) | Wrong / immoral (1) | Total |
| --- | ---: | ---: | ---: |
| Train | 3,872 | 2,789 | 6,661 |
| Test | 1,126 | 983 | 2,109 |

These are the filtered counts used in our experiments, not the size of the full
ETHICS benchmark. Acceptable examples include ordinary actions as well as
prosocial ones; that label does not necessarily mean virtuous or sentiment-neutral.

For each layer, we extract the final non-padding token's hidden state from:

```text
Scenario: {text}
Moral evaluation:
```

The direction is estimated from training examples:

```text
d = mean(h_wrong) - mean(h_acceptable)
u = d / ||d||
score(h) = h . u
```

Code and figures call this the candidate **immorality direction**. Higher
projection means stronger alignment with the training contrast. It is not a
calibrated measure of moral severity or proof of the model's beliefs.

### Recorded results

- The strongest ETHICS projection AUROC in the layer sweep was **0.8290 at layer
  15**, with several nearby middle and late layers performing similarly.
- In **1,000 stratified bootstrap resamples** of training activations, the 95th
  percentile angular deviation from the full-data direction was **5.79 degrees**.
- A sentiment direction was fit from **5,578 SST-2 training examples**, balanced
  between positive and negative sentiment. At layer 15, immorality and negative
  sentiment had cosine similarity **0.5219**, or **58.54 degrees** separation.
- Removing the component parallel to that sentiment direction left ETHICS
  projection AUROC of **0.8003**, compared with **0.8286** in the same run. This
  controls for one estimated sentiment axis, not all possible sentiment information.

### Causal ablation

The causal benchmark uses a separate judgment prompt:

```text
Scenario: {text}
Is this morally acceptable? Answer with one word:
```

It scores the full continuations ` acceptable` and ` unacceptable`, using
`log p(" unacceptable") - log p(" acceptable")`. These are constrained completion
scores, not freely generated answers. AUROC measures how well this gap ranks
wrong examples above acceptable ones; 0.5 is chance and 1.0 is perfect ranking.

For a unit direction `u`, removal at the layer-15 prompt-boundary token is
`h' = h - (h . u)u`: projection onto the orthogonal subspace.

| Direction removed | Baseline judgment AUROC | AUROC after removal |
| --- | ---: | ---: |
| Full ETHICS direction | 0.7758 | 0.6357 |
| SST-2 sentiment direction | 0.7763 | 0.6607 |
| ETHICS direction orthogonalized against sentiment | 0.7763 | 0.7407 |

Each benchmark used **2,109 evaluation examples**. These were separate runs with
slightly different baseline scores. Judgment AUROC is a different measurement
from the activation-projection AUROC above.

In a further control, **32 random unit-direction projection ablations** gave mean
judgment AUROC **0.7759**. The largest random AUROC drop was **0.00683**, compared
with **0.14011** for the ETHICS direction. Equal unit-vector norms do not ensure
equal perturbation magnitudes: the removed amount depends on `h . u`.

The results support a candidate direction that affects moral-judgment scores.
Error inspection also found sensitivity to violent or taboo wording, subtle
wrongdoing, and awkward phrasing. Latent projections and preferred completions
sometimes disagree; these discrepancies can reflect comprehension or prompt
effects as well as differences in the signals being read.

See the [bootstrap analysis](notes/bootstrap_uncertainty.md),
[error examples](notes/error_modes.md), and
[latent/completion comparison](notes/latent_expressed_mismatch.md).

## Programme 2: Six moral foundations

### Dataset and aggregation

The [Moral Foundations Reddit Corpus](https://arxiv.org/abs/2208.05545), available
as [USC-MOLA-Lab/MFRC](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC), annotates
moral language in Reddit comments. We use its six foundations: **Care, Equality,
Proportionality, Purity, Authority, and Loyalty**.

A comment can express multiple foundations, through praise, condemnation, or
discussion. **Non-Moral means no moral foundation was identified; it does not mean
morally acceptable, emotionally neutral, or politically neutral.** This programme
measures foundation expression, rather than six different kinds of wrongdoing.

We aggregate annotations by `(text, subreddit, bucket)`:

1. Discard annotations marked Not Confident.
2. Require at least two retained votes for a foundation.
3. For a positive example, also require fewer than two votes each for Non-Moral
   and Thin Morality.
4. For the strict Non-Moral baseline, require at least two Non-Moral votes, fewer
   than two votes for every foundation, and fewer than two Thin Morality votes.

Multi-foundation positives are retained. The baseline can still include a single
dissenting foundation vote. Splits are assigned deterministically by the aggregation
key. The derived datasets contain **8,308 strict Non-Moral comments**.

### Recorded results

| Foundation | Available positives | Sweep train / eval totals | Best layer | Eval AUROC |
| --- | ---: | ---: | ---: | ---: |
| Care | 1,670 | 1,024 / 512 | 4 | 0.871 |
| Loyalty | 450 | 873 / 345 | 6 | 0.784 |
| Proportionality | 620 | 1,009 / 379 | 6 | 0.779 |
| Authority | 670 | 1,024 / 385 | 19 | 0.777 |
| Equality | 1,129 | 1,024 / 494 | 22 | 0.886 |
| Purity | 238 | 696 / 310 | 22 | 0.800 |

Sweep totals include positives and negatives. Configurations cap sampling at
512 training and 256 evaluation examples **per class**; smaller classes supply
fewer examples. Positive pools overlap across foundations.

The representation prompt is `Comment: {text}\nFoundation signal:`. We fit each
foundation's mean-difference direction against the Non-Moral baseline across layers.

For each foundation's best-decoding layer, we compare its direction with **all five
other foundations at that same layer**, even when those foundations peak elsewhere.
Two aligned tables report angles with 1,000-replicate bootstrap intervals and the
corresponding evaluation AUROCs. Loyalty and Proportionality are about **31 degrees**
apart at layer 6; Equality and Purity are about **46 degrees** apart at layer 22.

The pattern is **different decoding peaks with substantial shared alignment**.
Different best layers do not establish that concepts exist only at those layers,
and the results do not establish six independent axes. Foundation-specific causal
validation remains to be done.

See the [aggregation specification](notes/mfrc_aggregation_design.md),
[derived dataset README](data/mfrc/README.md),
[layerwise analysis](notes/mfrc_layerwise_structure.md), and
[aligned tables](notes/mfrc_same_layer_tables.md).

## Comparison between programmes

An exploratory analysis projected the ETHICS direction onto the span of the six
MFRC directions **within each layer**. The captured fraction of the ETHICS vector's
**squared norm** was **19.1% at layer 15** and **40.8% at layer 23**, the largest
value in that sweep.

This is geometric alignment between two dataset contrasts, not the percentage of
moral judgment explained by foundations. ETHICS contrasts wrong with acceptable
scenarios; MFRC contrasts foundation expression with Non-Moral Reddit comments,
using a different prompt and domain. That mismatch limits any claim that one
decomposes the other. The [subspace note](notes/foundation_subspace_projection.md)
records the calculation; its interpretations should be read with this caveat.

## Interpretation and limitations

The reported numbers are exploratory results from March 2026, not independently
replicated or peer-reviewed findings.

- **Layer selection:** evaluation scores were used both to select best layers and
  report performance. An untouched confirmatory split is still needed.
- **Model and prompts:** most results use one small model and fixed English
  templates. Model comprehension, truncation of long inputs, and prompt effects
  need further checks.
- **Concept specificity:** sentiment and random-direction controls support the
  ETHICS result, but lexical, toxicity, and perturbation-size controls remain
  incomplete. The experiments do not identify a complete circuit.
- **MFRC confounds:** shared negatives, multi-label comments, topic, subreddit,
  and writing style can contribute to alignment. Negatives are not currently
  subreddit- or bucket-matched.
- **Bootstrap interpretation:** resampling measures variability within the observed
  datasets. MFRC table code resamples foundation datasets separately despite shared
  comments and controls; joint resampling by comment is needed for stronger
  statistical claims about distinctness.
- **Scope of inference:** an activation score does not establish endorsement,
  intentions, or agent behaviour. Transfer to the fellowship's AI-specific
  scenarios and more capable models remains untested here.

The [literature note](notes/morality_direction_literature_note.md) dates to March
2026 and needs refreshing before making novelty claims. The
[paper PDF](paper/main.pdf) is a draft scaffold containing notes, figures, and
tables. Some notes use stronger interpretive language than the caveats above.

## Running experiments

### Setup

Use Python **3.11 or 3.12** and run commands from the repository root. Inference
runs on a Modal A10G; a local GPU is unnecessary. Weights are downloaded and cached
on Modal.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
modal setup
```

CPU PyTorch is for local imports and analysis; the Modal image installs its own GPU
dependencies. Dependencies have lower bounds rather than a locked environment,
so a fresh install may need compatibility checks.

The Modal app expects a secret named `huggingface`, even for public models.
If it does not already exist, public Qwen runs can use an empty token:

```bash
modal secret create huggingface HF_TOKEN=""
```

Always pass an explicit Qwen config for the experiments below: the app's default
config selects Gemma. Gemma configs and optional SAE support are available in the
codebase but are not used in the reported results. Gated model runs require a real
Hugging Face token with the applicable model access.

### Programme 1 commands

Start with a small raw-activation sweep:

```bash
modal run modal_app.py --mode sweep --config configs/morality_direction_smoke_open.toml
```

Run the full ETHICS sweep, sentiment control sweep, and full evaluation ablation:

```bash
modal run modal_app.py --mode sweep --config configs/morality_direction_open_full.toml
modal run modal_app.py --mode sweep --config configs/sentiment_direction_open_matched.toml
modal run modal_app.py --mode causal --config configs/morality_direction_open_full_causal_all.toml --layer 15
```

Run the sentiment-controlled ablation or random-direction control:

```bash
modal run modal_app.py --mode orthogonalized-causal \
  --config configs/morality_direction_open_full_causal_all.toml \
  --control-config configs/sentiment_direction_open_matched.toml --layer 15

modal run modal_app.py --mode random-causal \
  --config configs/morality_direction_open_full_causal_all.toml \
  --layer 15 --num-random-directions 32
```

`--mode control-causal` with the same primary and control configs ablates pure
sentiment; `--mode orthogonalize` evaluates the sentiment-orthogonalized projection.
The causal configs also run steering at `-3, -1, +1, +3`; set
`[intervention].add_scales = []` in a config copy for a removal-only run.

### Programme 2 commands

The aggregated MFRC CSVs are included in git. Run the six sweeps with:

```bash
for config in configs/mfrc_*_vs_strict_nonmoral.toml; do
  modal run modal_app.py --mode sweep --config "$config"
done
```

Configs under [configs/mfrc_same_layer/](configs/mfrc_same_layer/) specify the
single-layer extraction runs used for bootstrap tables. Use `--mode run` with
those configs to save training activations. The offline scripts include
[bootstrap_direction_uncertainty.py](scripts/bootstrap_direction_uncertainty.py),
[build_mfrc_same_layer_tables.py](scripts/build_mfrc_same_layer_tables.py), and
[analyze_wrongness_foundation_subspace.py](scripts/analyze_wrongness_foundation_subspace.py).
Several scripts default to specific run directories; check their arguments and
paths before reuse.

The local CLI exposes the modes through `python -m mech_interp.cli --help`.
Its `run` mode also accepts `--statements data/statements/example_statements.txt`
for custom projections and constrained judgment scores. A suitable inference
environment is needed to run the model locally.

## Artifacts and reproducibility

Modal stores Hugging Face assets in volume `mech-interp-hf-cache` at
`/cache/hf` and results in `mech-interp-runs` at `/root/project/runs`.
The app is named `gemma-morality-interp`, including for Qwen runs. Access to
existing volumes requires the originating Modal workspace; another workspace
creates its own volumes when running the app.

Runs save `config.json`, `summary.json`, and mode-specific artifacts:

| Mode | Main artifacts |
| --- | --- |
| `run` | Direction, train/eval activation arrays, projection CSVs, histogram |
| `sweep` | Layer metrics, stability CSV, per-layer directions and projections |
| Causal modes | Directions, per-example intervention scores, aggregate intervention metrics |
| `random-causal` | Morality direction, per-random-direction aggregate metrics, summary |

Some causal aggregate CSVs omit AUROC; it can be recomputed from per-example
labels and gaps using [plot_auroc_comparison.py](scripts/plot_auroc_comparison.py).
Bootstrapping uses saved activations and requires no additional model forwards.

Download a run after locating its directory:

```bash
modal volume ls mech-interp-runs
modal volume get mech-interp-runs /RUN_DIRECTORY tmp/modal_results/RUN_DIRECTORY
```

Replace `RUN_DIRECTORY` with the actual run name. Local outputs under
`runs/` and downloaded results under `tmp/modal_results/` are **gitignored**.
A GitHub clone contains code, configs, derived foundation CSVs, notes, and the
compiled paper, but not activation arrays or full experiment outputs. Some notes
link to absolute local paths. The archived runs must be obtained from Allen or
regenerated to reproduce those analyses.

## Repository guide

| Location | Contents |
| --- | --- |
| [src/mech_interp/](src/mech_interp/) | Data loading, activation extraction, direction fitting, completion scoring, interventions |
| [modal_app.py](modal_app.py) | Remote GPU execution, source/config/data upload, persistent volumes |
| [configs/](configs/) | ETHICS, SST-2, MFRC, Gemma, and MFV experiment configurations |
| [data/mfrc/](data/mfrc/README.md) | Aggregated MFRC comments and binary foundation datasets |
| [data/moral_foundations/](data/moral_foundations/README.md) | Small Moral Foundations Vignettes dataset; separate from the MFRC results above |
| [data/statements/](data/statements/) | Example custom statements |
| [scripts/](scripts/) | Dataset preparation and offline analysis |
| [notes/](notes/) | Experiment records and literature notes |
| [paper/](paper/) | LaTeX draft, bibliography, and tracked PDF |

To rebuild the paper, restore its referenced figure and table files under
`tmp/modal_results/`, then run `latexmk -pdf main.tex` from `paper/`.
Those generated assets are not included in git, so the current LaTeX source does
not build from a fresh clone alone; the tracked PDF can be read directly.
