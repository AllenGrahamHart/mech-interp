# Moral Foundations Vignettes Data

This directory contains local CSV datasets derived from the original Moral Foundations Vignettes paper:

- Scott Clifford, Vijeth Iyengar, Roberto Cabeza, Walter Sinnott-Armstrong.
  *Moral foundations vignettes: a standardized stimulus database of scenarios based on moral foundations theory.*
  Behavior Research Methods, 2015.

Source PDF used for extraction:

- `https://scottaclifford.com/wp-content/uploads/2012/02/CICSA_MoralVignettes_BRM.pdf`

Files:

- `mfv_full.csv`
  Full extracted set of 132 scenarios with foundation labels and deterministic train/test splits.
- `mfv_<foundation>_vs_social_norms.csv`
  Binary foundation-vs-control datasets for:
  - care
  - fairness
  - loyalty
  - authority
  - sanctity
  - liberty

Important caveat:

The original paper discusses a recommended 90-item subset in Table 6. The accessible PDF text exposes the selection rule but not the actual table contents, so the current local dataset uses the full Table 1 scenario set rather than the hidden 90-item subset.

Regeneration:

```bash
python3 scripts/build_mfv_datasets.py
```
