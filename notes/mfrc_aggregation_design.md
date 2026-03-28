# MFRC aggregation design

We use the Hugging Face dataset `USC-MOLA-Lab/MFRC` as a comment-level moral-foundation corpus.

## Why MFRC

- It is much larger than the Moral Foundations Vignettes.
- It preserves annotator-level disagreement and confidence.
- It uses a revised but still recognizable MFT taxonomy:
  - `Care`
  - `Equality`
  - `Proportionality`
  - `Purity`
  - `Authority`
  - `Loyalty`
  - plus `Thin Morality` and `Non-Moral`

## Comment-level structure

The Hugging Face dataset stores one row per annotator judgment, not one row per unique comment.
We aggregate by `(text, subreddit, bucket)`.

Empirical structure in the train split:

- `17,886` unique comments
- Original annotations per comment:
  - `10,191` comments have 3 annotations
  - `7,534` comments have 4 annotations
- After dropping `Not Confident`, retained annotations per comment are usually still between 2 and 4.

## Main aggregation rule

1. Drop `Not Confident` annotations.
2. Split multi-label annotations on commas.
3. Count votes for each label.
4. Mark a foundation present if it receives at least `2` retained votes.

We define:

- `strict_nonmoral`:
  - `Non-Moral >= 2`
  - no foundation has `>= 2`
  - `Thin Morality < 2`

- `foundation-positive` for foundation `f`:
  - `f >= 2`
  - `Non-Moral < 2`
  - `Thin Morality < 2`

- `pure foundation-positive` for foundation `f`:
  - foundation-positive for `f`
  - no other foundation has `>= 2`

This keeps multi-label foundation comments in the main dataset, while excluding comments that are still mixed with `Non-Moral` or `Thin Morality`.

## Counts under the main rule

Foundation-present comments (`>=2` votes, before excluding `Non-Moral` / `Thin Morality` mixtures):

- `Care`: `1,782`
- `Equality`: `1,186`
- `Proportionality`: `828`
- `Purity`: `256`
- `Authority`: `734`
- `Loyalty`: `521`

Main high-precision positives:

- `Care`: `1,670`
- `Equality`: `1,129`
- `Proportionality`: `620`
- `Purity`: `238`
- `Authority`: `670`
- `Loyalty`: `450`

Pure positives:

- `Care`: `1,111`
- `Equality`: `683`
- `Proportionality`: `279`
- `Purity`: `135`
- `Authority`: `452`
- `Loyalty`: `225`

Strict non-moral negatives:

- `8,308`

Other relevant counts:

- comments with at least one foundation under the `>=2` rule: `4,133`
- comments with more than one present label under the `>=2` rule: `1,799`
- `Thin Morality` only comments: `1,133`

## Notes

- `Non-Moral` should be treated as a foundation-negative baseline, not as philosophically neutral or sentiment-neutral text.
- The bucket distribution is imbalanced, so later experiments may want bucket-matched negatives as an additional robustness check.
- The split assignment for local CSV datasets is deterministic at the comment level so the same comment does not leak across train and test in different derived datasets.
