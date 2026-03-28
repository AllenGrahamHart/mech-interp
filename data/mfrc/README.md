# MFRC Aggregation

This directory contains comment-level MFRC datasets derived from the Hugging Face dataset
`USC-MOLA-Lab/MFRC` using a high-precision aggregation rule.

Aggregation rule:
- Group by `(text, subreddit, bucket)`.
- Drop `Not Confident` annotations.
- Split multi-label annotations on commas.
- A foundation is present if it receives at least 2 retained votes.
- `strict_nonmoral` means `Non-Moral >= 2`, no foundation has 2 votes, and `Thin Morality < 2`.
- High-precision positive examples require the target foundation to have 2 votes and both `Non-Moral < 2` and `Thin Morality < 2`.
- Pure positives additionally require no other foundation to have 2 votes.

Aggregated comments: 17886
Strict non-moral comments: 8308

High-precision foundation counts:
- Care: 1670
- Equality: 1129
- Proportionality: 620
- Purity: 238
- Authority: 670
- Loyalty: 450

Pure foundation counts:
- Care: 1111
- Equality: 683
- Proportionality: 279
- Purity: 135
- Authority: 452
- Loyalty: 225

Files:
- `mfrc_aggregated_comments.csv`: comment-level votes and derived flags.
- `mfrc_<foundation>_vs_strict_nonmoral.csv`: main one-vs-non-moral datasets.
- `mfrc_<foundation>_pure_vs_strict_nonmoral.csv`: cleaner pure-positive variants.
