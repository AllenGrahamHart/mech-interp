# Bootstrap Uncertainty for Layer-15 Directions

This note records the bootstrap uncertainty estimates for the layer-15 `immorality` direction fit on ETHICS and the layer-15 `negative sentiment` direction fit on SST-2.

Method:
- Stratified bootstrap over the training activations only.
- For each replicate, resample each class with replacement at the original class size.
- Recompute the mean-difference direction and normalize it.
- Compare the bootstrap direction to the full-data direction using cosine similarity and angular deviation.
- Also record the bootstrap distribution of the cosine overlap between the immorality and negative-sentiment directions.

Artifacts:
- [summary.json](/home/u2470931/mech_interp/tmp/modal_results/bootstrap_uncertainty_layer15/summary.json)
- [bootstrap_replicates.csv](/home/u2470931/mech_interp/tmp/modal_results/bootstrap_uncertainty_layer15/bootstrap_replicates.csv)
- Script: [bootstrap_direction_uncertainty.py](/home/u2470931/mech_interp/scripts/bootstrap_direction_uncertainty.py)

## 1. Immorality Direction Uncertainty

Training set:
- Total examples: `6661`
- Immoral examples: `2789`
- Acceptable examples: `3872`

Bootstrap summary:
- Mean cosine to full-data direction: `0.9970`
- 95% cosine interval: `[0.9949, 0.9983]`
- Mean angular deviation: `4.37°`
- 95% angular interval: `[3.32°, 5.79°]`
- 97.5th percentile angle: `6.33°`

Interpretation:
- The learned ETHICS immorality direction is very stable under dataset resampling.
- A useful one-line summary is that the layer-15 immorality direction remains within about `6°` of the full-data estimate under `95–97.5%` bootstrap resampling.

## 2. Negative-Sentiment Direction Uncertainty

Training set:
- Total examples: `5578`
- Negative examples: `2789`
- Positive examples: `2789`

Bootstrap summary:
- Mean cosine to full-data direction: `0.9977`
- 95% cosine interval: `[0.9957, 0.9988]`
- Mean angular deviation: `3.82°`
- 95% angular interval: `[2.82°, 5.33°]`
- 97.5th percentile angle: `5.78°`

Interpretation:
- The negative-sentiment direction is also highly stable, and slightly tighter than the immorality direction in this experiment.

## 3. Uncertainty on Direction Overlap

Observed overlap:
- Cosine between full-data directions: `0.5219`
- Equivalent angle: `58.54°`

Bootstrap summary:
- Mean overlap cosine: `0.5194`
- 95% cosine interval: `[0.5012, 0.5359]`
- Mean overlap angle: `58.70°`
- 95% angular interval: `[57.59°, 59.92°]`

Interpretation:
- The relationship between the two directions is itself stable.
- The overlap is substantial, but the directions remain clearly distinct across bootstrap resamples.

## Suggested One-Sentence Summary

At layer 15, both the ETHICS-derived immorality direction and the SST-2-derived negative-sentiment direction are highly stable under stratified bootstrap resampling, remaining within roughly `6°` of their full-data estimates, while their mutual overlap is also stable at about `0.52` cosine (`~59°` apart).
