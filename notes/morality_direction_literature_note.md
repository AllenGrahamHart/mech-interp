# Literature Note: Linear Morality Representations in Language Models

Date: 2026-03-27

## Question

Has a linear "morality direction" or internal moral feature in language models already been studied, and if so, what remains novel for this project?

## Bottom Line

The broad claim "language models contain a linear morality feature" is no longer novel on its own.

There is clear prior work in three adjacent but distinct lines:

1. Moral direction in embedding space or sentence-representation space.
2. Linear internal directions for related high-level concepts such as sentiment, harmlessness, power-seeking, toxicity, and bias.
3. Very recent hidden-state studies that explicitly probe moral or ethical subspaces inside LLMs.

However, I did not find a paper that exactly matches the present design:

- a small open-weight autoregressive model,
- high-consensus ETHICS commonsense items,
- raw residual-stream activations,
- a simple layerwise mean-difference direction from final-token states,
- explicit sample-size stability analysis,
- and causal ablation/addition tests on that same direction.

That combination still looks plausibly novel and, more importantly, clean.

## Closest Prior Art

### 1. Schramowski et al. (2021)

Patrick Schramowski, Cigdem Turan, Nico Andersen, Constantin A. Rothkopf, Kristian Kersting.
"Large Pre-trained Language Models Contain Human-like Biases of What is Right and Wrong to Do."

Link: https://arxiv.org/abs/2103.11790

Why it matters:

- This is the clearest early "moral direction" paper.
- The paper says moral norms can be captured geometrically by a direction computed in embedding space.
- They use this direction to rate phrases by normativity and to steer generation away from toxic degeneration.

Why it does not fully overlap with the present project:

- The main object is an embedding-space or phrase-space moral direction, not a layerwise residual-stream feature in a modern decoder-only model.
- It is not a raw hidden-state mechanistic study in the present sense.

Direct support from the abstract:

- The paper says LMs "bring a 'moral direction' to surface."
- It also says this direction can be computed "in the embedding space."

### 2. Hammerl et al. (2023)

"Speaking Multiple Languages Affects the Moral Bias of Language Models."

Link: https://aclanthology.org/2023.findings-acl.134/

Why it matters:

- Extends the `MORALDIRECTION` idea cross-lingually.
- Shows that moral bias is language-sensitive rather than trivially universal.

Relevance:

- This is important for any future multilingual extension of the project.
- It suggests that "morality direction" may not be language-invariant even when the model is multilingual.

### 3. Zou et al. (2023)

Andy Zou et al.
"Representation Engineering: A Top-Down Approach to AI Transparency."

Link: https://arxiv.org/abs/2310.01405

Why it matters:

- Establishes the general methodology of extracting high-level linear directions in hidden-state space.
- Their case studies include honesty, harmlessness, and power-seeking.

Relevance:

- This is a strong methodological precedent for treating a high-level safety-relevant concept as a direction in activation space.
- It is not morality-specific, but it legitimizes the basic approach.

### 4. Tigges et al. (2023)

Curt Tigges, Oskar John Hollinsworth, Atticus Geiger, Neel Nanda.
"Linear Representations of Sentiment in Large Language Models."

Link: https://arxiv.org/abs/2310.15154

Why it matters:

- This is the cleanest precedent for the exact style of study we want to run.
- It shows sentiment is represented linearly and uses causal interventions to validate that direction.

Relevance:

- Sentiment is not morality, but it is the strongest methodological analogue.
- It shows how to move from a linear separator to a causal claim.

Direct support from the abstract:

- The paper says sentiment is represented linearly by "a single direction in activation space."
- It also reports causal validation through interventions and ablations.

### 5. Zhou et al. (2024)

Zhenhong Zhou et al.
"How Alignment and Jailbreak Work: Explain LLM Safety through Intermediate Hidden States."

Link: https://arxiv.org/abs/2406.05644

Why it matters:

- Hidden-state study of safety and jailbreak behavior.
- The paper argues that models learn ethical concepts during pretraining and transform them through later layers during aligned behavior.

Why it is adjacent rather than identical:

- The concept under study is closer to maliciousness / safety refusal than to a clean ETHICS-style moral acceptability dimension.
- The analysis uses weak classifiers over hidden states rather than isolating a simple consensus morality direction from moral scenarios.

### 6. Liu et al. (2024)

Guangliang Liu et al.
"Intrinsic Self-correction for Enhanced Morality: An Analysis of Internal Mechanisms and the Superficial Hypothesis."

Link: https://arxiv.org/abs/2407.15286

Why it matters:

- Explicitly studies "morality levels" in hidden states.
- Uses hidden-state analysis to argue that self-correction can improve outputs without truly removing immorality from internal representations.

Why it is not the same project:

- The focus is self-correction prompts and toxicity / bias-style benchmarks.
- The hidden-state signal is used to analyze self-correction rather than to isolate a consensus morality direction from ETHICS-style scenario judgments.

### 7. Liu et al. (2024/2026 revision)

Guangliang Liu et al.
"Self-correction is Not An Innate Capability in Language Models."

Link: https://arxiv.org/abs/2410.20513

Why it matters:

- Extends the self-correction line with both behavioral and mechanistic analysis.
- Again treats moral sensitivity as something visible in hidden states.

Why it is still only partial overlap:

- The paper is about self-correction and feedback use, not directly about a stable linear morality direction fit from consensus examples.

### 8. Hu et al. (2026)

Luoming Hu et al.
"The Straight and Narrow: Do LLMs Possess an Internal Moral Path?"

Link: https://arxiv.org/abs/2601.10307

Why it matters:

- This is the closest hit.
- The abstract explicitly claims cross-lingual linear probing, shared moral representations in middle layers, and steerable moral vectors.

Why it still does not fully subsume the present design:

- The paper is framed through Moral Foundations Theory and adaptive inference-time intervention.
- It is not obviously the same as a raw residual-stream ETHICS-commonsense study with simple final-token mean-difference estimation and sample-size stabilization.

This paper preempts the naive claim that nobody has studied an internal moral direction at all.

### 9. Xu et al. (2026)

Weilun Xu, Alexander Rusnak, Frederic Kaplan.
"Probing Ethical Framework Representations in Large Language Models: Structure, Entanglement, and Methodological Challenges."

Link: https://arxiv.org/abs/2603.23659

Why it matters:

- Very recent and highly relevant.
- Probes hidden representations across deontology, utilitarianism, virtue, justice, and commonsense.
- Explicitly asks whether ethics collapses to a single acceptability dimension.

Why it matters for our design:

- It suggests that the space may contain multiple ethical subspaces rather than one universal "good/bad" line.
- It also reports template dependence and surface-feature confounds, which is directly relevant to our controls.

This paper significantly raises the novelty bar.

## Novelty Assessment

### What is probably no longer novel

- The generic existence claim: "LLMs have an internal morality representation."
- The claim that moral or ethical concepts can be linearly probed from hidden states.
- The claim that moral vectors can be used for steering.

### What still looks plausibly novel

The following narrower contribution still appears defensible:

- Establish a simple, raw, layerwise morality direction in a small open model using ETHICS commonsense short scenarios.
- Use only final-token residual-stream activations with a mean-difference estimator.
- Show stability of the learned direction as sample size increases.
- Compare neighboring layers directly and identify where the signal stabilizes.
- Perform causal interventions on that same direction.
- Test whether the direction is distinct from sentiment or toxicity.
- Compare latent projection with explicit model judgments and with contentious examples.

That is a cleaner and more mechanistically specific study than the papers above.

## Recommended Framing

The project should not be framed as:

> Do LLMs have a morality feature at all?

That question is now too broad.

It should instead be framed as:

> How simple, stable, and causally meaningful is the consensus-commonsense morality representation in the raw residual stream of a small open LLM?

Or, more concretely:

> Can a single layerwise linear direction fit from high-consensus ETHICS commonsense scenarios explain and causally influence moral acceptability judgments in a small open model?

This framing preserves novelty and keeps the design clean.

## Experimental Implications

Given the literature, the cleanest next steps are:

1. Keep the main object linear.
2. Use ETHICS commonsense short scenarios for the core direction.
3. Focus on a single aligned readout position, namely the final token of a fixed prompt template.
4. Report sample-size stability explicitly.
5. Add controls for:
   - sentiment,
   - toxicity,
   - template artifacts,
   - lexical overlap.
6. Use causal ablation and addition at the best layer.
7. Treat richer nonlinear probes only as an upper bound on decodability, not as the main mechanistic object.

## Cross-Lingual Note

One useful correction for planning:

Qwen/Qwen2.5-0.5B-Instruct is not English-only. The model card states that Qwen2.5 has "multilingual support for over 29 languages," including English, Chinese, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, and Arabic.

Link: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

So a cross-lingual extension is possible even at 0.5B scale.

That said, multilingual support is not the same thing as equal moral competence across languages. The cross-lingual literature suggests moral representations can shift with language, so a multilingual experiment should be treated as a substantive research extension rather than a trivial replication.

## Practical Conclusion

The project is still viable and interesting, but the novelty claim should be modest and specific.

Best current claim:

- not "we discovered morality inside LLMs,"
- but "we provide a clean residual-stream analysis of consensus morality in a small open model, with stability and causal tests."

That is a solid and defensible contribution.

## References

1. Patrick Schramowski et al. "Large Pre-trained Language Models Contain Human-like Biases of What is Right and Wrong to Do." arXiv 2103.11790.
   https://arxiv.org/abs/2103.11790

2. Theresa Hammerl et al. "Speaking Multiple Languages Affects the Moral Bias of Language Models." Findings of ACL 2023.
   https://aclanthology.org/2023.findings-acl.134/

3. Andy Zou et al. "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv 2310.01405.
   https://arxiv.org/abs/2310.01405

4. Curt Tigges et al. "Linear Representations of Sentiment in Large Language Models." arXiv 2310.15154.
   https://arxiv.org/abs/2310.15154

5. Zhenhong Zhou et al. "How Alignment and Jailbreak Work: Explain LLM Safety through Intermediate Hidden States." arXiv 2406.05644.
   https://arxiv.org/abs/2406.05644

6. Guangliang Liu et al. "Intrinsic Self-correction for Enhanced Morality: An Analysis of Internal Mechanisms and the Superficial Hypothesis." arXiv 2407.15286.
   https://arxiv.org/abs/2407.15286

7. Guangliang Liu et al. "Self-correction is Not An Innate Capability in Language Models." arXiv 2410.20513.
   https://arxiv.org/abs/2410.20513

8. Luoming Hu et al. "The Straight and Narrow: Do LLMs Possess an Internal Moral Path?" arXiv 2601.10307.
   https://arxiv.org/abs/2601.10307

9. Weilun Xu, Alexander Rusnak, Frederic Kaplan. "Probing Ethical Framework Representations in Large Language Models: Structure, Entanglement, and Methodological Challenges." arXiv 2603.23659.
   https://arxiv.org/abs/2603.23659
