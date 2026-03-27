from __future__ import annotations

from typing import Sequence


def _sequence_logprob(model_bundle, prompt: str, continuation: str, max_length: int) -> float:
    import torch

    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    device = next(model.parameters()).device

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    full_ids = tokenizer(
        prompt + continuation,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    full_ids = {key: value.to(device) for key, value in full_ids.items()}

    prompt_length = prompt_ids.shape[1]
    full_length = full_ids["input_ids"].shape[1]
    if full_length <= prompt_length:
        raise ValueError("Continuation did not add any new tokens.")

    with torch.no_grad():
        outputs = model(**full_ids, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        targets = full_ids["input_ids"][:, 1:]
        token_logprobs = logits.log_softmax(dim=-1).gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)

    continuation_logprob = token_logprobs[:, prompt_length - 1 :].sum().item()
    return float(continuation_logprob)


def judge_statements(
    statements: Sequence[str],
    prompt_template: str,
    positive_completion: str,
    negative_completion: str,
    model_bundle,
    max_length: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for statement in statements:
        prompt = prompt_template.format(text=statement)
        positive_logprob = _sequence_logprob(
            model_bundle=model_bundle,
            prompt=prompt,
            continuation=positive_completion,
            max_length=max_length,
        )
        negative_logprob = _sequence_logprob(
            model_bundle=model_bundle,
            prompt=prompt,
            continuation=negative_completion,
            max_length=max_length,
        )
        rows.append(
            {
                "statement": statement,
                "positive_logprob": positive_logprob,
                "negative_logprob": negative_logprob,
                "logprob_gap": positive_logprob - negative_logprob,
                "preferred_completion": (
                    positive_completion.strip()
                    if positive_logprob >= negative_logprob
                    else negative_completion.strip()
                ),
            }
        )
    return rows

