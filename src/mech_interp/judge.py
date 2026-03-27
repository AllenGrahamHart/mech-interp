from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

from mech_interp.utils import batched


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


def _single_token_id(tokenizer, token_text: str) -> int:
    token_ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(
            f"Expected '{token_text}' to tokenize to exactly one token, got {len(token_ids)} tokens."
        )
    return int(token_ids[0])


def next_token_logit_gap(
    prompt: str,
    positive_token: str,
    negative_token: str,
    model_bundle,
    max_length: int,
) -> dict[str, object]:
    import torch

    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    device = next(model.parameters()).device

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    positive_token_id = _single_token_id(tokenizer, positive_token)
    negative_token_id = _single_token_id(tokenizer, negative_token)

    with torch.no_grad():
        outputs = model(**encoded, use_cache=False)
        last_logits = outputs.logits[:, -1, :].squeeze(0).float()
        logprobs = last_logits.log_softmax(dim=-1)

    positive_logprob = float(logprobs[positive_token_id].item())
    negative_logprob = float(logprobs[negative_token_id].item())
    return {
        "positive_logprob": positive_logprob,
        "negative_logprob": negative_logprob,
        "logprob_gap": positive_logprob - negative_logprob,
        "preferred_completion": (
            positive_token.strip()
            if positive_logprob >= negative_logprob
            else negative_token.strip()
        ),
    }


def completion_logprob_gap(
    prompt: str,
    positive_completion: str,
    negative_completion: str,
    model_bundle,
    max_length: int,
) -> dict[str, object]:
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
    return {
        "positive_logprob": positive_logprob,
        "negative_logprob": negative_logprob,
        "logprob_gap": positive_logprob - negative_logprob,
        "preferred_completion": (
            positive_completion.strip()
            if positive_logprob >= negative_logprob
            else negative_completion.strip()
        ),
    }


def _batched_sequence_logprobs(
    prompts: Sequence[str],
    continuation: str,
    model_bundle,
    max_length: int,
    batch_size: int,
    intervention_factory=None,
) -> list[float]:
    import torch

    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    device = next(model.parameters()).device

    logprobs: list[float] = []
    for prompt_batch in batched(list(prompts), batch_size):
        prompt_batch = list(prompt_batch)
        prompt_encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        prompt_lengths = prompt_encoded["attention_mask"].sum(dim=1)

        full_texts = [prompt + continuation for prompt in prompt_batch]
        full_encoded = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        full_lengths = full_encoded["attention_mask"].sum(dim=1)
        if torch.any(full_lengths <= prompt_lengths):
            raise ValueError("Continuation did not add any new tokens for at least one prompt in the batch.")

        sequence_length = full_encoded["input_ids"].shape[1]
        start_indices = (sequence_length - full_lengths + prompt_lengths - 1).to(dtype=torch.long)
        full_encoded = {key: value.to(device) for key, value in full_encoded.items()}

        intervention_context = nullcontext()
        if intervention_factory is not None:
            intervention_context = intervention_factory(start_indices.tolist())

        with intervention_context:
            with torch.no_grad():
                outputs = model(**full_encoded, use_cache=False)
                logits = outputs.logits[:, :-1, :]
                targets = full_encoded["input_ids"][:, 1:]
                token_logprobs = logits.log_softmax(dim=-1).gather(
                    dim=-1,
                    index=targets.unsqueeze(-1),
                ).squeeze(-1)

        positions = torch.arange(token_logprobs.shape[1], device=device).unsqueeze(0)
        continuation_mask = positions >= start_indices.to(device).unsqueeze(1)
        batch_logprobs = (token_logprobs * continuation_mask).sum(dim=1)
        logprobs.extend(float(value) for value in batch_logprobs.cpu().tolist())

    return logprobs


def batched_completion_logprob_gap(
    prompts: Sequence[str],
    positive_completion: str,
    negative_completion: str,
    model_bundle,
    max_length: int,
    batch_size: int,
    intervention_factory=None,
) -> list[dict[str, object]]:
    positive_logprobs = _batched_sequence_logprobs(
        prompts=prompts,
        continuation=positive_completion,
        model_bundle=model_bundle,
        max_length=max_length,
        batch_size=batch_size,
        intervention_factory=intervention_factory,
    )
    negative_logprobs = _batched_sequence_logprobs(
        prompts=prompts,
        continuation=negative_completion,
        model_bundle=model_bundle,
        max_length=max_length,
        batch_size=batch_size,
        intervention_factory=intervention_factory,
    )

    rows: list[dict[str, object]] = []
    for prompt, positive_logprob, negative_logprob in zip(
        prompts, positive_logprobs, negative_logprobs, strict=True
    ):
        rows.append(
            {
                "prompt": prompt,
                "positive_logprob": float(positive_logprob),
                "negative_logprob": float(negative_logprob),
                "logprob_gap": float(positive_logprob - negative_logprob),
                "preferred_completion": (
                    positive_completion.strip()
                    if positive_logprob >= negative_logprob
                    else negative_completion.strip()
                ),
            }
        )
    return rows


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
