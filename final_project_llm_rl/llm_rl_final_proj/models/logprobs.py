from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_per_token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
    """Returns log p(x_t | x_<t) for t in [1, L-1]. Shape: [B, L-1]."""
    with torch.set_grad_enabled(enable_grad):
        # (student): run the causal LM, align logits with the next-token targets,
        # and return per-token log-probabilities of the observed tokens.
        # Hint: use F.cross_entropy with reduction='none' for memory efficiency.
        B, L = input_ids.size()
        logits = model(
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1]
                ).logits
        targets = input_ids[:, 1:].clone()
        targets[attention_mask[:, 1:] == 0] = -100
        logprobs = - F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                   targets.reshape(-1),
                        ignore_index=-100, reduction='none')
        logprobs = logprobs.reshape(B, L - 1)
        return logprobs

def build_completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_input_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Mask over per-token positions [B, L-1], selecting completion tokens only."""
    del pad_token_id
    # (student): build a float mask of shape [B, L-1] that selects only completion tokens.
    # Be careful about the one-token shift between logits[:, :-1] and input_ids[:, 1:].
    B, L = input_ids.shape
    completion_mask = torch.zeros((B, L - 1), dtype=torch.float)
    completion_mask[:, prompt_input_len - 1: ] = 1.0
    return completion_mask


def masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum()

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)

def masked_sum_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1)

def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def approx_kl_from_logprobs(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Positive KL proxy from sampled actions.

    Uses estimator: exp(delta) - delta - 1 where delta = log p_ref(a) - log p_new(a).
    """
    del eps, log_ratio_clip
    # (student): implement the sampled-token KL proxy used throughout the codebase.
    # You should mask out non-completion positions and return a scalar batch mean.
    assert ref_logprobs.shape == new_logprobs.shape
    delta = ref_logprobs - new_logprobs
    estimator = torch.exp(delta) - delta - 1
    return torch.mean(masked_mean_per_row(estimator, mask))
    