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
        selected = input_ids[attention_mask == 0]
        if selected.min() == selected.max():
            pad_token_id = selected.min().item()
        else:
            raise ValueError("input_ids where attention_mask==0 shoud be filled with pad_token_id!")
        logits = model(input_ids[:, :-1])
        logprobs = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids[:, 1:].view(-1),
                        ignore_index=pad_token_id, reduction='none')
        logprobs = logprobs.view(B, L - 1)
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
    L = prompt_input_len
    assert L == input_ids.shape[1]
    completion_mask = attention_mask[:, 1:].to(dtype=torch.float)
    assert completion_mask.shape[1] == L - 1
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
    return masked_sum_per_row(estimator, mask).mean()
    