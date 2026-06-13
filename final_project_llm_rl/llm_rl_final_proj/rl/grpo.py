from __future__ import annotations

from typing import Dict

import torch

from llm_rl_final_proj.models.logprobs import compute_per_token_logprobs, masked_mean_per_row, approx_kl_from_logprobs
from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch, iter_minibatches


class GRPO(RLAlgorithm):
    """GRPO update with a PPO-style clipped surrogate over completion tokens."""

    name = "grpo"

    def update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        # del model, optimizer, rollout, grad_accum_steps
        # (student): implement one GRPO training iteration.
        # The intended structure is:
        #   1. loop over PPO epochs,
        #   2. iterate over rollout minibatches,
        #   3. recompute token log-probabilities under the current policy,
        #   4. form PPO ratios against mb.old_logprobs,
        #   5. apply token-level clipping with the sequence-level GRPO averaging used in this codebase,
        #   6. add KL regularization against mb.ref_logprobs,
        #   7. handle gradient accumulation / clipping / optimizer steps,
        #   8. return the logged metrics expected by the training script.
        
        model.train()
        device = next(model.parameters()).device
        optimizer.zero_grad(set_to_none=True)
        microbatch_count = 0

        for ep in range(self.cfg.ppo_epochs):
            Rollouts = iter_minibatches(rollout, 
                                        self.cfg.minibatch_size,
                                        device=device)
            for mb in Rollouts:
                new_logprobs = compute_per_token_logprobs(model, 
                                                      mb.input_ids,
                                                      mb.attention_mask)
                PPO_ratios = torch.exp(new_logprobs - mb.old_logprobs) # size: [B, L - 1]
                obj1 = PPO_ratios * mb.advantages.reshape(-1, 1)
                obj2 = torch.clamp(
                    PPO_ratios, 
                    1 - self.cfg.clip_eps, 
                    1 + self.cfg.clip_eps) * mb.advantages.reshape(-1, 1)
                loss_1 = torch.mean(masked_mean_per_row(torch.min(obj1, obj2), mb.completion_mask))
                loss_2 = - self.cfg.kl_coef * approx_kl_from_logprobs(
                    new_logprobs,
                    mb.ref_logprobs,
                    mb.completion_mask
                )
                loss = - (loss_1 + loss_2) # to use the GD optimizer, change the sign of the original objective.

                (loss / grad_accum_steps).backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.cfg.max_grad_norm
                ).item())
                microbatch_count += 1

                if microbatch_count % grad_accum_steps != 0:
                    continue

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        metrics = {
            "train/in_sample_loss": float(loss.item()),
            "train/grad_norm": grad_norm
        }
        return metrics
