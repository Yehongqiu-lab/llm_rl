from __future__ import annotations

from typing import Dict

import torch

from llm_rl_final_proj.models.logprobs import compute_per_token_logprobs, masked_mean_per_row, approx_kl_from_logprobs
from llm_rl_final_proj.rl.base import RLAlgorithm
from llm_rl_final_proj.rollout.rollout_buffer import RolloutBatch, iter_minibatches


class GSPO(RLAlgorithm):
    """Sequence-level clipped surrogate using geometric-mean likelihood ratios."""

    name = "gspo"

    def update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        # del model, optimizer, rollout, grad_accum_steps
        # (student): implement GSPO.
        # The main change relative to GRPO is that you should aggregate token log-ratios into
        # one sequence-level ratio before applying PPO-style clipping.
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
                PPO_ratios_seq_level = torch.exp(
                                        masked_mean_per_row(new_logprobs - mb.old_logprobs,
                                                            mask=mb.completion_mask)
                                        ) # size: [B, 1]
                obj1 = PPO_ratios_seq_level * mb.advantages.reshape(-1, 1)
                obj2 = torch.clamp(
                    PPO_ratios_seq_level, 
                    1 - self.cfg.clip_eps, 
                    1 + self.cfg.clip_eps) * mb.advantages.reshape(-1, 1)
                loss_1 = torch.mean(torch.min(obj1, obj2))
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