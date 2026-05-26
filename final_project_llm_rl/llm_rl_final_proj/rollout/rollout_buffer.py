from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:
    # del batch, minibatch_size, shuffle, generator, device

    # (student): iterate over the rollout in minibatches, optionally shuffling the row indices,
    # and yield RolloutBatch objects containing the selected subset.
    class RolloutBatchIterator:
        def __init__(self, batch, minibatch_size, shuffle, generator, device):
            self._batch = batch
            self._index = 0
            self._minibatch_size = minibatch_size
            self._shuffle = shuffle
            self._generator = generator
            self._device = device
        
        def __iter__(self):
            return self

        def __next__(self):
            if self._index + self._minibatch_size <= len(self._batch):
                minibatch = self._batch[self._index:self._index + self._minibatch_size]
                self._index += self._minibatch_size
                if self._shuffle:
                    shuffled_idx = torch.randperm(self._minibatch_size, 
                                                  generator=self._generator,
                                                  device=self._device)
                    minibatch = minibatch[shuffled_idx, :]
                return minibatch
            raise StopIteration
    
    if device:
        batch = batch.to(device)
    return RolloutBatchIterator(batch, minibatch_size, shuffle, generator, device)

    