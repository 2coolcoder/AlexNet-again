"""Accuracy metrics and running averages."""

from __future__ import annotations

import torch


class AverageMeter:
    """Running mean of a scalar, weighted by sample count."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)):
    """Number of correct predictions in the top-k, one entry per k.

    Counts (not percentages) are returned so they can be summed across ranks
    before dividing.
    """
    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return [correct[:, : min(k, maxk)].any(dim=1).sum() for k in topk]
