# Copyright AStarship <https://astarship.net>.
"""Tensor-parallel sharding primitives for ProppaTP.

Provides row-parallel and column-parallel linear layers plus the
collective communication primitives (all-reduce, all-gather) needed
to shard a transformer across multiple GPUs over PCIe.
"""

import torch
import torch.nn as nn
import torch.distributed as dist


def is_initialized() -> bool:
    """Check whether the default process group is initialized.

    Returns:
        True if a process group is active, False otherwise.
    """
    return dist.is_available() and dist.is_initialized()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """In-place all-reduce over the default process group.

    On a single process (CPU dev, TP=1) this is a no-op.

    Args:
        tensor: Tensor to reduce. Must be contiguous.

    Returns:
        The same tensor, now containing the reduced result.
    """
    if not is_initialized():
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor along dim 0.

    Args:
        tensor: Local shard.

    Returns:
        Concatenated tensor from all ranks.
    """
    if not is_initialized():
        return tensor
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def shard_rows(weight: torch.Tensor, tp_rank: int, tp_world: int) -> torch.Tensor:
    """Shard a weight matrix along rows (out_features).

    For a linear layer W of shape [out, in], row-parallel splits W
    into tp_world horizontal slices. Each rank keeps slice
    [tp_rank * out_per_rank : (tp_rank+1) * out_per_rank].

    Args:
        weight: Full weight tensor [out_features, in_features].
        tp_rank: This rank's index (0-based).
        tp_world: Total number of tensor-parallel ranks.

    Returns:
        Local weight shard [out_per_rank, in_features].
    """
    out_features = weight.shape[0]
    per_rank = out_features // tp_world
    start = tp_rank * per_rank
    return weight[start : start + per_rank].contiguous()


def shard_cols(weight: torch.Tensor, tp_rank: int, tp_world: int) -> torch.Tensor:
    """Shard a weight matrix along columns (in_features).

    For a linear layer W of shape [out, in], column-parallel splits W
    into tp_world vertical slices. Each rank keeps slice
    [ :, tp_rank * in_per_rank : (tp_rank+1) * in_per_rank ].

    Args:
        weight: Full weight tensor [out_features, in_features].
        tp_rank: This rank's index (0-based).
        tp_world: Total number of tensor-parallel ranks.

    Returns:
        Local weight shard [out_features, in_per_rank].
    """
    in_features = weight.shape[1]
    per_rank = in_features // tp_world
    start = tp_rank * per_rank
    return weight[:, start : start + per_rank].contiguous()


class RowParallelLinear(nn.Module):
    """Linear layer with row-parallel weight sharding.

    Y = X * W^T + b, where W is sharded along rows (out_features).
    Each rank computes a partial result over its out_features slice.
    An all-reduce over the partial results produces the full output.

    This is the preferred TP strategy for PCIe-only topologies:
    exactly one all-reduce per layer.

    Args:
        in_features: Input dimension.
        out_features: Total output dimension (before sharding).
        tp_rank: This rank's TP index.
        tp_world: Total TP world size.
        bias: Whether the layer has a bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_rank: int,
        tp_world: int,
        bias: bool = True,
    ):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.out_per_rank = out_features // tp_world

        self.weight = nn.Parameter(
            torch.empty(self.out_per_rank, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_rank))
        else:
            self.register_parameter("bias", None)

        # Note: in production the weight is loaded from a checkpoint
        # and sliced via shard_rows(). Here we just allocate.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the row-parallel linear.

        Args:
            x: Input tensor [*, in_features].

        Returns:
            Output tensor [*, out_per_rank] (local shard).
            Caller is responsible for all_reduce if the full
            [*, out_features] result is needed.
        """
        return nn.functional.linear(x, self.weight, self.bias)


class ColumnParallelLinear(nn.Module):
    """Linear layer with column-parallel weight sharding.

    Y = X * W^T + b, where W is sharded along columns (in_features).
    Each rank computes the full output dimension but only over its
    slice of the input. Requires an all-gather on the input before
    the matmul, or an all-reduce on the output after.

    Less preferred over RowParallelLinear for PCIe topologies because
    it adds communication volume.

    Args:
        in_features: Total input dimension (before sharding).
        out_features: Output dimension.
        tp_rank: This rank's TP index.
        tp_world: Total TP world size.
        bias: Whether the layer has a bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_rank: int,
        tp_world: int,
        bias: bool = True,
    ):
        super().__init__()
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.in_per_rank = in_features // tp_world

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_per_rank)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the column-parallel linear.

        Args:
            x: Input tensor [*, in_per_rank] (already sharded).

        Returns:
            Output tensor [*, out_features].
        """
        return nn.functional.linear(x, self.weight, self.bias)


def tp_rank() -> int:
    """Get this process's TP rank (0 if not in a distributed group)."""
    if not is_initialized():
        return 0
    return dist.get_rank()


def tp_world_size() -> int:
    """Get the TP world size (1 if not in a distributed group)."""
    if not is_initialized():
        return 1
    return dist.get_world_size()
