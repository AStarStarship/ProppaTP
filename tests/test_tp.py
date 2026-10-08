# Copyright AStarship <https://astarship.net>.
"""Tests for tensor-parallel sharding primitives."""

import torch
import pytest

from proppatp.tp import (
    shard_rows,
    shard_cols,
    RowParallelLinear,
    ColumnParallelLinear,
    all_reduce,
    all_gather,
)


def test_shard_rows_splits_evenly():
    """Row sharding should split out_features into tp_world slices."""
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    tp_world = 2

    shard0 = shard_rows(weight, 0, tp_world)
    shard1 = shard_rows(weight, 1, tp_world)

    assert shard0.shape == (2, 6)
    assert shard1.shape == (2, 6)
    # Shard 0 should be rows 0-1, shard 1 should be rows 2-3.
    assert torch.equal(shard0, weight[0:2])
    assert torch.equal(shard1, weight[2:4])


def test_shard_cols_splits_evenly():
    """Column sharding should split in_features into tp_world slices."""
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    tp_world = 2

    shard0 = shard_cols(weight, 0, tp_world)
    shard1 = shard_cols(weight, 1, tp_world)

    assert shard0.shape == (4, 3)
    assert shard1.shape == (4, 3)
    assert torch.equal(shard0, weight[:, 0:3])
    assert torch.equal(shard1, weight[:, 3:6])


def test_row_parallel_linear_shape():
    """RowParallelLinear output should have out_per_rank features."""
    layer = RowParallelLinear(
        in_features=8,
        out_features=16,
        tp_rank=0,
        tp_world=2,
        bias=True,
    )
    x = torch.randn(2, 8)
    out = layer(x)
    assert out.shape == (2, 8)  # 16 / 2 = 8 per rank.


def test_column_parallel_linear_shape():
    """ColumnParallelLinear should take sharded input, full output."""
    layer = ColumnParallelLinear(
        in_features=8,
        out_features=16,
        tp_rank=0,
        tp_world=2,
        bias=True,
    )
    x = torch.randn(2, 4)  # 8 / 2 = 4 in per rank.
    out = layer(x)
    assert out.shape == (2, 16)


def test_all_reduce_noop_single_process():
    """all_reduce should be a no-op when no process group is active."""
    tensor = torch.randn(4)
    result = all_reduce(tensor)
    assert torch.equal(tensor, result)


def test_all_gather_noop_single_process():
    """all_gather should return the input unchanged in single process."""
    tensor = torch.randn(3)
    result = all_gather(tensor)
    assert torch.equal(tensor, result)


def test_row_parallel_sharding_matches_full():
    """
    Sum of RowParallelLinear outputs across all ranks should match
    the full (unsharded) linear layer.
    """
    in_features = 6
    out_features = 8
    tp_world = 2

    # Full linear.
    full_weight = torch.randn(out_features, in_features)
    full_bias = torch.randn(out_features)

    x = torch.randn(2, in_features)
    full_out = torch.nn.functional.linear(x, full_weight, full_bias)

    # Sharded linear across 2 ranks.
    partial_sum = torch.zeros(2, out_features)
    for rank in range(tp_world):
        layer = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            tp_rank=rank,
            tp_world=tp_world,
            bias=True,
        )
        per_rank = out_features // tp_world
        start = rank * per_rank
        layer.weight.data.copy_(full_weight[start : start + per_rank])
        layer.bias.data.copy_(full_bias[start : start + per_rank])

        out = layer(x)
        partial_sum[:, start : start + per_rank] = out

    # The partial sum should match the full output.
    assert torch.allclose(partial_sum, full_out, atol=1e-5)
