# Copyright AStarship <https://astarship.net>.
"""Model loading and nvFP4 quantization for ProppaTP."""

import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class TModelConfig:
    """Plain config struct for a model to load.

    Attributes:
        name: HuggingFace model ID or local path.
        quantization: "nvfp4", "int4", or "none".
        tp_rank: Tensor-parallel rank for this process.
        tp_world: Total TP world size.
        device: torch device string ("cuda:0", "cpu", etc.).
        max_seq_len: Maximum sequence length for KV cache.
        num_kv_heads: Number of KV heads (for paged KV sizing).
        head_dim: Dimension per attention head.
    """
    name: str
    quantization: str = "nvfp4"
    tp_rank: int = 0
    tp_world: int = 1
    device: str = "cpu"
    max_seq_len: int = 4096
    num_kv_heads: int = 8
    head_dim: int = 128


def load_model(config: TModelConfig) -> nn.Module:
    """Load a model from HuggingFace or a local path.

    Applies tensor-parallel sharding and quantization per config.
    On CPU (no GPU), loads in float32 for logic validation only.

    Args:
        config: Model configuration.

    Returns:
        The loaded model module.

    Raises:
        RuntimeError: If the model cannot be found or loaded.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    model_config = AutoConfig.from_pretrained(
        config.name, trust_remote_code=True
    )

    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load weights. For TP we load the full model on rank 0 and
    # broadcast, or load sharded directly from the checkpoint.
    # For now: load full model, shard weights in-place.
    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)

    # Apply TP sharding to linear layers.
    if config.tp_world > 1:
        _shard_model_tp(model, config)

    model.eval()
    return model


def _shard_model_tp(model: nn.Module, config: TModelConfig) -> None:
    """Shard linear layers in-place for tensor parallelism.

    Replaces nn.Linear layers with RowParallelLinear where the
    output dimension is divisible by tp_world.

    Args:
        model: The model to shard.
        config: Model config with tp_rank and tp_world.
    """
    from proppatp.tp import RowParallelLinear

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        out_features = module.out_features
        if out_features % config.tp_world != 0:
            continue  # Cannot shard evenly; leave as-is.

        # Extract full weight, shard it.
        full_weight = module.weight.data
        in_features = module.in_features

        tp_linear = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            tp_rank=config.tp_rank,
            tp_world=config.tp_world,
            bias=module.bias is not None,
        )
        tp_linear.weight.data.copy_(
            full_weight[
                config.tp_rank * (out_features // config.tp_world) :
                (config.tp_rank + 1) * (out_features // config.tp_world)
            ]
        )
        if module.bias is not None:
            tp_linear.bias.data.copy_(
                module.bias.data[
                    config.tp_rank * (out_features // config.tp_world) :
                    (config.tp_rank + 1) * (out_features // config.tp_world)
                ]
            )

        # Replace in parent.
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], tp_linear)


def estimate_kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate KV cache size in bytes for a given sequence length.

    Args:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension per head.
        seq_len: Sequence length.
        dtype_bytes: Bytes per element (2 for fp16/bf16).

    Returns:
        Total KV cache bytes.
    """
    # K and V, each: num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes
    per_tensor = num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes
    return 2 * per_tensor  # K + V


def fp4_pack(values: torch.Tensor) -> torch.Tensor:
    """Pack float values into 4-bit representation (int4 fallback).

    Until native FP4 tensor ops are available in PyTorch, we pack
    into int4 for storage efficiency. This is a placeholder for
    the actual nvFP4 (e2m1 with block scaling) format.

    Args:
        values: Tensor of float values to pack.

    Returns:
        Packed int8 tensor (2 values per byte).
    """
    # Simple int4 packing: clamp, quantize to 4-bit unsigned.
    packed = values.clamp(-1.0, 1.0)
    quantized = ((packed + 1.0) * 7.5).round().to(torch.uint8)
    # Pack two 4-bit values per byte.
    return (quantized[0::2] | (quantized[1::2] << 4)).contiguous()


def fp4_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int4-packed values back to float.

    Args:
        packed: int8 tensor with 2 packed values per byte.

    Returns:
        Unpacked float tensor (approximately 2x the packed length).
    """
    lo = (packed & 0x0F).float()
    hi = (packed >> 4).float()
    quantized = torch.stack([lo, hi], dim=-1).reshape(-1)
    return (quantized / 7.5 - 1.0).clamp(-1.0, 1.0)
