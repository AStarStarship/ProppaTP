# Copyright AStarship <https://astarship.net>.
"""HTTP inference server for ProppaTP.

Exposes a minimal JSON API for prompt pre-fill and decode:
  POST /v1/completions   — full pre-fill + decode
  GET  /v1/health        — liveness probe
  GET  /v1/stats         — timing stats (TTFT, TPOT, throughput)
"""

import argparse
import json
import sys
import time

import torch

from proppatp.model import TModelConfig, load_model
from proppatp.prefill import PrefillStage
from proppatp.decode import DecodeStage


class InferenceServer:
    """ProppaTP inference server.

    Args:
        model_name: HuggingFace model ID or local path.
        aux_model_name: Optional 4B aux model (Nemotron 3 4B).
        tp: Tensor-parallel world size (1, 2, or 4).
        device: "cpu" or "cuda".
        quantization: "nvfp4", "int4", or "none".
        max_seq_len: Maximum sequence length.
        max_new_tokens: Default max decode tokens.
    """

    def __init__(
        self,
        model_name: str,
        aux_model_name: str | None = None,
        tp: int = 1,
        device: str = "cpu",
        quantization: str = "nvfp4",
        max_seq_len: int = 4096,
        max_new_tokens: int = 512,
    ):
        self.model_name = model_name
        self.aux_model_name = aux_model_name
        self.tp = tp
        self.device = device
        self.quantization = quantization
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens

        self.prefill_stage_: PrefillStage | None = None
        self.decode_stage_: DecodeStage | None = None
        self.tokenizer_ = None
        self.start_time_ = time.monotonic()

    def init(self) -> None:
        """Load models and initialize pre-fill/decode stages.

        Must be called before serving. On CPU this loads a small
        reference model for logic validation. On GPU it loads the
        real model with TP sharding.

        Raises:
            RuntimeError: If model loading fails.
        """
        from transformers import AutoTokenizer

        self.tokenizer_ = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Main model (Gemma 4 12B or test model).
        config = TModelConfig(
            name=self.model_name,
            quantization=self.quantization,
            tp_rank=0,  # Single-process for now; multi-process TP is a TODO.
            tp_world=self.tp,
            device=self.device,
            max_seq_len=self.max_seq_len,
        )
        model = load_model(config)

        # Aux model (Nemotron 3 4B) — only if specified and available.
        aux_model = None
        if self.aux_model_name:
            aux_config = TModelConfig(
                name=self.aux_model_name,
                quantization=self.quantization,
                tp_rank=0,
                tp_world=1,
                device=self.device,
            )
            aux_model = load_model(aux_config)

        self.prefill_stage_ = PrefillStage(
            model=model,
            tokenizer=self.tokenizer_,
            device=self.device,
            aux_model=aux_model,
        )
        self.decode_stage_ = DecodeStage(
            model=model,
            tokenizer=self.tokenizer_,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
        )

    def complete(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> dict:
        """Run a full pre-fill + decode pass.

        Args:
            prompt: Input text.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k (0 = off).
            top_p: Nucleus (1.0 = off).

        Returns:
            Dict with "text", "tokens", "ttft_s", "tpot_ms", "throughput_tps".
        """
        if self.prefill_stage_ is None or self.decode_stage_ is None:
            raise RuntimeError("Server not initialized. Call init() first.")

        # Pre-fill.
        input_ids, kv_cache = self.prefill_stage_.encode_prompt(prompt)
        ttft_s = self.prefill_stage_.prefill_time_s

        # Decode.
        tokens = self.decode_stage_.generate(
            input_ids,
            kv_cache=kv_cache,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        text = self.decode_stage_.decode_text(tokens)

        return {
            "text": text,
            "tokens": tokens,
            "num_tokens": len(tokens),
            "ttft_s": round(ttft_s, 6),
            "tpot_ms": round(self.decode_stage_.tpot_ms(), 3),
            "throughput_tps": round(self.decode_stage_.throughput_tps(), 2),
        }

    def health(self) -> dict:
        """Liveness probe.

        Returns:
            Dict with "status" and "uptime_s".
        """
        return {
            "status": "ok",
            "uptime_s": round(time.monotonic() - self.start_time_, 1),
            "tp": self.tp,
            "device": self.device,
        }

    def stats(self) -> dict:
        """Timing statistics.

        Returns:
            Dict with pre-fill and decode timing.
        """
        result = {"uptime_s": round(time.monotonic() - self.start_time_, 1)}
        if self.prefill_stage_:
            result["prefill"] = {
                "time_s": self.prefill_stage_.prefill_time_s,
                "tokens": self.prefill_stage_.prefill_tokens,
                "tps": round(self.prefill_stage_.throughput_tps(), 2),
            }
        if self.decode_stage_:
            result["decode"] = {
                "time_s": self.decode_stage_.decode_time_s,
                "tokens": self.decode_stage_.decode_tokens,
                "tpot_ms": round(self.decode_stage_.tpot_ms(), 3),
                "tps": round(self.decode_stage_.throughput_tps(), 2),
            }
        return result


def _print_json(obj: dict) -> None:
    """Print a dict as JSON to stdout.

    Args:
        obj: The dict to serialize.
    """
    print(json.dumps(obj, indent=2), flush=True)


def main() -> None:
    """CLI entry point: python -m proppatp.

    For now this runs a single-shot completion from the command line.
    A full HTTP server (uvicorn) will be added in a later commit.
    """
    parser = argparse.ArgumentParser(
        description="ProppaTP — Prompt Preprocessing + TP engine"
    )
    parser.add_argument(
        "--model", default="google/gemma-2b",
        help="HuggingFace model ID (default: google/gemma-2b for CPU test)",
    )
    parser.add_argument(
        "--aux-model", default=None,
        help="Aux model ID (e.g. nvidia/NVIDIA-Nemotron-3-4B)",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor-parallel world size (default: 1)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device (default: cpu)",
    )
    parser.add_argument(
        "--quant", default="none", choices=["none", "int4", "nvfp4"],
        help="Quantization (default: none for CPU)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64,
        help="Max tokens to generate (default: 64)",
    )
    parser.add_argument(
        "--prompt", default="Hello, world!",
        help="Prompt text (default: 'Hello, world!')",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=4096,
        help="Max sequence length (default: 4096)",
    )
    args = parser.parse_args()

    server = InferenceServer(
        model_name=args.model,
        aux_model_name=args.aux_model,
        tp=args.tp,
        device=args.device,
        quantization=args.quant,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_tokens,
    )

    print(f"Loading model: {args.model}", file=sys.stderr)
    server.init()
    print(f"Model loaded. Device: {args.device}, TP: {args.tp}", file=sys.stderr)

    result = server.complete(args.prompt)
    _print_json(result)


if __name__ == "__main__":
    main()
