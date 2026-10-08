# Copyright AStarship <https://astarship.net>.
"""Decode stage: autoregressive generation with TP=2 and paged KV cache.

Runs on the two 5060 Ti cards. Takes the KV cache produced by the
pre-fill stage and generates tokens one at a time (or in small
batches), applying tensor-parallel all-reduce at each layer.
"""

import time
import torch
import torch.nn as nn


class DecodeStage:
    """Autoregressive decode stage with TP and paged KV cache.

    Args:
        model: The transformer model (TP-sharded).
        tokenizer: HuggingFace tokenizer.
        device: torch device string.
        max_new_tokens: Default max tokens to generate.
        paged_kv: Whether to use paged KV cache (vs contiguous).
        page_size: Tokens per KV page (for paged cache).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cpu",
        max_new_tokens: int = 512,
        paged_kv: bool = True,
        page_size: int = 16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.paged_kv = paged_kv
        self.page_size = page_size

        # KV cache state.
        self.kv_cache_ = None
        self.seq_len_ = 0
        self.generated_ids_: list[int] = []

        # Timing.
        self.last_decode_time_s_ = 0.0
        self.last_decode_tokens_ = 0

    def reset(self, kv_cache=None, seq_len: int = 0) -> None:
        """Reset decode state, optionally seeding with a pre-fill KV cache.

        Args:
            kv_cache: KV cache from the pre-fill stage (or None to start fresh).
            seq_len: Current sequence length (from pre-fill).
        """
        self.kv_cache_ = kv_cache
        self.seq_len_ = seq_len
        self.generated_ids_ = []

    def generate(
        self,
        input_ids: torch.Tensor,
        kv_cache=None,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> list[int]:
        """Generate tokens autoregressively.

        Args:
            input_ids: Prompt token IDs [1, seq_len] (or last token [1, 1]
                       if kv_cache is provided from pre-fill).
            kv_cache: Pre-fill KV cache to continue from (optional).
            max_new_tokens: Override max tokens for this call.
            temperature: Sampling temperature.
            top_k: Top-k filtering (0 = disabled).
            top_p: Nucleus sampling threshold (1.0 = disabled).

        Returns:
            List of generated token IDs (excluding the prompt).
        """
        if kv_cache is not None:
            self.reset(kv_cache=kv_cache, seq_len=input_ids.shape[-1])
        else:
            self.reset(seq_len=input_ids.shape[-1])

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        start = time.monotonic()
        current_ids = input_ids
        new_tokens = []

        for _ in range(max_new_tokens):
            with torch.no_grad():
                output = self.model(
                    current_ids,
                    past_key_values=self.kv_cache_,
                    use_cache=True,
                )
            self.kv_cache_ = output.past_key_values
            self.seq_len_ += 1

            logits = output.logits[:, -1, :].squeeze(0)  # [vocab_size]
            token_id = self._sample(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            new_tokens.append(token_id)
            current_ids = torch.tensor(
                [[token_id]], dtype=torch.long, device=self.device
            )

            # Stop on EOS.
            if token_id == self._eos_token_id():
                break

        self.last_decode_time_s_ = time.monotonic() - start
        self.last_decode_tokens_ = len(new_tokens)
        self.generated_ids_ = new_tokens
        return new_tokens

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> int:
        """Sample a token from the logits.

        Args:
            logits: [1, vocab_size] logits.
            temperature: Sampling temperature.
            top_k: Top-k filtering (0 = off).
            top_p: Nucleus threshold (1.0 = off).

        Returns:
            Sampled token ID (int).
        """
        if temperature <= 0.0:
            # Greedy: argmax, no sampling.
            return logits.argmax(dim=-1).item()

        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            indices_to_remove = (
                logits
                < torch.topk(logits, top_k)[0][..., -1, None]
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = (
                cumulative_probs > top_p
            )
            # Keep the first token always.
            sorted_indices_to_remove[..., 1:] = (
                sorted_indices_to_remove[..., 1:]
            )
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()
        return token

    def _eos_token_id(self) -> int:
        """Get the EOS token ID from the tokenizer config."""
        if hasattr(self.tokenizer, "eos_token_id"):
            return self.tokenizer.eos_token_id
        return -1  # Never stop.

    def decode_text(self, tokens: list[int], skip_special: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            tokens: Token IDs to decode.
            skip_special: Whether to skip special tokens.

        Returns:
            Decoded text.
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special)

    @property
    def decode_time_s(self) -> float:
        """Wall-clock seconds for the last decode run."""
        return self.last_decode_time_s_

    @property
    def decode_tokens(self) -> int:
        """Number of tokens in the last decode run."""
        return self.last_decode_tokens_

    def tpot_ms(self) -> float:
        """Time per output token in milliseconds.

        Returns:
            ms/token (0.0 if no decode has run yet).
        """
        if self.last_decode_time_s_ <= 0.0 or self.last_decode_tokens_ == 0:
            return 0.0
        return (self.last_decode_time_s_ * 1000.0) / self.last_decode_tokens_

    def throughput_tps(self) -> float:
        """Tokens per second for the last decode run.

        Returns:
            tokens/s (0.0 if no decode has run yet).
        """
        if self.last_decode_time_s_ <= 0.0:
            return 0.0
        return self.last_decode_tokens_ / self.last_decode_time_s_
