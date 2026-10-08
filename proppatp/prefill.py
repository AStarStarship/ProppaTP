# Copyright AStarship <https://astarship.net>.
"""Pre-fill stage: prompt encoding on the 5070 Ti.

Runs on the pre-fill card (5070 Ti) which also hosts the
Nemotron 3 4B nvFP4 aux model. Takes a prompt, runs the encoder
and initial transformer layers, and produces the KV cache that
the decode stage (TP=2 on 5060 Tis) continues from.
"""

import time
import torch
import torch.nn as nn


class PrefillStage:
    """Pre-fill (prompt encoding) stage.

    Args:
        model: The transformer model (or the pre-fill portion).
        tokenizer: HuggingFace tokenizer.
        device: torch device for the pre-fill card.
        aux_model: Optional 4B aux model for routing/scoring.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cpu",
        aux_model: nn.Module | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.aux_model = aux_model
        self.last_prefill_time_s_ = 0.0
        self.last_prefill_tokens_ = 0

    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and run the pre-fill pass over the prompt.

        Args:
            prompt: The input prompt text.

        Returns:
            Tuple of (input_ids, kv_cache) where:
              input_ids — tensor [1, seq_len]
              kv_cache  — list of (K, V) pairs per layer,
                          each K/V shaped [1, num_kv_heads, seq_len, head_dim]
        """
        start = time.monotonic()

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids,
                use_cache=True,
            )

        kv_cache = output.past_key_values
        self.last_prefill_time_s_ = time.monotonic() - start
        self.last_prefill_tokens_ = input_ids.shape[-1]

        return input_ids, kv_cache

    def run_aux(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the 4B aux model over the encoded prompt.

        Used for routing, relevance scoring, or auxiliary inference.

        Args:
            input_ids: Tokenized prompt [1, seq_len].

        Returns:
            Aux model logits [1, seq_len, aux_vocab_size].
        """
        if self.aux_model is None:
            raise RuntimeError("No aux model loaded on pre-fill stage.")
        with torch.no_grad():
            output = self.aux_model(input_ids)
        return output.logits

    @property
    def prefill_time_s(self) -> float:
        """Wall-clock seconds for the last pre-fill pass."""
        return self.last_prefill_time_s_

    @property
    def prefill_tokens(self) -> int:
        """Number of tokens in the last pre-filled prompt."""
        return self.last_prefill_tokens_

    def throughput_tps(self) -> float:
        """Tokens per second for the last pre-fill pass.

        Returns:
            tokens/s (0.0 if no pre-fill has run yet).
        """
        if self.last_prefill_time_s_ <= 0.0:
            return 0.0
        return self.last_prefill_tokens_ / self.last_prefill_time_s_
