# Copyright AStarship <https://astarship.net>.
"""End-to-end inference smoke test.

Uses a tiny CPU model to validate the pre-fill → decode pipeline
without requiring a GPU or a large model download.
"""

import pytest
import torch
import torch.nn as nn


class TinyModel(nn.Module):
    """Minimal transformer-like model for smoke testing."""

    def __init__(self, vocab_size=100, hidden=16, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = torch.relu(layer(x))
        logits = self.head(x)
        # Fake KV cache: just return a sentinel.
        kv = None
        if use_cache:
            kv = ((torch.zeros(1), torch.zeros(1)),) * len(self.layers)
        return type("Out", (), {"logits": logits, "past_key_values": kv})


class FakeTokenizer:
    """Minimal tokenizer stub for testing."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.eos_token_id = 2

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        # Simple char-to-id mapping for testing.
        ids = [ord(c) % self.vocab_size for c in text]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor}

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(t % 128) for t in tokens)


@pytest.fixture
def tiny_setup():
    """Create a tiny model + tokenizer for testing."""
    model = TinyModel()
    tokenizer = FakeTokenizer()
    return model, tokenizer


def test_prefill_encode_prompt(tiny_setup):
    """Pre-fill should produce input_ids and a KV cache sentinel."""
    from proppatp.prefill import PrefillStage

    model, tokenizer = tiny_setup
    stage = PrefillStage(model=model, tokenizer=tokenizer, device="cpu")

    input_ids, kv_cache = stage.encode_prompt("hello")
    assert input_ids.shape == (1, 5)
    assert kv_cache is not None
    assert stage.prefill_tokens == 5
    assert stage.prefill_time_s > 0.0


def test_decode_generate(tiny_setup):
    """Decode should generate the requested number of tokens."""
    from proppatp.decode import DecodeStage

    model, tokenizer = tiny_setup
    stage = DecodeStage(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_new_tokens=10,
    )

    tokens = stage.generate(
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=5,
        temperature=0.0,  # Greedy for determinism.
    )
    # Should generate up to 5 tokens (may stop early on EOS=2).
    assert 1 <= len(tokens) <= 5
    assert stage.decode_tokens == len(tokens)
    assert stage.decode_time_s > 0.0
    assert stage.throughput_tps() > 0.0


def test_prefill_to_decode_pipeline(tiny_setup):
    """Full pre-fill → decode pipeline should work end-to-end."""
    from proppatp.prefill import PrefillStage
    from proppatp.decode import DecodeStage

    model, tokenizer = tiny_setup
    prefill = PrefillStage(model=model, tokenizer=tokenizer, device="cpu")
    decode = DecodeStage(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_new_tokens=8,
    )

    # Pre-fill.
    input_ids, kv_cache = prefill.encode_prompt("Hi")
    assert input_ids.shape[-1] == 2

    # Decode using pre-fill KV cache.
    tokens = decode.generate(
        input_ids,
        kv_cache=kv_cache,
        max_new_tokens=4,
        temperature=0.0,
    )
    assert len(tokens) >= 1

    # Decode text should be a string.
    text = decode.decode_text(tokens)
    assert isinstance(text, str)


def test_kv_cache_size_estimate():
    """KV cache estimation should match manual calculation."""
    from proppatp.model import estimate_kv_cache_bytes

    # 2 layers, 4 KV heads, head_dim 8, seq 100, fp16 (2 bytes).
    # Per tensor: 2 * 4 * 8 * 100 * 2 = 12800
    # K + V = 25600
    expected = 2 * (2 * 4 * 8 * 100 * 2)
    actual = estimate_kv_cache_bytes(
        num_layers=2,
        num_kv_heads=4,
        head_dim=8,
        seq_len=100,
        dtype_bytes=2,
    )
    assert actual == expected
