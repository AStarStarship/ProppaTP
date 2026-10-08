# ProppaTP

**Pro**mpt **P**reprocessing and **T**ensor **P**arallelism engine.

Pre-fill and tensor-parallel decode server for LLM inference.

## Hardware (current)

| Component | Spec |
|---|---|
| Motherboard | MSI Z890 Aero G (dual PCIe 5.0 x16) |
| Pre-fill GPU | 1x RTX 5070 Ti (16 GB) |
| Decode GPUs | 2x RTX 5060 Ti (16 GB each) |
| Interconnect | PCIe 5.0 x16 per card (no NVLink) |

No NVLink on any of these cards. All-reduce between the two 5060 Tis
goes over PCIe 5.0 x16 — the bottleneck for TP=2. Row-parallel linear
keeps it to one all-reduce per layer.

## Model allocation

| Model | Where | VRAM |
|---|---|---|
| Nemotron 3 4B (nvFP4) | 5070 Ti | ~12 GB |
| Gemma 4 12B (nvFP4) | 2x 5060 Ti, TP=2 | ~3 GB per card |

The 5070 Ti handles pre-fill: prompt encoding, KV-cache warm-up, and the
4B aux model (routing, scoring, or auxiliary inference). The two 5060 Tis
run Gemma 4 12B in tensor-parallel decode with paged KV cache.

## Build / run

```bash
# CPU dev (this VM — no NVIDIA GPU yet):
source .venv/bin/activate
python -m proppatp --tp 1 --device cpu

# GPU box (5070 Ti + 2x 5060 Ti passed through):
source .venv/bin/activate
python -m proppatp --tp 2 --device cuda
```

## Layout

```
ProppaTP/
  requirements.txt       # Python deps
  AGENTS.md              # Agent rules for this project
  README.md              # You are here
  proppatp/
    __init__.py
    __main__.py          # python -m proppatp entry
    server.py            # Inference server (HTTP API)
    tp.py                # Tensor-parallel sharding (row/col parallel, all-reduce)
    prefill.py           # Pre-fill stage (prompt encoding on 5070 Ti)
    decode.py            # Decode stage (autoregressive, TP=2 on 5060 Tis)
    model.py             # Model loading + nvFP4 quantization
  tests/
    test_tp.py           # Tensor-parallel correctness
    test_inference.py    # End-to-end inference smoke test
```

## Current state

- [x] Project initialized, CPU torch 2.14 installed
- [ ] GPU build (cu126 torch) — install when cards are passed through
- [ ] nvFP4 model loading (Gemma 4 12B + Nemotron 3 4B)
- [ ] TP=2 shard + all-reduce over PCIe
- [ ] Paged KV cache
- [ ] Pre-fill on 5070 Ti, decode on 5060 Ti pair
- [ ] HTTP inference server
- [ ] Benchmark harness (tokens/s, TTFT, TPOT)

## GPU timeline

This VM has no NVIDIA GPU. The Z890 Aero G box with 5070 Ti + 2x 5060 Ti
is the target. Do not install the 580 driver on this VM until the cards
are physically passed through. Install NVIDIA driver + CUDA 12.6 only
when a card is available. Until then all work is CPU-only for logic
validation.
