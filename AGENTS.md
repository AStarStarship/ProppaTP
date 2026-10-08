# Agent Rules for ProppaTP

**ProppaTP = Prompt Preprocessing + Tensor Parallelism engine.**
Pre-fill and TP decode server for LLM inference.

## Kanban board

This project belongs to the **AStarship** organization; its Kanban board
slug is `astarship`. Always target `astarship` explicitly.

## 1. Build, Lint, and Test Commands

### Environment
- **Python:** 3.12, venv at `.venv/`
- **Activate:** `source .venv/bin/activate`
- **CPU torch:** installed (2.14.0+cpu) — this VM has no NVIDIA GPU.
- **GPU torch:** install when 5070 Ti / 5060 Ti are passed through:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

### Run
- **Server (CPU dev):** `python -m proppatp --tp 1 --device cpu`
- **Server (GPU, TP=2):** `python -m proppatp --tp 2 --device cuda`

### Test
- **All tests:** `python -m pytest tests/ -v`
- **Single test:** `python -m pytest tests/test_tp.py::test_shard_rows -v`

### Lint
- No automated linter configured. Follow the style rules in section 2.

## 2. Code Style

Python in the AStarship fleet follows Chimera+ naming applied to Python:

- **Module / class names:** CamelCase (`PrefillStage`, `DecodeServer`).
- **Functions / methods:** snake_case, verb-first (`load_model`, `shard_rows`).
- **Local variables:** snake_case.
- **Class attributes (mutable):** snake_case with trailing underscore
  (`self.tp_rank_`).
- **Class attributes (immutable, set in `__init__`):** snake_case, no
  trailing underscore (`self.num_layers`).
- **Module-level constants:** UPPER_SNAKE_CASE (`MAX_SEQ_LEN`).
- **Type aliases / dataclasses:** CamelCase, `T` prefix for plain structs
  (`TPrefillConfig`), `A` prefix for owning classes (`ADecodeServer`).
- **No stdlib `datetime` for timing** — use `time.monotonic()` or
  `torch.cuda.Event` for GPU-accurate timing.

### Docstrings
NumPy-style or Google-style, one per public function. Include:
- `Args:` / `Returns:` / `Raises:` sections.
- Tensor shapes in the Returns section, e.g.
  `Returns: logits — tensor [batch, seq_len, vocab_size]`.

### File layout
- `proppatp/` — the package (inference engine).
- `tests/` — pytest tests.
- New modules start with:
  ```python
  # Copyright AStarship <https://astarship.net>.
  """One-line module docstring."""
  ```

## 3. Important Directories

- `proppatp/tp.py` — Tensor-parallel sharding (row/col parallel linear,
  all-reduce / all-gather primitives).
- `proppatp/model.py` — Model loading, nvFP4 quantization hooks.
- `proppatp/prefill.py` — Pre-fill stage (prompt encoding on 5070 Ti).
- `proppatp/decode.py` — Decode stage (autoregressive, TP=2 on 5060 Tis,
  paged KV cache).
- `proppatp/server.py` — HTTP inference server.

## 4. AI Behavior Rules

- **Verify before implementing:** read the relevant module before editing.
- **CPU-first development:** this VM has no GPU. All logic must be testable
  on CPU. GPU code paths are guarded with
  `if torch.cuda.is_available():` and tested on the GPU box only.
- **No fabricated benchmarks:** never report tokens/s or latency numbers
  that were not actually measured. If a test is skipped (no GPU), say so.
- **nvFP4 specifics:** Gemma 4 12B and Nemotron 3 4B use NVIDIA's FP4
  format (4-bit mantissa + shared exponent per 16-element block). Use
  `torch.float4_e2m1fn_x2` when available; fall back to int4 packing
  until CUDA 12.6 + torch 2.4+ supports native FP4 ops.
- **TP invariants:** after any tensor-parallel operation, the output shape
  must be identical to the single-GPU reference. Test this explicitly in
  `tests/test_tp.py`.

## 5. Hardware Constraints

- **Pre-fill card:** 1x RTX 5070 Ti (16 GB) — Blackwell, compute 12.0.
  Hosts Nemotron 3 4B nvFP4 aux model (~12 GB) + prompt pre-fill.
- **Decode cards:** 2x RTX 5060 Ti (16 GB each) — Blackwell, compute 12.0.
  Run Gemma 4 12B nvFP4 in TP=2 with paged KV cache.
- **Interconnect:** PCIe Gen 5 x16 per card (no NVLink on 5060 Ti).
  All-reduce over PCIe is the bottleneck — keep the all-reduce volume
  minimal by using row-parallel linear (one all-reduce per layer) rather
  than column-parallel (which would need all-gather + all-reduce).
- **Model sizes:**
  - Gemma 4 12B nvFP4 ≈ 6 GB weights. At TP=2: ~3 GB per card,
    ~13 GB free for KV cache + activations per card.
  - Nemotron 3 4B nvFP4 ≈ 3 GB weights + KV cache on the 5070 Ti.

## 6. Prohibited

- Do not install NVIDIA drivers or CUDA on this VM before a card is
  physically passed through.
- Do not commit `.venv/`, `__pycache__/`, or `*.pyc`.
- Do not use `torch.distributed` without first confirming NCCL or
  Gloo backend is available on the target hardware.
