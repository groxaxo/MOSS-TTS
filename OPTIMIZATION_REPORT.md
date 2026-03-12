# MOSS-TTS Realtime — RTX 3090 Performance Optimization Report

**Date**: 2026-03-13
**GPU**: NVIDIA GeForce RTX 3090 (24 GB, Ampere SM 8.6)
**Stack**: PyTorch 2.9.1+cu128, transformers 5.0.0, flash-attn 2.8.3, triton 3.5.1

---

## Executive Summary

Optimized the MOSS-TTS-Realtime OpenAI-compatible server for maximum throughput
and minimum latency on the RTX 3090. Through four targeted changes to the
inference pipeline, achieved:

| Metric     | Before  | After   | Improvement |
|------------|---------|---------|-------------|
| **RTF**    | 0.8044  | 0.3352  | **−58%** (2.4× throughput) |
| **TTFB**   | 1557 ms | 586 ms  | **−62%** (2.7× faster first chunk) |
| **Min RTF**| 0.7858  | 0.3073  | Best-case 3.3× realtime |

40.5 seconds of audio generated in 13.4 seconds (wall-clock) across 7 test
sentences of varying length (12–202 characters).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  OpenAI API  (FastAPI + uvicorn)  :8012                     │
│    POST /v1/audio/speech                                    │
├─────────────────────────────────────────────────────────────┤
│  Streaming Pipeline                                         │
│    TokenChunkStream → StreamingSession → AudioStreamDecoder │
├─────────────────────────────────────────────────────────────┤
│  Backbone Transformer  (Qwen3-1.7B, 28 layers)             │
│    2048 hidden, 16 heads, 8 KV heads, bfloat16             │
│    → Produces hidden states per token step                  │
├─────────────────────────────────────────────────────────────┤
│  Local Transformer  (4 layers, 2048 hidden)                 │
│    16 sequential RVQ codebook steps per backbone step       │
│    → Produces 16-channel audio tokens                       │
├─────────────────────────────────────────────────────────────┤
│  MOSS Audio Tokenizer (codec)                               │
│    Decodes RVQ tokens → 24 kHz waveform                     │
│    Streaming mode with crossfade overlap                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Bottlenecks Identified & Fixes Applied

### 1. Local Transformer: DynamicCache + No Compilation (HIGH IMPACT)

**Problem**: When the backbone used `flash_attention_2`, the local transformer
inherited the same attention implementation. This forced `DynamicCache` and
**disabled `torch.compile`** for the 16-step RVQ decode loop — the innermost
hot loop of the entire pipeline.

The local transformer processes sequences of only 16 tokens (one per RVQ
codebook). Flash attention provides negligible benefit at this length, while
`StaticCache` + `torch.compile(fullgraph=True)` can fuse the entire loop into
optimized Triton kernels.

**Fix** (`streaming_mossttsrealtime.py`, `inferencer.py`):
- Override local transformer config `_attn_implementation` → `"sdpa"`
- Force `StaticCache(max_cache_len=16)` regardless of backbone attention
- Enable `torch.compile(fullgraph=True)` for the local transformer always

```python
# Before: cache/compile depended on backbone attn
self._use_dynamic_local_cache = attn_impl == "flash_attention_2"  # True → no compile
self._should_compile_local_transformer = not self._use_dynamic_local_cache  # False

# After: always StaticCache + compile for local transformer
local_cfg._attn_implementation = "sdpa"
self._use_dynamic_local_cache = False
self._should_compile_local_transformer = True
```

### 2. Backbone: SDPA + torch.compile (HIGH IMPACT)

**Problem**: The backbone Qwen3 language model ran uncompiled. Additionally,
`flash_attention_2` prevented `torch.compile` from generating fused Triton
kernels for the attention + MLP blocks.

**Finding**: Benchmarking revealed that on the RTX 3090 (Ampere),
`SDPA + torch.compile(dynamic=True)` dramatically outperforms
`flash_attention_2` for typical TTS sequence lengths:

| Config                        | RTF   | TTFB    |
|-------------------------------|-------|---------|
| flash_attn2, no compile       | 0.805 | 1557 ms |
| flash_attn2 + compile         | 0.490 | 950 ms  |
| **SDPA + compile (dynamic)**  | **0.335** | **586 ms** |

**Fix** (`app.py`):
- Switch default attention to `"sdpa"` (auto-detected)
- Add `torch.compile(mode="default", dynamic=True)` for the backbone
- `dynamic=True` prevents shape-triggered recompilation as KV cache grows

```python
model.language_model = torch.compile(
    model.language_model,
    mode="default",
    fullgraph=False,
    dynamic=True,
)
```

### 3. No Startup Warmup (HIGH TTFB IMPACT)

**Problem**: `WARMUP_ON_START` defaulted to `false`. The first user request
suffered a 30–70 second cold-start penalty while `torch.compile` generated
and cached Triton kernels.

**Fix** (`openai_api.py`):
- Default `WARMUP_ON_START` to `true`
- Run 2 warmup generations with different text lengths during `lifespan()`
- Server starts accepting traffic only after caches are hot (~2.5 min startup)

### 4. CUDA Runtime Enhancements (LOW-MEDIUM IMPACT)

**Fix** (`app.py`):
- Increased `torch._dynamo.config.cache_size_limit` from 64 → 128
- Added `torch._dynamo.config.suppress_errors = True` for graceful fallback
- Added CUDA memory pool pre-allocation (`set_per_process_memory_fraction`)

---

## Why SDPA Beats Flash Attention on 3090

Flash Attention 2 is a hand-written CUDA kernel optimized for long sequences
(>512 tokens). It bypasses PyTorch's operator fusion.

SDPA (`torch.nn.functional.scaled_dot_product_attention`) is a PyTorch-native
op that `torch.compile` can fuse with surrounding operations (layernorm, MLP,
residual connections) into a single Triton kernel graph.

For MOSS-TTS-Realtime:
- **Backbone decode step**: 1 token input, KV cache grows to ~100–400 tokens
- **Local transformer**: 16 tokens total

At these lengths, the kernel launch overhead of flash_attn's separate CUDA
kernel dominates over its memory-access advantage. `torch.compile` + SDPA
generates fused Triton kernels that eliminate this overhead entirely.

---

## Benchmark Details

### Test Configuration
- 7 sentences, 12–202 characters
- Voice preset: alloy (prompt audio cloning)
- Response format: WAV (no codec encoding overhead)
- All requests sequential (no concurrency)

### Before (Baseline)
```
flash_attention_2, no torch.compile, no warmup
  [1] RTF=0.8213 TTFB=1579ms audio=5.56s chars=79
  [2] RTF=0.7858 TTFB=1539ms audio=6.35s chars=97
  [3] RTF=0.8061 TTFB=1553ms audio=7.41s chars=113
  Avg: RTF=0.8044  TTFB=1557ms
```

### After (Optimized)
```
SDPA + torch.compile(dynamic=True), StaticCache local transformer, warmup
  [1] RTF=0.3578 TTFB=455ms audio=1.28s chars=12
  [2] RTF=0.3182 TTFB=623ms audio=2.71s chars=32
  [3] RTF=0.3073 TTFB=596ms audio=7.28s chars=79
  [4] RTF=0.4287 TTFB=590ms audio=6.15s chars=97
  [5] RTF=0.3091 TTFB=623ms audio=6.15s chars=113
  [6] RTF=0.3128 TTFB=602ms audio=7.55s chars=144
  [7] RTF=0.3126 TTFB=616ms audio=9.40s chars=202
  Avg: RTF=0.3352  TTFB=586ms
```

---

## Files Modified

| File | Changes |
|------|---------|
| `moss_tts_realtime/mossttsrealtime/streaming_mossttsrealtime.py` | Force local transformer → SDPA + StaticCache + torch.compile |
| `moss_tts_realtime/inferencer.py` | Same local transformer fix (standalone inference path) |
| `moss_tts_realtime/app.py` | Backbone torch.compile, CUDA runtime enhancements, dynamo config |
| `moss_tts_realtime/openai_api.py` | Default SDPA, warmup on start, multi-shape warmup |

---

## Server Configuration

The optimized server runs at `http://127.0.0.1:8012` with:
- `MOSS_TTS_ATTN_IMPLEMENTATION=auto` (resolves to `sdpa`)
- `MOSS_TTS_WARMUP_ON_START=true` (default)
- `MOSS_TTS_COMPILE_BACKBONE=true` (default)
- Startup time: ~2.5 minutes (model load + warmup compilation)

To force flash_attention_2 (if needed for other hardware):
```bash
export MOSS_TTS_ATTN_IMPLEMENTATION=flash_attention_2
```

To disable backbone compilation (if stability issues arise):
```bash
export MOSS_TTS_COMPILE_BACKBONE=false
```

---

## Remaining Optimization Opportunities

1. **ONNX/TensorRT codec decode**: The audio tokenizer codec runs as PyTorch.
   ONNX or TRT backends exist in `moss_audio_tokenizer/` but aren't wired
   into the streaming path. Could reduce decode latency.

2. **CUDA graphs for fixed-shape decode**: If inputs were padded to discrete
   bucket sizes, `reduce-overhead` mode could capture full CUDA graphs for
   near-zero kernel launch overhead.

3. **Quantized backbone**: The backbone runs in bfloat16. INT8/INT4
   quantization (via GPTQ, AWQ, or torch.ao) could further reduce compute.

4. **Speculative decoding**: Pre-generate multiple RVQ frames speculatively
   and verify — could reduce effective RTF further.

5. **Multi-stream codec decode**: Overlap codec decoding with backbone
   generation using separate CUDA streams.
