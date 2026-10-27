# Metal decode speedups for Hadamard-rotated qwen35 models, part 2 (October 2026)

Follow-up to [metal-qwen35-decode.md](metal-qwen35-decode.md) (commit `49475161c`). Same machine, models and method: Mac Studio, Apple M3 Ultra (80-core GPU), [dendriteholdings/Ingot-IV-27B-Q2](https://huggingface.co/dendriteholdings/Ingot-IV-27B-Q2), `llama-bench -p 512 -n 128 -ngl 99 -fa 1`, other services idle on the GPU.

## Results

| Model | Decode tg128, upstream `e3de715` | Decode tg128, part 1 | Decode tg128, part 2 | Prefill pp512 |
|---|---:|---:|---:|---:|
| Qwen3.8-27B PQ2_0 | 44.5 tok/s | 56.6 tok/s | 58.6 tok/s (+32% over upstream) | 433 tok/s, unchanged |
| Qwen3.8-27B PTQ1_0 | 43.1 tok/s | 52.8 tok/s | 54.8 tok/s (+27% over upstream) | 368 tok/s, unchanged |

GPU busy per token about 17 ms to about 16.5 ms, of which 8.9 ms is the weight read.

Correctness: greedy `llama-completion --temp 0` output is byte-identical to part 1 and to the unfused path (`GGML_METAL_FUSION_DISABLE=1`), and the PQ2_0 MUL_MAT backend op tests pass.

## Changes

| # | Change | Files | What and why |
|---|---|---|---|
| 1 | Residual ADD + RMS_NORM + MUL + Hadamard kernel in float4 | `ggml-metal/kernels/norm.metal`, `ggml-metal-device.cpp` | The kernel works in float4 with the same thread-to-element split as the stock rms_norm kernel, so its sum of squares is the same bits. The transform runs its first two stages inside the float4, five by lane shuffles and three through threadgroup memory. |
| 2 | GDN tail as one 256-thread threadgroup per 1024-wide output block | `ggml-metal/kernels/norm.metal`, `ggml-metal-ops.cpp`, `ggml-metal-impl.h`, `ggml-metal-device.cpp` | Was one 1024-thread threadgroup per token row. Each thread now finds its source element through the inverse head permutation, so a simdgroup holds one head and the per-head sum is one simd_sum, and a row's traffic is spread over six cores instead of one. Its cost fell from about 0.75 ms to 0.25 ms per token. |

## Where the time goes now

Measured by swapping each fused kernel for an empty one: the residual-add + norm + Hadamard kernel about 1.5 ms per token, the GDN state update 0.75 ms, the GDN tail 0.25 ms, the GLU + Hadamard 0.3 ms, the conv front 0.1 ms; the dispatches themselves about 0.4 ms. A small kernel run as one threadgroup per row costs several microseconds per 20 KB array it touches, far above the chip's bandwidth.

## Tried and dropped

- Norm kernel: a transposed Hadamard, splitting the row over threadgroups, a two-dispatch per-block version (per-block sums, then per-block normalise) and hoisted loads were each neutral or slower.
- Folding the GDN L2 norm into the conv front: neutral (that kernel already overlaps the z projection) and not bit-identical.
- Speculative decoding with the DFlash 2 draft (`dflash2-27b-q8.gguf`, about 50% acceptance at 4 draft tokens): 24.8 tok/s at first, 43.8 tok/s with a PQ2_0 mat-vec that decodes each weight once for two columns, still below plain decode. Verifying 5 tokens costs about 3x one token, because PQ2_0 decode is ALU-bound at this width: a 2-column kernel (more columns spill registers) and simdgroup-matrix kernels (float, half, split-K) all stayed at 140 us or more for the 17408x5120 FFN matrix at n=5, against about 85 us needed to beat plain decode. Removed from the code.
