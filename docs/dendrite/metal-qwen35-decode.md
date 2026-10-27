# Metal decode speedups for Hadamard-rotated qwen35 models (October 2026)

Commit: `49475161c`. Follow-up: [metal-qwen35-decode-2.md](metal-qwen35-decode-2.md).

Single-sequence decode of Hadamard-rotated qwen35 models in the fork's 2-bit formats on Apple Silicon. The weights already streamed at about 93% of the chip's memory bandwidth, so the time left was in the roughly 1,700 small GPU dispatches per token; every change below either fuses neighbouring graph ops into one kernel or stops a kernel from re-reading data it already had.

## Results

Machine: Mac Studio, Apple M3 Ultra (80-core GPU, 512 GB), macOS 27.0.1.

Model: [dendriteholdings/Ingot-IV-27B-Q2](https://huggingface.co/dendriteholdings/Ingot-IV-27B-Q2), files `Ingot-IV-27B-PQ2_0.gguf` and `Ingot-IV-27B-PTQ1_0.gguf`.

Measured with `llama-bench -p 512 -n 128 -ngl 99 -fa 1`, other services idle on the GPU.

| Model | Decode tg128, upstream `e3de715` | Decode tg128, dendrite | Prefill pp512 |
|---|---:|---:|---:|
| Qwen3.8-27B PQ2_0 | 44.5 tok/s | 56.4 tok/s (+27%) | 432 tok/s, unchanged |
| Qwen3.8-27B PTQ1_0 | 43.1 tok/s | 52.8 tok/s (+23%) | 368 tok/s, unchanged |

Dispatches per decode token 1706 to about 1160. GPU busy per token 19.3 ms to about 17 ms, of which 8.9 ms is the weight read.

These changes do not affect CUDA or any other backend: all kernels and fusions are Metal-only, and the two model-file edits were checked on an NVIDIA GPU: greedy output and a teacher-forced hidden-state dump are byte-identical to the unpatched build.

Correctness: greedy `llama-completion --temp 0` output is byte-identical to the legacy paths (`GGML_METAL_FUSION_DISABLE=1 GGML_GDN_STATE_GATHER=1 GGML_METAL_PQ2_0_KS=0 GGML_METAL_PTQ1_0_KS=0`) and to the unpatched `e3de715` build, for both formats, after every step. `test-backend-ops` passes for MUL_MAT on both types, RMS_NORM and SSM_CONV.

## Changes

| # | Change | Files | What it does |
|---|---|---|---|
| 1 | Gated DeltaNet state updated in place for plain decode | `src/models/qwen35.cpp`, `src/models/delta-net-base.cpp` | The fork's "rows mode", where the fused GDN kernel reads each sequence's 3 MB recurrent state from its cache row and writes the new state back in its epilogue, was only on for speculative decoding. Plain decode went through GET_ROWS and CPY of the state, two dispatches per GDN layer, 48 layers. Rows mode is now on for plain decode too, gated to batches with no state relocation (`n_rs == n_seqs`). `GGML_GDN_STATE_GATHER=1` restores the old path. |
| 2 | PQ2_0 mat-vec: split-K, 4 rows x 4 simdgroups, activations staged as float4 | `ggml-metal/kernels/mul_mv.metal`, `ggml-metal-device.cpp` | The shipped kernel gave each simdgroup 8 output rows and re-read the activation vector per simdgroup with scalar loads; on 5120-row projections that left 640 simdgroups in flight on 80 cores. The new kernel gives a threadgroup 4 rows and 4 simdgroups that split the K blocks, stages the activation once per threadgroup and reduces through threadgroup memory. Default on GPUs without the tensor unit (before M5); `GGML_METAL_PQ2_0_KS=<rows>x<simdgroups>` picks a variant, `=0` restores the old kernel. A PTQ1_0 split-K variant with register prefetch exists too (`GGML_METAL_PTQ1_0_KS`, `GGML_METAL_PTQ1_0_PF`). |
| 3 | Residual ADD + RMS_NORM + weight MUL + Hadamard sign flip + 1024-block Walsh-Hadamard transform in one kernel | `ggml-metal/kernels/norm.metal`, `ggml-metal-ops.cpp`, `ggml-metal-device.{h,cpp}` | Before each attention, GDN and FFN block the graph ran five ops, the fork fused the last two. One 1024-thread kernel per token row now writes both the residual stream and the rotated activation. 128 per token. A plain ADD + RMS_NORM + MUL variant covers norms without a rotation. |
| 4 | Conv front of a GDN layer in one kernel | `ggml-metal/kernels/ssm.metal`, `ggml-metal-ops.cpp`, `ggml-metal-impl.h` | CONCAT of the 3 old conv-state columns with the new token, the CPY back into the conv cache, SSM_CONV (4 taps, launched as 10,240 single-thread threadgroups) and SiLU become one thread per channel that also writes the shifted state into the cache row. The CONCAT output is never materialised. 48 per token. |
| 5 | GDN tail in one kernel | `ggml-metal/kernels/norm.metal`, `ggml-metal-ops.cpp` | Per-head RMS_NORM, norm weight, SiLU gate, the CONT that reorders heads from tiled to grouped layout, the sign flip and the Hadamard transform: six graph nodes, four dispatches, now one 1024-thread kernel per token row. 48 per token. |
| 6 | Metal async tensor upload | `ggml-metal/ggml-metal-context.m` | Upstream's scheduler (`63f16c408`) uploads host inputs with `set_tensor_async`. On Metal that aborted on a zero-length input and otherwise allocated a staging buffer, a blit and a command buffer per input per token, costing about 2% decode. Empty uploads return early; uploads into shared (unified-memory) buffers are a memcpy after a synchronize. |

All fusions pattern-match the graph at encode time with `ggml_can_fuse_subgraph_ext`, which also checks that intermediate tensors are not used elsewhere, and fall back to the original ops when the pattern does not match (other models, shapes they do not cover, fusion disabled).

## Tooling

Off by default, each enabled by its environment variable:

| Variable | Effect |
|---|---|
| `GGML_METAL_PROFILE=<n>` | Each graph node in its own encoder with GPU timestamps sampled at encoder boundaries, aggregated per kernel label, table every n graphs. Serialised, so per-kernel numbers are upper bounds. |
| `GGML_METAL_GPUTIME=<n>` | GPU span and busy time per graph from the command buffers' timestamps next to wall time, every n graphs, to tell GPU-bound from CPU-bound. |
| `GGML_METAL_FUSION_DEBUG=2` | Upstream's switch; the new fusions also print why one was declined. |

## How to verify a change

```
# greedy output must match the legacy paths byte for byte
llama-completion -m model.gguf -ngl 99 -fa on -no-cnv --temp 0 -n 128 -p "..." > new.txt
GGML_METAL_FUSION_DISABLE=1 GGML_GDN_STATE_GATHER=1 GGML_METAL_PQ2_0_KS=0 GGML_METAL_PTQ1_0_KS=0 \
  llama-completion -m model.gguf -ngl 99 -fa on -no-cnv --temp 0 -n 128 -p "..." > legacy.txt
cmp new.txt legacy.txt
# then the speed
llama-bench -m model.gguf -p 512 -n 128 -ngl 99 -fa 1
```
