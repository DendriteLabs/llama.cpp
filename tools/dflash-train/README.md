# llama-dflash-train

Fine-tunes a DFlash / DFlash 2 drafter against a quantized target (a "king") inside llama.cpp, so the
drafter learns the king's own hidden states and answers. Output is a GGUF the server loads with
`--spec-type draft-dflash`, unchanged. Companion plan: `quantedo/dflash-king4-finetune-plan.md`.

## How it trains

- The king runs through the normal llama API; the features are the same per-layer inputs the
  speculative driver captures at serve time (`llama_set_embeddings_layer_inp`, the drafter's
  `target_layers`).
- The drafter's forward is built here from its own tensors with ggml ops, mirroring
  `src/models/dflash.cpp`: encoder (fc + RMS norm), context keys/values from the encoded features,
  block rows `[tok[a], MASK x (B-1)]` at positions `a..a+B-1`, causal attention over features of
  positions `< a` within the sliding window, bidirectional inside the block, DFlash 2 dynamic conv,
  the king's `output.weight` as the head.
- Loss: cross-entropy on rows `1..B-1` against `tok[a+1..a+B-1]`, weighted `exp(-(i-1)/gamma)`,
  many random anchors per sequence batched as independent blocks. AdamW through ggml-opt, cosine
  schedule with warmup, gradient accumulation over `--seqs-per-step` sequences.
- Trainable: draft layers, encoder, conv tensors. Frozen: the king, embeddings, head, and the
  DFlash 2 selector tensors unless `--train-selector`.
- Data parallel: one replica per device (every GPU by default, `--devices CUDA0,CUDA1,...` to pick),
  each with its own king, drafter copy and ggml-opt context. Every replica trains `--seqs-per-step`
  sequences, the gradient accumulators are averaged on the host, clipped to `--clip` (global norm),
  and the same AdamW step runs on every replica, so the copies stay identical (re-broadcast every
  `--sync-every` steps as a drift guard). ggml-opt is used for forward + backward only; AdamW is the
  tool's own `ggml_opt_step_adamw` graph. The host all-reduce moves 2.4 GB per replica per step,
  about a second on eight cards; an NCCL path would remove that if it ever matters.
- ggml changes for this: `GGML_OP_CONCAT` got a backward rule (`ggml/src/ggml.c`); ggml-opt's
  gradients-only build type no longer falls into the optimizer-step assertion and `ggml_opt_grad_acc`
  works for it (`ggml/src/ggml-opt.cpp`).
  CUDA has every backward kernel; Metal lacks OUT_PROD, RMS_NORM_BACK and CROSS_ENTROPY_LOSS_BACK,
  so on a Mac those run on the CPU through the scheduler (slow but correct).

## Build

```
cmake -B build -DGGML_CUDA=ON && cmake --build build --target llama-dflash-train -j
```

## Checks before training

```
# 1. gradients: ggml backward vs finite differences on a toy drafter (CPU, seconds); the second form also
#    checks the all-reduce and that replicas stay identical after an AdamW step
python3 tools/dflash-train/scripts/make_toy_drafter.py toy.gguf
build/bin/llama-dflash-train -md toy.gguf --gradcheck --devices CPU
build/bin/llama-dflash-train -md toy.gguf --gradcheck --devices CPU,CPU

# 2. the training graph equals the serving graph: same king features, same block, logits compared
build/bin/llama-dflash-train -m king.gguf -md dflash2-27b-f32.gguf --heldout heldout.jsonl --selfcheck 20

# 3. baseline acceptance of the untouched drafter on 100 held-out samples (must match the server's number)
build/bin/llama-dflash-train -m king.gguf -md dflash2-27b-f32.gguf --heldout heldout.jsonl --check 100
```

## Data

jsonl, one sample per line, token ids of the king's tokenizer, answers generated greedily by the king:

```
{"id": "...", "source": "code", "engine": "cuda", "prompt_tokens": [...], "response_tokens": [...]}
```

`scripts/make_prompts.py` draws the prompts from the ingot evaluation corpus with the eval service's
own domain- and language-balanced draw (held-out = an eval-sized draw, training = a larger one minus
its ids). `scripts/dflash_gen.py` produces the shards from a prompts file through one or more
`llama-server` instances (chat template with tool schemas and tokenizer applied by the server, answers
sampled as the evals sample the king, or `--greedy`, token ids returned). Shards are
written as `shard-NNNNN.jsonl.part` and renamed when complete; the trainer only reads complete shards
and rescans the directory every `--rescan-every` steps, so training can start while generation runs.

## Train

```
build/bin/llama-dflash-train -m king.gguf -md dflash2-27b-f32.gguf \
    --train shards/ --heldout heldout.jsonl --out ckpt/ \
    --anchors 64 --seqs-per-step 8 --seq-max 3072 --gamma 4 \
    --lr 1.5e-4 --lr-min 1e-5 --warmup 0.04 --wd 0.01 --epochs 2 \
    --eval-every 500 --eval-samples 100 --save-every 500 --metrics-port 9108
```

Checkpoints: `ckpt/ckpt-NNNNNN.gguf` and `ckpt/latest.gguf`, tensors in the drafter file's types
(`--save-type f32` keeps f32). Serve: `llama-server -m king.gguf -md ckpt/latest.gguf --spec-type draft-dflash`.
Requantize for shipping: `llama-quantize ckpt/latest.gguf drafter-q8.gguf q8_0`.

Generation on several cards: one `llama-server` per card with a wide `-np` (pick it with
`llama-batched-bench -npl 1,8,16,32,64,128`), all of them listed in `--servers`; the client splits the
prompts between them. Training can start on the first complete shards while the rest generate.

Memory per micro-step is dominated by the attention scores `[T + A*B, A*B, heads]` and the logits
`[vocab, A*(B-1)]`; lower `--anchors` if it does not fit. The king's `output.weight` is held in f32
(5 GB for a 248k vocab) because ggml's mat-mul backward (`ggml_out_prod`) wants f32.

## Watching it

`--metrics-port` serves Prometheus text at `/metrics`; `grafana/docker-compose.yml` starts Prometheus +
Grafana with a provisioned dashboard (loss, held-out acceptance per position, mean accepted length,
learning rate, step time, tokens per second). Point `grafana/prometheus.yml` at the trainer's host.
`ckpt/train.jsonl` holds the same numbers per step.

## First result (king IV, 2026-10-28)

Lean run, 12k prompts from the ingot corpus, 30M tokens, 52 minutes on 8 RTX PRO 6000: held-out mean
accepted tokens per 7-row block 3.01 -> 3.50; served single-stream on one card at draft length 5:
233 tok/s against 138 plain (1.69x) and 186 with the shipped z-lab drafter (1.35x). `scripts/spec_bench.py`
is the serving benchmark. Greedy outputs of any speculative run in this fork differ from plain decode on
a few near-tied tokens (batched verification rounds differently); plain decode itself is deterministic.

## Serving benchmarks

`scripts/bench_serve.sh` starts the target alone and then with the drafter on one GPU and runs one of two
benchmarks against each:

```bash
# held-out chat prompts, greedy, identity against plain
TARGET=king.gguf DRAFTER=drafter-q8.gguf PROMPTS=heldout/shard-00000.jsonl scripts/bench_serve.sh
# agent-style prompts up to 60k tokens built from source files, sampled at temperature 0.6
TARGET=king.gguf DRAFTER=drafter-q8.gguf SRC=src MODE=longctx scripts/bench_serve.sh
```

`BIN`, `GPU`, `PORT`, `DRAFT_N` (default 5), `CTX` and `OUT` override the defaults. `scripts/longctx_bench.py`
can also run on its own against any server.

Results with the full-run drafter (Q8_0, draft length 5, single stream):

| target, benchmark | plain tok/s | drafter tok/s | speedup | acceptance |
|---|---|---|---|---|
| king IV, held-out greedy | 135.5 | 243.9 | 1.80x | 55.8% |
| king XVI, held-out greedy | 135.8 | 245.2 | 1.81x | 56.4% |
| king XVI, 2k sampled | 134.7 | 175.2 | 1.30x | 35.5% |
| king XVI, 60k sampled | 98.3 | 138.2 | 1.41x | 30.7% |

The drafter trained on king IV transfers to later kings unchanged, and acceptance holds well past the
3k-token training sequences. Q8_0 serves faster than BF16 at the same acceptance.

## Not done

- NCCL all-reduce (the host one is fine at this size).
- The DFlash 2 selector is frozen by default; its training objective is not public.
- Labels are uploaded as dense one-hot matrices per micro-step (one `[vocab, anchors]` block per
  position). The loss is one cross-entropy per block position, combined with the position weights;
  ggml's cross-entropy backward assumes one-hot rows, so the weights cannot go into the labels
  (that was a real bug, caught by `--gradcheck`).
