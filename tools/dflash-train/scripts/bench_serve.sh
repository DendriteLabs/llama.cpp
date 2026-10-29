#!/bin/bash
# Plain target vs target + drafter on one GPU, one llama-server at a time.
#
#   TARGET=king.gguf DRAFTER=drafter.gguf PROMPTS=heldout.jsonl scripts/bench_serve.sh
#   TARGET=... DRAFTER=... SRC=llama.cpp/src MODE=longctx scripts/bench_serve.sh
#
# MODE=short (default): spec_bench.py, greedy, identity against the plain run.
# MODE=longctx: longctx_bench.py, sampled, prompts up to 60k tokens built from SRC.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-build/bin/llama-server}
GPU=${GPU:-0}
PORT=${PORT:-8095}
DRAFT_N=${DRAFT_N:-5}
MODE=${MODE:-short}
OUT=${OUT:-bench-out}
: "${TARGET:?set TARGET to the target gguf}" "${DRAFTER:?set DRAFTER to the drafter gguf}"
mkdir -p "$OUT"

if [ "$MODE" = longctx ]; then CTX=${CTX:-70000}; else CTX=${CTX:-8192}; fi

stop() { pkill -f "llama-server.*--port $PORT" || true; sleep 4; }

run() {
    local name=$1; shift
    CUDA_VISIBLE_DEVICES=$GPU "$BIN" -m "$TARGET" -c "$CTX" -np 1 -fa on --jinja -ngl 99 \
        --host 127.0.0.1 --port "$PORT" "$@" > "$OUT/server-$name.log" 2>&1 < /dev/null &
    for _ in $(seq 1 150); do curl -s -m 2 "localhost:$PORT/health" | grep -q ok && break; sleep 2; done
    echo "=== $name"
    if [ "$MODE" = longctx ]; then
        python3 "$HERE/longctx_bench.py" --server "http://127.0.0.1:$PORT" --src "${SRC:?set SRC}" --out "$OUT/longctx-$name.json"
    else
        python3 "$HERE/spec_bench.py" --server "http://127.0.0.1:$PORT" --prompts "${PROMPTS:?set PROMPTS}" \
            --n "${N:-16}" --max-tokens 256 --out "$OUT/short-$name.json" $CMP
    fi
    stop
}

stop
CMP=""
run plain
CMP="--compare $OUT/short-plain.json"
run drafter -md "$DRAFTER" --spec-type draft-dflash --spec-draft-n-max "$DRAFT_N"
