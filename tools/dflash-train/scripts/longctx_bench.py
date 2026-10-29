#!/usr/bin/env python3
"""Long-context serve benchmark: agent-style prompts (source files + a coding task) at several lengths,
sampled, single stream, against one llama-server with or without a drafter.

    python3 longctx_bench.py --server http://localhost:8095 --src path/to/llama.cpp/src --out plain.json
    python3 longctx_bench.py ... --out drafter.json        # same prompts, server started with -md

Prompts depend only on --src, the lengths and the seeds, so two runs see the same inputs. Prints per
length the decode tokens per second (server timings, generation phase only) and the draft acceptance.
The server needs -c at least as large as the longest length plus --max-tokens.
"""
import argparse
import json
import random
import urllib.request
from pathlib import Path

TASKS = [
    "Find a function above that could crash on an empty input and write a patch that fixes it, with a short explanation.",
    "Summarize what the files above do, then propose one refactor and show the code for it.",
    "Write a unit test for one non-trivial function above. Show the full test file.",
    "Pick one function above and rewrite it to be faster. Explain the change and show the new code.",
]


def post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def build(files, target_tokens, seed, tokenize):
    """Whole files in a seeded order until the token budget; the last one is trimmed to fill it."""
    rng = random.Random(seed)
    order = files[:]
    rng.shuffle(order)
    parts, total = [], 0
    for f in order:
        text = f.read_text(errors="replace")
        block = f"### File: {f.name}\n```cpp\n{text}\n```\n"
        n = len(tokenize(block))
        if total + n > target_tokens:
            if total == 0 or target_tokens - total > 2000:
                keep = int(len(text) * max(0.0, (target_tokens - total - 50) / n))
                parts.append(f"### File: {f.name}\n```cpp\n{text[:keep]}\n```\n")
            break
        parts.append(block)
        total += n
    return "You are working in this repository. Here are the relevant files.\n\n" + "".join(parts) + "\n" + TASKS[seed % len(TASKS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--src", required=True, help="directory of .cpp/.h files to build the prompts from")
    ap.add_argument("--lengths", default="2000,8000,16000,32000,60000")
    ap.add_argument("--n", type=int, default=4, help="prompts per length")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def tokenize(s):
        return post(f"{a.server}/tokenize", {"content": s})["tokens"]

    files = sorted(p for p in Path(a.src).rglob("*") if p.suffix in (".cpp", ".h") and p.stat().st_size > 2000)
    rows = []
    for length in [int(x) for x in a.lengths.split(",")]:
        for i in range(a.n):
            msg = build(files, length, 1000 * length + i, tokenize)
            r = post(f"{a.server}/v1/chat/completions", {
                "messages": [{"role": "user", "content": msg}], "max_tokens": a.max_tokens,
                "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "seed": i})
            t = r["timings"]
            row = {"len": length, "i": i, "prompt_n": t.get("prompt_n"), "predicted_n": t["predicted_n"],
                   "tok_s": t["predicted_per_second"], "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")}
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))

    for length in sorted({r["len"] for r in rows}):
        rs = [r for r in rows if r["len"] == length]
        tps = sum(r["predicted_n"] for r in rs) / sum(r["predicted_n"] / r["tok_s"] for r in rs)
        dn = sum(r["draft_n"] or 0 for r in rs)
        da = sum(r["draft_acc"] or 0 for r in rs)
        prompt = sum(r["prompt_n"] or 0 for r in rs) // len(rs)
        acc = f"{da / dn:.3f}" if dn else "-"
        print(f"len {length:6d} prompt~{prompt:6d} tok/s {tps:6.1f} accept {acc}")


if __name__ == "__main__":
    main()
