#!/usr/bin/env python3
"""Serve-time benchmark: the same prompts through a llama-server with and without a drafter, single stream.

    python3 spec_bench.py --server http://localhost:8090 --prompts heldout.jsonl --n 20 --max-tokens 256 --out plain.json
    python3 spec_bench.py ... --out trained.json --compare plain.json     # greedy identity against the plain run

Prints decode tokens per second (server timings, generation phase only) and, with --compare, whether the
generated token ids are identical to the reference run (they must be at temperature 0).
"""
import argparse
import json
import urllib.request


def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--prompts", required=True, help="jsonl with prompt_tokens (a generated shard) or messages")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default="")
    args = ap.parse_args()

    items = []
    with open(args.prompts) as fh:
        for line in fh:
            if not line.strip():
                continue
            it = json.loads(line)
            if it.get("prompt_tokens"):
                items.append(it)
            if len(items) >= args.n:
                break
    ref = {}
    if args.compare:
        for r in json.load(open(args.compare))["runs"]:
            ref[r["id"]] = r["tokens"]

    runs, gen_tokens, gen_time, prompt_tokens, prompt_time = [], 0, 0.0, 0, 0.0
    identical = 0
    for it in items:
        body = {"prompt": it["prompt_tokens"], "n_predict": args.max_tokens, "return_tokens": True, "cache_prompt": False,
                "n_probs": 0, "temperature": args.temperature}
        if args.temperature == 0.0:
            body.update({"top_k": 1, "samplers": ["top_k"]})
        r = post(args.server + "/completion", body)
        t = r["timings"]
        gen_tokens += t["predicted_n"]; gen_time += t["predicted_ms"] / 1000.0
        prompt_tokens += t["prompt_n"]; prompt_time += t["prompt_ms"] / 1000.0
        toks = r.get("tokens") or []
        same = None
        if it["id"] in ref:
            same = toks == ref[it["id"]]
            identical += same
        runs.append({"id": it["id"], "tokens": toks, "predicted_n": t["predicted_n"], "predicted_ms": t["predicted_ms"],
                     "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"), "identical": same})
        print(f"{it['id'][:48]:48s} gen {t['predicted_n']:4d} tok {t['predicted_n']/(t['predicted_ms']/1000):7.1f} tok/s"
              + (f"  draft {t.get('draft_n')} acc {t.get('draft_n_accepted')}" if t.get("draft_n") is not None else "")
              + ("" if same is None else ("  identical" if same else "  DIFFERENT")), flush=True)
    summary = {"server": args.server, "n": len(runs), "gen_tokens": gen_tokens, "gen_tok_s": gen_tokens / gen_time if gen_time else 0,
               "prompt_tok_s": prompt_tokens / prompt_time if prompt_time else 0, "identical": identical if ref else None}
    dn = sum(r["draft_n"] or 0 for r in runs); da = sum(r["draft_n_accepted"] or 0 for r in runs)
    if dn:
        summary["draft_acceptance"] = da / dn
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "runs": runs}, open(args.out, "w"))


if __name__ == "__main__":
    main()
