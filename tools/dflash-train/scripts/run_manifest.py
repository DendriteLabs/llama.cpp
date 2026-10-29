#!/usr/bin/env python3
"""Write a record of a training run: which samples it trained on and held out, how the answers were generated
(per shard: when, how many, how many tokens, which engine), per-step and per-epoch timings, totals, evaluations.

    python3 run_manifest.py --shards shards/ --heldout heldout/shard-00000.jsonl --ckpt ckpt-lean/ --gen-log gen-train.log \
        --name lean --out runs/lean/
Writes <out>/manifest.json (everything), <out>/manifest.md (readable summary), <out>/train_ids.txt, <out>/heldout_ids.txt.
"""
import argparse
import glob
import json
import os
import re
import time


def read_jsonl(path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--ckpt", required=True, help="dir with train.jsonl")
    ap.add_argument("--gen-log", default="")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs-per-step", type=int, default=8)
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--shard-limit", type=int, default=0, help="only the first N shards (a run that used a prefix of a growing directory)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    shards = []
    train_ids, by_source, by_engine, errors = [], {}, {}, 0
    shard_files = sorted(glob.glob(os.path.join(args.shards, "shard-*.jsonl")))
    if args.shard_limit:
        shard_files = shard_files[:args.shard_limit]
    for p in shard_files:
        n, ptok, rtok = 0, 0, 0
        for r in read_jsonl(p):
            if r.get("finish", "").startswith("error") or not r.get("prompt_tokens"):
                errors += 1
                continue
            n += 1
            ptok += len(r["prompt_tokens"]); rtok += len(r["response_tokens"])
            train_ids.append(r["id"])
            by_source[r.get("source", "")] = by_source.get(r.get("source", ""), 0) + 1
            by_engine[r.get("engine", "")] = by_engine.get(r.get("engine", ""), 0) + 1
        st = os.stat(p)
        shards.append({"file": os.path.basename(p), "samples": n, "prompt_tokens": ptok, "answer_tokens": rtok,
                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))})
    held_ids, held_tokens = [], 0
    for r in read_jsonl(args.heldout):
        if r.get("prompt_tokens"):
            held_ids.append(r["id"]); held_tokens += len(r["prompt_tokens"]) + len(r["response_tokens"])

    gen = {}
    if args.gen_log and os.path.exists(args.gen_log):
        lines = open(args.gen_log).read().splitlines()
        m = [re.match(r"(\d+)/(\d+) answers, ([\d.]+)M answer tokens, (\d+) tok/s", l) for l in lines]
        m = [x for x in m if x]
        if m:
            last = m[-1]
            gen = {"answers": int(last[1]), "of": int(last[2]), "answer_tokens_m": float(last[3]), "mean_tok_s": int(last[4])}
        st = os.stat(args.gen_log)
        gen["log_last_write"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))

    steps, evals = [], []
    for r in read_jsonl(os.path.join(args.ckpt, "train.jsonl")):
        (evals if "heldout_tau" in r else steps).append(r)
    epochs = {}
    for s in steps:
        e = epochs.setdefault(s["epoch"], {"epoch": s["epoch"] + 1, "steps": 0, "seconds": 0.0, "compute_seconds": 0.0, "king_seconds": 0.0, "first_step": s["step"], "last_step": s["step"]})
        e["steps"] += 1; e["seconds"] += s["seconds"]; e["compute_seconds"] += s.get("compute_seconds", 0.0); e["king_seconds"] += s.get("feat_seconds", 0.0); e["last_step"] = s["step"]
    totals = {
        "optimizer_steps": len(steps),
        "sequences": len(steps) * args.seqs_per_step * args.replicas,
        "sequence_tokens": steps[-1]["tokens"] if steps else 0,
        "anchors_blocks": steps[-1]["anchors"] if steps else 0,
        "wall_seconds": steps[-1]["elapsed"] if steps else 0,
        "mean_step_seconds": sum(s["seconds"] for s in steps) / len(steps) if steps else 0,
        "mean_king_forward_seconds_per_step": sum(s.get("feat_seconds", 0) for s in steps) / len(steps) if steps else 0,
    }
    manifest = {
        "run": args.name, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": {"train_samples": len(train_ids), "train_by_source": by_source, "train_by_engine": by_engine, "generation_errors": errors,
                 "train_prompt_tokens": sum(s["prompt_tokens"] for s in shards), "train_answer_tokens": sum(s["answer_tokens"] for s in shards),
                 "heldout_samples": len(held_ids), "heldout_tokens": held_tokens, "shards": shards, "generation": gen},
        "training": {"totals": totals, "epochs": list(epochs.values()),
                     "evaluations": [{"step": e["step"], "heldout_loss": e["heldout_loss"], "mean_accepted": e["heldout_tau"], "per_position": e["heldout_acc"]} for e in evals],
                     "steps": [{k: s[k] for k in ("step", "epoch", "loss", "gnorm", "lr", "seconds", "tokens", "anchors") if k in s} for s in steps]},
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    # ids are corpus keys (dataset/split/entry_name/language) and an entry_name may hold any character: one JSON string per line
    open(os.path.join(args.out, "train_ids.txt"), "w").write("\n".join(json.dumps(i, ensure_ascii=False) for i in train_ids) + "\n")
    open(os.path.join(args.out, "heldout_ids.txt"), "w").write("\n".join(json.dumps(i, ensure_ascii=False) for i in held_ids) + "\n")
    d, t = manifest["data"], totals
    md = [f"# Run {args.name}", "", "## Data", f"- training samples: {d['train_samples']} ({d['train_prompt_tokens']/1e6:.2f}M prompt tokens, {d['train_answer_tokens']/1e6:.2f}M answer tokens), ids in train_ids.txt",
          f"- held-out samples: {d['heldout_samples']} ({d['heldout_tokens']/1e6:.2f}M tokens), ids in heldout_ids.txt",
          f"- by source: " + ", ".join(f"{k} {v}" for k, v in sorted(by_source.items())),
          f"- by engine: " + ", ".join(f"{k} {v}" for k, v in sorted(by_engine.items())) + f"; generation errors {errors}",
          f"- generation: {gen}" if gen else "- generation: no log", "", "| shard | samples | prompt tokens | answer tokens | finished (UTC) |", "|---|---:|---:|---:|---|"]
    md += [f"| {s['file']} | {s['samples']} | {s['prompt_tokens']} | {s['answer_tokens']} | {s['finished_at']} |" for s in shards]
    md += ["", "## Training", f"- optimizer steps {t['optimizer_steps']}, sequences {t['sequences']}, sequence tokens {t['sequence_tokens']/1e6:.2f}M, blocks (anchors) {t['anchors_blocks']}",
           f"- wall {t['wall_seconds']/60:.1f} min, mean step {t['mean_step_seconds']:.1f}s of which king forward {t['mean_king_forward_seconds_per_step']:.1f}s",
           "", "| epoch | steps | wall s | compute s | king forward s |", "|---|---:|---:|---:|---:|"]
    md += [f"| {e['epoch']} | {e['steps']} | {e['seconds']:.0f} | {e['compute_seconds']:.0f} | {e['king_seconds']:.0f} |" for e in epochs.values()]
    md += ["", "| eval step | held-out loss | mean accepted of 7 | p1 | p7 |", "|---|---:|---:|---:|---:|"]
    md += [f"| {e['step']} | {e['heldout_loss']:.4f} | {e['heldout_tau']:.3f} | {e['heldout_acc'][0]:.3f} | {e['heldout_acc'][-1]:.3f} |" for e in evals]
    open(os.path.join(args.out, "manifest.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
