#!/usr/bin/env python3
"""Draw training and held-out prompts for the drafter from the ingot evaluation corpus, with the eval
service's own draw (domain-balanced, spread evenly over languages, agent trajectories cut before a random
assistant turn), so the drafter trains on what the evals score.

    PYTHONPATH=/path/to/ingot/src python3 make_prompts.py --tokenizer qwen38-meta --out prompts/ \
        --per-domain 600 --agent 1500 --heldout-per-domain 40 --heldout-agent 150 --max-prompt-tokens 2048

Needs: the ingot repo's src on PYTHONPATH (eval_service.dataset, common.models), transformers for the
tokenizer, and network access to Hugging Face (or --corpus-dir with the data/chungus-*.parquet files).
Writes prompts/heldout.jsonl and prompts/train.jsonl: {"id", "source" (domain), "language", "messages",
"tools"?}. The held-out draw uses the eval's quotas; the training draw excludes its ids.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="dir with the king's tokenizer (Qwen3.8-27B)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus-dir", default="", help="dir holding data/chungus-*.parquet; default: download")
    ap.add_argument("--repo", default="dendriteholdings/ingot_corpus")
    ap.add_argument("--revision", default="027dc1d65bcbbd329f9e48ec7c83ab21e0124add")
    ap.add_argument("--split", default="chungus")
    ap.add_argument("--per-domain", type=int, default=600, help="training prompts per domain")
    ap.add_argument("--agent", type=int, default=1500, help="training prompts for each of swe and terminal")
    ap.add_argument("--heldout-per-domain", type=int, default=40, help="the eval's own quota")
    ap.add_argument("--heldout-agent", type=int, default=150)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048,
                    help="longest rendered prompt kept (the trainer's --seq-max minus the answer cap)")
    ap.add_argument("--seed", default="dflash-king4")
    args = ap.parse_args()

    from eval_service import config, dataset  # noqa: E402  (ingot/src on PYTHONPATH)
    from transformers import AutoTokenizer  # noqa: E402

    if args.corpus_dir:
        root = Path(args.corpus_dir)
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(args.repo, revision=args.revision, repo_type="dataset",
                                      allow_patterns=f"data/{args.split}-*.parquet"))
    files = tuple(sorted(root.glob(f"data/{args.split}-*.parquet")))
    if not files:
        sys.exit(f"no {args.split} parquet files under {root}")
    print(f"{len(files)} parquet files, reading the index (this holds the corpus in memory)", flush=True)
    index = dataset.load_index(files)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def fits(sample):
        text = tok.apply_chat_template(sample.messages, tools=sample.tools, tokenize=False, add_generation_prompt=True)
        if len(text.encode()) > args.max_prompt_tokens * 16:   # far too long to be worth tokenizing
            return False
        return len(tok.encode(text, add_special_tokens=False)) <= args.max_prompt_tokens

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(samples, path):
        with open(path, "w") as fh:
            for s in samples:
                rec = {"id": s.id, "source": s.domain, "language": s.language, "messages": s.messages}
                if s.tools:
                    rec["tools"] = s.tools
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # held-out: the eval's quotas, a fixed seed
    config.DOMAIN_SAMPLES = {"swe": args.heldout_agent, "terminal": args.heldout_agent}
    held = dataset._draw_index(index, f"{args.seed}/heldout", args.heldout_per_domain, fits)
    write(held, out / "heldout.jsonl")
    held_ids = {s.id for s in held}
    print(f"heldout: {len(held)} prompts -> {out / 'heldout.jsonl'}", flush=True)

    # training: larger quotas, another seed, held-out ids removed
    config.DOMAIN_SAMPLES = {"swe": args.agent, "terminal": args.agent}
    train = [s for s in dataset._draw_index(index, f"{args.seed}/train", args.per_domain, fits) if s.id not in held_ids]
    write(train, out / "train.jsonl")
    by_domain = {}
    for s in train:
        by_domain[s.domain] = by_domain.get(s.domain, 0) + 1
    print(f"train: {len(train)} prompts -> {out / 'train.jsonl'}", flush=True)
    for d, n in sorted(by_domain.items()):
        print(f"  {d:12s} {n}")


if __name__ == "__main__":
    main()
