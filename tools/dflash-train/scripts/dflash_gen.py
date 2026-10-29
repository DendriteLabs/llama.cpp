#!/usr/bin/env python3
"""Regenerate answers with the king through llama-server and write jsonl shards for llama-dflash-train.

    python3 dflash_gen.py --prompts prompts.jsonl --out shards/ --servers http://gpu6:8080,http://mac:8012 \
        --concurrency 16 --max-tokens 1024 --shard-size 1000

prompts.jsonl: one per line, {"id": ..., "source": "code", "messages": [...], "tools": [...]?}
               or {"id": ..., "source": ..., "text": "<already templated prompt>"} (make_prompts.py writes these).
Output shard line: {"id", "source", "language", "engine", "prompt_tokens", "response_tokens", "finish"}.
The prompt goes through the server's own chat template (/apply-template, with the tools) and tokenizer
(/tokenize); the answer is sampled the way the evals sample the king (temperature 1.0, top-p 0.95, top-k 20)
unless --greedy, and returned as token ids, so the trainer never re-tokenizes anything.
Several servers split the prompt ids between them round-robin; each line records which server (engine) wrote it.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.request


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def engine_name(server):
    try:
        props = get(server + "/props")
        return props.get("engine", "") or server
    except Exception:
        return server


def worker(server, engine, q, out_q, args):
    while True:
        item = q.get()
        if item is None:
            return
        try:
            if "text" in item:
                prompt_text = item["text"]
            else:
                body = {"messages": item["messages"]}
                if item.get("tools"):
                    body["tools"] = item["tools"]
                prompt_text = post(server + "/apply-template", body)["prompt"]
            toks = post(server + "/tokenize", {"content": prompt_text, "add_special": False})["tokens"]
            toks = [t if isinstance(t, int) else t["id"] for t in toks]
            sampling = ({"temperature": 0.0, "top_k": 1, "samplers": ["top_k"]} if args.greedy else
                        {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                         "samplers": ["top_k", "top_p", "temperature"], "seed": abs(hash(str(item.get("id")))) % (2**31)})
            r = post(server + "/completion", {
                "prompt": toks,
                "n_predict": args.max_tokens,
                "return_tokens": True,
                "cache_prompt": True,
                "n_probs": 0,
                **sampling,
            })
            resp = r.get("tokens") or []
            finish = r.get("stop_type", "") or ("limit" if r.get("truncated") else "eos")
            out_q.put({"id": item.get("id"), "source": item.get("source", ""), "language": item.get("language", ""), "engine": engine,
                       "prompt_tokens": toks, "response_tokens": resp, "finish": finish})
        except Exception as e:  # noqa: BLE001
            item["_tries"] = item.get("_tries", 0) + 1
            sys.stderr.write(f"[{server}] {item.get('id')} try {item['_tries']}: {e}\n")
            if item["_tries"] < 3:
                q.put(item)
            else:
                out_q.put({"id": item.get("id"), "source": item.get("source", ""), "language": item.get("language", ""),
                           "engine": engine, "prompt_tokens": [], "response_tokens": [], "finish": f"error: {e}"})
            time.sleep(2)
        finally:
            q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--servers", required=True, help="comma-separated llama-server base URLs")
    ap.add_argument("--concurrency", type=int, default=8, help="requests in flight per server (= its -np)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--start-shard", type=int, default=0)
    ap.add_argument("--greedy", action="store_true", help="argmax answers instead of the evals' sampling")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    servers = [s.rstrip("/") for s in args.servers.split(",") if s.strip()]
    done_ids = set()
    for f in os.listdir(args.out):
        if f.endswith(".jsonl"):
            with open(os.path.join(args.out, f)) as fh:
                for line in fh:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except Exception:  # noqa: BLE001
                        pass
    q = queue.Queue()
    n_total = 0
    with open(args.prompts) as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("id") in done_ids:
                continue
            q.put(item)
            n_total += 1
    print(f"{n_total} prompts to do ({len(done_ids)} already in {args.out})", flush=True)

    out_q = queue.Queue()
    threads = []
    for s in servers:
        eng = engine_name(s)
        for _ in range(args.concurrency):
            th = threading.Thread(target=worker, args=(s, eng, q, out_q, args), daemon=True)
            th.start()
            threads.append(th)

    shard = args.start_shard
    while os.path.exists(os.path.join(args.out, f"shard-{shard:05d}.jsonl")):
        shard += 1
    n_in_shard, n_done, tokens, t0 = 0, 0, 0, time.time()
    fh = open(os.path.join(args.out, f"shard-{shard:05d}.jsonl.part"), "w")
    while n_done < n_total:
        rec = out_q.get()
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        n_in_shard += 1
        n_done += 1
        tokens += len(rec["response_tokens"])
        if n_done % 50 == 0:
            el = time.time() - t0
            print(f"{n_done}/{n_total} answers, {tokens/1e6:.2f}M answer tokens, {tokens/el:.0f} tok/s", flush=True)
        if n_in_shard >= args.shard_size:
            fh.close()
            os.rename(os.path.join(args.out, f"shard-{shard:05d}.jsonl.part"), os.path.join(args.out, f"shard-{shard:05d}.jsonl"))
            shard += 1
            n_in_shard = 0
            fh = open(os.path.join(args.out, f"shard-{shard:05d}.jsonl.part"), "w")
    fh.close()
    if n_in_shard:
        os.rename(os.path.join(args.out, f"shard-{shard:05d}.jsonl.part"), os.path.join(args.out, f"shard-{shard:05d}.jsonl"))
    else:
        os.remove(os.path.join(args.out, f"shard-{shard:05d}.jsonl.part"))
    for _ in threads:
        q.put(None)
    print("done", flush=True)


if __name__ == "__main__":
    main()
