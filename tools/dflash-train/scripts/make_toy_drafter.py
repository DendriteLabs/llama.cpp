#!/usr/bin/env python3
"""Write a tiny DFlash 2 drafter GGUF (with its own token_embd and output) for llama-dflash-train --gradcheck.

    python3 make_toy_drafter.py toy.gguf
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "gguf-py"))
import gguf  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "toy-dflash.gguf"
rng = np.random.default_rng(0)

n_embd, n_layer, n_head, n_kv, hd, n_ff, vocab = 64, 2, 4, 2, 16, 96, 97
block, kernel, group, rank, topk = 4, 2, 16, 8, 4
target_layers = [2, 5, 7]   # inputs to these layers of a toy king; gradcheck feeds random features anyway
mask_id = vocab - 1

w = gguf.GGUFWriter(out, "dflash")
w.add_name("toy-dflash2")
w.add_block_count(n_layer)
w.add_context_length(4096)
w.add_embedding_length(n_embd)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_kv)
w.add_key_length(hd)
w.add_value_length(hd)
w.add_rope_dimension_count(hd)
w.add_rope_freq_base(10000.0)
w.add_layer_norm_rms_eps(1e-6)
w.add_sliding_window(8)
w.add_block_size(block)
w.add_conv_kernel_size(kernel)
w.add_conv_group_size(group)
w.add_selector_rank(rank)
w.add_selector_top_k(topk)
w.add_target_layers(target_layers)
w.add_mask_token_id(mask_id)
w.add_tokenizer_model("none")
w.add_token_list([f"t{i}" for i in range(vocab)])
w.add_file_type(gguf.LlamaFileType.ALL_F32)


def t(name, shape, scale=0.05):
    a = (rng.standard_normal(shape) * scale).astype(np.float32)
    w.add_tensor(name, a)


# ggml shape convention: numpy shape reversed, so [rows=out, cols=in]
t("token_embd.weight", (vocab, n_embd), 0.3)
t("output.weight", (vocab, n_embd), 0.3)
t("fc.weight", (n_embd, len(target_layers) * n_embd), 0.1)
w.add_tensor("enc.output_norm.weight", (1.0 + 0.1 * rng.standard_normal(n_embd)).astype(np.float32))
w.add_tensor("output_norm.weight", (1.0 + 0.1 * rng.standard_normal(n_embd)).astype(np.float32))
for il in range(n_layer):
    p = f"blk.{il}."
    w.add_tensor(p + "attn_norm.weight", (1.0 + 0.1 * rng.standard_normal(n_embd)).astype(np.float32))
    t(p + "attn_q.weight", (n_head * hd, n_embd))
    t(p + "attn_k.weight", (n_kv * hd, n_embd))
    t(p + "attn_v.weight", (n_kv * hd, n_embd))
    t(p + "attn_output.weight", (n_embd, n_head * hd))
    w.add_tensor(p + "attn_q_norm.weight", (1.0 + 0.1 * rng.standard_normal(hd)).astype(np.float32))
    w.add_tensor(p + "attn_k_norm.weight", (1.0 + 0.1 * rng.standard_normal(hd)).astype(np.float32))
    w.add_tensor(p + "ffn_norm.weight", (1.0 + 0.1 * rng.standard_normal(n_embd)).astype(np.float32))
    t(p + "ffn_gate.weight", (n_ff, n_embd))
    t(p + "ffn_down.weight", (n_embd, n_ff))
    t(p + "ffn_up.weight", (n_ff, n_embd))
    groups = n_embd // group
    t(p + "attn_conv_base", (2, kernel, n_embd), 0.5)
    t(p + "attn_conv_proj.weight", (2 * kernel * groups, n_embd), 0.1)
    t(p + "ffn_conv_base", (2, kernel, n_embd), 0.5)
    t(p + "ffn_conv_proj.weight", (2 * kernel * groups, n_embd), 0.1)
t("selector_predecessor", (vocab, rank), 0.1)
t("selector_successor", (vocab, rank), 0.1)
t("selector_hidden.weight", (rank, n_embd), 0.1)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
