// dendrite: fine-tune a DFlash / DFlash 2 drafter against a quantized target (the "king") inside llama.cpp.
//
// The target runs through the normal llama API and hands over the same per-layer input features the
// speculative driver feeds the drafter at serve time (llama_set_embeddings_layer_inp). The drafter's
// training graph is built here from the drafter's own tensors with ggml ops and trained with ggml-opt
// (AdamW). Teacher forcing: every sampled anchor position a makes one block of B rows
// [tok[a], MASK x (B-1)] at positions a..a+B-1 that attends causally to the encoded target features of
// positions < a (sliding window n_swa) and bidirectionally inside the block; cross-entropy on rows 1..B-1
// against tok[a+1..a+B-1], weighted exp(-(i-1)/gamma). Saves a GGUF the server loads unchanged.
//
//   llama-dflash-train -m king.gguf -md drafter-f32.gguf --train shards/ --heldout heldout.jsonl --out ckpt/
//   llama-dflash-train -m king.gguf -md drafter.gguf --heldout heldout.jsonl --check 100
//   llama-dflash-train -m king.gguf -md drafter.gguf --heldout heldout.jsonl --selfcheck 20
//   llama-dflash-train -md toy.gguf --gradcheck
//
// Data: jsonl, one sample per line: {"prompt_tokens":[...], "response_tokens":[...]} (ids of the target's
// tokenizer; the response must be the target's own greedy answer, that is what verification accepts).

#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "gguf.h"
#include "../../src/llama-ext.h"
#include "../../src/llama-model.h"
#include "../../src/llama-graph.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "metrics.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

#define LOGI(...) do { fprintf(stderr, __VA_ARGS__); fflush(stderr); } while (0)
#define DIE(...)  do { fprintf(stderr, "error: " __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------------------------------------
// options

struct options {
    std::string king_path;
    std::string draft_path;
    std::vector<std::string> train_paths;
    std::string heldout_path;
    std::string out_dir = "dflash-ckpt";
    std::string devices;       // comma-separated ggml device names, one replica each; empty = every GPU, else CPU
    float clip           = 1.0f;   // global gradient-norm clip, 0 = off
    int   sync_every     = 200;    // steps between re-broadcasts of replica 0's weights (guards against drift)
    std::string save_type = "src"; // src | f32

    int   anchors        = 64;
    int   block          = 0;      // 0 = the drafter's trained block size
    float gamma          = 4.0f;
    int   seq_max        = 3072;
    int   seqs_per_step  = 8;
    float lr             = 1.5e-4f;
    float lr_min         = 1e-5f;
    float warmup         = 0.04f;
    float wd             = 0.01f;
    float beta1          = 0.9f;
    float beta2          = 0.95f;
    float eps            = 1e-8f;
    int   epochs         = 2;
    int   max_steps      = 0;      // 0 = epochs * pool / seqs_per_step
    int   eval_every     = 500;
    int   save_every     = 500;
    int   eval_samples   = 100;
    int   eval_anchors   = 32;
    int   check          = 0;      // > 0: evaluate heldout on N samples and exit
    int   selfcheck      = 0;      // > 0: compare N blocks against the serving graph and exit
    bool  gradcheck      = false;
    int   metrics_port   = 9108;
    int   threads        = 16;
    int   seed           = 1234;
    int   rescan_every   = 50;     // steps between looks for new shards
    bool  train_selector = false;
    int   n_gpu_layers   = 999;
};

static void usage() {
    fprintf(stderr,
        "usage: llama-dflash-train -md drafter.gguf [-m king.gguf] [options]\n"
        "  -m FILE            target (king) GGUF; omitted only for --gradcheck on a toy drafter that carries token_embd + output\n"
        "  -md FILE           drafter GGUF (any tensor types; trainable tensors are held in f32)\n"
        "  --train P [P...]   jsonl shard files or directories (rescanned during training for new shards)\n"
        "  --heldout FILE     jsonl held-out samples\n"
        "  --out DIR          checkpoints, train.jsonl log (default dflash-ckpt)\n"
        "  --anchors N        anchors per sequence per micro-step (64)\n"
        "  --block B          block rows incl. anchor (default: drafter's block_size)\n"
        "  --gamma G          loss weight exp(-(i-1)/G) for row i (4)\n"
        "  --seq-max N        longest sequence, prompt + answer (3072)\n"
        "  --seqs-per-step N  sequences per optimizer step, gradient accumulation (8)\n"
        "  --lr --lr-min --warmup --wd --beta1 --beta2 --eps   AdamW + cosine schedule\n"
        "  --epochs N  --max-steps N  --eval-every N  --save-every N  --eval-samples N  --eval-anchors N\n"
        "  --check N          evaluate the drafter on N held-out samples, print acceptance, exit\n"
        "  --selfcheck N      compare the training graph with the serving graph on N blocks, exit\n"
        "  --gradcheck        finite-difference check of the gradients on a toy drafter, exit\n"
        "  --train-selector   also train the DFlash 2 selector tensors (default: frozen)\n"
        "  --save-type src|f32  checkpoint tensor types: the drafter file's (bf16/f16) or f32 (src)\n"
        "  --devices A,B,...  ggml devices, one data-parallel replica each (default: every GPU, else CPU)\n"
        "  --clip X           global gradient-norm clip (1.0, 0 = off)\n"
        "  --sync-every N     steps between re-broadcasts of replica 0's weights to the others (200)\n"
        "  --metrics-port P   Prometheus /metrics port (9108, 0 = off)\n"
        "  --threads N  --seed N  --rescan-every N  -ngl N\n");
}

static options parse_args(int argc, char ** argv) {
    options o;
    auto need = [&](int & i, const char * flag) -> const char * {
        if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", flag); usage(); exit(1); }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "-m"  || a == "--king")      o.king_path  = need(i, "-m");
        else if (a == "-md" || a == "--draft")     o.draft_path = need(i, "-md");
        else if (a == "--train") { while (i + 1 < argc && argv[i + 1][0] != '-') o.train_paths.push_back(argv[++i]); }
        else if (a == "--heldout")       o.heldout_path = need(i, "--heldout");
        else if (a == "--out")           o.out_dir = need(i, "--out");
        else if (a == "--anchors")       o.anchors = atoi(need(i, "--anchors"));
        else if (a == "--block")         o.block = atoi(need(i, "--block"));
        else if (a == "--gamma")         o.gamma = (float) atof(need(i, "--gamma"));
        else if (a == "--seq-max")       o.seq_max = atoi(need(i, "--seq-max"));
        else if (a == "--seqs-per-step") o.seqs_per_step = atoi(need(i, "--seqs-per-step"));
        else if (a == "--lr")            o.lr = (float) atof(need(i, "--lr"));
        else if (a == "--lr-min")        o.lr_min = (float) atof(need(i, "--lr-min"));
        else if (a == "--warmup")        o.warmup = (float) atof(need(i, "--warmup"));
        else if (a == "--wd")            o.wd = (float) atof(need(i, "--wd"));
        else if (a == "--beta1")         o.beta1 = (float) atof(need(i, "--beta1"));
        else if (a == "--beta2")         o.beta2 = (float) atof(need(i, "--beta2"));
        else if (a == "--eps")           o.eps = (float) atof(need(i, "--eps"));
        else if (a == "--epochs")        o.epochs = atoi(need(i, "--epochs"));
        else if (a == "--max-steps")     o.max_steps = atoi(need(i, "--max-steps"));
        else if (a == "--eval-every")    o.eval_every = atoi(need(i, "--eval-every"));
        else if (a == "--save-every")    o.save_every = atoi(need(i, "--save-every"));
        else if (a == "--eval-samples")  o.eval_samples = atoi(need(i, "--eval-samples"));
        else if (a == "--eval-anchors")  o.eval_anchors = atoi(need(i, "--eval-anchors"));
        else if (a == "--check")         o.check = atoi(need(i, "--check"));
        else if (a == "--selfcheck")     o.selfcheck = atoi(need(i, "--selfcheck"));
        else if (a == "--gradcheck")     o.gradcheck = true;
        else if (a == "--train-selector") o.train_selector = true;
        else if (a == "--save-type")     o.save_type = need(i, "--save-type");
        else if (a == "--devices")       o.devices = need(i, "--devices");
        else if (a == "--clip")          o.clip = (float) atof(need(i, "--clip"));
        else if (a == "--sync-every")    o.sync_every = atoi(need(i, "--sync-every"));
        else if (a == "--metrics-port")  o.metrics_port = atoi(need(i, "--metrics-port"));
        else if (a == "--threads")       o.threads = atoi(need(i, "--threads"));
        else if (a == "--seed")          o.seed = atoi(need(i, "--seed"));
        else if (a == "--rescan-every")  o.rescan_every = atoi(need(i, "--rescan-every"));
        else if (a == "-ngl")            o.n_gpu_layers = atoi(need(i, "-ngl"));
        else if (a == "-h" || a == "--help") { usage(); exit(0); }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); usage(); exit(1); }
    }
    if (o.draft_path.empty()) { usage(); exit(1); }
    if (o.king_path.empty() && !o.gradcheck) DIE("-m king.gguf is required unless --gradcheck");
    return o;
}

// ---------------------------------------------------------------------------------------------------------
// backends

struct backends_t {
    ggml_backend_dev_t dev = nullptr;
    ggml_backend_t gpu = nullptr;   // the training backend (may itself be a CPU backend)
    ggml_backend_t cpu = nullptr;   // fallback for ops the training backend lacks
    ggml_backend_sched_t sched = nullptr;
    bool gpu_is_cpu = false;
};

static void set_backend_threads(ggml_backend_t be, int n) {
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) return;
    auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) fn(be, n);
}

// the devices to train on: --devices, else every GPU, else the CPU
static std::vector<ggml_backend_dev_t> pick_devices(const options & o) {
    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devs;
    if (!o.devices.empty()) {
        size_t pos = 0;
        while (pos <= o.devices.size()) {
            size_t e = o.devices.find(',', pos);
            if (e == std::string::npos) e = o.devices.size();
            std::string name = o.devices.substr(pos, e - pos);
            if (!name.empty()) {
                ggml_backend_dev_t d = ggml_backend_dev_by_name(name.c_str());
                if (!d) DIE("device %s not found", name.c_str());
                devs.push_back(d);
            }
            pos = e + 1;
        }
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) devs.push_back(d);
        }
        if (devs.empty()) devs.push_back(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    }
    return devs;
}

static backends_t init_backends(ggml_backend_dev_t dev, int threads) {
    backends_t b;
    b.dev = dev;
    b.gpu = ggml_backend_dev_init(dev, nullptr);
    if (!b.gpu) DIE("cannot init backend for %s", ggml_backend_dev_name(dev));
    b.gpu_is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (b.gpu_is_cpu) {
        b.cpu = b.gpu;
    } else {
        b.cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!b.cpu) DIE("no CPU backend");
    }
    set_backend_threads(b.cpu, threads);
    std::vector<ggml_backend_t> list = { b.gpu };
    if (!b.gpu_is_cpu) list.push_back(b.cpu);
    b.sched = ggml_backend_sched_new(list.data(), nullptr, (int) list.size(), 65536, false, true);
    return b;
}

// ---------------------------------------------------------------------------------------------------------
// gguf helpers

static int64_t kv_id(const gguf_context * g, const std::string & key, bool required = true) {
    int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 && required) DIE("gguf key %s missing", key.c_str());
    return id;
}
static uint32_t kv_u32(const gguf_context * g, const std::string & key, uint32_t def = 0, bool required = true) {
    int64_t id = kv_id(g, key, required);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (uint32_t) gguf_get_val_u64(g, id);
        default: DIE("gguf key %s is not an integer", key.c_str());
    }
}
static float kv_f32(const gguf_context * g, const std::string & key, float def, bool required = true) {
    int64_t id = kv_id(g, key, required);
    if (id < 0) return def;
    return gguf_get_val_f32(g, id);
}

// read one tensor's raw bytes out of a gguf file
static void read_tensor_bytes(FILE * f, const gguf_context * g, int64_t tid, std::vector<uint8_t> & out, size_t nbytes) {
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);
    out.resize(nbytes);
#ifdef _WIN32
    _fseeki64(f, (long long) off, SEEK_SET);
#else
    fseeko(f, (off_t) off, SEEK_SET);
#endif
    if (fread(out.data(), 1, nbytes, f) != nbytes) DIE("short read of tensor %s", gguf_get_tensor_name(g, tid));
}

static void to_f32(ggml_type type, const void * src, float * dst, int64_t n) {
    if (type == GGML_TYPE_F32) { memcpy(dst, src, n * sizeof(float)); return; }
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    if (!tr->to_float) DIE("no dequantizer for type %s", ggml_type_name(type));
    tr->to_float(src, dst, n);
}

// ---------------------------------------------------------------------------------------------------------
// drafter

struct draft_hp {
    int32_t n_embd = 0, n_layer = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0, n_vocab = 0;
    int32_t n_rot = 0, n_swa = 0, block = 0, conv_kernel = 0, conv_group = 0, sel_rank = 0, sel_topk = 0;
    int32_t n_ctx_train = 262144;
    float   eps = 1e-6f, rope_base = 10000.0f;
    int32_t mask_id = -1;
    std::vector<int32_t> target_layers;   // the gguf's target_layers = layer whose INPUT is captured
    int n_cap() const { return (int) target_layers.size(); }
};

struct draft_layer {
    ggml_tensor * attn_norm = nullptr, * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
    ggml_tensor * q_norm = nullptr, * k_norm = nullptr, * ffn_norm = nullptr;
    ggml_tensor * gate = nullptr, * down = nullptr, * up = nullptr;
    ggml_tensor * aconv_base = nullptr, * aconv_proj = nullptr, * fconv_base = nullptr, * fconv_proj = nullptr;
};

struct drafter {
    draft_hp hp;
    std::string path;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> all;                 // every drafter tensor, file order
    std::map<std::string, ggml_type> src_type;      // type in the file
    std::vector<ggml_tensor *> params;              // trainable
    ggml_tensor * fc = nullptr, * enc_norm = nullptr, * out_norm = nullptr;
    std::vector<draft_layer> layers;
    // the target's shared tensors (or the toy's own)
    ggml_tensor * tok_embd = nullptr;   // f16 [n_embd, n_vocab]
    ggml_tensor * lm_head  = nullptr;   // f32 [n_embd, n_vocab]  (f32: ggml_out_prod wants it)
    gguf_context * meta = nullptr;      // the source file's metadata, copied into checkpoints
};

static bool is_selector_name(const std::string & n) {
    return n.rfind("selector_", 0) == 0 || n == "d2t";
}

static ggml_tensor * get_t(drafter & D, const std::string & name, bool required = true) {
    for (auto * t : D.all) if (name == ggml_get_name(t)) return t;
    if (required) DIE("drafter tensor %s missing", name.c_str());
    return nullptr;
}

static void load_drafter(drafter & D, const options & o, ggml_backend_t backend) {
    D.path = o.draft_path;
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta_ctx };
    gguf_context * g = gguf_init_from_file(o.draft_path.c_str(), gp);
    if (!g) DIE("cannot open %s", o.draft_path.c_str());
    D.meta = g;

    const std::string arch = gguf_get_val_str(g, kv_id(g, "general.architecture"));
    if (arch != "dflash") DIE("%s: architecture is %s, expected dflash", o.draft_path.c_str(), arch.c_str());
    auto & h = D.hp;
    h.n_embd    = kv_u32(g, arch + ".embedding_length");
    h.n_layer   = kv_u32(g, arch + ".block_count");
    h.n_head    = kv_u32(g, arch + ".attention.head_count");
    h.n_head_kv = kv_u32(g, arch + ".attention.head_count_kv");
    h.head_dim  = kv_u32(g, arch + ".attention.key_length", h.n_embd / h.n_head, false);
    h.n_ff      = kv_u32(g, arch + ".feed_forward_length");
    h.n_rot     = kv_u32(g, arch + ".rope.dimension_count", h.head_dim, false);
    h.n_ctx_train = kv_u32(g, arch + ".context_length", 262144, false);
    h.n_swa     = kv_u32(g, arch + ".attention.sliding_window", 0, false);
    h.block     = kv_u32(g, arch + ".block_size", 16, false);
    h.conv_kernel = kv_u32(g, arch + ".conv_kernel_size", 0, false);
    h.conv_group  = kv_u32(g, arch + ".conv_group_size", 0, false);
    h.sel_rank    = kv_u32(g, arch + ".selector_rank", 0, false);
    h.sel_topk    = kv_u32(g, arch + ".selector_top_k", 0, false);
    h.eps       = kv_f32(g, arch + ".attention.layer_norm_rms_epsilon", 1e-6f, false);
    h.rope_base = kv_f32(g, arch + ".rope.freq_base", 10000.0f, false);
    h.mask_id   = (int32_t) kv_u32(g, "tokenizer.ggml.mask_token_id", (uint32_t) -1, false);
    {
        int64_t id = kv_id(g, arch + ".target_layers");
        const size_t n = gguf_get_arr_n(g, id);
        const void * data = gguf_get_arr_data(g, id);
        const gguf_type at = gguf_get_arr_type(g, id);
        for (size_t i = 0; i < n; ++i) {
            if (at == GGUF_TYPE_INT32)       h.target_layers.push_back(((const int32_t  *) data)[i]);
            else if (at == GGUF_TYPE_UINT32) h.target_layers.push_back((int32_t) ((const uint32_t *) data)[i]);
            else DIE("target_layers has an unexpected array type");
        }
    }
    {
        int64_t id = kv_id(g, "tokenizer.ggml.tokens", false);
        h.n_vocab = id >= 0 ? (int32_t) gguf_get_arr_n(g, id) : 0;
    }

    // tensors: trainable -> f32, the rest keep their type
    const int64_t n_tensors = gguf_get_n_tensors(g);
    ggml_init_params ip = { (size_t) (n_tensors + 8) * ggml_tensor_overhead(), nullptr, true };
    D.ctx = ggml_init(ip);
    std::vector<ggml_tensor *> metas;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * m = ggml_get_tensor(meta_ctx, name);
        metas.push_back(m);
        const bool frozen = is_selector_name(name) && !o.train_selector;
        const bool own_embd = (std::string(name) == "token_embd.weight" || std::string(name) == "output.weight");
        ggml_type ty = (frozen || own_embd) ? m->type : GGML_TYPE_F32;
        if (own_embd) ty = std::string(name) == "output.weight" ? GGML_TYPE_F32 : GGML_TYPE_F16;
        ggml_tensor * t = ggml_new_tensor(D.ctx, ty, ggml_n_dims(m), m->ne);
        ggml_set_name(t, name);
        D.src_type[name] = m->type;
        D.all.push_back(t);
        if (own_embd) {
            if (std::string(name) == "output.weight") D.lm_head = t; else D.tok_embd = t;
        } else if (!frozen) {
            if (ty == GGML_TYPE_F32) { ggml_set_param(t); D.params.push_back(t); }
        }
    }
    D.buf = ggml_backend_alloc_ctx_tensors(D.ctx, backend);
    if (!D.buf) DIE("cannot allocate drafter tensors");

    FILE * f = fopen(o.draft_path.c_str(), "rb");
    if (!f) DIE("cannot open %s", o.draft_path.c_str());
    std::vector<uint8_t> raw;
    std::vector<float> f32;
    std::vector<ggml_fp16_t> f16;
    size_t bytes_trainable = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * m = metas[i];
        ggml_tensor * t = D.all[i];
        read_tensor_bytes(f, g, i, raw, ggml_nbytes(m));
        if (t->type == m->type) {
            ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
        } else if (t->type == GGML_TYPE_F32) {
            f32.resize(ggml_nelements(m));
            to_f32(m->type, raw.data(), f32.data(), ggml_nelements(m));
            ggml_backend_tensor_set(t, f32.data(), 0, f32.size() * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            f32.resize(ggml_nelements(m));
            to_f32(m->type, raw.data(), f32.data(), ggml_nelements(m));
            f16.resize(f32.size());
            ggml_fp32_to_fp16_row(f32.data(), f16.data(), (int64_t) f32.size());
            ggml_backend_tensor_set(t, f16.data(), 0, f16.size() * sizeof(ggml_fp16_t));
        } else DIE("unexpected load type");
        if (t->flags & GGML_TENSOR_FLAG_PARAM) bytes_trainable += ggml_nbytes(t);
    }
    fclose(f);
    ggml_free(meta_ctx);

    D.fc       = get_t(D, "fc.weight");
    D.enc_norm = get_t(D, "enc.output_norm.weight");
    D.out_norm = get_t(D, "output_norm.weight");
    D.layers.resize(h.n_layer);
    for (int il = 0; il < h.n_layer; ++il) {
        auto & L = D.layers[il];
        auto nm = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
        L.attn_norm = get_t(D, nm("attn_norm.weight"));
        L.wq = get_t(D, nm("attn_q.weight"));
        L.wk = get_t(D, nm("attn_k.weight"));
        L.wv = get_t(D, nm("attn_v.weight"));
        L.wo = get_t(D, nm("attn_output.weight"));
        L.q_norm = get_t(D, nm("attn_q_norm.weight"));
        L.k_norm = get_t(D, nm("attn_k_norm.weight"));
        L.ffn_norm = get_t(D, nm("ffn_norm.weight"));
        L.gate = get_t(D, nm("ffn_gate.weight"));
        L.down = get_t(D, nm("ffn_down.weight"));
        L.up   = get_t(D, nm("ffn_up.weight"));
        if (h.conv_kernel > 0) {
            L.aconv_base = get_t(D, nm("attn_conv_base"));
            L.aconv_proj = get_t(D, nm("attn_conv_proj.weight"));
            L.fconv_base = get_t(D, nm("ffn_conv_base"));
            L.fconv_proj = get_t(D, nm("ffn_conv_proj.weight"));
        }
    }
    if (h.n_vocab == 0 && D.tok_embd) h.n_vocab = (int32_t) D.tok_embd->ne[1];

    LOGI("drafter: %s\n", o.draft_path.c_str());
    LOGI("  n_embd=%d n_layer=%d heads=%d/%d head_dim=%d n_ff=%d vocab=%d block=%d swa=%d rope_base=%.0f eps=%g mask=%d\n",
         h.n_embd, h.n_layer, h.n_head, h.n_head_kv, h.head_dim, h.n_ff, h.n_vocab, h.block, h.n_swa, h.rope_base, h.eps, h.mask_id);
    LOGI("  conv kernel=%d group=%d selector rank=%d topk=%d target_layers=[", h.conv_kernel, h.conv_group, h.sel_rank, h.sel_topk);
    for (size_t i = 0; i < h.target_layers.size(); ++i) LOGI("%s%d", i ? "," : "", h.target_layers[i]);
    LOGI("]\n  %zu tensors, %zu trainable (%.2f GB f32)%s\n", D.all.size(), D.params.size(),
         bytes_trainable / 1e9, o.train_selector ? ", selector trainable" : ", selector frozen");
}

// apply one Hadamard block transform to every row of a [n_cols, n_rows] f32 table (multithreaded).
// out[i] = sum_k rot[i*n + k] * in[k] per block of n, i.e. ggml_mul_mat(rot, x) as llama_mul_mat_hadamard does;
// mode 0: h = (R z) * s      (embedding lookup, the inverse transform the graph applies after get_rows)
// mode 1: w' = s * (R w)     (head: logits = W . R(x*s) = (s*(R^T... see README) folded into the rows)
static void fold_hadamard(std::vector<float> & table, int64_t n_cols, int64_t n_rows, const std::vector<float> & rot, int64_t n,
                          const std::vector<float> & signs, int mode, int threads) {
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t) {
        th.emplace_back([&, t] {
            std::vector<float> tmp(n);
            for (int64_t r = n_rows * t / threads; r < n_rows * (t + 1) / threads; ++r) {
                float * row = table.data() + r * n_cols;
                for (int64_t b = 0; b < n_cols; b += n) {
                    float * x = row + b;
                    for (int64_t i = 0; i < n; ++i) {
                        float acc = 0.0f;
                        // mode 0: out[i] = sum_k rot[i,k] z[k]   (as ggml_mul_mat(rot, z))
                        // mode 1: w'[i]  = sum_k rot[k,i] w[k]   (rot^T: logits = W . rot(x*s) = (rot^T W * s) . x)
                        if (mode == 0) { const float * ri = rot.data() + i * n; for (int64_t k = 0; k < n; ++k) acc += ri[k] * x[k]; }
                        else           { for (int64_t k = 0; k < n; ++k) acc += rot[k * n + i] * x[k]; }
                        tmp[i] = acc;
                    }
                    for (int64_t i = 0; i < n; ++i) x[i] = tmp[i];
                    if (!signs.empty()) for (int64_t k = 0; k < n; ++k) x[k] *= signs[b + k];   // signs on the output index in both modes
                }
            }
        });
    }
    for (auto & x : th) x.join();
}

// the target's token_embd (f16) and output (f32): dequantized from its gguf, with the king's Hadamard
// rotation and signs folded in so the training graph can use plain get_rows / mul_mat on them
// the folded tables are computed once per process (host RAM) and uploaded to every replica
struct shared_cache {
    std::vector<ggml_fp16_t> tok_embd;
    std::vector<float> lm_head;
    int64_t n_embd = 0, n_vocab = 0;
    bool ready = false;
};
static shared_cache g_shared;
static std::mutex g_shared_mu;

static void load_shared_from_king(drafter & D, const std::string & king_path, ggml_backend_t backend,
                                  const llama_model * king, int threads) {
    std::lock_guard<std::mutex> lk(g_shared_mu);
    if (g_shared.ready) {
        D.hp.n_vocab = (int32_t) g_shared.n_vocab;
        ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        D.tok_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, g_shared.n_embd, g_shared.n_vocab);
        ggml_set_name(D.tok_embd, "king.token_embd");
        D.lm_head  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, g_shared.n_embd, g_shared.n_vocab);
        ggml_set_name(D.lm_head, "king.output");
        if (!ggml_backend_alloc_ctx_tensors(ctx, backend)) DIE("cannot allocate the king's embedding and output tensors");
        ggml_backend_tensor_set(D.tok_embd, g_shared.tok_embd.data(), 0, ggml_nbytes(D.tok_embd));
        ggml_backend_tensor_set(D.lm_head, g_shared.lm_head.data(), 0, ggml_nbytes(D.lm_head));
        return;
    }
    threads = std::max(threads, (int) std::thread::hardware_concurrency());   // the fold runs once, use the whole box
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gp = { true, &meta_ctx };
    gguf_context * g = gguf_init_from_file(king_path.c_str(), gp);
    if (!g) DIE("cannot open %s", king_path.c_str());
    ggml_tensor * me = ggml_get_tensor(meta_ctx, "token_embd.weight");
    ggml_tensor * mo = ggml_get_tensor(meta_ctx, "output.weight");
    if (!me) DIE("king has no token_embd.weight");
    if (!mo) { LOGI("king has no output.weight: tied embeddings, using token_embd\n"); mo = me; }
    if (me->ne[0] != D.hp.n_embd) DIE("king n_embd %" PRId64 " != drafter n_embd %d", me->ne[0], D.hp.n_embd);
    D.hp.n_vocab = (int32_t) me->ne[1];

    ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    D.tok_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, me->ne[0], me->ne[1]);
    ggml_set_name(D.tok_embd, "king.token_embd");
    D.lm_head  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mo->ne[0], mo->ne[1]);
    ggml_set_name(D.lm_head, "king.output");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) DIE("cannot allocate the king's embedding and output tensors");

    FILE * f = fopen(king_path.c_str(), "rb");
    std::vector<uint8_t> raw;
    std::vector<float> f32(ggml_nelements(me));
    const int64_t n_t = gguf_get_n_tensors(g);
    auto find_tid = [&](const char * name) -> int64_t {
        for (int64_t i = 0; i < n_t; ++i) if (!strcmp(gguf_get_tensor_name(g, i), name)) return i;
        return -1;
    };
    // the king's transforms (llama_model::hadamard_inverses for the embedding lookup, hadamard_rotations for the head)
    auto get_transform = [&](const llama_hadamard_rotations & map, const ggml_tensor * w, std::vector<float> & rot, int64_t & n,
                             std::vector<float> & signs, const char * what) -> bool {
        auto it = map.find(w);
        if (it == map.end()) { LOGI("king %s: no Hadamard transform\n", what); return false; }
        const auto & t = it->second;
        if (t.perm_rep > 1) DIE("king %s: permuted Hadamard fold (perm_rep %" PRId64 ") not supported here", what, t.perm_rep);
        n = t.rot->ne[0];
        rot.resize((size_t) n * n);
        ggml_backend_tensor_get(t.rot, rot.data(), 0, rot.size() * sizeof(float));
        if (t.signs) {
            if (t.signs->type != GGML_TYPE_F32) DIE("king %s: signs are %s, expected f32", what, ggml_type_name(t.signs->type));
            signs.resize(ggml_nelements(t.signs));
            ggml_backend_tensor_get(t.signs, signs.data(), 0, signs.size() * sizeof(float));
        }
        LOGI("king %s: Hadamard block %" PRId64 "%s\n", what, n, t.signs ? " with signs" : "");
        return true;
    };
    std::vector<float> rot, signs;
    int64_t n = 0;
    // token_embd -> f16, un-rotated
    read_tensor_bytes(f, g, find_tid("token_embd.weight"), raw, ggml_nbytes(me));
    to_f32(me->type, raw.data(), f32.data(), ggml_nelements(me));
    if (get_transform(king->hadamard_inverses, king->tok_embd, rot, n, signs, "token_embd")) {
        fold_hadamard(f32, me->ne[0], me->ne[1], rot, n, signs, 0, threads);
    }
    {
        g_shared.tok_embd.resize(f32.size());
        ggml_fp32_to_fp16_row(f32.data(), g_shared.tok_embd.data(), (int64_t) f32.size());
        ggml_backend_tensor_set(D.tok_embd, g_shared.tok_embd.data(), 0, g_shared.tok_embd.size() * sizeof(ggml_fp16_t));
    }
    // output -> f32, with the activation-side rotation folded into the rows
    if (mo != me) {
        read_tensor_bytes(f, g, find_tid("output.weight"), raw, ggml_nbytes(mo));
        f32.resize(ggml_nelements(mo));
        to_f32(mo->type, raw.data(), f32.data(), ggml_nelements(mo));
    }
    rot.clear(); signs.clear();
    if (king->output && get_transform(king->hadamard_rotations, king->output, rot, n, signs, "output")) {
        fold_hadamard(f32, mo->ne[0], mo->ne[1], rot, n, signs, 1, threads);
    }
    if (king->output_s) {
        std::vector<float> ws(ggml_nelements(king->output_s));
        ggml_backend_tensor_get(king->output_s, ws.data(), 0, ws.size() * sizeof(float));
        LOGI("king output scale: %zu values\n", ws.size());
        for (int64_t v = 0; v < mo->ne[1]; ++v) {
            const float sc = ws.size() == 1 ? ws[0] : ws[v];
            float * row = f32.data() + v * mo->ne[0];
            for (int64_t i = 0; i < mo->ne[0]; ++i) row[i] *= sc;
        }
    }
    ggml_backend_tensor_set(D.lm_head, f32.data(), 0, f32.size() * sizeof(float));
    g_shared.lm_head = std::move(f32);
    g_shared.n_embd = me->ne[0];
    g_shared.n_vocab = D.hp.n_vocab;
    g_shared.ready = true;
    fclose(f);
    ggml_free(meta_ctx);
    gguf_free(g);
    LOGI("king shared tensors: token_embd %s -> f16, output %s -> f32, vocab %d\n",
         ggml_type_name(me->type), ggml_type_name(mo->type), D.hp.n_vocab);
}

// write a checkpoint: the source file's metadata, every tensor in its source type (or f32)
static void save_drafter(drafter & D, const std::string & path, const std::string & save_type) {
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, D.meta);
    size_t total = 0;
    for (auto * t : D.all) {
        if (t == D.tok_embd || t == D.lm_head) continue;
        ggml_type ty = t->type;
        if (t->flags & GGML_TENSOR_FLAG_PARAM) {
            ggml_type st = D.src_type[ggml_get_name(t)];
            ty = (save_type == "f32" || !(st == GGML_TYPE_F16 || st == GGML_TYPE_BF16 || st == GGML_TYPE_F32)) ? GGML_TYPE_F32 : st;
        }
        total += ggml_row_size(ty, t->ne[0]) * (t->ne[1] * t->ne[2] * t->ne[3]);
    }
    ggml_init_params ip = { total + D.all.size() * ggml_tensor_overhead() + 1024 * 1024, nullptr, false };
    ggml_context * hctx = ggml_init(ip);
    std::vector<float> f32;
    for (auto * t : D.all) {
        if (t == D.tok_embd || t == D.lm_head) continue;
        ggml_type ty = t->type;
        if (t->flags & GGML_TENSOR_FLAG_PARAM) {
            ggml_type st = D.src_type[ggml_get_name(t)];
            ty = (save_type == "f32" || !(st == GGML_TYPE_F16 || st == GGML_TYPE_BF16 || st == GGML_TYPE_F32)) ? GGML_TYPE_F32 : st;
        }
        ggml_tensor * h = ggml_new_tensor(hctx, ty, ggml_n_dims(t), t->ne);
        ggml_set_name(h, ggml_get_name(t));
        if (ty == t->type) {
            ggml_backend_tensor_get(t, h->data, 0, ggml_nbytes(t));
        } else {
            f32.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, f32.data(), 0, f32.size() * sizeof(float));
            if (ty == GGML_TYPE_F16)       ggml_fp32_to_fp16_row(f32.data(), (ggml_fp16_t *) h->data, (int64_t) f32.size());
            else if (ty == GGML_TYPE_BF16) ggml_fp32_to_bf16_row(f32.data(), (ggml_bf16_t *) h->data, (int64_t) f32.size());
            else DIE("save: cannot convert to %s", ggml_type_name(ty));
        }
        gguf_add_tensor(out, h);
    }
    const std::string tmp = path + ".tmp";
    if (!gguf_write_to_file(out, tmp.c_str(), false)) DIE("cannot write %s", tmp.c_str());
    fs::rename(tmp, path);
    gguf_free(out);
    ggml_free(hctx);
    LOGI("saved %s (%.2f GB)\n", path.c_str(), total / 1e9);
}

// ---------------------------------------------------------------------------------------------------------
// the king: target features per position, exactly what the speculative driver captures

struct king_t {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    int32_t n_embd = 0;
    int32_t n_batch = 0;
    std::vector<int32_t> layers;   // drafter target_layers
    llama_batch batch{};
};

static void init_king(king_t & K, const options & o, const draft_hp & hp, ggml_backend_dev_t dev) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = o.n_gpu_layers;
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        mp.devices = devs;         // this replica's king lives on this replica's card
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
    }
    K.model = llama_model_load_from_file(o.king_path.c_str(), mp);
    if (!K.model) DIE("cannot load king %s", o.king_path.c_str());
    K.n_embd = llama_model_n_embd(K.model);
    if (K.n_embd != hp.n_embd) DIE("king n_embd %d != drafter n_embd %d", K.n_embd, hp.n_embd);
    const int n_layer = llama_model_n_layer(K.model);
    for (int l : hp.target_layers) if (l < 0 || l > n_layer) DIE("target layer %d outside the king's %d layers", l, n_layer);
    K.layers = hp.target_layers;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = o.seq_max + hp.block + 8;
    cp.n_batch   = o.seq_max;
    cp.n_ubatch  = o.seq_max;
    cp.n_seq_max = 1;
    cp.n_threads = o.threads;
    cp.n_threads_batch = o.threads;
    K.n_batch = o.seq_max;
    K.ctx = llama_init_from_model(K.model, cp);
    if (!K.ctx) DIE("cannot create the king's context");
    for (int l : K.layers) llama_set_embeddings_layer_inp(K.ctx, (uint32_t) l, true);
    K.batch = llama_batch_init(o.seq_max, 0, 1);
    LOGI("king on %s: context %d, capture layer inputs [", ggml_backend_dev_name(dev), cp.n_ctx);
    for (size_t i = 0; i < K.layers.size(); ++i) LOGI("%s%d", i ? "," : "", K.layers[i]);
    LOGI("]\n");
}

// feat: T rows of [n_cap * n_embd], row t = [layer0 | layer1 | ...] of position t
static void king_features(king_t & K, const std::vector<int32_t> & toks, std::vector<float> & feat) {
    const int T = (int) toks.size();
    if (T > K.n_batch) DIE("sequence of %d tokens exceeds --seq-max %d", T, K.n_batch);
    llama_memory_clear(llama_get_memory(K.ctx), true);
    K.batch.n_tokens = T;
    for (int i = 0; i < T; ++i) {
        K.batch.token[i]     = toks[i];
        K.batch.pos[i]       = i;
        K.batch.n_seq_id[i]  = 1;
        K.batch.seq_id[i][0] = 0;
        K.batch.logits[i]    = 0;
    }
    if (llama_decode(K.ctx, K.batch) != 0) DIE("king decode failed (T=%d)", T);
    const int n_cap = (int) K.layers.size();
    feat.resize((size_t) T * n_cap * K.n_embd);
    for (int k = 0; k < n_cap; ++k) {
        const float * src = llama_get_embeddings_layer_inp(K.ctx, (uint32_t) K.layers[k]);
        if (!src) DIE("king: layer %d input not captured", K.layers[k]);
        for (int t = 0; t < T; ++t) {
            memcpy(feat.data() + ((size_t) t * n_cap + k) * K.n_embd, src + (size_t) t * K.n_embd, K.n_embd * sizeof(float));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// data

struct sample {
    std::vector<int32_t> toks;   // prompt + answer
    int32_t prompt_len = 0;
};

static std::vector<std::string> list_shards(const std::vector<std::string> & paths) {
    std::vector<std::string> files;
    for (const auto & p : paths) {
        if (fs::is_directory(p)) {
            for (const auto & e : fs::directory_iterator(p)) {
                if (e.is_regular_file() && e.path().extension() == ".jsonl") files.push_back(e.path().string());
            }
        } else if (fs::is_regular_file(p)) {
            files.push_back(p);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// a shard still being written ends without a newline on its last line; such a line is skipped
static size_t load_shard(const std::string & path, int seq_max, int block, std::vector<sample> & out, size_t * skipped_long) {
    std::ifstream in(path);
    if (!in) DIE("cannot open %s", path.c_str());
    std::string line;
    size_t n = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        json j;
        try { j = json::parse(line); } catch (...) { continue; }
        if (!j.contains("prompt_tokens") || !j.contains("response_tokens")) continue;
        sample s;
        const auto & p = j["prompt_tokens"];
        const auto & r = j["response_tokens"];
        s.prompt_len = (int32_t) p.size();
        if (s.prompt_len + block + 1 > seq_max) { if (skipped_long) ++*skipped_long; continue; }
        s.toks.reserve(p.size() + r.size());
        for (const auto & t : p) s.toks.push_back(t.get<int32_t>());
        for (const auto & t : r) s.toks.push_back(t.get<int32_t>());
        if ((int) s.toks.size() > seq_max) s.toks.resize(seq_max);   // long answers are cut, the prompt stays
        if ((int) s.toks.size() < s.prompt_len + block) { if (skipped_long) ++*skipped_long; continue; }
        out.push_back(std::move(s));
        ++n;
    }
    return n;
}

struct pool_t {
    std::vector<sample> samples;
    std::set<std::string> seen;
    std::vector<int> order;     // training order, grows when new shards arrive
    size_t cursor = 0;
    size_t tokens = 0;
};

static void rescan_pool(pool_t & P, const options & o, int block, std::mt19937 & rng) {
    size_t skipped = 0, added = 0;
    for (const auto & f : list_shards(o.train_paths)) {
        if (P.seen.count(f)) continue;
        const size_t before = P.samples.size();
        load_shard(f, o.seq_max, block, P.samples, &skipped);
        P.seen.insert(f);
        std::vector<int> idx;
        for (size_t i = before; i < P.samples.size(); ++i) { idx.push_back((int) i); P.tokens += P.samples[i].toks.size(); }
        std::shuffle(idx.begin(), idx.end(), rng);
        P.order.insert(P.order.end(), idx.begin(), idx.end());
        added += idx.size();
    }
    if (added) LOGI("pool: +%zu samples -> %zu samples, %.1fM tokens (%zu skipped as too long)\n", added, P.samples.size(), P.tokens / 1e6, skipped);
}

// ---------------------------------------------------------------------------------------------------------
// the training graph

struct graph_io {
    ggml_tensor * feat = nullptr;     // f32 [n_cap*n_embd, T]
    ggml_tensor * pos_ctx = nullptr;  // i32 [T]
    ggml_tensor * tok_blk = nullptr;  // i32 [A*B]
    ggml_tensor * pos_blk = nullptr;  // i32 [A*B]
    ggml_tensor * mask = nullptr;     // f32 [T + A*B, A*B]
    // per block position i = 1..B-1 (index i-1): ggml's cross-entropy backward assumes one-hot rows, so the
    // position weights go on the per-position losses, not into the labels
    std::vector<ggml_tensor *> labels; // f32 [V, A] one-hot
    std::vector<ggml_tensor *> logits; // f32 [V, A]
    std::vector<ggml_tensor *> pred;   // i32 [A]
    ggml_tensor * loss = nullptr;     // f32 [1]  = sum_i w_i/sum(w) * CE_i
    ggml_tensor * enc = nullptr;      // f32 [n_embd, T]  encoder output, for the self-check
    int T = 0, A = 0, B = 0;
};

// DFlash 2 dynamic depthwise conv over the rows of each block (mirror of build_dflash2_conv in src/models/dflash.cpp)
static ggml_tensor * conv_block(ggml_context * c, const draft_hp & hp, ggml_tensor * hidden, ggml_tensor * dynamic,
                                ggml_tensor * base, int side, int A, int B) {
    const int64_t n_embd = hp.n_embd, AB = (int64_t) A * B, kernel = hp.conv_kernel, group = hp.conv_group;
    const int64_t n_groups = n_embd / group;
    ggml_tensor * blocks = ggml_reshape_3d(c, hidden, n_embd, B, A);
    ggml_tensor * coeffs = ggml_reshape_4d(c, dynamic, n_groups, kernel, 2, AB);
    ggml_tensor * coeffs_side = ggml_view_3d(c, coeffs, n_groups, kernel, AB, coeffs->nb[1], coeffs->nb[3], side * coeffs->nb[2]);
    ggml_tensor * coeff_all = ggml_cont(c, coeffs_side);
    coeff_all = ggml_reshape_4d(c, coeff_all, 1, n_groups, kernel, AB);
    coeff_all = ggml_repeat_4d(c, coeff_all, group, n_groups, kernel, AB);
    ggml_tensor * base_side = ggml_reshape_4d(c, ggml_view_1d(c, base, n_embd * kernel, side * base->nb[2]), group, n_groups, kernel, 1);
    ggml_tensor * weight_all = ggml_add(c, coeff_all, base_side);
    ggml_tensor * result = nullptr;
    for (int64_t tap = 0; tap < kernel; ++tap) {
        ggml_tensor * values = blocks;
        if (tap > 0) {
            ggml_tensor * zeros = ggml_fill(c, ggml_new_tensor_3d(c, GGML_TYPE_F32, n_embd, std::min<int64_t>(tap, B), A), 0.0f);
            if (tap < B) {
                ggml_tensor * previous = ggml_view_3d(c, blocks, n_embd, B - tap, A, blocks->nb[1], blocks->nb[2], 0);
                values = ggml_concat(c, zeros, previous, 1);
            } else {
                values = zeros;
            }
        }
        values = ggml_reshape_2d(c, values, n_embd, AB);
        ggml_tensor * weight = ggml_reshape_2d(c,
            ggml_cont(c, ggml_view_4d(c, weight_all, group, n_groups, 1, AB, weight_all->nb[1], weight_all->nb[2], weight_all->nb[3], tap * weight_all->nb[2])),
            n_embd, AB);
        ggml_tensor * term = ggml_mul(c, weight, values);
        result = result ? ggml_add(c, result, term) : term;
    }
    return result;
}

static ggml_tensor * rms(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

// every mat-mul in f32: CUDA's default precision runs f32 x f32 through f16 GEMMs, too coarse for training
static ggml_tensor * mm(ggml_context * c, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * t = ggml_mul_mat(c, a, b);
    ggml_mul_mat_set_prec(t, GGML_PREC_F32);
    return t;
}

static graph_io build_graph(ggml_context * c, ggml_cgraph * gf, drafter & D, int T, int A, int B, float gamma, float loss_scale) {
    const draft_hp & hp = D.hp;
    const int64_t n_embd = hp.n_embd, hd = hp.head_dim, n_head = hp.n_head, n_kv = hp.n_head_kv;
    const int64_t AB = (int64_t) A * B, Tk = T + AB;
    const float kq_scale = 1.0f / sqrtf((float) hd);
    graph_io io;
    io.T = T; io.A = A; io.B = B;

    io.feat    = ggml_new_tensor_2d(c, GGML_TYPE_F32, (int64_t) hp.n_cap() * n_embd, T); ggml_set_name(io.feat, "feat");    ggml_set_input(io.feat);
    io.pos_ctx = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);                                 ggml_set_name(io.pos_ctx, "pos_ctx"); ggml_set_input(io.pos_ctx);
    io.tok_blk = ggml_new_tensor_1d(c, GGML_TYPE_I32, AB);                                ggml_set_name(io.tok_blk, "tok_blk"); ggml_set_input(io.tok_blk);
    io.pos_blk = ggml_new_tensor_1d(c, GGML_TYPE_I32, AB);                                ggml_set_name(io.pos_blk, "pos_blk"); ggml_set_input(io.pos_blk);
    io.mask    = ggml_new_tensor_2d(c, GGML_TYPE_F32, Tk, AB);                            ggml_set_name(io.mask, "mask");    ggml_set_input(io.mask);

    auto rope = [&](ggml_tensor * x, ggml_tensor * pos) {
        return ggml_rope_ext(c, x, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX, hp.n_ctx_train, hp.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    };

    // encoder: fc + rms norm over the concatenated target features
    ggml_tensor * g = rms(c, mm(c, D.fc, io.feat), D.enc_norm, hp.eps);   // [n_embd, T]
    ggml_set_name(g, "enc"); ggml_set_output(g);
    io.enc = g;

    ggml_tensor * x = ggml_get_rows(c, D.tok_embd, io.tok_blk);                     // [n_embd, AB] f32
    for (int il = 0; il < hp.n_layer; ++il) {
        const auto & L = D.layers[il];
        // context keys and values from the encoded target features (serve time: injected into the cache)
        ggml_tensor * Kc = ggml_reshape_3d(c, mm(c, L.wk, g), hd, n_kv, T);
        Kc = rope(rms(c, Kc, L.k_norm, hp.eps), io.pos_ctx);
        ggml_tensor * Vc = ggml_reshape_3d(c, mm(c, L.wv, g), hd, n_kv, T);

        ggml_tensor * h = rms(c, x, L.attn_norm, hp.eps);
        ggml_format_name(h, "noise_norm-%d", il);
        ggml_tensor * adyn = nullptr;
        if (L.aconv_proj) {
            adyn = mm(c, L.aconv_proj, h);
            h = conv_block(c, hp, h, adyn, L.aconv_base, 0, A, B);
            ggml_format_name(h, "attn_conv_in-%d", il);
        }
        ggml_tensor * Q  = ggml_reshape_3d(c, mm(c, L.wq, h), hd, n_head, AB);
        ggml_tensor * Kb = ggml_reshape_3d(c, mm(c, L.wk, h), hd, n_kv, AB);
        ggml_tensor * Vb = ggml_reshape_3d(c, mm(c, L.wv, h), hd, n_kv, AB);
        Q  = rope(rms(c, Q,  L.q_norm, hp.eps), io.pos_blk);
        Kb = rope(rms(c, Kb, L.k_norm, hp.eps), io.pos_blk);
        ggml_format_name(Q, "Qcur-%d", il); ggml_format_name(Kb, "Kcur-%d", il); ggml_format_name(Vb, "Vcur-%d", il);
        ggml_format_name(Kc, "Kcur_injected-%d", il); ggml_format_name(Vc, "Vcur_injected-%d", il);

        ggml_tensor * K = ggml_concat(c, Kc, Kb, 2);   // [hd, n_kv, Tk]
        ggml_tensor * V = ggml_concat(c, Vc, Vb, 2);   // [hd, n_kv, Tk]

        ggml_tensor * q = ggml_permute(c, Q, 0, 2, 1, 3);               // [hd, AB, n_head]
        ggml_tensor * k = ggml_permute(c, K, 0, 2, 1, 3);               // [hd, Tk, n_kv]
        ggml_tensor * kq = mm(c, k, q);                       // [Tk, AB, n_head]
        kq = ggml_soft_max_ext(c, kq, io.mask, kq_scale, 0.0f);
        ggml_tensor * v = ggml_cont(c, ggml_permute(c, V, 1, 2, 0, 3)); // [Tk, hd, n_kv]
        ggml_tensor * kqv = mm(c, v, kq);                     // [hd, AB, n_head]
        ggml_tensor * o = ggml_cont(c, ggml_permute(c, kqv, 0, 2, 1, 3)); // [hd, n_head, AB]
        o = ggml_reshape_2d(c, o, hd * n_head, AB);                         // attention width, not n_embd
        ggml_format_name(o, "kqv_merged_cont-%d", il);
        ggml_tensor * attn = mm(c, L.wo, o);
        ggml_format_name(attn, "kqv_out-%d", il);
        if (adyn) { attn = conv_block(c, hp, attn, adyn, L.aconv_base, 1, A, B); ggml_format_name(attn, "attn_conv_out-%d", il); }
        x = ggml_add(c, x, attn);
        ggml_format_name(x, "ffn_inp-%d", il);

        ggml_tensor * h2 = rms(c, x, L.ffn_norm, hp.eps);
        ggml_format_name(h2, "ffn_norm-%d", il);
        ggml_tensor * fdyn = nullptr;
        if (L.fconv_proj) {
            fdyn = mm(c, L.fconv_proj, h2);
            h2 = conv_block(c, hp, h2, fdyn, L.fconv_base, 0, A, B);
        }
        ggml_tensor * ff = mm(c, L.down, ggml_swiglu_split(c, mm(c, L.gate, h2), mm(c, L.up, h2)));
        ggml_format_name(ff, "ffn_out-%d", il);
        if (fdyn) { ff = conv_block(c, hp, ff, fdyn, L.fconv_base, 1, A, B); ggml_format_name(ff, "ffn_conv_out-%d", il); }
        x = ggml_add(c, x, ff);
        ggml_format_name(x, "l_out-%d", il);
    }
    ggml_tensor * out = rms(c, x, D.out_norm, hp.eps);                               // [n_embd, AB]
    ggml_set_name(out, "result_norm");
    double wsum = 0.0;
    for (int i = 1; i < B; ++i) wsum += exp(-(double) (i - 1) / gamma);
    io.loss = nullptr;
    for (int i = 1; i < B; ++i) {
        // row i of every block: a strided view over the block-major rows
        ggml_tensor * rows = ggml_cont(c, ggml_view_2d(c, out, n_embd, A, (size_t) B * out->nb[1], (size_t) i * out->nb[1]));
        ggml_tensor * logits = mm(c, D.lm_head, rows);                      // [V, A]
        ggml_format_name(logits, "logits_%d", i); ggml_set_output(logits);
        ggml_tensor * pred = ggml_argmax(c, logits);
        ggml_format_name(pred, "pred_%d", i); ggml_set_output(pred);
        ggml_build_forward_expand(gf, pred);
        ggml_tensor * labels = ggml_new_tensor_2d(c, GGML_TYPE_F32, logits->ne[0], A);
        ggml_format_name(labels, "labels_%d", i); ggml_set_input(labels);
        ggml_tensor * ce = ggml_cross_entropy_loss(c, logits, labels);                // mean over the A rows
        ce = ggml_scale(c, ce, (float) (exp(-(double) (i - 1) / gamma) / wsum) * loss_scale);
        io.loss = io.loss ? ggml_add(c, io.loss, ce) : ce;
        io.logits.push_back(logits);
        io.pred.push_back(pred);
        io.labels.push_back(labels);
    }
    ggml_set_name(io.loss, "loss");
    ggml_build_forward_expand(gf, io.loss);
    return io;
}

// host-side inputs for one sequence and its anchors
struct step_data {
    std::vector<float>   feat;
    std::vector<int32_t> pos_ctx, tok_blk, pos_blk, targets;
    std::vector<float>   mask;
    std::vector<std::vector<float>> labels;   // per position i-1: [V, A] one-hot
    int T = 0, A = 0, B = 0;
};

static void fill_step(step_data & d, const draft_hp & hp, const std::vector<int32_t> & toks, const std::vector<int> & anchors,
                      int B, int64_t n_vocab) {
    const int T = (int) toks.size(), A = (int) anchors.size();
    const int64_t AB = (int64_t) A * B, R = (int64_t) A * (B - 1), Tk = T + AB;
    d.T = T; d.A = A; d.B = B;
    d.pos_ctx.resize(T);
    for (int t = 0; t < T; ++t) d.pos_ctx[t] = t;
    d.tok_blk.resize(AB); d.pos_blk.resize(AB); d.targets.resize(R);
    for (int k = 0; k < A; ++k) {
        const int a = anchors[k];
        for (int i = 0; i < B; ++i) {
            d.tok_blk[k * B + i] = i == 0 ? toks[a] : hp.mask_id;
            d.pos_blk[k * B + i] = a + i;
            if (i > 0) d.targets[k * (B - 1) + i - 1] = toks[a + i];
        }
    }
    // mask [Tk, AB]: row r (query) x key j
    d.mask.assign((size_t) Tk * AB, -INFINITY);
    for (int k = 0; k < A; ++k) {
        const int a = anchors[k];
        for (int i = 0; i < B; ++i) {
            float * row = d.mask.data() + (size_t) (k * B + i) * Tk;
            const int p = a + i;
            const int j0 = hp.n_swa > 0 ? std::max(0, p - hp.n_swa + 1) : 0;
            for (int j = j0; j < a; ++j) row[j] = 0.0f;
            for (int i2 = 0; i2 < B; ++i2) row[T + k * B + i2] = 0.0f;
        }
    }
    d.labels.resize(B - 1);
    for (int i = 1; i < B; ++i) {
        auto & lab = d.labels[i - 1];
        lab.assign((size_t) n_vocab * A, 0.0f);
        for (int k = 0; k < A; ++k) lab[(size_t) k * n_vocab + d.targets[(int64_t) k * (B - 1) + i - 1]] = 1.0f;
    }
}

static void upload_step(const graph_io & io, const step_data & d) {
    ggml_backend_tensor_set(io.feat,    d.feat.data(),    0, ggml_nbytes(io.feat));
    ggml_backend_tensor_set(io.pos_ctx, d.pos_ctx.data(), 0, ggml_nbytes(io.pos_ctx));
    ggml_backend_tensor_set(io.tok_blk, d.tok_blk.data(), 0, ggml_nbytes(io.tok_blk));
    ggml_backend_tensor_set(io.pos_blk, d.pos_blk.data(), 0, ggml_nbytes(io.pos_blk));
    ggml_backend_tensor_set(io.mask,    d.mask.data(),    0, ggml_nbytes(io.mask));
    for (size_t i = 0; i < io.labels.size(); ++i) ggml_backend_tensor_set(io.labels[i], d.labels[i].data(), 0, ggml_nbytes(io.labels[i]));
}

static std::vector<int> sample_anchors(const sample & s, int B, int n, std::mt19937 & rng) {
    // anchors from the last prompt token to the last position that still has B-1 targets after it
    const int lo = std::max(0, s.prompt_len - 1);
    const int hi = (int) s.toks.size() - B;   // inclusive
    std::vector<int> all;
    for (int a = lo; a <= hi; ++a) all.push_back(a);
    if ((int) all.size() > n) {
        std::shuffle(all.begin(), all.end(), rng);
        all.resize(n);
        std::sort(all.begin(), all.end());
    }
    return all;
}

// ---------------------------------------------------------------------------------------------------------
// replicas: one per device, each with its own king, drafter copy, ggml-opt context (forward + backward only)
// and AdamW state. Gradients are averaged across replicas on the host every optimizer step, clipped, and the
// same AdamW step is applied on every replica, so the copies stay identical.

struct replica {
    int id = 0;
    backends_t be;
    drafter D;
    king_t K;
    bool has_king = false;
    ggml_opt_context_t opt = nullptr;
    ggml_opt_result_t  res = nullptr;
    ggml_context * ctx_cur = nullptr;
    ggml_context * ctx_prev = nullptr;   // kept alive until the next alloc (ggml-opt still touches the old graph)
    std::vector<ggml_tensor *> grad_accs; // one per param, persistent in ggml-opt's static context
    // AdamW
    ggml_context * ctx_adam = nullptr;
    ggml_backend_buffer_t buf_adam = nullptr;
    std::vector<ggml_tensor *> m, v;
    ggml_context * ctx_adam_cpu = nullptr;
    ggml_backend_buffer_t buf_adam_cpu = nullptr;
    ggml_tensor * adam_params = nullptr;
    ggml_context * ctx_adam_graph = nullptr;
    ggml_cgraph * gf_adam = nullptr;
    // host mirror of the accumulators for the all-reduce
    std::vector<float> hbuf;
    std::vector<size_t> hoff;
    // per-step scratch
    step_data d;
    std::mt19937 rng;
    double loss_sum = 0.0, t_feat = 0.0;
    int n_seq = 0;
    size_t tokens = 0, anchors = 0;
};

struct trainer {
    options o;
    std::vector<replica> reps;
    int B = 0;
    int step = 0, total_steps = 1, iter = 1;
    float lr_now = 0.0f;
    std::mt19937 rng;
    float lr_at(int s) const {
        const int warm = std::max(1, (int) (o.warmup * total_steps));
        if (s < warm) return o.lr * (float) (s + 1) / (float) warm;
        const float t = total_steps > warm ? (float) (s - warm) / (float) (total_steps - warm) : 1.0f;
        return o.lr_min + 0.5f * (o.lr - o.lr_min) * (1.0f + cosf((float) M_PI * std::min(1.0f, t)));
    }
};

static ggml_opt_optimizer_params opt_pars_unused(void *) {
    return ggml_opt_get_default_optimizer_params(nullptr);
}

// DFLASH_DUMP=1: print every node's sum and max|x| as the graph runs (compare backends)
static bool dump_named(const char * name) {
    static const char * keys[] = { "noise_norm-", "attn_conv_in-", "Qcur-", "Kcur-", "Vcur-", "Kcur_injected-", "Vcur_injected-",
                                   "kqv_merged_cont-", "kqv_out-", "attn_conv_out-", "ffn_inp-", "ffn_norm-", "ffn_out-",
                                   "ffn_conv_out-", "l_out-", "result_norm", "inp_noise_embd", "enc_norm_out", "enc" };
    for (const char * k : keys) if (strncmp(name, k, strlen(k)) == 0) return true;
    return false;
}
static bool dump_cb(ggml_tensor * t, bool ask, void *) {
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->op == GGML_OP_NONE) return true;
    const char * mode = getenv("DFLASH_DUMP");
    if (mode && !strcmp(mode, "names") && !dump_named(ggml_get_name(t))) return true;
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    double sum = 0.0, amax = 0.0;
    for (float x : v) { sum += x; amax = std::max(amax, (double) fabsf(x)); }
    LOGI("node %-32s %-12s [%5" PRId64 ",%5" PRId64 ",%3" PRId64 "] sum % .6e amax %.6e\n",
         ggml_get_name(t), ggml_op_desc(t), t->ne[0], t->ne[1], t->ne[2], sum, amax);
    return true;
}

static void init_opt(replica & r) {
    if (getenv("DFLASH_DUMP")) ggml_backend_sched_set_eval_callback(r.be.sched, dump_cb, nullptr);
    ggml_opt_params p = ggml_opt_default_params(r.be.sched, GGML_OPT_LOSS_TYPE_SUM);
    p.ctx_compute = nullptr;       // dynamic graphs
    p.opt_period = 1 << 30;        // ggml-opt never applies an optimizer step; AdamW is ours (see adam_step)
    p.get_opt_pars = opt_pars_unused;
    p.build_type = GGML_OPT_BUILD_TYPE_GRAD;
    r.opt = ggml_opt_init(p);
    r.res = ggml_opt_result_init();
}

static void init_adam(replica & r) {
    const size_t n = r.D.params.size();
    ggml_init_params ip = { (2 * n + 8) * ggml_tensor_overhead(), nullptr, true };
    r.ctx_adam = ggml_init(ip);
    size_t total = 0;
    for (auto * p : r.D.params) {
        ggml_tensor * m = ggml_dup_tensor(r.ctx_adam, p); ggml_format_name(m, "adam_m.%s", ggml_get_name(p));
        ggml_tensor * v = ggml_dup_tensor(r.ctx_adam, p); ggml_format_name(v, "adam_v.%s", ggml_get_name(p));
        r.m.push_back(m); r.v.push_back(v);
        total += ggml_nelements(p);
    }
    r.buf_adam = ggml_backend_alloc_ctx_tensors(r.ctx_adam, r.be.gpu);
    if (!r.buf_adam) DIE("cannot allocate AdamW state");
    for (size_t i = 0; i < n; ++i) { ggml_backend_tensor_memset(r.m[i], 0, 0, ggml_nbytes(r.m[i])); ggml_backend_tensor_memset(r.v[i], 0, 0, ggml_nbytes(r.v[i])); }
    ggml_init_params ipc = { 4 * ggml_tensor_overhead(), nullptr, true };
    r.ctx_adam_cpu = ggml_init(ipc);
    r.adam_params = ggml_new_tensor_1d(r.ctx_adam_cpu, GGML_TYPE_F32, 7);
    ggml_set_name(r.adam_params, "adamw_params");
    ggml_set_input(r.adam_params);
    // on the training device itself: a host-side tensor would be copied by the scheduler on every step
    r.buf_adam_cpu = ggml_backend_alloc_ctx_tensors(r.ctx_adam_cpu, r.be.gpu);
    r.hoff.resize(n + 1);
    r.hoff[0] = 0;
    for (size_t i = 0; i < n; ++i) r.hoff[i + 1] = r.hoff[i] + ggml_nelements(r.D.params[i]);
    r.hbuf.resize(r.hoff[n]);
}

// builds the graph for (T, A), allocates it, uploads the inputs; the caller then calls run_step
static graph_io prepare_step(replica & r, const options & o, const step_data & d, bool backward) {
    ggml_init_params ip = { 64u * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 65536, true);
    graph_io io = build_graph(ctx, gf, r.D, d.T, d.A, d.B, o.gamma, 1.0f / (float) o.seqs_per_step);
    ggml_opt_prepare_alloc(r.opt, ctx, gf, io.feat, io.loss);
    ggml_opt_alloc(r.opt, backward);
    if (backward && r.grad_accs.empty()) {
        for (auto * p : r.D.params) {
            ggml_tensor * ga = ggml_opt_grad_acc(r.opt, p);
            if (!ga) DIE("no gradient accumulator for %s", ggml_get_name(p));
            r.grad_accs.push_back(ga);
        }
    }
    if (r.ctx_prev) ggml_free(r.ctx_prev);
    r.ctx_prev = r.ctx_cur;
    r.ctx_cur = ctx;
    upload_step(io, d);
    return io;
}

// the loss is scaled by 1/seqs_per_step in the graph, so the sum over a cycle is the mean micro-step loss
static float run_step(replica & r) {
    ggml_opt_result_reset(r.res);
    ggml_opt_eval(r.opt, r.res);
    double loss = 0.0;
    ggml_opt_result_loss(r.res, &loss, nullptr);
    return (float) loss;
}

static void zero_grads(replica & r) {
    for (auto * ga : r.grad_accs) ggml_backend_tensor_memset(ga, 0, 0, ggml_nbytes(ga));
}

static void download_grads(replica & r) {
    for (size_t i = 0; i < r.grad_accs.size(); ++i) {
        ggml_backend_tensor_get(r.grad_accs[i], r.hbuf.data() + r.hoff[i], 0, ggml_nbytes(r.grad_accs[i]));
    }
}
static void upload_grads(replica & r, const std::vector<float> & src) {
    for (size_t i = 0; i < r.grad_accs.size(); ++i) {
        ggml_backend_tensor_set(r.grad_accs[i], src.data() + r.hoff[i], 0, ggml_nbytes(r.grad_accs[i]));
    }
}

// average the accumulators over replicas (host), clip the global norm, push the result back everywhere
// returns the pre-clip gradient norm
static double all_reduce_grads(trainer & tr) {
    const int world = (int) tr.reps.size();
    const bool need_host = world > 1 || tr.o.clip > 0.0f;
    if (!need_host) return -1.0;
    std::vector<std::thread> th;
    for (auto & r : tr.reps) th.emplace_back([&r] { download_grads(r); });
    for (auto & t : th) t.join();
    th.clear();
    auto & acc = tr.reps[0].hbuf;
    const size_t n = acc.size();
    const int nth = std::max(1, std::min(tr.o.threads, 32));
    std::vector<double> sq(nth, 0.0);
    {
        std::vector<std::thread> w;
        for (int t = 0; t < nth; ++t) {
            w.emplace_back([&, t] {
                const size_t a = n * t / nth, b = n * (t + 1) / nth;
                const float inv = 1.0f / (float) world;
                double s = 0.0;
                for (size_t i = a; i < b; ++i) {
                    float g = acc[i];
                    for (int rr = 1; rr < world; ++rr) g += tr.reps[rr].hbuf[i];
                    g *= inv;
                    acc[i] = g;
                    s += (double) g * g;
                }
                sq[t] = s;
            });
        }
        for (auto & x : w) x.join();
    }
    double norm = 0.0;
    for (double x : sq) norm += x;
    norm = sqrt(norm);
    if (tr.o.clip > 0.0f && norm > tr.o.clip) {
        const float scale = (float) (tr.o.clip / (norm + 1e-6));
        for (size_t i = 0; i < n; ++i) acc[i] *= scale;
    }
    for (auto & r : tr.reps) th.emplace_back([&r, &acc] { upload_grads(r, acc); });
    for (auto & t : th) t.join();
    return norm;
}

static void adam_step(replica & r, const options & o, float lr, int iter) {
    if (!r.gf_adam) {
        ggml_init_params ip = { (size_t) (4 * r.D.params.size() + 64) * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true };
        r.ctx_adam_graph = ggml_init(ip);
        r.gf_adam = ggml_new_graph_custom(r.ctx_adam_graph, 4096, false);
        for (size_t i = 0; i < r.D.params.size(); ++i) {
            ggml_tensor * st = ggml_opt_step_adamw(r.ctx_adam_graph, r.D.params[i], r.grad_accs[i], r.m[i], r.v[i], r.adam_params);
            ggml_build_forward_expand(r.gf_adam, st);
        }
    }
    const float pars[7] = { lr, o.beta1, o.beta2, o.eps, o.wd,
                            1.0f / (1.0f - powf(o.beta1, (float) iter)), 1.0f / (1.0f - powf(o.beta2, (float) iter)) };
    ggml_backend_tensor_set(r.adam_params, pars, 0, sizeof(pars));
    ggml_backend_sched_reset(r.be.sched);
    if (!ggml_backend_sched_alloc_graph(r.be.sched, r.gf_adam)) DIE("cannot allocate the AdamW graph");
    ggml_backend_sched_graph_compute(r.be.sched, r.gf_adam);
    ggml_backend_sched_reset(r.be.sched);
}

// copy replica 0's weights to the others (float drift guard)
static void broadcast_weights(trainer & tr) {
    if (tr.reps.size() < 2) return;
    auto & r0 = tr.reps[0];
    std::vector<float> buf;
    for (size_t i = 0; i < r0.D.params.size(); ++i) {
        buf.resize(ggml_nelements(r0.D.params[i]));
        ggml_backend_tensor_get(r0.D.params[i], buf.data(), 0, buf.size() * sizeof(float));
        for (size_t rr = 1; rr < tr.reps.size(); ++rr) {
            ggml_backend_tensor_set(tr.reps[rr].D.params[i], buf.data(), 0, buf.size() * sizeof(float));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// held-out evaluation on replica 0: per-position accuracy, mean accepted length (matching prefix), loss

struct eval_result {
    std::vector<double> acc;   // per position 1..B-1
    double tau = 0.0;          // mean matching prefix length per block
    double loss = 0.0;
    int blocks = 0, samples = 0;
    double seconds = 0.0;
};

static eval_result evaluate(trainer & tr, const std::vector<sample> & held, int n_samples, int n_anchors) {
    replica & r = tr.reps[0];
    const int B = tr.B;
    eval_result res;
    res.acc.assign(B - 1, 0.0);
    std::mt19937 rng(777);
    step_data d;
    std::vector<int32_t> pred;
    const double t0 = now_s();
    double loss_sum = 0.0;
    const int n = std::min<int>(n_samples, (int) held.size());
    for (int si = 0; si < n; ++si) {
        const sample & s = held[si];
        std::vector<int> anchors = sample_anchors(s, B, n_anchors, rng);
        if (anchors.empty()) continue;
        if (r.has_king) king_features(r.K, s.toks, d.feat);
        fill_step(d, r.D.hp, s.toks, anchors, B, r.D.hp.n_vocab);
        graph_io io = prepare_step(r, tr.o, d, false);
        const float loss = run_step(r) * (float) tr.o.seqs_per_step;
        const int A = (int) anchors.size();
        pred.resize((size_t) A * (B - 1));
        for (int i = 1; i < B; ++i) {
            std::vector<int32_t> pi(A);
            ggml_backend_tensor_get(io.pred[i - 1], pi.data(), 0, ggml_nbytes(io.pred[i - 1]));
            for (int k = 0; k < A; ++k) pred[(size_t) k * (B - 1) + i - 1] = pi[k];
        }
        for (int k = 0; k < A; ++k) {
            bool prefix = true;
            for (int i = 1; i < B; ++i) {
                const int64_t rr = (int64_t) k * (B - 1) + i - 1;
                const bool ok = pred[rr] == d.targets[rr];
                if (ok) res.acc[i - 1] += 1.0;
                if (prefix && ok) res.tau += 1.0; else prefix = false;
            }
        }
        res.blocks += A;
        loss_sum += loss;
        res.samples += 1;
    }
    if (res.blocks) { for (auto & a : res.acc) a /= res.blocks; res.tau /= res.blocks; }
    if (res.samples) res.loss = loss_sum / res.samples;
    res.seconds = now_s() - t0;
    return res;
}

static void print_eval(const eval_result & r, int B) {
    LOGI("heldout: %d samples, %d blocks, loss %.4f, mean accepted %.3f of %d (acceptance %.1f%%), %.1fs\n",
         r.samples, r.blocks, r.loss, r.tau, B - 1, 100.0 * r.tau / (B - 1), r.seconds);
    LOGI("  per position:");
    for (int i = 0; i < B - 1; ++i) LOGI(" p%d=%.3f", i + 1, r.acc[i]);
    LOGI("\n");
}

// ---------------------------------------------------------------------------------------------------------
// self-check: the training graph against the fork's serving graph (llama API, same king features)

static void selfcheck(trainer & tr, const std::vector<sample> & held, int n_blocks) {
    replica & r = tr.reps[0];
    const draft_hp & hp = r.D.hp;
    const int B = tr.B;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = tr.o.n_gpu_layers;
    ggml_backend_dev_t devs[2] = { r.be.dev, nullptr };
    if (!r.be.gpu_is_cpu) { mp.devices = devs; mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.main_gpu = 0; }
    llama_model * mdft = llama_model_load_from_file(tr.o.draft_path.c_str(), mp);
    if (!mdft) DIE("selfcheck: cannot load the drafter through llama");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = tr.o.seq_max + B + 8;
    cp.n_batch = tr.o.seq_max;
    cp.n_ubatch = tr.o.seq_max;
    cp.n_seq_max = 1;
    cp.n_threads = tr.o.threads;
    cp.n_threads_batch = tr.o.threads;
    cp.ctx_other = r.K.ctx;
    if (getenv("DFLASH_DUMP")) { cp.cb_eval = dump_cb; cp.cb_eval_user_data = nullptr; }
    llama_context * cdft = llama_init_from_model(mdft, cp);
    if (!cdft) DIE("selfcheck: cannot create the drafter context");
    llama_set_embeddings_nextn(cdft, true, /*masked*/ false);
    llama_set_causal_attn(cdft, false);
    const int n_embd_dec = llama_model_n_embd_out(mdft);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(mdft));
    if (n_vocab != hp.n_vocab) LOGI("selfcheck: vocab %d (llama) vs %d (trainer)\n", n_vocab, hp.n_vocab);

    std::mt19937 rng(4242);
    step_data d;
    std::vector<float> mine;
    llama_batch inj = llama_batch_init(tr.o.seq_max, n_embd_dec, 1);
    llama_batch blk = llama_batch_init(B, 0, 1);
    int done = 0, argmax_agree = 0, rows = 0;
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t si = 0; si < held.size() && done < n_blocks; ++si) {
        const sample & s = held[si];
        std::vector<int> anchors = sample_anchors(s, B, 1, rng);
        if (anchors.empty()) continue;
        const int a = anchors[0];
        king_features(r.K, s.toks, d.feat);

        fill_step(d, hp, s.toks, anchors, B, hp.n_vocab);
        graph_io io = prepare_step(r, tr.o, d, false);
        run_step(r);
        mine.resize((size_t) (B - 1) * hp.n_vocab);
        for (int i = 1; i < B; ++i) {
            ggml_backend_tensor_get(io.logits[i - 1], mine.data() + (size_t) (i - 1) * hp.n_vocab, 0, ggml_nbytes(io.logits[i - 1]));
        }

        // serving graph: encode features of positions < a, inject, decode the block
        llama_memory_clear(llama_get_memory(cdft), true);
        llama_batch enc = { a, nullptr, d.feat.data(), nullptr, nullptr, nullptr, nullptr };
        if (llama_encode(cdft, enc) != 0) DIE("selfcheck: llama_encode failed");
        const float * g = llama_get_embeddings_nextn(cdft);
        if (!g) DIE("selfcheck: encoder produced no output");
        {
            // encoder output: mine (enc [n_embd, T], rows < a) against the serving encoder
            std::vector<float> enc(ggml_nelements(io.enc));
            ggml_backend_tensor_get(io.enc, enc.data(), 0, ggml_nbytes(io.enc));
            double mad = 0.0, scale = 0.0;
            for (int t = 0; t < a; ++t) for (int e = 0; e < hp.n_embd; ++e) {
                const float mine_v = enc[(size_t) t * hp.n_embd + e], theirs_v = g[(size_t) t * n_embd_dec + e];
                mad = std::max(mad, (double) fabsf(mine_v - theirs_v));
                scale = std::max(scale, (double) fabsf(theirs_v));
            }
            LOGI("selfcheck: encoder rows<%d max |diff| %.4g (scale %.4g), n_embd_dec %d, feat[0..3] %.4f %.4f %.4f\n",
                 a, mad, scale, n_embd_dec, d.feat[0], d.feat[1], d.feat[2]);
        }
        inj.n_tokens = a;
        memcpy(inj.embd, g, (size_t) a * n_embd_dec * sizeof(float));
        for (int i = 0; i < a; ++i) { inj.pos[i] = i; inj.n_seq_id[i] = 1; inj.seq_id[i][0] = 0; inj.logits[i] = 0; }
        if (llama_decode(cdft, inj) != 0) DIE("selfcheck: inject decode failed");
        blk.n_tokens = B;
        for (int i = 0; i < B; ++i) {
            blk.token[i] = i == 0 ? s.toks[a] : hp.mask_id;
            blk.pos[i] = a + i; blk.n_seq_id[i] = 1; blk.seq_id[i][0] = 0; blk.logits[i] = 1;
        }
        if (llama_decode(cdft, blk) != 0) DIE("selfcheck: block decode failed");
        for (int i = 1; i < B; ++i) {
            const float * theirs = llama_get_logits_ith(cdft, i);
            if (!theirs) DIE("selfcheck: no logits for row %d", i);
            const float * ours = mine.data() + (size_t) (i - 1) * hp.n_vocab;
            int am_o = 0, am_t = 0;
            double scale = 0.0, mad = 0.0;
            for (int vv = 0; vv < hp.n_vocab; ++vv) {
                if (ours[vv] > ours[am_o]) am_o = vv;
                if (theirs[vv] > theirs[am_t]) am_t = vv;
                scale = std::max(scale, (double) fabsf(theirs[vv]));
                mad = std::max(mad, (double) fabsf(ours[vv] - theirs[vv]));
            }
            max_abs = std::max(max_abs, mad);
            if (scale > 0) max_rel = std::max(max_rel, mad / scale);
            argmax_agree += am_o == am_t;
            rows++;
            if (done == 0 && i <= 2) {
                LOGI("  row %d: mine argmax %d (%.3f) theirs argmax %d (%.3f); mine at theirs %.3f, theirs at mine %.3f; target %d\n",
                     i, am_o, ours[am_o], am_t, theirs[am_t], ours[am_t], theirs[am_o], d.targets[i - 1]);
            }
        }
        done++;
        LOGI("selfcheck block %d: sample %zu anchor %d, max |diff| so far %.4g (rel %.3g), argmax agree %d/%d\n",
             done, si, a, max_abs, max_rel, argmax_agree, rows);
    }
    LOGI("selfcheck: %d blocks, %d rows, argmax agreement %.1f%%, max abs diff %.4g, max rel diff %.3g\n",
         done, rows, rows ? 100.0 * argmax_agree / rows : 0.0, max_abs, max_rel);
    LOGI("  (f16 embeddings and an f32 head here vs the king's quantized tensors there: expect small diffs, not argmax flips)\n");
    llama_batch_free(inj);
    llama_batch_free(blk);
    llama_free(cdft);
    llama_model_free(mdft);
}

// ---------------------------------------------------------------------------------------------------------
// gradient check on a toy drafter: ggml backward (averaged over replicas, same data) vs central finite differences

static void gradcheck(trainer & tr) {
    replica & r0 = tr.reps[0];
    drafter & D = r0.D;
    if (!D.tok_embd || !D.lm_head) DIE("gradcheck needs a drafter that carries token_embd.weight and output.weight (make_toy_drafter.py)");
    const int B = tr.B, T = 12;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<int32_t> toks(T);
    for (auto & t : toks) t = (int32_t) (rng() % (uint32_t) D.hp.n_vocab);
    std::vector<int> anchors = { 2, T - B };
    step_data d;
    d.feat.resize((size_t) T * D.hp.n_cap() * D.hp.n_embd);
    for (auto & f : d.feat) f = nd(rng);
    fill_step(d, D.hp, toks, anchors, B, D.hp.n_vocab);

    float loss0 = 0.0f;
    for (auto & r : tr.reps) { prepare_step(r, tr.o, d, true); loss0 = run_step(r); }
    const float clip_saved = tr.o.clip;
    tr.o.clip = 0.0f;
    all_reduce_grads(tr);          // averages identical gradients: must leave them unchanged
    tr.o.clip = clip_saved;
    LOGI("gradcheck: loss %.6f, %zu trainable tensors, %zu replicas\n", loss0, D.params.size(), tr.reps.size());
    struct probe { ggml_tensor * t; int64_t idx; float g; };
    std::vector<probe> probes;
    for (size_t ip = 0; ip < D.params.size(); ++ip) {
        ggml_tensor * p = D.params[ip];
        std::vector<float> g(ggml_nelements(p));
        ggml_backend_tensor_get(r0.grad_accs[ip], g.data(), 0, g.size() * sizeof(float));
        for (int k = 0; k < 2; ++k) {
            const int64_t idx = (int64_t) (rng() % (uint32_t) g.size());
            probes.push_back({ p, idx, g[idx] });
        }
    }
    auto forward = [&]() {
        prepare_step(r0, tr.o, d, false);
        return run_step(r0);
    };
    {
        // the forward-only graph must reproduce the loss of the forward+backward graph
        const float f1 = forward(), f2 = forward();
        LOGI("gradcheck: forward-only loss %.6f %.6f vs backward graph %.6f\n", f1, f2, loss0);
    }
    int bad = 0;
    for (auto & pr : probes) {
        float v;
        ggml_backend_tensor_get(pr.t, &v, pr.idx * sizeof(float), sizeof(float));
        auto central = [&](float h) {
            float vp = v + h, vm = v - h;
            ggml_backend_tensor_set(pr.t, &vp, pr.idx * sizeof(float), sizeof(float));
            const float lp = forward();
            ggml_backend_tensor_set(pr.t, &vm, pr.idx * sizeof(float), sizeof(float));
            const float lm = forward();
            ggml_backend_tensor_set(pr.t, &v, pr.idx * sizeof(float), sizeof(float));
            return (double) (lp - lm) / (2.0 * h);
        };
        const float h = std::max(4e-3f, 4e-2f * fabsf(v));
        const double fd1 = central(h), fd2 = central(2 * h);
        const double fd = (4.0 * fd1 - fd2) / 3.0;
        const double err = fabs(fd - pr.g);
        const double tol = 2e-2 * std::max(fabs(fd), (double) fabs(pr.g)) + 2e-4;
        const bool ok = err <= tol;
        if (!ok) bad++;
        LOGI("  %-28s [%6" PRId64 "] analytic % .6e  fd % .6e  %s\n", ggml_get_name(pr.t), pr.idx, pr.g, fd, ok ? "ok" : "MISMATCH");
        if (!ok) LOGI("      v=% .5f h=%.2e fd(h)=% .6e fd(2h)=% .6e fd(4h)=% .6e\n", v, h, fd1, fd2, central(4 * h));
    }
    // one AdamW step on every replica must leave the replicas identical
    for (auto & r : tr.reps) adam_step(r, tr.o, 1e-3f, 1);
    double max_dev = 0.0;
    if (tr.reps.size() > 1) {
        std::vector<float> a, b;
        for (size_t ip = 0; ip < D.params.size(); ++ip) {
            a.resize(ggml_nelements(D.params[ip])); b.resize(a.size());
            ggml_backend_tensor_get(D.params[ip], a.data(), 0, a.size() * sizeof(float));
            ggml_backend_tensor_get(tr.reps[1].D.params[ip], b.data(), 0, b.size() * sizeof(float));
            for (size_t i = 0; i < a.size(); ++i) max_dev = std::max(max_dev, (double) fabsf(a[i] - b[i]));
        }
        LOGI("gradcheck: after one AdamW step replicas differ by at most %.3g\n", max_dev);
    }
    LOGI("gradcheck: %zu probes, %d mismatches -> %s\n", probes.size(), bad, (bad || max_dev > 0) ? "FAILED" : "PASSED");
    if (bad || max_dev > 0) exit(2);
}

// ---------------------------------------------------------------------------------------------------------
// main

int main(int argc, char ** argv) {
    options o = parse_args(argc, argv);
    setenv("GGML_CUDA_NO_TF32", "1", /*overwrite*/ 0);   // f32 GEMMs stay f32 on CUDA (see ggml-cuda/common.cuh)
    llama_backend_init();
    trainer tr;
    tr.o = o;
    tr.rng.seed(o.seed);

    std::vector<ggml_backend_dev_t> devs = pick_devices(o);
    tr.reps.resize(devs.size());
    const int threads_each = std::max(1, o.threads / (int) devs.size());
    for (size_t i = 0; i < devs.size(); ++i) {
        replica & r = tr.reps[i];
        r.id = (int) i;
        r.rng.seed(o.seed + 1000 * (int) i + 1);
        r.be = init_backends(devs[i], threads_each);
        load_drafter(r.D, o, r.be.gpu);
        if (!o.king_path.empty()) {
            init_king(r.K, o, r.D.hp, devs[i]);
            load_shared_from_king(r.D, o.king_path, r.be.gpu, r.K.model, threads_each);
            r.has_king = true;
        }
        LOGI("replica %zu on %s (%s)\n", i, ggml_backend_dev_name(devs[i]), ggml_backend_dev_description(devs[i]));
    }
    drafter & D0 = tr.reps[0].D;
    tr.B = o.block > 0 ? o.block : D0.hp.block;
    if (tr.B > D0.hp.block) LOGI("warning: --block %d exceeds the drafter's trained block size %d\n", tr.B, D0.hp.block);
    if (tr.B < 2) DIE("block must be at least 2");
    if (D0.hp.mask_id < 0) DIE("the drafter has no mask token id (tokenizer.ggml.mask_token_id)");

    if (o.gradcheck) {
        o.seqs_per_step = 1; tr.o.seqs_per_step = 1;
        for (auto & r : tr.reps) { init_opt(r); init_adam(r); }
        gradcheck(tr);
        return 0;
    }

    std::vector<sample> held;
    if (!o.heldout_path.empty()) {
        size_t skipped = 0;
        load_shard(o.heldout_path, o.seq_max, tr.B, held, &skipped);
        LOGI("heldout: %zu samples (%zu skipped as too long)\n", held.size(), skipped);
    }
    if (o.selfcheck > 0) {
        if (held.empty()) DIE("--selfcheck needs --heldout");
        if (!tr.reps[0].has_king) DIE("--selfcheck needs -m king.gguf");
        init_opt(tr.reps[0]);
        selfcheck(tr, held, o.selfcheck);
        return 0;
    }
    if (o.check > 0) {
        if (held.empty()) DIE("--check needs --heldout");
        init_opt(tr.reps[0]);
        eval_result r = evaluate(tr, held, o.check, o.eval_anchors);
        print_eval(r, tr.B);
        return 0;
    }

    // ---- training
    if (o.train_paths.empty()) DIE("--train is required");
    if (!tr.reps[0].has_king) DIE("training needs -m king.gguf");
    fs::create_directories(o.out_dir);
    pool_t pool;
    rescan_pool(pool, o, tr.B, tr.rng);
    if (pool.samples.empty()) DIE("no training samples found");
    const int world = (int) tr.reps.size();
    tr.total_steps = o.max_steps > 0 ? o.max_steps : std::max(1, (int) (o.epochs * pool.samples.size() / ((size_t) o.seqs_per_step * world)));
    LOGI("training: %zu samples, %d replicas x %d seqs/step, %d anchors/seq, block %d, %d steps planned, lr %g -> %g, warmup %.0f%%, clip %g\n",
         pool.samples.size(), world, o.seqs_per_step, o.anchors, tr.B, tr.total_steps, o.lr, o.lr_min, 100.0 * o.warmup, o.clip);
    for (auto & r : tr.reps) { init_opt(r); init_adam(r); }

    metrics_server metrics(o.metrics_port);
    std::ofstream log(o.out_dir + "/train.jsonl", std::ios::app);

    if (!held.empty()) {
        eval_result r = evaluate(tr, held, o.eval_samples, o.eval_anchors);
        print_eval(r, tr.B);
        metrics.set_eval(r.loss, r.tau, r.acc, 0);
        log << json{{"step", 0}, {"heldout_loss", r.loss}, {"heldout_tau", r.tau}, {"heldout_acc", r.acc}}.dump() << "\n" << std::flush;
    }

    std::mutex pool_mu;
    int epoch = 0;
    auto next_sample = [&]() -> const sample * {
        std::lock_guard<std::mutex> lk(pool_mu);
        if (pool.cursor >= pool.order.size()) {
            epoch++;
            if (o.rescan_every > 0) rescan_pool(pool, o, tr.B, tr.rng);
            std::shuffle(pool.order.begin(), pool.order.end(), tr.rng);
            pool.cursor = 0;
        }
        return &pool.samples[pool.order[pool.cursor++]];
    };

    const double t_train0 = now_s();
    size_t tokens_seen = 0, anchors_seen = 0;
    for (tr.step = 0; tr.step < tr.total_steps; ++tr.step) {
        tr.lr_now = tr.lr_at(tr.step);
        const double t0 = now_s();
        std::vector<std::thread> th;
        for (auto & r : tr.reps) {
            th.emplace_back([&] {
                r.loss_sum = 0.0; r.t_feat = 0.0; r.n_seq = 0;
                for (int s = 0; s < o.seqs_per_step; ++s) {
                    const sample * smp = next_sample();
                    std::vector<int> anchors = sample_anchors(*smp, tr.B, o.anchors, r.rng);
                    if (anchors.empty()) { s--; continue; }
                    const double tf = now_s();
                    king_features(r.K, smp->toks, r.d.feat);
                    r.t_feat += now_s() - tf;
                    fill_step(r.d, r.D.hp, smp->toks, anchors, tr.B, r.D.hp.n_vocab);
                    prepare_step(r, o, r.d, true);
                    r.loss_sum += run_step(r);
                    r.tokens += smp->toks.size();
                    r.anchors += anchors.size();
                    r.n_seq++;
                }
            });
        }
        for (auto & t : th) t.join();
        const double t_fwd = now_s() - t0;
        const double gnorm = all_reduce_grads(tr);
        th.clear();
        for (auto & r : tr.reps) th.emplace_back([&] { adam_step(r, o, tr.lr_now, tr.iter); zero_grads(r); });
        for (auto & t : th) t.join();
        tr.iter++;
        double loss = 0.0, t_feat = 0.0;
        tokens_seen = 0; anchors_seen = 0;
        for (auto & r : tr.reps) { loss += r.loss_sum; t_feat = std::max(t_feat, r.t_feat); tokens_seen += r.tokens; anchors_seen += r.anchors; }
        loss /= world;   // each replica's sum is already its mean micro-step loss
        const double dt = now_s() - t0;
        LOGI("step %d/%d  loss %.4f  gnorm %.3f  lr %.2e  %.1fs (compute %.1fs, king %.1fs)  epoch %d  tokens %.2fM\n",
             tr.step + 1, tr.total_steps, loss, gnorm, tr.lr_now, dt, t_fwd, t_feat, epoch, tokens_seen / 1e6);
        metrics.set_step(tr.step + 1, loss, tr.lr_now, dt, t_feat, tokens_seen, anchors_seen, epoch);
        log << json{{"step", tr.step + 1}, {"loss", loss}, {"gnorm", gnorm}, {"lr", tr.lr_now}, {"seconds", dt}, {"compute_seconds", t_fwd},
                    {"feat_seconds", t_feat}, {"tokens", tokens_seen}, {"anchors", anchors_seen}, {"epoch", epoch}, {"elapsed", now_s() - t_train0}}.dump() << "\n" << std::flush;

        if (o.rescan_every > 0 && (tr.step + 1) % o.rescan_every == 0) { std::lock_guard<std::mutex> lk(pool_mu); rescan_pool(pool, o, tr.B, tr.rng); }
        if (o.sync_every > 0 && (tr.step + 1) % o.sync_every == 0) broadcast_weights(tr);
        const bool last = tr.step + 1 == tr.total_steps;
        if (!held.empty() && o.eval_every > 0 && ((tr.step + 1) % o.eval_every == 0 || last)) {
            eval_result r = evaluate(tr, held, o.eval_samples, o.eval_anchors);
            print_eval(r, tr.B);
            metrics.set_eval(r.loss, r.tau, r.acc, tr.step + 1);
            log << json{{"step", tr.step + 1}, {"heldout_loss", r.loss}, {"heldout_tau", r.tau}, {"heldout_acc", r.acc}}.dump() << "\n" << std::flush;
        }
        if (o.save_every > 0 && ((tr.step + 1) % o.save_every == 0 || last)) {
            char name[64];
            snprintf(name, sizeof(name), "/ckpt-%06d.gguf", tr.step + 1);
            save_drafter(D0, o.out_dir + name, o.save_type);
            fs::copy_file(o.out_dir + name, o.out_dir + "/latest.gguf", fs::copy_options::overwrite_existing);
        }
    }
    LOGI("done: %d steps, %.2fM tokens, %.1f min\n", tr.total_steps, tokens_seen / 1e6, (now_s() - t_train0) / 60.0);
    return 0;
}
