// Prometheus text endpoint for llama-dflash-train (scraped by the Grafana stack in grafana/)
#pragma once

#include <cpp-httplib/httplib.h>

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct metrics_server {
    int port = 0;
    httplib::Server srv;
    std::thread th;
    std::mutex mu;
    // training
    int    step = 0, epoch = 0;
    double loss = 0, lr = 0, step_s = 0, feat_s = 0;
    size_t tokens = 0, anchors = 0;
    // held-out
    int    eval_step = -1;
    double eval_loss = 0, eval_tau = 0;
    std::vector<double> eval_acc;
    double started = 0;

    explicit metrics_server(int port_) : port(port_) {
        if (port <= 0) return;
        srv.Get("/metrics", [this](const httplib::Request &, httplib::Response & res) {
            res.set_content(render(), "text/plain; version=0.0.4");
        });
        srv.Get("/", [](const httplib::Request &, httplib::Response & res) {
            res.set_content("llama-dflash-train: see /metrics\n", "text/plain");
        });
        th = std::thread([this] {
            if (!srv.listen("0.0.0.0", port)) fprintf(stderr, "metrics: cannot listen on port %d\n", port);
        });
        fprintf(stderr, "metrics: http://0.0.0.0:%d/metrics\n", port);
    }
    ~metrics_server() {
        if (port <= 0) return;
        srv.stop();
        if (th.joinable()) th.join();
    }
    void set_step(int s, double l, double lr_, double dt, double tf, size_t tok, size_t anc, int ep) {
        std::lock_guard<std::mutex> lk(mu);
        step = s; loss = l; lr = lr_; step_s = dt; feat_s = tf; tokens = tok; anchors = anc; epoch = ep;
    }
    void set_eval(double l, double tau, const std::vector<double> & acc, int at_step) {
        std::lock_guard<std::mutex> lk(mu);
        eval_loss = l; eval_tau = tau; eval_acc = acc; eval_step = at_step;
    }
    std::string render() {
        std::lock_guard<std::mutex> lk(mu);
        std::string s;
        char b[256];
        auto line = [&](const char * name, const char * help, const char * type, double v) {
            snprintf(b, sizeof(b), "# HELP %s %s\n# TYPE %s %s\n%s %.10g\n", name, help, name, type, name, v);
            s += b;
        };
        line("dflash_train_step", "optimizer steps done", "gauge", step);
        line("dflash_train_epoch", "epochs started", "gauge", epoch);
        line("dflash_train_loss", "weighted cross-entropy per block row, last step", "gauge", loss);
        line("dflash_train_lr", "learning rate", "gauge", lr);
        line("dflash_train_step_seconds", "wall seconds of the last optimizer step", "gauge", step_s);
        line("dflash_train_feat_seconds", "seconds of the last step spent in the king's forward", "gauge", feat_s);
        line("dflash_train_tokens_total", "sequence tokens seen", "counter", (double) tokens);
        line("dflash_train_anchors_total", "anchors (blocks) trained", "counter", (double) anchors);
        if (eval_step >= 0) {
            line("dflash_heldout_loss", "held-out weighted loss", "gauge", eval_loss);
            line("dflash_heldout_tau", "held-out mean accepted tokens per block", "gauge", eval_tau);
            line("dflash_heldout_step", "step of the last held-out evaluation", "gauge", eval_step);
            s += "# HELP dflash_heldout_acc held-out per-position accuracy\n# TYPE dflash_heldout_acc gauge\n";
            for (size_t i = 0; i < eval_acc.size(); ++i) {
                snprintf(b, sizeof(b), "dflash_heldout_acc{pos=\"%zu\"} %.6f\n", i + 1, eval_acc[i]);
                s += b;
            }
        }
        return s;
    }
};
