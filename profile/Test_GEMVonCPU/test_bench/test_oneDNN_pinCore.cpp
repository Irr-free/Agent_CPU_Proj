#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
#include <oneapi/dnnl/dnnl.hpp>
#include <tbb/task_scheduler_observer.h>
#include <atomic>
#include <pthread.h>
#include <sched.h>
#include <thread>

#include "../gemv_kernels.h"

int main(int argc, char** argv) {
    // Pin oneTBB worker threads to distinct CPUs to avoid migration jitter.
    struct PinObserver : public tbb::task_scheduler_observer {
        std::vector<int> cpus;
        std::atomic<int> next{0};
        std::mutex map_mu;
        std::unordered_map<std::thread::id, int> assigned;

        PinObserver() {
            const int n = std::max(1u, std::thread::hardware_concurrency());
            cpus.resize(n);
            for (int i = 0; i < n; ++i) cpus[i] = i;
            observe(true);
        }
        void on_scheduler_entry(bool) override {
            if (cpus.empty()) return;
            const auto tid = std::this_thread::get_id();
            int core;
            {
                std::lock_guard<std::mutex> lock(map_mu);
                auto it = assigned.find(tid);
                if (it == assigned.end()) {
                    core = next.fetch_add(1, std::memory_order_relaxed) % cpus.size();
                    assigned.emplace(tid, core);
                } else {
                    core = it->second;
                }
            }
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus[core], &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
    };
    static PinObserver pin_observer;

    int M = 128;   // seq_len
    int N = 128;    // head_dim
    int B = 32;     // batch
    int iters = 20000;
    int warmup = 20; // default warmup iterations (not timed)

    // Usage: ./a.out [M] [N] [B] [iters]
    if (argc >= 2) M = std::stoi(argv[1]);
    if (argc >= 3) N = std::stoi(argv[2]);
    if (argc >= 4) B = std::stoi(argv[3]);
    if (argc >= 5) iters = std::stoi(argv[4]);
    warmup = std::min(std::max(5, iters / 20), std::max(20, iters)); // at least 5, ~5% of iters, cap reasonable

    std::cout << "Batched GEMV test (oneDNN)\n"
              << "M(seq_len)=" << M << " N(head_dim)=" << N
              << " B(batch)=" << B << " iters=" << iters << "\n";

    static_assert(sizeof(bf16) == 2, "bf16 must be 2 bytes to be compatible with oneDNN bf16");
    static_assert(std::is_trivially_copyable_v<bf16>, "bf16 should be trivially copyable");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // K: [B, M, N]
    std::vector<bf16> K_all(static_cast<size_t>(B) * M * N);
    for (auto& v : K_all) v = bf16(dist(rng));

    // q: [B, N]
    std::vector<bf16> q_all(static_cast<size_t>(B) * N);
    for (auto& v : q_all) v = bf16(dist(rng));

    // Batched implementation under test.
    std::vector<float> y(static_cast<size_t>(B) * M);

    // Warmup (not timed): build primitives, pin threads, fault pages.
    for (int it = 0; it < warmup; ++it) {
        gemv_bf16_onednn_batched(K_all.data(), q_all.data(), y.data(), B, M, N);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        gemv_bf16_onednn_batched(K_all.data(), q_all.data(), y.data(), B, M, N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double gflops = (2.0 * M * N * B) / (ms * 1e6);
    float checksum = std::accumulate(y.begin(), y.end(), 0.0f);

    std::cout << "bf16_onednn_batched: " << ms << " ms/iter, "
              << gflops << " GFLOPS, checksum=" << checksum << "\n";

    return 0;
}
