#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "gemv_kernels.h"

void usage(const char* exe) {
    std::cout << "Usage: " << exe
              << " [--m M] [--n N] [--iters I] "
              << "[--mode omp|avx512|amx|onednn|all]\n";
}

int main(int argc, char** argv) {
    int M = 1024;
    int N = 1024;
    int iters = 100;
    std::string mode = "all";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--m" && i + 1 < argc) {
            M = std::stoi(argv[++i]);
        } else if (arg == "--n" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        }
    }

    __builtin_cpu_init();
    const bool has_avx512 = __builtin_cpu_supports("avx512bf16");
    const bool has_amx = __builtin_cpu_supports("amx_bf16");

    std::cout << "GEMV BF16 M=" << M << " N=" << N << " iters=" << iters
              << " mode=" << mode << "\n";
    std::cout << "CPU ISA: AVX512BF16=" << has_avx512
              << " AMX_BF16=" << has_amx << "\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<bf16> A(static_cast<size_t>(M) * N);
    std::vector<bf16> x(static_cast<size_t>(N));
    for (auto& v : A) v = bf16(dist(rng));
    for (auto& v : x) v = bf16(dist(rng));

    std::vector<float> ref_y;
    auto run_kernel = [&](const std::string& name,
                          auto&& fn,
                          bool supported) {
        if (!supported) {
            std::cout << "Skip " << name << " (ISA not supported)\n";
            return std::vector<float>{};
        }
        std::vector<float> y(M);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) {
            fn(A.data(), x.data(), y.data(), M, N);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double gflops = (2.0 * M * N) / (ms * 1e6);
        float checksum = std::accumulate(y.begin(), y.end(), 0.0f);
        float diff = ref_y.empty() ? 0.0f : max_abs_diff(y, ref_y);
        std::cout << name << ": " << ms << " ms/iter, " << gflops
                  << " GFLOPS, checksum=" << checksum;
        if (!ref_y.empty()) std::cout << " max|diff|=" << diff;
        std::cout << "\n";
        return y;
    };

    const bool run_all = mode == "all";
    if (run_all || mode == "omp") {
        ref_y = run_kernel("omp", gemv_omp, true);
    } else {
        ref_y.resize(M);
        gemv_omp(A.data(), x.data(), ref_y.data(), M, N);
    }
    if (run_all || mode == "avx512") {
        run_kernel("avx512_dpbf16", gemv_avx512, has_avx512);
    }
    if (run_all || mode == "amx") {
        run_kernel("amx_dpbf16", gemv_amx, has_amx);
    }
    if (run_all || mode == "onednn") {
        run_kernel("onednn_matmul", gemv_bf16_onednn, true);
    }

    return 0;
}
