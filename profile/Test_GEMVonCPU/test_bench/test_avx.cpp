#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "../gemv_kernels.h"

int main(int argc, char** argv) {
    int M = 4096;
    int N = 4096;
    int iters = 1000;

    // std::cout << "GEMV BF16 (oneDNN tests) M=" << M << " N=" << N
    //           << " iters=" << iters << "\n";
    // __builtin_cpu_init();
    // const bool has_avx512 = __builtin_cpu_supports("avx512bf16");
    // std::cout << "CPU ISA: AVX512BF16=" << has_avx512 << "\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<bf16> A(static_cast<size_t>(M) * N);
    for (auto& v : A) v = bf16(dist(rng));

    // x_ax for y = A * x, length N; x_xa for y = x * A, length M.
    std::vector<bf16> x_ax(static_cast<size_t>(N));
    for (auto& v : x_ax) v = bf16(dist(rng));

    std::vector<float> ref_y_ax;

    auto run_kernel_ax = [&](const std::string& name, auto&& fn, bool supported) {
        if (!supported) {
            std::cout << "Skip " << name << " (not supported)\n";
            return std::vector<float>{};
        }
        std::vector<float> y(M);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) {
            fn(A.data(), x_ax.data(), y.data(), M, N);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double gflops = (2.0 * M * N) / (ms * 1e6);
        float checksum = std::accumulate(y.begin(), y.end(), 0.0f);
        float diff = ref_y_ax.empty() ? 0.0f : max_abs_diff(y, ref_y_ax);
        std::cout << name << " (A*x): " << ms << " ms/iter, " << gflops
                  << " GFLOPS, checksum=" << checksum;
        if (!ref_y_ax.empty()) std::cout << " max|diff|=" << diff;
        std::cout << "\n";
        return y;
    };

    // Baselines
    ref_y_ax = run_kernel_ax("bf16_onednn", gemv_avx512, true);

    return 0;
}
