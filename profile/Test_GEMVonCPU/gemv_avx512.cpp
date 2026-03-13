#include "gemv_kernels.h"

inline float hsum512_ps(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 sum = _mm256_add_ps(lo, hi);

    __m128 lo128 = _mm256_castps256_ps128(sum);
    __m128 hi128 = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(lo128, hi128);

    __m128 shuf = _mm_movehdup_ps(sum128);
    sum128 = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sum128);
    sum128 = _mm_add_ss(sum128, shuf);
    return _mm_cvtss_f32(sum128);
}

// AVX-512 BF16 GEMV using dpbf16.
// with OpenMP parallelization over rows.
void gemv_avx512(const bf16* A, const bf16* x, float* y, int M, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        const bf16* row = A + static_cast<size_t>(i) * N;
        __m512 acc = _mm512_setzero_ps();
        int j = 0;
        for (; j + 32 <= N; j += 32) {
            __m512bh a = (__m512bh)_mm512_loadu_si512((const void*)(row + j));
            __m512bh b = (__m512bh)_mm512_loadu_si512((const void*)(x + j));
            acc = _mm512_dpbf16_ps(acc, a, b);
        }
        float sum = hsum512_ps(acc);
        for (; j < N; ++j) {
            sum += row[j].to_float() * x[j].to_float();
        }
        y[i] = sum;
    }
}
