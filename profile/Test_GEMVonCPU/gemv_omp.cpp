#include "gemv_kernels.h"

// Scalar BF16 GEMV: y = A * x (A: MxN, row-major), accumulate in FP32.
void gemv_scalar_rowM_Ax(const bf16* A, const bf16* x, float* y, int M, int N) {
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        const bf16* row = A + static_cast<size_t>(i) * N;
        for (int j = 0; j < N; ++j) {
            sum += row[j].to_float() * x[j].to_float();
        }
        y[i] = sum;
    }
}

// Scalar BF16 GEMV: y = x * A (A: MxN row-major), accumulate in FP32.
void gemv_scalar_colM_xA(const bf16* A, const bf16* x, float* y, int M, int N) {
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += x[i].to_float() * A[static_cast<size_t>(i) * N + j].to_float();
        }
        y[j] = sum;
    }
}

// Scalar BF16 GEMV with OpenMP with A row-major: y = A * x.
void gemv_omp_rowM_Ax(const bf16* A, const bf16* x, float* y, int M, int N) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        const bf16* row = A + static_cast<size_t>(i) * N;
        for (int j = 0; j < N; ++j) {
            sum += row[j].to_float() * x[j].to_float();
        }
        y[i] = sum;
    }
}

// Scalar BF16 GEMV with OpenMP: y = x * A (row-major).
void gemv_omp_colM_xA(const bf16* A, const bf16* x, float* y, int M, int N) {
    std::fill(y, y + N, 0.0f);
#pragma omp parallel
    {
        std::vector<float> local_y(N, 0.0f);
#pragma omp for schedule(static)
        for (int i = 0; i < M; ++i) {
            float x_val = x[i].to_float();
            if (x_val == 0.0f) continue;
            const bf16* row_ptr = A + static_cast<size_t>(i) * N;
            for (int j = 0; j < N; ++j) {
                local_y[j] += row_ptr[j].to_float() * x_val;
            }
        }
#pragma omp critical
        {
            for (int j = 0; j < N; ++j) {
                y[j] += local_y[j];
            }
        }
    }
}
