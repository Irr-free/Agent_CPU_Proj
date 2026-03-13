#pragma once

#include <immintrin.h>
#include "gemv_utils.h"

void gemv_avx512(const bf16* A, const bf16* x, float* y, int M, int N);
void gemv_amx(const bf16* A, const bf16* x, float* y, int M, int N);
void gemv_bf16_onednn(const bf16* A, const bf16* x, float* y, int M, int N);
// Batched GEMV using oneDNN matmul: K[B,M,N] * q[B,N] -> y[B,M]
void gemv_bf16_onednn_batched(const bf16* K, const bf16* q, float* y,
                              int B, int M, int N);

// OpenMP / scalar helpers (row-major data).
void gemv_scalar_rowM_Ax(const bf16* A, const bf16* x, float* y, int M, int N);
void gemv_scalar_colM_xA(const bf16* x, const bf16* A, float* y, int M, int N);
void gemv_omp_rowM_Ax(const bf16* A, const bf16* x, float* y, int M, int N);
void gemv_omp_colM_xA(const bf16* A, const bf16* x, float* y, int M, int N);
