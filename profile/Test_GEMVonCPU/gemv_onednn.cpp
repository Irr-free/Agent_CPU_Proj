#include "gemv_kernels.h"
#include <oneapi/dnnl/dnnl.hpp>

// oneDNN BF16 GEMV: uses matmul with strided row-major A and contiguous x.
void gemv_bf16_onednn(const bf16* A, const bf16* x, float* y, int M, int N) {
    using namespace dnnl;

    // Cache primitives per (M, N) to avoid recreating descriptors every call.
    static engine eng(engine::kind::cpu, 0);
    static stream s(eng);
    static int cached_M = -1;
    static int cached_N = -1;
    static memory::desc a_md;
    static memory::desc x_md;
    static memory::desc y_md;
    static matmul matmul_prim;

    const bool need_rebuild = (M != cached_M) || (N != cached_N);
    if (need_rebuild) {
        memory::dims a_dims = {M, N};
        memory::dims x_dims = {N, 1};
        memory::dims y_dims = {M, 1};
        memory::dims a_strides = {N, 1};
        memory::dims x_strides = {1, 1};
        memory::dims y_strides = {1, 1};

        a_md = memory::desc(a_dims, memory::data_type::bf16, a_strides);
        x_md = memory::desc(x_dims, memory::data_type::bf16, x_strides);
        y_md = memory::desc(y_dims, memory::data_type::f32, y_strides);

        auto matmul_d  = matmul::desc(a_md, x_md, y_md);
        auto matmul_pd = matmul::primitive_desc(matmul_d, eng);
        matmul_prim = matmul(matmul_pd);
        
        cached_M = M;
        cached_N = N;
    }

    memory a_mem(a_md, eng, const_cast<bf16*>(A));
    memory x_mem(x_md, eng, const_cast<bf16*>(x));
    memory y_mem(y_md, eng, y);

    matmul_prim.execute(s,
                        {{DNNL_ARG_SRC, a_mem},
                         {DNNL_ARG_WEIGHTS, x_mem},
                         {DNNL_ARG_DST, y_mem}});
    s.wait();
}
