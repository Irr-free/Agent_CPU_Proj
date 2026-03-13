#include "gemv_kernels.h"
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

// Batched BF16 GEMV implemented with oneDNN batched matmul:
// K shape: [B, M, N] (row-major contiguous), q: [B, N], y: [B, M].
void gemv_bf16_onednn_batched(const bf16* K, const bf16* q, float* y,
                              int B, int M, int N) {
    namespace dnnl = oneapi::dnnl;

    static dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    static dnnl::stream strm(eng);

    // Cache primitive by (B, M, N) to avoid rebuilding on every call.
    struct Cached {
        int B = -1, M = -1, N = -1;
        dnnl::memory::desc src_md, wei_md, dst_md, sp_md;
        dnnl::matmul prim;
        std::vector<uint8_t> scratchpad;
    };
    static Cached cache;

    if (cache.B != B || cache.M != M || cache.N != N) {
        dnnl::memory::dims src_dims = {B, M, N}; // K
        dnnl::memory::dims wei_dims = {B, N, 1}; // q (as weights)
        dnnl::memory::dims dst_dims = {B, M, 1}; // y

        cache.src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::bf16,
                                          dnnl::memory::format_tag::abc);
        cache.wei_md = dnnl::memory::desc(wei_dims, dnnl::memory::data_type::bf16,
                                          dnnl::memory::format_tag::abc);
        cache.dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::abc);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#if DNNL_VERSION_MAJOR < 3
        auto matmul_d = dnnl::matmul::desc(cache.src_md, cache.wei_md, cache.dst_md);
        dnnl::matmul::primitive_desc matmul_pd(matmul_d, attr, eng);
#else
        dnnl::matmul::primitive_desc matmul_pd(
            eng, cache.src_md, cache.wei_md, cache.dst_md, attr);
#endif
        cache.prim = dnnl::matmul(matmul_pd);
        cache.sp_md = matmul_pd.scratchpad_desc();
        cache.scratchpad.resize(cache.sp_md.get_size());

        cache.B = B; cache.M = M; cache.N = N;
    }

    dnnl::memory src_mem(cache.src_md, eng, const_cast<bf16*>(K));
    dnnl::memory wei_mem(cache.wei_md, eng, const_cast<bf16*>(q));
    dnnl::memory dst_mem(cache.dst_md, eng, y);
    dnnl::memory sp_mem(cache.sp_md, eng, cache.scratchpad.data());

    cache.prim.execute(strm, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, wei_mem},
        {DNNL_ARG_DST, dst_mem},
        {DNNL_ARG_SCRATCHPAD, sp_mem},
    });
    strm.wait();
}
