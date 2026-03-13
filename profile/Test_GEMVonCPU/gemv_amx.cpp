#include "gemv_kernels.h"

// AMX BF16 GEMV (16x32 tiles). Pads tails into scratch tiles.
void gemv_amx(const bf16* A, const bf16* x, float* y, int M, int N) {
#if !defined(__AMX_BF16__)
#error "Compile with -mamx-bf16 to enable AMX BF16."
#endif
    constexpr int TM = 16;  // rows per tile
    constexpr int TK = 32;  // columns per tile

    alignas(64) bf16 Ablk[TM * TK];
    alignas(64) bf16 xblk[TK];
    alignas(64) float Cblk[TM];

    struct alignas(64) tilecfg {
        uint8_t palette_id;
        uint8_t start_row;
        uint8_t reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    };

    tilecfg cfg{};
    cfg.palette_id = 1;  // AMX palette
    cfg.start_row = 0;
    cfg.rows[0] = TM;            // A tile rows
    cfg.colsb[0] = TK * 2;       // bf16 bytes per row for A
    cfg.rows[1] = TK;            // B tile rows
    cfg.colsb[1] = 2;            // bf16 bytes per row for vector chunk
    cfg.rows[2] = TM;            // C tile rows
    cfg.colsb[2] = 4;            // fp32 bytes per row for output
    _tile_loadconfig(&cfg);

    for (int m0 = 0; m0 < M; m0 += TM) {
        const int mb = std::min(TM, M - m0);
        _tile_zero(2);  // reset accumulation tile

        for (int k0 = 0; k0 < N; k0 += TK) {
            const int kb = std::min(TK, N - k0);
            std::fill(std::begin(Ablk), std::end(Ablk), bf16{});
            std::fill(std::begin(xblk), std::end(xblk), bf16{});

            for (int r = 0; r < mb; ++r) {
                const bf16* src = A + static_cast<size_t>(m0 + r) * N + k0;
                std::copy(src, src + kb, Ablk + r * TK);
            }
            std::copy(x + k0, x + k0 + kb, xblk);

            _tile_loadd(0, Ablk, TK * sizeof(bf16));
            _tile_loadd(1, xblk, sizeof(bf16));
            _tile_dpbf16ps(2, 0, 1);
        }

        _tile_stored(2, Cblk, sizeof(float));
        for (int r = 0; r < mb; ++r) {
            y[m0 + r] = Cblk[r];
        }
    }
    _tile_release();
}
