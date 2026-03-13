#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>

// Minimal BF16 helper with round-to-nearest-even conversion.
struct alignas(2) bf16 {
    uint16_t v;
    bf16() : v(0) {}
    explicit bf16(float f) { v = float_to_bf16_bits(f); }
    static uint16_t float_to_bf16_bits(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        // Round to nearest even on the truncation boundary.
        uint32_t lsb = (u >> 16) & 1;
        uint32_t rounding_bias = 0x7FFF + lsb;
        return static_cast<uint16_t>((u + rounding_bias) >> 16);
    }
    static float bf16_to_float(uint16_t bits) {
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }
    float to_float() const { return bf16_to_float(v); }
};

inline float max_abs_diff(const std::vector<float>& a,
                          const std::vector<float>& b) {
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::abs(a[i] - b[i]));
    }
    return diff;
}
