在CPU上研究针对GEMV的优化加速
前期测试部分：
测试对象：
1. 单纯部署GEMV在CPU上 [BF16]
2. 部署GEMV在CPU的AVX-512上 [BF16]
3. 部署GEMV在CPU的AMX上 [BF16]
4. 部署GEMV通过 oneDNN matmul [BF16 输入累加到 FP32]

## 代码结构
- `gemv_utils.h`：BF16 类型与通用工具。
- `gemv_kernels.h`：各实现的声明。
- `gemv_scalar.cpp`：纯标量 BF16 GEMV。
- `gemv_avx512.cpp`：AVX-512 `_mm512_dpbf16_ps` GEMV。
- `gemv_amx.cpp`：AMX `_tile_dpbf16ps` GEMV。
- `gemv_onednn.cpp`：oneDNN `matmul` 路径（BF16 输入累加到 FP32）。
- `GEMV2CPU.cpp`：测试驱动，命令行可选择模式/shape。
- shape 可通过命令行传参灵活配置：`--m`（行数）、`--n`（列数）、`--iters`（迭代次数）、`--mode scalar|avx512|amx|onednn|all`。

## 编译
- 建议统一编译一个二进制，包含所有路径（CPU 不支持的路径会在运行时跳过）：
  ```
  g++ -O3 -std=c++17 -mavx512bf16 -mamx-bf16 \
      Test_GEMVonCPU/GEMV2CPU.cpp \
      Test_GEMVonCPU/gemv_scalar.cpp \
      Test_GEMVonCPU/gemv_avx512.cpp \
      Test_GEMVonCPU/gemv_amx.cpp \
      Test_GEMVonCPU/gemv_onednn.cpp \
      -ldnnl -o gemv_bf16
  ```

## 运行示例
- 运行全部路径并用标量结果校验：`./gemv_bf16 --m 4096 --n 4096 --iters 100 --mode all`
- 只测 AVX-512 路径：`./gemv_bf16 --m 2048 --n 4096 --iters 200 --mode avx512`
- 只测 AMX 路径：`./gemv_bf16 --m 2048 --n 4096 --iters 200 --mode amx`
- 只测 oneDNN 路径：`./gemv_bf16 --m 2048 --n 4096 --iters 200 --mode onednn`
