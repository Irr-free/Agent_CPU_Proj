# vLLM + ITT + VTune 研究方案：分析 Llama3-8B Decode 阶段 Q*K 算子性能

## 方案概述

| 项目 | 配置 |
|------|------|
| 模型 | meta-llama/Meta-Llama-3-8B-Instruct |
| 推理框架 | vLLM 0.15.1 (CPU 版本) |
| 分析工具 | Intel VTune Profiler + ittapi 1.2.1 |
| 目标算子 | Decode 阶段的 Q*K (Query × Key^T) |
| CPU | Intel Xeon Gold 6242R (支持 AVX-512, AVX-512_VNNI) |

---

## 研究背景

在 Transformer 的 Decode 阶段，每个新 token 都需要与之前所有的 Key 向量计算注意力分数，即 Q*K^T。这个操作是 Decode 阶段的性能瓶颈之一。

vLLM CPU 后端使用 `cpu_attention_with_kv_cache` 算子来实现 PagedAttention，其中包含了 Q*K 计算、Softmax、以及 Attention×Value 计算。

---

## 阶段一：环境准备

### 1.1 验证当前环境

```bash
# 进入 apptainer sandbox
cd /home/bing/profile
apptainer shell --writable cpu-vllm-sandbox

# 验证安装
/opt/venv/bin/python -c "
import vllm
import torch
import transformers
import ittapi
print(f'vLLM: {vllm.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'ittapi: installed')
print(f'AVX-512 available: {torch.cpu._is_avx512_supported()}')
"
```

### 1.2 准备模型访问权限

Llama3-8B 需要 HuggingFace Token：

```bash
# 申请 token: https://huggingface.co/settings/tokens
# 登录 HuggingFace
/opt/venv/bin/python -c "
from huggingface_hub import login
login(token='your_hf_token_here')
"
```

或者使用环境变量：
```bash
export HF_TOKEN=your_token_here
```

---

## 阶段二：ITT 标记注入方案

### 方案 A：Python 层标记（简单，推荐先尝试）

在 Python 层标记 `cpu_attention_with_kv_cache` 调用，可分析整个 attention 算子。

**实施步骤：**

1. **创建 ITT 标记 Patch 脚本** (`patch_cpu_attn.py`)

```python
"""Patch vLLM CPU attention with ITT markers"""
import ittapi
import vllm._custom_ops as ops

# 保存原始函数
_original_cpu_attention = ops.cpu_attention_with_kv_cache

# 创建 ITT Domain
DOMAIN_ATTN = ittapi.Domain("CPU_Attention")
DOMAIN_QK = ittapi.Domain("QK_Computation")

def patched_cpu_attention_with_kv_cache(*args, **kwargs):
    """包装原始函数，添加 ITT 标记"""
    with ittapi.Task(DOMAIN_ATTN, "cpu_attention_with_kv_cache"):
        # 可以在内部进一步细分 Q*K, Softmax, Attention×Value
        with ittapi.Task(DOMAIN_QK, "QK_MatMul"):
            result = _original_cpu_attention(*args, **kwargs)
        return result

# 替换原始函数
ops.cpu_attention_with_kv_cache = patched_cpu_attention_with_kv_cache
print("[INFO] Patched cpu_attention_with_kv_cache with ITT markers")
```

2. **创建主分析脚本** (`profile_llama3_decode.py`)

```python
#!/usr/bin/env python3
"""
Llama3-8B Decode 阶段 Q*K 性能分析
"""
import os
import sys
import time
import torch

# 必须首先设置环境变量
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "40"  # 40GB KV Cache
os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "0-19"

# 导入并应用 patch
import patch_cpu_attn  # noqa

from vllm import LLM, SamplingParams
import ittapi

# ITT Domains
DOMAIN_MAIN = ittapi.Domain("Main")
DOMAIN_DECODE = ittapi.Domain("Decode_Phase")

def run_profiling(
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    num_tokens: int = 64,
    batch_size: int = 1,
):
    """运行 Profiling"""
    
    # 长 prompt 确保充分 prefill
    prompt = """Artificial intelligence (AI) is transforming the way we live and work.
    From autonomous vehicles to medical diagnosis, AI systems are becoming increasingly
    sophisticated and capable. The foundation of modern AI is deep learning.
    """ * 10  # 重复以获得长序列
    
    prompts = [prompt] * batch_size
    
    # 模型加载
    with ittapi.Task(DOMAIN_MAIN, "Model_Load"):
        print("[INFO] Loading model...")
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            device="cpu",
            max_model_len=2048,
            trust_remote_code=True,
        )
    
    # Prefill 阶段（不计入分析）
    print("[INFO] Prefill phase...")
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,  # 只生成 1 个 token 完成 prefill
        ignore_eos=True,
    )
    outputs = llm.generate(prompts, sampling_params)
    
    # Decode 阶段分析
    print("[INFO] Decode phase with ITT profiling...")
    ittapi.resume()  # 开始 VTune 收集
    
    with ittapi.Task(DOMAIN_MAIN, "Decode_Analysis"):
        sampling_params_decode = SamplingParams(
            temperature=0.0,
            max_tokens=num_tokens,
            ignore_eos=True,
        )
        outputs = llm.generate(prompts, sampling_params_decode)
    
    ittapi.pause()  # 停止 VTune 收集
    
    print(f"[INFO] Generated {num_tokens} tokens")
    print(f"[INFO] Output: {outputs[0].outputs[0].text[:100]}...")
    
    return outputs

if __name__ == "__main__":
    run_profiling()
```

### 方案 B：C++ 层标记（精确到 Q*K，需要重新编译）

如果需要精确标记 Q*K 计算（而非整个 attention），需要修改 vLLM C++ 源码。

**文件位置**：`vllm/csrc/cpu/attention.cpp` 或相关文件

**修改步骤：**

1. 找到 vLLM 源码（需要重新安装 editable 模式）
2. 在 Q*K 计算循环中添加 ITT API C++ 调用
3. 重新编译 vLLM

```cpp
// 在 C++ attention 实现中添加
#include "ittnotify.h"

// 在 Q*K 计算前
__itt_task_begin(domain, __itt_null, __itt_null, __itt_string_handle_create("QK_MatMul"));

// Q*K 计算代码...

// 在 Q*K 计算后
__itt_task_end(domain);
```

**注意**：此方法复杂度高，建议先用方案 A 验证可行性。

---

## 阶段三：VTune 数据收集

### 3.1 运行 VTune 分析

```bash
cd /home/bing/profile/Test_GEMVonCPU

# 方式 1: Hotspots 分析（查看 CPU 时间分布）
apptainer exec cpu-vllm-sandbox \
  vtune -collect hotspots \
        -result-dir vtune_hotspots \
        -app-working-dir /home/bing/profile/Test_GEMVonCPU \
        -- /opt/venv/bin/python profile_llama3_decode.py

# 方式 2: Microarchitecture 分析（查看 AVX-512 利用率）
apptainer exec cpu-vllm-sandbox \
  vtune -collect u-arch-exploration \
        -result-dir vtune_uarch \
        -app-working-dir /home/bing/profile/Test_GEMVonCPU \
        -- /opt/venv/bin/python profile_llama3_decode.py

# 方式 3: Memory Access 分析（查看内存带宽）
apptainer exec cpu-vllm-sandbox \
  vtune -collect memory-access \
        -result-dir vtune_memory \
        -app-working-dir /home/bing/profile/Test_GEMVonCPU \
        -- /opt/venv/bin/python profile_llama3_decode.py
```

### 3.2 利用 ITT 标记精确收集

```bash
# 只在标记区域收集数据（减少噪声）
apptainer exec cpu-vllm-sandbox \
  vtune -collect hotspots \
        -knob enable-user-tasks=true \
        -result-dir vtune_itt_marked \
        -app-working-dir /home/bing/profile/Test_GEMVonCPU \
        -- /opt/venv/bin/python profile_llama3_decode.py
```

---

## 阶段四：结果分析

### 4.1 生成报告

```bash
# 生成 Summary 报告
apptainer exec cpu-vllm-sandbox \
  vtune -report summary -result-dir vtune_hotspots -format html \
        -report-output report_hotspots.html

# 生成 Top-down 树报告
apptainer exec cpu-vllm-sandbox \
  vtune -report top-down -result-dir vtune_hotspots -format html \
        -report-output report_topdown.html

# 查看 ITT 标记的任务时间线
apptainer exec cpu-vllm-sandbox \
  vtune -report tasks -result-dir vtune_itt_marked
```

### 4.2 关键指标

| 指标 | 说明 | 预期关注点 |
|------|------|-----------|
| CPU Time | Q*K 算子耗时 | Decode 阶段主要开销 |
| Instructions Retired | 执行指令数 | AVX-512 向量化效率 |
| CPI Rate | 每指令周期数 | 内存延迟 vs 计算 |
| AVX-512 Utilization | AVX-512 指令比例 | 向量化程度 |
| Memory Bandwidth | 内存带宽使用 | 访存瓶颈分析 |
| L1/L2/L3 Miss | 缓存未命中 | 数据局部性 |

---

## 阶段五：对比实验（可选）

### 5.1 不同 Batch Size

```python
for bs in [1, 2, 4, 8]:
    run_profiling(batch_size=bs, num_tokens=64)
```

### 5.2 不同序列长度

```python
for seq_len in [512, 1024, 2048]:
    run_profiling(prompt_len=seq_len, num_tokens=64)
```

### 5.3 对比 PyTorch SDPA

修改 vLLM 配置使用 SDPA 而非自定义 attention，对比性能差异。

---

## 预期成果

1. **Q*K 算子性能特征**：
   - 在 Decode 阶段的时间占比
   - AVX-512 指令利用率
   - 内存访问模式

2. **优化建议**：
   - 是否存在向量化不足
   - 缓存未命中问题
   - 内存带宽瓶颈

3. **性能数据**：
   - Tokens/second
   - Q*K 计算延迟
   - CPU 利用率

---

## 风险与缓解

| 风险 | 可能性 | 缓解措施 |
|------|--------|---------|
| 模型下载失败 | 中 | 提前下载，检查 HF Token |
| 内存不足 | 低 | 376GB 充足，监控内存使用 |
| vLLM 版本兼容 | 中 | 使用已知稳定版本 |
| ITT 标记开销 | 低 | 只在关键路径标记，对比有无 ITT 的结果 |
| VTune 许可 | 低 | 使用免费版本或确保许可 |

---

## 下一步行动

1. [ ] 用户审核方案
2. [ ] 确认 HuggingFace Token
3. [ ] 实施方案 A（Python 层 ITT 标记）
4. [ ] 运行初步 VTune 分析
5. [ ] 根据初步结果决定是否实施方案 B（C++ 层标记）

---

## 附录

### 参考命令速查

```bash
# 进入 apptainer
cd /home/bing/profile && apptainer shell cpu-vllm-sandbox

# 验证 ITT
/opt/venv/bin/python -c "import ittapi; print('OK')"

# 运行分析
/opt/venv/bin/python profile_llama3_decode.py

# VTune GUI（如支持）
vtune-gui vtune_hotspots/
```
