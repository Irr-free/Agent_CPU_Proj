# vLLM Llama3-8B Decode Phase Q*K Profiling

本项目基于 ITT (Intel Trace Collector API) 和 VTune Profiler，对 vLLM 在 CPU 上运行的 Llama3-8B 模型的 Decode 阶段进行性能分析，重点分析 Q*K（Query × Key^T）算子的性能。

## 文件结构

```
Test_GEMVonCPU/
├── README_VLLM_PROFILING.md      # 本文件
├── RESEARCH_PLAN.md              # 详细研究方案
├── patch_cpu_attn.py             # ITT 标记补丁
├── profile_llama3_decode.py      # 主分析脚本
├── run_vtune_analysis.sh         # VTune 分析脚本
├── quick_test.py                 # 环境快速测试
└── vllm_itt_profile.py           # 早期版本（保留）
```

## 环境要求

### 硬件
- CPU: Intel Xeon Gold 6242R 或支持 AVX-512 的 CPU
- 内存: 建议 64GB+（Llama3-8B BF16 约需 16GB 模型 + KV Cache）
- 磁盘: 50GB+ 可用空间

### 软件（已在 apptainer 中配置）
- vLLM 0.15.1+ (CPU backend)
- PyTorch 2.10.0+ (CPU version)
- Transformers 4.57.6+
- ittapi 1.2.1
- Intel VTune Profiler

## 快速开始

### 1. 进入 Apptainer

```bash
cd /home/bing/profile
apptainer shell cpu-vllm-sandbox
```

### 2. 运行环境测试

```bash
cd /home/bing/profile/Test_GEMVonCPU
/opt/venv/bin/python quick_test.py
```

### 3. 设置 HuggingFace Token（访问 Llama3 需要）

```bash
export HF_TOKEN=your_huggingface_token_here
```

获取 token: https://huggingface.co/settings/tokens

### 4. 运行基本分析

```bash
# 基本运行（生成 64 个 token）
/opt/venv/bin/python profile_llama3_decode.py

# 自定义参数
/opt/venv/bin/python profile_llama3_decode.py \
    --decode-tokens 128 \
    --batch-size 1 \
    --prefill-tokens 1024 \
    --dtype bfloat16
```

### 5. VTune 分析

```bash
# 在宿主机上运行（退出 apptainer）
exit

# 运行 VTune 分析
cd /home/bing/profile/Test_GEMVonCPU
./run_vtune_analysis.sh hotspots    # Hotspots 分析
./run_vtune_analysis.sh uarch       # 微架构分析
./run_vtune_analysis.sh memory      # 内存访问分析
./run_vtune_analysis.sh itt         # 带 ITT 标记的分析
./run_vtune_analysis.sh all         # 运行所有分析
```

## 工作原理

### ITT 标记

通过 `patch_cpu_attn.py` 对 vLLM 的 CPU attention 算子进行 monkey-patch：

```python
# 原始函数
ops.cpu_attention_with_kv_cache(...)

# Patch 后
with ittapi.Task(DOMAIN_ATTENTION, "cpu_attention_with_kv_cache"):
    ops.cpu_attention_with_kv_cache(...)
```

这样 VTune 可以精确捕获 Decode 阶段 attention 计算的时间。

### 分析流程

1. **Prefill 阶段**：模型处理输入 prompt（不计入分析）
2. **ITT Resume**：开始 VTune 数据收集
3. **Decode 阶段**：生成新 token，Q*K 计算在此阶段执行
4. **ITT Pause**：停止 VTune 数据收集
5. **报告生成**：生成性能分析报告

## 输出解读

### 控制台输出示例

```
======================================================================
vLLM Llama3-8B Decode Phase Profiling
======================================================================
Model: meta-llama/Meta-Llama-3-8B-Instruct
Device: CPU with AVX-512
Data type: bfloat16
Batch size: 1
...

[Phase 3] Decode completed in 45.23s
[Phase 3] Throughput: 1.41 tokens/sec
[Phase 3] Latency: 706.72 ms/token
```

### VTune 报告关键指标

| 指标 | 说明 | 优化目标 |
|------|------|---------|
| CPU Time | 函数执行时间 | 减少热点函数时间 |
| Instructions Retired | 执行指令总数 | 降低指令数 |
| CPI Rate | 每指令周期数 | 接近 1.0（理想） |
| AVX-512 Utilization | AVX-512 指令占比 | 越高越好 |
| Memory Bandwidth | 内存带宽使用 | 避免带宽瓶颈 |
| L3 Miss | L3 缓存未命中 | 降低未命中率 |

## 进阶使用

### 批量测试不同配置

```bash
for bs in 1 2 4; do
    for tokens in 64 128; do
        echo "Testing batch_size=$bs, decode_tokens=$tokens"
        /opt/venv/bin/python profile_llama3_decode.py \
            --batch-size $bs \
            --decode-tokens $tokens \
            --quiet
    done
done
```

### 自定义 ITT 标记

在 `patch_cpu_attn.py` 中添加更多细粒度标记：

```python
# 添加新的 domain
DOMAIN_CUSTOM = ittapi.Domain("Custom")

# 在特定位置标记
with ittapi.Task(DOMAIN_CUSTOM, "MySection"):
    # 你的代码
    pass
```

### 导出原始数据

```bash
# 导出 CSV 格式
vtune -report top-down -result-dir vtune_results -format csv \
      -csv-delimiter comma -report-output report.csv
```

## 故障排除

### 1. ITT 标记不生效

检查 `quick_test.py` 输出，确认 patch 已应用。

### 2. 模型下载失败

```bash
# 手动下载模型
/opt/venv/bin/python -c "
from huggingface_hub import snapshot_download, login
login(token='$HF_TOKEN')
snapshot_download(repo_id='meta-llama/Meta-Llama-3-8B-Instruct')
"
```

### 3. 内存不足

减少 KV Cache 大小：
```bash
export VLLM_CPU_KVCACHE_SPACE=20  # 减少到 20GB
```

### 4. VTune 找不到

在 apptainer 中安装 VTune：
```bash
apptainer exec --writable cpu-vllm-sandbox bash
apt-get update
apt-get install -y intel-oneapi-vtune
```

## 研究目标

本研究旨在回答以下问题：

1. **Q*K 算子在 Decode 阶段的性能特征**：
   - 时间占比
   - AVX-512 指令使用效率
   - 内存访问模式

2. **性能瓶颈识别**：
   - 计算瓶颈 vs 内存瓶颈
   - 缓存命中率
   - 向量化效率

3. **优化建议**：
   - 基于数据的优化方向
   - 与理论 peak performance 的对比

## 参考

- [vLLM Documentation](https://docs.vllm.ai/)
- [Intel VTune Profiler User Guide](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/)
- [ITT API Reference](https://intel.github.io/ittapi/)
- [Llama3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## 作者

Research Plan created for: Test_GEMVonCPU Project
