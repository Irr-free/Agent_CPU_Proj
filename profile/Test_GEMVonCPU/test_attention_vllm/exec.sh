# 解析命令行参数
COLLECT_TYPE=${1:-hs}
echo "命令行取到类型: $COLLECT_TYPE"

# 验证
VALID_TYPES="hs:hotspots micro:uarch-exploration ma:memory-access t:threading ps:performance-snapshot hp:hpc-performance"

# echo "收集类型: $COLLECT_TYPE"

case "$COLLECT_TYPE" in
    hs)
        COLLECT_TYPE="hotspots"
        KNOBS="-knob sampling-mode=hw -knob sampling-interval=0.1 -knob enable-stack-collection=true"
        ;;
    micro)
        COLLECT_TYPE="uarch-exploration"
        KNOBS="-knob sampling-interval=0.1 -knob collect-memory-bandwidth=true"
        ;;
    ma)
        COLLECT_TYPE="memory-access"
        KNOBS="-knob sampling-interval=0.1"
        ;;
    t)
        COLLECT_TYPE="threading"
        KNOBS="-knob sampling-interval=0.1 -knob sampling-and-waits=hw -knob stack-size=2048"
        ;;
    ps)
        COLLECT_TYPE="performance-snapshot"
        KNOBS="-knob collect-memory-bandwidth=true -knob analyze-openmp=true"
        ;;
    hp)
        COLLECT_TYPE="hpc-performance"
        KNOBS=""
        ;;
    *)
        echo "未知的收集类型: $COLLECT_TYPE"
        echo "有效类型: $VALID_TYPES"
esac

RESULT_DIR="/home/bing/profile/vtune-results/${COLLECT_TYPE}/decode_layer0_l2k_b1"
# mkdir -p "$RESULT_DIR"

vtune \
    -collect "$COLLECT_TYPE" \
    -mrte-mode=native \
    $KNOBS \
    -start-paused \
    -result-dir "$RESULT_DIR" \
    -- python -u /home/bing/profile/Test_GEMVonCPU/test_attention_vllm/profile_llama3_decode.py \
    --enforce-eager \
    --decode-tokens 4 \
    --batch-size 1 \
    --prefill-tokens 2048

# hotspots uarch-exploration memory-access threading performance-snapshot hpc-performance