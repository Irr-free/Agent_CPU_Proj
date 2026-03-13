#!/usr/bin/env python3
"""
Basic vLLM test - Run Llama3-8B on CPU without ITT/VTune
vLLM 0.10.0 version
"""

import os
import sys
import time

# Environment Setup for vLLM 0.10.0
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "10"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["HF_HOME"] = "/home/bing/.cache/huggingface"


def get_prompt():
    return "Artificial intelligence is transforming the way we live and work. " * 50


def main():
    print("=" * 70)
    print("Basic vLLM Test - Llama3-8B on CPU (v0.10.0)")
    print("=" * 70)
    
    # Import inside main
    from vllm import LLM, SamplingParams
    
    prompt = get_prompt()
    
    try:
        print("\n[1/3] Loading model...")
        start = time.time()
        
        # vLLM 0.10.0 API - remove unsupported parameters
        llm = LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=2048,
            trust_remote_code=True,
        )
        
        print(f"Model loaded in {time.time() - start:.2f}s")
        
        print("\n[2/3] Running prefill (1 token)...")
        params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        outputs = llm.generate([prompt], params)
        print(f"Prefill done. First token: {outputs[0].outputs[0].text!r}")
        
        print("\n[3/3] Running decode (4 tokens)...")
        start = time.time()
        params = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
        outputs = llm.generate([prompt], params)
        elapsed = time.time() - start
        print(f"Decode done in {elapsed:.2f}s")
        print(f"Generated: {outputs[0].outputs[0].text}")
        
        print("\n[4/4] Releasing resources...")
        del llm
        import gc
        gc.collect()
        print("Resources released.")
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
