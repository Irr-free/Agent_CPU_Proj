#!/usr/bin/env python3
"""
vLLM Llama3-8B Decode Phase Profiling with ITT/VTune

This script profiles the Q*K computation in the decode phase of Llama3-8B
running on CPU with AVX-512.

Usage:
    python profile_llama3_decode.py --decode-tokens 64
"""

import os
import sys
import time
import argparse


def get_long_prompt(target_length: int = 512) -> str:
    """Generate a long prompt to ensure sufficient prefill computation"""
    base_text = """Artificial intelligence (AI) is transforming the way we live and work.
From autonomous vehicles to medical diagnosis, AI systems are becoming increasingly
sophisticated and capable. The foundation of modern AI is deep learning, which uses
neural networks with many layers to learn complex patterns in data. Large language
models like GPT and Llama have demonstrated remarkable capabilities in understanding
and generating human-like text. These models use a transformer architecture with
self-attention mechanisms to process sequences of tokens efficiently.

The key innovation in transformers is the attention mechanism, which allows the model
to focus on relevant parts of the input when generating each output token. Specifically,
the query-key-value (QKV) attention computes attention scores by taking the dot product
between query vectors and key vectors. This operation, often denoted as Q*K^T, is
fundamental to the transformer's ability to capture long-range dependencies."""
    
    words_needed = int(target_length / 0.75)
    words_per_block = len(base_text.split())
    repeats = max(1, words_needed // words_per_block)
    return (base_text + " ") * repeats


def main():
    # ============================================================
    # ALL INITIALIZATION MUST BE INSIDE main() for multiprocessing
    # ============================================================
    
    parser = argparse.ArgumentParser(description="Profile Llama3-8B Decode Phase Q*K Performance")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--itt-layer", type=int,
                        default=int(os.environ.get("VLLM_ITT_LAYER_IDX", "0")),
                        help="Target transformer layer index for ITT instrumentation")
    parser.add_argument("--itt-debug-log", type=str,
                        default="/home/bing/profile/itt_layer_hit.log",
                        help="Path to ITT debug log file")
    parser.add_argument("--allow-compile", action="store_true",
                        help="Allow vLLM torch.compile (not recommended for precise ITT gating)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable torch.compile to ensure ITT hooks execute (recommended)")
    parser.add_argument("--strict-eager", action="store_true",
                        help="Actually pass enforce_eager=True to vLLM (may crash with SIGILL)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.decode_tokens < 1:
        parser.error("--decode-tokens must be >= 1")
    
    verbose = not args.quiet
    
    # ============================================================
    # Environment Setup for ITT/VTune Profiling
    # ============================================================
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    os.environ["VLLM_CPU_KVCACHE_SPACE"] = "20"
    os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "0-79"
    os.environ["OMP_NUM_THREADS"] = "80"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    # os.environ.setdefault("KMP_AFFINITY", "granularity=fine,proclist=[0-39],explicit")
    os.environ["HF_HOME"] = "/home/bing/.cache/huggingface"
    os.environ["VLLM_EXEC_MODE_DETECT"] = "1"


    # os.environ.setdefault("OMP_PROC_BIND", "true")
    # os.environ.setdefault("OMP_PLACES", "cores")
    # os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

    
    # ITT/VTune Configuration
    # Specify which layer to instrument (0-31 for Llama3-8B)
    os.environ["VLLM_ITT_LAYER_IDX"] = str(args.itt_layer)
    # Enable ITT debug logging to verify instrumentation
    os.environ["VLLM_ITT_DEBUG_LOG"] = args.itt_debug_log
    try:
        with open(args.itt_debug_log, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass
    # Enable ITT collection control for vtune -start-paused flow.
    os.environ["VLLM_ITT_CONTROL_COLLECTION"] = "1"
    
    # vLLM Execution Configuration
    # Keep execution in one process so ITT resume/pause is in the profiled process.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Use V1 engine (required for CPU backend)
    os.environ["VLLM_USE_V1"] = "1"
    # Allow graph breaks so Python-level ITT hooks in model code can execute.
    os.environ["VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE"] = "0"
    # Default to disabling compile for precise Python-level ITT gating.
    if not args.allow_compile:
        os.environ["VLLM_DISABLE_COMPILE"] = "1"
        os.environ["VLLM_COMPILATION_LEVEL"] = "0"
    
    # Import everything INSIDE main()
    import torch
    torch._dynamo.config.suppress_errors = True
    import ittapi
    from vllm import LLM, SamplingParams
    
    # ITT Domains
    DOMAIN_MAIN = ittapi.domain("Main")
    DOMAIN_PREFILL = ittapi.domain("Prefill")
    DOMAIN_DECODE = ittapi.domain("Decode")
    
    if verbose:
        print("=" * 70)
        print("vLLM Llama3-8B Decode Phase Profiling")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Decode steps target: {args.decode_tokens}")
        print(f"ITT target layer: {args.itt_layer}")
        print(f"allow_compile: {args.allow_compile}")
        print(f"enforce_eager: {args.enforce_eager}")
        print(f"strict_eager: {args.strict_eager}")
        print("=" * 70)

    # Determine eager mode: --enforce-eager alone disables torch.compile safely
    # --strict-eager actually passes enforce_eager=True (may cause SIGILL)
    llm_enforce_eager = args.strict_eager
    if (args.enforce_eager and not args.strict_eager) or (not args.allow_compile):
        # Safe eager mode: disable torch.compile without passing enforce_eager
        # This prevents ITT hooks from being compiled away
        if verbose:
            print("[INFO] Using safe eager mode: torch.compile disabled via environment.")
            print("[INFO] ITT hooks will execute in Python.")
        os.environ["VLLM_DISABLE_COMPILE"] = "1"
    
    # Prepare prompts
    prompt = get_long_prompt(args.prefill_tokens)
    prompts = [prompt] * args.batch_size
    
    try:
        # Phase 1: Model Loading (INSIDE main())
        with ittapi.task(DOMAIN_MAIN, "Phase1_Model_Loading"):
            if verbose:
                print("\n[Phase 1] Loading model...")
            
            load_start = time.time()
            
            # LLM INITIALIZATION INSIDE main()
            llm = LLM(
                model=args.model,
                dtype=args.dtype,
                tensor_parallel_size=1,
                max_model_len=8192,
                trust_remote_code=True,
                download_dir=os.environ.get("HF_HOME"),
                enforce_eager=llm_enforce_eager,
            )
            
            load_time = time.time() - load_start
            if verbose:
                print(f"[Phase 1] Model loaded in {load_time:.2f}s")
        
        # Phase 2: Prefill
        with ittapi.task(DOMAIN_MAIN, "Phase2_Prefill"):
            if verbose:
                print("\n[Phase 2] Running prefill...")
            
            prefill_params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
            outputs = llm.generate(prompts, prefill_params)
            
            if verbose:
                print(f"[Phase 2] Prefill done")
        
        # Phase 3: Decode (profiled)
        if verbose:
            print("\n[Phase 3] Running decode with ITT profiling...")
        
        decode_start = time.time()
        
        with ittapi.task(DOMAIN_MAIN, "Phase3_Decode"):
            decode_params = SamplingParams(
                temperature=0.0,
                # In vLLM, max_tokens=1 can be prefill-only (no true decode step).
                # Add one token so decode-tokens=N means N iterative decode steps.
                max_tokens=args.decode_tokens + 1,
                ignore_eos=True,
            )
            with ittapi.task(DOMAIN_DECODE, "Generate"):
                outputs = llm.generate(prompts, decode_params)
        
        decode_time = time.time() - decode_start
        
        if verbose:
            print(f"\n[Phase 3] Decode: {decode_time:.2f}s")
            print(f"Throughput: {args.decode_tokens * args.batch_size / decode_time:.2f} tokens/sec")
            # print(f"Latency: {decode_time * 1000 / args.decode_tokens:.2f} ms/token")
            print(f"\nGenerated: {outputs[0].outputs[0].text[:100]}...")
            if os.path.exists(args.itt_debug_log):
                with open(args.itt_debug_log, "r", encoding="utf-8") as f:
                    debug_lines = [line.strip() for line in f if line.strip()]
                resume_cnt = sum("ITT collection resumed" in line for line in debug_lines)
                pause_cnt = sum("ITT collection paused" in line for line in debug_lines)
                task_cnt = sum("ITT task executed" in line for line in debug_lines)
                print("\n[ITT] Debug summary")
                print(f"[ITT] log={args.itt_debug_log}")
                print(f"[ITT] target_layer={args.itt_layer} task_hits={task_cnt} resume={resume_cnt} pause={pause_cnt}")
                tail_n = 8
                if debug_lines:
                    print(f"[ITT] tail({min(tail_n, len(debug_lines))})")
                    for line in debug_lines[-tail_n:]:
                        print(f"[ITT] {line}")
                else:
                    print("[ITT] debug log is empty")
            else:
                print(f"\n[ITT] Debug log not found: {args.itt_debug_log}")
        
        # Release vLLM resources to avoid EngineCore crash on exit
        if verbose:
            print("\n[INFO] Releasing vLLM resources...")
        del llm
        import gc
        gc.collect()
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
