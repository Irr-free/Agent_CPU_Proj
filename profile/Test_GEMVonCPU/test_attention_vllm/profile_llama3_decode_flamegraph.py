#!/usr/bin/env python3
"""
vLLM Llama3-8B Decode Phase Profiling for perf/FlameGraph
"""

import os
import sys
import time
import argparse


def get_long_prompt(target_length: int = 512) -> str:
    """Generate a long prompt to ensure sufficient prefill computation."""
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
    parser = argparse.ArgumentParser(
        description="Profile Llama3-8B decode performance for perf/FlameGraph")
    parser.add_argument("--model",
                        type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--dtype",
                        type=str,
                        default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--allow-compile",
                        action="store_true",
                        help="Allow vLLM torch.compile")
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="Disable torch.compile to keep execution predictable")
    parser.add_argument("--strict-eager",
                        action="store_true",
                        help="Actually pass enforce_eager=True to vLLM")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.decode_tokens < 1:
        parser.error("--decode-tokens must be >= 1")

    verbose = not args.quiet

    # Runtime environment
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    os.environ["VLLM_CPU_KVCACHE_SPACE"] = "20"
    os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "0-39"
    os.environ["OMP_NUM_THREADS"] = "40"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["HF_HOME"] = "/home/bing/.cache/huggingface"
    os.environ["VLLM_EXEC_MODE_DETECT"] = "0"

    # vLLM execution behavior
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE"] = "0"

    if not args.allow_compile:
        os.environ["VLLM_DISABLE_COMPILE"] = "1"
        os.environ["VLLM_COMPILATION_LEVEL"] = "0"

    import torch
    torch._dynamo.config.suppress_errors = True
    from vllm import LLM, SamplingParams

    if verbose:
        print("=" * 70)
        print("vLLM Llama3-8B Decode Profiling (perf/FlameGraph)")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Decode steps target: {args.decode_tokens}")
        print(f"allow_compile: {args.allow_compile}")
        print(f"enforce_eager: {args.enforce_eager}")
        print(f"strict_eager: {args.strict_eager}")
        print("=" * 70)

    llm_enforce_eager = args.strict_eager
    if (args.enforce_eager and not args.strict_eager) or (not args.allow_compile):
        if verbose:
            print("[INFO] Using safe eager mode: torch.compile disabled via environment.")
        os.environ["VLLM_DISABLE_COMPILE"] = "1"

    prompt = get_long_prompt(args.prefill_tokens)
    prompts = [prompt] * args.batch_size

    try:
        if verbose:
            print("\n[Phase 1] Loading model...")
        load_start = time.time()
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

        if verbose:
            print("\n[Phase 2] Running prefill...")
        prefill_params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        llm.generate(prompts, prefill_params)
        if verbose:
            print("[Phase 2] Prefill done")

        if verbose:
            print("\n[Phase 3] Running decode...")
        decode_start = time.time()
        decode_params = SamplingParams(
            temperature=0.0,
            # max_tokens=1 can be prefill-only in vLLM; +1 forces decode steps.
            max_tokens=args.decode_tokens + 1,
            ignore_eos=True,
        )
        outputs = llm.generate(prompts, decode_params)
        decode_time = time.time() - decode_start

        if verbose:
            print(f"\n[Phase 3] Decode: {decode_time:.2f}s")
            print(
                f"Throughput: {args.decode_tokens * args.batch_size / decode_time:.2f} tokens/sec"
            )
            print(f"\nGenerated: {outputs[0].outputs[0].text[:100]}...")

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
