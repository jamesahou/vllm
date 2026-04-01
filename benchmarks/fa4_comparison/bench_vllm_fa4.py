import json
import os
import statistics
import time
from random import randint, seed

from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig
from vllm.inputs import TokensPrompt
from vllm.v1.attention.backends.registry import AttentionBackendEnum

MODEL = "Qwen/Qwen3-14B"
NUM_SEQS = 512
NUM_TRIALS = 5
NUM_WARMUP = 2
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024

# ATTN_BACKEND mirrors mini-sglang naming:
#   fa4      -> FlashAttention v4
#   fa3 / fa -> FlashAttention v3
#   fa2      -> FlashAttention v2
#   fi       -> FlashInfer
# vLLM separates flash_attn_version from backend enum, so we translate here.
ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "fa4")
RESULT_FILE = f"vllm_{ATTN_BACKEND.replace(',', '_')}_result.json"


def build_attention_config(backend: str) -> AttentionConfig:
    mapping = {
        "fa4": AttentionConfig(flash_attn_version=4),
        "fa3": AttentionConfig(flash_attn_version=3),
        "fa":  AttentionConfig(flash_attn_version=3),
        "fa2": AttentionConfig(flash_attn_version=2),
        "fi":  AttentionConfig(backend=AttentionBackendEnum.FLASHINFER),
    }
    if backend not in mapping:
        raise ValueError(
            f"Unknown ATTN_BACKEND '{backend}'. "
            f"Valid options: {list(mapping.keys())}"
        )
    return mapping[backend]


def make_inputs(rng_seed: int):
    # Same seed and length distribution as bench_minisgl_fa4.py for fair comparison
    seed(rng_seed)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))]
        for _ in range(NUM_SEQS)
    ]
    max_tokens_list = [randint(100, MAX_OUTPUT_LEN) for _ in range(NUM_SEQS)]
    # Consume the same random numbers as SamplingParams in the minisgl script
    # (temperature is not drawn from the rng there, so just max_tokens matters)
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n)
        for n in max_tokens_list
    ]
    return prompts, sampling_params, max_tokens_list


def main():
    print(f"Loading {MODEL} with backend: {ATTN_BACKEND}")
    llm = LLM(
        model=MODEL,
        attention_config=build_attention_config(ATTN_BACKEND),
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )

    prompts, sampling_params, max_tokens_list = make_inputs(rng_seed=0)
    total_tokens = sum(max_tokens_list)

    print(f"Warming up ({NUM_WARMUP} batches)...")
    for i in range(NUM_WARMUP):
        llm.generate(prompts, sampling_params)
        print(f"  warmup {i + 1}/{NUM_WARMUP} done")

    print(f"Running {NUM_TRIALS} timed trials ({NUM_SEQS} seqs, {total_tokens} total output tokens each)...")
    times = []
    for i in range(NUM_TRIALS):
        t = time.perf_counter()
        llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - t
        times.append(elapsed)
        print(f"  trial {i + 1}/{NUM_TRIALS}: {elapsed:.2f}s  ({total_tokens / elapsed:.1f} tok/s)")

    # Discard min and max, compute stats on remaining
    times_trimmed = sorted(times)[1:-1]
    throughputs = [total_tokens / t for t in times_trimmed]
    mean_tp = statistics.mean(throughputs)
    std_tp = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0

    print(f"\n=== vLLM FA4 Results ===")
    print(f"  Throughput (trimmed mean): {mean_tp:.1f} tok/s")
    print(f"  Throughput std:            {std_tp:.1f} tok/s")
    print(f"  All trial throughputs:     {[f'{x:.1f}' for x in [total_tokens / t for t in times]]}")

    result = {
        "system": "vllm",
        "backend": ATTN_BACKEND,
        "model": MODEL,
        "num_seqs": NUM_SEQS,
        "total_tokens": total_tokens,
        "num_trials": NUM_TRIALS,
        "num_warmup": NUM_WARMUP,
        "throughput_mean": round(mean_tp, 2),
        "throughput_std": round(std_tp, 2),
        "trial_throughputs": [round(total_tokens / t, 2) for t in times],
        "all_times_s": [round(t, 4) for t in times],
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {RESULT_FILE}")


if __name__ == "__main__":
    main()
