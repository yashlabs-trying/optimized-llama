#!/usr/bin/env python3
"""vLLM throughput benchmark (offline batched inference).

Sweeps concurrency to find max throughput. Reports:
  - total throughput (tok/s)
  - requests/s
  - mean TTFT and ITL per request (via async engine under load)

Usage:
  python benchmarks/bench_vllm.py --model /workspace/models/llama-3.2-3b-hf \
      --quantization none --num-prompts 256 --max-tokens 128
"""
import argparse
import asyncio
import json
import random
import time

PROMT_POOL = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main causes of climate change?",
    "Describe how photosynthesis works.",
    "What is the difference between RAM and ROM?",
    "List five healthy breakfast ideas.",
    "How does a blockchain work?",
    "What is machine learning?",
    "Explain the water cycle.",
]


async def run_async(model: str, quantization: str, num_prompts: int,
                    max_tokens: int, concurrency: int):
    from vllm import AsyncLLMEngine, SamplingParams, EngineArgs

    engine = AsyncLLMEngine.from_engine_args(
        EngineArgs(model=model, quantization=quantization, dtype="auto",
                   enforce_eager=True)
    )
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                        ignore_eos=False)

    prompts = [random.choice(PROMT_POOL) for _ in range(num_prompts)]

    ttfts = []
    itls = []
    total_tokens = 0

    async def worker(pidx):
        nonlocal total_tokens
        t0 = time.perf_counter()
        first = True
        last = None
        tok_count = 0
        async for out in engine.generate(prompts[pidx], sp, request_id=str(pidx)):
            if first:
                ttfts.append(time.perf_counter() - t0)
                first = False
            else:
                itls.append(time.perf_counter() - last)
            last = time.perf_counter()
            tok_count = len(out.outputs[0].token_ids)
        total_tokens += tok_count

    t0 = time.perf_counter()
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for i in range(num_prompts):
        async def wrapped(i=i):
            async with sem:
                await worker(i)
        tasks.append(asyncio.create_task(wrapped()))
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    await engine.close()

    return {
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "total_tokens": total_tokens,
        "wall_s": wall,
        "throughput_tok_s": total_tokens / wall if wall else 0,
        "requests_per_s": num_prompts / wall if wall else 0,
        "mean_ttft_s": sum(ttfts) / len(ttfts) if ttfts else 0,
        "mean_itl_s": sum(itls) / len(itls) if itls else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default="none")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrencies", default="1,4,8,16,32,64,128")
    args = ap.parse_args()

    results = []
    for c in [int(x) for x in args.concurrencies.split(",")]:
        r = asyncio.run(run_async(
            args.model, args.quantization, args.num_prompts,
            args.max_tokens, c))
        results.append(r)
        print(json.dumps(r))

    best = max(results, key=lambda r: r["throughput_tok_s"])
    print("\n=== BEST ===")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
