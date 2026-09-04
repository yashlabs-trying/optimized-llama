import traceback
import asyncio
from vllm import AsyncLLMEngine, SamplingParams, EngineArgs

async def main():
    engine = AsyncLLMEngine.from_engine_args(
        EngineArgs(model="/workspace/models/llama-3.2-3b-hf",
                   quantization=None, dtype="auto", enforce_eager=True)
    )
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    async for out in engine.generate("Hello", sp, request_id="x"):
        pass
    await engine.close()
    print("OK")

try:
    asyncio.run(main())
except Exception:
    with open("/workspace/err.txt", "w") as f:
        traceback.print_exc(file=f)
    print("ERR")
