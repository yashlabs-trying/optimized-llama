import traceback
try:
    from vllm import LLM
    llm = LLM(model="/workspace/models/llama-3.2-3b-hf", max_model_len=2048,
              enforce_eager=True)
    out = llm.generate(["The capital of France is"],
                       sampling_params=None)
    print("VLLM OK")
except Exception:
    traceback.print_exc()
