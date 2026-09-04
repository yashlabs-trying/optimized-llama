import traceback
try:
    from vllm import LLM
    llm = LLM(model="/workspace/models/llama-3.2-3b-hf", max_model_len=2048,
              enforce_eager=True)
    print("VLLM OK")
except Exception:
    with open("/workspace/vllm_err.txt", "w") as f:
        traceback.print_exc(file=f)
    print("wrote err")
