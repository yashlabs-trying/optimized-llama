import re
data = open("/workspace/vllm_fp16.log", "rb").read().decode("utf-8", "replace")
for line in data.splitlines():
    if "Error" in line or "error" in line or "raise" in line or "self._" in line and "assert" in line:
        print(line)
