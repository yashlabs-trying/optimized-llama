import json
d = json.load(open("/workspace/fp16.json"))
for k, v in d.items():
    print(f"{k}: {v}")
