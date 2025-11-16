import sys
from adapters.infer import ModelClient
mc = ModelClient()
prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello!"
print(mc.generate(prompt, max_new_tokens=128, temperature=0.0))
