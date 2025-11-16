import os, json
MODE = os.environ.get("EVAL_MODE", "hf_local")

# Local HF defaults (you can override via env)
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
LORA_ADAPTER_PATH = os.environ.get("LORA_ADAPTER_PATH")  # e.g., /path/to/kg_lora_output_chat
HF_DEVICE = os.environ.get("HF_DEVICE", "cpu")           # "mps" on Apple Silicon, else "cpu"

# HTTP (if your model is served behind an endpoint)
HTTP_ENDPOINT = os.environ.get("HTTP_ENDPOINT", "http://localhost:8000/generate")
HTTP_HEADERS = {"Content-Type": "application/json"}

class ModelClient:
    def __init__(self):
        if MODE == "hf_local":
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            self.torch = torch
            # Qwen-style models may need trust_remote_code
            self.tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, trust_remote_code=True).to(HF_DEVICE)
            if LORA_ADAPTER_PATH:
                try:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, LORA_ADAPTER_PATH)
                except Exception as e:
                    raise RuntimeError(f"Failed to load LoRA adapter at {LORA_ADAPTER_PATH}: {e}")
            self.model.eval()
        elif MODE == "http":
            import requests
            self.requests = requests
        else:
            raise ValueError(f"Unknown MODE: {MODE}")

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0, seed: int = 42) -> str:
        if MODE == "hf_local":
            if hasattr(self.torch, "manual_seed"): self.torch.manual_seed(seed)
            inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    do_sample=(temperature>0.0),
                    temperature=(temperature or 1.0),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tok.eos_token_id
                )
            text = self.tok.decode(out[0], skip_special_tokens=True)
            return text[len(prompt):].strip() if text.startswith(prompt) else text
        else:
            import json
            payload = {"prompt": prompt, "max_new_tokens": max_new_tokens, "temperature": temperature, "seed": seed}
            r = self.requests.post(HTTP_ENDPOINT, headers=HTTP_HEADERS, data=json.dumps(payload), timeout=120)
            r.raise_for_status()
            return r.json().get("text","").strip()

def maybe_token_count(text: str) -> int:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=True)
        return len(tok.encode(text))
    except Exception:
        return len(text.split())
