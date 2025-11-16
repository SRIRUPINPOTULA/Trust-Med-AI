import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base HF model id or path")
    ap.add_argument("--lora", required=True, help="LoRA adapter path")
    ap.add_argument("--out", required=True, help="Merged model output dir")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.lora)
    merged = model.merge_and_unload()
    tok.save_pretrained(args.out); merged.save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")

if __name__ == "__main__":
    main()
