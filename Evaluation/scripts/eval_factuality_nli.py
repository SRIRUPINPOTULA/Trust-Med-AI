import argparse, csv, json
from adapters.infer import ModelClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

LABELS = {0: "contradiction", 1: "neutral", 2: "entailment"}  # roberta-large-mnli

def nli_score(premise: str, hypothesis: str, tok, model, device):
    inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0].softmax(-1).cpu().tolist()
    return {LABELS[i]: float(logits[i]) for i in range(3)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV: claim,evidence")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--nli_model", default="roberta-large-mnli")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device).eval()

    mc = ModelClient()
    rows = list(csv.DictReader(open(args.data)))
    evals = []

    for r in tqdm(rows):
        claim = r["claim"]
        evidence = r["evidence"]
        prompt = f"Answer the claim with a concise factual statement (no explanations):\nClaim: {claim}\nAnswer:"
        pred = mc.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        scores = nli_score(premise=evidence, hypothesis=pred, tok=tok, model=model, device=device)
        evals.append({"claim": claim, "evidence": evidence, "model_answer": pred, "nli": scores})

    entail = sum(e["nli"]["entailment"] for e in evals) / len(evals) if evals else 0.0
    contra = sum(e["nli"]["contradiction"] for e in evals) / len(evals) if evals else 0.0
    out = {"summary": {"mean_entailment": entail, "mean_contradiction": contra}, "rows": evals[:50]}
    open(args.out,"w").write(json.dumps(out, indent=2))
    print(out)

if __name__ == "__main__":
    main()
