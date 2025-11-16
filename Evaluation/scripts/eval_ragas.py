import argparse, csv, json
from adapters.infer import ModelClient
from tqdm import tqdm

# RAGAS
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate as ragas_evaluate
from datasets import Dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV: question,contexts,ground_truth  (contexts separated by |||)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    mc = ModelClient()
    qs, ctxs, gts, preds = [], [], [], []

    rows = list(csv.DictReader(open(args.data)))
    for r in tqdm(rows):
        q = r["question"]
        contexts = [c.strip() for c in r["contexts"].split("|||") if c.strip()]
        gt = r["ground_truth"]
        prompt = "Use ONLY the context to answer. If not answerable, say 'I don't know'.\n"
        prompt += "\n".join(f"- {c}" for c in contexts)
        prompt += f"\n\nQuestion: {q}\nAnswer:"
        pred = mc.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        qs.append(q); ctxs.append(contexts); gts.append(gt); preds.append(pred)

    ds = Dataset.from_dict({"question": qs, "contexts": ctxs, "answer": preds, "ground_truth": gts})
    result = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    out = {k: float(v) for k, v in result.items()}
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(out)

if __name__ == "__main__":
    main()
