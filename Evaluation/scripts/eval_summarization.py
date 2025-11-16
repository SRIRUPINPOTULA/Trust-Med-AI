import argparse, csv, json
from tqdm import tqdm
from adapters.infer import ModelClient
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV: article,reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    mc = ModelClient()
    rows = list(csv.DictReader(open(args.data)))
    preds, refs = [], []
    for r in tqdm(rows):
        article, ref = r["article"], r["reference"]
        prompt = f"Summarize the following article in 3-5 sentences:\n\n{article}\n\nSummary:"
        pred = mc.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        preds.append(pred); refs.append(ref)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = sum(scorer.score(refs[i], preds[i])['rougeL'].fmeasure for i in range(len(preds))) / len(preds)

    P, R, F1 = bert_score(preds, refs, lang="en", verbose=True)
    out = {"rougeL_f": rougeL, "bert_precision": float(P.mean()), "bert_recall": float(R.mean()), "bert_f1": float(F1.mean()), "n": len(preds)}
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(out)

if __name__ == "__main__":
    main()
