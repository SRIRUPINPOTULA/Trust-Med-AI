import argparse, csv, json, re
from tqdm import tqdm
from adapters.infer import ModelClient

def norm(s): return re.sub(r'\W+', ' ', s.strip().lower())
def em(pred, gold): return 1.0 if norm(pred)==norm(gold) else 0.0
def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g: return 1.0 if p==g else 0.0
    inter = sum(min(p.count(w), g.count(w)) for w in set(p))
    if inter==0: return 0.0
    prec, rec = inter/len(p), inter/len(g)
    return 2*prec*rec/(prec+rec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    mc = ModelClient()
    rows = list(csv.DictReader(open(args.data)))
    res = []
    for r in tqdm(rows):
        q, gold = r["question"], r["answer"]
        pred = mc.generate(q, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        res.append({"q": q, "gold": gold, "pred": pred, "em": em(pred,gold), "f1": f1(pred,gold)})

    EM = sum(x["em"] for x in res)/len(res) if res else 0.0
    F1 = sum(x["f1"] for x in res)/len(res) if res else 0.0
    with open(args.out, "w") as f:
        for x in res: f.write(json.dumps(x)+"\n")
        f.write(json.dumps({"SUMMARY": {"EM": EM, "F1": F1, "n": len(res)}})+"\n")
    print({"EM": EM, "F1": F1, "n": len(res)})

if __name__ == "__main__":
    main()
