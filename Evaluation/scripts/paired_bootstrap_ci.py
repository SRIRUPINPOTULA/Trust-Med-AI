
import argparse, json, random, statistics

def load_scores(path):
    items = []
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "f1" in obj:
                items.append(obj["f1"])
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_jsonl", required=True, help="per-example JSONL with field f1 (e.g., base)")
    ap.add_argument("--b_jsonl", required=True, help="per-example JSONL with field f1 (e.g., lora)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = load_scores(args.a_jsonl)
    B = load_scores(args.b_jsonl)
    if len(A)!=len(B) or not A:
        raise SystemExit("Mismatch or empty scores; ensure both files were run on the same set and include 'f1'.")

    diffs = [b - a for a,b in zip(A,B)]
    mean_diff = statistics.mean(diffs)

    boots = []
    n = len(diffs)
    rnd = random.Random(0)
    for _ in range(args.iters):
        sample = [diffs[rnd.randrange(n)] for _ in range(n)]
        boots.append(statistics.mean(sample))
    boots.sort()
    lo = boots[int(0.025*args.iters)]
    hi = boots[int(0.975*args.iters)]

    # Approx p-value: fraction of bootstrap means <= 0 (for B not better than A), two-sided
    p = min(1.0, 2*min(sum(1 for x in boots if x<=0)/len(boots), sum(1 for x in boots if x>=0)/len(boots)))

    summ = {"n": n, "mean_diff_B_minus_A": mean_diff, "ci95": [lo, hi], "p_two_sided": p}
    with open(args.out,"w") as f:
        f.write(json.dumps({"SUMMARY": summ})+"\n")
    print(summ)

if __name__ == "__main__":
    main()
