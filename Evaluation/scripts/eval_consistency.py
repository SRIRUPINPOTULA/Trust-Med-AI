import argparse, json, random, itertools
from adapters.infer import ModelClient

def normalize(s):
    import re
    return re.sub(r"\W+"," ", s.strip().lower())

def jaccard(a, b):
    A, B = set(normalize(a).split()), set(normalize(b).split())
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B)/len(A | B)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--t_low", type=float, default=0.0)
    ap.add_argument("--t_high", type=float, default=0.7)
    args = ap.parse_args()

    mc = ModelClient()
    prompts = [p.strip() for p in open(args.prompts) if p.strip()]
    random.seed(0)

    report = {"prompts": [], "summary": {}}
    det_flags, var_scores = [], []

    for p in prompts:
        lows = [mc.generate(p, max_new_tokens=args.max_new_tokens, temperature=args.t_low, seed=100+i) for i in range(args.samples)]
        det = int(len(set(lows)) == 1)
        det_flags.append(det)

        highs = [mc.generate(p, max_new_tokens=args.max_new_tokens, temperature=args.t_high, seed=200+i) for i in range(args.samples)]
        sims = [jaccard(a,b) for a,b in itertools.combinations(highs, 2)]
        avg_sim = sum(sims)/len(sims) if sims else 1.0
        variability = 1 - avg_sim
        var_scores.append(variability)

        report["prompts"].append({"prompt": p, "deterministic_at_t_low": bool(det), "avg_pairwise_similarity_t_high": avg_sim, "variability_t_high": variability})

    report["summary"] = {"determinism_rate_t_low": sum(det_flags)/len(det_flags) if det_flags else 0.0,
                         "avg_variability_t_high": sum(var_scores)/len(var_scores) if var_scores else 0.0}
    open(args.out,"w").write(json.dumps(report, indent=2))
    print(report)

if __name__ == "__main__":
    main()
