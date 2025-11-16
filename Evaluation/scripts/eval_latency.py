import argparse, time, json, statistics
from adapters.infer import ModelClient, maybe_token_count

def pct(vs, q):
    vs = sorted(vs)
    idx = int((q/100.0) * (len(vs)-1))
    return vs[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="Text file, one prompt per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    mc = ModelClient()
    prompts = [p.strip() for p in open(args.prompts) if p.strip()]
    latencies, token_counts = [], []

    for p in prompts:
        t0 = time.perf_counter()
        text = mc.generate(p, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        t1 = time.perf_counter()
        latencies.append(t1-t0)
        token_counts.append(maybe_token_count(text))

    report = {
        "n": len(prompts),
        "p50_latency_s": pct(latencies, 50) if latencies else None,
        "p95_latency_s": pct(latencies, 95) if latencies else None,
        "avg_tokens": sum(token_counts)/len(token_counts) if token_counts else 0.0,
        "tokens_per_second": (sum(token_counts)/sum(latencies)) if latencies and sum(latencies)>0 else None
    }
    with open(args.out, "w") as f: json.dump(report, f, indent=2)
    print(report)

if __name__ == "__main__":
    main()
