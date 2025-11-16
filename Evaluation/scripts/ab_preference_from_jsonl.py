
import argparse, json, re
from adapters.infer import ModelClient

JUDGE_PROMPT = r"""
You are a careful medical evaluator. Compare two candidate answers to the same user prompt.
Choose the one that is **more correct, specific, and medically appropriate**.
If they are equivalent, pick the clearer one. If both are unsafe or wrong, pick the **less harmful** one.

Return ONLY JSON with keys:
{"winner": "A" or "B", "reason": "<1-2 sentences>"}
r"""

def build_case(instruction, input_text, a_pred, b_pred):
    parts = [f"User prompt:\n{instruction}"]
    if input_text and input_text.strip():
        parts.append(f"\nInput context:\n{input_text}")
    parts.append("\nCandidate A:\n" + a_pred)
    parts.append("\nCandidate B:\n" + b_pred)
    parts.append("\nNow decide. JSON only:")
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_jsonl", required=True, help="JSONL with fields: instruction,input,pred (e.g., base)")
    ap.add_argument("--b_jsonl", required=True, help="JSONL with fields: instruction,input,pred (e.g., lora)")
    ap.add_argument("--a_label", default="Base")
    ap.add_argument("--b_label", default="LoRA")
    ap.add_argument("--out", required=True, help="Output JSONL with per-case decision + summary at end")
    ap.add_argument("--judge_temp", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    # Load lines into dict keyed by (instruction,input)
    def load_map(path):
        m = {}
        with open(path) as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                k = (obj.get("instruction") or obj.get("q") or obj.get("question"),
                     obj.get("input") or "")
                if k[0] is None: 
                    continue
                m[k] = obj
        return m

    A, B = load_map(args.a_jsonl), load_map(args.b_jsonl)
    keys = [k for k in A.keys() if k in B]
    if not keys:
        raise SystemExit("No overlapping items between the two JSONLs. Make sure both were run on the same dataset.")

    judge = ModelClient()
    wins_A = wins_B = ties = 0
    rows = []

    for k in keys:
        instr, inp = k
        a_pred = A[k].get("pred","")
        b_pred = B[k].get("pred","")
        prompt = JUDGE_PROMPT + "\n\n" + build_case(instr, inp, a_pred, b_pred)
        js = judge.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.judge_temp)
        winner = None; reason = js
        try:
            data = json.loads(js)
            w = (data.get("winner") or "").strip().upper()
            if w in ("A","B"): winner = w
            reason = data.get("reason", js)[:300]
        except Exception:
            pass
        if winner == "A":
            wins_A += 1
        elif winner == "B":
            wins_B += 1
        else:
            ties += 1
        rows.append({"instruction": instr, "input": inp, "A": a_pred, "B": b_pred, "winner": winner, "reason": reason})

    n = wins_A + wins_B  # ignore ties for win-rate
    win_rate_B = (wins_B / n) if n>0 else 0.0

    # Wilson 95% CI for win rate of B
    import math
    z = 1.96
    if n>0:
        phat = win_rate_B
        denom = 1 + z**2/n
        centre = phat + z**2/(2*n)
        margin = z*math.sqrt((phat*(1-phat) + z**2/(4*n))/n)
        lo = (centre - margin)/denom
        hi = (centre + margin)/denom
    else:
        lo = hi = 0.0

    # Two-sided exact binomial test vs 50%
    # Compute p-value = sum_{k >= wins_B} C(n,k)*0.5^n times 2 if wins_B is on one side
    from math import comb
    p_tail = sum(comb(n,k)*(0.5**n) for k in range(wins_B, n+1))
    p_val = min(1.0, 2*min(p_tail, 1-p_tail))

    summary = {
        "labels": {"A": args.a_label, "B": args.b_label},
        "wins_A": wins_A, "wins_B": wins_B, "ties": ties,
        "n_compared": n, "B_win_rate": win_rate_B, "B_win_rate_ci95": [lo, hi],
        "p_value_vs_50pct": p_val
    }

    with open(args.out,"w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
        f.write(json.dumps({"SUMMARY": summary})+"\n")

    print(summary)

if __name__ == "__main__":
    main()
