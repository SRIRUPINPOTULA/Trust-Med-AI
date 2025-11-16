import argparse, csv, json
from adapters.infer import ModelClient

JUDGE_PROMPT = """You are a strict evaluator.
Score the assistant's answer on a 1-5 scale for each criterion:
- Helpfulness (does it directly address the user's request?)
- Completeness (does it include all key steps/facts?)
- Clarity (is it easy to follow and unambiguous?)

Return ONLY JSON:
{"helpfulness": <1-5>, "completeness": <1-5>, "clar": <1-5>, "overall": <1-5>, "reason": "<1-2 sentences>"}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV: prompt[,gold]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--judge_temp", type=float, default=0.0)
    args = ap.parse_args()

    gen = ModelClient()
    judge = ModelClient()  # ideally point to a stronger model via env vars

    rows = list(csv.DictReader(open(args.data)))
    with open(args.out, "w") as f:
        for r in rows:
            prompt = r["prompt"]
            answer = gen.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            judge_input = f"{JUDGE_PROMPT}\n\nUser prompt:\n{prompt}\n\nAssistant answer:\n{answer}\n\nJSON:"
            js = judge.generate(judge_input, max_new_tokens=160, temperature=args.judge_temp)
            try:
                data = json.loads(js)
            except Exception:
                data = {"helpfulness": None, "completeness": None, "clar": None, "overall": None, "reason": js[:200]}
            row = {"prompt": prompt, "answer": answer, "scores": data}
            f.write(json.dumps(row)+"\n")
            print(row)

if __name__ == "__main__":
    main()
