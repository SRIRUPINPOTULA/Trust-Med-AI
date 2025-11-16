import argparse, csv, json, os, tempfile, subprocess, random
from math import comb
from adapters.infer import ModelClient

TEMPLATE = """
# Implement the required function(s) below.
{starter}

if __name__ == "__main__":
    pass
"""

TEST_RUNNER = """
import importlib.util, sys, json
def run_tests(sol_path, tests_code):
    spec = importlib.util.spec_from_file_location("solution", sol_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = {"solution": mod, "ALL_PASS": False, "DETAILS": []}
    exec(tests_code, env, env)
    return env.get("ALL_PASS", False), env.get("DETAILS", [])
if __name__ == "__main__":
    sol_path = sys.argv[1]
    tests_code = open(sys.argv[2]).read()
    ok, details = run_tests(sol_path, tests_code)
    print(json.dumps({"ok": bool(ok), "details": details}))
"""

def pass_at_k(n, c, k):
    if k > n: k = n
    if c == 0: return 0.0
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV: problem_id,prompt,starter,tests")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    mc = ModelClient()
    rows = list(csv.DictReader(open(args.data)))
    random.seed(0)

    os.makedirs("tmp_passk", exist_ok=True)
    open("tmp_passk/test_runner.py","w").write(TEST_RUNNER)

    results = []
    values = []
    for r in rows:
        pid, prompt, starter, tests = r["problem_id"], r["prompt"], r["starter"], r["tests"]
        correct = 0
        for i in range(args.samples):
            seed = 1000 + i
            full_prompt = f"""You are a senior engineer. Write Python code only.
{prompt}

Starter:
{starter}

Provide only the code without explanations.
"""
            code = mc.generate(full_prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=seed)
            with open(f"tmp_passk/{pid}_sol_{i}.py","w") as f: f.write(TEMPLATE.format(starter=code))
            with open(f"tmp_passk/{pid}_tests.py","w") as f: f.write(tests)
            p = subprocess.run(["python","tmp_passk/test_runner.py", f"tmp_passk/{pid}_sol_{i}.py", f"tmp_passk/{pid}_tests.py"],
                               capture_output=True, text=True, timeout=20)
            ok = False
            try:
                data = json.loads(p.stdout.strip())
                ok = bool(data.get("ok", False))
            except Exception:
                ok = False
            if ok: correct += 1
        p_at_k = pass_at_k(args.samples, correct, args.k)
        values.append(p_at_k)
        results.append({"problem_id": pid, "n": args.samples, "c": correct, "k": args.k, "pass_at_k": p_at_k})

    out = {"summary": {"problems": len(rows), "mean_pass_at_k": (sum(values)/len(values) if values else 0.0)}, "problems": results}
    open(args.out,"w").write(json.dumps(out, indent=2))
    print(out)

if __name__ == "__main__":
    main()
