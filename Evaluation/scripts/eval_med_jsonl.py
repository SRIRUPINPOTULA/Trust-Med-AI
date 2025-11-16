import argparse, json, re, sys
from tqdm import tqdm
from adapters.infer import ModelClient

def norm(s): 
    return re.sub(r'\W+', ' ', (s or "").strip().lower())

def em(pred, gold):
    return 1.0 if norm(pred) == norm(gold) else 0.0

def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g: return 1.0 if p==g else 0.0
    inter = sum(min(p.count(w), g.count(w)) for w in set(p))
    if inter==0: return 0.0
    prec, rec = inter/len(p), inter/len(g)
    return 2*prec*rec/(prec+rec)

def build_prompt(instruction, input_text, with_input=True):
    if with_input and input_text and input_text.strip():
        return f"""You are a concise medical assistant.
Instruction: {instruction}

Input: {input_text}

Answer:"""
    else:
        return f"""You are a concise medical assistant.
Instruction: {instruction}

Answer:"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSONL with fields: instruction,input,output")
    ap.add_argument("--out", required=True, help="Path to JSONL with per-example results + summary row")
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--use_input", action="store_true", help="Include the 'input' field in the prompt if present")
    ap.add_argument("--add_rouge", action="store_true", help="Also compute ROUGE-L (extra dep)")
    ap.add_argument("--add_bert", action="store_true", help="Also compute BERTScore (extra dep, slower)")
    args = ap.parse_args()

    mc = ModelClient()
    preds, refs, rows = [], [], []
    n=0
    with open(args.data) as f:
        for line in tqdm(f, desc="Evaluating"):
            line=line.strip()
            if not line: continue
            obj = json.loads(line)
            instr = obj.get("instruction") or obj.get("question") or obj.get("prompt")
            inp = obj.get("input") or obj.get("context") or ""
            gold = obj.get("output") or obj.get("answer") or obj.get("reference") or ""
            if not instr:
                # skip malformed rows
                continue
            prompt = build_prompt(instr, inp, with_input=args.use_input)
            pred = mc.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            preds.append(pred); refs.append(gold)
            rows.append({"instruction": instr, "input": inp, "gold": gold, "pred": pred, "em": em(pred,gold), "f1": f1(pred,gold)})
            n+=1

    EM = sum(x["em"] for x in rows)/n if n else 0.0
    F1 = sum(x["f1"] for x in rows)/n if n else 0.0
    summary = {"SUMMARY": {"n": n, "EM": EM, "F1": F1}}

    # Optional metrics
    if args.add_rouge:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rougeL = sum(scorer.score(refs[i], preds[i])['rougeL'].fmeasure for i in range(len(preds))) / len(preds) if preds else 0.0
        summary["SUMMARY"]["rougeL_f"] = rougeL
    if args.add_bert:
        from bert_score import score as bert_score
        P, R, F1b = bert_score(preds, refs, lang="en", verbose=True)
        summary["SUMMARY"]["bert_precision"] = float(P.mean())
        summary["SUMMARY"]["bert_recall"] = float(R.mean())
        summary["SUMMARY"]["bert_f1"] = float(F1b.mean())

    with open(args.out, "w") as out:
        for r in rows:
            out.write(json.dumps(r)+"\n")
        out.write(json.dumps(summary)+"\n")

    print(summary)

if __name__ == "__main__":
    main()
