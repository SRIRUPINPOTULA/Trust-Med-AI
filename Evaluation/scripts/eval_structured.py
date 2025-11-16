import argparse, csv, json
from adapters.infer import ModelClient
from tqdm import tqdm
from jsonschema import Draft202012Validator

SYSTEM_INSTRUCTIONS = "Return ONLY a JSON object that matches the provided schema. Do not include explanations."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with column: prompt")
    ap.add_argument("--schema", required=True, help="Path to JSONSchema file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    schema = json.load(open(args.schema))
    validator = Draft202012Validator(schema)
    mc = ModelClient()

    rows = list(csv.DictReader(open(args.data)))
    ok_parse, ok_schema = 0, 0
    details = []

    for r in tqdm(rows):
        p = r["prompt"]
        prompt = f"""{SYSTEM_INSTRUCTIONS}
JSONSchema:
{json.dumps(schema)}
---
Prompt:
{p}
JSON:
"""
        text = mc.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        parsed, parse_err, schema_errs = None, None, None
        try:
            parsed = json.loads(text)
            ok_parse += 1
            errs = sorted(validator.iter_errors(parsed), key=lambda e: e.path)
            if not errs:
                ok_schema += 1
            else:
                schema_errs = [f"{'/'.join(map(str,e.path))}: {e.message}" for e in errs]
        except Exception as e:
            parse_err = str(e)

        details.append({"prompt": p, "raw": text, "parsed": parsed is not None, "schema_valid": schema_errs is None, "schema_errors": schema_errs, "parse_error": parse_err})

    report = {
        "total": len(rows),
        "parseable_json_rate": ok_parse/len(rows) if rows else 0.0,
        "schema_valid_rate": ok_schema/len(rows) if rows else 0.0,
        "examples": details[:20]
    }
    json.dump(report, open(args.out, "w"), indent=2)
    print(report)

if __name__ == "__main__":
    main()
