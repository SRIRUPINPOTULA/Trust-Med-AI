import argparse, csv, json, pandas as pd
from adapters.infer import ModelClient

def normalize(s):
    import re
    return re.sub(r"\W+"," ", s.strip().lower())

def em(pred, truth): 
    return 1.0 if normalize(pred)==normalize(truth) else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV: prompt,topic,length_bucket,difficulty[,gold]")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    mc = ModelClient()
    df = pd.read_csv(args.data)
    df["pred"] = [mc.generate(p, max_new_tokens=args.max_new_tokens, temperature=args.temperature) for p in df["prompt"].tolist()]

    if "gold" in df.columns:
        df["em"] = [em(p,g) for p,g in zip(df["pred"], df["gold"])]
        metric = "em"
    else:
        df["resp_len"] = df["pred"].str.len()
        metric = "resp_len"

    piv = df.pivot_table(values=metric, index=["topic","length_bucket","difficulty"], aggfunc="mean")
    df.to_csv(args.out_csv, index=False)
    open(args.out_json,"w").write(json.dumps({"by_bucket_mean": piv.to_dict()}, indent=2))
    print({"by_bucket_mean": piv.to_dict()})

if __name__ == "__main__":
    main()
