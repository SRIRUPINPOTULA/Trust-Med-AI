import argparse, json, collections, itertools
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    keys=collections.Counter()
    samples=[]
    with open(args.data) as f:
        for i, line in enumerate(f):
            line=line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception as e:
                obj={"__parse_error__":str(e)}
            keys.update(obj.keys())
            if len(samples)<args.n: samples.append(obj)
    print("Key counts:", dict(keys))
    print("Samples:")
    for s in samples:
        print("-"*40); print(s)
if __name__ == "__main__":
    main()
