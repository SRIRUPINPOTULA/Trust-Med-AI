import argparse, json
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)  # base
    ap.add_argument("--b", required=True)  # lora
    args = ap.parse_args()
    la=[json.loads(x) for x in open(args.a) if x.strip() and '"SUMMARY"' not in x]
    lb=[json.loads(x) for x in open(args.b) if x.strip() and '"SUMMARY"' not in x]
    wa=wb=ti=0
    for a,b in zip(la,lb):
        fa=a.get("f1",0.0); fb=b.get("f1",0.0)
        if fa>fb: wa+=1
        elif fb>fa: wb+=1
        else: ti+=1
    n=wa+wb
    print({"wins_base":wa,"wins_lora":wb,"ties":ti,"n_compared":n,"lora_win_rate": (wb/n if n else 0.0)})
if __name__=="__main__":
    main()
