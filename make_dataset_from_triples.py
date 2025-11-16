#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Simple, safe phrasing helper
DISCLAIMER = ("Note: This information is for educational purposes only "
              "and is not a substitute for professional medical advice.")

def make_examples(df: pd.DataFrame, max_per_head=10):
    """
    Build instruction-tuning examples from triples.
    Patterns:
      - Symptoms-of-Disease
      - Treatments-for-Disease
      - Side-effects-of-Drug/Treatment
      - What-causes-Disease (if CAUSE)
      - Contrast queries (if multiple relations available)
    """
    # bucket by head and tail for quick grouping
    by_head = defaultdict(list)
    by_tail = defaultdict(list)
    for _, r in df.iterrows():
        by_head[r["head"]].append((r["relation"], r["tail"], r.get("weight", 1)))
        by_tail[r["tail"]].append((r["relation"], r["head"], r.get("weight", 1)))

    samples = []

    # 1) Symptoms-of-Disease / Disease-by-symptoms
    for disease, rels in by_head.items():
        # gather symptom/treatment/side effect lists
        symptoms = sorted({t for (rel, t, _) in rels if rel.upper() in {"HAS_SYMPTOM", "SYMPTOM_OF"} })
        treatments = sorted({t for (rel, t, _) in rels if rel.upper() in {"TREATED_BY", "TREATS"} })
        sidefx = sorted({t for (rel, t, _) in rels if rel.upper() in {"HAS_SIDE_EFFECT", "SIDE_EFFECT_OF"} })
        causes = sorted({t for (rel, t, _) in rels if rel.upper() in {"CAUSED_BY", "CAUSES"} })

        # Symptoms-of
        if symptoms:
            prompt = f"What are common symptoms of {disease}?"
            answer = (f"Commonly linked symptoms of {disease} include: "
                      f"{', '.join(symptoms[:10])}. {DISCLAIMER}")
            samples.append({"instruction": prompt, "input": "", "output": answer})

        # Treatments-for
        if treatments:
            prompt = f"What treatments or medications are associated with {disease}?"
            answer = (f"Treatments/medications connected to {disease} include: "
                      f"{', '.join(treatments[:10])}. {DISCLAIMER}")
            samples.append({"instruction": prompt, "input": "", "output": answer})

        # Side-effects
        if sidefx:
            prompt = f"What side effects are mentioned in relation to {disease}?"
            answer = (f"Reported side effects linked with {disease} include: "
                      f"{', '.join(sidefx[:10])}. {DISCLAIMER}")
            samples.append({"instruction": prompt, "input": "", "output": answer})

        # Causes
        if causes:
            prompt = f"What factors are associated with {disease}?"
            answer = (f"Associated factors/causes for {disease} include: "
                      f"{', '.join(causes[:10])}. {DISCLAIMER}")
            samples.append({"instruction": prompt, "input": "", "output": answer})

    # 2) Reverse: disease candidates given a symptom (from tail buckets)
    for symptom, rels in by_tail.items():
        diseases = sorted({h for (rel, h, _) in rels if rel.upper() in {"HAS_SYMPTOM", "SYMPTOM_OF"} })
        if diseases and symptom.strip():
            prompt = f"What conditions are commonly associated with the symptom: {symptom}?"
            answer = (f"Conditions associated with {symptom} in the knowledge graph include: "
                      f"{', '.join(diseases[:10])}. {DISCLAIMER}")
            samples.append({"instruction": prompt, "input": "", "output": answer})

    # de-dup and cap
    seen = set(); deduped = []
    for s in samples:
        key = (s["instruction"], s["output"])
        if key in seen: continue
        seen.add(key); deduped.append(s)
    return deduped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples_csv", default="kg_out/triples_clean.csv")
    ap.add_argument("--out_dir", default="kg_dataset")
    ap.add_argument("--val_size", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.triples_csv)
    df = df.dropna(subset=["head", "relation", "tail"])
    df["relation"] = df["relation"].astype(str)

    examples = make_examples(df)
    print(f"Generated {len(examples)} instruction samples.")

    X_train, X_val = train_test_split(examples, test_size=args.val_size, shuffle=True, random_state=42)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/train.jsonl", "w", encoding="utf-8") as f:
        for ex in X_train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(f"{args.out_dir}/val.jsonl", "w", encoding="utf-8") as f:
        for ex in X_val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(X_train)} train and {len(X_val)} val to {args.out_dir}/")

import os
if __name__ == "__main__":
    main()


