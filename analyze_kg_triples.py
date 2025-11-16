#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze and understand a medical KG triples JSON:
- Loads JSON (array or JSONL)
- Normalizes into (head, relation, tail) + optional types
- Cleans and deduplicates
- Exports summary CSVs and a relation histogram
- (Optional) Generates Neo4j import CSVs + Cypher

Usage:
  python analyze_kg_triples.py --input medical_kg_triples.json --outdir ./kg_out --neo4j
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# --------------------------
# IO & normalization helpers
# --------------------------

def load_any_json(path: str) -> List[dict]:
    """Load JSON (array/object) or JSON Lines. Returns a list of dict triples."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Try standard JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "triples" in data and isinstance(data["triples"], list):
            return data["triples"]
        if isinstance(data, list):
            return data
        # Single dict → single triple
        return [data]
    except json.JSONDecodeError:
        pass

    # Fallback to JSON Lines
    triples: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                triples.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip bad lines
                continue
    return triples


HEAD_KEYS = ["head", "subject", "subj", "s"]
REL_KEYS  = ["relation", "predicate", "rel", "p", "edge", "type"]
TAIL_KEYS = ["tail", "object", "obj", "o"]

def find_key(d: dict, candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None

def norm_node(x: Any) -> Tuple[str, Optional[str]]:
    """
    Normalize a node field to a display string + optional type.
    Accepts strings or dicts like:
      {"name": "...", "type": "Disease"} or {"id": "C0011849", "label": "..."}
    """
    if isinstance(x, dict):
        # Prefer human-friendly label
        for key in ("name", "label", "text", "value"):
            if key in x and isinstance(x[key], str) and x[key].strip():
                return x[key].strip(), x.get("type") or x.get("category")
        # Fallback to IDs
        for key in ("id", "cui", "uid", "iri"):
            if key in x and isinstance(x[key], str) and x[key].strip():
                return x[key].strip(), x.get("type") or x.get("category")
        return json.dumps(x, ensure_ascii=False), x.get("type") or x.get("category")
    elif x is None:
        return "", None
    else:
        return str(x).strip(), None

def norm_rel(x: Any) -> str:
    if isinstance(x, dict):
        for key in ("name", "label", "type", "relation", "p", "id"):
            v = x.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return json.dumps(x, ensure_ascii=False)
    elif x is None:
        return ""
    else:
        return str(x).strip()

def normalize_triples(raw: List[dict]) -> pd.DataFrame:
    rows = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        hk = find_key(t, HEAD_KEYS)
        rk = find_key(t, REL_KEYS)
        tk = find_key(t, TAIL_KEYS)
        head_val = t.get(hk) if hk else None
        rel_val  = t.get(rk) if rk else None
        tail_val = t.get(tk) if tk else None

        head, head_type = norm_node(head_val)
        tail, tail_type = norm_node(tail_val)
        rel = norm_rel(rel_val)

        row = {
            "head": head, "relation": rel, "tail": tail,
            "head_type": head_type, "tail_type": tail_type
        }

        # Keep provenance / extras
        for k, v in t.items():
            if k not in (hk, rk, tk):
                row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    # Basic clean
    for c in ("head", "relation", "tail"):
        if c in df:
            df[c] = df[c].astype(str).str.strip()
    # Drop empties
    df = df.dropna(subset=["head", "relation", "tail"])
    df = df[(df["head"] != "") & (df["relation"] != "") & (df["tail"] != "")]
    # Deduplicate
    df = df.drop_duplicates(subset=["head", "relation", "tail"]).reset_index(drop=True)
    return df


# --------------------------
# Analytics
# --------------------------

def compute_relation_counts(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["relation"].value_counts().reset_index()
    vc.columns = ["relation", "count"]
    return vc

def compute_node_degrees(df: pd.DataFrame) -> pd.DataFrame:
    G = nx.MultiDiGraph()
    for _, r in df.iterrows():
        G.add_node(r["head"], ntype=r.get("head_type"))
        G.add_node(r["tail"], ntype=r.get("tail_type"))
        G.add_edge(r["head"], r["tail"], relation=r["relation"])
    records = []
    for n in G.nodes():
        records.append({
            "node": n,
            "in_degree": G.in_degree(n),
            "out_degree": G.out_degree(n),
            "total_degree": G.degree(n),
            "type": G.nodes[n].get("ntype")
        })
    deg = pd.DataFrame(records).sort_values("total_degree", ascending=False).reset_index(drop=True)
    return deg

def plot_relation_hist(rel_counts: pd.DataFrame, out_png: str, top_k: int = 20):
    if rel_counts.empty:
        return
    rel_counts = rel_counts.head(top_k)
    plt.figure()
    plt.bar(rel_counts["relation"], rel_counts["count"])
    plt.title("Top Relations by Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Relation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# --------------------------
# Neo4j export
# --------------------------

def slugify(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

def make_neo4j_files(df: pd.DataFrame, outdir: str):
    """
    Creates:
      nodes.csv (id:ID,name,type:LABEL)
      relationships.csv (:START_ID,relation,:END_ID)
      load_neo4j.cypher (to import via neo4j-admin or LOAD CSV)
    Deduplicates nodes by 'name' string (head/tail)
    """
    # Create node table
    head_nodes = df[["head", "head_type"]].rename(columns={"head": "name", "head_type": "type"})
    tail_nodes = df[["tail", "tail_type"]].rename(columns={"tail": "name", "tail_type": "type"})
    nodes = pd.concat([head_nodes, tail_nodes], ignore_index=True)
    nodes["name"] = nodes["name"].astype(str)
    nodes["type"] = nodes["type"].fillna("Entity").astype(str)

    nodes = nodes.drop_duplicates(subset=["name"]).reset_index(drop=True)
    nodes["id"] = nodes["name"].apply(slugify)
    nodes = nodes[["id", "name", "type"]]

    # Map name -> id
    id_map = dict(zip(nodes["name"], nodes["id"]))

    # Relationship table
    rels = df[["head", "relation", "tail"]].copy()
    rels[":START_ID"] = rels["head"].map(id_map)
    rels[":END_ID"] = rels["tail"].map(id_map)
    rels = rels[[":START_ID", "relation", ":END_ID"]]

    nodes_csv = os.path.join(outdir, "nodes.csv")
    rels_csv = os.path.join(outdir, "relationships.csv")
    nodes.to_csv(nodes_csv, index=False)
    rels.to_csv(rels_csv, index=False)

    # Minimal Cypher (LOAD CSV)
    cypher = f"""// Run inside Neo4j Browser (after placing CSVs in import folder)
// 1) Create constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;

// 2) Load nodes (Entity label + 'type' as additional label)
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:Entity {{id: row.id}})
SET n.name = row.name
WITH n, row
CALL {{
  WITH n, row
  // Apply an extra label from the 'type' column if present
  WITH n, row
  CALL apoc.create.addLabels(n, [row.type]) YIELD node
  RETURN node
}} IN TRANSACTIONS OF 10000 ROWS;

// 3) Load relationships
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (s:Entity {{id: row.`:START_ID`}})
MATCH (t:Entity {{id: row.`:END_ID`}})
CALL apoc.create.relationship(s, row.relation, {{}}, t) YIELD rel
RETURN count(rel);
"""
    cypher_path = os.path.join(outdir, "load_neo4j.cypher")
    with open(cypher_path, "w", encoding="utf-8") as f:
        f.write(cypher)
    return nodes_csv, rels_csv, cypher_path


# --------------------------
# Main
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to triples JSON (array/dict) or JSONL.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--neo4j", action="store_true", help="Also create Neo4j CSVs + Cypher.")
    parser.add_argument("--topk_rel_plot", type=int, default=20, help="Top-K relations to plot.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[+] Loading: {args.input}")
    raw = load_any_json(args.input)
    print(f"[+] Raw items: {len(raw)}")

    print("[+] Normalizing triples…")
    df = normalize_triples(raw)
    print(f"[+] Clean triples: {len(df)}")

    # Save main triples
    triples_csv = os.path.join(args.outdir, "triples_clean.csv")
    df.to_csv(triples_csv, index=False)
    print(f"[+] Wrote {triples_csv}")

    # Summary CSVs
    rel_counts = compute_relation_counts(df)
    rel_csv = os.path.join(args.outdir, "relation_counts.csv")
    rel_counts.to_csv(rel_csv, index=False)
    print(f"[+] Wrote {rel_csv}")

    deg_df = compute_node_degrees(df)
    deg_csv = os.path.join(args.outdir, "node_degrees.csv")
    deg_df.to_csv(deg_csv, index=False)
    print(f"[+] Wrote {deg_csv}")

    # Plot
    plot_path = os.path.join(args.outdir, "relation_hist.png")
    plot_relation_hist(rel_counts, plot_path, top_k=args.topk_rel_plot)
    print(f"[+] Wrote {plot_path}")

    # Optional Neo4j files
    if args.neo4j:
        try:
            nodes_csv, rels_csv, cypher_path = make_neo4j_files(df, args.outdir)
            print(f"[+] Neo4j nodes: {nodes_csv}")
            print(f"[+] Neo4j relationships: {rels_csv}")
            print(f"[+] Cypher loader: {cypher_path}")
            print("\nNeo4j import tips:")
            print("  1) Move nodes.csv / relationships.csv into Neo4j's 'import' folder.")
            print("  2) In Neo4j Browser, run the commands in load_neo4j.cypher.")
            print("  3) Ensure APOC is enabled for apoc.create.* calls.")
        except Exception as e:
            print(f"[!] Neo4j export failed: {e}")

    # Small summary to console
    summary = {
        "num_triples": len(df),
        "num_unique_heads": df["head"].nunique(),
        "num_unique_tails": df["tail"].nunique(),
        "num_unique_nodes_approx": pd.Index(pd.concat([df["head"], df["tail"]])).nunique(),
        "num_rel_types": df["relation"].nunique()
    }
    print("\n[Summary]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
    
# Results:
#     (venv) srirupin@macbookpro Project  % python analyze_kg_triples.py --input medical_kg_triples.json --outdir ./kg_out --neo4j
# [+] Loading: medical_kg_triples.json
# [+] Raw items: 511
# [+] Normalizing triples…
# [+] Clean triples: 429
# [+] Wrote ./kg_out/triples_clean.csv
# [+] Wrote ./kg_out/relation_counts.csv
# [+] Wrote ./kg_out/node_degrees.csv
# [+] Wrote ./kg_out/relation_hist.png
# [+] Neo4j nodes: ./kg_out/nodes.csv
# [+] Neo4j relationships: ./kg_out/relationships.csv
# [+] Cypher loader: ./kg_out/load_neo4j.cypher

# Neo4j import tips:
#   1) Move nodes.csv / relationships.csv into Neo4j's 'import' folder.
#   2) In Neo4j Browser, run the commands in load_neo4j.cypher.
#   3) Ensure APOC is enabled for apoc.create.* calls.

# [Summary]
#   num_triples: 429
#   num_unique_heads: 396
#   num_unique_tails: 396
#   num_unique_nodes_approx: 792
#   num_rel_types: 5