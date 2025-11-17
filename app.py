# app.py
# -----------------------------------------------------------------------------
# TrustMed AI — Conversational Agent (Gradio)
# Now using alias-aware, intent-specific, KG-constrained answering logic
# similar to your CLI pattern (knowledge lines + allowed entities + bullet output),
# with deterministic KG-only fallback and a clean disclaimer section.
# -----------------------------------------------------------------------------

import os
import re
from typing import List, Tuple, Optional, Set, Dict

import pandas as pd
import gradio as gr

# =============== Config ===============
CONTEXT_CSV = os.getenv("CONTEXT_CSV", "kg_out/triples_clean.csv")
BASE_MODEL  = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "")   # optional LoRA path
SKIP_LORA   = os.getenv("SKIP_LORA", "0").lower() in ("1","true","yes")
FORCE_DEVICE = os.getenv("FORCE_DEVICE", "").lower()  # "cpu" | "mps" | "cuda" (auto if unset)

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))
GEN_TIMEOUT_SECS = int(os.getenv("GEN_TIMEOUT_SECS", "45"))
BULLETS_DEFAULT = int(os.getenv("BULLETS", "4"))

DISCLAIMER = (
    "⚠️ I am a research assistant and not a medical professional. "
    "This is informational and not medical advice. For urgent or personal care, consult a licensed clinician."
)

# =============== Aliases & Relations ===============
ALIAS_CLUSTERS: Dict[str, Set[str]] = {
    # Type 2 diabetes
    "type 2 diabetes": {
        "type 2 diabetes", "type-2 diabetes", "t2d", "t2dm",
        "type ii diabetes", "dm2", "diabetes mellitus type 2",
    },
    # Strep throat
    "strep throat": {
        "strep throat", "streptococcal pharyngitis", "group a strep pharyngitis",
        "gas pharyngitis", "streptococcus pyogenes pharyngitis", "strep a pharyngitis",
    },
    # Hay fever
    "hay fever": {
        "hay fever", "allergic rhinitis", "seasonal allergic rhinitis",
    },
}

REL_GROUPS = {
    "treats": {
        "TREATS", "TREATED_BY", "INDICATION_FOR", "MANAGES", "THERAPY_FOR"
    },
    "caused_by": {
        "CAUSED_BY", "CAUSES", "ETIOLOGY", "RESULT_OF", "DUE_TO"
    },
    "symptom_of": {
        "HAS_SYMPTOM", "SYMPTOM_OF", "PRESENTS_WITH", "SIGN_OF"
    },
    "side_effect": {
        "HAS_SIDE_EFFECT", "SIDE_EFFECT_OF", "ADVERSE_EVENT", "ADVERSE_EFFECT"
    },
}

# =============== Intent & Emergencies ===============
RED_FLAGS = [
    "chest pain", "shortness of breath", "anaphylaxis", "stroke",
    "suicidal", "overdose", "severe bleeding", "unconscious"
]

def emergency_filter(q: str) -> bool:
    ql = q.lower()
    return any(flag in ql for flag in RED_FLAGS)


def classify_intent(q: str) -> str:
    ql = q.lower()
    if any(w in ql for w in ["treat", "therapy", "manage", "medication", "drug", "intervention"]):
        return "treats"
    if any(w in ql for w in ["cause", "risk", "etiolog", "due to", "result of"]):
        return "caused_by"
    if any(w in ql for w in ["symptom", "sign"]):
        return "symptom_of"
    if any(w in ql for w in ["side effect", "adverse", "reaction"]):
        return "side_effect"
    return "treats"  # helpful default

# =============== Light sanitization ===============
_SAN_BAD = re.compile(r"[\uFFFD\u00A0]")

def _sanitize(s: str) -> str:
    return _SAN_BAD.sub(" ", str(s)).strip()

# =============== Entity helpers ===============

def detect_entity(q: str) -> Optional[str]:
    ql = q.lower()
    for canon, aliases in ALIAS_CLUSTERS.items():
        if any(a in ql for a in aliases):
            return canon
    if "diabetes" in ql and ("type 2" in ql or "type-2" in ql or "ii" in ql or "t2" in ql):
        return "type 2 diabetes"
    if "strep" in ql or "pharyngitis" in ql:
        return "strep throat"
    if "hay fever" in ql or ("allergic" in ql and "rhinitis" in ql):
        return "hay fever"
    return None


def expand_aliases(entity: Optional[str]) -> Set[str]:
    if not entity:
        return set()
    e = entity.lower().strip()
    if e in ALIAS_CLUSTERS:
        return set(ALIAS_CLUSTERS[e])
    for canon, aliases in ALIAS_CLUSTERS.items():
        if e in aliases:
            return set(aliases)
    return {e}

# =============== Relation normalization ===============

def normalize_relation(r: str) -> str:
    rl = str(r or "").lower()
    if any(k in rl for k in ["treat", "indication", "manage", "therapy"]):
        return "TREATS"
    if any(k in rl for k in ["cause", "etiolog", "due to", "result"]):
        return "CAUSED_BY"
    if any(k in rl for k in ["symptom", "sign", "presents with"]):
        return "HAS_SYMPTOM"
    if any(k in rl for k in ["side effect", "adverse", "reaction"]):
        return "HAS_SIDE_EFFECT"
    return rl.upper()

# =============== Triple selection (alias + intent) ===============

def _row_mentions_any(text: str, needles: Set[str]) -> bool:
    tl = text.lower()
    return any(n in tl for n in needles)


def select_triples(df: pd.DataFrame, q: str, intent: str, entity: Optional[str], max_triples: int = 20) -> List[Tuple[str,str,str]]:
    if df.empty:
        return []
    rels = REL_GROUPS.get(intent, set())
    if not rels:
        rels = set().union(*REL_GROUPS.values())

    cand = df.copy()
    cand["rel_norm"] = cand["relation"].map(normalize_relation)
    cand = cand[cand["rel_norm"].isin(rels)]

    aliases = expand_aliases(entity)
    if aliases:
        mask = cand.apply(lambda r: _row_mentions_any(str(r["head"]), aliases) or _row_mentions_any(str(r["tail"]), aliases), axis=1)
        cand = cand[mask]
    else:
        # loose: any overlap with query tokens
        qtok = set(re.findall(r"[a-z0-9\-+]+", q.lower()))
        loose = {t for t in qtok if len(t) >= 3}
        mask = cand.apply(lambda r: _row_mentions_any(str(r["head"]), loose) or _row_mentions_any(str(r["tail"]), loose), axis=1)
        cand = cand[mask]

    if cand.empty:
        return []

    # scoring
    qtokens = set(re.findall(r"[a-z0-9\-+]+", q.lower())) | aliases
    def score_row(r):
        text = f"{r['head']} {r['tail']} {r['relation']}".lower()
        hits = sum(1 for t in qtokens if t in text)
        w = float(r.get("weight", 1.0))
        tail_bonus = 1.0
        if intent == "treats" and aliases:
            if _row_mentions_any(str(r["tail"]), aliases):
                tail_bonus = 1.5
        return hits * 10 * tail_bonus + w

    cand["score"] = cand.apply(score_row, axis=1)
    cand = cand.sort_values("score", ascending=False).head(max_triples)

    triples = list(zip(map(_sanitize, cand["head"]), map(_sanitize, cand["relation"]), map(_sanitize, cand["tail"])) )
    return triples

# =============== Prompt construction (knowledge-constrained) ===============

def build_prompt(question: str, triples: List[Tuple[str,str,str]], bullets: int = 4, intent: str = "treats") -> str:
    knowledge_lines = [f"- {h} — {r} → {t}" for (h,r,t) in triples]
    knowledge = "Knowledge (KG facts — use ONLY these):\n" + "\n".join(knowledge_lines) + "\n\n"
    entities = sorted({h for h,_,_ in triples} | {t for _,_,t in triples})
    entity_list = "Allowed entities: " + ", ".join(entities[:80]) + ".\n\n"
    intent_phrase = {
        "treats": "List evidence-backed treatments/medications.",
        "caused_by": "List main causes/etiologies.",
        "symptom_of": "List common symptoms/signs.",
        "side_effect": "List known side effects/adverse effects.",
    }.get(intent, "Answer briefly.")
    system = (
        "Instruction: You are a careful medical assistant. "
        f"Answer in {bullets}-{bullets+2} concise bullet points. "
        f"{intent_phrase} Use ONLY the allowed entities; do NOT invent new facts. "
        "Do NOT add a disclaimer line; it will be added separately.\n"
    )
    return f"{system}{knowledge}{entity_list}Question: {question}\nAnswer:"

# =============== Post-processing (no disclaimer here) ===============

def post_process_no_disc(text: str) -> str:
    txt = text.strip()
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in txt.splitlines() if ln.strip()]
    out, seen = [], set()
    for ln in lines:
        low = ln.lower()
        if "not medical advice" in low:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(ln)
    # keep at most ~10 lines
    return "\n".join(out[:12])

# =============== Text generation backend ===============
_textgen_pipe = None

def _pick_device_and_dtype():
    import torch
    if FORCE_DEVICE in ("cpu","mps","cuda"):
        dev = FORCE_DEVICE
    else:
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    if dev == "cuda":
        dtype = "float16"
    elif dev == "mps":
        dtype = "float32"
    else:
        dtype = "float32"
    return dev, dtype


def ensure_pipe():
    global _textgen_pipe
    if _textgen_pipe is not None:
        return _textgen_pipe
    if not BASE_MODEL:
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
        device_str, _ = _pick_device_and_dtype()
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        # Optional: LoRA
        if ADAPTER_DIR and os.path.isdir(ADAPTER_DIR) and not SKIP_LORA:
            try:
                from peft import PeftModel
                base = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
            except Exception:
                pass
        device = torch.device(device_str)
        base.to(device).eval()
        gen = pipeline(
            "text-generation",
            model=base,
            tokenizer=tok,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            device=device,
            return_full_text=False,
        )
        _textgen_pipe = gen
    except Exception as e:
        print(f"[WARN] Could not init generator: {e}")
        _textgen_pipe = None
    return _textgen_pipe

# =============== Deterministic KG-only (bullets) ===============

def kg_only_answer(triples: List[Tuple[str,str,str]], intent: str, entity: Optional[str]) -> str:
    aliases = expand_aliases(entity)
    items: List[str] = []
    for h, r, t in triples:
        R = normalize_relation(r)
        if intent == "treats" and R in REL_GROUPS["treats"]:
            if aliases and any(a in h.lower() for a in aliases):
                items.append(t)
            elif aliases and any(a in t.lower() for a in aliases):
                items.append(h)
            else:
                items.append(h)
        elif intent == "caused_by" and R in REL_GROUPS["caused_by"]:
            items.append(t if R == "CAUSED_BY" else h)
        elif intent == "symptom_of" and R in REL_GROUPS["symptom_of"]:
            items.append(t if R == "HAS_SYMPTOM" else h)
        elif intent == "side_effect" and R in REL_GROUPS["side_effect"]:
            items.append(t if R == "HAS_SIDE_EFFECT" else h)
    items = [s for s in map(_sanitize, items) if s]
    items = list(dict.fromkeys(items))
    bullets = "\n".join(f"- {m}" for m in items[:10]) if items else "- (no facts found in KG context)"
    return bullets

# =============== Gradio chat function ===============

def chat_answer(question: str, history):
    if emergency_filter(question):
        return (
            "[Answer]\nThis may be urgent. Please seek immediate medical attention or contact local emergency services.\n\n"
            "[Facts Used]\n(none)\n\n[Disclaimer]\n" + DISCLAIMER
        )

    # Load triples
    try:
        df = pd.read_csv(CONTEXT_CSV)
    except Exception as e:
        return (
            "[Answer]\nI couldn't load the knowledge graph file.\n\n[Facts Used]\n(none)\n\n[Disclaimer]\n" + DISCLAIMER
        )
    df.columns = [c.lower() for c in df.columns]
    if not set(["head","relation","tail"]).issubset(df.columns):
        return (
            "[Answer]\nThe knowledge graph CSV is missing required columns (head, relation, tail).\n\n[Facts Used]\n(none)\n\n[Disclaimer]\n" + DISCLAIMER
        )

    intent = classify_intent(question)
    entity = detect_entity(question)

    triples = select_triples(df, question, intent, entity, max_triples=20)
    if not triples:
        return (
            "[Answer]\nI don't have enough grounded information to answer that from the available knowledge graph.\n\n"
            "[Facts Used]\n(none)\n\n[Disclaimer]\n" + DISCLAIMER
        )

    # Try LLM generation with knowledge-constrained prompt
    pipe = ensure_pipe()
    if pipe is not None:
        try:
            prompt = build_prompt(question, triples, bullets=BULLETS_DEFAULT, intent=intent)
            out = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS)[0]["generated_text"]
            bullets = post_process_no_disc(out)
        except Exception:
            bullets = kg_only_answer(triples, intent, entity)
    else:
        bullets = kg_only_answer(triples, intent, entity)

    # Facts Used block (first 6 triples)
    facts = []
    for h, r, t in triples[:6]:
        facts.append(f"- [H:{_sanitize(h)} | R:{_sanitize(r)} | T:{_sanitize(t)}]")

    return (
        "[Answer]\n" + bullets + "\n\n" +
        "[Facts Used]\n" + ("\n".join(facts) if facts else "(none)") + "\n\n" +
        "[Disclaimer]\n" + DISCLAIMER
    )

# =============== UI ===============
with gr.Blocks() as demo:
    gr.Markdown("# TrustMed AI — Conversational Agent")
    gr.Markdown(
        "This tool provides general medical information grounded in a curated knowledge graph. "
        "It is not a substitute for professional medical advice."
    )
    examples = [
        "How is type 2 diabetes managed?",
        "What are the treatments for strep throat?",
        "Which medicines treat hay fever?",
        "What causes tonsillitis?",
        "What are symptoms of a UTI in children?",
        "What are the side effects of antibiotics?",
    ]
    gr.ChatInterface(
        fn=chat_answer,
        title="TrustMed Chat",
        examples=examples,
        type="messages",
    )

if __name__ == "__main__":
    demo.launch(share=True)

# export CONTEXT_CSV="kg_out/triples_clean.csv"
# export BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
# export ADAPTER_DIR="kg_lora_out"
# (venv) srirupin@Potulas-MacBook-Pro Maked code % python app.py
