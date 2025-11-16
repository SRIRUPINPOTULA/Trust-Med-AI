# app.py
# -----------------------------------------------------------------------------
# TrustMed AI — Conversational Agent (entity-locked + device override + timeout)
# + Qwen chat template + KG summarizer fallback + treats-filter + grammar fixes
# + phrase-aware entity detection & strict entity filtering
# + KG sanitization + intent cascade + output validation
# -----------------------------------------------------------------------------

import os
import re
import time
import pandas as pd
import gradio as gr
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# ============================ Config ============================

CONTEXT_CSV = os.getenv("CONTEXT_CSV", "kg_out/triples_clean.csv")
BASE_MODEL  = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "")  # optional LoRA path
SKIP_LORA   = os.getenv("SKIP_LORA", "0").lower() in ("1","true","yes")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))  # lowered for speed
TOP_K = int(os.getenv("TOP_K", "20"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

# Generation timeout (seconds) — if we exceed it, we return KG-only
GEN_TIMEOUT_SECS = int(os.getenv("GEN_TIMEOUT_SECS", "45"))

# Force device: "cpu" | "mps" | "cuda" (auto if unset)
FORCE_DEVICE = os.getenv("FORCE_DEVICE", "").lower()

DISCLAIMER = (
    "⚠️ I am a research assistant and not a medical professional. "
    "This is informational and not medical advice. For urgent or personal care, consult a licensed clinician."
)

# ======================= Intent & Guardrails =====================

def classify_intent(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["treat", "therapy", "manage", "medication", "drug", "intervention"]):
        return "treats"
    if any(k in ql for k in ["cause", "risk", "etiology"]):
        return "caused_by"
    if "side effect" in ql or "adverse" in ql or "reaction" in ql:
        return "side_effect"
    if "symptom" in ql or "sign" in ql:
        return "symptom_of"
    return "general"

RED_FLAGS = [
    "chest pain", "shortness of breath", "anaphylaxis", "stroke",
    "suicidal", "overdose", "severe bleeding", "unconscious"
]

def emergency_filter(q: str) -> bool:
    ql = q.lower()
    return any(flag in ql for flag in RED_FLAGS)

# ===================== Data loading / sanitization ==================

# Sanitizer for odd bytes (NBSP/replacement char)
_SAN_BAD = re.compile(r"[\uFFFD\u00A0]")

# Extra filters for fragment-y/junky rows
_BAD_STARTS = re.compile(r"^(for example|eg\.|e\.g\.|example:|note:)\b", re.I)
_TRAIL_TO = re.compile(r"\b(to|used to)\s*$", re.I)
_MANY_COMMAS = re.compile(r"(?:[^,]*,){3,}")  # 3+ commas → clause dump
_NON_WORDY = re.compile(r"^[^a-zA-Z]*$")

def _sanitize(s: str) -> str:
    return _SAN_BAD.sub(" ", (s or "")).replace("â", "").strip()

_QWORD_RX = re.compile(r"^\s*(what|which|how|when|where|why)\b", re.I)
_HAS_QMARK_RX = re.compile(r"\?\s*$")
_HAS_URL_RX = re.compile(r"https?://|www\.", re.I)
_FIRST_PERSON_RX = re.compile(r"\b(I|my|me|we|our|us)\b", re.I)
_MIN_LEN = 3
_MAX_LEN = 220

def _row_is_noisy(h: str, t: str) -> bool:
    if _QWORD_RX.search(h) or _QWORD_RX.search(t) or _HAS_QMARK_RX.search(h) or _HAS_QMARK_RX.search(t):
        return True
    if _HAS_URL_RX.search(h) or _HAS_URL_RX.search(t):
        return True
    if _FIRST_PERSON_RX.search(h) or _FIRST_PERSON_RX.search(t):
        return True
    if not (_MIN_LEN <= len(h) <= _MAX_LEN) or not (_MIN_LEN <= len(t) <= _MAX_LEN):
        return True
    if _BAD_STARTS.search(h) or _BAD_STARTS.search(t):
        return True
    if _TRAIL_TO.search(h) or _TRAIL_TO.search(t):
        return True
    if _NON_WORDY.match(h) or _NON_WORDY.match(t):
        return True
    if _MANY_COMMAS.search(h) or _MANY_COMMAS.search(t):
        return True
    return False

def sanitize_triples(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("head","relation","tail"):
        df[col] = df[col].astype(str).map(_sanitize)
    mask = ~df.apply(lambda r: _row_is_noisy(r["head"], r["tail"]), axis=1)
    df = df[mask].drop_duplicates(subset=["head","relation","tail"]).reset_index(drop=True)
    return df

def load_triples(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    for col in ["head", "relation", "tail"]:
        if col not in df.columns:
            raise ValueError("Expected columns: head, relation, tail")
    df = sanitize_triples(df)
    return df

# ===================== Entity utilities =====================

STOPWORDS = {
    "what","whats","the","a","an","for","of","in","to","on","and","or","is","are","how",
    "does","do","with","about","tell","me","please","can","you","i","we","it","that"
}
DOMAIN_STOP = {
    "treat","treats","treatment","treatments","treated","therapy","therapies",
    "management","manage","managing","medication","medications","drug","drugs",
    "symptom","symptoms","sign","signs","cause","causes","risk","risks","infection","infections",
    "disease","disorder","syndrome","pain","care","doctor","nurse","clinic","consult","guidance"
}
DISEASE_SUFFIX_RE = re.compile(r"(itis|osis|emia|algia|oma|pathy|phobia|plegia|rrhea|rrhoea)$")

def tokenize_question(q: str):
    toks_all = [t for t in re.findall(r"[a-z0-9\-]+", q.lower()) if len(t) > 2]
    return [t for t in toks_all if t not in STOPWORDS]

def count_token_hits(df: pd.DataFrame, tok: str) -> int:
    head_hit = df["head"].astype(str).str.lower().str.contains(tok, regex=False)
    tail_hit = df["tail"].astype(str).str.lower().str.contains(tok, regex=False)
    return int((head_hit | tail_hit).sum())

def _make_ngrams(tokens, nmin=2, nmax=3):
    for n in range(nmin, nmax + 1):
        for i in range(0, len(tokens) - n + 1):
            yield " ".join(tokens[i:i+n])

def _phrase_hits_in_kg(df: pd.DataFrame, phrase: str) -> int:
    words = phrase.split()
    if not words:
        return 0
    prefix = " ".join(map(re.escape, words[:-1]))
    last = re.escape(words[-1]) + r"(?:s|es)?"
    pat = rf"\b{prefix}\s+{last}\b" if prefix else rf"\b{last}\b"
    rx = re.compile(pat, flags=re.I)
    head_hit = df["head"].astype(str).str.contains(rx, na=False)
    tail_hit = df["tail"].astype(str).str.contains(rx, na=False)
    return int((head_hit | tail_hit).sum())

def _compile_entity_regex(entity: str) -> re.Pattern:
    """
    Acronyms/short tokens (<=3 chars or ALL CAPS) match EXACTLY.
    Longer words allow simple plurals (s|es) only.
    """
    words = entity.split()
    def tok(w: str) -> str:
        is_acronym = (w.isupper() and len(w) <= 6) or len(w) <= 3
        if is_acronym:
            return r"\b" + re.escape(w) + r"\b"
        return r"\b" + re.escape(w) + r"(?:s|es)?\b"
    if len(words) == 1:
        return re.compile(tok(words[0]), flags=re.I)
    prefix = r"\b" + r"\s+".join(map(re.escape, words[:-1])) + r"\s+"
    last = tok(words[-1]).lstrip(r"\b")
    return re.compile(prefix + last, flags=re.I)

def pick_main_entity(df: pd.DataFrame, q: str) -> str | None:
    toks = tokenize_question(q)
    singles = [t for t in toks if t not in DOMAIN_STOP and len(t) >= 3]
    disease_like = [t for t in singles if DISEASE_SUFFIX_RE.search(t)]
    single_pool = disease_like if disease_like else singles
    phrases = list(_make_ngrams(singles, 2, 3))

    best_phrase, best_p_hits = None, 0
    for ph in phrases:
        hits = _phrase_hits_in_kg(df, ph)
        if hits > best_p_hits:
            best_phrase, best_p_hits = ph, hits
    if best_p_hits > 0:
        return best_phrase

    if not single_pool:
        return None
    scored = [(t, count_token_hits(df, t)) for t in single_pool]
    scored.sort(key=lambda x: x[1], reverse=True)
    ent = scored[0][0] if scored and scored[0][1] > 0 else None
    if ent:
        return ent
    # Fallback: trust obvious condition mentions even if KG has no hits
    q_low = q.lower()
    obvious = [w for w in re.findall(r"[A-Za-z0-9\-]+", q) if w.isupper() or w.lower() in {"hiv","aids","covid","copd","uti","ibd","ibs"}]
    if obvious:
        return obvious[0].lower()
    if "hiv" in q_low:
        return "hiv"
    return None

# ===================== Retrieval & scoring =====================

def _treat_hint_mask(frame: pd.DataFrame):
    pat = r"\b(treat|treatment|medication|steroid|corticosteroid|antibiotic|antifungal|antiviral|antiretroviral|haart|art|arv)\b"
    head = frame["head"].astype(str).str.contains(pat, case=False, regex=True, na=False)
    tail = frame["tail"].astype(str).str.contains(pat, case=False, regex=True, na=False)
    return head | tail

def retrieve_triples(df: pd.DataFrame, question: str, intent: str, main_entity: str | None, k: int = TOP_K) -> pd.DataFrame:
    toks = tokenize_question(question)
    dfx = df.copy()

    # For treatment questions, we must know the entity (prevents irrelevant facts)
    if intent == "treats" and not main_entity:
        return pd.DataFrame(columns=df.columns)

    if "relation" in dfx.columns and intent != "general":
        dfx.loc[:, "rel_boost"] = dfx["relation"].astype(str).str.lower().apply(lambda r: 1 if intent in r else 0)
    else:
        dfx.loc[:, "rel_boost"] = 0

    # Prefer treat*; otherwise soft treatment hints
    if intent == "treats":
        treat_mask = dfx["relation"].astype(str).str.lower().str.contains("treat", na=False)
        dfx_treats = dfx[treat_mask].copy()
        if not dfx_treats.empty:
            dfx = dfx_treats
        else:
            hint_mask = _treat_hint_mask(dfx)
            dfx_hints = dfx[hint_mask].copy()
            if not dfx_hints.empty:
                dfx = dfx_hints

    # Strict entity filter
    ent_rx = _compile_entity_regex(main_entity) if main_entity else None
    if ent_rx is not None:
        ent_mask = (
            dfx["head"].astype(str).str.contains(ent_rx, na=False) |
            dfx["tail"].astype(str).str.contains(ent_rx, na=False)
        )
        dfx = dfx[ent_mask].copy()
        if dfx.empty:
            return dfx

        # Special case: avoid HIV≈hives collisions
        if (main_entity or "").lower() == "hiv":
            HIVES_RX = re.compile(r"\bhives?\b", re.I)
            bad = dfx["head"].astype(str).str.contains(HIVES_RX, na=False) | dfx["tail"].astype(str).str.contains(HIVES_RX, na=False)
            if bad.any():
                dfx = dfx[~bad].copy()
                if dfx.empty:
                    return dfx

    # For treats, prefer rows where entity appears in tail and drop diagnostic/fitness clutter
    if intent == "treats" and ent_rx is not None and "tail" in dfx.columns:
        prefer_tail = dfx["tail"].astype(str).str.contains(ent_rx, na=False)
        if prefer_tail.any():
            dfx = dfx[prefer_tail].copy()
        BAD_TREAT_HEAD_RX = re.compile(r"\b(diagnose|diagnosis|monitor|know when|doctor will|exercise|exercises|stand about|podiatrist)\b", re.I)
        bad_head = dfx["head"].astype(str).str.contains(BAD_TREAT_HEAD_RX, na=False)
        if bad_head.any():
            dfx = dfx[~bad_head].copy()
            if dfx.empty:
                return dfx

    def score_row(r):
        h = str(r.get("head","")).lower()
        t = str(r.get("tail","")).lower()
        base = 0
        for tok in toks:
            if tok in DOMAIN_STOP:
                continue
            if tok in h or tok in t:
                base += 2 if len(tok) >= 6 else 1
        if ent_rx is not None and (ent_rx.search(h) or ent_rx.search(t)):
            base += 5
        base += 2 * r["rel_boost"]
        return base

    dfx.loc[:, "score"] = dfx.apply(score_row, axis=1)
    dfx = dfx.sort_values("score", ascending=False)
    return dfx[dfx["score"] > 0].head(k).copy()

# ---- Intent cascade ----

INTENT_NEIGHBORS = {
    "treats": ["treats"],  # no pivot for treatment questions
    "symptom_of": ["symptom_of", "caused_by", "treats", "general"],
    "caused_by": ["caused_by", "symptom_of", "treats", "general"],
    "side_effect": ["side_effect", "treats", "general"],
    "general": ["treats", "caused_by", "symptom_of", "general"],
}

def retrieve_with_cascade(df: pd.DataFrame, question: str, intent: str, main_entity: str | None, k: int = TOP_K):
    tried = []
    best_rows = pd.DataFrame()
    best_intent = intent
    for it in INTENT_NEIGHBORS.get(intent, [intent]):
        rows = retrieve_triples(df, question, intent=it, main_entity=main_entity, k=k)
        tried.append((it, len(rows)))
        if len(rows) >= 2:
            return rows, it, tried
        if not rows.empty and best_rows.empty:
            best_rows, best_intent = rows, it
    return best_rows, best_intent, tried

def format_context(rows: pd.DataFrame, max_facts: int = 20) -> str:
    if rows is None or rows.empty:
        return "(no facts)"
    lines = []
    for _, r in rows.iterrows():
        h = _sanitize(str(r.get("head", "")))
        rel = _sanitize(str(r.get("relation", "")))
        t = _sanitize(str(r.get("tail", "")))
        lines.append(f"[H:{h} | R:{rel} | T:{t}]")
        if len(lines) >= max_facts:
            break
    return "\n".join(lines)

# ================= Prompt & post-processing / parsing =============

PROMPT_TMPL = """You are TrustMed, a careful medical information assistant.
Use ONLY the facts listed in <CONTEXT>. If information is missing, say you don't know.
Do not add other knowledge. Do not include examples.
Focus only on the condition: "{entity}". Ignore facts about other conditions.

<QUESTION>
{q}
</QUESTION>

<CONTEXT>
{ctx}
</CONTEXT>

<OUTPUT_FORMAT>
Begin immediately with:
[Answer]
- 2 to 5 clear sentences summarizing what the facts say about "{entity}". No speculation.

[Facts Used]
- 3 to 6 bullet points. Each bullet must be in the form: [H:head | R:relation | T:tail]
- Use only facts that appear in <CONTEXT> and that mention "{entity}".

[Disclaimer]
- Exactly this line: {disc}
</OUTPUT_FORMAT>
"""

def kg_only_response(ctx_rows: pd.DataFrame) -> str:
    if ctx_rows is None or ctx_rows.empty:
        return (
            "[Answer]\nI don't have enough grounded information to answer that from the available knowledge graph.\n\n"
            "[Facts Used]\n(none)\n\n"
            f"[Disclaimer]\n{DISCLAIMER}"
        )
    bullets = []
    for h, r, t in ctx_rows[["head", "relation", "tail"]].values[:6]:
        bullets.append(f"- [H:{_sanitize(str(h))} | R:{_sanitize(str(r))} | T:{_sanitize(str(t))}]")
    return (
        "[Answer]\nThe following knowledge graph facts are relevant to your question. "
        "I will not speculate beyond these facts.\n\n"
        "[Facts Used]\n" + "\n".join(bullets) + "\n\n" +
        f"[Disclaimer]\n{DISCLAIMER}"
    )

def extract_three_sections(text: str) -> str | None:
    m = re.search(r"\[Answer\].*?\[Facts\s*Used\].*?\[Disclaimer\].*", text, flags=re.S | re.I)
    if not m:
        return None
    out = m.group(0)
    lines = out.splitlines()
    clean = []
    disclaimer_normed = False
    for ln in lines:
        if ln.strip().lower().startswith("[disclaimer]"):
            clean.append("[Disclaimer]")
            clean.append(DISCLAIMER)
            disclaimer_normed = True
            break
        clean.append(ln)
    out = "\n".join(clean).strip()
    if "[Disclaimer]" not in out or not disclaimer_normed:
        out += f"\n\n[Disclaimer]\n{DISCLAIMER}"
    return out

# -------- Output validation to avoid bad fragments --------
_BAD_FACT_PATTERNS = [
    re.compile(r"^\[H:\s*for example", re.I),
    re.compile(r"\bused to\s*$", re.I),
    re.compile(r"\bexample\b", re.I),
]

def output_looks_sus(raw_out: str) -> bool:
    sect = re.search(r"\[Facts\s*Used\](.*?)(?:\n\[|$)", raw_out, flags=re.S|re.I)
    if not sect:
        return True
    body = sect.group(1)
    if len(body.strip()) == 0:
        return True
    for rx in _BAD_FACT_PATTERNS:
        if rx.search(body):
            return True
    return False

# ===================== Text generation (HF/PEFT) =================

_textgen_pipe = None  # lazy global

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
        dtype = torch.float16
    elif dev == "mps":
        dtype = torch.float32  # more stable
    else:
        dtype = torch.float32
    return dev, dtype

def build_generator(base_model: str, adapter_dir: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    device_str, dtype = _pick_device_and_dtype()
    print(f"[INFO] Using device={device_str}, dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        attn_implementation="eager",
        device_map=None,
        low_cpu_mem_usage=False,
    )

    if adapter_dir and os.path.isdir(adapter_dir) and not SKIP_LORA:
        try:
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
            print(f"[INFO] Loaded LoRA adapter from: {adapter_dir}")
        except Exception as e:
            print(f"[WARN] Could not load LoRA adapter '{adapter_dir}': {e}. Continuing without adapter.")
    elif SKIP_LORA:
        print("[INFO] SKIP_LORA=1 → not loading adapter")

    import torch
    device = torch.device(device_str)
    base.to(device).eval()

    gen = pipeline(
        "text-generation",
        model=base,
        tokenizer=tok,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=False,
        device=device,
        return_full_text=False,
    )

    # Warm-up
    try:
        _ = gen("[Answer]\n", max_new_tokens=1)[0]["generated_text"]
        print("[INFO] Warm-up done")
    except Exception as e:
        print(f"[WARN] Warm-up failed: {e}")

    # Attach tokenizer for prompt building
    gen.tokenizer_ref = tok
    return gen

def ensure_pipe():
    global _textgen_pipe
    if _textgen_pipe is not None:
        return _textgen_pipe
    if not BASE_MODEL:
        return None
    try:
        _textgen_pipe = build_generator(BASE_MODEL, ADAPTER_DIR or None)
        return _textgen_pipe
    except Exception as e:
        print(f"[WARN] Falling back to KG-only mode (generator init failed): {e}")
        _textgen_pipe = None
        return None

# --------- Qwen chat-style prompt helpers ----------

def build_messages(question: str, ctx_txt: str, entity: str | None):
    system = (
        "You are TrustMed, a careful medical information assistant. "
        "Use ONLY the facts listed in <CONTEXT>. If information is missing, say you don't know. "
        "Do not add other knowledge. Do not include examples. "
        f'Focus only on the condition: "{entity or "this condition"}". Ignore facts about other conditions. '
        "You MUST reply using exactly the following three sections, in order, nothing else:\n\n"
        "[Answer]\n- 2 to 5 clear sentences summarizing what the facts say about the condition. No speculation.\n\n"
        "[Facts Used]\n- 3 to 6 bullet points. Each bullet must be in the form: [H:head | R:relation | T:tail]\n"
        "- Use only facts that appear in <CONTEXT> and that mention the condition.\n\n"
        f"[Disclaimer]\n- Exactly this line: {DISCLAIMER}"
    )
    user = (
        "<QUESTION>\n" + question + "\n</QUESTION>\n\n"
        "<CONTEXT>\n" + ctx_txt + "\n</CONTEXT>\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]

def build_prompt(question: str, ctx_txt: str, entity: str | None, tok=None) -> str:
    if tok is not None and hasattr(tok, "apply_chat_template"):
        msgs = build_messages(question, ctx_txt, entity)
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return PROMPT_TMPL.format(
        q=question, ctx=ctx_txt, disc=DISCLAIMER, entity=(entity or "this condition")
    )

# --------- Deterministic, relation-aware summarizer ----------

def summarize_answer_from_facts(ctx_rows: pd.DataFrame, entity: str | None,
                                requested_intent: str, eff_intent: str,
                                max_sents: int = 5) -> str:
    """
    Deterministic, grammar-safe summarizer. Keeps outputs entity-locked and clear.
    """
    ent = (entity or "").strip()
    if not ent:
        return "I don't have enough grounded information about the specific condition to summarize treatments."

    # exact-ish entity regex (HIV won't match 'hives')
    def _compile_entity_regex_exact(e: str) -> re.Pattern:
        words = e.split()
        def tok(w: str) -> str:
            is_acronym = (w.isupper() and len(w) <= 6) or len(w) <= 3
            if is_acronym:
                return r"\b" + re.escape(w) + r"\b"
            return r"\b" + re.escape(w) + r"(?:s|es)?\b"
        if len(words) == 1:
            return re.compile(tok(words[0]), flags=re.I)
        prefix = r"\b" + r"\s+".join(map(re.escape, words[:-1])) + r"\s+"
        last = tok(words[-1]).lstrip(r"\b")
        return re.compile(prefix + last, flags=re.I)

    ent_rx = _compile_entity_regex_exact(ent)

    def fix(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s+([,.;:?!])", r"\1", s)
        s = re.sub(r"\bthe the\b", "the", s, flags=re.I)
        s = re.sub(r"\bis is\b", "is", s, flags=re.I)
        s = re.sub(r"\bare is\b", "are", s, flags=re.I)
        s = re.sub(r"\bis are\b", "are", s, flags=re.I)
        s = re.sub(r"\b(hiv)\b", "HIV", s, flags=re.I)
        if not s.endswith((".", "!", "?")):
            s += "."
        return s[:1].upper() + s[1:]

    def copula_for(head: str) -> str:
        if re.search(r"\b(drugs|medicines|medications|steroids|antiretrovirals|antivirals|antibiotics)\b", head, re.I) or head.rstrip().endswith("s"):
            return "are"
        return "is"

    BAD_HEAD = re.compile(r"\b(diagnos|monitor|doctor will|exercise|exercises|stand about|podiatrist|know when)\b", re.I)
    def mentions_ent(s: str) -> bool:
        return bool(ent_rx.search(s or ""))

    def sent_treat(h, t) -> str | None:
        h, t = _sanitize(h), _sanitize(t)
        if BAD_HEAD.search(h):
            return None
        if re.search(r"\bused to\b", h, re.I):
            cop = copula_for(h)
            h = re.sub(r"\b(is|are|was|were)?\s*(mainly\s+)?used\s+to\b", f"{cop} used to treat", h, flags=re.I)
            return fix(f"{h} {ent}")
        if not re.search(r"\btreat\b", h, re.I):
            return fix(f"{h} {copula_for(h)} used to treat {ent}")
        if not mentions_ent(h) and mentions_ent(t):
            return fix(f"{h} {t}")
        return fix(h)

    def sent_caused_by(h, t) -> str | None:
        h, t = _sanitize(h), _sanitize(t)
        if mentions_ent(h) and not mentions_ent(t):
            h_clean = re.sub(r"\s+(be|is|are)\s*$", "", h, flags=re.I)
            return fix(f"{h_clean} is caused by {t}")
        if mentions_ent(t):
            return fix(f"{ent} can be caused by {h}")
        return None

    def sent_symptom_of(h, t) -> str | None:
        h, t = _sanitize(h), _sanitize(t)
        if mentions_ent(t):
            return fix(f"Symptoms of {ent} may include {h}")
        if mentions_ent(h):
            return fix(f"{h} may be a symptom of {t}")
        return None

    def sent_side_effect(h, t) -> str | None:
        h, t = _sanitize(h), _sanitize(t)
        if mentions_ent(t):
            return fix(f"{h} can be a side effect of treatments for {ent}")
        if mentions_ent(h):
            return fix(f"{h} can be a side effect of {t}")
        return None

    buckets = {"treats": [], "caused_by": [], "symptom_of": [], "side_effect": [], "other": []}
    for _, r in ctx_rows.iterrows():
        h = str(r.get("head", "") or "")
        rel = str(r.get("relation", "") or "").strip().lower()
        t = str(r.get("tail", "") or "")

        if not (mentions_ent(h) or mentions_ent(t)):
            continue

        if rel.startswith("treat"):
            s = sent_treat(h, t)
            if s: buckets["treats"].append(s)
        elif rel.startswith("cause"):
            s = sent_caused_by(h, t)
            if s: buckets["caused_by"].append(s)
        elif rel.startswith("symptom"):
            s = sent_symptom_of(h, t)
            if s: buckets["symptom_of"].append(s)
        elif rel.startswith("side"):
            s = sent_side_effect(h, t)
            if s: buckets["side_effect"].append(s)
        else:
            txt = _sanitize(h + " " + rel + " " + t).strip()
            if txt:
                buckets["other"].append(fix(txt))

    order_map = {
        "treats":      ["treats", "caused_by", "symptom_of", "side_effect", "other"],
        "caused_by":   ["caused_by", "treats", "symptom_of", "side_effect", "other"],
        "symptom_of":  ["symptom_of", "treats", "caused_by", "side_effect", "other"],
        "side_effect": ["side_effect", "treats", "caused_by", "symptom_of", "other"],
        "general":     ["treats", "caused_by", "symptom_of", "side_effect", "other"],
    }
    pick_order = order_map.get(requested_intent, order_map["general"])

    out, seen = [], set()
    for key in pick_order:
        for s in buckets[key]:
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
            if len(out) >= max_sents:
                break
        if len(out) >= max_sents:
            break

    if requested_intent == "treats" and not buckets["treats"]:
        if out:
            out.insert(0, f"I don't have treatment-specific facts for {ent}. Here are other grounded facts.")
        else:
            return f"I don't have enough grounded information to answer treatments for {ent}."

    if not out:
        return "I don't have enough grounded information to answer that from the available knowledge graph."
    return " ".join(out[:max_sents])

# -----------------------------------------------------

def _call_pipe(pipe, prompt: str) -> str:
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: pipe(prompt)[0]["generated_text"])
        return fut.result(timeout=GEN_TIMEOUT_SECS)

def generate_answer(pipe, question: str, ctx_txt: str, entity: str | None) -> str:
    tok = getattr(pipe, "tokenizer_ref", None) or getattr(pipe, "tokenizer", None)
    prompt = build_prompt(question, ctx_txt, entity, tok)
    try:
        raw = _call_pipe(pipe, prompt)
    except FuturesTimeout:
        print(f"[WARN] Generation exceeded {GEN_TIMEOUT_SECS}s; falling back to KG-only.")
        return ""
    except Exception as e:
        print(f"[WARN] Generation error: {e}")
        return ""
    keep = extract_three_sections(raw)
    if keep and not output_looks_sus(keep):
        return keep
    try:
        raw2 = _call_pipe(pipe, prompt)
        keep2 = extract_three_sections(raw2)
        if keep2 and not output_looks_sus(keep2):
            return keep2
    except Exception as e:
        print(f"[WARN] Retry generation error: {e}")
    return ""

# ============================== App ==============================

TRIPLES = load_triples(CONTEXT_CSV)

def chat_answer(question: str, history):
    if emergency_filter(question):
        return ("[Answer]\nThis may be urgent. Please seek immediate medical attention or "
                "contact local emergency services.\n\n[Facts Used]\n(none)\n\n[Disclaimer]\n" + DISCLAIMER)

    intent = classify_intent(question)
    entity = pick_main_entity(TRIPLES, question)
    # Trust obvious mentions (e.g., HIV) if KG couldn’t resolve
    if not entity:
        qlow = question.lower()
        if "hiv" in qlow:
            entity = "hiv"
        else:
            m_acr = re.search(r"\b[A-Z]{2,5}\b", question)
            if m_acr:
                entity = m_acr.group(0).lower()

    ctx_rows, eff_intent, _tried = retrieve_with_cascade(
        TRIPLES, question, intent=intent, main_entity=entity, k=TOP_K
    )

    if ctx_rows is None or ctx_rows.empty:
        return (
            "[Answer]\nI don't have enough grounded information to answer that from the available knowledge graph.\n\n"
            "[Facts Used]\n(none)\n\n"
            f"[Disclaimer]\n{DISCLAIMER}"
        )

    ctx_txt = format_context(ctx_rows)
    pipe = ensure_pipe()
    if pipe is None:
        return kg_only_response(ctx_rows)

    out = generate_answer(pipe, question, ctx_txt, entity)
    if not out:
        answer_text = summarize_answer_from_facts(ctx_rows, entity, requested_intent=intent, eff_intent=eff_intent, max_sents=3)
        ent_rx = _compile_entity_regex(entity) if entity else None
        bullets = []
        for h, r, t in ctx_rows[["head", "relation", "tail"]].values:
            hs, rs, ts = _sanitize(str(h)), _sanitize(str(r)), _sanitize(str(t))
            if ent_rx is None or ent_rx.search(hs) or ent_rx.search(ts):
                bullets.append(f"- [H:{hs} | R:{rs} | T:{ts}]")
            if len(bullets) >= 6:
                break
        if not bullets:
            for h, r, t in ctx_rows[["head", "relation", "tail"]].values[:3]:
                bullets.append(f"- [H:{_sanitize(str(h))} | R:{_sanitize(str(r))} | T:{_sanitize(str(t))}]")
        out = "[Answer]\n" + answer_text + "\n\n[Facts Used]\n" + "\n".join(bullets) + "\n\n[Disclaimer]\n" + DISCLAIMER
    return out

with gr.Blocks() as demo:
    gr.Markdown("# TrustMed AI — Conversational Agent")
    gr.Markdown(
        "This tool provides general medical information grounded in a curated knowledge graph. "
        "It is not a substitute for professional medical advice."
    )
    examples = [
        "What are the treatments for a kidney infection?",
        "What are the side effects of antibiotics?",
        "What causes tonsillitis?",
        "What are symptoms of a UTI in children?",
        "How do you treat hay fever?",
        "How is Strep A treated?",
        "What causes VTEC O157 infections?"
    ]
    gr.ChatInterface(
        fn=chat_answer,
        title="TrustMed Chat",
        examples=examples,
        type="messages",
    )

# --- launch (public link)
if __name__ == "__main__":
    demo.launch(share=True)

# export BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
# export ADAPTER_DIR="kg_lora_out_chat"
# export SKIP_LORA=0
# export FORCE_DEVICE=cpu
# export MAX_NEW_TOKENS=160
# python app.py
