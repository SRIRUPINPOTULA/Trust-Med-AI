# 🏥 TrustMed AI — Conversational Medical Agent

---

## 🧠 Project Overview

**TrustMed AI** is a medical conversational agent that leverages **Knowledge Graphs (KGs)** and **trusted medical sources** to provide structured, evidence-based responses about symptoms, treatments, drugs, causes, and side effects.

It combines **web-scraped data**, **language models**, and **graph-based reasoning** to answer medical-related queries in a **factual and explainable** way.

---

## 💡 Features

- ✅ Pulls verified information from trusted sources such as **NEJM**, **JAMA**, **Mayo Clinic**, and **WebMD**.  
- 🩺 Organizes **symptoms**, **treatments**, **drugs**, **causes**, and **side-effects** into structured relationships.  
- 💬 Supports **natural-language queries** through a chatbot built using **Gradio**.  
- 📚 Cites **authoritative references** and blends medical evidence with community insights.  
- 🧩 Generates **structured datasets** for model fine-tuning using **Knowledge Graph triples**.  
- ⚡ Uses **LoRA adapters** for efficient fine-tuning of large language models.  

---

## 🌐 Sources and Targets

The agent integrates medical information from reputable websites and communities:

- Reddit (e.g., `r/AskDocs`, `r/Medical`, diabetes-related subreddits)  
- [PatientsLikeMe](https://www.patientslikeme.com)  
- [HealthBoards](https://www.healthboards.com)  
- [Diabetes.co.uk](https://www.diabetes.co.uk)  
- [WebMD](https://messageboards.webmd.com)  
- [Patient.info](https://patient.info/forums)  
- [Mayo Clinic](https://connect.mayoclinic.org/groups)  
- **Ontologies:** Uses **UMLS** for medical entity alignment and knowledge integration.

---

## 🛠️ Tools & Technologies

| **Category** | **Tools** |
|---------------|------------|
| **Scraping & Automation** | Selenium, Playwright |
| **Data Processing** | Pandas, Scikit-learn, NetworkX |
| **Deep Learning** | PyTorch, Transformers, PEFT (LoRA) |
| **Visualization** | Matplotlib |
| **Interface** | Gradio |
| **Ontology Integration** | UMLS |
| **Storage** | CSV / JSON datasets, Neo4j export |

---

## ⚙️ Pipeline Overview

The project consists of four modular components forming a full **Knowledge Graph workflow**:

| **Stage** | **Script** | **Description** | **Input** | **Output** | **Command** |
|------------|-------------|------------------|------------|-------------|--------------|
| **1️⃣ KG Analysis** | `analyze_kg_triples.py` | Cleans and normalizes medical triples; creates CSVs, visualizations, and Neo4j files. | `medical_kg_triples.json` | `triples_clean.csv`, `relation_counts.csv`, `relation_hist.png` | ```bash\npython analyze_kg_triples.py --input medical_kg_triples.json --outdir ./kg_out --neo4j\n``` |
| **2️⃣ Dataset Creation** | `make_dataset_from_triples.py` | Builds instruction-style QA pairs from triples for model training. | `kg_out/triples_clean.csv` | `dataset/train.json`, `dataset/test.json` | ```bash\npython make_dataset_from_triples.py --input ./kg_out/triples_clean.csv --output ./dataset/\n``` |
| **3️⃣ LoRA Fine-tuning** | `train_lora_masked.py` | Fine-tunes transformer model (e.g., Qwen) using LoRA on dataset. | `dataset/` | `lora_adapter/` | ```bash\npython train_lora_masked.py --model Qwen/Qwen2.5-1.5B-Instruct --data ./dataset/ --output ./lora_adapter/\n``` |
| **4️⃣ Chatbot Interface** | `app.py` | Launches interactive **Gradio app** for querying the KG and fine-tuned model. | `triples_clean.csv`, model weights | Web UI | ```bash\npython app.py\n``` |

---

## ⚡ Example Usage

```bash
# Step 1: Analyze KG triples
python analyze_kg_triples.py --input medical_kg_triples.json --outdir ./kg_out --neo4j

# Step 2: Generate dataset
python make_dataset_from_triples.py --input ./kg_out/triples_clean.csv --output ./dataset/

# Step 3: Fine-tune model
python train_lora_masked.py --model Qwen/Qwen2.5-1.5B-Instruct --data ./dataset/ --output ./lora_adapter/

# Step 4: Launch the chatbot
python app.py
```

# 🧪 **Model Evaluation Guide (README)**

This guide shows you how to evaluate your **LoRA‑trained** model versus the **base** model on fixed validation and training splits, then run **judge‑free A/B testing** with per‑example F1 and a **paired bootstrap CI** for the F1 lift.

# 📁 **Before You Start**

* Change into your evaluation directory:

```bash
cd Evaluation/
```

* (Optional but recommended) ensure the output folder exists:

```bash
mkdir -p results
```

# ⚙️ **0) One‑Time Setup**

Make packages importable and install dependencies.

```bash
# make packages importable
touch adapters/__init__.py scripts/__init__.py

# install dependencies
pip install -r requirements.txt
```

# 🧩 **1) Evaluate LoRA (Trained) on Validation & Training Sets**

## 🔧 **Environment**

```bash
export EVAL_MODE=hf_local
export BASE_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
export LORA_ADAPTER_PATH=/Users/srirupin/Desktop/SWM/SWM_Evaluation/kg_lora_out_chat
export HF_DEVICE=mps   # or: cpu
```

## 🫧 **Sanity Check (quick smoke test)**

```bash
python -m scripts.smoke_infer "Say hi in one sentence."
```

## ✅ **Validation Set (Full metrics — ROUGE, BERT)**

```bash
python -m scripts.eval_med_jsonl \
  --data ./kg_dataset/val.jsonl \
  --out results/med_val_eval_full.jsonl \
  --use_input --add_rouge --add_bert
```

## ✅ **Training Set (Full metrics — ROUGE, BERT)**

```bash
python -m scripts.eval_med_jsonl \
  --data ./kg_dataset/train.jsonl \
  --out results/med_train_eval_full.jsonl \
  --use_input --add_rouge --add_bert
```

# 🧠 **2) Evaluate Base (Standard) Model on the Same Splits**

Disable LoRA by unsetting the adapter path, then rerun on the same data.

## 🔧 **Environment (Base only)**

```bash
unset LORA_ADAPTER_PATH   # IMPORTANT: disables LoRA
export BASE_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
export HF_DEVICE=mps      # or: cpu
```

## ✅ **Validation Set (Full metrics — ROUGE, BERT)**

```bash
python -m scripts.eval_med_jsonl \
  --data ./kg_dataset/val.jsonl \
  --out results/base_val_eval_full.jsonl \
  --use_input --add_rouge --add_bert
```

## ✅ **Training Set (Full metrics — ROUGE, BERT)**

```bash
python -m scripts.eval_med_jsonl \
  --data ./kg_dataset/train.jsonl \
  --out results/base_train_eval_full.jsonl \
  --use_input --add_rouge --add_bert
```

# 🔎 **Quick Result Check**

Peek at the last (most recent) record in each JSONL to confirm the job completed and metrics are present.

```bash
tail -n 1 results/med_val_eval_full.jsonl

tail -n 1 results/base_val_eval_full.jsonl
```

> Tip: Each line is a self‑contained JSON object (per‑example or aggregate, depending on your script). Keep these around for downstream analysis.

# 📊 **3) Judge‑Free A/B Testing — Per‑Example F1 + Bootstrap CI**

Quantify how much the LoRA adapter helps (or hurts) on the **same** examples.

## 🥇 **Per‑Example Winners by F1**

```bash
python -m scripts.ab_pref_f1_fast \
  --a results/base_val_eval_full.jsonl \
  --b results/med_val_eval_full.jsonl
```

## 📈 **Paired Bootstrap CI for F1 Lift (LoRA − Base)**

```bash
python -m scripts.paired_bootstrap_ci \
  --a_jsonl results/base_val_eval_full.jsonl \
  --b_jsonl results/med_val_eval_full.jsonl \
  --iters 5000 \
  --out results/f1_lift_val.jsonl
```

### 🔍 **Check the final lift**

```bash
tail -n 1 results/f1_lift_val.jsonl
```

> Interpretation: a **positive** F1 lift means LoRA outperforms the base model on average; a **negative** lift means the base model is better. The bootstrap CI tells you how stable that estimate is across resamples of the same validation set.

# 🧰 **Troubleshooting & Tips**

* **No MPS on this machine**: set `export HF_DEVICE=cpu`.
* **macOS MPS memory issues**: close GPU‑heavy apps; if you still see allocation errors, try smaller batch sizes in your scripts.
* **Missing metrics packages**: make sure `requirements.txt` includes any metric deps (e.g., ROUGE, BERTScore). Re‑run `pip install -r requirements.txt` if needed.
* **Results folder not found**: create it with `mkdir -p results` before running.

# ♻️ **Reproducibility Checklist**

* Record the exact values of these env vars in your run logs: `EVAL_MODE`, `BASE_MODEL_ID`, `LORA_ADAPTER_PATH` (for LoRA runs), `HF_DEVICE`.
* Keep the exact commit hash of this repo and the versions from `pip freeze`.
* Archive the produced `results/*.jsonl` files alongside the dataset snapshot used (`./kg_dataset/{train,val}.jsonl`).

# 📦 **What You Should Have After Completing the Guide**

* LoRA eval artifacts:

  * `results/med_val_eval_full.jsonl`
  * `results/med_train_eval_full.jsonl`
* Base eval artifacts:

  * `results/base_val_eval_full.jsonl`
  * `results/base_train_eval_full.jsonl`
* A/B testing artifact:

  * `results/f1_lift_val.jsonl` (paired bootstrap CI summary for F1 lift)

# ✅ **Done!**

You now have a clean, repeatable process for comparing **LoRA vs. Base** with **full metrics** and a **statistically principled** F1 lift estimate. Commit this README with your results so collaborators can reproduce and review your evaluation end‑to‑end.
