# Trust-Med-AI

🧠 Medical Knowledge Graph Pipeline

This repository provides a complete pipeline for building and using a medical Knowledge Graph (KG) —
from cleaning and analyzing triples, to generating fine-tuning datasets, training LoRA adapters, and deploying an interactive chatbot using Gradio.

🚀 Overview

The workflow includes the following stages:

Analyze and Clean KG Triples → analyze_kg_triples.py

Generate Instruction Dataset → make_dataset_from_triples.py

Fine-tune Model (LoRA) → train_lora_masked.py

Launch Chatbot App → app.py

Each script is modular and can be used independently, or together as a unified pipeline.

⚙️ Installation
1️⃣ Clone the repository
git clone <your_repo_url>
cd <your_repo_name>

2️⃣ Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt


If no requirements.txt is available, install the main dependencies manually:

pip install pandas matplotlib networkx scikit-learn gradio torch transformers peft

🔄 Pipeline Summary
Stage	Script	Description	Input	Output	Command
1️⃣ KG Analysis	analyze_kg_triples.py	Cleans and normalizes triples, generates CSVs, plots, and Neo4j import files.	medical_kg_triples.json	triples_clean.csv, relation_counts.csv, node_degrees.csv, relation_hist.png, (optional) Neo4j CSVs	bash\npython analyze_kg_triples.py --input medical_kg_triples.json --outdir ./kg_out --neo4j\n
2️⃣ Dataset Creation	make_dataset_from_triples.py	Converts triples into instruction–response pairs for model training.	kg_out/triples_clean.csv	dataset/train.json, dataset/test.json	bash\npython make_dataset_from_triples.py --input ./kg_out/triples_clean.csv --output ./dataset/\n
3️⃣ LoRA Fine-tuning	train_lora_masked.py	Fine-tunes a base language model (e.g. Qwen) on generated dataset using LoRA adapters.	dataset/	lora_adapter/	bash\npython train_lora_masked.py --model Qwen/Qwen2.5-1.5B-Instruct --data ./dataset/ --output ./lora_adapter/\n
4️⃣ Chatbot Interface	app.py	Launches Gradio-based conversational AI that uses KG and fine-tuned model to answer queries.	kg_out/triples_clean.csv + model	Gradio Web UI	bash\npython app.py\n
🧭 Data Flow Diagram
graph TD
A[medical_kg_triples.json] --> B[analyze_kg_triples.py<br>→ Clean & Visualize]
B --> C[make_dataset_from_triples.py<br>→ Create Instruction Dataset]
C --> D[train_lora_masked.py<br>→ Fine-tune LoRA Model]
D --> E[app.py<br>→ Interactive Chatbot]

⚡ Example Workflow
# Step 1: Analyze and clean the KG triples
python analyze_kg_triples.py --input medical_kg_triples.json --outdir ./kg_out --neo4j

# Step 2: Create an instruction dataset
python make_dataset_from_triples.py --input ./kg_out/triples_clean.csv --output ./dataset/

# Step 3: Train LoRA adapter
python train_lora_masked.py --model Qwen/Qwen2.5-1.5B-Instruct --data ./dataset/ --output ./lora_adapter/

# Step 4: Launch the Gradio chatbot
python app.py


Or with environment variables:

export CONTEXT_CSV=kg_out/triples_clean.csv
export BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export ADAPTER_DIR="./lora_adapter"
python app.py

🧩 Environment Variables (for app.py)
Variable	Description	Default
CONTEXT_CSV	Path to cleaned triples CSV	kg_out/triples_clean.csv
BASE_MODEL	Base Hugging Face model name	Qwen/Qwen2.5-1.5B-Instruct
ADAPTER_DIR	Path to LoRA adapter directory	(empty)
SKIP_LORA	Skip LoRA adapter loading (1/true/yes)	0
FORCE_DEVICE	Force device (cpu, cuda, mps)	Auto-detected
📊 Output Summary
File	Description
triples_clean.csv	Cleaned and deduplicated triples
relation_counts.csv	Relation frequency summary
node_degrees.csv	Node degree statistics
relation_hist.png	Visualization of top relations
dataset/*.json	Instruction–response pairs
lora_adapter/	Fine-tuned LoRA model weights
💬 Example Interaction

User:

What are the treatments for diabetes?

Model Response:

Metformin and insulin are commonly used to treat diabetes.

Diet and exercise management are essential components of therapy.

Some patients may require oral hypoglycemic agents.

⚠️ This information is for educational purposes only and not a substitute for professional medical advice.

🩺 Disclaimer

⚠️ This tool is designed for research and educational purposes only.
It is not a substitute for professional medical advice.
Always consult a qualified healthcare provider for diagnosis or treatment.

🧾 Summary Comparison
Criteria	analyze_kg_triples.py	make_dataset_from_triples.py	train_lora_masked.py	app.py
Role	Clean & analyze data	Generate dataset	Fine-tune model	Deploy chatbot
Input Type	JSON/JSONL	CSV	Dataset	CSV + Model
Output Type	CSV/PNG	JSON	Model weights	Web interface
Core Libraries	pandas, matplotlib, networkx	pandas, sklearn	torch, peft	gradio, transformers
Execution Type	CLI	CLI	CLI	Web UI
