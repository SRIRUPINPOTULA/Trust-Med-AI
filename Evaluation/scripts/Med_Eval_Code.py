!pip install -q evaluate bert-score rouge_score sacrebleu pandas numpy nltk

import json, pandas as pd, evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer, util
import torch


def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

train_df = load_jsonl("/content/med_train_eval.jsonl")
val_df   = load_jsonl("/content/med_val_eval.jsonl")
print(f"Loaded {len(train_df)} training and {len(val_df)} validation samples")

print("\n🧾 Columns detected in validation file:")
print(val_df.columns.tolist())


references = val_df["gold"].fillna("").tolist()   # true answers
predictions = val_df["pred"].fillna("").tolist()  # model outputs


# ROUGE
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=predictions, references=references)

# BLEU
smooth = SmoothingFunction().method4
bleu_scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
               for ref, pred in zip(references, predictions)]
bleu_score = 100 * sum(bleu_scores) / len(bleu_scores)

# BERTScore
bertscore = evaluate.load("bertscore")
bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
bert_f1 = 100 * sum(bertscore_result["f1"]) / len(bertscore_result["f1"])

# METEOR
meteor_values = [meteor_score([ref.split()], pred.split()) for ref, pred in zip(references, predictions)]
meteor_avg = 100 * sum(meteor_values) / len(meteor_values)

# CHRF (character-level)
chrf = evaluate.load("chrf")
chrf_result = chrf.compute(predictions=predictions, references=references)
chrf_score = chrf_result["score"]

# Exact Match
exact_match = 100 * sum([1 for r, p in zip(references, predictions) if r.strip() == p.strip()]) / len(references)

# Cosine Similarity (semantic)
model_sim = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
ref_emb = model_sim.encode(references, convert_to_tensor=True, show_progress_bar=False)
pred_emb = model_sim.encode(predictions, convert_to_tensor=True, show_progress_bar=False)
cosine_scores = util.cos_sim(ref_emb, pred_emb)
cosine_mean = 100 * torch.mean(torch.diag(cosine_scores)).item()


print("Text Generation Evaluation Results (Gold vs Pred):")

print(f"ROUGE-1         : {rouge_result['rouge1']*100:.2f}%")
print(f"ROUGE-L         : {rouge_result['rougeL']*100:.2f}%")
print(f"BLEU            : {bleu_score:.2f}%")
print(f"BERTScore (F1)  : {bert_f1:.2f}%")
print(f"METEOR          : {meteor_avg:.2f}%")
print(f"CHRF            : {chrf_score:.2f}%")
print(f"Exact Match     : {exact_match:.2f}%")
print(f"Cosine Similar. : {cosine_mean:.2f}%")

print(f"Queries evaluated: {len(predictions)}")
