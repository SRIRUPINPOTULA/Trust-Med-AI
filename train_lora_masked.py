#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lora_masked.py
- Prompt/answer label masking (no loss on prompt tokens)
- EOS appended to answers
- Optional Qwen chat-template prompting for alignment with app.py
- MPS-friendly defaults (batch=1 + grad_accum)
"""
import argparse, os, torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="kg_dataset/train.jsonl")
    ap.add_argument("--val_path",   default="kg_dataset/val.jsonl")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out_dir",    default="kg_lora_out")
    ap.add_argument("--epochs",     type=int,   default=1)
    ap.add_argument("--lr",         type=float, default=2e-4)
    ap.add_argument("--train_bs",   type=int,   default=1)
    ap.add_argument("--eval_bs",    type=int,   default=1)
    ap.add_argument("--grad_accum", type=int,   default=16)
    ap.add_argument("--cutoff_len", type=int,   default=512)
    ap.add_argument("--use_chat_template", action="store_true",
                    help="Wrap (instruction,input) using Qwen's chat template for better alignment with inference.")
    return ap.parse_args()


def main():
    args = parse_args()

    # Prefer MPS on Apple Silicon
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps") if use_mps else torch.device("cpu")

    # Load JSONL datasets (columns: instruction, input, output)
    ds_train = load_dataset("json", data_files=args.train_path, split="train")
    ds_val   = load_dataset("json", data_files=args.val_path,   split="train")

    # Tokenizer / Model
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model.config.use_cache = False  # important for training
    model.to(device)

    # Attach LoRA adapters
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # ------- Prompt builders (two modes) -------

    def build_prompt_and_answer_plain(ex):
        instr = (ex["instruction"] or "").strip()
        inp   = (ex.get("input") or "").strip()
        out   = (ex["output"] or "").strip()
        if inp:
            prompt = f"Instruction: {instr}\nInput: {inp}\nAnswer:"
        else:
            prompt = f"Instruction: {instr}\nAnswer:"
        return prompt, out

    def build_prompt_and_answer_chat(ex):
        """Use Qwen chat template so training matches inference."""
        instr = (ex["instruction"] or "").strip()
        inp   = (ex.get("input") or "").strip()
        user  = instr if not inp else f"{instr}\n\n{inp}"
        msgs = [
            {
                "role": "system",
                # Keep system brief; your app also supplies a system message at inference
                "content": (
                    "You are TrustMed. Answer carefully and use the exact 3-section format: "
                    "[Answer] ... [Facts Used] ... [Disclaimer] ..."
                )
            },
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out    = (ex["output"] or "").strip()
        return prompt, out

    build_prompt_and_answer = build_prompt_and_answer_chat if args.use_chat_template else build_prompt_and_answer_plain

    # ------- Tokenize with prompt masking -------
    # We keep batch_size=1 so no special collator is needed (labels have variable lengths).

    def tokenize_with_labels(batch):
        input_ids, attention_mask, labels = [], [], []
        for i in range(len(batch["instruction"])):
            ex = {k: batch[k][i] for k in batch.keys()}
            prompt, answer = build_prompt_and_answer(ex)

            # Append EOS to answer
            if not answer.endswith(tok.eos_token):
                answer = answer + tok.eos_token

            enc_p = tok(
                prompt,
                padding=False, truncation=True,
                max_length=args.cutoff_len,
                return_attention_mask=False,
                add_special_tokens=False,  # chat template already inserts specials
            )
            enc_a = tok(
                answer,
                padding=False, truncation=True,
                max_length=args.cutoff_len,
                return_attention_mask=False,
                add_special_tokens=False,
            )

            p_ids = enc_p["input_ids"]
            a_ids = enc_a["input_ids"]

            ids  = p_ids + a_ids
            attn = [1] * len(ids)
            lbls = [-100] * len(p_ids) + a_ids[:]  # mask prompt, learn only on answer

            # Truncate consistently
            if len(ids) > args.cutoff_len:
                ids  = ids[:args.cutoff_len]
                attn = attn[:args.cutoff_len]
                lbls = lbls[:args.cutoff_len]

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lbls)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    keep_cols = ds_train.column_names  # ['instruction','input','output',...]
    ds_train_tok = ds_train.map(tokenize_with_labels, batched=True, remove_columns=keep_cols)
    ds_val_tok   = ds_val.map(tokenize_with_labels,   batched=True, remove_columns=keep_cols)

    # ------- Training setup -------
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        gradient_checkpointing=False,  # safer on MPS
        fp16=False,  # ignored on MPS; keep False
        bf16=False,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train_tok,
        eval_dataset=ds_val_tok,
        tokenizer=tok,  # OK even if deprecated warning; fine for now
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"Saved LoRA adapter + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
