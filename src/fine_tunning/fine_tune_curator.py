#!/usr/bin/env python3
"""
Curator fine-tuning script (per triplet verification).

Objective:
Given:
  - text
  - verified triplets 
Predict:
  - normalize entity names so that different mentions referring to the same entity are unified into a single canonical form.
  - minimal set of missing triplets that can be inferred from the text and are not already in the verified set.
"""

import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend, 
    Qwen3VLForConditionalGeneration,
)

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# -------------------- Config --------------------
MODEL_NAME = "mistralai/Ministral-3-8B-Reasoning-2512" 
# "mistralai/Ministral-3-3B-Instruct-2512-BF16" 
# "Qwen/Qwen3-8B"  
# "fdtn-ai/Foundation-Sec-8B-Reasoning"   
# "Qwen/Qwen3-VL-8B-Thinking"
OUTPUT_DIR = "models/ministral-3-8B-Reasoning-2512_curator_lora_v0.2" 
# "models/ministral-3-3B_curator_lora_v0.1" 
# "models/qwen3_8b_curator_lora_v0.1"
# "models/foundation-sec-8B_curator_lora_v0.1"
# "models/qwen3_vl_8b_curator_lora_v0.1"

DATA_PATH = "data/datasets/train_curation_dataset.json"
MAX_LENGTH = 2048
TRAIN_TEST_SPLIT = 0.95
SEED = 42

BATCH_SIZE = 1
EPOCHS = 3
LR = 2e-5
GRAD_ACCUM_STEPS = 8

SAVE_STRATEGY = "epoch"

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

# -------------------- Utils --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------- Prompt --------------------
CURATOR_INSTRUCTIONS = (
    "You are a Cyber Threat Intelligence curator.\n"
    "Your task is to identify missing triplets using ONLY evidence from the TEXT.\n"
    "Rules:\n"
    " - Normalize entity names so that different mentions referring to the same entity are unified into a single canonical form. \n"
    " - Only propose triplets supported by the TEXT.\n"
    " - Do NOT invent relations or entities not mentioned in the TEXT.\n"
    " - Output valid JSON only (a list of triplets)\n"
)

def build_curator_prompt(text, main_graph):
    return (
        f"TEXT:\n{text}\n\n"
        f"MAIN_GRAPH_TRIPLETS:\n{json.dumps(main_graph, ensure_ascii=False)}\n\n"
        f"{CURATOR_INSTRUCTIONS}\n"
        f"REPLY (JSON):\n"
    )

# -------------------- Dataset --------------------
class CuratorDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for r in records:
            explicit = r.get("explicit_triplets", r.get("explicit", []))
            implicit = r.get("implicit_triplets", [])

            target = implicit if implicit else []

            if len(target) == 0:
                   continue

            prompt = build_curator_prompt(
                    text=r.get("text", ""),
                    main_graph=explicit,
            )

            target_json = json.dumps(target, ensure_ascii=False)

            full = prompt + target_json
            enc = tokenizer(
                    full,
                    truncation=True,
                    max_length=self.max_length,
                )

            labels = enc["input_ids"].copy()

            # Mask loss on prompt (only learn output)
            prompt_len = len(
                    tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"]
                )
            labels[:prompt_len] = [-100] * prompt_len

            enc["labels"] = labels
            self.samples.append(enc)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.samples[idx].items()}

# -------------------- Training --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    set_seed(SEED)

    records = load_json(args.data_path)
    random.shuffle(records)
    split = int(TRAIN_TEST_SPLIT * len(records))
    train_records = records[:split]
    eval_records = records[split:]
    print("Loading tokenizer & base model...")
    if "Ministral" in MODEL_NAME:
        tokenizer = MistralCommonBackend.from_pretrained(MODEL_NAME)
        base_model = Mistral3ForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
    elif "Qwen3-VL" in MODEL_NAME:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
        base_model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Thinking", dtype="auto", device_map="auto")       
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        #load_in_4bit=False
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.padding_side = "left"
    base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = CuratorDataset(train_records, tokenizer, MAX_LENGTH)
    eval_ds = CuratorDataset(eval_records, tokenizer, MAX_LENGTH) if eval_records else None

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args_train = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True if torch.cuda.is_available() else False,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy=SAVE_STRATEGY,
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Curator model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
