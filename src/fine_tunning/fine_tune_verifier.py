#!/usr/bin/env python3
"""
Verifier fine-tuning script (per triplet verification).

Objective:
Given:
  - text
  - extracted triplets 
Predict:
  - for each triplet, a label: SUPPORTED or NOT_SUPPORTED based on the evidence in the text.
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
    AutoProcessor
)

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# -------------------- Config --------------------
MODEL_NAME = "mistralai/Ministral-3-8B-Reasoning-2512"
# "mistralai/Ministral-3-3B-Instruct-2512-BF16" 
# "Qwen/Qwen3-8B"  
# "fdtn-ai/Foundation-Sec-8B-Reasoning"   
# "Qwen/Qwen3-VL-8B-Thinking" 
OUTPUT_DIR = "models/ministral-3-8B-Reasoning-2512_verifier_lora_v0.1" 
# "models/qwen3_vl_8b_verifier_lora_v0.1" 
# "models/foundation-sec-8B_verifier_lora_v0.1" 
# "models/ministral-3-3B_verifier_lora_v0.1"  
# "models/qwen3_vl_8b_thinking_verifier_lora_v1.0"  

DATA_PATH = "data/datasets/train_verification_dataset.json"
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
VIRIFIER_INSTRUCTIONS = (
    "You are a Cyber Threat Intelligence verifier.\n"
    "Your task is to verify extracted triplets using evidence from the text.\n\n"
    "Rules:\n"
    "You must classify each triplet into ONE of the following labels:\n\n"

    "SUPPORTED:\n"
    "- The TEXT clearly states the relation between subject and object.\n\n"

    "NOT_SUPPORTED:\n"
    "- The TEXT does NOT explicitly state the relation.\n"
    "- The relation may be plausible but is not present.\n"
    )

def build_verifier_prompt(text, trips):
    return (
        f"TEXT:\n{text}\n\n"
        f"EXTRACTED_TRIPLETS:\n{json.dumps(trips, ensure_ascii=False)}\n\n"
        f"{VIRIFIER_INSTRUCTIONS}\n"
        f"REPLY (JSON):\n"
    )

import json
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------
# Dataset for Triplet Verification Fine-tuning
# ---------------------------------------------------------
class VerifierDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for r in records:

            text = r.get("text", "")
            verification = r.get("verified_triplets", [])

            if not text or not verification:
                continue

            # Train ONE triplet at a time (much cleaner supervision)
            for item in verification:

                triplet = {
                    "subject": item["subject"],
                    "relation": item["relation"],
                    "object": item["object"]
                }

                label = item["label"]

                # Build prompt
                prompt = build_verifier_prompt(
                    text=text,
                    trips=[triplet]   # single triplet
                )

                # Expected output format
                target_json = json.dumps([
                    {
                        "subject": item["subject"],
                        "relation": item["relation"],
                        "object": item["object"],
                        "label": label
                    }
                ], ensure_ascii=False)

                full_text = prompt + target_json

                enc = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.max_length,
                )

                labels = enc["input_ids"].copy()

                # -----------------------------
                # Mask prompt tokens
                # -----------------------------
                prompt_len = len(
                    tokenizer(
                        prompt,
                        truncation=True,
                        max_length=self.max_length
                    )["input_ids"]
                )

                labels[:prompt_len] = [-100] * prompt_len
                enc["labels"] = labels

                self.samples.append(enc)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            k: torch.tensor(v)
            for k, v in self.samples[idx].items()
        }
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

    train_ds = VerifierDataset(train_records, tokenizer, MAX_LENGTH)
    eval_ds = VerifierDataset(eval_records, tokenizer, MAX_LENGTH) if eval_records else None

    print("Train dataset size:", len(train_ds))

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

    print(f"✅ Verifier model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
