#!/usr/bin/env python3
"""
Fine-tune a small model with LoRA to extract CTI triplets.

Objective:
Given:
  - text
Predict:
  - extract all CTI triplets in the text and return them in a normalized JSON format.

"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, Mistral3ForConditionalGeneration, MistralCommonBackend, Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# ---------------- CONFIG ----------------

MODEL_NAME =  "mistralai/Ministral-3-8B-Reasoning-2512" 
# "mistralai/Ministral-3-3B-Instruct-2512-BF16" 
# "Qwen/Qwen3-8B"  
# "fdtn-ai/Foundation-Sec-8B-Reasoning"   
# "Qwen/Qwen3-VL-8B-Thinking" 
OUTPUT_DIR = "models/ministral-3-8B_extractor_lora_v2.0" 
# "models/ministral-3-3B_extractor_lora_v1.0"  
# "models/qwen3_8b_extractor_lora_v1.0" 
# "models/foundation-sec-8B_extractor_lora_v1.0"  
# "models/qwen3_vl_8b_thinking_extractor_lora_v1.0" 

DATA_PATH = "data/datasets/train_extraction_typing_dataset.json"
MAX_LENGTH = 2048
TRAIN_TEST_SPLIT = 0.95
Fraction = 1.0  # Use 100% of data (set to <1.0 for quick testing)
SEED = 42
BATCH_SIZE = 1
EPOCHS = 3
LR = 2e-4
WARMUP_STEPS = 100
SAVE_STRATEGY = "epoch"
GRAD_ACCUM_STEPS = 8

# LoRA settings
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]  # attention layers

# Prompt template
INSTRUCTION = (
    "As a security analyst, extract all CTI triplets from the report.\n"
    "Return ONLY a valid JSON array with the following format (no explanation):\n"
    """[
      {"subject": "<entity>", "subject_type": "<Entity Class>", 
       "relation": "<relation>", "object": "<entity>", "object_type": "<Entity Class>"}
    ]"""
)


# ---------------- Dataset ----------------
class TripletExtractionDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for ex in examples:
            prompt = ex["prompt"]
            target = ex["target"]
            full_text = prompt + target
            tokenized = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)

            # Mask prompt tokens
            prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"])
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100

            self.data.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: v.clone() for k, v in self.data[idx].items()}

# ---------------- Utilities ----------------
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_into_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]

def chunk_text_with_overlap(text: str, triplets: List[Dict], max_sents: int = 10, overlap: int = 3) -> List[Dict]:
    sents = split_into_sentences(text)
    chunks = []
    i = 0
    while i < len(sents):
        end = min(len(sents), i + max_sents)
        chunk_text = " ".join(sents[i:end])
        chunk_triplets = [t for t in triplets if t.get("subject","").lower() in chunk_text.lower() 
                                                  or t.get("object","").lower() in chunk_text.lower()]
        if chunk_triplets:
            chunks.append({"text": chunk_text, "triplets": chunk_triplets})
        i = end - overlap if end - overlap > i else end
    return chunks

def build_example(text: str, triplets: List[Dict]) -> Dict:
    prompt = INSTRUCTION + "\nReport:\n" + text.strip() + "\n\nJSON:\n"
    target = json.dumps(triplets, ensure_ascii=False, separators=(",", ":"))
    return {"prompt": prompt, "target": target}

def load_examples(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    examples = []
    for doc in docs:
        text = doc.get("text", "")
        triplets = doc.get("triplets", [])
        if not text or not triplets or len(triplets) >= 30:
            continue
        chunks = chunk_text_with_overlap(text, triplets)
        for chunk in chunks:
            examples.append(build_example(chunk["text"], chunk["triplets"]))
    return examples

# ---------------- JSON Extraction ----------------
JSON_ARRAY_RE = re.compile(r"(\[\s*\{.*?\}\s*\])", re.DOTALL)

def extract_first_json_array(text: str):
    m = JSON_ARRAY_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            cleaned = m.group(1).replace("'", '"')
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                return json.loads(cleaned)
            except Exception:
                return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return None
    return None

def normalize_triplets_list(triplets_list):
    normalized = []
    for t in triplets_list:
        if not isinstance(t, dict):
            continue
        normalized.append((t.get("subject","").strip().lower(),
                           t.get("relation","").strip().lower(),
                           t.get("object","").strip().lower()))
    return set(normalized)

# ---------------- Main ----------------
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading tokenizer & model...")
    if "Ministral" in MODEL_NAME:
        tokenizer = MistralCommonBackend.from_pretrained(MODEL_NAME)
        base_model = Mistral3ForConditionalGeneration.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
    elif "Qwen3-VL" in MODEL_NAME:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
         "Qwen/Qwen3-VL-8B-Thinking",
         dtype="auto",
         device_map="auto"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
          MODEL_NAME,
          trust_remote_code=True,
          device_map="auto",
          torch_dtype=torch.bfloat16
        )
   
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.padding_side = "left"

    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    print("LoRA attached to model.")

    print("Model device:", next(model.parameters()).device)

    examples = load_examples(DATA_PATH)
    print(f"Loaded {len(examples)} examples")
    random.shuffle(examples)
    cutoff = int(len(examples) * TRAIN_TEST_SPLIT * Fraction)
    train_ex = examples[:cutoff]
    val_ex = examples[cutoff:]
    print(f"Train: {len(train_ex)}, Val: {len(val_ex)}")

    train_ds = TripletExtractionDataset(train_ex, tokenizer)
    val_ds = TripletExtractionDataset(val_ex, tokenizer)

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch])
        }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True,
        eval_strategy="epoch",
        save_strategy=SAVE_STRATEGY,
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        optim="paged_adamw_32bit"
    )

    print("Model device:", next(model.parameters()).device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete. Model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
