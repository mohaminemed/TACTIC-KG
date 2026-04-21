import json
import re
from pathlib import Path
from difflib import SequenceMatcher
import torch.nn.functional as F
import random
import math

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MistralCommonBackend, Mistral3ForConditionalGeneration, Qwen3VLForConditionalGeneration
from peft import PeftModel

import yaml
from pathlib import Path
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

args = parse_args()

# Load YAML
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Global config
BASE_MODEL = config["models"]["base_model"]
NUM_DOCS = config["experiment"]["num_docs"]
START = config["experiment"]["start"]
isLoRa = config["experiment"]["islora"]
TEST = config["experiment"]["test"]

# Agent-specific config 
LORA_DIR = config["agents"]["extractor"]["lora_dir"]
TEST_DATA_PATH = f"data/datasets/unseen_test_dataset_{TEST}.json"

if isLoRa : 
   OUTPUT_PATH = f"outputs/unseen/ft_{BASE_MODEL}/pred_triplets.json"
   SAVE_DIR = Path(f"outputs/unseen/ft_{BASE_MODEL}")
else :   
   OUTPUT_PATH = f"outputs/unseen/{BASE_MODEL}/pred_triplets.json"
   SAVE_DIR = Path(f"outputs/unseen/{BASE_MODEL}")

MAX_LENGTH = 4096  # 2048
MAX_NEW_TOKENS = 1024
CHUNK_SIZE = 2048  # 1024
OVERLAP = 50


# ---------- JSON extraction helper ----------
def extract_json_objects(text: str):
    """Extract all JSON objects inside brackets from text, even if incomplete arrays"""
    objs = []
    # matches {...} patterns
    for m in re.finditer(r"(\{.*?\})", text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            objs.append(obj)
        except json.JSONDecodeError:
            continue  # skip incomplete objects
    return objs

# ---------- Chunking helper ----------
def chunk_text(text, tokenizer, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks

# ---------- Load model ----------

if "Ministral" in BASE_MODEL:
    tokenizer = MistralCommonBackend.from_pretrained(BASE_MODEL)
    base_model = Mistral3ForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map= "auto")
elif "Qwen3-VL" in BASE_MODEL:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, trust_remote_code=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Thinking", dtype="auto", device_map="auto")      
else :
   tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
   base_model = AutoModelForCausalLM.from_pretrained(
      BASE_MODEL,
      trust_remote_code=True,
      device_map= {"": 0},  #"auto",
      torch_dtype=torch.bfloat16
    )

if isLoRa:
   print(f"Loading tokenizer and LoRA fine-tuned {BASE_MODEL} extractor...")
   model = PeftModel.from_pretrained(base_model, LORA_DIR)
   model = model.merge_and_unload()   # merge LoRA weights into base weights
else: 
   print(f"Loading tokenizer and base {BASE_MODEL} extractor...")
   model = base_model   

model.eval()
device = next(model.parameters()).device


# ---------- Load test dataset ----------
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    all_test_docs = json.load(f)
    print(f"✅ Loaded test dataset with {len(all_test_docs)} documents.")

CLASSES = [
    "Account",
    "Credential",
    "Tool",
    "Attacker",
    "Event",
    "Exploit Target",
    {
        "Indicator": [
            "File",
            "IP",
            "URL",
            "Domain",
            "Registry Key",
            "Hash",
            "Mutex",
            "User Agent",
            "Email",
            "Yara Rule",
            "SSL Certificate",
        ]
    },
    "Information",
    "Location",
    "Malware",
    {
        "Malware Characteristic": [
            "Behavior",
            "Capability",
            "Feature",
            "Payload",
            "Variants",
        ]
    },
    "Organization",
    "Infrastructure",
    "Time",
    "Vulnerability",
    "Unknown",
]

def build_prompt(text):
    return (f"""
           You are cybersecurity EXTRACTOR agent, your task is to extract key information from a given cyber threat intelligence (CTI) report in the form of subject-relation-object triplets.
           Output format (NO TYPES):
           [
           {{
            "subject": "...", 
            "relation": "...",
            "object": "...", 
           }}
           ]
           Rules:
            1. Extract ALL entities belonging to the following classes: {CLASSES}
            2. Do not type the entities, only extract subject, relation, and object.
            3. Your extraction must be COMPLETE and ACCURATE: 
                     - No hallucinations or external knowledge.
                     - No redundancy: each triplet is a fact and captures exactly one unique claim.
            5. Entity consistency and canonicalization (MANDATORY):
                     - Use a single, consistent surface form for the same real-world entity throughout the entire report.
                     - If multiple expressions refer to the same entity (e.g., “threat actor”, “threat actors”, “adversary”), normalize them to one canonical form and use it consistently in all triplets.
            4. Return valid JSON only (no explanations, no elaborations).

            ### TARGET report: (Process THIS only)
            CTI: {text}

            JSON:
            """)

def deduplicate_triplets(triplets):
    """
    Deduplicate triplets even if key names differ or some fields are missing.
    Robust to bool, None, and invalid types.
    """

    seen = set()
    deduped = []

    for t in triplets:

        subj = t.get("subject", t.get("subject_name", None))
        obj = t.get("object", t.get("object_name", None))
        rel = t.get("relation", None)
        score = t.get("conf_extraction", 0.0)

        # Reject invalid values
        if not isinstance(subj, str) or not isinstance(obj, str) or not isinstance(rel, str):
            continue

        subj = subj.strip()
        obj = obj.strip()
        rel = rel.strip()

        if subj == "" or obj == "" or rel == "":
            continue

        key = (subj.lower(), rel.lower(), obj.lower())

        if key not in seen:

            seen.add(key)

            deduped.append({
                "subject": subj,
                "relation": rel,
                "object": obj,
                "conf_extraction": float(score)
            })

    return deduped


def field_confidence(token_logprobs):
    """
    token_logprobs: list of log-probabilities for tokens in a field
    returns: confidence in [0,1]
    """
    if not token_logprobs:
        return 0.0

    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    return math.exp(avg_logprob)


def clean_token(tok):
    return tok.replace("Ġ", "").replace("▁", "").strip()


def align_tokens_to_triplet(triplet, token_logprobs):
    aligned = {"subject": [], "relation": [], "object": []}
    gen_tokens = [t["token"].replace("Ġ", "").replace("▁", "").strip() for t in token_logprobs]

    for field in ["subject", "relation", "object"]:

        value = triplet.get(field, "")

        # FIX: ensure string
        if not isinstance(value, str):
            value = ""

        text = value.strip().lower()
        if not text:
            continue
        target_tokens = text.split()
        L = len(target_tokens)

        found = False
        # Try exact match first
        for i in range(len(gen_tokens) - L + 1):
            window = " ".join(gen_tokens[i:i+L]).lower()
            if window == text:
                aligned[field] = [token_logprobs[i+j]["logprob"] for j in range(L)]
                found = True
                break
        if not found:
            # Fallback: partial match, count only matching tokens
            matched = []
            for i, tok in enumerate(gen_tokens):
                if tok.lower() in target_tokens:
                    matched.append(token_logprobs[i]["logprob"])
            aligned[field] = matched  # can be empty
    return aligned

def score_extraction_confidence(preds, token_logprobs):
    scored = []

    for t in preds:
        field_tokens = align_tokens_to_triplet(t, token_logprobs)

        conf_sub = field_confidence(field_tokens["subject"])
        conf_rel = field_confidence(field_tokens["relation"])
        conf_obj = field_confidence(field_tokens["object"])

        t = t.copy()
        t["conf_extraction"] = min(conf_sub, conf_rel, conf_obj)
        scored.append(t)

    return scored


def generate_triplets_ft(chunk, tokenizer=tokenizer, model=model, device=device):
    """Generate triplets using the fine-tuned LoRA extractor."""
   
    prompt = build_prompt(chunk)
    print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    # Generate
    
    gen = model.generate(
      **inputs,
      #max_new_tokens=MAX_NEW_TOKENS,
      do_sample=False,
      repetition_penalty=1.0,
      eos_token_id=tokenizer.eos_token_id,
      pad_token_id=tokenizer.pad_token_id,
      return_dict_in_generate=True,
      output_scores=True
    )

    input_len = inputs["input_ids"].shape[1]
    gen_tokens = gen.sequences[0][input_len:]
    
    token_logprobs = []

    for step, scores in enumerate(gen.scores):
      log_probs = F.log_softmax(scores[0], dim=-1)
      token_id = gen_tokens[step]
      token_logprobs.append({
        "token_id": token_id.item(),
        "token": tokenizer.convert_ids_to_tokens([token_id.item()])[0],
        "logprob": log_probs[token_id].item()
      })

      out_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    

    # Safe JSON extraction: handle partial outputs
    preds = extract_json_objects(out_text)
    if not preds:
        print("⚠️ Warning: No valid JSON extracted from chunk. Saving raw output for inspection.")

        raw_path = SAVE_DIR / "last_chunk_raw.txt"
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(out_text)
    return preds, token_logprobs

# ---------- Evaluation ----------
results = []
start_time = time.time()
for i, doc in enumerate(all_test_docs[START:NUM_DOCS]):  # first docs for testing
    text = doc.get("text","")
    gold_triplets = doc.get("triplets",[])
    source_id = doc.get("source_id", f"doc_{i}")
    if not text or not gold_triplets: continue

    chunks = chunk_text(text, tokenizer)
    print(f"\nProcessing document {i} with {len(chunks)} chunks...")
    print(f"Source ID: {source_id}")
    all_preds = []

    for chunk in chunks:
        preds, token_logprobs = generate_triplets_ft(chunk)
        # Only keep valid triplets
        required_keys = ["subject", "relation", "object"]
        valid_preds = [t for t in preds if all(k in t for k in required_keys)]
        if not valid_preds:
            print("⚠️ Warning: No valid triplets extracted from chunk. Skipping confidence scoring.")
            continue
        scored_preds = score_extraction_confidence(valid_preds, token_logprobs)
        all_preds.extend(scored_preds)
    all_preds = deduplicate_triplets(all_preds)
    print(f"Extracted {len(all_preds)} triplets for document {i}.")
    #print(f"Triplets: {all_preds}")
    results.append({
        "id": i,
        "source_id": source_id,
        "text": text,
        "gold_triplets": gold_triplets,
        "pred_triplets": all_preds
    })


end_time = time.time()
elapsed = end_time - start_time
print(f"\n⏱️ Extraction using fine-tuned model {BASE_MODEL} completed in {elapsed:.2f} seconds.")
# Save results
avg_time = elapsed / len(results)
final_output = {
    "model": BASE_MODEL,
    "total_time_sec": round(elapsed, 4),
    "avg_time_per_sample_sec": round(avg_time, 4),
    "num_samples": len(results),
    "results": results
}
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print(f"\n✅ Triplet extraction done. Saved to {OUTPUT_PATH}")