import json
import re
from pathlib import Path
from difflib import SequenceMatcher
import random

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
isLoRa = config["experiment"]["islora"]

# Agent-specific config 
LORA_DIR = config["agents"]["typer"]["lora_dir"]

if isLoRa:
  EXTRACTOR_DATA_PATH = f"outputs/unseen/ft_{BASE_MODEL}/pred_triplets.json"
  OUTPUT_PATH = f"outputs/unseen/ft_{BASE_MODEL}/typed_triplets.json"
  SAVE_DIR = Path(f"outputs/unseen/ft_{BASE_MODEL}/")
else : 
  EXTRACTOR_DATA_PATH = f"outputs/unseen/{BASE_MODEL}/pred_triplets.json"
  OUTPUT_PATH = f"outputs/unseen/{BASE_MODEL}/typed_triplets.json"
  SAVE_DIR = Path(f"outputs/unseen/{BASE_MODEL}/")    

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 2048
CHUNK_SIZE = 2048
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
    base_model = Mistral3ForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
elif "Qwen3-VL" in BASE_MODEL:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, trust_remote_code=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Thinking", dtype="auto", device_map="auto")
else : 
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    device_map={"": 0},  # "auto",
    torch_dtype=torch.bfloat16
)
if isLoRa :
    print(f"Loading tokenizer and LoRA {BASE_MODEL} fine-tuned typer...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model = model.merge_and_unload()   # merge LoRA weights into base weights
else:
    print(f"Loading tokenizer and base {BASE_MODEL} typer...")
    model = base_model

model.eval()
device = next(model.parameters()).device
print(f"Model loaded on device: {device}")

# ---------- Load test dataset ----------
with open(EXTRACTOR_DATA_PATH, "r", encoding="utf-8") as f:
    all_test_docs = json.load(f)
    all_test_docs = all_test_docs["results"]


def select_short_demo(docs, tokenizer, max_tokens=1024, k=3):
    """
    Select up to k demos from docs whose tokenized text length <= max_tokens.
    If not enough candidates, fall back to the k shortest docs.
    """
    # Filter candidates that fit within max_tokens
    candidates = [d for d in docs if len(tokenizer.encode(d.get("text", ""))) <= max_tokens]
    
    if len(candidates) < k:
        # Fallback: pick the k shortest available
        candidates = sorted(docs, key=lambda d: len(tokenizer.encode(d.get("text", ""))))[:k]
    
    # Randomly sample k (or all if fewer)
    return random.sample(candidates, min(k, len(candidates)))

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
    "This entity cannot be classified into any of the existing types",
]

def build_prompt(text, extracted_triplets=""):
    return (f"""
                You are a Typer, a cyber threat intelligence graph reasoning agent. You are given a CTI report and the corresponding set of extracted triplets.
                Your task is to read carefully the text and type all entities (subject, object) from the set of triplets using the provided nested classes.
                Output format:
                [
                 {{
                    "subject": "<entity>",
                    "subject_type": "<Entity Class>",
                    "relation": "<relation>",
                    "object": "<entity>",
                    "object_type": "<Entity Class>",
                 }},
                ...
                ]
                Rules:
                   1. Each entity type must accurately reflect the context provided in the text.
                   2. All subjects and objects in the input triplets must be typed.
                   3. Use only the nested entity classes provided: {CLASSES}.
                   4. Be consistent: each entity (subject or object) must have a unique type across all triplets.
                   5. Return valid JSON **only**. Do not include explanations or extra text.

                ### TARGET REPORT: (Process THIS)
                CTI: {text}
                ### EXTRACTED TRIPLETS:
                {extracted_triplets}

                JSON:
                """)

def deduplicate_triplets(triplets):
    """
    Deduplicate triplets even if key names differ or some fields are missing.
    """
    seen = set()
    deduped = []

    for t in triplets:
        subj = t.get("subject") or t.get("subject_name") or ""
        subj_type = t.get("subject_type") or ""
        subj_score = t.get("subject_type_confidence") or 0.00
        obj = t.get("object") or t.get("object_name") or ""
        obj_type = t.get("object_type") or ""
        obj_score = t.get("object_type_confidence") or 0.00
        rel = t.get("relation") or ""
        if not subj or not obj or not rel:
            continue  # skip incomplete triplets

        key = (subj.strip().lower(), rel.strip().lower(), obj.strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append({"subject": subj, "subject_type": subj_type, "subject_type_confidence": subj_score, "relation": rel, "object": obj, "object_type": obj_type, "object_type_confidence": obj_score,})
    return deduped

def align_tokens_to_typed_triplet(triplet, token_logprobs):
    """
    Align subject_type and object_type to generated tokens to compute confidence
    """
    aligned = {"subject_type": [], "object_type": []}

    for field in ["subject_type", "object_type"]:
        text = triplet.get(field, "").strip().lower()
        if not text:
            continue

        target_tokens = text.split()
        L = len(target_tokens)
        found = False

        # Exact match first
        gen_tokens = [t["token"].lower() for t in token_logprobs]
        for i in range(len(gen_tokens) - L + 1):
            window = " ".join(gen_tokens[i:i+L])
            if window == " ".join(target_tokens):
                aligned[field] = [token_logprobs[i + j]["logprob"] for j in range(L)]
                found = True
                break

        if not found:
            # Partial match fallback
            aligned[field] = [t["logprob"] for t in token_logprobs if t["token"].lower() in target_tokens]

    return aligned

def field_confidence(logprobs):
    if not logprobs:
        return 0.0
    # geometric mean is more forgiving than min
    prod = 1.0
    for p in logprobs:
        prod *= max(p, 1e-8)  # avoid zero
    return prod ** (1/len(logprobs))

def generate_typed_triplets_ft(chunk, extracted_triplets="", tokenizer=tokenizer, model=model, device=device):
    """Generate triplets using the fine-tuned LoRA extractor."""

    prompt = build_prompt(chunk, extracted_triplets=extracted_triplets)
    print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
    #print(f"Prompt: {prompt}")
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
        output_scores=True,
    )

    gen_tokens = gen.sequences[0][inputs["input_ids"].shape[1]:]  # skip prompt
    out_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Tokenize prompt + generated text together  
    full_text = prompt + out_text
    inputs_full = tokenizer(full_text, return_tensors="pt").to(device)

    with torch.no_grad():
       outputs = model(**inputs_full)
       logits = outputs.logits  # [1, seq_len, vocab_size]

    # Slice logits for the generated part
    gen_logits = logits[:, inputs["input_ids"].shape[1]-1:-1, :]  # [1, gen_len, vocab_size]
    gen_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][:, inputs["input_ids"].shape[1]:]

    # Compute logprobs
    token_logprobs = []
    for i, token_id in enumerate(gen_ids[0]):
       token_id_int = int(token_id)
       logprob = torch.log_softmax(gen_logits[0, i], dim=-1)[token_id_int]
       token_logprobs.append({
        "token": tokenizer.convert_ids_to_tokens(token_id_int),
        "logprob": float(logprob)
       })
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
for i, doc in enumerate(all_test_docs):  
    text = doc.get("text","")
    extracted_triplets = doc.get("pred_triplets", [])
    gold_triplets = doc.get("gold_triplets", [])
    source_id = doc.get("source_id", f"doc_{i}")
    if not text or not gold_triplets:
         continue

    print(f"\nProcessing document {i} ...")
    print(f"Source ID: {source_id}")
    #all_preds = []
    typed_preds, token_logprobs = generate_typed_triplets_ft(text, extracted_triplets)
    
    # align and score
    for t in typed_preds:
      aligned = align_tokens_to_typed_triplet(t, token_logprobs)
      t["conf_subject_type"] = field_confidence(aligned.get("subject_type", []))
      t["conf_object_type"] = field_confidence(aligned.get("object_type", []))

    #if preds:
        #all_preds.extend(preds)
    all_preds = deduplicate_triplets(typed_preds)
    #print(f"Typed {len(typed_preds)} triplets for document {i}.")
    results.append({
        "id": i,
        "source_id": source_id,
        "text": text,
        "gold_triplets": gold_triplets,
        "typed_triplets": all_preds
    })
    print(f" Gold: {len(doc['gold_triplets'])} →  Predicted: {len(extracted_triplets)} →  Typed: {len(all_preds)}") 


end_time = time.time()
elapsed = end_time - start_time
print(f"\n⏱️ Typing using fine-tuned model {BASE_MODEL} completed in {elapsed:.2f} seconds.")
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

print(f"\n✅ Triplet typing done. Saved to {OUTPUT_PATH}")