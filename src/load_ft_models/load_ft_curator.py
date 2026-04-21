
import json
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MistralCommonBackend, Mistral3ForConditionalGeneration, Qwen3VLForConditionalGeneration
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import numpy as np

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
HYBRID_MODEL = config["models"].get("hybrid_model", None)
NUM_DOCS = config["experiment"]["num_docs"]
isLoRa = config["experiment"]["islora"]
HYBRID = config["experiment"]["hybrid"]
EMBEDDING_MERGE_MODEL = config["models"]["embedding_merge_model"]
NODE_SIM_THRESHOLD = config["experiment"]["entity_merge_threshold"]
REL_SIM_THRESHOLD = config["experiment"]["relation_merge_threshold"]

# Agent-specific config 
LORA_DIR = config["agents"]["curator"]["lora_dir"]
LORA_HYBRID_DIR = config["agents"]["verifier"]["lora_hybrid_dir"]

if isLoRa:
   TYPER_OUTPUT_PATH = f"outputs/unseen/ft_{BASE_MODEL}/verified_triplets.json"
   OUTPUT_PATH = f"outputs/unseen/ft_{BASE_MODEL}/final_triplets.json"
else:
   TYPER_OUTPUT_PATH = f"outputs/unseen/{BASE_MODEL}/verified_triplets.json"
   OUTPUT_PATH = f"outputs/unseen/{BASE_MODEL}/final_triplets.json"

SAVE_DIR = Path(OUTPUT_PATH).parent

MAX_LENGTH = 8128
MAX_NEW_TOKENS = 4096


# ------------------------- HELPERS ------------------------
def safe_parse_json(s):
    """Try to parse JSON safely."""
    if not s: return []
    s = s.strip()
    try:
        start = s.index("[")
        end = s.rindex("]") + 1
        return json.loads(s[start:end])
    except (ValueError, json.JSONDecodeError):
        return []

def normalize_text(s: str) -> str:
    if not s: return ""
    return " ".join(s.lower().strip().split())

def seq_sim(a: str, b: str) -> float:
    if not a or not b: return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def cosine(a, b):
    return float(np.dot(a, b))

def embed(name):
    # Encode returns a vector already normalized if normalize_embeddings=True
    emb = merge_model.encode(name.lower().strip(), normalize_embeddings=True)
    return np.array(emb)

def deduplicate_triplets(triplets):
    seen = set()
    deduped = []
    for t in triplets:
        if not isinstance(t, dict):
            continue
        subj = t.get("subject") or ""
        obj = t.get("object") or ""
        rel = t.get("relation") or ""
        stype = t.get("subject_type") or ""
        otype = t.get("object_type") or ""
        if not subj or not obj or not rel:
            continue
        key = (subj.strip().lower(), rel.strip().lower(), obj.strip().lower(),
               stype.lower(), otype.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    return deduped

def count_support(triplets):
    c = Counter()
    for t in triplets:
        key = (t["subject"], t["relation"], t["object"])
        c[key] += 1
    return c

def low_confidence_triplets(triplets, type_thresh=0.8, support_thresh=2):
    """Return triplets with low typing confidence or low support."""
    low_conf = []
    support = count_support(triplets)
    for t in triplets:
        st_conf = t.get("subject_type_confidence", 1.0)
        ot_conf = t.get("object_type_confidence", 1.0)
        sup = t.get("support_count", support.get((t["subject"], t["relation"], t["object"]), 1))
        if st_conf < type_thresh or ot_conf < type_thresh or sup < support_thresh:
            low_conf.append(t)
    return low_conf

# -------------------- CANONICALIZATION -------------------
def collect_all_entity_names(triplets):
    return {normalize_text(t.get("subject","")) for t in triplets} | {normalize_text(t.get("object","")) for t in triplets}

def build_similarity_clusters(names, threshold=NODE_SIM_THRESHOLD):
    names = sorted(list(names))
    canonical = {}
    visited = set()
    for i, n in enumerate(names):
        if n in visited: continue
        cluster = [n]
        visited.add(n)
        for m in names[i+1:]:
            if m in visited: continue
            if seq_sim(n, m) >= threshold:
                cluster.append(m)
                visited.add(m)
        canon = sorted(cluster, key=lambda x: (-len(x), x))[0]
        for member in cluster:
            canonical[member] = canon
    return canonical

def build_embedding_clusters(names, threshold=NODE_SIM_THRESHOLD):
    embs = {n: embed(n) for n in names}
    canon = {}
    used = set()

    for n in names:
        if n in used:
            continue
        cluster = [n]
        used.add(n)

        for m in names:
            if m in used:
                continue
            if cosine(embs[n], embs[m]) >= threshold:
                cluster.append(m)
                used.add(m)

        # choose representative deterministically
        representative = sorted(cluster, key=lambda x: (-len(x), x))[0]
        for x in cluster:
            canon[x] = representative

    return canon

def canonicalize_triplets(triplets):
    tcopy = []
    names = collect_all_entity_names(triplets)
    #mapping = build_similarity_clusters(names)
    mapping = build_embedding_clusters(names)
    for t in triplets:
        s = normalize_text(t.get("subject",""))
        o = normalize_text(t.get("object",""))
        tcopy.append({
            **t,
            "subject": mapping.get(s, s),
            "object": mapping.get(o, o),
            "relation": normalize_text(t.get("relation","")),
            "subject_type": t.get("subject_type", "Unknown"),
            "object_type": t.get("object_type", "Unknown")
        })
    return tcopy

def collapse_relations(triplets, threshold=REL_SIM_THRESHOLD):
    rels = sorted({normalize_text(t["relation"]) for t in triplets})
    rel_map, visited = {}, set()
    for i, r in enumerate(rels):
        if r in visited: continue
        cluster = [r]
        visited.add(r)
        for m in rels[i+1:]:
            if m in visited: continue
            if seq_sim(r, m) >= threshold:
                cluster.append(m)
                visited.add(m)
        canon = sorted(cluster, key=lambda x: (-len(x), x))[0]
        for member in cluster:
            rel_map[member] = canon
    out = []
    for t in triplets:
        r = normalize_text(t.get("relation",""))
        t2 = dict(t)
        t2["relation"] = rel_map.get(r, r)
        out.append(t2)
    return out, rel_map

def merge_duplicate_triplets(triplets):
    counts = count_support(triplets)
    merged = {}
    for t in triplets:
        key = (t["subject"], t["relation"], t["object"])
        if key not in merged:
            merged[key] = dict(t)
            merged[key]["support_count"] = counts[key]
    return list(merged.values())


def infer_type(entity, triplets):
    for t in triplets:
        if t.get("subject") == entity and t.get("subject_type") != "Unknown":
            return t["subject_type"]
        if t.get("object") == entity and t.get("object_type") != "Unknown":
            return t["object_type"]
    return "Unknown"

def build_curator_prompt(text, triplets, focus_triplets=None):

    prompt = f"""
    You are a Cyber Threat Intelligence graph curator.
    Your task is to identify missing triplets using ONLY evidence from the TEXT.
    CRITICAL RULES:
    - Normalize entity names so that different mentions referring to the same entity are unified into a single canonical form. 
    - Only propose triplets supported by the TEXT.
    - Do NOT invent relations or entities not mentioned in the TEXT.
    - Do NOT repeat existing triplets.
    - Output ONLY NEW valid triplets with label "PREDICTED".

    Extracted TRIPLETS:
    {json.dumps(triplets, indent=2, ensure_ascii=False)}
    """

    if focus_triplets:
        prompt += "\nLOW-CONFIDENCE TRIPLETS (optional refinement):\n"
        prompt += json.dumps(focus_triplets, indent=2, ensure_ascii=False)

    prompt += f"""

    TEXT:
    {text}

    OUTPUT (JSON only):
    """
    return prompt


# -------------------- LOAD MODELS -------------------------
if not HYBRID: 
   if "Ministral" in BASE_MODEL:
      tokenizer = MistralCommonBackend.from_pretrained(BASE_MODEL)
      base_model = Mistral3ForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto") 
   elif "Qwen3-VL" in BASE_MODEL:
      tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, trust_remote_code=True)
      base_model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Thinking", dtype="auto", device_map="auto")    
   else:
      tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
      base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
      )

   if isLoRa :
       print(f"Loading tokenizer and LoRA {BASE_MODEL} fine-tuned verifier...")
       model = PeftModel.from_pretrained(base_model, LORA_DIR)
       model = model.merge_and_unload()   # merge LoRA weights into base weights
   else:
       print(f"Loading tokenizer and base {BASE_MODEL} verifier...")
       model = base_model

else:      
    tokenizer = MistralCommonBackend.from_pretrained(HYBRID_MODEL)
    base_model = Mistral3ForConditionalGeneration.from_pretrained(HYBRID_MODEL, torch_dtype=torch.bfloat16, device_map="auto")    

    if isLoRa :
       print(f"Loading tokenizer and LoRA {HYBRID_MODEL} fine-tuned verifier...")
       model = PeftModel.from_pretrained(base_model, LORA_HYBRID_DIR)
       model = model.merge_and_unload()   # merge LoRA weights into base weights
    else:
       print(f"Loading tokenizer and base {HYBRID_MODEL} verifier...")
       model = base_model

model.eval()
device = next(model.parameters()).device


merge_model = SentenceTransformer(EMBEDDING_MERGE_MODEL)


# -------------------- LOAD TYPER OUTPUT -------------------
with open(TYPER_OUTPUT_PATH, "r", encoding="utf-8") as f:
    all_test_docs = json.load(f)
    all_test_docs = all_test_docs["results"]


# -------------------- CURATOR FUNCTION -------------------
def curator_agent(doc):
    triplets = [t for t in doc.get("verified_triplets", []) if t.get("label") != "NOT_SUPPORTED"]
    text = doc.get("text", "")
    if not triplets: return []

    # Canonicalize + normalize
    t_canon = canonicalize_triplets(triplets)

    # Collapse relations
    t_rel, _ = collapse_relations(t_canon)

    # Merge duplicates
    t_merged = merge_duplicate_triplets(t_rel)

    # Focus on low-confidence triplets
    t_low_conf = low_confidence_triplets(t_merged)

    # curation prompt - focus on low-confidence triplets to improve precision, while allowing model to add missing links for recall 
    prompt = build_curator_prompt(text, t_merged) #, focus_triplets=t_low_conf)
    print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
    #print(f"PROMPT: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    #max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=1.05,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
    out_tokens = gen[0][inputs["input_ids"].shape[1]:]
    out_text = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
    generated_triplets = safe_parse_json(out_text)
    print(f"[Curator] Proposed triplets: {generated_triplets}")

    # Combine all
    all_triplets = t_merged + generated_triplets

    # Final deduplication
    all_triplets = deduplicate_triplets(all_triplets)
    return all_triplets

# -------------------- RUN CURATOR ------------------------
results = []
start_time = time.time()
for i, doc in enumerate(all_test_docs):
    source_id = doc.get("source_id", f"doc_{i}")
    print(f"\n🔍 Curating document {i}...")
    print(f"Source ID: {source_id}")
    final_triplets = curator_agent(doc)
    results.append({
        "id": doc["id"],
        "source_id": source_id,
        "text": doc["text"],
        "verified_triplets": doc["verified_triplets"],
        "final_triplets": final_triplets
    })
    print(f"Initial triplets: {len(doc['verified_triplets'])} | Final: {len(final_triplets)}")


end_time = time.time()
elapsed = end_time - start_time 
print(f"\n⏱️ Final Curation using fine-tuned model {BASE_MODEL} completed in {elapsed:.2f} seconds.")
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

print(f"\n✅ Curator complete. Saved to {OUTPUT_PATH}")
