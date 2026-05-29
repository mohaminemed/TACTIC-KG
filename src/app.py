# app.py
import os
import re
import json
import webbrowser
import yaml
import math
import torch
import torch.nn.functional as F
import streamlit as st
from pathlib import Path
import webbrowser
import gc


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MistralCommonBackend,
    Mistral3ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from peft import PeftModel
import streamlit.components.v1 as components

from utils.graph_visualizer import build_graph

from sentence_transformers import SentenceTransformer
from collections import Counter
from difflib import SequenceMatcher
import numpy as np


# =========================================================
# CONFIG
# =========================================================

MAX_LENGTH = 4096
MAX_NEW_TOKENS = 4096
MAX_NEW_TOKENS_HYBRID = 8192
CHUNK_SIZE = 4096
OVERLAP = 50

MODELS_DIR = "models"
CONFIGS_DIR = "configs"

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


# =========================================================
# HELPERS
# =========================================================

def cleanup_model(model=None, tokenizer=None):

    try:

        if model is not None:
            del model

        if tokenizer is not None:
            del tokenizer

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            torch.cuda.ipc_collect()

    except Exception as e:

        print(f"Cleanup error: {e}")

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_json_objects(text: str):
    objs = []

    for m in re.finditer(r"(\{.*?\})", text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            objs.append(obj)
        except json.JSONDecodeError:
            continue

    return objs


def chunk_text(text, tokenizer, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    tokens = tokenizer.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        chunk_text_decoded = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        )

        chunks.append(chunk_text_decoded)

        start += chunk_size - overlap

    return chunks


def build_extractor_prompt(text):
    return f"""
You are cybersecurity EXTRACTOR agent, your task is to extract key information
from a given cyber threat intelligence (CTI) report in the form of
subject-relation-object triplets.

Output format (NO TYPES):

[
 {{
   "subject": "...",
   "relation": "...",
   "object": "..."
 }}
]

Rules:

1. Extract ALL entities belonging to the following classes: {CLASSES}
2. Do not type the entities, only extract subject, relation, and object.
3. Your extraction must be COMPLETE and ACCURATE:
   - No hallucinations or external knowledge.
   - No redundancy.
4. Entity consistency and canonicalization (MANDATORY):
   - Use consistent surface forms.
5. Return valid JSON only.

### TARGET report:
CTI: {text}

JSON:
"""


def deduplicate_triplets(triplets):

    seen = set()
    deduped = []

    for t in triplets:

        subj = t.get("subject")
        rel = t.get("relation")
        obj = t.get("object")

        score = t.get("conf_extraction", 0.0)

        if not all(isinstance(x, str) for x in [subj, rel, obj]):
            continue

        subj = subj.strip()
        rel = rel.strip()
        obj = obj.strip()

        if not subj or not rel or not obj:
            continue

        key = (subj.lower(), rel.lower(), obj.lower())

        if key not in seen:

            seen.add(key)

            deduped.append({
                "subject": subj,
                "relation": rel,
                "object": obj,
                "conf_extraction": round(float(score), 4)
            })

    return deduped


def field_confidence(token_logprobs):

    if not token_logprobs:
        return 0.0

    avg_logprob = sum(token_logprobs) / len(token_logprobs)

    return math.exp(avg_logprob)


def align_tokens_to_triplet(triplet, token_logprobs):

    aligned = {
        "subject": [],
        "relation": [],
        "object": []
    }

    gen_tokens = [
        t["token"].replace("Ġ", "").replace("▁", "").strip()
        for t in token_logprobs
    ]

    for field in ["subject", "relation", "object"]:

        value = triplet.get(field, "")

        if not isinstance(value, str):
            value = ""

        text = value.strip().lower()

        if not text:
            continue

        target_tokens = text.split()
        L = len(target_tokens)

        found = False

        for i in range(len(gen_tokens) - L + 1):

            window = " ".join(gen_tokens[i:i + L]).lower()

            if window == text:

                aligned[field] = [
                    token_logprobs[i + j]["logprob"]
                    for j in range(L)
                ]

                found = True
                break

        if not found:

            matched = []

            for i, tok in enumerate(gen_tokens):

                if tok.lower() in target_tokens:
                    matched.append(token_logprobs[i]["logprob"])

            aligned[field] = matched

    return aligned


def score_extraction_confidence(preds, token_logprobs):

    scored = []

    for t in preds:

        field_tokens = align_tokens_to_triplet(
            t,
            token_logprobs
        )

        conf_sub = field_confidence(field_tokens["subject"])
        conf_rel = field_confidence(field_tokens["relation"])
        conf_obj = field_confidence(field_tokens["object"])

        t = t.copy()

        t["conf_extraction"] = min(
            conf_sub,
            conf_rel,
            conf_obj
        )

        scored.append(t)

    return scored


# =========================================================
# MODEL LOADING
# =========================================================

#@st.cache_resource
def load_model(base_model_name, lora_dir=None):

    st.info(f"Loading model: {base_model_name}")

    if "Ministral" in base_model_name:

        tokenizer = MistralCommonBackend.from_pretrained(
            base_model_name
        )

        base_model = Mistral3ForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map= {"": 0} #"auto"
        )

    elif "Qwen3-VL" in base_model_name:

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=False,
            trust_remote_code=True
        )

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            dtype="auto",
            device_map= {"": 0} #"auto"
        )

    else:

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=True
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            device_map= {"": 0}, #"auto"
            torch_dtype=torch.bfloat16
        )

    if lora_dir and os.path.exists(lora_dir):

        st.info(f"Loading LoRA adapter: {lora_dir}")

        model = PeftModel.from_pretrained(
            base_model,
            lora_dir
        )

        model = model.merge_and_unload()

    else:

        model = base_model

    model.eval()

    device = next(model.parameters()).device

    return tokenizer, model, device


# =========================================================
# GENERATION
# =========================================================

def generate_triplets(
    chunk,
    tokenizer,
    model,
    device
):

    prompt = build_extractor_prompt(chunk)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():

        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
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
            "token": tokenizer.convert_ids_to_tokens(
                [token_id.item()]
            )[0],
            "logprob": log_probs[token_id].item()
        })

    out_text = tokenizer.decode(
        gen_tokens,
        skip_special_tokens=True
    ).strip()

    preds = extract_json_objects(out_text)

    return preds, token_logprobs, out_text


# =========================================================
# TYPER HELPERS
# =========================================================

def build_typer_prompt(text, extracted_triplets=""):

    return f"""
You are a Typer, a cyber threat intelligence graph reasoning agent.

You are given a CTI report and extracted triplets.

Your task is to assign entity types to all subjects and objects.

Output format:

[
 {{
    "subject": "<entity>",
    "subject_type": "<Entity Class>",
    "relation": "<relation>",
    "object": "<entity>",
    "object_type": "<Entity Class>"
 }}
]

Rules:

1. Each entity type must reflect the report context.
2. All subjects and objects must be typed.
3. Use ONLY these classes:
{CLASSES}

4. Keep type consistency across triplets.
5. Return VALID JSON only.

### TARGET REPORT:
CTI:
{text}

### EXTRACTED TRIPLETS:
{json.dumps(extracted_triplets, indent=2)}

JSON:
"""


def align_tokens_to_typed_triplet(
    triplet,
    token_logprobs
):

    aligned = {
        "subject_type": [],
        "object_type": []
    }

    gen_tokens = [
        t["token"].replace("Ġ", "").replace("▁", "").strip().lower()
        for t in token_logprobs
    ]

    for field in ["subject_type", "object_type"]:

        text = triplet.get(field, "").strip().lower()

        if not text:
            continue

        target_tokens = text.split()

        L = len(target_tokens)

        found = False

        for i in range(len(gen_tokens) - L + 1):

            window = " ".join(gen_tokens[i:i + L])

            if window == text:

                aligned[field] = [
                    token_logprobs[i + j]["logprob"]
                    for j in range(L)
                ]

                found = True
                break

        if not found:

            matched = []

            for i, tok in enumerate(gen_tokens):

                if tok in target_tokens:
                    matched.append(
                        token_logprobs[i]["logprob"]
                    )

            aligned[field] = matched

    return aligned


def generate_typed_triplets(
    text,
    extracted_triplets,
    tokenizer,
    model,
    device
):

    prompt = build_typer_prompt(
        text,
        extracted_triplets
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():

        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    input_len = inputs["input_ids"].shape[1]

    gen_tokens = gen.sequences[0][input_len:]

    out_text = tokenizer.decode(
        gen_tokens,
        skip_special_tokens=True
    ).strip()

    token_logprobs = []

    for step, scores in enumerate(gen.scores):

        log_probs = F.log_softmax(scores[0], dim=-1)

        token_id = gen_tokens[step]

        token_logprobs.append({
            "token_id": token_id.item(),
            "token": tokenizer.convert_ids_to_tokens(
                [token_id.item()]
            )[0],
            "logprob": log_probs[token_id].item()
        })

    preds = extract_json_objects(out_text)

    return preds, token_logprobs, out_text


def deduplicate_typed_triplets(triplets):

    seen = set()

    deduped = []

    for t in triplets:

        subj = t.get("subject", "")
        subj_type = t.get("subject_type", "")

        rel = t.get("relation", "")

        obj = t.get("object", "")
        obj_type = t.get("object_type", "")

        if not subj or not rel or not obj:
            continue

        key = (
            subj.lower(),
            rel.lower(),
            obj.lower()
        )

        if key not in seen:

            seen.add(key)

            deduped.append({
                "subject": subj,
                "subject_type": subj_type,
                "relation": rel,
                "object": obj,
                "object_type": obj_type,
                "conf_subject_type": round(
                    t.get("conf_subject_type", 0.0),
                    4
                ),
                "conf_object_type": round(
                    t.get("conf_object_type", 0.0),
                    4
                )
            })

    return deduped


def normalize_text(s: str) -> str:

    if not s:
        return ""

    return " ".join(
        s.lower().strip().split()
    )


def safe_parse_json(text):

    try:

        start = text.index("[")
        end = text.rindex("]") + 1

        return json.loads(text[start:end])

    except Exception:
        return []
    


def build_verifier_prompt(text, triplets):

    return f"""
You are a Cyber Threat Intelligence knowledge graph verifier.

Your task is to evaluate whether each triplet is explicitly supported by the CTI report.

You MUST classify each triplet into ONE label:

SUPPORTED
NOT_SUPPORTED

Rules:

1. Use ONLY the report.
2. No external knowledge.
3. Verify ALL triplets.
4. Provide evidence sentence when possible.
5. Return VALID JSON ONLY.

OUTPUT FORMAT:

[
 {{
   "subject": "...",
   "subject_type": "...",
   "relation": "...",
   "object": "...",
   "object_type": "...",
   "label": "SUPPORTED",
   "evidence": "..."
 }}
]

### CTI REPORT:
{text}

### TRIPLETS:
{json.dumps(triplets, indent=2)}

JSON:
"""


def generate_verified_triplets(
    text,
    typed_triplets,
    tokenizer,
    model,
    device
):

    prompt = build_verifier_prompt(
        text,
        typed_triplets
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():

        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_HYBRID,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    out_tokens = gen[0][inputs["input_ids"].shape[1]:]

    out_text = tokenizer.decode(
        out_tokens,
        skip_special_tokens=True
    ).strip()

    preds = safe_parse_json(out_text)

    return preds, out_text


def deduplicate_verified_triplets(triplets):

    seen = set()

    deduped = []

    for t in triplets:

        subj = t.get("subject", "")
        rel = t.get("relation", "")
        obj = t.get("object", "")

        label = t.get("label", "")
        evidence = t.get("evidence", "")

        if not subj or not rel or not obj:
            continue

        key = (
            subj.lower(),
            rel.lower(),
            obj.lower(),
            label.lower()
        )

        if key not in seen:

            seen.add(key)

            deduped.append(t)

    return deduped


def cosine(a, b):

    return float(np.dot(a, b))

#@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

def collect_all_entity_names(triplets):

    return (
        {
            normalize_text(t.get("subject", ""))
            for t in triplets
        }
        |
        {
            normalize_text(t.get("object", ""))
            for t in triplets
        }
    )

def build_embedding_clusters(
    names,
    embedding_model,
    threshold=0.85
):

    embs = {
        n: embedding_model.encode(
            n,
            normalize_embeddings=True
        )
        for n in names
    }

    canonical = {}

    used = set()

    for n in names:

        if n in used:
            continue

        cluster = [n]

        used.add(n)

        for m in names:

            if m in used:
                continue

            sim = cosine(
                embs[n],
                embs[m]
            )

            if sim >= threshold:

                cluster.append(m)

                used.add(m)

        representative = sorted(
            cluster,
            key=lambda x: (-len(x), x)
        )[0]

        for x in cluster:

            canonical[x] = representative

    return canonical


def canonicalize_triplets(
    triplets,
    embedding_model
):

    names = collect_all_entity_names(
        triplets
    )

    mapping = build_embedding_clusters(
        names,
        embedding_model
    )

    out = []

    for t in triplets:

        s = normalize_text(
            t.get("subject", "")
        )

        o = normalize_text(
            t.get("object", "")
        )

        out.append({
            **t,
            "subject": mapping.get(s, s),
            "object": mapping.get(o, o),
            "relation": normalize_text(
                t.get("relation", "")
            )
        })

    return out


def collapse_relations(
    triplets,
    threshold=0.90
):

    rels = sorted({
        normalize_text(t["relation"])
        for t in triplets
    })

    rel_map = {}

    visited = set()

    for i, r in enumerate(rels):

        if r in visited:
            continue

        cluster = [r]

        visited.add(r)

        for m in rels[i + 1:]:

            if m in visited:
                continue

            sim = SequenceMatcher(
                None,
                r,
                m
            ).ratio()

            if sim >= threshold:

                cluster.append(m)

                visited.add(m)

        canon = sorted(
            cluster,
            key=lambda x: (-len(x), x)
        )[0]

        for member in cluster:

            rel_map[member] = canon

    out = []

    for t in triplets:

        r = normalize_text(
            t.get("relation", "")
        )

        t2 = dict(t)

        t2["relation"] = rel_map.get(r, r)

        out.append(t2)

    return out


def count_support(triplets):

    c = Counter()

    for t in triplets:

        key = (
            t["subject"],
            t["relation"],
            t["object"]
        )

        c[key] += 1

    return c


def merge_duplicate_triplets(triplets):

    counts = count_support(
        triplets
    )

    merged = {}

    for t in triplets:

        key = (
            t["subject"],
            t["relation"],
            t["object"]
        )

        if key not in merged:

            merged[key] = dict(t)

            merged[key]["support_count"] = counts[key]

    return list(merged.values())


def build_curator_prompt(
    text,
    triplets
):

    return f"""
You are a Cyber Threat Intelligence graph curator.

Your task is to identify NEW missing triplets
supported by the CTI report.

Rules:

1. Use ONLY the CTI report.
2. Do NOT hallucinate.
3. Do NOT repeat existing triplets.
4. Normalize entities consistently.
5. Return ONLY NEW triplets.
6. All generated triplets must include:
   - subject
   - subject_type
   - relation
   - object
   - object_type
   - label = "PREDICTED"

### EXISTING TRIPLETS:
{json.dumps(triplets, indent=2)}

### CTI REPORT:
{text}

JSON:
"""


def generate_curated_triplets(
    text,
    triplets,
    tokenizer,
    model,
    device
):

    prompt = build_curator_prompt(
        text,
        triplets
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():

        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    out_tokens = gen[0][inputs["input_ids"].shape[1]:]

    out_text = tokenizer.decode(
        out_tokens,
        skip_special_tokens=True
    ).strip()

    preds = safe_parse_json(
        out_text
    )

    return preds, out_text


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="TACTIC-KG Interactive Pipeline",
    layout="wide"
)

#st.title("TACTIC-KG: Agentic CSKG Construction")

col1, col2 = st.columns([1, 8])

with col1:
    st.image(
        "tactic-kg.png",
        width=100
    )

with col2:
    st.title(
        "TACTIC-KG: Agentic CSKG Construction"
    )

st.markdown("""
Paste a CTI report and run the extraction pipeline interactively.
The output will be shown as structured JSON.
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Pipeline Configuration")

config_files = []

if os.path.exists(CONFIGS_DIR):
    config_files = [
        f for f in os.listdir(CONFIGS_DIR)
        if f.endswith(".yaml")
    ]

selected_config = st.sidebar.selectbox(
    "Select Config",
    config_files
)

config_path = os.path.join(
    CONFIGS_DIR,
    selected_config
)

config = load_yaml_config(config_path)

default_base_model = config["models"]["base_model"]
default_base_hybrid_model = config["models"]["hybrid_model"]

base_model = st.sidebar.text_input(
    "Base Model",
    value=default_base_model
)

base_hybrid_model = st.sidebar.text_input(
    "Base Hybrid Model",
    value=default_base_hybrid_model
)

available_loras = []

if os.path.exists(MODELS_DIR):
    available_loras = sorted(os.listdir(MODELS_DIR))

selected_lora = st.sidebar.selectbox(
    "Select LoRA Adapter for Extractor",
    ["None"] + available_loras
)

lora_path = None

if selected_lora != "None":
    lora_path = os.path.join(
        MODELS_DIR,
        selected_lora
    )

# =========================================================
# INPUT
# =========================================================

report = st.text_area(
    "Paste CTI Report",
    height=400,
    placeholder="Paste cybersecurity report here..."
)

run_button = st.button(
    "Run Pipeline",
    use_container_width=True
)


# =========================================================
# Typer configuration
# =========================================================

st.header("Typer Agent")

enable_typer = st.checkbox(
    "Enable Typer Agent",
    value=True
)

available_typer_loras = []

if os.path.exists(MODELS_DIR):
    available_typer_loras = sorted(os.listdir(MODELS_DIR))

selected_typer_lora = st.selectbox(
    "Select Typer LoRA Adapter",
    ["None"] + available_typer_loras,
    key="typer_lora"
)

typer_lora_path = None

if selected_typer_lora != "None":
    typer_lora_path = os.path.join(
        MODELS_DIR,
        selected_typer_lora
    )


# =========================================================
# Verifier configuration
# =========================================================

st.header("Verifier Agent")

enable_verifier = st.checkbox(
    "Enable Verifier Agent",
    value=True
)

available_verifier_loras = []

if os.path.exists(MODELS_DIR):
    available_verifier_loras = sorted(os.listdir(MODELS_DIR))

selected_verifier_lora = st.selectbox(
    "Select Verifier LoRA Adapter",
    ["None"] + available_verifier_loras,
    key="verifier_lora"
)

verifier_lora_path = None

if selected_verifier_lora != "None":
    verifier_lora_path = os.path.join(
        MODELS_DIR,
        selected_verifier_lora
    )


# =========================================================
# CURATOR CONFIGURATION
# =========================================================

st.header("Curator Agent")

enable_curator = st.checkbox(
    "Enable Curator Agent",
    value=True
)

available_curator_loras = []

if os.path.exists(MODELS_DIR):
    available_curator_loras = sorted(os.listdir(MODELS_DIR))

selected_curator_lora = st.selectbox(
    "Select Curator LoRA Adapter",
    ["None"] + available_curator_loras,
    key="curator_lora"
)

curator_lora_path = None

if selected_curator_lora != "None":

    curator_lora_path = os.path.join(
        MODELS_DIR,
        selected_curator_lora
    )


# =========================================================
# EXECUTION
# =========================================================

if run_button:

    if not report.strip():
        st.error("Please provide a CTI report.")
        st.stop()

    # =====================================================
    # LOAD EXTRACTOR
    # =====================================================

    with st.spinner("Loading extractor model..."):

        tokenizer, extractor_model, device = load_model(
            base_model,
            lora_path
        )

    # =====================================================
    # EXTRACTOR
    # =====================================================

    st.header("Extractor Execution")

    chunks = chunk_text(report, tokenizer)

    st.success(f"Created {len(chunks)} chunk(s)")

    all_preds = []

    raw_outputs = []

    progress = st.progress(0)

    for idx, chunk in enumerate(chunks):

        preds, token_logprobs, raw_output = generate_triplets(
            chunk,
            tokenizer,
            extractor_model,
            device
        )

        raw_outputs.append(raw_output)

        required_keys = [
            "subject",
            "relation",
            "object"
        ]

        valid_preds = [
            t for t in preds
            if all(k in t for k in required_keys)
        ]

        if valid_preds:

            scored_preds = score_extraction_confidence(
                valid_preds,
                token_logprobs
            )

            all_preds.extend(scored_preds)

        progress.progress((idx + 1) / len(chunks))

    extractor_triplets = deduplicate_triplets(all_preds)

    st.session_state["extractor_triplets"] = extractor_triplets

    extractor_json = {
        "base_model": base_model,
        "extractor_lora": selected_lora,
        "num_triplets": len(extractor_triplets),
        "triplets": extractor_triplets
    }

    st.subheader("Extractor JSON")

    st.json(extractor_json)

    cleanup_model(
    extractor_model,
    tokenizer
    )

    graph_path = None
    if "extractor_triplets" not in st.session_state:
        st.warning("No extractor triplets available.")
    else:
        graph_path = build_graph(
            st.session_state["extractor_triplets"],
            graph_name="Extractor Graph",
            origin="extractor",
            agent_name="Extractor"
        )

    # =====================================================
    # EXTRACTOR GRAPH
    # =====================================================
    if graph_path and os.path.exists(graph_path):
         
        with open(graph_path, "r", encoding="utf-8") as f:
            html = f.read()

        st.download_button(
            label="Download Extractor Graph HTML",
            data=html,
            file_name="extractor_triplets.html",
            mime="text/html"
        )    



    # =====================================================
    # TYPER
    # =====================================================

    if enable_typer:

        st.header("Typer Execution")

        with st.spinner("Loading typer model..."):

            typer_tokenizer, typer_model, typer_device = load_model(
                base_model,
                typer_lora_path
            )

        typed_preds, typer_logprobs, typer_raw = generate_typed_triplets(
            report,
            extractor_triplets,
            typer_tokenizer,
            typer_model,
            typer_device
        )

        for t in typed_preds:

            aligned = align_tokens_to_typed_triplet(
                t,
                typer_logprobs
            )

            t["conf_subject_type"] = field_confidence(
                aligned.get("subject_type", [])
            )

            t["conf_object_type"] = field_confidence(
                aligned.get("object_type", [])
            )

        typed_triplets = deduplicate_typed_triplets(
            typed_preds
        )

        st.session_state["typed_triplets"] = typed_triplets

        typer_json = {
            "base_model": base_model,
            "extractor_lora": selected_lora,
            "typer_lora": selected_typer_lora,
            "num_typed_triplets": len(typed_triplets),
            "typed_triplets": typed_triplets
        }

        st.subheader("Typer JSON")

        st.json(typer_json)

        cleanup_model(
            typer_model,
            typer_tokenizer
        )

        graph_path = None
        if "typed_triplets" not in st.session_state:
            st.warning("No typed triplets available.")
        else:
            graph_path = build_graph(
                st.session_state["typed_triplets"],
                graph_name="Typer Graph",
                origin="typer",
                agent_name="Typer"
            )   

        # =====================================================
        # TYPER GRAPH
        # =====================================================

        
        if graph_path and os.path.exists(graph_path):
         
           with open(graph_path, "r", encoding="utf-8") as f:
             html = f.read()

           st.download_button(
              label="Download Typer Graph HTML",
              data=html,
              file_name="typer_triplets.html",
              mime="text/html"
           )    

        st.download_button(
            label="Download Typed JSON",
            data=json.dumps(
                typer_json,
                indent=2,
                ensure_ascii=False
            ),
            file_name="typed_triplets.json",
            mime="application/json"
        )

        with st.expander("Raw Typer Output"):

            st.code(
                typer_raw,
                language="json"
            )

        # =====================================================
        # VERIFIER
        # =====================================================

        if enable_verifier:

           st.header("Verifier Execution")

           with st.spinner("Loading verifier model..."):

              verifier_tokenizer, verifier_model, verifier_device = load_model(
                base_hybrid_model,
                verifier_lora_path
              )

           verified_preds, verifier_raw = generate_verified_triplets(
              report,
              typed_triplets,
              verifier_tokenizer,
              verifier_model,
              verifier_device
           )

           verified_triplets = deduplicate_verified_triplets(
              verified_preds
           )

           st.session_state["verified_triplets"] = verified_triplets

           verifier_json = {
              "base_model": base_hybrid_model,
              "extractor_lora": selected_lora,
              "typer_lora": selected_typer_lora,
              "verifier_lora": selected_verifier_lora,
              "num_verified_triplets": len(verified_triplets),
              "verified_triplets": verified_triplets
           }

           st.subheader("Verifier JSON")

           st.json(verifier_json)

           cleanup_model(
              verifier_model,
              verifier_tokenizer
           )

           graph_path = None

           if "verified_triplets" not in st.session_state:

              st.warning("No verified triplets available.")

           else:

              graph_path = build_graph(
                st.session_state["verified_triplets"],
                graph_name="Verifier Graph",
                origin="verifier",
                agent_name="Verifier"
              )

           # =====================================================
           # VERIFIER GRAPH
           # =====================================================

           if graph_path and os.path.exists(graph_path):

              with open(graph_path, "r", encoding="utf-8") as f:

                html = f.read()

              st.download_button(
                label="Download Verifier Graph HTML",
                data=html,
                file_name="verified_triplets.html",
                mime="text/html"
              )

           st.download_button(
             label="Download Verified JSON",
             data=json.dumps(
                verifier_json,
                indent=2,
                ensure_ascii=False
            ),
            file_name="verified_triplets.json",
            mime="application/json"
           )

           with st.expander("Raw Verifier Output"):

             st.code(
                verifier_raw,
                language="json"
             )


           # =====================================================
           # CURATOR
           # =====================================================

           if enable_curator:

              st.header("Curator Execution")

              with st.spinner("Loading curator model..."):

                curator_tokenizer, curator_model, curator_device = load_model(
                base_hybrid_model,
                curator_lora_path
              )

              with st.spinner("Loading embedding model..."):

                embedding_model = load_embedding_model()

              verified_supported = [

                t for t in verified_triplets

                if t.get("label") != "NOT_SUPPORTED"
              ]

              # ================================================
              # CANONICALIZATION
              # ================================================

              t_canon = canonicalize_triplets(
                verified_supported,
                embedding_model
               )

              # ================================================
              # RELATION COLLAPSE
              # ================================================

              t_rel = collapse_relations(
                t_canon
              )

              # ================================================
              # MERGE DUPLICATES
              # ================================================

              t_merged = merge_duplicate_triplets(
                t_rel
              )

              # ================================================
              # GENERATE NEW TRIPLETS
              # ================================================

              generated_triplets, curator_raw = generate_curated_triplets(
                report,
                t_merged,
                curator_tokenizer,
                curator_model,
                curator_device
              )

              final_triplets = deduplicate_verified_triplets(
               t_merged + generated_triplets
              )

              st.session_state["final_triplets"] = final_triplets

              curator_json = {
                "base_model": base_hybrid_model,
                "extractor_lora": selected_lora,
                "typer_lora": selected_typer_lora,
                "verifier_lora": selected_verifier_lora,
                "curator_lora": selected_curator_lora,
                "num_final_triplets": len(final_triplets),
                "final_triplets": final_triplets
              }

              st.subheader("Curator JSON")

              st.json(curator_json)

              cleanup_model(
                curator_model,
                curator_tokenizer
              )

              graph_path = None

              if "final_triplets" not in st.session_state:

                st.warning("No curated triplets available.")

              else:

               graph_path = build_graph(
                st.session_state["final_triplets"],
                graph_name="Curated Graph",
                origin="curator",
                agent_name="Curator"
               )

              # =====================================================
              # CURATOR GRAPH
              # =====================================================

              if graph_path and os.path.exists(graph_path):

                with open(graph_path, "r", encoding="utf-8") as f:

                   html = f.read()

                st.download_button(
                  label="Download Curated Graph HTML",
                  data=html,
                  file_name="curated_triplets.html",
                  mime="text/html"
                )

              st.download_button(
                label="Download Final JSON",
                data=json.dumps(
                curator_json,
                indent=2,
                ensure_ascii=False
               ),
               file_name="final_triplets.json",
               mime="application/json"
              )

              with st.expander("Raw Curator Output"):

               st.code(
                curator_raw,
                language="json"
               )  

