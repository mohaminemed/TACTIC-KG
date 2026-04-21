import json
import csv
import os
import argparse
from pathlib import Path

import torch
import yaml
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# ------------------ Args ------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

args = parse_args()

# ------------------ Load YAML ------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

NUM_DOCS = config["experiment"]["num_docs"]
SIM_THRESHOLD = 0.60

# ------------------ Normalization ------------------
def normalize_triplet(t):
    subj = str(t.get("subject", "")).lower().strip()
    rel  = str(t.get("relation", "")).lower().strip()
    obj  = str(t.get("object", "")).lower().strip()
    return (subj, rel, obj)

def normalize_entity_type(entity_type):
    if not entity_type:
        return ""
    return entity_type.split(":")[0].strip().lower()

# ------------------ Embeddings ------------------

#model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device=device)
#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
#model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device=device)

def embed_triplet(t):
    text = " ".join([
        str(t.get("subject","")).lower().strip(),
        str(t.get("relation","")).lower().strip(),
        str(t.get("object","")).lower().strip()
    ])
    return model.encode(text, normalize_embeddings=True)

def embedding_match(e1, e2, threshold=SIM_THRESHOLD):
    return float(util.dot_score(e1, e2)) >= threshold

# ------------------ Triplet Set Scoring ------------------
def score_triplet_sets_semantic(gold, pred, embed_threshold=SIM_THRESHOLD):
    gold_enc = [(normalize_triplet(t), embed_triplet(t)) for t in gold]
    pred_enc = [(normalize_triplet(t), embed_triplet(t)) for t in pred]

    matched_gold = set()
    matched_pred = set()

    for i, (g_text, g_emb) in enumerate(gold_enc):
        for j, (p_text, p_emb) in enumerate(pred_enc):
            if j in matched_pred:
                continue
            if embedding_match(g_emb, p_emb, embed_threshold):
                matched_gold.add(i)
                matched_pred.add(j)
                break

    tp = len(matched_gold)
    fp = len(pred_enc) - tp
    fn = len(gold_enc) - tp

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1

# ------------------ Entity Typing ------------------
def entity_typing_metrics(preds, golds, embed_threshold=SIM_THRESHOLD):
    type_correct = 0
    partial_type_correct = 0
    total = 0

    gold_subjects = {g["subject"].lower(): normalize_entity_type(g.get("subject_type")) for g in golds}
    gold_objects  = {g["object"].lower():  normalize_entity_type(g.get("object_type"))  for g in golds}

    gold_subject_embs = {k: embed_triplet({"subject": k, "relation":"","object":""}) for k in gold_subjects}
    gold_object_embs  = {k: embed_triplet({"subject":"","relation":"", "object": k}) for k in gold_objects}

    for pred in preds:
        subj = pred.get("subject", "").lower()
        obj  = pred.get("object", "").lower()

        pred_subj_type = normalize_entity_type(pred.get("subject_type"))
        pred_obj_type  = normalize_entity_type(pred.get("object_type"))
       
        subj_match = any(embedding_match(embed_triplet({"subject":subj,"relation":"","object":""}), gold_subject_embs[gsubj], embed_threshold) for gsubj in gold_subjects)
        obj_match  = any(embedding_match(embed_triplet({"subject":"","relation":"","object":obj}), gold_object_embs[gobj], embed_threshold) for gobj in gold_objects)
      
        subj_type_correct = any((embedding_match(embed_triplet({"subject":subj,"relation":"","object":""}), gold_subject_embs[gsubj], embed_threshold)) and pred_subj_type == gold_subjects[gsubj] for gsubj in gold_subjects)
        obj_type_correct  = any((embedding_match(embed_triplet({"subject":"","relation":"","object":obj}), gold_object_embs[gobj], embed_threshold)) and pred_obj_type == gold_objects[gobj] for gobj in gold_objects)

        if subj_match or obj_match:
            total += 1
            if subj_type_correct and obj_type_correct:
                type_correct += 1
            elif subj_type_correct or obj_type_correct:
                partial_type_correct += 1

    if total == 0:
        return 0.0, 0.0

    type_acc = type_correct / total
    partial_type_acc = (type_correct + 0.5 * partial_type_correct) / total
    return type_acc, partial_type_acc

# ------------------ Graph Similarity ------------------
def graph_similarity(gold, pred, embed_threshold=SIM_THRESHOLD):
    gold_enc = [(" ".join(normalize_triplet(t)), embed_triplet(t)) for t in gold]
    pred_enc = [(" ".join(normalize_triplet(t)), embed_triplet(t)) for t in pred]

    matched_pred = set()
    matches = 0

    for g_text, g_emb in gold_enc:
        for i, (p_text, p_emb) in enumerate(pred_enc):
            if i in matched_pred:
                continue
            if embedding_match(g_emb, p_emb, embed_threshold):
                matches += 1
                matched_pred.add(i)
                break

    union = len(gold_enc) + len(pred_enc) - matches
    return matches / (union + 1e-9)

# ------------------ Evaluation ------------------
for model_cfg in config["models"]:
    model_name = model_cfg["name"]
    file_path = model_cfg["file"]

    print(f"\n\n==============================")
    print(f"EVALUATING {model_name}")
    print(f"==============================")

    with open(file_path) as f:
        data = json.load(f)
        data = data["results"]

    scores = []
    per_doc_results = []

    for i, doc in enumerate(data):
        gold = doc["gold_triplets"]
        pred = doc["pred_triplets"]

        print(f"\n--- Document {i} ---")

        p1, r1, f1_1 = score_triplet_sets_semantic(gold, pred)
        t1, pt1 = entity_typing_metrics(pred, gold)
        g1 = graph_similarity(gold, pred)

        scores.append((p1, r1, f1_1, t1, pt1, g1))

        per_doc_results.append({
            "doc_id": i,
            "ctinexus": {
                "precision": p1,
                "recall": r1,
                "f1": f1_1,
                "typing_accuracy": t1,
                "partial_typing_accuracy": pt1,
                "graph_similarity": g1
            }
        })

    def avg(scores):
        return tuple(sum(x[i] for x in scores) / len(scores) for i in range(6))

    P_ex, R_ex, F1_ex, Typ_ex, PTyp_ex, Graph_ex = avg(scores)

    print("\n=== CTINEXUS Extraction Performance ===")
    print(f"Precision: {P_ex:.3f}  Recall: {R_ex:.3f}  F1: {F1_ex:.3f}")
    print(f"Typing Accuracy: {Typ_ex:.3f}  Partial Typing Accuracy: {PTyp_ex:.3f}  Graph: {Graph_ex:.3f}")

    # ------------------ SAVE RESULTS ----------------
    output_dir = f"evaluation_results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save per-document CSV
    with open(f"{output_dir}/results_per_doc_{NUM_DOCS}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "precision", "recall", "f1", "typing", "partial_typing", "graph"])
        for r in per_doc_results:
            writer.writerow([
                r["doc_id"],
                r["ctinexus"]["precision"],
                r["ctinexus"]["recall"],
                r["ctinexus"]["f1"],
                r["ctinexus"]["typing_accuracy"],
                r["ctinexus"]["partial_typing_accuracy"],
                r["ctinexus"]["graph_similarity"],
            ])

    # Save summary CSV
    with open(f"{output_dir}/summary_{NUM_DOCS}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "precision", "recall", "f1", "typing", "partial_typing", "graph"])
        writer.writerow([model_name, P_ex, R_ex, F1_ex, Typ_ex, PTyp_ex, Graph_ex])

    print("\nSaved evaluation to:", output_dir)
