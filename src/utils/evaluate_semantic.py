import json
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

import yaml
import argparse
import csv
import os
device = "cuda:1" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

args = parse_args()

# Load YAML
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Global config
MODEL_NAME = config["models"]["base_model"]
isLoRa = config["experiment"]["islora"]
NUM_DOCS = config["experiment"]["num_docs"]

SIM_THRESHOLD = config["experiment"]["evaluation_threshold"]
EVAL_MODEL = config["models"]["evaluation_model"]

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

# ------------------ Semantic Matching Model ------------------
model = SentenceTransformer(EVAL_MODEL, device=device)


def embed_triplet(t):
    text = " ".join([
        str(t.get("subject","")).lower().strip(),
        str(t.get("relation","")).lower().strip(),
        str(t.get("object","")).lower().strip()
    ])
    return model.encode(text, normalize_embeddings=True)

def embedding_match(e1, e2, threshold=SIM_THRESHOLD):
    return float(util.dot_score(e1, e2)) >= threshold


# ------------------ Triplet Set Scoring (Optimal Matching) ------------------
def score_triplet_sets_semantic_optimal(gold, pred, embed_threshold=SIM_THRESHOLD):

    if len(gold) == 0 and len(pred) == 0:
        return 1.0, 1.0, 1.0
    if len(gold) == 0:
        return 0.0, 0.0, 0.0
    if len(pred) == 0:
        return 0.0, 0.0, 0.0

    # Encode triplets
    gold_embs = [embed_triplet(t) for t in gold]
    pred_embs = [embed_triplet(t) for t in pred]

    # Build similarity matrix
    sim_matrix = cosine_similarity(gold_embs, pred_embs)

    # Hungarian algorithm minimizes cost → convert similarity to cost
    cost_matrix = 1 - sim_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Count matches above threshold
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if sim_matrix[r, c] >= embed_threshold:
            tp += 1

    fp = len(pred) - tp
    fn = len(gold) - tp

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    return prec, rec, f1

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

# ------------------ Load Data ------------------
if isLoRa :
  with open(f"outputs/unseen/ft_{MODEL_NAME}/typed_triplets.json") as f:
    extractor_data = json.load(f)["results"]
  with open(f"outputs/unseen/ft_{MODEL_NAME}/verified_triplets.json") as f:
    verified_data = json.load(f)["results"] 
  with open(f"outputs/unseen/ft_{MODEL_NAME}/final_triplets.json", "r", encoding="utf-8") as f:
    final_data = json.load(f)["results"]  # each doc includes "final_triplets"
else:
  with open(f"outputs/unseen/{MODEL_NAME}/typed_triplets.json") as f:
    extractor_data = json.load(f)["results"]
  with open(f"outputs/unseen/{MODEL_NAME}/verified_triplets.json") as f:
    verified_data = json.load(f)["results"] 
  with open(f"outputs/unseen/{MODEL_NAME}/final_triplets.json", "r", encoding="utf-8") as f:
    final_data = json.load(f)["results"]  # each doc includes "final_triplets" 
    
  
def keep_supported_only(verified_triplets):
    return [
        item
        for item in verified_triplets
        if item["label"] != "NOT_SUPPORTED"
    ]

# ------------------ Evaluation ------------------
per_doc_results = []

for i in range(NUM_DOCS):
    #if i < 21 : #[22, 23]:  # docs 23 and 24 have malformed JSON in final_triplets, skip them
        #continue
    gold = extractor_data[i]["gold_triplets"]
    pred_init = extractor_data[i]["typed_triplets"]
    verified = verified_data[i]["verified_triplets"]
    verified = keep_supported_only(verified)
    final_triplets = final_data[i]["final_triplets"]  # final output from curator
    final_triplets = keep_supported_only(final_triplets)

    print(f"\n--- Document {i} ---")

    # Extractor
    p1, r1, f1_1 = score_triplet_sets_semantic_optimal(gold, pred_init)
    t1, pt1 = entity_typing_metrics(pred_init, gold)
    g1 = graph_similarity(gold, pred_init)

    # Curator (SUPPORT only)
    p3, r3, f1_3 = score_triplet_sets_semantic_optimal(gold, verified)
    t3, pt3 = entity_typing_metrics(verified, gold)
    g3 = graph_similarity(gold, verified)

    # Final Triplets (after full curation)
    p4, r4, f1_4 = score_triplet_sets_semantic_optimal(gold, final_triplets)
    t4, pt4 = entity_typing_metrics(final_triplets, gold)
    g4 = graph_similarity(gold, final_triplets)


    print(f"Extractor - Precision: {p1:.3f}  Recall: {r1:.3f}  F1: {f1_1:.3f}  Typing Acc: {t1:.3f}  Partial Typing Acc: {pt1:.3f}  Graph Sim: {g1:.3f}")
    print(f"Verifier  - Precision: {p3:.3f}  Recall: {r3:.3f}  F1: {f1_3:.3f}  Typing Acc: {t3:.3f}  Partial Typing Acc: {pt3:.3f}  Graph Sim: {g3:.3f}")
    print(f"Curator   - Precision: {p4:.3f}  Recall: {r4:.3f}  F1: {f1_4:.3f}  Typing Acc: {t4:.3f}  Partial Typing Acc: {pt4:.3f}  Graph Sim: {g4:.3f}")

    per_doc_results.append({
        "doc_id": i,
        "extractor": {
            "precision": p1,
            "recall": r1,
            "f1": f1_1,
            "typing_accuracy": t1,
            "partial_typing_accuracy": pt1,
            "graph_similarity": g1
        },
        "verifier": {
            "precision": p3,
            "recall": r3,
            "f1": f1_3,
            "typing_accuracy": t3,
            "partial_typing_accuracy": pt3,
            "graph_similarity": g3
        },
        "curator": {
            "precision": p4,
            "recall": r4,
            "f1": f1_4,
            "typing_accuracy": t4,
            "partial_typing_accuracy": pt4,
            "graph_similarity": g4
        }
    })

# ------------------ Average ------------------
def avg(scores, key):
    """Compute average for six metrics from dict scores."""
    metrics = ["precision", "recall", "f1", "typing_accuracy", "partial_typing_accuracy", "graph_similarity"]
    return tuple(sum(x[key][m] for x in scores) / len(scores) for m in metrics)

P_ex, R_ex, F1_ex, Typ_ex, PTyp_ex, Graph_ex = avg(per_doc_results, "extractor")
P_cr, R_cr, F1_cr, Typ_cr, PTyp_cr, Graph_cr = avg(per_doc_results, "verifier")
P_fn, R_fn, F1_fn, Typ_fn, PTyp_fn, Graph_fn = avg(per_doc_results, "curator")

print("\n=== Extraction Performance ===")
print(f"Precision: {P_ex:.3f}  Recall: {R_ex:.3f}  F1: {F1_ex:.3f}")
print(f"Typing Accuracy: {Typ_ex:.3f}  Partial Typing Accuracy: {PTyp_ex:.3f}  Graph: {Graph_ex:.3f}")

print("\n=== Verification Performance (SUPPORTED) ===")
print(f"Precision: {P_cr:.3f}  Recall: {R_cr:.3f}  F1: {F1_cr:.3f}")
print(f"Typing Accuracy: {Typ_cr:.3f}  Partial Typing Accuracy: {PTyp_cr:.3f}  Graph: {Graph_cr:.3f}")

print("\n=== Final Curated Triplets Performance ===")
print(f"Precision: {P_fn:.3f}  Recall: {R_fn:.3f}  F1: {F1_fn:.3f}")
print(f"Typing Accuracy: {Typ_fn:.3f}  Partial Typing Accuracy: {PTyp_fn:.3f}  Graph: {Graph_fn:.3f}")



# ================= SAVE RESULTS =================

output_dir = f"evaluation_results/{MODEL_NAME}/unseen"
os.makedirs(output_dir, exist_ok=True)



# ---------- SAVE CSV PER DOC ----------

with open(f"{output_dir}/results_per_doc_{NUM_DOCS}.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow([
        "doc_id",

        "extractor_precision",
        "extractor_recall",
        "extractor_f1",
        "extractor_typing",
        "extractor_partial_typing",
        "extractor_graph",

        "verifier_precision",
        "verifier_recall",
        "verifier_f1",
        "verifier_typing",
        "verifier_partial_typing",
        "verifier_graph", 

        "curator_precision",
        "curator_recall",
        "curator_f1",
        "curator_typing",
        "curator_partial_typing",
        "curator_graph"   
    ])
    for r in per_doc_results:
        writer.writerow([

            r["doc_id"],
            r["extractor"]["precision"],
            r["extractor"]["recall"],
            r["extractor"]["f1"],
            r["extractor"]["typing_accuracy"],
            r["extractor"]["partial_typing_accuracy"],
            r["extractor"]["graph_similarity"],

            r["curator"]["precision"],
            r["verifier"]["recall"],
            r["verifier"]["f1"],
            r["verifier"]["typing_accuracy"],
            r["verifier"]["partial_typing_accuracy"],
            r["verifier"]["graph_similarity"], 

            r["curator"]["precision"],
            r["curator"]["recall"],
            r["curator"]["f1"],
            r["curator"]["typing_accuracy"],
            r["curator"]["partial_typing_accuracy"],
            r["curator"]["graph_similarity"] 


        ])

# ---------- SAVE SUMMARY CSV ----------
with open(f"{output_dir}/summary_{NUM_DOCS}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "precision", "recall", "f1", "typing", "partial_typing", "graph"])
    writer.writerow(["extractor", P_ex, R_ex, F1_ex, Typ_ex, PTyp_ex, Graph_ex])
    writer.writerow(["verifier", P_cr, R_cr, F1_cr, Typ_cr, PTyp_cr, Graph_cr])
    writer.writerow(["curator", P_fn, R_fn, F1_fn, Typ_fn, PTyp_fn, Graph_fn])


print("\nSaved evaluation to:", output_dir)
