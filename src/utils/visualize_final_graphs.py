import os
import json
import networkx as nx
from pyvis.network import Network

# ---------- Config ----------

MODEL_NAME = "ft_mistralai/Ministral-3-3B-Instruct-2512-BF16" 
INPUT_PATH = f"outputs/unseen/{MODEL_NAME}/typed_triplets.json"
VERIFIER_PATH = f"outputs/unseen/{MODEL_NAME}/verified_triplets.json"
CURATED_PATH = f"outputs/unseen/{MODEL_NAME}/final_triplets.json"  # <-- new
OUTPUT_DIR = f"outputs/unseen/{MODEL_NAME}/graphs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load data ----------
import json

with open(INPUT_PATH, "r") as f:
    extractor_json = json.load(f)
    extractor_data = {item["id"]: item for item in extractor_json["results"]}

with open(VERIFIER_PATH, "r") as f:
    verifier_json = json.load(f)
    verifier_data = {item["id"]: item for item in verifier_json["results"]}

with open(CURATED_PATH, "r") as f:
    curated_json = json.load(f)
    curated_data = {item["id"]: item for item in curated_json["results"]}

print(f"Loaded {len(extractor_data)} extractor entries, {len(verifier_data)} verifier entries, and {len(curated_data)} curated entries.\n")

# ---------- Colors ----------
edge_colors = {
    "gold_triplets": "#2ecc71",      # green
    "typed_triplets": "#e74c3c",      # red
    "reflected_triplets": "#3498db", # blue
    "curated_triplets": "#9b59b6"    # purple
}

node_colors = {
    "Attacker": "#ff6b6b",
    "Organization": "#4dabf7",
    "Malware Characteristic:Behavior": "#82c91e",
    "Malware Characteristic:Capability": "#f59f00",
    "Event": "#9775fa",
    "Infrastructure": "#495057",
    "Account": "#228be6",
    "Unknown": "#999999"
}

# ---------- Deduplication function ----------
def deduplicate_triplets(triplets):
    seen = set()
    deduped = []

    for t in triplets:
        subj = t.get("subject") or ""
        subj_type = t.get("subject_type") or ""
        obj = t.get("object") or ""
        obj_type = t.get("object_type") or ""
        rel = t.get("relation") or ""
        if not subj or not obj or not rel:
            continue

        key = (subj.strip().lower(), rel.strip().lower(), obj.strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append({
                "subject": subj,
                "subject_type": subj_type,
                "relation": rel,
                "object": obj,
                "object_type": obj_type
            })
    return deduped

# ---------- Process each sample ----------
for graph_id in extractor_data.keys():
    G = nx.DiGraph()
    item_ext = extractor_data[graph_id]
    item_ref = verifier_data.get(graph_id, {})
    item_cur = curated_data.get(graph_id, {})

    # Merge and deduplicate triplets from gold, predicted, reflected, and curated
    for label, data_item, key in [
        ("gold_triplets", item_ext, "gold_triplets"),
        ("typed_triplets", item_ext, "typed_triplets"),
        ("verified_triplets", item_ref, "verified_triplets"),
        ("curated_triplets", item_cur, "curated_triplets")
    ]:
        triplets = data_item.get(key, [])
        if not triplets:
            continue

        triplets = deduplicate_triplets(triplets)

        for t in triplets:
            subj = t.get("subject", "Unknown")
            obj = t.get("object", "Unknown")
            rel = t.get("relation", "Unknown")
            subj_type = t.get("subject_type", "Unknown")
            obj_type = t.get("object_type", "Unknown")

            subj_pref = f"{label}_{subj}"
            obj_pref = f"{label}_{obj}"

            # Add nodes
            G.add_node(subj_pref, label=subj, type=subj_type, origin=label)
            G.add_node(obj_pref, label=obj, type=obj_type, origin=label)

            # Add edge
            G.add_edge(subj_pref, obj_pref, relation=rel, origin=label)

    if len(G.nodes) == 0:
        print(f"⚠️ Graph {graph_id} is empty, skipping.")
        continue

    # ---------- Visualization ----------
    net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff", notebook=False)
    net.from_nx(G)

    # Node styling
    for node in net.nodes:
        origin = G.nodes[node["id"]].get("origin", "")
        ntype = G.nodes[node["id"]].get("type", "Unknown")
        node["color"] = node_colors.get(ntype, "#cccccc")
        node["title"] = f"{node['label']} ({ntype}) - {origin}"
        node["label"] = G.nodes[node["id"]].get("label", node["id"])

    # Edge styling
    for edge in net.edges:
        origin = G[edge["from"]][edge["to"]].get("origin", "")
        rel = G[edge["from"]][edge["to"]].get("relation", "")
        edge["color"] = edge_colors.get(origin, "#999999")
        edge["title"] = f"{rel} ({origin})"
        edge["label"] = rel

    # Layout
    net.repulsion(node_distance=250, spring_length=150)
    net.toggle_physics(True)

    # ---------- Save ----------
    output_path = f"{OUTPUT_DIR}/graph_{graph_id}.html"
    net.save_graph(output_path)
    print(f"✅ Saved: {output_path}")

print("\nAll graphs generated with gold, predicted, and curated triplets!")
