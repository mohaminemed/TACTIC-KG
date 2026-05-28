import networkx as nx
from pyvis.network import Network
from pathlib import Path



# =========================================================
# COLORS
# =========================================================

EDGE_COLORS = {
    "gold_triplets": "#2ecc71",
    "extracted_triplets": "#e74c3c",    
    "typed_triplets": "#e74c3c",
    "verified_triplets": "#3498db",
    "curated_triplets": "#9b59b6"
}

NODE_COLORS = {
    "Attacker": "#ff6b6b",
    "Organization": "#4dabf7",
    "Malware": "#e03131",
    "Tool": "#f08c00",
    "Infrastructure": "#495057",
    "Event": "#9775fa",
    "Account": "#228be6",
    "Credential": "#12b886",
    "Location": "#15aabf",
    "Vulnerability": "#fa5252",
    "Time": "#868e96",
    "Unknown": "#999999",
}


# =========================================================
# DEDUPLICATION
# =========================================================

def deduplicate_triplets(triplets):

    seen = set()

    deduped = []

    for t in triplets:

        subj = t.get("subject", "")
        rel = t.get("relation", "")
        obj = t.get("object", "")

        subj_type = t.get("subject_type", "Unknown")
        obj_type = t.get("object_type", "Unknown")

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
                "relation": rel,
                "object": obj,
                "subject_type": subj_type,
                "object_type": obj_type
            })

    return deduped


# =========================================================
# GRAPH CREATION
# =========================================================




def build_graph(triplets, graph_name="Graph", origin="graph", agent_name="Agent"):
    # 1. Create NetworkX graph
    G = nx.DiGraph()

    # store node types
    node_types = {}

    for t in triplets:
        s = t.get("subject")
        r = t.get("relation")
        o = t.get("object")

        if not s or not r or not o:
            continue

        s_type = t.get("subject_type", "Unknown")
        o_type = t.get("object_type", "Unknown")

        G.add_node(s)
        G.add_node(o)
        G.add_edge(s, o, label=r)

        node_types[s] = s_type
        node_types[o] = o_type

    # 2. PyVis network
    net = Network(height="750px", width="100%", directed=True, notebook=False)
    net.from_nx(G)

    # 3. Apply node colors based on type
    for node in net.nodes:
        node_id = node["id"]

        n_type = node_types.get(node_id, "Unknown")
        node["color"] = NODE_COLORS.get(n_type, NODE_COLORS["Unknown"])

        # optional: improve readability
        node["font"] = {"color": "#333333"}

    # 4. (Optional) edge styling by relation category
    for edge in net.edges:
        label = edge.get("label", "").lower()

        if agent_name.lower() == "extractor":
            edge["color"] = EDGE_COLORS["extracted_triplets"]
        elif agent_name.lower() == "typer":
            edge["color"] = EDGE_COLORS["typed_triplets"]
        elif agent_name.lower() == "verifier":
            edge["color"] = EDGE_COLORS["verified_triplets"]
        elif agent_name.lower() == "curator":
            edge["color"] = EDGE_COLORS["curated_triplets"]
        else:
            edge["color"] = EDGE_COLORS["gold_triplets"]

    # 5. Output directory
    output_dir = Path("graphs") / origin
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"{graph_name.replace(' ', '_')}.html"

    # 6. Save
    net.save_graph(str(file_path))

    if not file_path.exists():
        raise RuntimeError(f"Graph file was not created: {file_path}")

    return str(file_path.resolve())