import networkx as nx
import re
from html import escape
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
    "Malware Characteristic": "#c86c8c",
    "Tool": "#f08c00",
    "Infrastructure": "#495057",
    "Event": "#9775fa",
    "Account": "#228be6",
    "Credential": "#12b886",
    "Location": "#15aabf",
    "Vulnerability": "#fa5252",
    "Time": "#57718b",
    "Unknown": "#999999",
}


def node_color(node_type):
        parent_type = node_type.split(":", 1)[0]
        return NODE_COLORS.get(node_type, NODE_COLORS.get(parent_type, NODE_COLORS["Unknown"]))


def build_legend(node_types):
        legend_items = []
        for node_type in sorted(set(node_types.values())):
                color = node_color(node_type)
                legend_items.append(
                        f'<div class="tactic-kg-legend-item">'
                        f'<span class="tactic-kg-legend-swatch" style="background:{color}"></span>'
                        f'<span>{escape(str(node_type))}</span>'
                        f"</div>"
                )

        return """
<aside class="tactic-kg-legend" aria-label="Node type legend">
    <strong>Node types</strong>
    {items}
</aside>
<style>
    .tactic-kg-legend {{
        position: fixed;
        top: 12px;
        right: 12px;
        z-index: 1000;
        max-width: 230px;
        padding: 10px 12px;
        border: 1px solid #d0d7de;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.96);
        color: #24292f;
        font: 13px/1.4 Arial, sans-serif;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
    }}
    .tactic-kg-legend-item {{
        display: flex;
        align-items: center;
        gap: 7px;
        margin-top: 5px;
        overflow-wrap: anywhere;
    }}
    .tactic-kg-legend-swatch {{
        flex: 0 0 12px;
        width: 12px;
        height: 12px;
        border: 1px solid rgba(0, 0, 0, 0.25);
        border-radius: 50%;
    }}
</style>
""".format(items="\n".join(legend_items))


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
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )
    net.from_nx(G)

    # 3. Apply node colors based on type
    for node in net.nodes:
        node_id = node["id"]

        n_type = node_types.get(node_id, "Unknown")
        node["color"] = node_color(n_type)

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

    html = file_path.read_text(encoding="utf-8")
    html = re.sub(
        r"^[^\n]*https://cdn\.jsdelivr\.net/npm/bootstrap[^\n]*\n?",
        "",
        html,
        flags=re.MULTILINE,
    )
    html = html.replace("</body>", build_legend(node_types) + "\n</body>")
    file_path.write_text(html, encoding="utf-8")

    if not file_path.exists():
        raise RuntimeError(f"Graph file was not created: {file_path}")

    return str(file_path.resolve())