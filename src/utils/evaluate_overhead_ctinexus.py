import os
import json
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
BASE_INPUT_DIR = "outputs/unseen/ctinexus"
BASE_OUTPUT_DIR = "evaluation_results"

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# -----------------------------
# GET ALL JSON FILES
# -----------------------------
files = [f for f in os.listdir(BASE_INPUT_DIR) if f.endswith(".json")]
print(f"[INFO] Found {len(files)} files")

# For global summary across models
global_summary = []

# -----------------------------
# PROCESS EACH MODEL
# -----------------------------
for file_name in files:
    input_path = os.path.join(BASE_INPUT_DIR, file_name)

    # Extract model name from filename
    model_name = file_name.replace("_seen_test.json", "")
    output_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Processing model: {model_name}")

    # -----------------------------
    # LOAD JSON
    # -----------------------------
    with open(input_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])

    if len(results) == 0:
        print(f"[WARNING] No results for {model_name}")
        continue

    # -----------------------------
    # PER-DOC OVERHEAD
    # -----------------------------
    rows = []

    for sample in results:
        rt = sample.get("response_time", {})
        rows.append({
            "id": sample.get("id"),
            "source_id": sample.get("source_id"),
            "IE": rt.get("IE", 0),
            "ET": rt.get("ET", 0),
            "EA": rt.get("EA", 0),
            "LP": rt.get("LP", 0),
            "E2E": rt.get("E2E", 0),
        })

    df = pd.DataFrame(rows)

    # Save per-document CSV
    per_doc_path = os.path.join(output_dir, "overhead_per_doc.csv")
    df.to_csv(per_doc_path, index=False)

    # -----------------------------
    # PER-MODEL SUMMARY
    # -----------------------------
    summary_rows = []
    metrics = ["IE", "ET", "EA", "LP", "E2E"]

    for m in metrics:
        summary_rows.append({
            "metric": m,
            "mean": df[m].mean(),
            "std": df[m].std(),
            "min": df[m].min(),
            "max": df[m].max(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "overhead_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Add to global summary (model-level means)
    global_summary.append({
        "model": model_name,
        **{f"{m}_mean": df[m].mean() for m in metrics}
    })

    print(f"[OK] Saved results for {model_name}")

# -----------------------------
# GLOBAL SUMMARY ACROSS MODELS
# -----------------------------
if global_summary:
    global_df = pd.DataFrame(global_summary)
    global_summary_path = os.path.join(BASE_OUTPUT_DIR, "ctinexus_global_summary.csv")
    global_df.to_csv(global_summary_path, index=False)
    print(f"\n[INFO] Global per-model summary saved: {global_summary_path}")

print("\n[INFO] All models processed successfully.")