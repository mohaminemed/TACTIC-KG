import os
import json
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
BASE_INPUT_DIR = "outputs/unseen"
BASE_OUTPUT_DIR = "evaluation_results"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Stage → (filename, field_name)
STAGES = {
    "IE": ("pred_triplets.json", "pred_triplets"),
    "ET": ("typed_triplets.json", "typed_triplets"),
    "VER": ("verified_triplets.json", "verified_triplets"),
    "CUR": ("final_triplets.json", "final_triplets"),
}

# -----------------------------
# FIND MODELS
# -----------------------------
agentic_dirs = [d for d in os.listdir(BASE_INPUT_DIR) if d.startswith("ft_")]

global_summary = []

for agentic_dir in agentic_dirs:
    agentic_path = os.path.join(BASE_INPUT_DIR, agentic_dir)

    for model_name in os.listdir(agentic_path):
        model_path = os.path.join(agentic_path, model_name)
        if not os.path.isdir(model_path):
            continue

        print(f"\n[INFO] Processing {model_name}")

        stage_counts = {}
        stage_times = {}
        total_triplets = 0
        num_samples = None
        total_time = None

        # -----------------------------
        # EXTRACT TRIPLETS & TIMINGS PER STAGE
        # -----------------------------
        for stage, (filename, field_name) in STAGES.items():
            file_path = os.path.join(model_path, filename)
            if not os.path.exists(file_path):
                print(f"[WARNING] Missing {file_path}")
                stage_counts[stage] = 0
                stage_times[stage] = 0
                continue

            with open(file_path, "r") as f:
                data = json.load(f)

            # Top-level timing info
            if total_time is None:
                total_time = data.get("total_time_sec", 0)
            if num_samples is None:
                num_samples = data.get("num_samples", len(data.get("results", [])))


            # Count triplets
            results = data.get("results", [])
            count = sum(len(sample.get(field_name, [])) for sample in results)
            stage_counts[stage] = count
            total_triplets += count

            # Use the avg time per stage if available
            stage_times[stage] = data.get("avg_time_per_sample_sec", 0)

        if total_triplets == 0:
            print(f"[WARNING] No triplets for {model_name}")
            continue

        total_time_ext =  stage_times.get("IE", 0) + stage_times.get("ET", 0) 
        
        # -----------------------------
        # CREATE DATAFRAME
        # -----------------------------
        row = {
            "IE": stage_times.get("IE", 0),
            "ET": stage_times.get("ET", 0),
            "VER": stage_times.get("VER", 0),
            "CUR": stage_times.get("CUR", 0),
            "E2E": total_time_ext or 0,
        }
        df = pd.DataFrame([row])

        # -----------------------------
        # SAVE PER-MODEL CSV
        # -----------------------------
        output_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(os.path.join(output_dir, "agentic_overhead_per_stage.csv"), index=False)

        # Summary CSV (mean, std, min, max) is trivial here, since we have 1 row only
        summary = [
            {"metric": col, "mean": df[col].mean(), "std": df[col].std(), "min": df[col].min(), "max": df[col].max()}
            for col in df.columns
        ]
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, "agentic_overhead_summary.csv"), index=False)

        # -----------------------------
        # ADD TO GLOBAL SUMMARY
        # -----------------------------
        global_summary.append({
            "model": model_name,
            "IE_mean": df["IE"].iloc[0],
            "ET_mean": df["ET"].iloc[0],
            "VER_mean": df["VER"].iloc[0],
            "CUR_mean": df["CUR"].iloc[0],
            "E2E_mean": df["E2E"].iloc[0],
        })

        print(f"[OK] Done {model_name}")

# -----------------------------
# SAVE GLOBAL SUMMARY
# -----------------------------
global_df = pd.DataFrame(global_summary)
global_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "agentic_global_summary.csv"), index=False)

print("\n[INFO] All models processed successfully.")