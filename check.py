import pandas as pd
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = Path("data")
FILEPATH = DATA_DIR / "hitl_green_100_qlora_structured_a1_1_2.csv"
HITL_PATH = DATA_DIR / "hitl.csv"
PARQUET_PATH = Path("patents_50k_green.parquet")
reviewed_path = FILEPATH.with_name(FILEPATH.stem + "_reviewed.csv")
# ----------------------------
# Load CSVs
# ----------------------------
file1 = pd.read_csv(FILEPATH)
human_in_the_loop = pd.read_csv(HITL_PATH)

# Sanity check: same length/order assumption
if len(file1) != len(human_in_the_loop):
    raise ValueError(
        f"Length mismatch: file1 has {len(file1)} rows, "
        f"hitl.csv has {len(human_in_the_loop)} rows. "
        "If order/length differs, you must merge on an ID instead."
    )

# Add HITL columns
required_hitl_cols = ["human_label_is_green", "human_notes"]
missing = [c for c in required_hitl_cols if c not in human_in_the_loop.columns]
if missing:
    raise KeyError(f"hitl.csv is missing required columns: {missing}")

file1["human_label_is_green"] = human_in_the_loop["human_label_is_green"].values
file1["human_notes"] = human_in_the_loop["human_notes"].values

# Only save if the reviewed file doesn't already exist
if not reviewed_path.exists():
    file1.to_csv(reviewed_path, index=False)
    print(f"Saved reviewed file to: {reviewed_path}")
else:
    print(f"Reviewed file already exists: {reviewed_path}")

# ----------------------------
# Load parquet + filter pool split
# ----------------------------
parquet_file = pd.read_parquet(PARQUET_PATH)

pool_data = parquet_file.loc[parquet_file["split"] == "pool_unlabeled"].copy()
pool_data = pool_data.rename(columns={"id": "doc_id"})

# Ensure required columns exist
for col in ["doc_id", "is_green_silver"]:
    if col not in pool_data.columns:
        raise KeyError(f"Parquet pool data is missing required column: {col}")

# Merge is_green_silver onto file1
file1 = file1.merge(pool_data[["doc_id", "is_green_silver"]], on="doc_id", how="left")

# Warn if merge didn't find matches
missing_is_green = file1["is_green_silver"].isna().sum()
if missing_is_green:
    print(f"WARNING: {missing_is_green} rows have missing is_green_silver after merge (doc_id not found in pool_data).")

# ----------------------------
# Consistency checks
# ----------------------------
file1["llm_green_suggested_match"] = file1["llm_green_suggested"] == file1["is_green_silver"]
file1["human_in_the_loop_match"] = file1["human_label_is_green"] == file1["is_green_silver"]

file1_llm_matches = int(file1["llm_green_suggested_match"].sum())
file1_human_matches = int(file1["human_in_the_loop_match"].sum())

print(f"File 1 - LLM Green Suggested Matches: {file1_llm_matches}")
print(f"File 1 - Human in the Loop Matches: {file1_human_matches}")

file1_llm_matches_percentage = file1_llm_matches / len(file1) * 100
file1_human_matches_percentage = file1_human_matches / len(file1) * 100

print(f"File 1 - LLM Green Suggested Matches Percentage: {file1_llm_matches_percentage:.2f}%")
print(f"File 1 - Human in the Loop Matches Percentage: {file1_human_matches_percentage:.2f}%")

# ----------------------------
# Discrepancies: human vs llm suggestion
# ----------------------------
discrepancies_file1 = file1[file1["human_label_is_green"] != file1["llm_green_suggested"]]

print(f"File 1 - Discrepancies between Human Label and LLM Suggestion: {len(discrepancies_file1)}")
print("Discrepancies in File 1:")

cols_to_print = ["doc_id", "human_label_is_green", "llm_green_suggested", "is_green_silver", "human_notes"]
for col in cols_to_print:
    if col not in discrepancies_file1.columns:
        raise KeyError(f"Expected column missing from file1: {col}")

for _, row in discrepancies_file1.iterrows():
    print(
        f"ID: {row['doc_id']}, "
        f"Human Label: {row['human_label_is_green']}, "
        f"LLM Suggestion: {row['llm_green_suggested']}, "
        f"Is Green Silver(Y02): {row['is_green_silver']}, "
        f"Human Notes: {row['human_notes']}"
    )