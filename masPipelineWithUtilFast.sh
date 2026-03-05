#!/bin/bash
#SBATCH --job-name=masPipelineWithUtilFast
#SBATCH --output=logs/%x/%x_output_%j.txt
#SBATCH --error=logs/%x/%x_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p "logs/${SLURM_JOB_NAME}"

# If your cluster needs HF auth for private models, keep this.


source .venv/bin/activate

# Optional: reduce tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# -----------------------------
# Easy-to-edit arguments
# -----------------------------
INPUT_PATH="data/hitl_green_100"                 # with or without .csv
OUTPUT_PATH="data/hitl_green_100_mas_output.csv" # script will suffix SLURM_JOB_ID automatically

BASE_MODEL_ID="unsloth/llama-3-8b-bnb-4bit"
ADVOCATE_ADAPTER_DIR="./qlora_patent_advocate"
SKEPTIC_ADAPTER_DIR="./qlora_patent_skeptic"

BATCH_SIZE=8

# Start safer (less truncation risk). If outputs look complete, lower later.
ADV_TOKENS=180
SKP_TOKENS=180
JUDGE_TOKENS=240

TEMPERATURE=0.2
TOP_P=0.9

# -----------------------------
# Run
# -----------------------------
python3 masPipelineWithUtilFast.py \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --base_model "${BASE_MODEL_ID}" \
  --adv_adapter "${ADVOCATE_ADAPTER_DIR}" \
  --skp_adapter "${SKEPTIC_ADAPTER_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --adv_max_new_tokens "${ADV_TOKENS}" \
  --skp_max_new_tokens "${SKP_TOKENS}" \
  --judge_max_new_tokens "${JUDGE_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}"
