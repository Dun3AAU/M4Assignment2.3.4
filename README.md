# Green Patent Classification Pipeline

A multi-stage machine-learning pipeline for classifying patent claims as **green technology (Y02)** or **not green**, using a combination of active learning, large language models, and fine-tuned transformers.

---

## Project Overview

This project implements an end-to-end pipeline that:

1. Builds a balanced 50 k patent dataset from the 1.5 M-claim `AI-Growth-Lab/patents_claims_1.5m_traim_test` dataset.
2. Trains a **baseline PatentSBERTa + Logistic Regression** classifier on *silver* labels derived from Y02* IPC codes.
3. Selects the **100 highest-uncertainty** pool examples and exports them for **Human-in-the-Loop (HITL)** review.
4. Uses an **LLM Judge** (Llama-3.1-8B via vLLM) to pre-annotate the HITL batch.
5. Runs a **Multi-Agent System (MAS)** – an Advocate and a Skeptic – to debate each claim before a final verdict.
6. **Fine-tunes PatentSBERTa** with QLoRA on the combined silver + gold-100 dataset.
7. Evaluates all runs and logs results to `output/evaluation_log.csv`.

---

## Repository Structure

```
.
├── data/                                              # All CSV data files
│   ├── hitl.csv                                       # Raw human labels (doc_id, human_label_is_green, human_notes)
│   ├── hitl_green_100.csv                             # 100 high-uncertainty pool examples exported for review
│   ├── hitl_green_100_mas_output_286940.csv           # MAS pipeline output for the 100 examples
│   └── hitl_green_100_qlora_structured_a1_1_2_reviewed.csv  # Final reviewed file (LLM + human labels merged)
│
├── patents_50k_green.parquet                          # Balanced 50 k dataset (train/eval/pool splits)
│
├── creatingData.py                   # Step 1 – build & save patents_50k_green.parquet
├── baselineTrainEvalExport.py        # Step 2 – baseline model + HITL export to data/hitl_green_100.csv
├── llmJudge.py                       # Step 3a – LLM judge v1 (permissive prompt)
├── llmJudge.2.py                     # Step 3b – LLM judge v2 (conservative prompt)
├── llmJudge.3.py                     # Step 3c – LLM judge v3 (balanced prompt)
├── llmJudgeQLoRA1.py                 # Step 3d – LLM judge with QLoRA-structured output
├── masPipelineWithUtilFast.py        # Step 4  – Multi-Agent System (Advocate + Skeptic + Judge)
├── masPipelineWithUtilFast.sh        # Shell launcher for the MAS pipeline
├── check.py                          # Step 5  – merge human labels; consistency checks
├── FTPatentSBERTA.py                 # Step 6  – fine-tune PatentSBERTa (silver + gold)
├── qLoRAAdvocate.py                  # QLoRA Advocate agent
├── qLoRASkeptic.py                   # QLoRA Skeptic agent
├── eval.sh                           # Evaluation shell script
├── llmjudge.sh                       # Shell launcher for LLM judge
├── pyproject.toml                    # Project dependencies (uv)
└── .python-version                   # Python version pin (3.12)
```

---

## Pipeline Steps

### Step 1 – Create the Dataset
```bash
python creatingData.py
```
Downloads the patent claims dataset, creates a **balanced 50 k sample** (25 k green / 25 k non-green), and splits it into `train_silver` (70 %), `eval_silver` (15 %), and `pool_unlabeled` (15 %). Saves to `patents_50k_green.parquet`.

---

### Step 2 – Baseline Model & HITL Export
```bash
python baselineTrainEvalExport.py
```
- Encodes text with the frozen **PatentSBERTa** model.
- Trains a **Logistic Regression** classifier on the silver training set.
- Evaluates on `eval_silver` and appends metrics to `output/evaluation_log.csv`.
- Exports the **100 highest-uncertainty** pool examples (highest `u = 1 − 2|p_green − 0.5|`) to `data/hitl_green_100.csv`.

---

### Step 3 – LLM Judge
Start a [vLLM](https://github.com/vllm-project/vllm) server with `meta-llama/Llama-3.1-8B-Instruct`, then run one of:

```bash
python llmJudge.py          # v1 – uses structured output via LangChain
python llmJudge.2.py        # v2 – conservative prompt, JSON mode
python llmJudge.3.py        # v3 – balanced/expert prompt, JSON mode
python llmJudgeQLoRA1.py    # v4 – structured output written for QLoRA downstream use
```

Each script reads `data/hitl_green_100.csv` and writes an annotated CSV to `data/`.

---

### Step 4 – Multi-Agent System (MAS)
```bash
bash masPipelineWithUtilFast.sh
# or directly:
python masPipelineWithUtilFast.py --input data/hitl_green_100 --output data/hitl_green_100_mas_output.csv
```

Two LLM agents debate each patent claim:
- **Advocate** – argues *for* a Y02 (green) label.
- **Skeptic** – argues *against* a Y02 label.
- A **Judge** agent delivers the final verdict.

Requires a running vLLM server (or a QLoRA adapter via `--model` / `--adapter` flags).

---

### Step 5 – Human-in-the-Loop Review & Consistency Checks
1. Fill in `data/hitl.csv` with your human labels (`human_label_is_green`, `human_notes`).
2. Run:
```bash
python check.py
```
This merges the human labels onto the LLM-annotated file, runs consistency checks, and reports the percentage agreement between human labels and LLM suggestions.

---

### Step 6 – Fine-tune PatentSBERTa (Silver + Gold)
```bash
python FTPatentSBERTA.py
```
Fine-tunes the PatentSBERTa model on the combined silver training set **plus** the 100 gold-reviewed examples. Evaluates on `eval_silver` and the gold-100 split, appending results to `output/evaluation_log.csv`.

---

## Data Files (`data/`)

| File | Description |
|------|-------------|
| `hitl_green_100.csv` | 100 highest-uncertainty pool examples; columns: `doc_id`, `text`, `p_green`, `u`, `human_label_is_green`, `human_notes` |
| `hitl.csv` | Human annotations: `human_label_is_green` (0/1) and `human_notes` |
| `hitl_green_100_mas_output_286940.csv` | MAS pipeline output with Advocate/Skeptic reasoning and final verdict |
| `hitl_green_100_qlora_structured_a1_1_2_reviewed.csv` | Final reviewed file with LLM suggestions and human labels merged |

---

## Dependencies

Dependencies are managed with [uv](https://github.com/astral-sh/uv). Install them with:

```bash
uv sync
```

Key packages: `transformers`, `peft`, `trl`, `sentence-transformers`, `langchain-openai`, `datasets`, `scikit-learn`, `torch`, `vllm`.

---

## Requirements

- Python 3.12
- CUDA-capable GPU (recommended for fine-tuning and vLLM inference)
- A running **vLLM** server with `meta-llama/Llama-3.1-8B-Instruct` for Steps 3 & 4
