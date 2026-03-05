import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

"""
llmJudgeQLoRA1.py (rewritten)

This version MATCHES the current qLoRA1 training format:
- Prompt style: "### Instruction / ### Claim / ### Response:"
- Model output expected: "Yes" or "No" (no JSON)
- We then map Yes->1, No->0 for llm_green_suggested
- Rationale is not part of training, so we fill a simple placeholder.
"""

# -------------------------
# 1) Load the Fine-Tuned Model
# -------------------------
base_model_id = "unsloth/llama-3-8b-bnb-4bit"
adapter_path = "./qlora_patent_adapter1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading model and adapter...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# -------------------------
# 2) Prompt template (MATCH TRAINING)
# -------------------------
PROMPT_TEMPLATE = """### Instruction:
Classify whether the following patent claim belongs to Y02 (green technology).
Respond with Yes or No.

### Claim:
{claim}

### Response:
"""

# -------------------------
# 3) Parsing helpers
# -------------------------
YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
NO_RE = re.compile(r"\bno\b", re.IGNORECASE)

def parse_yes_no(text: str) -> int:
    """
    Returns:
      1 if model says Yes
      0 if model says No
    Raises:
      ValueError if cannot confidently parse.
    """
    if text is None:
        raise ValueError("Empty output")

    t = text.strip()

    # First-token check (most reliable when generation is short)
    first = re.split(r"\s+", t, maxsplit=1)[0].strip().strip('"\',.:;!?[](){}')
    if first.lower().startswith("yes"):
        return 1
    if first.lower().startswith("no"):
        return 0

    # Fallback: search anywhere (handles "Yes." / "No - ..." etc.)
    if YES_RE.search(t) and not NO_RE.search(t):
        return 1
    if NO_RE.search(t) and not YES_RE.search(t):
        return 0

    # If both appear, try to prefer the earliest occurrence
    yes_i = next((m.start() for m in YES_RE.finditer(t)), None)
    no_i = next((m.start() for m in NO_RE.finditer(t)), None)
    if yes_i is not None and no_i is not None:
        return 1 if yes_i < no_i else 0

    raise ValueError(f"Could not parse Yes/No from: {t!r}")

# -------------------------
# 4) Inference
# -------------------------
@torch.no_grad()
def predict_is_green(claim_text: str) -> tuple[int, str, str]:
    """
    Returns:
      (llm_green_suggested, llm_confidence, llm_rationale)

    Note:
      Training only taught Yes/No, so confidence + rationale are placeholders.
    """
    prompt = PROMPT_TEMPLATE.format(claim=claim_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=4,         # enough for "Yes" / "No" (+ punctuation)
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    try:
        y = parse_yes_no(gen_text)
        # placeholders (the model wasn't trained for these fields)
        confidence = "N/A"
        rationale = f"Model response: {gen_text.strip()}"
        return y, confidence, rationale
    except Exception:
        print(f"FAILED PARSE. RAW OUTPUT: {gen_text!r}")
        return 0, "N/A", "Parsing Error (expected Yes/No)"

# -------------------------
# 5) Process the Data
# -------------------------
input_path = "data/hitl_green_100.csv"
output_path = "data/hitl_green_100_qlora_structured_a1_1.csv"

df = pd.read_csv(input_path)

llm_green = []
llm_conf = []
llm_rat = []

for text in tqdm(df["text"], desc="QLoRA Yes/No Inference"):
    y, c, r = predict_is_green(text)
    llm_green.append(y)
    llm_conf.append(c)
    llm_rat.append(r)

df["llm_green_suggested"] = llm_green
df["llm_confidence"] = llm_conf
df["llm_rationale"] = llm_rat

# Keep the same column order you used before (if columns exist)
cols = [
    "doc_id", "text", "p_green", "u",
    "llm_green_suggested", "llm_confidence", "llm_rationale",
    "human_label_is_green", "human_notes"
]
existing_cols = [c for c in cols if c in df.columns]

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
df[existing_cols].to_csv(output_path, index=False)

print(f"Done! Results saved to {output_path}")