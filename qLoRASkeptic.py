import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# -------------------------
# Load Model + Tokenizer
# -------------------------

model_id = "unsloth/llama-3-8b-bnb-4bit"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = False  # Important for QLoRA

# -------------------------
# Prepare LoRA
# -------------------------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

model.print_trainable_parameters()  # sanity check

# -------------------------
# Load Silver Data
# -------------------------

df = pd.read_parquet("patents_50k_green.parquet")
train_df = df[df["split"] == "train_silver"].sample(3000, random_state=42)

def format_instruction(row):
    label = "Labeled green" if row["is_green_silver"] == 1 else "Labeled not green"
    return f"""### You are Agent 2 (The Skeptic). Your job is to argue AGAINST classifying the claim as Y02.
Look for greenwashing: vague benefits, generic optimization, ordinary industrial processes, no explicit environmental mechanism.
Use only the claim text; quote short snippets as evidence.
Return:
- reasons_not_y02: bullet list
- greenwashing_flags: bullet list (optional)
- reasoning: 3–6 sentences
- confidence: Low/Medium/High
- what_would_change_my_mind: 1–3 bullets (optional)

### Claim:
{row['text']}

### Response:
{label}"""

train_df["text"] = train_df.apply(format_instruction, axis=1)

dataset = Dataset.from_pandas(train_df[["text"]])

# -------------------------
# Training Arguments
# -------------------------

training_args = TrainingArguments(
    output_dir="./qlora_patent_adapter_skeptic",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=100,
    learning_rate=2e-4,
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_32bit",  # important for QLoRA
    save_strategy="no",
)

# -------------------------
# SFT Trainer (0.8.6 version)
# -------------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    packing=True,
)

trainer.train()

model.save_pretrained("./qlora_patent_skeptic")
tokenizer.save_pretrained("./qlora_patent_skeptic")