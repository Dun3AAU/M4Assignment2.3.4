import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime
import os
from datasets import Dataset

# Define the metrics function once at the top
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 1. Load data
df_full = pd.read_parquet("patents_50k_green.parquet")

# 2. Prepare Training Data (Silver + Gold)
train_silver = df_full[df_full["split"] == "train_silver"].copy()
train_silver['label'] = train_silver['is_green_silver']

gold_100 = pd.read_csv("data/hitl_green_100_qlora_structured_a1_1_2_reviewed.csv") 
gold_100['label'] = gold_100['llm_green_suggested']

train_combined = pd.concat([
    train_silver[['text', 'label']], 
    gold_100[['text', 'label']]
], ignore_index=True)

# 3. Prepare Eval Data
eval_df = df_full[df_full["split"] == "eval_silver"].copy()
eval_df['label'] = eval_df['is_green_silver']

# 4. Tokenization
model_name = "AI-Growth-Lab/PatentSBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = Dataset.from_pandas(train_combined[['text', 'label']]).map(tokenize_function, batched=True)
tokenized_eval = Dataset.from_pandas(eval_df[['text', 'label']]).map(tokenize_function, batched=True)
tokenized_gold = Dataset.from_pandas(gold_100[['text', 'label']]).map(tokenize_function, batched=True)

# 5. Define Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./patent-classifierWeek2",
    num_train_epochs=1,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch", # Evaluates once at the end of the epoch
    per_device_eval_batch_size=32,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
    save_total_limit=1,
    load_best_model_at_end=True,
    push_to_hub=False 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval, 
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 7. Train
trainer.train()

# 8. Final Evaluations
print("\nEvaluating on General Eval Split...")
results_general = trainer.evaluate(tokenized_eval)

print("Evaluating on Gold 100 Split...")
results_gold = trainer.evaluate(tokenized_gold)

# 9. Logging
log_file = "output/evaluation_log.csv"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

new_entries = [
    {
        "timestamp": timestamp,
        "run_type": "QloraPatentSBERTa_FineTuned_Silver+Gold",
        "precision": round(results_general['eval_precision'], 4),
        "recall": round(results_general['eval_recall'], 4),
        "f1_score": round(results_general['eval_f1'], 4)
    },
    {
        "timestamp": timestamp,
        "run_type": "QloraPatentSBERTa_FineTuned_Gold100",
        "precision": round(results_gold['eval_precision'], 4),
        "recall": round(results_gold['eval_recall'], 4),
        "f1_score": round(results_gold['eval_f1'], 4)
    }
]

log_df = pd.DataFrame(new_entries)
if not os.path.exists("output"):
    os.makedirs("output")
log_df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

print(f"\n--- Final Results ---\nGeneral F1: {results_general['eval_f1']:.4f}\nGold 100 F1: {results_gold['eval_f1']:.4f}")