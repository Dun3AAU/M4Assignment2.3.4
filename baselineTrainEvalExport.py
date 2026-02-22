import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import os

os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)

# 1. Load the working file
df = pd.read_parquet('patents_50k_green.parquet')

# Separate train and eval splits
df_train = df[df['split'] == 'train_silver']
df_eval = df[df['split'] == 'eval_silver']

# 2. Load the frozen PatentSBERTa model
print("Loading PatentSBERTa model...")
model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa")

# 3. Generate Embeddings (Frozen by default in inference mode)
print("Encoding training data...")
X_train_emb = model.encode(df_train['text'].tolist(), show_progress_bar=True, batch_size=32)
y_train = df_train['is_green_silver'].tolist()

print("Encoding evaluation data...")
X_eval_emb = model.encode(df_eval['text'].tolist(), show_progress_bar=True, batch_size=32)
y_eval = df_eval['is_green_silver'].tolist()

# 4. Train the baseline classifier
print("Training Logistic Regression classifier...")
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_emb, y_train)

# 5. Evaluate the model
print("Evaluating on eval_silver...")
y_pred = clf.predict(X_eval_emb)

precision = precision_score(y_eval, y_pred)
recall = recall_score(y_eval, y_pred)
f1 = f1_score(y_eval, y_pred)

print(f"\n--- Baseline Results ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


#Saving evals
eval_results = {
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'run_type': 'baseline', # You can change this manually for future script iterations
    'precision': round(precision, 4),
    'recall': round(recall, 4),
    'f1_score': round(f1, 4)
}

results_path = 'output/evaluation_log.csv'
results_df = pd.DataFrame([eval_results])

if os.path.exists(results_path):
    results_df.to_csv(results_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(results_path, mode='w', header=True, index=False)
    
print(f"Evaluation metrics appended to {results_path}")


#Creating the HITL export for the pool_unlabeled split

# 1. Isolate the pool_unlabeled split
df_pool = df[df['split'] == 'pool_unlabeled'].copy()

# 2. Extract embeddings for the unlabeled pool
# (Note: This will take a moment depending on the size of the pool)
print("Encoding unlabeled pool data...")
X_pool_emb = model.encode(df_pool['text'].tolist(), show_progress_bar=True, batch_size=32)

# 3. Compute p_green (predicted probabilities)
# .predict_proba() returns an array of shape (n_samples, n_classes). 
# Index 1 corresponds to the positive class (is_green_silver = 1).
print("Predicting probabilities...")
p_green = clf.predict_proba(X_pool_emb)[:, 1]
df_pool['p_green'] = p_green

# 4. Compute the uncertainty score 'u'
df_pool['u'] = 1 - 2 * np.abs(df_pool['p_green'] - 0.5)

# 5. Select the top 100 highest-risk (highest uncertainty) examples
# We sort descending by 'u' and take the head(100)
df_high_risk = df_pool.sort_values(by='u', ascending=False).head(100).copy()

# 6. Prepare the HITL export dataframe
# Note: Adjust 'publication_number' to whatever ID column exists in the dataset.
# The prompt asks for 'doc_id' and 'text', so we will rename columns to match perfectly.
df_export = pd.DataFrame({
    'doc_id': df_high_risk['id'],
    'text': df_high_risk['text'],
    'p_green': df_high_risk['p_green'],
    'u': df_high_risk['u'],
    'human_label_is_green': '', # Empty column for human labeling
    'human_notes': ''           # Optional empty column for annotator comments
})

# 7. Export to CSV
export_filename = 'data/hitl_green_100.csv'
df_export.to_csv(export_filename, index=False)
print(f"\nSuccessfully exported 100 high-risk examples to {export_filename}")