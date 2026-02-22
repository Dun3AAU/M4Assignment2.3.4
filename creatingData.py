import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
def main():
    ds = load_dataset("AI-Growth-Lab/patents_claims_1.5m_traim_test", split="train")
    df = ds.to_pandas()

    #Creating a binary column for green tech based on the presence of Y02* codes
    y02_cols = [c for c in df.columns if str(c).startswith("Y02")]
    if len(y02_cols) > 0:
        df["is_green_silver"] = (df[y02_cols].sum(axis=1) > 0).astype(int)
    else:
        raise ValueError("No Y02* columns found. We need to inspect how green tech is encoded in this dataset.")
    
    #creating a balanced 50k sample dataset
    SEED = 42
    np.random.seed(SEED)
    green_silver_df = df[df["is_green_silver"] == 1].sample(n=25000, random_state=SEED)
    non_green_silver_df = df[df["is_green_silver"] == 0].sample(n=25000, random_state=SEED)
    balanced_df = pd.concat([green_silver_df, non_green_silver_df]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    #creating cplits and save as parquet files
    train = balanced_df.sample(frac=0.70, random_state=SEED)
    rest = balanced_df.drop(train.index)

    eval_ = rest.sample(frac=0.50, random_state=SEED)   # half of remaining = 15%
    pool  = rest.drop(eval_.index)                      # remaining = 15%

    train = train.copy(); eval_ = eval_.copy(); pool = pool.copy()
    train["split"] = "train_silver"
    eval_["split"]  = "eval_silver"
    pool["split"]   = "pool_unlabeled"

    final_df = pd.concat([train, pool, eval_], ignore_index=True)

    final_df.to_parquet("patents_50k_green.parquet", index=False)
    
    #print some info about the final dataset, to check that everything looks good
    print(final_df["split"].value_counts())
    print(final_df["is_green_silver"].value_counts())
    print(final_df.groupby("split")["is_green_silver"].value_counts())

    #and head of the final dataset
    print(final_df.head())

if __name__ == "__main__":
    main()
