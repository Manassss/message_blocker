import pandas as pd

def merge_all():
    df_blocked_1 = pd.read_csv("data/processed/labeled_dataset.csv")
    df_blocked_2 = pd.read_csv("data/processed/labeled__dataset.csv")
    df_blocked_3 = pd.read_csv("data/processed/generated_restricted.csv")
    df_clean_1 = pd.read_csv("data/processed/generated_clean.csv")

    df_merged = pd.concat([df_blocked_1, df_blocked_2, df_blocked_3, df_clean_1], ignore_index=True)
    df_merged = df_merged.sample(frac=1, random_state=42)  # shuffle

    df_merged.to_csv("data/processed/final_dataset.csv", index=False)
    print("✅ Merged dataset saved to data/processed/final_dataset.csv")

if __name__ == "__main__":
    merge_all()