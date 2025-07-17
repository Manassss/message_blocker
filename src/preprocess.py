import pandas as pd
import os

def label_all_as_blocked():
    input_path = "data/raw/blocker_dataset.csv"
    output_path = "data/processed/labeled__dataset.csv"

    # Load messages
    df = pd.read_csv(input_path)

    # Add label column with all 1s (blocked)
    df["label"] = 1

    # Save processed file
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ All messages labeled as blocked and saved to {output_path}")

if __name__ == "__main__":
    label_all_as_blocked()