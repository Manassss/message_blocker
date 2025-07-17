import pandas as pd
import random

clean_phrases = [
    "This looks good.",
    "What’s the price?",
    "How soon can I get it?",
    "Nice product!",
    "Can you show more photos?",
    "I’m interested in this item.",
    "Do you have it in stock?",
    "Does it come in other colors?",
    "Where do you ship from?",
    "What are the dimensions?"
]

def generate_clean_dataset(n=100000, seed=1042):
    random.seed(seed)
    clean_data = {
        "message": [random.choice(clean_phrases) for _ in range(n)],
        "label": [0] * n
    }
    df_clean = pd.DataFrame(clean_data)
    df_clean.to_csv("data/processed/generated_clean.csv", index=False)
    print("✅ Clean data saved to data/processed/generated_clean.csv")

if __name__ == "__main__":
    generate_clean_dataset()