import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def augment_with_clean_samples(df, num_samples=1000):
    import random
    clean_templates = [
        "Is this still available?",
        "Can I see more photos?",
        "How much does it cost?",
        "What’s the condition of the item?",
        "Where are you located?",
        "Is shipping available?",
        "Looks good. Can I pick it up tomorrow?",
        "Do you have other colors?",
        "I'm interested. Please tell me more.",
        "Can I schedule a viewing?"
    ]
    synthetic_clean = pd.DataFrame({
        "message": random.choices(clean_templates, k=num_samples),
        "label": [0] * num_samples
    })
    return pd.concat([df, synthetic_clean], ignore_index=True).sample(frac=1, random_state=42)
    
def train_and_save_model():
    # Load the labeled data
    df = pd.read_csv("data/processed/final_dataset.csv").drop_duplicates()
    df = augment_with_clean_samples(df, num_samples=1000)
    # Clean and validate label column
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(["0", "1"])]
    df["label"] = df["label"].astype(int)

    # Show label distribution
    print("✅ Cleaned label distribution:")
    print(df["label"].value_counts())
    print("✅ Dataset preview:")
    print(df.sample(5))
    print("Total rows:", len(df))

    # Prepare messages and labels
    messages = df["message"]
    labels = df["label"].astype(int)

    # Print label distribution
    label_counts = labels.value_counts()
    print("Label distribution before split:")
    print(label_counts)

    if label_counts.min() < 2:
        print("⚠️ Not enough samples in one of the classes for stratification. Proceeding without stratify.")
        stratify = None
    else:
        stratify = labels

    # Split before resampling
    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42, stratify=stratify
    )

    print("✅ Test set distribution:")
    print(y_test.value_counts())

    # Combine into DataFrame
    train_df = pd.DataFrame({"message": X_train, "label": y_train})

    # Handle class imbalance on training set
    from sklearn.utils import resample
    majority = train_df[train_df.label == 1]
    minority = train_df[train_df.label == 0]

    if len(minority) > 0 and len(minority) < len(majority):
        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=42
        )
        balanced_train_df = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)
        print("✅ Balanced training set:")
        print(balanced_train_df['label'].value_counts())
    else:
        print("⚠️ Skipping resampling on training data.")
        balanced_train_df = train_df

    # Final train sets
    X_train = balanced_train_df["message"]
    y_train = balanced_train_df["label"]

    # Vectorize text
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/contact_blocker.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("✅ Model and vectorizer saved to /models/")

if __name__ == "__main__":
    train_and_save_model()