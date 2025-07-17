import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

def train_and_save_model():
    # Load the labeled data
    df = pd.read_csv("data/processed/labeled_dataset.csv")
    messages = df["message"]
    labels = df["label"]
    
    print(df['message'].drop_duplicates)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

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

    # Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/contact_blocker.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("✅ Model and vectorizer saved to /models/")

if __name__ == "__main__":
    train_and_save_model()