"""Train a TF-IDF + Multinomial Naive Bayes finance sentiment classifier."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from preprocess import clean_text

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "sample_data.csv"
MODEL_PATH = ROOT / "sentiment_model.pkl"
VECTORIZER_PATH = ROOT / "vectorizer.pkl"

VALID_LABELS = {"positive", "negative", "neutral"}

def load_and_preprocess_data(path=DATA_PATH):
    """Load text,label CSV and clean its text column."""
    df = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("sample_data.csv must contain 'text' and 'label' columns.")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[(df["text"].str.len() > 0) & df["label"].isin(VALID_LABELS)]
    if df["label"].nunique() < 2:
        raise ValueError("The dataset must contain at least two valid sentiment labels.")
    if df["label"].value_counts().min() < 2:
        raise ValueError("Each label needs at least two rows for a stratified train/test split.")
    return df

def train_sentiment_model(path=DATA_PATH):
    df = load_and_preprocess_data(path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("=" * 62)
    print("FinSight AI — Finance News Sentiment Model")
    print("=" * 62)
    print(f"Rows used: {len(df)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.3f}")
    print("\nClassification report:\n")
    print(report)
    print(f"\nSaved model: {MODEL_PATH.name}")
    print(f"Saved vectorizer: {VECTORIZER_PATH.name}")
    print("\nNote: sample_data.csv is a small teaching/demo dataset.")
    print("Its metrics are not evidence of real-world financial prediction performance.")
    return model, vectorizer, accuracy, report

if __name__ == "__main__":
    train_sentiment_model()
