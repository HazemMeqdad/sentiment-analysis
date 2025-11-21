import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from clean_text import clean_arabic_text

def train():
    print("Loading dataset...")
    df = pd.read_csv("../data/arabic_comments.csv")

    if "comment" not in df.columns or "sentiment" not in df.columns:
        raise Exception("CSV must contain 'comment' and 'sentiment' columns.")

    print("Cleaning text...")
    df["cleaned"] = df["comment"].apply(clean_arabic_text)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["sentiment"], test_size=0.2, random_state=42
    )

    print("Vectorizing text...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("Training model...")
    model = LogisticRegression(max_iter=300)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print("Saving model...")
    dump(model, "../models/sentiment_model.pkl")
    dump(tfidf, "../models/tfidf_vectorizer.pkl")

    print("Training complete!")

if __name__ == "__main__":
    train()
