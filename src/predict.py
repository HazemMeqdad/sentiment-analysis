from joblib import load
from clean_text import clean_arabic_text

# Load saved model + vectorizer
model = load("../models/sentiment_model.pkl")
tfidf = load("../models/tfidf_vectorizer.pkl")

def predict_sentiment(text):
    cleaned = clean_arabic_text(text)
    vec = tfidf.transform([cleaned])
    return model.predict(vec)[0]

if __name__ == "__main__":
    # Demo predictions
    examples = [
        "الخدمة ممتازة جداً",
        "التطبيق سيء جداً وما بنصح فيه",
        "عادي ما في شي مميز"
    ]

    for e in examples:
        print(f"{e} → {predict_sentiment(e)}")
