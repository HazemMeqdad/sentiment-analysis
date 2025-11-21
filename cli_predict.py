from joblib import load
from src.clean_text import clean_arabic_text

# Load model and vectorizer
model = load("./models/sentiment_model.pkl")
tfidf = load("./models/tfidf_vectorizer.pkl")

def predict_sentiment(text):
    cleaned = clean_arabic_text(text)
    vector = tfidf.transform([cleaned])
    return model.predict(vector)[0]

if __name__ == "__main__":
    print("===============================================")
    print("   Arabic Sentiment Analysis (Command Line)")
    print("===============================================\n")

    while True:
        comment = input("اكتب تعليقك (أو اكتب exit للخروج): ")

        if comment.lower() == "exit":
            print("تم الإنهاء.")
            break

        prediction = predict_sentiment(comment)
        print(f"\n➡ التصنيف: {prediction}\n")
