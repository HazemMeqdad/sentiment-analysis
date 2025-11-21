import re
import arabicstopwords.arabicstopwords as stp

def clean_arabic_text(text):
    """
    Cleans Arabic text by removing non-Arabic characters,
    punctuation, English, numbers, and stopwords.
    """

    if not isinstance(text, str):
        return ""

    # Remove non-Arabic chars
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    cleaned_words = []
    for word in text.split():
        if word not in stp.stopwords_list():
            cleaned_words.append(word)

    return " ".join(cleaned_words)
