import os
import pickle
import string

import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# NLTK
# ============================================================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

MODEL_FILE = os.path.join(BASE_DIR, "model.pickle")
VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer.pkl")


with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


with open(VECTORIZER_FILE, "rb") as file:
    tfidf = pickle.load(file)


# ============================================================
# TEXT TRANSFORMATION
# ============================================================

ps = PorterStemmer()

stop_words = set(
    stopwords.words("english")
)


def transform_text(text):

    text = str(text).lower()

    tokens = nltk.word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word.isalnum()
    ]

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        ps.stem(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)


st.title("📩 SMS Spam Classifier")

st.write(
    "Enter an SMS message below to check whether it is "
    "Spam or Not Spam."
)


# ============================================================
# MESSAGE INPUT
# ============================================================

message = st.text_area(
    "Enter your SMS message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize!"
)


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🔍 Predict"):

    if message.strip() == "":
        st.warning("⚠️ Please enter an SMS message.")

    else:

        # Transform text
        transformed_message = transform_text(message)

        # Convert text into TF-IDF
        vector_input = tfidf.transform(
            [transformed_message]
        )

        # Prediction
        prediction = model.predict(
            vector_input
        )[0]

        # Display result
        if prediction == 1:

            st.error("🚨 SPAM MESSAGE")

            st.write(
                "This message is likely to be spam."
            )

        else:

            st.success("✅ NOT SPAM")

            st.write(
                "This message appears to be a legitimate SMS."
            )