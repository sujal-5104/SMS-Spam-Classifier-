import os
import pandas as pd
import pickle
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

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
# LOAD DATASET
# ============================================================

CSV_FILE = os.path.join(BASE_DIR, "spam.csv")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        "spam.csv not found. Please put spam.csv in the same "
        "folder as train.py."
    )

df = pd.read_csv(
    CSV_FILE,
    encoding="latin-1"
)

print("Dataset loaded successfully.")


# ============================================================
# HANDLE DATASET COLUMNS
# ============================================================

if "v1" in df.columns and "v2" in df.columns:

    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

elif "label" in df.columns and "message" in df.columns:

    df = df[["label", "message"]]

else:

    raise Exception(
        "Could not find message and label columns."
    )


# ============================================================
# CLEAN DATA
# ============================================================

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df.dropna(subset=["label"], inplace=True)

df["label"] = df["label"].astype(int)


# ============================================================
# TEXT PREPROCESSING
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


print("Processing text...")

df["transformed_text"] = df["message"].apply(
    transform_text
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X = df["transformed_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    max_features=5000
)

print("Fitting TF-IDF...")

X_train_tfidf = tfidf.fit_transform(
    X_train
)

X_test_tfidf = tfidf.transform(
    X_test
)


# ============================================================
# VERIFY VECTORIZER
# ============================================================

if not hasattr(tfidf, "vocabulary_"):

    raise Exception(
        "TF-IDF vectorizer was NOT fitted."
    )

print(
    "TF-IDF vocabulary size:",
    len(tfidf.vocabulary_)
)

print("TF-IDF vectorizer fitted successfully.")


# ============================================================
# TRAIN MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Logistic Regression model trained successfully.")


# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

print()
print("==============================")
print("MODEL RESULTS")
print("==============================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")


# ============================================================
# SAVE FITTED VECTORIZER
# ============================================================

VECTORIZER_FILE = os.path.join(
    BASE_DIR,
    "vectorizer.pkl"
)

with open(
    VECTORIZER_FILE,
    "wb"
) as file:

    pickle.dump(
        tfidf,
        file
    )


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

MODEL_FILE = os.path.join(
    BASE_DIR,
    "model.pickle"
)

with open(
    MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ============================================================
# VERIFY SAVED FILES
# ============================================================

with open(
    VECTORIZER_FILE,
    "rb"
) as file:

    saved_vectorizer = pickle.load(file)


with open(
    MODEL_FILE,
    "rb"
) as file:

    saved_model = pickle.load(file)


# ============================================================
# TEST SAVED MODEL
# ============================================================

test_message = transform_text(
    "Congratulations you have won a free prize"
)

test_vector = saved_vectorizer.transform(
    [test_message]
)

test_result = saved_model.predict(
    test_vector
)[0]


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("==============================")
print("FILES CREATED SUCCESSFULLY")
print("==============================")
print("vectorizer.pkl : OK")
print("model.pickle   : OK")

if test_result == 1:
    print("Test message: SPAM")
else:
    print("Test message: NOT SPAM")

print()
print("Everything is ready for Streamlit.")