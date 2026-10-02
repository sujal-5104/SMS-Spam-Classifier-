import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# -----------------------------
# Download required NLTK data
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# -----------------------------
# Initialize
# -----------------------------
ps = PorterStemmer()


# -----------------------------
# Text Transformation Function
# -----------------------------
def transform_text(text):
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Remove special characters
    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    # Remove stopwords and punctuation
    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    # Stemming
    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    # Convert list to string
    return " ".join(y)


# -----------------------------
# Load Model and Vectorizer
# -----------------------------
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pickle', 'rb'))
except FileNotFoundError:
    st.error("model.pickle or vectorizer.pkl file not found.")
    st.stop()


# -----------------------------
# Streamlit App
# -----------------------------
st.title("📩 SMS Spam Classifier")

st.write("Enter an SMS message below to check whether it is Spam or Not Spam.")

input_sms = st.text_area(
    "Enter The Message",
    placeholder="Example: Congratulations! You have won a free prize..."
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter an SMS message.")
    else:

        # 1. Preprocess
        transform_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transform_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display Result
        if result == 1:
            st.error("🚨 SPAM")
        else:
            st.success("✅ NOT SPAM")