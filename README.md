# 📩 SMS Spam Classifier

A Machine Learning web application that classifies SMS messages as **Spam** or **Not Spam** using **TF-IDF Vectorization** and **Logistic Regression**. The application is built with Python and Streamlit and deployed online using Render.

## 🚀 Live Demo

👉 [Try the SMS Spam Classifier](https://sms-spam-classifier-g1z4.onrender.com)

---

## 📌 Project Overview

SMS spam is a common problem where unwanted promotional, fraudulent, or suspicious messages are sent to users.

This project uses Natural Language Processing (NLP) and Machine Learning to automatically analyze an SMS message and predict whether it is:

* 🚨 **SPAM**
* ✅ **NOT SPAM**

The user simply enters an SMS message into the web application and receives an instant prediction.

---

## 🧠 Technologies Used

* 🐍 Python
* 📊 Pandas
* 🔢 NumPy
* 🧠 Scikit-learn
* 📝 NLTK
* 📈 TF-IDF Vectorization
* 🤖 Logistic Regression
* 🌐 Streamlit
* 🚀 Render
* 🔧 Git & GitHub

---

## ⚙️ How It Works

The project follows these main steps:

1. Load the SMS dataset.
2. Clean and preprocess the text.
3. Convert text to lowercase.
4. Tokenize the SMS.
5. Remove punctuation and unnecessary words.
6. Remove English stopwords.
7. Apply stemming using Porter Stemmer.
8. Convert processed text into numerical features using TF-IDF.
9. Train a Logistic Regression model.
10. Save the trained model and TF-IDF vectorizer using Pickle.
11. Load the saved files in the Streamlit application.
12. Predict whether a new SMS is Spam or Not Spam.

---

## 📂 Project Structure

```text
SMS-Spam-Classifier/
│
├── app.py
├── train.py
├── spam.csv
├── model.pickle
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Machine Learning Model

### TF-IDF Vectorizer

TF-IDF (Term Frequency–Inverse Document Frequency) converts SMS text into numerical features that can be understood by the machine learning model.

The project uses:

```python
TfidfVectorizer(max_features=5000)
```

### Logistic Regression

The classification model used is:

```python
LogisticRegression(max_iter=1000)
```

The model predicts:

```text
0 → NOT SPAM
1 → SPAM
```

---

## 🖥️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/sujal-5104/SMS-Spam-Classifier-.git
```

### 2. Open the project

```bash
cd SMS-Spam-Classifier-
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the model

```bash
python train.py
```

This creates:

```text
model.pickle
vectorizer.pkl
```

### 7. Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🌐 Deployment

This project is deployed using **Render**.

### Render Build Command

```bash
pip install -r requirements.txt
```

### Render Start Command

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### Live Application

👉 https://sms-spam-classifier-g1z4.onrender.com

---

## 📦 Requirements

The main Python libraries used in this project are:

```text
pandas
numpy
nltk
scikit-learn
streamlit
```

---

## 🧪 Example

### Input

```text
Congratulations! You have won a free prize. Call now to claim your reward!
```

### Output

```text
🚨 SPAM MESSAGE
```

Another example:

### Input

```text
Hey, are you coming to college today?
```

### Output

```text
✅ NOT SPAM
```

---

## 🎯 Project Goals

* Detect unwanted SMS messages automatically.
* Apply NLP techniques to real-world text data.
* Build an end-to-end Machine Learning project.
* Deploy the ML model as a web application.
* Provide a simple and user-friendly interface.

---

## 🔮 Future Improvements

* Add more SMS datasets.
* Improve model accuracy.
* Compare multiple classification algorithms.
* Add prediction probability.
* Add a message history feature.
* Improve the Streamlit UI.
* Deploy an API version of the model.

---

## 👨‍💻 Author

**Sujal-5104**

### ⭐ If you find this project useful, consider giving the repository a star!

## 📜 License

This project is created for educational and learning purposes.
