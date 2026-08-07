# Spam-mail-prediction

# 📧 Spam Mail Prediction Using Machine Learning

## 📌 Overview

Electronic mail is a cornerstone of modern digital communication, but it is heavily plagued by spam—bulk, unsolicited messages that often contain phishing attempts or fraudulent advertisements. This project implements a **Natural Language Processing (NLP)** and **Machine Learning-based classifier** to automatically read email text and predict whether an incoming message is **"Spam"** (malicious) or **"Ham"** (legitimate).

## 📊 Dataset

* **Source File:** `mail_data.csv`
* **Total Records:** 5,572 text messages
* **Features:** 2 columns (`Category`, `Message`)
* **Class Distribution:** 4,825 Ham (Legitimate) / 747 Spam (Malicious)

## 🛠️ Technologies & Libraries

* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning & NLP:** `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`, `train_test_split`, `accuracy_score`)
* **Environment:** Jupyter Notebook / Google Colab

## ⚙️ System Architecture & Workflow

1. **Data Ingestion:** Load the raw CSV dataset into a Pandas DataFrame.
2. **Data Cleaning:** Handle missing/null values by replacing them with empty strings.
3. **Label Encoding:** Map the categorical targets to binary integers (`Spam = 0`, `Ham = 1`).
4. **Data Splitting:** Divide the dataset into Training (80%) and Testing (20%) subsets to prevent model overfitting.
5. **Feature Extraction:** Use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert unstructured text strings into a structured matrix of numerical weights.
6. **Model Training:** Train a **Logistic Regression** classifier on the vectorized text data.
7. **Evaluation:** Test the model on unseen data using Accuracy Scores and Confusion Matrices.

## 📈 Results & Performance

The model is highly effective and lightweight, prioritizing the preservation of legitimate emails (0 False Positives).

* **Training Accuracy:** ~96.7%
* **Testing Accuracy:** ~96.6%
* **False Positives (Ham marked as Spam):** 0
* **False Negatives (Spam missed):** 37

> **Note:** The near-identical training and testing accuracies indicate that the model generalized perfectly to unseen data without overfitting.

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/spam-mail-prediction.git
cd spam-mail-prediction

```


2. **Install the required dependencies:**
Ensure you have Python installed, then run:
```bash
pip install pandas numpy scikit-learn

```


3. **Run the Project:**
Open the Jupyter Notebook (`Spam_Mail_Prediction.ipynb`) and execute the cells sequentially, or run the Python script directly.

## 💻 Usage / Predictive System

You can test the model on custom, real-world emails by modifying the `input_mail` variable in the script:

```python
# Provide your custom email text
input_mail = ["Congratulations! You've won a free ticket to the Bahamas. Click here to claim your prize."]

# The system vectorizes the text and predicts
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

if (prediction[0] == 1):
  print('Ham mail')
else:
  print('Spam mail')

```

## 🔮 Future Scope

* **Hyperparameter Tuning:** Experiment with `GridSearchCV` to optimize the `C` parameter in Logistic Regression and test different n-gram ranges in the TF-IDF vectorizer.
* **Deep Learning:** Apply Word Embeddings (Word2Vec) and recurrent neural networks (LSTMs) to capture the semantic context of sentences.
* **Web Deployment:** Wrap the predictive pipeline in a web framework (like Flask, Streamlit, or FastAPI) to create a user-friendly graphical interface.
