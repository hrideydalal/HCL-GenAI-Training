# ============================================================
# Project Overview
# ============================================================
# In this task, I built a sentiment analysis pipeline using NLP techniques.
# First, I loaded the dataset and cleaned the text by converting it to lowercase,
# removing punctuation, removing stopwords, and applying lemmatization.
# Then, I converted the text into numerical features using two methods:
# Bag of Words and TF-IDF.
#
# After that, I trained two machine learning models:
# Logistic Regression and Naive Bayes.
#
# Finally, I evaluated the models using accuracy, classification report,
# and confusion matrix, and generated a word cloud to visualize the most
# frequent words in the dataset.
# ============================================================


# ============================================================
# Assignment 1 - Sentiment Analysis using NLP
# ============================================================

# =========================
# 0. Install Libraries (Colab only)
# =========================
!pip install -q wordcloud

# =========================
# 1. Import Libraries
# =========================
import nltk
import string
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from wordcloud import WordCloud

# =========================
# 2. Download Required NLTK Data
# =========================
# These resources are needed only once
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# =========================
# 3. Load Dataset
# =========================
# We are using the NLTK movie_reviews dataset
# It has two classes:
# pos = positive review
# neg = negative review

documents = []
labels = []

for category in movie_reviews.categories():
    for file_id in movie_reviews.fileids(category):
        review_text = movie_reviews.raw(file_id)
        documents.append(review_text)
        labels.append(category)

print("Total number of reviews:", len(documents))
print("Sample labels:", set(labels))

# =========================
# 4. Text Preprocessing
# =========================
# Steps:
# - convert text to lowercase
# - remove punctuation
# - tokenize text
# - remove stopwords
# - apply lemmatization

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize into words
    tokens = word_tokenize(text)

    # Remove stopwords and keep only alphabetic words
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into one string
    return " ".join(tokens)

# Apply preprocessing to all reviews
cleaned_documents = [preprocess_text(doc) for doc in documents]

print("Text preprocessing completed.")

# =========================
# 5. Visualize Most Frequent Words using Word Cloud
# =========================
all_text = " ".join(cleaned_documents)

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Most Frequent Words")
plt.show()

# =========================
# 6. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_documents,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# 7. Feature Extraction - Bag of Words
# ============================================================
# Bag of Words converts text into word-count vectors

bow_vectorizer = CountVectorizer(max_features=5000)

X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

print("Bag of Words feature extraction completed.")

# ============================================================
# 8. Model 1 - Logistic Regression using Bag of Words
# ============================================================
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_bow, y_train)

y_pred_lr = lr_model.predict(X_test_bow)

print("\nLogistic Regression with Bag of Words")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['neg', 'pos'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression (BoW)")
plt.show()

# ============================================================
# 9. Feature Extraction - TF-IDF
# ============================================================
# TF-IDF gives importance to useful words
# and reduces the weight of very common words

tfidf_vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("TF-IDF feature extraction completed.")

# ============================================================
# 10. Model 2 - Naive Bayes using TF-IDF
# ============================================================
nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

y_pred_nb = nb_model.predict(X_test_tfidf)

print("\nNaive Bayes with TF-IDF")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

cm_nb = confusion_matrix(y_test, y_pred_nb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=['neg', 'pos'])
disp.plot(cmap='Greens')
plt.title("Confusion Matrix - Naive Bayes (TF-IDF)")
plt.show()

# ============================================================
# 11. Final Accuracy Comparison
# ============================================================
lr_accuracy = accuracy_score(y_test, y_pred_lr)
nb_accuracy = accuracy_score(y_test, y_pred_nb)

print("\nFinal Comparison")
print("Logistic Regression Accuracy:", lr_accuracy)
print("Naive Bayes Accuracy:", nb_accuracy)

# ============================================================
# 12. Assignment Questions - Answers
# ============================================================

# 1. Why is preprocessing important in NLP?
# Preprocessing is important because raw text contains punctuation,
# stopwords, and unnecessary noise. Cleaning the text helps the model
# focus on meaningful words and improves performance.

# 2. Difference between BoW and TF-IDF?
# Bag of Words counts how many times each word appears in a document.
# TF-IDF also measures word importance by reducing the weight of very
# common words and giving more value to meaningful words.

# 3. Why do RNN/LSTM models perform better on text than traditional models?
# RNN and LSTM models process text in sequence, so they can understand
# word order and context better than traditional models like BoW or TF-IDF.

# 4. What are the limitations of simple sentiment analysis?
# Simple sentiment analysis may struggle with sarcasm, mixed opinions,
# context, slang, and sentences where both positive and negative feelings
# are present.
