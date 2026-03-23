import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# STEP 1: Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# STEP 2: Add labels
fake["label"] = 0   # Fake news
real["label"] = 1   # Real news

# STEP 3: Combine datasets
data = pd.concat([fake, real])

# STEP 4: Keep only text and label columns
data = data[["text", "label"]]

# STEP 5: Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    return text

data["text"] = data["text"].apply(clean_text)

# STEP 6: Split dataset
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# STEP 7: Convert text into numbers (TF-IDF)
vectorizer = TfidfVectorizer()
Xv_train = vectorizer.fit_transform(X_train)
Xv_test = vectorizer.transform(X_test)

# STEP 8: Train model
model = LogisticRegression(max_iter=1000)
model.fit(Xv_train, y_train)

# STEP 9: Predict
predictions = model.predict(Xv_test)

# STEP 10: Results
print("Final Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))