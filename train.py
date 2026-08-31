import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

import pickle
import os


# 1. Load dataset
data = pd.read_csv("data/tickets.csv")

X = data["text"]
y = data["category"]


# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Create NLP pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])


# 4. Train model
model.fit(X_train, y_train)


# 5. Make predictions
y_pred = model.predict(X_test)


# 6. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")


print("\n===== MODEL RESULTS =====")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))


# 7. Save model
os.makedirs("model", exist_ok=True)

with open("model/ticket_classifier.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")
print("Location: model/ticket_classifier.pkl")