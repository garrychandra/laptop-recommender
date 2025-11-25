import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = os.path.join("data", "intent_id.json")

# Load dataset
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

texts = [d["text"] for d in data]
labels = [d["intent"] for d in data]

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# Model Training
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, clf.predict(X_test)))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/intent_vectorizer.pkl")
joblib.dump(clf, "models/intent_model.pkl")
print("Model saved in models/")
