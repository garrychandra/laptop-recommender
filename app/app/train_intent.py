import os
import json
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ==========================================
# PREPROCESSING FUNCTION
# ==========================================
def preprocess_text(text):
    text = text.lower() # Lowercasing

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Remove emojis (basic pattern)
    emoji_pattern = re.compile("["
                           "\U0001F600-\U0001F64F"  # emoticons
                           "\U0001F300-\U0001F5FF"  # symbols & pictographs
                           "\U0001F680-\U0001F6FF"  # transport & map symbols
                           "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "\U00002702-\U000027B0"
                           "\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Remove special characters (punctuation and other non-alphanumeric except space)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove any remaining non-alphanumeric chars not covered by string.punctuation

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
    
# ==========================================
# PERSIAPAN DATA & TRAINING MODEL
# ==========================================
DATA_PATH = os.path.join("data", "intent_id.json")

# Load dataset
with open(DATA_PATH, encoding="utf-8") as f:
    training_data = json.load(f)

# Preprocess training texts
texts = [preprocess_text(data["text"]) for data in training_data]
labels = [data["label"] for data in training_data]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)

# Simpan model (Opsional, untuk arsip)
os.makedirs("models", exist_ok=True)
with open("models/intent_model_simple.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model berhasil dilatih dan siap digunakan.\n")

