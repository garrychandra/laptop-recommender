from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os
from pathlib import Path
import pickle
import re
import string

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the simple intent model (pickle)
INTENT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "intent_model_simple.pkl")
with open(INTENT_MODEL_PATH, "rb") as f:
    intent_model = pickle.load(f)

def preprocess_text(text):
    """Preprocess text for intent prediction"""
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    emoji_pattern = re.compile("["
                           "\U0001F600-\U0001F64F"
                           "\U0001F300-\U0001F5FF"
                           "\U0001F680-\U0001F6FF"
                           "\U0001F1E0-\U0001F1FF"
                           "\U00002702-\U000027B0"
                           "\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_intent(text: str):
    """Predict intent using the simple sklearn model"""
    preprocessed = preprocess_text(text)
    return intent_model.predict([preprocessed])[0]

# Load NER model (IndoBERT)
NER_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "indobert_ner")
ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH, local_files_only=True, trust_remote_code=False)
ner_model     = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH, local_files_only=True, trust_remote_code=False).to(device)
ner_id2label  = ner_model.config.id2label

def predict_entities(text: str) -> dict:
    tokens     = ner_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = ner_model(**tokens).logits
    predictions = outputs.argmax(dim=-1)[0].cpu().numpy()
    tokens_ids  = tokens["input_ids"][0].cpu().numpy()
    words       = ner_tokenizer.convert_ids_to_tokens(tokens_ids)
    ents        = {}
    for idx,pred in enumerate(predictions):
        label     = ner_id2label[int(pred)]
        word      = words[idx]
        if label != "O":
            ents.setdefault(label, []).append(word)
    return ents

# make analyze() combine both
def analyze(text: str):
    intent   = predict_intent(text)
    entities = predict_entities(text)
    return intent, entities
