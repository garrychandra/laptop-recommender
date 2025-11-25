from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("models/indobert_intent")
model = AutoModelForSequenceClassification.from_pretrained("models/indobert_intent").to(device)

id2label = model.config.id2label

def predict_intent(text: str):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = logits.argmax(dim=-1).item()
    return id2label[predicted_id]

ner_tokenizer = AutoTokenizer.from_pretrained("models/indobert_ner")
ner_model     = AutoModelForTokenClassification.from_pretrained("models/indobert_ner").to(device)
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
