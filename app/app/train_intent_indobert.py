from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import json
import os

DATA_PATH = os.path.join("data", "intent_id.json")
dataset = load_dataset("json", data_files=DATA_PATH)

label_set = sorted(list(set([d["intent"] for d in dataset["train"]])))
label2id = {label: i for i, label in enumerate(label_set)}
id2label = {i: label for label, i in label2id.items()}

def encode(batch):
    batch["labels"] = [label2id[label] for label in batch["intent"]]
    return tokenizer(batch["text"], truncation=True, max_length=512)

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModelForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p1",
    num_labels=len(label_set),
    id2label=id2label,
    label2id=label2id,
)

dataset = dataset.map(encode, batched=True)

training_args = TrainingArguments(
    output_dir="models/indobert_intent",
    eval_strategy="epoch",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    logging_steps=10,
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("models/indobert_intent")
tokenizer.save_pretrained("models/indobert_intent")
