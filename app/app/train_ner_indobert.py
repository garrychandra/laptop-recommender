from datasets import load_dataset, Sequence, ClassLabel
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import os
import evaluate

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "ner_id.json")
dataset = load_dataset("json", data_files=DATA_PATH)

# Split dataset: 80% train, 20% test
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

label_list = ["O", "B-BRAND", "I-BRAND", "B-MODEL", "I-MODEL", "B-RAM", "I-RAM", "B-STORAGE", "I-STORAGE", 
              "B-SCREEN_SIZE", "I-SCREEN_SIZE", "B-BUDGET", "I-BUDGET", "B-USAGE", "I-USAGE",
              "B-TOUCHSCREEN", "I-TOUCHSCREEN"]
label2id = {l:i for i,l in enumerate(label_list)}
id2label = {i:l for l,i in label2id.items()}

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, max_length=512)
    labels = []
    
    for i in range(len(examples["text"])):
        text = examples["text"][i]
        entities = examples["entities"][i]
        
        # Create character-level labels
        char_labels = ["O"] * len(text)
        for entity in entities:
            label = entity["label"]
            start = entity["start"]
            end = min(entity["end"], len(text))  # Bounds check
            # Mark first character with B- prefix
            if start < len(text):
                char_labels[start] = f"B-{label}"
            # Mark rest with I- prefix
            for j in range(start + 1, end):
                char_labels[j] = f"I-{label}"
        
        # Align with tokens
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Get character offset for this token
                char_span = tokenized_inputs.token_to_chars(i, token_idx)
                if char_span is not None:
                    label_ids.append(label2id[char_labels[char_span.start]])
                else:
                    label_ids.append(label2id["O"])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
model = AutoModelForTokenClassification.from_pretrained(
    "indobenchmark/indobert-base-p1",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

args = TrainingArguments(
    output_dir=os.path.join(SCRIPT_DIR, "models", "indobert_ner"),
    num_train_epochs=5,
    per_device_train_batch_size=4,
    eval_strategy="epoch",
    logging_steps=10,
    save_strategy="no"  # Disable checkpoint saving
)

metric = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_preds  = [[label_list[p] for (p,l) in zip(pred, lab) if l!=-100]
                   for pred,lab in zip(predictions, labels)]
    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
      "precision": results["overall_precision"],
      "recall": results["overall_recall"],
      "f1": results["overall_f1"]
    }

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
MODELS_NER_DIR = os.path.join(SCRIPT_DIR, "models", "indobert_ner")
trainer.save_model(MODELS_NER_DIR)
tokenizer.save_pretrained(MODELS_NER_DIR)

# Evaluate on test set and print results
print("\n" + "="*80)
print("FINAL TEST SET EVALUATION")
print("="*80)
test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print(f"\nTest Results:")
print(f"  Precision: {test_results['eval_precision']*100:.2f}%")
print(f"  Recall: {test_results['eval_recall']*100:.2f}%")
print(f"  F1 Score: {test_results['eval_f1']*100:.2f}%")
print("="*80)
