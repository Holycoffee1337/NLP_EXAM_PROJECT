from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
import matplotlib.pyplot as plt
from datasets import DatasetDict, Dataset
import evaluate
import numpy as np
import torch
import json
import os
import pandas as pd
import ast

training_args = {
    "output_dir": "finetuned_model",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.1,
    "report_to": "none",
    "save_total_limit": 2,  # How many checkpoints to keep during training.
}


def load_data_json(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    examples = []
    for entry in data:
        text = entry["data"]["text"]
        tokens = []
        labels = []
        current_index = 0

        for result in entry["predictions"][0]["result"]:
            start, end = result["value"]["start"], result["value"]["end"]
            label = result["value"]["labels"][0]

            # Add non-entity tokens before the current entity
            while current_index < start:
                token = text[current_index:start].strip()
                if token:
                    tokens.append(token)
                    labels.append("O")
                current_index = start

            # Add the entity token
            entity = text[start:end]
            tokens.append(entity)
            labels.append(label)
            current_index = end

        # Add remaining non-entity tokens
        remaining_text = text[current_index:].strip()
        if remaining_text:
            tokens.extend(remaining_text.split())
            labels.extend(["O"] * len(remaining_text.split()))

        examples.append({"tokens": tokens, "tags": labels})

    # Extract unique labels and add 'O'
    unique_labels = set(label for ex in examples for label in ex["tags"])
    unique_labels.add('O')
    global id2label, label2id
    id2label = {i: label for i, label in enumerate(sorted(unique_labels))}
    label2id = {label: i for i, label in id2label.items()}

    # Create Hugging Face Dataset
    dataset = Dataset.from_dict({
        "tokens": [ex["tokens"] for ex in examples],
        "tags": [ex["tags"] for ex in examples]
    })

    # Split into train and validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)  # 90% train, 10% validation
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
    

def load_data_csv(csv_path):
    examples = []
    
    # Read the CSV using pandas
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        text = row["text"]
        
        if not text:  # Check if the text is empty
            print(f"Skipping empty text at row {index}")
            continue

        try:
            # Safely evaluate the string as a Python literal (list of dictionaries)
            annotations = ast.literal_eval(text)
        except (ValueError, SyntaxError) as e:
            print(f"Error evaluating string at row {index}: {e}")
            print(f"Problematic text: {text}")
            continue

        tokens = []
        labels = []
        current_index = 0

        for annotation in annotations:
            start, end = annotation["start"], annotation["end"]
            label = annotation["labels"][0]  # Assuming single label for simplicity

            # Add non-entity tokens before the current entity
            while current_index < start:
                token = row["text"][current_index:start].strip()
                if token:
                    tokens.append(token)
                    labels.append("O")  # "O" for non-entity tokens
                current_index = start

            # Add the entity token
            entity = row["text"][start:end]
            tokens.append(entity)
            labels.append(label)
            current_index = end

        # Add remaining non-entity tokens
        remaining_text = row["text"][current_index:].strip()
        if remaining_text:
            tokens.extend(remaining_text.split())
            labels.extend(["O"] * len(remaining_text.split()))

        examples.append({"tokens": tokens, "tags": labels})

    # Extract unique labels and add 'O'
    unique_labels = set(label for ex in examples for label in ex["tags"])
    unique_labels.add('O')
    global id2label, label2id
    id2label = {i: label for i, label in enumerate(sorted(unique_labels))}
    label2id = {label: i for i, label in id2label.items()}

    # Create Hugging Face Dataset
    dataset = Dataset.from_dict({
        "tokens": [ex["tokens"] for ex in examples],
        "tags": [ex["tags"] for ex in examples]
    })

    # Split into train and validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)  # 90% train, 10% validation
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

    



def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:  # Special tokens
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Start of a new word
                label_ids.append(label2id.get(label[word_idx], label2id['O']))
            else:  # Continuation of a word
                label_ids.append(-100)  # Ignore sub-word tokens

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


seqeval = evaluate.load("seqeval")

def compute_metrics(predictions_and_labels):
    predictions, labels = predictions_and_labels
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Dataset Loading
csv_path = "NLP_EXAM_PROJECT\studio-label-annotation.csv"  # Update with your JSON file path
dataset = load_data_csv(csv_path)

model_checkpoint = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Trainer Configuration
args = TrainingArguments(
    **training_args,
    logging_dir=os.path.join(training_args["output_dir"], "logs"),  # Logs are saved in a folder in the output directory
    logging_strategy="epoch",
    logging_steps=20,  # Log every n steps
    push_to_hub=False  # Not using Hugging Face Hub
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],  # Add validation dataset
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Optional: Early stopping
)

model_save_path = os.path.join(training_args["output_dir"], "model")
trainer.save_model(model_save_path)

# Save Tokenizer
tokenizer.save_pretrained(model_save_path)
print(f"Training complete. Model and tokenizer saved!")

# Print dataset sizes
print(f"Training dataset size: {len(dataset['train'])}")
print(f"Validation dataset size: {len(dataset['validation'])}")

# Train and save model
train_result = trainer.train()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print("Training complete. Model and tokenizer saved!")