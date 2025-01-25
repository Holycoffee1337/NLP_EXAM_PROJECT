from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import matplotlib.pyplot as plt
from datasets import DatasetDict, Dataset
import evaluate
import numpy as np
import torch
import os
import json
model_checkpoint = "TODO: MODEL"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def load_data(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    examples = []
    for prediction in data["predictions"]:
        text = data["data"]["text"]
        tokens, labels = [], []
        for result in prediction["result"]:
            start, end = result["value"]["start"], result["value"]["end"]
            label = result["value"]["labels"][0]
            tokens.append(text[start:end])
            labels.append(label)
        examples.append({"tokens": tokens, "tags": labels})

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({"tokens": [ex["tokens"] for ex in examples], 
                                  "tags": [ex["tags"] for ex in examples]})
    return DatasetDict({"train": dataset})

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
            if word_idx is None:  # Special tokens (e.g., [CLS], [SEP])
                label_ids.append(-100)  # The -100 label is commonly used for token to not be used in loss calculation
            elif word_idx != previous_word_idx:  # Start of a new word
                label_ids.append(label[word_idx])  # Add the correct label
            else:  # Continuation of a word (sub-word token)
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
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }