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
import os
import json

model_checkpoint = "TODO: MODEL"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

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
            "save_total_limit": 2, #How many checkpoints to keep during training.
                                        #Will save at least two. The last and the best.
        }

id2label = {

}
label2id = {

}
label_list = [label2id.keys()]

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

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True  # Used to ignore discrepency in the size of output layer
)

#TODO Something like this
print("Loading data...")
dataset = load_data()

print("Starting training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Training on: {device}")

# Print out more detailed model information
print(f"Model configuration:")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

args = TrainingArguments(
    **training_args,
    logging_dir=os.path.join(training_args["output_dir"], "logs"),  # Logs are saved in a folder in the output directory
    logging_strategy="epoch",
    logging_steps=20,  # Log every n steps
    push_to_hub=False  # Not using hugginface hub
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Add early stopping callback
)


# Dataset stats
print(f"Training dataset size: {len(dataset['train'])}")
print(f"Validation dataset size: {len(dataset['validation'])}")

# Train and save results to variable
train_result = trainer.train()

# Save training results and metrics
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)


model_save_path = os.path.join(training_args["output_dir"], "model")
# The finale/best fine-tuned model is saved to a model folder in the output directory
trainer.save_model(model_save_path)

# Save the tokenizer to the same subfolder
tokenizer.save_pretrained(model_save_path)
print(f"Training complete. Model and tokenizer saved!")