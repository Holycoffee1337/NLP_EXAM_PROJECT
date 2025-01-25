import ast
from datasets import Dataset, DatasetDict
import pandas as pd
import evaluate
import numpy as np
import json
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)

def load_data_from_csv(csv_path):
    examples = []
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        text = row["text"]
        
        if not text:
            continue

        try:
            annotations = ast.literal_eval(row['label'])
        except (ValueError, SyntaxError):
            continue

        tokens = text.split()
        labels = ['O'] * len(tokens)

        for annotation in sorted(annotations, key=lambda x: x['start']):
            start, end = annotation['start'], annotation['end']
            label = annotation['labels'][0]
            
            try:
                entity_tokens = text[start:end].split()
                start_idx = tokens.index(entity_tokens[0])
                end_idx = start_idx + len(entity_tokens)
                
                labels[start_idx] = f'B-{label}'
                for i in range(start_idx + 1, end_idx):
                    labels[i] = f'I-{label}'
            except ValueError:
                continue

        examples.append({"tokens": tokens, "tags": labels})

    unique_labels = set(label.replace('B-', '').replace('I-', '') for ex in examples for label in ex["tags"])
    unique_labels.add('O')

    dataset = Dataset.from_dict({
        "tokens": [ex["tokens"] for ex in examples],
        "tags": [ex["tags"] for ex in examples]
    })

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

dataset= load_data_from_csv("DATA\labelstudio-annotation.csv")
label2id = {
    'O': 0,
    'B-PLAYR': 1,
    'I-PLAYR': 2,
    'B-TEAM': 3,
    'I-TEAM': 4
}
id2label = {v: k for k, v in label2id.items()}

model_checkpoint = "dslim/bert-base-NER"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

#\cite{https://colab.research.google.com/drive/1o0jjpWMgG1cX7eAYsV7hf2ptrOeS1fqS?usp=sharing#scrollTo=dONBxQI67sXq}
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        label = [label2id[l] for l in label]  # Convert string labels to integers
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_data = dataset.map(tokenize_and_align_labels, batched=True)
print(tokenized_data)
label_list = list(label2id.keys())
seqeval = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
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

training_args = TrainingArguments(
    output_dir="soccer_NER",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    save_total_limit = 2, #How many checkpoints to keep during training.
                                        #Will save at least two. The last and the best.
)
# Load the pre-trained model
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id, 
    ignore_mismatched_sizes=True
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,  # Fixed: Use `tokenizer` instead of `processing_class`
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
results = trainer.train()
print(results)

# Save the trained model and tokenizer
output_dir = "soccer_NER_model"  # Replace with your desired folder name
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

# Evaluate the final model on the validation set
evaluation_results = trainer.evaluate()
print("Evaluation results:", evaluation_results)

# Save evaluation results to a JSON file
evaluation_results_file = "RESULTS\evaluation_results.json"
with open(evaluation_results_file, "w") as f:
    json.dump(evaluation_results, f, indent=4)
print(f"Evaluation results saved to {evaluation_results_file}")