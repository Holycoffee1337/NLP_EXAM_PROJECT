import os
import ast
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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
    
    id2label = {i: label for i, label in enumerate(sorted(unique_labels))}
    label2id = {label: i for i, label in id2label.items()}

    dataset = Dataset.from_dict({
        "tokens": [ex["tokens"] for ex in examples],
        "tags": [ex["tags"] for ex in examples]
    })

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]}), id2label, label2id

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id.get(label[word_idx], label2id['O']))
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(predictions_and_labels, id2label):
    predictions, labels = predictions_and_labels
    predictions = np.argmax(predictions, axis=2)

    flat_predictions = []
    flat_labels = []
    for pred, label in zip(predictions, labels):
        flat_predictions.extend([p for p, l in zip(pred, label) if l != -100])
        flat_labels.extend([l for l in label if l != -100])

    flat_predictions = [id2label[p] for p in flat_predictions]
    flat_labels = [id2label[l] for l in flat_labels]

    precision, recall, f1, _ = precision_recall_fscore_support(flat_labels, flat_predictions, average='weighted')
    accuracy = accuracy_score(flat_labels, flat_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

def train_ner_model(
    csv_path, 
    model_checkpoint="dslim/bert-base-NER", 
    output_dir="finetuned_model"
):
    dataset, id2label, label2id = load_data_from_csv(csv_path)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id), 
        batched=True
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        report_to="none",
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    train_result = trainer.train()
    
    model_save_path = os.path.join(output_dir, "model")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    print(f"Training complete. Model saved to {model_save_path}")
    print(f"Training dataset size: {len(tokenized_dataset['train'])}")
    print(f"Validation dataset size: {len(tokenized_dataset['validation'])}")

def eval(trainer):    
    # Evaluate the model
    eval_results = trainer.evaluate(eval_dataset=tokenized_dataset)

    print("Evaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    csv_path = "studio-label-annotation.csv"
    #train_ner_model(csv_path)