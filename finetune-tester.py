


def test_ner_model(
    csv_path, 
    model_path="finetuned_model/model", 
    model_checkpoint="dslim/bert-base-NER"
):
    # Load the dataset
    dataset, id2label, label2id = load_data_from_csv(csv_path)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Tokenize the validation dataset
    tokenized_dataset = dataset["validation"].map(
        lambda x: tokenize_and_align_labels({"tokens": x["tokens"], "tags": x["tags"]}, tokenizer, label2id), 
        batched=True
    )

    # Create a Trainer for evaluation
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=lambda p: compute_metrics(p, id2label)
    )

    # Evaluate the model
    eval_results = trainer.evaluate(eval_dataset=tokenized_dataset)

    print("Evaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    csv_path = "studio-label-annotation.csv"
    test_ner_model(csv_path)