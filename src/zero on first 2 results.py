"""
File: zero_shot_filter.py
Purpose: Filter texts using zero-shot classification with thresholds.
"""

from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import wandb
import time


class ZeroShotFilter:
    def __init__(self, model_name, candidate_labels):
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.candidate_labels = candidate_labels

    def filter_text_batch(self, batch, topics, threshold):
        results = self.classifier(batch, self.candidate_labels)
        df = pd.DataFrame(results)
        df["original_article"] = batch  # Include original texts
        return df[
            (df["labels"].apply(lambda labels: labels[0] in topics)) &
            (df["scores"].apply(lambda scores: scores[0] > threshold))
        ]

    def sequential_filter(self, texts, topics, threshold=0.5, batch_size=100):
        total_batches = (len(texts) + batch_size - 1) // batch_size
        all_results = []
        for i in range(total_batches):
            batch = texts[i * batch_size: (i + 1) * batch_size]
            print(f"Processing batch {i + 1}/{total_batches}...")
            all_results.append(self.filter_text_batch(batch, topics, threshold))
        return pd.concat(all_results, ignore_index=True)


def execute():
    wandb.init(
        project="zero-shot-filtering",
        config={
            "model_name": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
            "threshold": 0.5,
            "batch_size": 10,
            "topics": ["Soccer", "Football"],
            "candidates": ["Soccer", "Football", "Technology", "Business"]
        }
    )
    start_time = time.time()

    # Load dataset
    ds = load_dataset("SetFit/bbc-news")
    all_data = ds['train']['text']

    # Filter texts
    config = wandb.config
    filter = ZeroShotFilter(config["model_name"], config["candidates"])
    df = filter.sequential_filter(all_data, config["topics"], config["threshold"], config["batch_size"])

    # Save and log results
    file_name = "zero_results_with_original_articles.csv"
    df.to_csv(file_name, index=False, encoding="utf-8")
    wandb.log({"filtered_texts": len(df), "output_file": file_name})
    wandb.log({"runtime_minutes": (time.time() - start_time) / 60})
    wandb.finish()
    print(f"Results saved to {file_name}. Total time: {(time.time() - start_time) / 60:.2f} minutes.")
