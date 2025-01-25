"""
File: zero_shot_filter.py
Purpose: Filter texts using zero-shot classification to identify soccer-related content and exclude rugby-related content.
"""

from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import wandb
import time


class ZeroShotFilter:
    """
    A class to filter texts using zero-shot classification.
    """

    def __init__(self, model_name, candidate_labels, soccer_labels, rugby_labels):
        # Initialize the zero-shot classifier and set label categories
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.candidate_labels = candidate_labels
        self.soccer_labels = soccer_labels
        self.rugby_labels = rugby_labels

    def filter_text_batch(self, batch_texts, soccer_threshold, rugby_threshold):
        """
        Filter a batch of texts based on soccer and rugby thresholds.
        """
        results = self.classifier(batch_texts, self.candidate_labels)
        filtered_data = []

        for text, result in zip(batch_texts, results):
            label_scores = dict(zip(result["labels"], result["scores"]))
            soccer_score = max(label_scores[label] for label in self.soccer_labels)
            rugby_score = max(label_scores[label] for label in self.rugby_labels)
            keep = (soccer_score >= soccer_threshold) and (rugby_score < rugby_threshold)

            filtered_data.append({
                "original_article": text,
                "soccer_score": soccer_score,
                "rugby_score": rugby_score,
                "keep": keep
            })

        df = pd.DataFrame(filtered_data)
        return df[df["keep"] == True]

    def sequential_filter(self, texts, soccer_threshold=0.5, rugby_threshold=0.3, batch_size=100):
        """
        Filter texts sequentially in batches for large datasets.
        """
        total_batches = (len(texts) + batch_size - 1) // batch_size
        all_results = []

        for i in range(total_batches):
            batch = texts[i * batch_size: (i + 1) * batch_size]
            print(f"Processing batch {i + 1}/{total_batches}...")
            batch_results = self.filter_text_batch(batch, soccer_threshold, rugby_threshold)
            all_results.append(batch_results)

        return pd.concat(all_results, ignore_index=True)


def execute():
    """
    Execute the filtering process and log results.
    """
    wandb.init(
        project="zero-shot-filtering",
        config={
            "model_name": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
            "soccer_threshold": 0.5,
            "rugby_threshold": 0.3,
            "batch_size": 10,
            "candidate_labels": [
                "Soccer", "Football", "Association Football",
                "Rugby", "Rugby Football",
                "Technology", "Business"
            ],
            "soccer_labels": ["Soccer", "Football", "Association Football"],
            "rugby_labels": ["Rugby", "Rugby Football"]
        }
    )

    start_time = time.time()

    # Load dataset
    ds = load_dataset("SetFit/bbc-news")
    all_data = ds['train']['text']

    # Initialize filter
    config = wandb.config
    zero_shot_filter = ZeroShotFilter(
        config["model_name"],
        config["candidate_labels"],
        config["soccer_labels"],
        config["rugby_labels"]
    )

    # Filter data
    filtered_df = zero_shot_filter.sequential_filter(
        texts=all_data,
        soccer_threshold=config["soccer_threshold"],
        rugby_threshold=config["rugby_threshold"],
        batch_size=config["batch_size"]
    )

    # Save results
    output_file = "results_50_advanced.csv"
    filtered_df.to_csv(output_file, index=False, encoding="utf-8")

    # Log results
    wandb.log({
        "filtered_texts_count": len(filtered_df),
        "output_file": output_file
    })

    end_time = time.time()
    wandb.log({"runtime_minutes": (end_time - start_time) / 60})
    wandb.finish()
    print(f"Filtered data saved as {output_file}.")
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes.")
