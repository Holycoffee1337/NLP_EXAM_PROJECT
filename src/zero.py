from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import wandb
import time

class ZeroShotFilter:
    def __init__(self, model_name, candidate_labels, soccer_labels, rugby_labels):
        """
        Initialize Zero-Shot Text Classifier

        Args:
            model_name (str): Transformer model for zero-shot classification
            candidate_labels (list): Predefined labels for classification
            soccer_labels (list): Labels that indicate soccer/football content
            rugby_labels (list): Labels that indicate rugby content
        """
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.candidate_labels = candidate_labels
        self.soccer_labels = soccer_labels
        self.rugby_labels = rugby_labels

    def filter_text_batch(
        self,
        batch_texts,
        soccer_threshold,
        rugby_threshold
    ):
        """
        Classify a batch of texts and filter based on two conditions:
            1) The maximum score among soccer_labels is >= soccer_threshold.
            2) The maximum score among rugby_labels is < rugby_threshold.

        Args:
            batch_texts (list): List of text/articles in the batch.
            soccer_threshold (float): Minimum confidence to consider an article about soccer.
            rugby_threshold (float): Maximum confidence allowed for rugby content to exclude it.

        Returns:
            pd.DataFrame: Filtered DataFrame with columns [original_article, soccer_score, rugby_score, keep].
        """
        # Run classification
        results = self.classifier(batch_texts, self.candidate_labels)

        filtered_data = []
        for text, result in zip(batch_texts, results):
            # Create a dictionary of label->score for easier lookup
            label_scores = dict(zip(result["labels"], result["scores"]))

            # Get max soccer/football score
            soccer_score = max(label_scores[label] for label in self.soccer_labels)
            # Get max rugby score
            rugby_score = max(label_scores[label] for label in self.rugby_labels)

            # Determine whether to keep the article
            keep = (soccer_score >= soccer_threshold) and (rugby_score < rugby_threshold)

            filtered_data.append({
                "original_article": text,
                "soccer_score": soccer_score,
                "rugby_score": rugby_score,
                "keep": keep
            })

        df = pd.DataFrame(filtered_data)
        # Only return rows that meet the keep condition
        return df[df["keep"] == True]

    def sequential_filter(
        self,
        texts,
        soccer_threshold=0.5,
        rugby_threshold=0.3,
        batch_size=100
    ):
        """
        Sequentially filter texts in batches. This is useful for large datasets.

        Args:
            texts (list): List of all texts/articles to filter.
            soccer_threshold (float): Minimum confidence score to include as soccer/football.
            rugby_threshold (float): Maximum allowed rugby confidence to exclude articles.
            batch_size (int): Number of texts per batch.

        Returns:
            pd.DataFrame: Combined results from all batches.
        """
        total_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
        all_results = []

        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch = texts[start_idx:end_idx]

            print(f"Processing batch {i + 1}/{total_batches}...")
            batch_results = self.filter_text_batch(batch, soccer_threshold, rugby_threshold)
            all_results.append(batch_results)

        return pd.concat(all_results, ignore_index=True)

def execute():
    # Initialize Weights & Biases
    wandb.init(
        project="zero-shot-filtering",
        config={
            "model_name": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
            "soccer_threshold": 0.5,    # Confidence must be at least 0.5 for soccer
            "rugby_threshold": 0.3,     # Confidence must be below 0.3 for rugby
            "batch_size": 10,           # Use smaller batch size for demonstration
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
    all_data = ds['train']['text']  # For example, keep the entire training set

    # Get configuration from W&B
    config = wandb.config

    # Initialize filter
    zero_shot_filter = ZeroShotFilter(
        config["model_name"],
        config["candidate_labels"],
        config["soccer_labels"],
        config["rugby_labels"]
    )

    # Perform filtering in batches
    filtered_df = zero_shot_filter.sequential_filter(
        texts=all_data,
        soccer_threshold=config["soccer_threshold"],
        rugby_threshold=config["rugby_threshold"],
        batch_size=config["batch_size"]
    )

    # Save results
    output_file = "results_50_advanced.csv"
    filtered_df.to_csv(output_file, index=False, encoding="utf-8")

    # Log results to W&B
    wandb.log({
        "filtered_texts_count": len(filtered_df),
        "output_file": output_file
    })

    end_time = time.time()
    wandb.log({"runtime_minutes": (end_time - start_time) / 60})
    wandb.finish()
    print(f"Filtered data saved as {output_file}.")
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes.")

# Uncomment this line to run the execute function when the script is executed
# if __name__ == "__main__":
#     execute()

