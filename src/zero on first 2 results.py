from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import wandb
import time

class ZeroShotFilter:
    def __init__(self, model_name, candidate_labels):
        """
        Initialize Zero-Shot Text Classifier
        Args:
            model_name (str): Transformer model for zero-shot classification
            candidate_labels (list): Predefined labels for classification
        """
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.candidate_labels = candidate_labels

    def filter_text_batch(self, batch, topics, threshold):
        """
        Filter a batch of texts with original articles included
        Args:
            batch (list): List of texts in a batch
            topics (list): Topics of interest
            threshold (float): Minimum confidence for filtering
        Returns:
            DataFrame: Filtered results with original articles
        """
        results = self.classifier(batch, self.candidate_labels)
        df = pd.DataFrame(results)

        # Add the original articles to the DataFrame
        df["original_article"] = batch

        # Filter by topics and confidence threshold
        topics_articles = df[df["labels"].apply(lambda labels: labels[0] in topics)]
        high_conf_articles = topics_articles[topics_articles["scores"].apply(lambda scores: scores[0] > threshold)]

        return high_conf_articles

    def sequential_filter(self, texts, topics, threshold=0.5, batch_size=100):
        """
        Sequentially filter texts in batches
        Args:
            texts (list): List of all texts
            topics (list): Topics of interest
            threshold (float): Minimum confidence for filtering
            batch_size (int): Number of texts per batch
        Returns:
            DataFrame: Combined results from all batches
        """
        total_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
        all_results = []
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch = texts[start_idx:end_idx]

            print(f"Processing batch {i + 1}/{total_batches}...")
            batch_results = self.filter_text_batch(batch, topics, threshold)
            all_results.append(batch_results)

        return pd.concat(all_results, ignore_index=True)

def execute():
    # Initialize wandb
    wandb.init(
        project="zero-shot-filtering",
        config={
            "model_name": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
            "threshold": 0.5,
            "batch_size": 10,  # Use smaller batch size for testing
            "topics": ["Soccer", "Football"],
            "candidates": [
                "Soccer", "Football", "Technology", "Business", 
            ]
        }
    )

    start_time = time.time()

    # Load dataset
    ds = load_dataset("SetFit/bbc-news")
    all_data = ds['train']['text']  # Only take the first 50 articles

    # Get configuration
    config = wandb.config
    filter = ZeroShotFilter(config["model_name"], config["candidates"])

    # Filter texts sequentially
    df = filter.sequential_filter(all_data, config["topics"], config["threshold"], config["batch_size"])

    # Save results with original articles included
    file_name = "zero_results_with_original_articles_50_rugby.csv"
    df.to_csv(file_name, index=False, encoding="utf-8")
    wandb.log({"filtered_texts": len(df), "output_file": file_name})

    end_time = time.time()
    wandb.log({"runtime_minutes": (end_time - start_time) / 60})
    wandb.finish()
    print(f"Filtered data saved as {file_name}. Total time: {(end_time - start_time) / 60:.2f} minutes.")

# Uncomment to run the execute function when the script is executed
# execute()

