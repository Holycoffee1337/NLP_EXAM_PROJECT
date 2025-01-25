import pandas as pd
from transformers import pipeline
import torch

# Check if GPU is available and set device accordingly
device = 0 if torch.cuda.is_available() else -1

# Load your soccer_NER model
model_path = "soccer_NER_model"
ner_pipeline = pipeline("ner", model=model_path, tokenizer=model_path, aggregation_strategy="simple", device=device)

def extract_entities(text):
    """
    Extract soccer teams and players from the given text using the soccer_NER model.
    """
    ner_results = ner_pipeline(text)
    teams = [entity["word"] for entity in ner_results if entity["entity_group"] == "TEAM"]
    players = [entity["word"] for entity in ner_results if entity["entity_group"] == "PLAYER"]
    return ", ".join(teams), ", ".join(players)

# Load the CSV file
input_csv = r"RESULTS\results_50_advanced_checked.csv"
output_csv = r"RESULTS\article_entities.csv"
df = pd.read_csv(input_csv)

# Extract entities and add them to new columns
df["teams"], df["players"] = zip(*df["original_article"].apply(extract_entities))

# Save the updated CSV
df.to_csv(output_csv, index=False)
print(f"Updated CSV saved to {output_csv}")