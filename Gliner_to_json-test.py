from gliner import GLiNER
import pandas as pd
from datasets import load_dataset
import json

# Load the GLiNER model
model = GLiNER.from_pretrained("EmergentMethods/gliner_medium_news-v2.1")

# Load the SetFit BBC News dataset
dataset = load_dataset("SetFit/bbc-news")

# Define the list of labels (these should match what GLiNER expects)
labels = ["Person", "Player", "Place", "Date", "Organization", "Team", "Score"]

# Extract a sample text for testing
text = dataset["train"][0]["text"]  # Example text from the dataset

# Predict entities using the GLiNER model
entities = model.predict_entities(text, labels)

# Prepare the JSON data for Label Studio
ner_data = {
    "data": {
        "text": text
    },
    "predictions": [
        {
            "result": [
                {
                    "value": {
                        "start": entity["start"],
                        "end": entity["end"],
                        "text": entity["text"],
                        "labels": [entity["label"]]
                    },
                    "id": f"ner_{index}",
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels"
                }
                for index, entity in enumerate(entities)
            ]
        }
    ]
}

# Save the JSON to a file
output_file = "ner_annotations.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(ner_data, f, ensure_ascii=False, indent=4)

print(f"NER annotations saved to {output_file}")
