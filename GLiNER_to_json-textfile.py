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

# Input and output file paths
input_file = "input_texts.txt"  # Text file with one input text per line
output_file = "ner_annotations.json"  # Output JSON file

# Initialize a list to store NER data for all lines
all_ner_data = []

# Process each line in the input text file
with open(input_file, "r", encoding="utf-8") as f:
    for line_index, line in enumerate(f):
        text = line.strip()  # Remove leading/trailing whitespace
        if not text:  # Skip empty lines
            continue
        
        # Predict entities using the GLiNER model
        entities = model.predict_entities(text, labels)
        
        # Prepare the NER data for the current line
        ner_data = {
            "line_number": line_index + 1,  # Track line numbers for reference
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
        # Add the current line's NER data to the list
        all_ner_data.append(ner_data)

# Save all NER data to a JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_ner_data, f, ensure_ascii=False, indent=4)

print(f"NER annotations saved to {output_file}")