from gliner import GLiNER
import json

# Load the GLiNER model
model = GLiNER.from_pretrained("EmergentMethods/gliner_medium_news-v2.1")

# Define the labels that GLiNER should use during prediction
labels = ["Person", "Player Name", "Organization", "League", "Team Name", "Location"]

# Mapping for the BIO standard labels
bio_label_mapping = {
    "Team Name": "TEAM",
    "Player Name": "PLAYR"
}

# Input and output file paths
input_file = "NLP_EXAM_PROJECT\src\DATA\sentences.txt"  # Text file with one input text per line
output_file = "NLP_EXAM_PROJECT\src\DATA\gliner_annotations_name.json"  # Output JSON file

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
        
        # Filter entities and remap the labels to match BIO standards
        filtered_entities = [
            {
                **entity,
                "label": bio_label_mapping[entity["label"]]
            }
            for entity in entities if entity["label"] in bio_label_mapping
        ]
        
        # Skip lines with no relevant entities
        if not filtered_entities:
            continue

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
                        for index, entity in enumerate(filtered_entities)
                    ]
                }
            ]
        }
        # Add the current line's NER data to the list
        all_ner_data.append(ner_data)

# Save all filtered NER data to a JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_ner_data, f, ensure_ascii=False, indent=4)

print(f"Filtered NER annotations saved to {output_file}")
