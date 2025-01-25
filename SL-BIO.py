import json
import re

def convert_to_bio(input_file, output_file):
    """
    Converts JSON or JSONL (Label Studio export) to BIO format.
    
    Args:
        input_file (str): Path to the input JSON or JSONL file.
        output_file (str): Path to the output BIO file.
    """
    def tokenize_with_offsets(text):
        """
        Tokenizes text and keeps track of character-level offsets.
        Returns a list of (token, start, end) tuples.
        """
        tokens = []
        for match in re.finditer(r'\S+', text):  # Match non-whitespace sequences
            tokens.append((match.group(), match.start(), match.end()))
        return tokens

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Parse each JSON line
            record = json.loads(line)
            text = record['text']
            entities = record.get('entities', [])

            # Create a token-to-label mapping initialized to "O" (outside)
            tokens_with_offsets = tokenize_with_offsets(text)
            labels = ["O"] * len(tokens_with_offsets)

            # Assign entity labels
            for entity in entities:
                entity_start = entity['start']
                entity_end = entity['end']
                entity_label = entity['label']

                for i, (token, token_start, token_end) in enumerate(tokens_with_offsets):
                    if entity_start <= token_start < entity_end:  # Token starts inside the entity
                        labels[i] = f"B-{entity_label}" if labels[i] == "O" else f"I-{entity_label}"
                    elif token_start < entity_start < token_end:  # Entity starts in the middle of a token
                        labels[i] = f"B-{entity_label}" if labels[i] == "O" else f"I-{entity_label}"

            # Write to BIO format
            for (token, _, _), label in zip(tokens_with_offsets, labels):
                outfile.write(f"{token} {label}\n")
            outfile.write("\n")  # Sentence boundary

# Example usage:
convert_to_bio(
    r"NLP_EXAM_PROJECT\src\DATA\gliner_annotations_name.json",
    r"NLP_EXAM_PROJECT\src\DATA\nerdata.bio"
)