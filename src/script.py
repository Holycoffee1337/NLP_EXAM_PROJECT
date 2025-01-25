"""
File: script.py
Purpose: Manually classify articles as football-related and calculate percentages.
"""

import pandas as pd
import os

def manually_classify(file_path, output_file):
    """
    Manually label articles as football-related.
    Args:
        file_path (str): Path to the input CSV file.
        output_file (str): Path to save the labeled CSV file.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    data = pd.read_csv(file_path)
    
    if "original_article" not in data.columns:
        print("Column 'original_article' not found.")
        return
    
    if "is_football_article" not in data.columns:
        data["is_football_article"] = None
    
    for index, row in data.iterrows():
        if pd.notnull(row["is_football_article"]):
            continue
        
        print(f"Article {index + 1}:{row['original_article']}")
        answer = None
        while answer not in ["y", "n"]:
            answer = input("Is this a football article? (y/n): ").strip().lower()
        
        data.at[index, "is_football_article"] = "Yes" if answer == "y" else "No"
        data.to_csv(output_file, index=False)  # Save after every input
        print("Answer saved.")
    
    print(f"All articles labeled and saved to {output_file}.")

def calculate_football_percentage(file_path):
    """
    Calculate the percentage of football-related articles.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        float: Percentage of football-related articles.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return 0

    df = pd.read_csv(file_path)

    if 'is_football_article' not in df.columns:
        raise ValueError("Column 'is_football_article' not found.")

    total_rows = len(df)
    yes_count = df['is_football_article'].str.strip().str.lower().value_counts().get('yes', 0)
    return (yes_count / total_rows) * 100 if total_rows > 0 else 0

