"""
File: visualization.py
Purpose: Functions to visualize results and data,
         including removal of stopwords from unfiltered
         and filtered datasets.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

def common_words(df_format, category_name):
    """
    Analyzes the text data in the 'text' column of the provided DataFrame
    and plots the top 10 most common words along with their frequencies,
    removing common English stopwords.
    """
    # Handle NaN values
    df_format['text'] = df_format['text'].replace(np.nan, '', regex=True)
    
    # Pass stop_words='english' to remove common English stopwords
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df_format['text'])
    feature_names = vectorizer.get_feature_names_out()

    # Count word frequencies
    word_frequencies = Counter(dict(zip(feature_names, X.sum(axis=0).A1)))
    most_common_words = word_frequencies.most_common(10)

    # Plot the top 10 words
    plt.bar(*zip(*most_common_words))
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f"{category_name}: Top 10 Most Common Words (Stopwords Removed)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


ds = load_dataset("SetFit/bbc-news")
df_unfiltered = pd.DataFrame(ds['train']['text'], columns=["text"])

common_words(df_unfiltered, category_name="Unfiltered Dataset")

filtered_csv_path = "./RESULTS/results_50_advanced_checked.csv"
df_filtered = pd.read_csv(filtered_csv_path)

if "original_article" in df_filtered.columns:
    df_filtered.rename(columns={"original_article": "text"}, inplace=True)


common_words(df_filtered, category_name="Filtered Dataset")

