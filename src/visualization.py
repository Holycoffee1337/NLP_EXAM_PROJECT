"""
File: visualization.py
Purpose: Functions to visualize results and data,
         including saving common words (with and without stopword removal)
         and word count distribution plots.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from datasets import load_dataset
import os

# Ensure results directory exists
os.makedirs("./RESULTS", exist_ok=True)

def common_words(df, category_name, save_path, remove_stopwords=True):
    """
    Plots top 10 common words.
    """
    df['text'] = df['text'].fillna('')

    # CountVectorizer setup
    vectorizer = CountVectorizer(stop_words='english' if remove_stopwords else None)
    X = vectorizer.fit_transform(df['text'])
    word_frequencies = Counter(dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1)))

    # Top 10 words
    most_common = word_frequencies.most_common(10)
    plt.bar(*zip(*most_common))
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f"{category_name}: Top 10 Most Common Words ({'Stopwords Removed' if remove_stopwords else 'With Stopwords'})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def count_words(text):
    """
    Returns word count.
    """
    return len(text.split()) if isinstance(text, str) else 0

def word_count_distribution(df, dataset_name, save_path):
    """
    Saves histogram of word counts.
    """
    df['Word_Count'] = df['text'].apply(count_words)
    plt.hist(df['Word_Count'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.title(f'Word Count Distribution in {dataset_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_plots():
    """
    Creates all visualizations and saves them to the results folder.
    """
    # Load datasets
    ds = load_dataset("SetFit/bbc-news")
    df_unfiltered = pd.DataFrame(ds['train']['text'], columns=["text"])

    filtered_csv_path = "./RESULTS/results_50_advanced_checked.csv"
    df_filtered = pd.read_csv(filtered_csv_path)

    if "original_article" in df_filtered.columns:
        df_filtered.rename(columns={"original_article": "text"}, inplace=True)

    # Generate common words plots
    common_words(df_unfiltered, "Unfiltered Dataset (With Stopwords)", "./RESULTS/threshold_50_with_stopwords.png", remove_stopwords=False)
    common_words(df_filtered, "Filtered Dataset (With Stopwords)", "./RESULTS/advanced_threshold_50_with_stopwords.png", remove_stopwords=False)
    common_words(df_unfiltered, "Unfiltered Dataset", "./RESULTS/threshold_50.png", remove_stopwords=True)
    common_words(df_filtered, "Filtered Dataset", "./RESULTS/advanced_threshold_50.png", remove_stopwords=True)

    # Generate word count distribution plots
    word_count_distribution(df_unfiltered, "Unfiltered Dataset", "./RESULTS/unfiltered_word_count_dist.png")
    word_count_distribution(df_filtered, "Filtered Dataset", "./RESULTS/filtered_word_count_dist.png")

    print("Plots saved in the './RESULTS' folder:")
    print("- Top 10 Common Words (Unfiltered, With Stopwords): ./RESULTS/threshold_50_with_stopwords.png")
    print("- Top 10 Common Words (Filtered, With Stopwords): ./RESULTS/advanced_threshold_50_with_stopwords.png")
    print("- Top 10 Common Words (Unfiltered, Stopwords Removed): ./RESULTS/threshold_50.png")
    print("- Top 10 Common Words (Filtered, Stopwords Removed): ./RESULTS/advanced_threshold_50.png")
    print("- Word Count Distribution (Unfiltered): ./RESULTS/unfiltered_word_count_dist.png")
    print("- Word Count Distribution (Filtered): ./RESULTS/filtered_word_count_dist.png")

