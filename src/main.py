"""
File: main.py
Purpose: Entry point to execute logic from zero.py, visualization.py, and script.py.
"""

import zero  # Import zero.py to execute its logic
import visualization  # Import visualization for creating plots
import script  # Import script.py for article classification and percentage calculation

def manually_classify_articles():
    """
    Call script to manually classify articles.
    """
    input_file = "./RESULTS/results_50_advanced.csv"
    output_file = "./RESULTS/results_50_advanced_checked.csv"
    script.manually_classify(input_file, output_file)

def calculate_percentage():
    """
    Call script to calculate the percentage of football articles.
    """
    output_file = "./RESULTS/results_50_advanced_checked.csv"
    percentage = script.calculate_football_percentage(output_file)
    print(f"Percentage of football articles: {percentage:.2f}%")

if __name__ == "__main__":
    ##### Create and save 6 plots #####
    # visualization.create_plots()

    ##### Train zero-shot model #####
    # zero.execute()

    ##### Manually classify articles #####
    # manually_classify_articles()

    ##### Calculate percentage of football articles #####
    # calculate_percentage()

