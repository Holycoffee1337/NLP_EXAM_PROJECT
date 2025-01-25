# Football Article Analysis Project

## 🏟️ Introduction
This project uses **zero-shot classification** and **visualization** techniques to identify and analyze football-related articles. The goal is to filter, preprocess, and visualize datasets, providing insights into football.

---

## 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Usage](#usage)
   - [Install Dependencies](#install-dependencies)
   - [Run the Project](#run-the-project)
4. [Outputs](#outputs)
   - [Generated Plots](#generated-plots)
   - [Filtered Results](#filtered-results)
5. [Acknowledgments](#acknowledgments)

---

## 🗂️ File Structure

### **`main.py`**
- Entry point to execute all core logic.
- Integrates zero-shot filtering, data visualization, and football article classification.

### **`data_preprocessing.py`**
- Handles preprocessing of football datasets:
  - **Player stats**: [Kaggle Dataset](https://www.kaggle.com/datasets/vivovinco/20212022-football-player-stats)
  - **Team stats**: [Kaggle Dataset](https://www.kaggle.com/datasets/vivovinco/20212022-football-team-stats)
  - **Football Twitter Data**: [Kaggle Dataset](https://www.kaggle.com/datasets/ibrahimserouis99/twitter-sentiment-analysis-and-word-embeddings)

### **`zero.py`**
- Filters football-related articles using **zero-shot classification**.
- Saves results to `results_50_advanced.csv`.

### **`results.py`**
- Similar to `zero.py`, but uses different **candidate labels** and **threshold parameters** for filtering.  
- Designed for experiments and fine-tuning classification logic.

### **`zero 2 results.py`**
- Another variation of `zero.py`, optimized for specific **topics** like "Soccer" and "Football."  
- Saves additional data, including original articles, in `zero_results_with_original_articles.csv`.

### **`./RESULTS`**
- Stores all outputs created by the project:
  - Filtered datasets.
  - Visualization plots.

### **`visualization.py`**
- Generates **6 visualizations**:
  - **Top 10 Common Words** (with and without stopwords).
  - **Word Count Distributions** (filtered and unfiltered datasets).
- Saves all plots in the `./RESULTS/` folder.

---

## 🚀 Usage (Maybe include this??)

### 1. Install Dependencies
Make sure to install the required libraries before running the project:
```bash
pip install -r requirements.txt

