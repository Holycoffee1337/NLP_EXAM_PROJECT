"""
File: data_preprocessing.py
Purpose: Provides DataHandler to manage and preprocess all data for the project.

Data sources: 
    - Kaggle football stats: https://www.kaggle.com/discussions/general/327257
    - Football player stats: https://www.kaggle.com/datasets/vivovinco/20212022-football-player-stats/data
    - Football team stats: https://www.kaggle.com/datasets/vivovinco/20212022-football-team-stats
    - Football Twitter: https://www.kaggle.com/datasets/ibrahimserouis99/twitter-sentiment-analysis-and-word-embeddings
"""

import pandas as pd
import pickle
import os
import json


class DataHandler:
    """
    Handles loading and preprocessing of player stats, team stats, and sentences.
    """

    def __init__(self, class_name):
        self.class_name = class_name
        self.df_player_stats = self._load_player_stats()
        self.df_team_stats = self._load_team_stats()
        self.df_our_sentences = self._load_our_sentences()

    # Analyze data
    def get_unique_teams(self, df_team_stats):
        return df_team_stats['Squad'].unique()

    def get_unique_players(self, df_player_stats):
        df_player_stats['Normalized_Player'] = df_player_stats['Player'].apply(unidecode)
        return df_player_stats['Normalized_Player'].unique()

    def get_position(self, player_name, player_data):
        match = player_data[player_data['Player'] == player_name]
        return match['Standardized_Pos'].values[0] if not match.empty else None

    # Load player stats
    def _load_player_stats(self):
        file_path = './DATA/player_stats.csv'
        df_player_stats = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1")
        position_mapping = {
            "DF": "Defender", "MF": "Midfielder", "FW": "Forward", "GK": "Goalkeeper",
            "MFFW": "Midfielder/Forward", "FWMF": "Forward/Midfielder", "DFMF": "Defender/Midfielder",
            "FWDF": "Forward/Defender", "MFDF": "Midfielder/Defender", "DFFW": "Defender/Forward",
            "GKMF": "Goalkeeper/Midfielder",
        }
        df_player_stats = df_player_stats[['Player', 'Pos']].dropna()
        df_player_stats['Standardized_Pos'] = df_player_stats['Pos'].map(position_mapping)
        return df_player_stats

    # Load team stats
    def _load_team_stats(self):
        file_path = './DATA/team_stats.csv'
        return pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1")

    # Load annotated text
    def _load_annotated_text(self):
        file_path = './DATA/Test4.json'
        return pd.read_json(file_path).set_index('Id')['Sentence'].to_dict()

    # Load sentences
    def _load_our_sentences(self):
        file_path = './DATA/our_sentences.json'
        return pd.read_json(file_path).set_index('Id')['Sentence'].to_dict()

    # Save DataHandler
    def save_data_handler(self, file_name):
        file_path = os.path.join('./DATA/', file_name)
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    # Getters
    def get_class_name(self):
        return self.class_name

    def get_team_stats(self):
        return self.df_team_stats

    def get_our_sentences(self):
        return self.df_our_sentences

    def get_player_stats(self):
        return self.df_player_stats


# Load DataHandler object
def load_data_handler(class_name):
    file_path = f'./DATA/{class_name}'
    with open(file_path, 'rb') as file:
        return pickle.load(file)


# Download datasets (run once)
def download_player_stats():
    print("Path to dataset files:", kagglehub.dataset_download("vivovinco/20212022-football-player-stats"))

def download_team_stats():
    print("Path to dataset files:", kagglehub.dataset_download("vivovinco/20212022-football-team-stats"))

