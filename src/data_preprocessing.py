"""
File: data_preprocessing.py
Purpose: Provides Datahandler, 
that handle all the data.

Data sources: 
    Kaggle football stats:  https://www.kaggle.com/discussions/general/327257
    - Football player stats: https://www.kaggle.com/datasets/vivovinco/20212022-football-player-stats/data
    - Football team stats: https://www.kaggle.com/datasets/vivovinco/20212022-football-team-stats
    - Football Twitter: https://www.kaggle.com/datasets/ibrahimserouis99/twitter-sentiment-analysis-and-word-embeddings
"""
import pandas as pd
import pickle
import os
import json

"""
TODO: 
- Privatize class
- Read the new annotated txt data
- Make functions that can use the data. 
use: label-studio start

- Look at GLiNer for model to use. Bidirectional model. 
    It's a good model for adding new entities to find - that the model have not trained on.

- Look into inception for annotation. 
    - Prodigy is the best one, but cost 500 dollars. 

"""


class DataHandler:
    """
    DataHandler is a class to handle data preprocessing,
    for all the data in the project: 
        - Player stats
        - Team stats
        - Twitter
        - Articles

    Preconditions:
        /DATA/our_sentences.csv
        /DATA/player_stats.csv
        /DATA/team_stats.csv
    
    Parameters:
        ...
        
    """

    def __init__(self, class_name):
        self.class_name = class_name

        # Data for the project
        self.df_player_stats = self._load_player_stats()
        self.df_team_stats = self._load_team_stats()
        self.df_our_sentences = self._load_our_sentences()
        # self.df_annotated_text = self._load_annotated_text()


    """
    Functions to analyze the data

    """

    def get_unique_teams(df_team_stats):
        unique_teams = df_team_stats['Squad'].unique()
        return unique_teams


    def get_unique_players(df_player_stats):
        # Normalize player names
        df_player_stats['Normalized_Player'] = df_player_stats['Player'].apply(unidecode)
        
        # Find rows where the Player name was changed
        changes = df_player_stats[df_player_stats['Player'] != df_player_stats['Normalized_Player']]
        
        ##### Print the changed rows for inspection #####
        # print(changes[['Player', 'Normalized_Player']])
        
        # Return unique normalized player names
        unique_players = df_player_stats['Normalized_Player'].unique()
        return unique_players
    
    
    def get_position(player_name, player_data):
        match = player_data[player_data['Player'] == player_name]
        if not match.empty:
            return match['Standardized_Pos'].values[0]  # Return position
        return None  # If player not found
        

    """
    Functions to load the needed data.

    """ 

    def _load_player_stats(self):
        file_path = './DATA/player_stats.csv'
        df_player_stats = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1")
        position_mapping = {
            "DF": "Defender",
            "MF": "Midfielder",
            "FW": "Forward",
            "GK": "Goalkeeper",
            "MFFW": "Midfielder/Forward",  # Hybrid role
            "FWMF": "Forward/Midfielder", # Hybrid role
            "DFMF": "Defender/Midfielder",  # Hybrid role
            "FWDF": "Forward/Defender",  # Rare, but possible hybrid
            "MFDF": "Midfielder/Defender",  # Hybrid role
            "DFFW": "Defender/Forward",  # Rare hybrid
            "GKMF": "Goalkeeper/Midfielder",  # Unusual hybrid
        }
    
        # position_mapping = self.position_mapping

        df_player_stats = df_player_stats[['Player', 'Pos']]
        df_player_stats = df_player_stats.dropna()

        # Apply mapping
        df_player_stats['Standardized_Pos'] = df_player_stats['Pos'].map(position_mapping)

        return df_player_stats

    def _load_team_stats(self):
        file_path = './DATA/team_stats.csv'
        df_team_stats = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1")

        return df_team_stats

    def _load_annotated_text(self):
        file_path = './DATA/Test4.json'
        df_our_sentences = pd.read_json(file_path)
        df_our_sentences = df_our_sentences.set_index('Id')['Sentence'].to_dict()

    def _load_our_sentences(self):
        file_path = './DATA/our_sentences.json'
        df_our_sentences = pd.read_json(file_path)
        df_our_sentences = df_our_sentences.set_index('Id')['Sentence'].to_dict()

        return df_our_sentences 
    
    # Saving the datahandler 
    def saveDataHandlerClass(self, file_name):
        folder_path = './DATA/'
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
    
    """
    Getter functions
    """
    def get_class_name(self):
        return self.class_name
    
    def get_team_stats(self):
        return self.df_team_stats

    def get_our_sentences(self):
        return self.df_our_sentences

    def get_player_stats(self):
        return self.df_player_stats

    def get_positions(self):
        return self.position_mapping


          
def loadDataHandler(class_name):
    path = './DATA/'
    class_path = path + class_name
    with open(class_path, 'rb') as input:
        data_handler = pickle.load(input)
        data_handler.get_class_name = data_handler.get_class_name()
        return data_handler


# These only need to be called one time
def download_player_stats():
    path = kagglehub.dataset_download("vivovinco/20212022-football-player-stats")
    print("Path to dataset files:", path)

def download_team_stats():
    path = kagglehub.dataset_download("vivovinco/20212022-football-team-stats")
    print(path)

