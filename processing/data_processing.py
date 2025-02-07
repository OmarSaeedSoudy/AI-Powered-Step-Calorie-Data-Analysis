from processing.data_cleaning import DataCleaner
import pandas as pd

class DataProcessor:
    def __init__(self, df_steps, df_calories, df_distance):
        self.df_steps = df_steps
        self.df_calories = df_calories
        self.df_distance = df_distance
    
    def merge_data(self):
        df = pd.merge(self.df_steps, self.df_calories, how="left", on="date")
        df = pd.merge(df, self.df_distance, how="left", on="date")

        return df
        