import pandas as pd


class DataCleaner:
    def __init__(self):
        pass

    def ingest_data(self):
        self.df_steps = pd.read_csv("input_data/step_counts.csv", parse_dates=["creation_datetime", "end_datetime", "start_datetime"])

        self.df_calories = pd.read_csv("input_data/cals.csv", parse_dates=["date_components"])

        self.df_distance = pd.read_csv("input_data/dist_walking_running.csv", parse_dates=["creation_datetime", "end_datetime", "start_datetime"])
    
    def clean_data(self):
        # Rename columns for easier handling
        self.df_calories.rename(columns={"date_components": "date"}, inplace=True)

        # Ensure date columns are properly formatted
        self.df_steps["date"] = self.df_steps["start_datetime"].dt.date
        self.df_calories["date"] = self.df_calories["date"].dt.date
        self.df_distance["date"] = self.df_distance["start_datetime"].dt.date

        # Drop duplicates if any
        self.df_steps.drop_duplicates(inplace=True)
        self.df_calories.drop_duplicates(inplace=True)
        self.df_distance.drop_duplicates(inplace=True)

        # Group by date 
        self.df_steps = self.df_steps.groupby("date", as_index=False).agg({
            "value": "sum"
            }).reset_index()
        self.df_distance = self.df_distance.groupby("date", as_index=False).agg({
            "value": "sum",
            "unit": "first"
            }).reset_index()
        
        # Rename columns
        self.df_distance = self.df_distance.rename(columns={"value": "distance"})
        self.df_steps = self.df_steps.rename(columns={"value": "steps"})
        self.df_calories = self.df_calories.rename(columns={"active_energy_burned": "calories_burned"})
        

        # change unit from miles to km
        self.df_distance = self.df_distance[self.df_distance["unit"] == "mi"].assign(distance=self.df_distance["distance"] * 1.60934)
        self.df_distance["unit"] = "km"

        # Select important columns
        self.df_calories = self.df_calories[["date", "calories_burned"]]
        self.df_steps = self.df_steps[["date", "steps"]]
        self.df_distance = self.df_distance[["date", "distance", "unit"]]
