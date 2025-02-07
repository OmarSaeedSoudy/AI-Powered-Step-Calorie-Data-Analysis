import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import numpy as np



class DataAnalyser:
    def __init__(self, df):
        self.df = df
    
    def plot_steps_over_time(self):
        # Visualizing Step Count Trends Over Time
        plt.figure(figsize=(12, 5))
        sns.lineplot(x="date", y="steps", data=self.df, marker="o", color="blue")
        plt.xticks(rotation=45)
        plt.title("Daily Step Count Over Time")
        plt.xlabel("Date")
        plt.ylabel("Steps")
        plt.grid()
        plt.savefig("output_data/steps_over_time.png")
        plt.show()
    
    def plot_calories_burned_vs_steps(self):
        # Analyze Calories Burned vs. Steps Correlation
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x="steps", y="calories_burned", data=self.df)
        plt.title("Steps vs Calories Burned")
        plt.xlabel("Steps")
        plt.ylabel("Calories Burned")
        plt.grid()
        plt.savefig("output_data/calories_burned_vs_steps.png")
        plt.show()

        # Drop non-numeric columns before correlation calculation
        numeric_df = self.df.select_dtypes(include=["number"])

        # Correlation heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.savefig("output_data/correlation_heatmap.png")
        plt.show()


    
    def identify_anomalies(self):
        #Identify Patterns & Anomalies
        self.df["z_score"] = np.abs(zscore(self.df["steps"]))  # Calculate z-score for steps
        outliers = self.df[self.df["z_score"] > 2.5]  # Threshold > 2.5 considered outlier
        print("Outliers in Step Count:", outliers)

        # Plot Outliers
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=self.df["steps"])
        plt.title("Outlier Detection in Step Count")
        plt.savefig("output_data/outlier_detection.png")
        plt.show()