from processing.data_cleaning import DataCleaner
from processing.data_processing import DataProcessor
from processing.data_analysis import DataAnalyser
from machine_learning.linear_regression import LinearRegressionModel
from machine_learning.k_means import KMeansClustering
from machine_learning.k_nearest_neighbors import KNearestNeighbors
from machine_learning.hypothesis_testing import HypothesisTesting
import pandas as pd


def process_data():
    """
    Data Processing and Analysis
    """
    # Ingesting and Cleaning
    data_cleaner = DataCleaner()
    data_cleaner.ingest_data()
    data_cleaner.clean_data()

    # Transforming and Creating the final dataset
    data_processor = DataProcessor(data_cleaner.df_steps, data_cleaner.df_calories, data_cleaner.df_distance)
    df = data_processor.merge_data()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].between("2024-01-01", "2024-12-30")]

    # Analyzing and Visualizing
    data_analyzer = DataAnalyser(df)
    data_analyzer.plot_steps_over_time()
    data_analyzer.plot_calories_burned_vs_steps()
    data_analyzer.identify_anomalies()
    return df


def train_and_evaluate_models(df):
    """
    Statistical & Machine Learning Models
    """
    # Linear Regression (Predicting Calories from Steps)
    linear_regression_model = LinearRegressionModel(df)
    linear_regression_model.train()
    linear_regression_model.evaluate()
    linear_regression_model.plot()

    # K-Means Clustering (Grouping Walking Habits)
    k_means_clustering = KMeansClustering(df)
    k_means_clustering.apply_kmeans()
    k_means_clustering.plot_clusters()

    # K-Nearest Neighbors (Classifying Activity Levels)
    k_nearest_neighbors = KNearestNeighbors(df)
    k_nearest_neighbors.train()
    k_nearest_neighbors.predict()

    # Hypothesis Testing (Do More Steps Burn More Calories?)
    hypothesis_testing = HypothesisTesting(df)
    hypothesis_testing.high_low_test()


if __name__ == "__main__":
    df = process_data()
    train_and_evaluate_models(df)
    print("****************************************")
    print("All done!")