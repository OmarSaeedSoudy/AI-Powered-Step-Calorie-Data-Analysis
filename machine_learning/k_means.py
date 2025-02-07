import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansClustering:
    def __init__(self, df):
        self.df = df
        self.n_clusters = 2
        self.max_iter = 100
        self.random_state = None
        self.scaler = StandardScaler()
        self.df_scaled = self.scaler.fit_transform(df[["steps", "calories_burned"]])
    
    def apply_kmeans(self):
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.df["cluster"] = kmeans.fit_predict(self.df_scaled)
    
    def plot_clusters(self):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x="steps", y="calories_burned", hue=self.df["cluster"], palette="viridis", data=self.df)
        plt.title("K-Means Clustering of Walking Habits")
        plt.xlabel("Steps")
        plt.ylabel("Calories Burned")
        plt.show()
    