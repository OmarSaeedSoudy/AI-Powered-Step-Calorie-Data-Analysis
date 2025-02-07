import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNearestNeighbors:
    def __init__(self, df):
        self.df = df 
        self.scaler = StandardScaler()
        self.k = 3
        self.median_steps = df["steps"].median()
        self.df["activity_level"] = np.where(df["steps"] > self.median_steps, 1, 0)

        self.X = df[['steps']]
        self.y = df['activity_level']

        self.X_scaled = self.scaler.fit_transform(self.X)   

    def train(self):
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42)
        
        self.knn.fit(self.X_train, self.y_train)
    
    def predict(self):
        self.y_pred = self.knn.predict(self.X_test)
        print("KNN Accuracy:", accuracy_score(self.y_test, self.y_pred))