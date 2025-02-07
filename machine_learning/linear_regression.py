from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


class LinearRegressionModel:
    def __init__(self, df):
        self.X = df[['steps']]
        self.y = df['calories_burned']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def train(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"R2: {r2}")
    
    def plot(self):
        plt.scatter(self.X_test, self.y_test, color='blue')
        plt.plot(self.X_test, self.model.predict(self.X_test), color='red')
        plt.xlabel('Steps')
        plt.ylabel('Calories Burned')
        plt.title("Linear Regression Model")
        plt.savefig("output_data/linear_regression.png")
        plt.show()

