from scipy.stats import ttest_ind


class HypothesisTesting:
    def __init__(self, df):
        self.df = df
    
    def high_low_test(self):
        # Define two groups: High & Low Step Count
        median_steps = self.df["steps"].median()
        high_steps = self.df[self.df["steps"] > median_steps]["calories_burned"]
        low_steps = self.df[self.df["steps"] <= median_steps]["calories_burned"]

        # Perform t-test
        t_stat, p_value = ttest_ind(high_steps, low_steps)
        print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

        # Conclusion
        if p_value < 0.05:
            print("Significant difference: More steps lead to more calories burned.")
        else:
            print("No significant difference detected.")