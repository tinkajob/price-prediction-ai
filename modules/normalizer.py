class Normalizer:
    def __init__(self):
        self.means = {} # (key, value) pairs, consisting of a feature and it's average value across the dataset 
        self.stds = {} # (key, value) pairs, consisting of a feature and it's standard deviation (how close the values are to the average)

    def fit(self, df, features):
        """Sets the values for transforming data"""
        # Here we find average value and standard deviation for each feature
        for feature in features:
            self.means[feature] = df[feature].mean()
            self.stds[feature] = df[feature].std() if df[feature].std() != 0 else 1 # To avoid division by 0

    def transform(self, df, features):
        """Calculates the normalized values"""
        df = df.copy() # To prevent modifying the existing dataframe
        for feature in features:
            df[feature] = (df[feature] - self.means[feature]) / self.stds[feature]
        return df
    
    def invert_transform(self, df, features):
        """Inverts the normalized values back to 'original'."""
        df = df.copy()
        for feature in features:
            df[feature] *= self.stds[feature]
            df[feature] += self.means[feature]
        return df
    
    def invert_value(self, value, feature):
        """Inverts the normalized value back to 'original' scale."""
        return value * self.stds[feature] + self.means[feature]