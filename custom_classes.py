from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Add engineered features
        X['Support Call Rate'] = X['Support Calls'] / X['Tenure']
        X['Avg Monthly Spend'] = X['Total Spend'] / X['Tenure']
        X['Recent Activity'] = X['Tenure'] - X['Last Interaction']
        return X