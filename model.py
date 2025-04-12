import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import joblib

from feature_engineering import FeatureEngineer

# Define column types
cat_cols = ['Gender', 'Subscription Type', 'Contract Length']
num_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 
            'Total Spend', 'Last Interaction']



def train_and_save_model():
    # Load and prepare data
    df = pd.read_csv('data/customer_churn_dataset-testing-master.csv')
    
    # Split features and target
    X = df[cat_cols + num_cols]
    y = df['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ])
    
    # Create model pipeline
    model = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'static/model/churn_model.pkl')
    
    return model

def predict_single(data):
    """
    Predict churn for single customer
    Returns:
        tuple: (prediction, probability)
        prediction: 0 for no churn, 1 for churn
        probability: probability of churn
    """
    model = joblib.load('static/model/churn_model.pkl')
    probability = model.predict_proba(data)[0][1]
    prediction = 1 if probability >= 0.5 else 0
    return prediction, probability

def get_input_features():
    """Return list of features needed for prediction"""
    return cat_cols + num_cols

if __name__ == "__main__":
    model = train_and_save_model()
    print("Model trained and saved successfully!")