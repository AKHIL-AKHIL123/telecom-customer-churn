import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """Custom preprocessor for telecom churn data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        
    def fit(self, X, y=None):
        """Fit the preprocessor"""
        X = X.copy()
        
        # Handle TotalCharges conversion
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            X['TotalCharges'].fillna(X['TotalCharges'].median(), inplace=True)
        
        # Drop customerID if present
        if 'customerID' in X.columns:
            X = X.drop('customerID', axis=1)
        
        # Identify feature types
        self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Fit label encoders for categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Transform for fitting scaler
        X_transformed = self._encode_features(X)
        self.scaler.fit(X_transformed[self.numerical_features])
        
        self.feature_names = X.columns.tolist()
        return self
    
    def transform(self, X):
        """Transform the data"""
        X = X.copy()
        
        # Handle TotalCharges conversion
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            X['TotalCharges'].fillna(0, inplace=True)
        
        # Drop customerID if present
        if 'customerID' in X.columns:
            X = X.drop('customerID', axis=1)
        
        # Encode categorical features
        X_transformed = self._encode_features(X)
        
        # Scale numerical features
        X_transformed[self.numerical_features] = self.scaler.transform(
            X_transformed[self.numerical_features]
        )
        
        return X_transformed
    
    def _encode_features(self, X):
        """Encode categorical features"""
        X = X.copy()
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        return X
    
    def save(self, path):
        """Save preprocessor"""
        joblib.dump(self, path)
    
    @staticmethod
    def load(path):
        """Load preprocessor"""
        return joblib.load(path)
