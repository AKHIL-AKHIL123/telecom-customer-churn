# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    df = df.copy()

    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Binary encode
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Encode gender
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df
