import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH, RANDOM_STATE, TEST_SIZE
from src.ml.preprocessor import ChurnPreprocessor
from src.ml.model import ChurnModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__, 'training.log')

def load_data():
    """Load the dataset"""
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def prepare_data(df):
    """Prepare features and target"""
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    logger.info(f"Features: {X.shape[1]}, Target distribution: {y.value_counts().to_dict()}")
    return X, y

def train_model(X_train, y_train, X_test, y_test):
    """Train and evaluate the model"""
    
    # Initialize preprocessor
    logger.info("Initializing preprocessor...")
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(X_train)
    
    # Transform data
    logger.info("Transforming data...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Train model
    logger.info("Training XGBoost model...")
    model = ChurnModel(
        model_type='xgboost',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_transformed, y_train)
    
    # Predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    logger.info("Model Performance:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Save model and preprocessor
    logger.info(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    
    logger.info(f"Saving preprocessor to {PREPROCESSOR_PATH}")
    preprocessor.save(PREPROCESSOR_PATH)
    
    # Save metrics
    logger.info(f"Saving metrics to {METRICS_PATH}")
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return model, preprocessor, metrics

def main():
    """Main training pipeline"""
    logger.info("Starting training pipeline...")
    
    # Load data
    df = load_data()
    
    # Prepare data
    X, y = prepare_data(df)
    
    # Split data
    logger.info(f"Splitting data (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Train model
    model, preprocessor, metrics = train_model(X_train, y_train, X_test, y_test)
    
    logger.info("Training pipeline completed successfully!")
    return model, preprocessor, metrics

if __name__ == "__main__":
    main()
