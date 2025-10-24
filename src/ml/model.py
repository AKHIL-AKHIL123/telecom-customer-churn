import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib

class ChurnModel:
    """Wrapper class for churn prediction models"""
    
    def __init__(self, model_type='xgboost', **kwargs):
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.feature_importance = None
        
    def _create_model(self, model_type, **kwargs):
        """Create model based on type"""
        models = {
            'xgboost': XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42),
                eval_metric='logloss'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42)
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 5),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42)
            ),
            'logistic': LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        }
        return models.get(model_type, models['xgboost'])
    
    def fit(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        
        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return scores
    
    def get_feature_importance(self, feature_names):
        """Get feature importance as dictionary"""
        if self.feature_importance is not None:
            return dict(zip(feature_names, self.feature_importance))
        return None
    
    def save(self, path):
        """Save model"""
        joblib.dump(self, path)
    
    @staticmethod
    def load(path):
        """Load model"""
        return joblib.load(path)
