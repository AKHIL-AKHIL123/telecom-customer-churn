from pydantic import BaseModel, Field
from typing import List, Optional

class CustomerData(BaseModel):
    """Schema for single customer prediction"""
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., ge=0, description="Number of months")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Payment method type")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    TotalCharges: str = Field(..., description="Total charges")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85"
            }
        }

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    churn_prediction: int = Field(..., description="0 for No Churn, 1 for Churn")
    churn_probability: float = Field(..., description="Probability of churn")
    risk_level: str = Field(..., description="Low, Medium, or High")

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction"""
    customers: List[CustomerData]

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[PredictionResponse]

class ModelInfo(BaseModel):
    """Schema for model information"""
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

class HealthResponse(BaseModel):
    """Schema for health check"""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
