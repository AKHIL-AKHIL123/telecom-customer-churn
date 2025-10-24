from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH, API_HOST, API_PORT
from src.api.schemas import (
    CustomerData, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, ModelInfo, HealthResponse
)
from src.ml.model import ChurnModel
from src.ml.preprocessor import ChurnPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__, 'api.log')

# Initialize FastAPI app
app = FastAPI(
    title="Telecom Churn Prediction API",
    description="API for predicting customer churn in telecom industry",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
metrics = None

@app.on_event("startup")
async def load_model():
    """Load model and preprocessor on startup"""
    global model, preprocessor, metrics
    
    try:
        logger.info("Loading model and preprocessor...")
        model = ChurnModel.load(MODEL_PATH)
        preprocessor = ChurnPreprocessor.load(PREPROCESSOR_PATH)
        
        if METRICS_PATH.exists():
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        
        logger.info("Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Telecom Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None
    )

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model information and metrics"""
    if metrics is None:
        raise HTTPException(status_code=404, detail="Model metrics not found")
    
    return ModelInfo(
        model_type=model.model_type,
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1_score=metrics['f1_score'],
        roc_auc=metrics['roc_auc']
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerData):
    """Predict churn for a single customer"""
    try:
        # Convert to DataFrame
        customer_dict = customer.model_dump()
        df = pd.DataFrame([customer_dict])
        
        # Preprocess
        df_transformed = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(df_transformed)[0]
        probability = model.predict_proba(df_transformed)[0][1]
        risk_level = get_risk_level(probability)
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
        
        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=float(probability),
            risk_level=risk_level
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for multiple customers"""
    try:
        # Convert to DataFrame
        customers_data = [customer.model_dump() for customer in request.customers]
        df = pd.DataFrame(customers_data)
        
        # Preprocess
        df_transformed = preprocessor.transform(df)
        
        # Predict
        predictions = model.predict(df_transformed)
        probabilities = model.predict_proba(df_transformed)[:, 1]
        
        # Create response
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(PredictionResponse(
                churn_prediction=int(pred),
                churn_probability=float(prob),
                risk_level=get_risk_level(prob)
            ))
        
        logger.info(f"Batch prediction completed for {len(results)} customers")
        
        return BatchPredictionResponse(predictions=results)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
