# Telecom Churn Prediction System

A complete machine learning system for predicting customer churn in telecom industry with FastAPI backend and Streamlit frontend.

## Project Structure

```
telecom-churn-prediction/
├── data/                      # Dataset directory
├── models/                    # Trained models
├── src/
│   ├── api/                  # FastAPI backend
│   ├── frontend/             # Streamlit frontend
│   ├── ml/                   # ML pipeline
│   └── utils/                # Utilities
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks
├── requirements.txt
└── README.md
```

## Quick Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (REQUIRED)
python src/ml/train.py

# 4. Run the application
python run.py
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

## Usage

**API Documentation:** http://localhost:8000/docs

**Frontend Dashboard:** http://localhost:8501

## API Endpoints

- `POST /predict` - Single customer prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model metrics
- `GET /health` - Health check

## Run Tests

```bash
pytest tests/
```
