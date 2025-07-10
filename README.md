telecom-customer-churn/
│
├── data/                    # Dataset and preprocessed files
│   └── telecom_churn.csv
│
├── notebooks/               # Jupyter notebooks for EDA and model training
│   └── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src/                     # Python scripts for reusable code
│   └── preprocessing.py
│   └── model.py
│
├── app/                     # Optional: Streamlit or Flask app for deployment
│   └── streamlit_app.py
│
├── requirements.txt         # Project dependencies
├── README.md
└── churn_predictor.ipynb    # Main notebook (optional summary)
# 📞 Telecom Customer Churn Prediction

This project predicts whether a telecom customer will churn based on customer demographics, services subscribed, and account details using machine learning.

It includes:
- 🧪 A machine learning model trained on the Telco Customer Churn dataset
- 📊 EDA, preprocessing, and model training using Jupyter notebooks
- 🌐 A deployed Streamlit app for interactive churn prediction

---

## 🚀 Features

- 🔍 Exploratory Data Analysis (EDA)
- 🧼 Clean and modular preprocessing pipeline
- 🤖 Random Forest-based churn predictor (extendable to XGBoost, LightGBM, etc.)
- 🧠 Easily replaceable or tunable ML model
- 🌈 Streamlit frontend for uploading customer data and visualizing predictions
- 📉 Pie chart visual of churn distribution
- 📦 Download predictions as CSV

---

## 📁 Project Structure

"# telecom-customer-churn" 
