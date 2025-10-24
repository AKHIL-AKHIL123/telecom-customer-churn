import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import API_URL

# Page config
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_URL}/model/info")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def predict_churn(customer_data):
    """Make prediction via API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
    return None

def main():
    # Header
    st.markdown('<p class="main-header">Telecom Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict customer churn with machine learning</p>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("API is not running. Please start the FastAPI backend first.")
        st.code("python src/api/main.py")
        return
    
    st.success("Connected to API")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Model Info"])
    
    if page == "Single Prediction":
        show_single_prediction()
    elif page == "Batch Prediction":
        show_batch_prediction()
    else:
        show_model_info()

def show_single_prediction():
    st.header("Single Customer Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
    with col2:
        st.subheader("Additional Services")
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        st.subheader("Account Information")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
    
    if st.button("Predict Churn", type="primary"):
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": str(total_charges)
        }
        
        with st.spinner("Making prediction..."):
            result = predict_churn(customer_data)
        
        if result:
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Prediction", 
                         "Will Churn" if result['churn_prediction'] == 1 else "Will Stay")
            
            with col2:
                st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
            
            with col3:
                st.metric("Risk Level", result['risk_level'])
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['churn_probability'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

def show_batch_prediction():
    st.header("Batch Prediction")
    st.write("Upload a CSV file with customer data for batch predictions")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        if st.button("Predict All", type="primary"):
            st.info("Batch prediction feature - implement by calling /predict/batch endpoint")

def show_model_info():
    st.header("Model Information")
    
    model_info = get_model_info()
    
    if model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Type", model_info['model_type'].upper())
            st.metric("Accuracy", f"{model_info['accuracy']:.4f}")
            st.metric("Precision", f"{model_info['precision']:.4f}")
        
        with col2:
            st.metric("Recall", f"{model_info['recall']:.4f}")
            st.metric("F1 Score", f"{model_info['f1_score']:.4f}")
            st.metric("ROC AUC", f"{model_info['roc_auc']:.4f}")
        
        # Metrics visualization
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            'Score': [
                model_info['accuracy'],
                model_info['precision'],
                model_info['recall'],
                model_info['f1_score'],
                model_info['roc_auc']
            ]
        })
        
        fig = px.bar(metrics_df, x='Metric', y='Score', 
                     title='Model Performance Metrics',
                     color='Score',
                     color_continuous_scale='Blues')
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model information not available. Train the model first.")

if __name__ == "__main__":
    main()
