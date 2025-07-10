import sys
import os
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Add parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stFileUploader {border: 2px dashed #4CAF50; border-radius: 5px; padding: 10px;}
    .stAlert {border-radius: 5px;}
    h1 {color: #2c3e50; font-family: Arial, sans-serif;}
    .sidebar .sidebar-content {background-color: #2c3e50; color: white;}
    .footer {text-align: center; color: #7f8c8d; font-size: 12px; margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

# Set page configuration
st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📞", layout="wide")

# Sidebar with project info
with st.sidebar:
    st.header("About")
    st.markdown("""
        **Telecom Churn Prediction App**  
        Upload a CSV file with customer data to predict churn using a pre-trained machine learning model.  
        Ensure the CSV matches the expected format (e.g., includes columns like tenure, MonthlyCharges, etc.).  
        [Learn more about the dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
    """)
    st.image("https://via.placeholder.com/150", caption="Telecom Analytics")

# Main title and description
st.title("📞 Telecom Churn Prediction")
st.markdown("""
    Welcome to the Telecom Churn Prediction App! Upload a customer dataset (CSV) to predict which customers are likely to churn. 
    The app uses a pre-trained machine learning model to provide accurate predictions.
""")

# File uploader
st.subheader("Upload Customer Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV with customer data (e.g., Telco Customer Churn dataset).")

if uploaded_file:
    with st.container():
        try:
            # Read the uploaded CSV
            input_df = pd.read_csv(uploaded_file)
            
            # Show data preview in an expander
            with st.expander("View Uploaded Data Preview", expanded=True):
                st.write("**Data Preview (First 5 Rows)**")
                st.dataframe(input_df.head(), use_container_width=True)

            # Load the model with debugging
            model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
            if not os.path.exists(model_path):
                st.error(f"Model file not found at {model_path}. Please ensure it exists in the project root.")
                st.stop()
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            st.write("Expected feature names:", getattr(model, 'feature_names_in_', "Not available (check model type)"))  # Debug output

            # Import preprocess_data
            try:
                from src.preprocessing import preprocess_data
                processed = preprocess_data(input_df)
            except ImportError:
                st.error("Could not import preprocess_data from src.preprocessing. Check if src/preprocessing.py exists.")
                st.stop()

            # Debug processed feature names
            st.write("Processed feature names:", processed.columns.tolist())  # Debug output

            # Align feature names (basic fix)
            expected_features = getattr(model, 'feature_names_in_', None)
            if expected_features is not None:
                # Reindex processed to match expected features, filling missing with 0
                processed = processed.reindex(columns=expected_features, fill_value=0)
            else:
                st.warning("Could not determine expected features. Ensure the model was fitted with feature names.")

            # Make predictions
            predictions = model.predict(processed)
            input_df['Churn Prediction'] = predictions
            input_df['Churn Prediction'] = input_df['Churn Prediction'].map({1: 'Yes', 0: 'No'})

            # Display predictions
            st.subheader("Churn Predictions")
            if 'customerID' in input_df.columns:
                display_df = input_df[['customerID', 'Churn Prediction']]
            else:
                display_df = input_df[['Churn Prediction']]
                st.warning("Note: 'customerID' column not found in uploaded file.")

            # Show predictions in a styled table
            st.dataframe(display_df, use_container_width=True)

            # Visualization: Churn Distribution
            churn_counts = input_df['Churn Prediction'].value_counts()
            fig = px.bar(x=churn_counts.index, y=churn_counts.values, labels={'x': 'Churn', 'y': 'Count'}, 
                         title="Churn Prediction Distribution", color=churn_counts.index, 
                         color_discrete_map={'Yes': '#FF4136', 'No': '#2ECC40'})
            st.plotly_chart(fig)

            # Download button for predictions
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv",
                help="Download the predictions as a CSV file"
            )

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty or invalid.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown('<div class="footer">Developed by xAI • Powered by Streamlit</div>', unsafe_allow_html=True)