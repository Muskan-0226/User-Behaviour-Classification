#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model and scaler
model = joblib.load('user_behavior_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("User Behavior Prediction App")

# Option to choose between file upload or manual input
option = st.selectbox("Select Input Method", ["Upload Data File", "Manual Input"])

# Function to load and display uploaded CSV data
def load_data(file):
    try:
        data = pd.read_csv(file)
        st.write("### Dataset Overview")
        st.dataframe(data.head())
        st.write(data.describe())
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

if option == "Upload Data File":
    # File uploader for CSV files
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        data = load_data(uploaded_file)

        # Encode categorical features in the dataset if needed
        if data is not None:
            categorical_columns = ['Device Model', 'Operating System', 'Gender']
            for col in categorical_columns:
                if col in data.columns:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])

else:
    # Manual Input option
    st.write("### Manual Input for Prediction")

    # Define manual input fields for user data
    def user_input_features():
        device_model = st.selectbox('Device Model', ['Google Pixel 5', 'OnePlus 9', 'Xiaomi Mi 11', 'iPhone 12', 'Samsung Galaxy S21'])
        os = st.selectbox('Operating System', ['Android', 'iOS'])
        app_usage_time = st.slider('App Usage Time (min/day)', min_value=30, max_value=600, step=10)
        screen_on_time = st.slider('Screen On Time (hours/day)', min_value=1.0, max_value=12.0, step=0.1)
        battery_drain = st.slider('Battery Drain (mAh/day)', min_value=300, max_value=3000, step=100)
        num_apps_installed = st.slider('Number of Apps Installed', min_value=10, max_value=100, step=5)
        data_usage = st.slider('Data Usage (MB/day)', min_value=100, max_value=2500, step=100)
        age = st.slider('Age', min_value=18, max_value=60, step=1)
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        
        input_data = {
            'Device Model': device_model,
            'Operating System': os,
            'App Usage Time (min/day)': app_usage_time,
            'Screen On Time (hours/day)': screen_on_time,
            'Battery Drain (mAh/day)': battery_drain,
            'Number of Apps Installed': num_apps_installed,
            'Data Usage (MB/day)': data_usage,
            'Age': age,
            'Gender': gender
        }
        
        # Create DataFrame for manual input
        return pd.DataFrame([input_data])

    # Get input data for prediction
    input_data = user_input_features()

    # Encode categorical features if needed
    label_encoders = {}
    categorical_columns = ['Device Model', 'Operating System', 'Gender']
    for col in categorical_columns:
        le = LabelEncoder()
        input_data[col] = le.fit_transform(input_data[col])

    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict and display the result
    if st.button("Predict Behavior Class"):
        prediction = model.predict(input_data_scaled)
        st.write(f"Predicted User Behavior Class: {prediction[0]}")
