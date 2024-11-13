#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model and scaler
model = joblib.load('user_behavior_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("User Behavior Prediction App")

# File uploader for dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Load the dataset from the uploaded file
    data = pd.read_csv(uploaded_file)

    # Display basic information about the dataset
    st.write("### Dataset Overview")
    st.dataframe(data.head())
    st.write(data.describe())

    # Encode categorical features if needed
    label_encoders = {}
    categorical_columns = ['Device Model', 'Operating System', 'Gender']
    
    for col in categorical_columns:
        if col in data.columns:
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])

    # Define input fields for user data
    def user_input_features():
        device_model = st.selectbox('Device Model', options=data['Device Model'].unique())  # Example categories
        os = st.selectbox('Operating System', options=data['Operating System'].unique())  # Example categories
        app_usage_time = st.slider('App Usage Time (min/day)', min_value=30, max_value=600, step=10)
        screen_on_time = st.slider('Screen On Time (hours/day)', min_value=1.0, max_value=12.0, step=0.1)
        battery_drain = st.slider('Battery Drain (mAh/day)', min_value=300, max_value=3000, step=100)
        num_apps_installed = st.slider('Number of Apps Installed', min_value=10, max_value=100, step=5)
        data_usage = st.slider('Data Usage (MB/day)', min_value=100, max_value=2500, step=100)
        age = st.slider('Age', min_value=18, max_value=60, step=1)
        gender = st.selectbox('Gender', options=data['Gender'].unique())  # Example categories
        
        # Convert input data into a dataframe
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
        return np.array(list(input_data.values())).reshape(1, -1)

    # Main function to get user input and predict behavior class
    st.subheader("Enter user details to predict behavior class")
    input_data = user_input_features()

    # Scale input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict and display the result
    if st.button("Predict Behavior Class"):
        prediction = model.predict(input_data_scaled)
        st.write(f"Predicted User Behavior Class: {prediction[0]}")
