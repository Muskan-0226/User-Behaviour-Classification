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

    # Encode categorical features if needed
    label_encoders = {}
    categorical_columns = ['Device Model', 'Operating System', 'Gender']
    
    for col in categorical_columns:
        if col in data.columns:
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])

    # Create a mapping for displaying original labels
    device_model_mapping = dict(enumerate(label_encoders['Device Model'].classes_))
    operating_system_mapping = dict(enumerate(label_encoders['Operating System'].classes_))
    gender_mapping = dict(enumerate(label_encoders['Gender'].classes_))

    # Define input fields for user data
    def user_input_features():
        device_model = st.selectbox('Device Model', options=list(device_model_mapping.keys()), format_func=lambda x: device_model_mapping[x])
        os = st.selectbox('Operating System', options=list(operating_system_mapping.keys()), format_func=lambda x: operating_system_mapping[x])
        app_usage_time = st.slider('App Usage Time (min/day)', min_value=30, max_value=600, step=10)
        screen_on_time = st.slider('Screen On Time (hours/day)', min_value=1.0, max_value=12.0, step=0.1)
        battery_drain = st.slider('Battery Drain (mAh/day)', min_value=300, max_value=3000, step=100)
        num_apps_installed = st.slider('Number of Apps Installed', min_value=10, max_value=100, step=5)
        data_usage = st.slider('Data Usage (MB/day)', min_value=100, max_value=2500, step=100)
        age = st.slider('Age', min_value=18, max_value=60, step=1)
        gender = st.selectbox('Gender', options=list(gender_mapping.keys()), format_func=lambda x: gender_mapping[x])
        
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
        prediction = model.predict(input_data_scaled)[0]
        
        # Display selected categories and their corresponding labels
        selected_device_model = device_model_mapping[input_data[0][0]]
        selected_os = operating_system_mapping[input_data[0][1]]
        
        try:
            selected_gender = gender_mapping[input_data[0][7]]
        except KeyError:
            selected_gender = "Unknown"

        # General statements based on prediction (customize as needed)
        behavior_statements = {
            0: "Low usage detected. Consider exploring more apps to enhance your experience.",
            1: "Moderate usage detected. You're balancing your phone usage well.",
            2: "High usage detected. Be mindful of your screen time for better well-being.",
            3: "Excessive usage detected. It might be beneficial to take breaks from your device.",
            4: "Critical usage level detected. Consider reducing your screen time significantly.",
            5: "No specific advice available for this behavior class."
        }

        # Handle cases where the predicted class is not in behavior_statements
        general_statement = behavior_statements.get(prediction, "No specific advice available for this behavior class.")

        # Display prediction and general statement
        st.write(f"Predicted User Behavior Class: {prediction}")
        st.write("You selected:")
        st.write(f"Device Model: {selected_device_model} (Encoded: {input_data[0][0]})")
        st.write(f"Operating System: {selected_os} (Encoded: {input_data[0][1]})")
        st.write(f"Gender: {selected_gender} (Encoded: {input_data[0][7]})")
        
        # Display general statement regarding predicted behavior
        st.write(general_statement)
