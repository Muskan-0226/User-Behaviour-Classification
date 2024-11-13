#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Let's load the dataset and examine its structure to understand the columns, data types, missing values, etc.
import pandas as pd

# Load the dataset
file_path = r"C:\Users\ASUS\Downloads\user_behavior_dataset.csv"
data = pd.read_csv(file_path)

# Display basic information about the dataset
data_info = data.info()
data_head = data.head()
data_describe = data.describe()
data_nulls = data.isnull().sum()

data_info, data_head, data_describe, data_nulls


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('seaborn-darkgrid')

# 1. Distribution of Numerical Features
numerical_columns = ['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 
                     'Number of Apps Installed', 'Data Usage (MB/day)', 'Age']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(numerical_columns):
    sns.histplot(data[col], kde=True, ax=axes[i // 3, i % 3])
    axes[i // 3, i % 3].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# 2. Distribution of Target Variable (User Behavior Class)
plt.figure(figsize=(8, 5))
sns.countplot(x='User Behavior Class', data=data)
plt.title('Distribution of User Behavior Class')
plt.xlabel('User Behavior Class')
plt.ylabel('Count')
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()


# In[6]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import pandas as pd

# Load the dataset
data = pd.read_csv(r"C:\Users\ASUS\Downloads\user_behavior_dataset.csv")

# Encode categorical features
label_encoders = {}
categorical_columns = ['Device Model', 'Operating System', 'Gender']
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Separate features and target variable
X = data.drop(columns=['User ID', 'User Behavior Class'])
y = data['User Behavior Class']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Dictionary to store model performance
performance = {}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store the results
    performance[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    # Print classification report
    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Display performance summary
performance


# In[7]:


from sklearn.model_selection import GridSearchCV, cross_val_score

# Select the best-performing model based on initial results
# (Assuming Random Forest performed best in this example; change if another model performed better)
best_model = RandomForestClassifier()

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='f1_weighted', verbose=2)

# Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Display the best parameters
print("Best Parameters from Grid Search:", grid_search.best_params_)

# Evaluate the tuned model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display the optimized model's performance
print("\nOptimized Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(classification_report(y_test, y_pred))


# In[10]:


import joblib

# Save the scaler
scaler = StandardScaler().fit(X_train)  # Fit the scaler on training data
joblib.dump(scaler, 'scaler.pkl')

# Save the optimized model (assuming best_model is optimized)
joblib.dump(best_model, 'user_behavior_model.pkl')



# In[11]:


import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('user_behavior_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("User Behavior Prediction App")

# Define input fields for user data
def user_input_features():
    device_model = st.selectbox('Device Model', options=[0, 1, 2])  # Example categories
    os = st.selectbox('Operating System', options=[0, 1])  # Example categories
    app_usage_time = st.slider('App Usage Time (min/day)', min_value=30, max_value=600, step=10)
    screen_on_time = st.slider('Screen On Time (hours/day)', min_value=1.0, max_value=12.0, step=0.1)
    battery_drain = st.slider('Battery Drain (mAh/day)', min_value=300, max_value=3000, step=100)
    num_apps_installed = st.slider('Number of Apps Installed', min_value=10, max_value=100, step=5)
    data_usage = st.slider('Data Usage (MB/day)', min_value=100, max_value=2500, step=100)
    age = st.slider('Age', min_value=18, max_value=60, step=1)
    gender = st.selectbox('Gender', options=[0, 1])  # Example categories
    
    # Convert input data into a dataframe
    data = {
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
    return np.array(list(data.values())).reshape(1, -1)

# Main function
st.subheader("Enter user details to predict behavior class")
input_data = user_input_features()

# Scale input data using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# Predict and display the result
if st.button("Predict Behavior Class"):
    prediction = model.predict(input_data_scaled)
    st.write(f"Predicted User Behavior Class: {prediction[0]}")


# In[ ]:




