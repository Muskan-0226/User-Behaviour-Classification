# -*- coding: utf-8 -*-
"""UserBehaviourclass.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10f1XI68lvmyPq_JufOMXQ_rVIPHtNb6Y
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('/content/user_behavior_dataset.csv')
df.head()

# Check for missing values
print(df.isnull().sum())

# Encoding categorical features
categorical_features = ['Device Model', 'Operating System', 'Gender']
encoder = LabelEncoder()
for feature in categorical_features:
    df[feature] = encoder.fit_transform(df[feature])

# Scaling numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Screen On Time (hours/day)', 'Data Usage (MB/day)', 'Battery Drain (mAh/day)']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize feature distributions
for feature in numerical_features:
    plt.figure()
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Analyze target variable (User Behavior Class) for imbalance
sns.countplot(x='User Behavior Class', data=df)
plt.title('Class Distribution')

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Split data
X = df.drop('User Behavior Class', axis=1)
y = df['User Behavior Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'Logistic Regression': LogisticRegression(solver='saga', max_iter=300),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred))

from sklearn.metrics import classification_report, accuracy_score

# Evaluate each model
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\nModel: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'class_weight': ['balanced', 'balanced_subsample']  # Adjust weights for imbalance
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_res, y_res)

# Best model and parameters
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate the tuned model on the test set
y_pred_tuned = best_model.predict(X_test)
print("Tuned Model Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation with F1 scoring for multiclass problems
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1_weighted')  # or use 'f1_macro' or 'f1_micro'
print("Cross-Validation F1 Scores:", cv_scores)
print("Mean CV F1 Score:", cv_scores.mean())

# Feature importance for best model
importances = best_model.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]

print("Top Influential Features:")
for idx in sorted_indices[:5]:  # top 5 features
    print(f"{feature_names[idx]}: {importances[idx]}")

!pip install streamlit

# Save the model
import joblib
joblib.dump(best_model, 'best_model.pkl')

# Streamlit app
import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('best_model.pkl')

import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# Title and input form
st.title("User Behavior Classification")
age = st.number_input("Age")
screen_time = st.number_input("Screen Time")
data_usage = st.number_input("Data Usage")
battery_drain = st.number_input("Battery Drain")

# Predict button
if st.button("Predict"):
    features = np.array([[age, screen_time, data_usage, battery_drain]])
    prediction = model.predict(features)
    st.write("Predicted User Behavior Class:", prediction[0])

# Run locally
# streamlit run <script_name.py>

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("/content/user_behavior_dataset.csv")

# Feature columns
X = data[['Age', 'App Usage Time (min/day)', 'Screen On Time (hours/day)',
          'Battery Drain (mAh/day)', 'Number of Apps Installed', 'Data Usage (MB/day)', 'Gender']]  # Adjust columns as needed

# Encode categorical feature columns (e.g., 'Gender') using LabelEncoder
label_encoder_X = LabelEncoder()

# Apply label encoding only to categorical columns in X (e.g., 'Gender')
X['Gender'] = label_encoder_X.fit_transform(X['Gender'])

# Target variable
y = data['User Behavior Class']  # The target variable (categorical)

# Encode the target variable
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the classifier (RandomForestClassifier in this case)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Save the trained model to a .pkl file
joblib.dump(classifier, 'classifier_model.pkl')

# Save the LabelEncoders (both for 'Gender' and 'User Behavior Class')
joblib.dump(label_encoder_X, 'label_encoder_X.pkl')
joblib.dump(label_encoder_y, 'label_encoder_y.pkl')

import streamlit as st
import pandas as pd
import joblib

# Load models
classifier = joblib.load('classifier_model.pkl')  # Model for Experience Level classification
label_encoder_X = joblib.load('label_encoder_X.pkl')  # Label encoder for Gender in X features
label_encoder_y = joblib.load('label_encoder_y.pkl')  # Label encoder for Experience Level

st.title("Mobile User Behaviour Classification")

# Data upload
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", input_data.head())

# Input sliders and number inputs (features based on your dataset)
age = st.slider("Age", 18, 80, 25)
app_usage_time = st.number_input("App Usage Time (min/day)", 0, 1000, 150)
screen_on_time = st.slider("Screen On Time (hours/day)", 0.0, 24.0, 2.0)
battery_drain = st.number_input("Battery Drain (mAh/day)", 0, 10000, 500)
num_apps = st.number_input("Number of Apps Installed", 0, 500, 30)
data_usage = st.number_input("Data Usage (MB/day)", 0, 1000, 200)
gender = st.selectbox("Gender", ["Male", "Female"])

# Encode the gender for the prediction
encoded_gender = label_encoder_X.transform([gender])[0]

# Display Predictions
if st.button("Predict Calories Burned"):
    user_input = pd.DataFrame([[age, app_usage_time, screen_on_time, battery_drain, num_apps, data_usage, encoded_gender]],
                              columns=['Age', 'App Usage Time (min/day)', 'Screen On Time (hours/day)',
                                       'Battery Drain (mAh/day)', 'Number of Apps Installed', 'Data Usage (MB/day)', 'Gender'])
    prediction = regressor.predict(user_input)
    st.write(f"Predicted Calories Burned: {prediction[0]:.2f}")

if st.button("Classify Experience Level"):
    user_input = pd.DataFrame([[age, app_usage_time, screen_on_time, battery_drain, num_apps, data_usage, encoded_gender]],
                              columns=['Age', 'App Usage Time (min/day)', 'Screen On Time (hours/day)',
                                       'Battery Drain (mAh/day)', 'Number of Apps Installed', 'Data Usage (MB/day)', 'Gender'])
    prediction = classifier.predict(user_input)
    predicted_class = label_encoder_y.inverse_transform(prediction)  # Convert numeric prediction back to original label
    st.write(f"Predicted Experience Level: {predicted_class[0]}")