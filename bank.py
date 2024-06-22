import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('bank_dataset.csv')
    return data

data = load_data()

# Data exploration
st.title('Bank Fraud Detection - Data Exploration')

# Display first few rows of the dataset
st.subheader('Dataset Overview')
st.write(data.head())

# Summary statistics
st.subheader('Summary Statistics')
st.write(data.describe())

# Class distribution
st.subheader('Class Distribution')
class_counts = data['label'].value_counts()
st.bar_chart(class_counts)

# Correlation matrix
st.subheader('Correlation Matrix')
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot()

# Data preprocessing
X = data.drop('label', axis=1)
y = data['label']

# Handle imbalanced data
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selection and training
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC()
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
    
    st.write(f"Model: {name}")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    st.write()

st.write(f"Best Model: {type(best_model).__name__}")
st.write(f"Best Accuracy: {best_accuracy}")
st.write()

# Hyperparameter tuning (Random Forest as an example)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier()
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

st.write("Best Hyperparameters for Random Forest:")
st.write(grid_search.best_params_)
st.write()

# Evaluation of best model
y_pred_best = grid_search.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)

st.write("Best Model Evaluation:")
st.write(f"Accuracy: {accuracy_best}")
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred_best))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_best))
