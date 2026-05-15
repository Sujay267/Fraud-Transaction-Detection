# Fraud Transaction Detection Using Machine Learning

## Project Overview

This project is an advanced machine learning-based fraud transaction detection system developed using Python.

The system analyzes transaction datasets and predicts whether a transaction is fraudulent or legitimate using multiple machine learning algorithms.

The project includes:
- Data cleaning
- Feature engineering
- Exploratory Data Analysis (EDA)
- SMOTE balancing
- Multiple ML models
- Model evaluation
- Confusion matrix
- ROC curve analysis
- Feature importance analysis
- Model saving
- Logging system

---

# Dataset Information

The dataset consists of 183 `.pkl` transaction files.

Main columns:
- TRANSACTION_ID
- TX_DATETIME
- CUSTOMER_ID
- TERMINAL_ID
- TX_AMOUNT
- TX_FRAUD

Fraud labels are simulated using transaction behavior patterns.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SMOTE
- Joblib

---

# Machine Learning Models Used

1. Logistic Regression
2. Decision Tree
3. Random Forest

---

# Features Implemented

## Data Processing
- Missing value checking
- Duplicate removal
- Datetime feature extraction
- Log transformation

## Feature Engineering
- Transaction hour
- Transaction day
- Transaction month
- Weekday extraction
- Weekend indicator
- Amount log transformation

## Data Visualization
- Fraud distribution graph
- Transaction amount distribution
- Fraud by hour graph
- Correlation heatmap
- Feature importance graph
- ROC curves
- Confusion matrices

## Machine Learning
- SMOTE balancing
- Multi-model training
- Accuracy evaluation
- Precision evaluation
- Recall evaluation
- F1-score evaluation
- ROC-AUC evaluation

---

# Project Structure

```text
Fraud_Transaction_Detection/
│
├── data/
├── outputs/
├── logs/
├── final_fraud_detection_project.py
├── requirements.txt
└── README.md
