
# ============================================================
# FRAUD TRANSACTION DETECTION USING MACHINE LEARNING
# Advanced Internship Project
# ============================================================

# ============================================================
# IMPORTING REQUIRED LIBRARIES
# ============================================================

import os
import glob
import logging
import warnings
import time
import gc

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE

import joblib

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION VARIABLES
# ============================================================

DATA_PATH = "data/*.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42
START_TIME = time.time()


# ============================================================
# CREATING REQUIRED FOLDERS
# ============================================================

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    filename="logs/project.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ============================================================
# FUNCTION TO LOAD DATASET
# ============================================================

def load_dataset():
    """
    Load and merge all PKL files.
    """

    try:

        file_list = glob.glob(DATA_PATH)

        if len(file_list) == 0:
            raise FileNotFoundError("No PKL files found inside data folder.")

        logging.info(f"Total Files Found: {len(file_list)}")

        dataframe_list = []

        # Reading all PKL files
        for file in file_list:

            print(f"Loading File: {file}")

            df = pd.read_pickle(file)

            dataframe_list.append(df)

        # Combining all datasets
        data = pd.concat(dataframe_list, ignore_index=True)

        # Free memory
        del dataframe_list
        gc.collect()

        logging.info("Dataset loaded successfully.")

        return data

    except FileNotFoundError as error:

        logging.error(error)

        print(error)

    except Exception as error:

        logging.error(error)

        print(error)


# ============================================================
# FUNCTION FOR DATA CLEANING
# ============================================================

def clean_data(data):
    """
    Perform data cleaning.
    """

    try:

        print("\nChecking Missing Values...")
        print(data.isnull().sum())

        # Remove duplicate rows
        duplicate_rows = data.duplicated().sum()

        print(f"Duplicate Rows Found: {duplicate_rows}")

        data.drop_duplicates(inplace=True)

        logging.info("Duplicate rows removed successfully.")

        return data

    except Exception as error:

        logging.error(error)

        print(error)


# ============================================================
# FUNCTION FOR FEATURE ENGINEERING
# ============================================================

def feature_engineering(data):
    """
    Create additional useful features.
    """

    try:

        # Convert datetime column
        data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'])

        # Time-based features
        data['TX_HOUR'] = data['TX_DATETIME'].dt.hour
        data['TX_DAY'] = data['TX_DATETIME'].dt.day
        data['TX_MONTH'] = data['TX_DATETIME'].dt.month
        data['TX_WEEKDAY'] = data['TX_DATETIME'].dt.weekday

        # Weekend indicator
        data['IS_WEEKEND'] = np.where(
            data['TX_WEEKDAY'] >= 5,
            1,
            0
        )

        # Log transform transaction amount
        # Prevents data leakage issue
        data['AMOUNT_LOG'] = np.log1p(data['TX_AMOUNT'])

        logging.info("Feature engineering completed successfully.")

        return data

    except Exception as error:

        logging.error(error)

        print(error)


# ============================================================
# FUNCTION FOR EXPLORATORY DATA ANALYSIS
# ============================================================

def perform_eda(data):
    """
    Create visualizations.
    """

    try:

        # ----------------------------------------------------
        # FRAUD DISTRIBUTION GRAPH
        # ----------------------------------------------------

        plt.figure(figsize=(6, 5))

        sns.countplot(x='TX_FRAUD', data=data)

        plt.title("Fraud vs Non-Fraud Transactions")

        plt.tight_layout()

        plt.savefig("outputs/plots/fraud_distribution.png")

        plt.close()


        # ----------------------------------------------------
        # TRANSACTION AMOUNT DISTRIBUTION
        # ----------------------------------------------------

        plt.figure(figsize=(10, 5))

        sns.histplot(data['TX_AMOUNT'], bins=50)

        plt.title("Transaction Amount Distribution")

        plt.tight_layout()

        plt.savefig(
            "outputs/plots/transaction_amount_distribution.png"
        )

        plt.close()


        # ----------------------------------------------------
        # FRAUD BY HOUR
        # ----------------------------------------------------

        plt.figure(figsize=(10, 5))

        sns.countplot(
            x='TX_HOUR',
            hue='TX_FRAUD',
            data=data
        )

        plt.title("Fraud Transactions by Hour")

        plt.tight_layout()

        plt.savefig("outputs/plots/fraud_by_hour.png")

        plt.close()


        # ----------------------------------------------------
        # CORRELATION HEATMAP
        # ----------------------------------------------------

        numeric_data = data.select_dtypes(include=np.number)

        plt.figure(figsize=(14, 10))

        sns.heatmap(
            numeric_data.corr(),
            cmap='coolwarm'
        )

        plt.title("Correlation Heatmap")

        plt.tight_layout()

        plt.savefig("outputs/plots/correlation_heatmap.png")

        plt.close()

        logging.info("EDA visualizations created successfully.")

    except Exception as error:

        logging.error(error)

        print(error)


# ============================================================
# FUNCTION FOR PREPROCESSING
# ============================================================

def preprocess_data(data):
    """
    Perform preprocessing.
    """

    try:

        # Drop datetime column
        data = data.drop(columns=['TX_DATETIME'])

        # Encode categorical columns
        encoder = LabelEncoder()

        for column in data.columns:

            if data[column].dtype == 'object':

                data[column] = encoder.fit_transform(
                    data[column]
                )

        # Feature and target separation
        X = data.drop('TX_FRAUD', axis=1)

        y = data['TX_FRAUD']

        logging.info("Preprocessing completed successfully.")

        return X, y

    except Exception as error:

        logging.error(error)

        print(error)


# ============================================================
# FUNCTION FOR TRAIN TEST SPLIT
# ============================================================

def split_dataset(X, y):
    """
    Split dataset into training and testing.
    """

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        logging.info("Train-test split completed successfully.")

        return X_train, X_test, y_train, y_test

    except Exception as error:

        logging.error(error)

        print(error)


# ============================================================
# FUNCTION FOR SMOTE BALANCING
# ============================================================

def apply_smote(X_train, y_train):
    """
    Apply SMOTE to balance dataset.
    """

    try:

        smote = SMOTE(random_state=RANDOM_STATE)

        X_resampled, y_resampled = smote.fit_resample(
            X_train,
            y_train
        )

        logging.info("SMOTE balancing completed successfully.")

        return X_resampled, y_resampled

    except Exception as error:

        logging.error(error)

        print(error)


# ============================================================
# FUNCTION FOR MODEL TRAINING
# ============================================================

def train_models(X_train, y_train):
    """
    Train multiple machine learning models.
    """

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    trained_models = {}

    try:

        for name, model in models.items():

            print(f"\nTraining {name}...")

            model.fit(X_train, y_train)

            trained_models[name] = model

            logging.info(f"{name} trained successfully.")

        return trained_models

    except Exception as error:

        logging.error(error)

        print(error)


# ============================================================
# FUNCTION FOR MODEL EVALUATION
# ============================================================

def evaluate_models(models, X_test, y_test, feature_names):
    """
    Evaluate trained models.
    """

    best_model = None
    best_accuracy = 0

    results = []

    try:

        for name, model in models.items():

            print(f"\nEvaluating {name}...")

            # Predictions
            y_pred = model.predict(X_test)

            y_prob = model.predict_proba(X_test)[:, 1]

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            # Save results
            results.append([
                name,
                accuracy,
                precision,
                recall,
                f1,
                roc_auc
            ])

            # Display metrics
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC-AUC Score: {roc_auc:.4f}")

            # Classification report
            report = classification_report(y_test, y_pred)

            report_path = (
                f"outputs/reports/{name}_classification_report.txt"
            )

            with open(report_path, "w") as file:

                file.write(report)


            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(6, 5))

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues'
            )

            plt.title(f"Confusion Matrix - {name}")

            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            plt.tight_layout()

            plt.savefig(
                f"outputs/plots/{name}_confusion_matrix.png"
            )

            plt.close()


            # ------------------------------------------------
            # ROC CURVE
            # ------------------------------------------------

            fpr, tpr, _ = roc_curve(y_test, y_prob)

            plt.figure(figsize=(8, 6))

            plt.plot(fpr, tpr)

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")

            plt.title(f"ROC Curve - {name}")

            plt.tight_layout()

            plt.savefig(
                f"outputs/plots/{name}_roc_curve.png"
            )

            plt.close()


            # ------------------------------------------------
            # BEST MODEL SELECTION
            # ------------------------------------------------

            if accuracy > best_accuracy:

                best_accuracy = accuracy

                best_model = model


        # ----------------------------------------------------
        # MODEL COMPARISON TABLE
        # ----------------------------------------------------

        results_df = pd.DataFrame(
            results,
            columns=[
                'Model',
                'Accuracy',
                'Precision',
                'Recall',
                'F1 Score',
                'ROC-AUC'
            ]
        )

        print("\nModel Comparison:")
        print(results_df)

        # Save model comparison
        results_df.to_csv(
            "outputs/reports/model_comparison.csv",
            index=False
        )


        # ----------------------------------------------------
        # FEATURE IMPORTANCE GRAPH
        # ----------------------------------------------------

        if isinstance(best_model, RandomForestClassifier):

            importance = best_model.feature_importances_

            importance_df = pd.DataFrame({

                'Feature': feature_names,

                'Importance': importance
            })

            importance_df = importance_df.sort_values(
                by='Importance',
                ascending=False
            )

            plt.figure(figsize=(12, 6))

            sns.barplot(
                x='Importance',
                y='Feature',
                data=importance_df.head(10)
            )

            plt.title("Top 10 Important Features")

            plt.tight_layout()

            plt.savefig(
                "outputs/plots/feature_importance.png"
            )

            plt.close()

        logging.info("Model evaluation completed successfully.")

        return best_model

    except Exception as error:

        logging.error(error)

        print(error)


# ============================================================
# FUNCTION FOR SAVING MODEL
# ============================================================

def save_model(model):
    """
    Save best trained model.
    """

    try:

        joblib.dump(
            model,
            "outputs/models/best_fraud_detection_model.pkl"
        )

        logging.info("Best model saved successfully.")

        print("\nBest model saved successfully.")

    except Exception as error:

        logging.error(error)

        print(error)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """
    Main execution function.
    """

    print("=" * 60)
    print("FRAUD TRANSACTION DETECTION PROJECT")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    data = load_dataset()

    # Check dataset loading
    if data is None:

        print("Dataset loading failed.")

        return


    # --------------------------------------------------------
    # DATA CLEANING
    # --------------------------------------------------------

    data = clean_data(data)


    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    data = feature_engineering(data)


    # --------------------------------------------------------
    # EXPLORATORY DATA ANALYSIS
    # --------------------------------------------------------

    perform_eda(data)


    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    X, y = preprocess_data(data)


    # --------------------------------------------------------
    # FRAUD PERCENTAGE
    # --------------------------------------------------------

    fraud_percentage = (y.mean()) * 100

    print(f"\nFraud Percentage: {fraud_percentage:.2f}%")


    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )


    # --------------------------------------------------------
    # APPLY SMOTE
    # --------------------------------------------------------

    X_train, y_train = apply_smote(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    models = train_models(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # EVALUATE MODELS
    # --------------------------------------------------------

    best_model = evaluate_models(
        models,
        X_test,
        y_test,
        X.columns
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    save_model(best_model)


    # --------------------------------------------------------
    # EXECUTION TIME
    # --------------------------------------------------------

    end_time = time.time()

    execution_time = (end_time - START_TIME) / 60

    print(f"\nExecution Time: {execution_time:.2f} minutes")

    print("\nProject Completed Successfully.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
