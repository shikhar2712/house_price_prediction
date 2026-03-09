"""
Train House Price Prediction Model
----------------------------------
This script loads the dataset, preprocesses the data,
trains a Random Forest model, evaluates it, and saves
the trained model for future predictions.
"""

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw.csv"
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    return df


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_data(df):

    print("Preprocessing data...")

    # Remove ID column if present
    df = df.drop(columns=["PID"], errors="ignore")

    # Separate target
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    return X, y


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(X_train, y_train):

    print("Training Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    print("Evaluating model...")

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    print(f"Model R² Score: {r2:.4f}")

    return r2


# --------------------------------------------------
# Save Model
# --------------------------------------------------

def save_model(model):

    print("Saving trained model...")

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved at: {MODEL_PATH}")


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def main():

    df = load_data()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()