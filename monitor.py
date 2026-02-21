import os
import joblib
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Thresholds
TUNE_THRESHOLD = 0.80
RETRAIN_THRESHOLD = 0.70

# Paths
DATA_PATH = "data/ipl auction.csv"   # IPL dataset
MODEL_PATH = "model/poly_linear_regression_pipeline.pkl"   # trained IPL model

FEATURE_COLS = ["Inns", "NO", "Avg", "BF", "4s", "6s"]

def retrain_model(target_column="Runs"):
    """Retrain polynomial regression (degree 2) and save model with joblib."""
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)

        # Check required columns
        for col in FEATURE_COLS + [target_column]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        X = df[FEATURE_COLS]
        y = df[target_column]

        if X.empty or y.empty:
            raise ValueError("Dataset is empty or missing values.")

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        # Save model and polynomial transformer together
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump((model, poly), MODEL_PATH)

        print("Model retrained and saved successfully.")
    except Exception as e:
        print(f"Error during retraining: {e}")

def monitor_model(target_column="Runs"):
    """Monitor IPL model performance and retrain/tune if needed."""
    try:
        if not os.path.exists(DATA_PATH):
            print(f"Dataset not found at {DATA_PATH}")
            return

        df = pd.read_csv(DATA_PATH)

        # Check required columns
        for col in FEATURE_COLS + [target_column]:
            if col not in df.columns:
                print(f"Missing required column: {col}")
                return

        if not os.path.exists(MODEL_PATH):
            print("No existing model found. Training a fresh one...")
            retrain_model(target_column)
            return

        # Load model
        model, poly = joblib.load(MODEL_PATH)

        # Prepare data
        X = df[FEATURE_COLS]
        y = df[target_column]
        X_poly = poly.transform(X)

        # Evaluate
        preds = model.predict(X_poly)
        score = r2_score(y, preds)
        print(f"Current IPL model R²: {score:.4f}")

        # Decision logic
        if score < RETRAIN_THRESHOLD:
            print("IPL model performance low — retraining...")
            retrain_model(target_column)

        elif score < TUNE_THRESHOLD:
            print("IPL model performance degraded — tuning (retrain with same data)...")
            retrain_model(target_column)

        else:
            print(" IPL model performance acceptable. No action taken.")

    except Exception as e:
        print(f" Error during monitoring: {e}")

if __name__ == "__main__":
    monitor_model()
