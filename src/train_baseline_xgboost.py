import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib
import os

DATA_PATH = "data/processed/clean_data.csv"


# -----------------------------
# Load Data
# -----------------------------
def load_data(path):
    print(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    return df


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(df):
    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop non-feature columns
    drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Target
    if "Label" not in df.columns:
        raise ValueError("Label column not found for multi-class classification")

    y = df["Label"]
    X = df.drop(columns=["Label", "BinaryLabel"])

    return X, y


# -----------------------------
# Encode Labels
# -----------------------------
def encode_labels(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("\nClasses:")
    for idx, label in enumerate(encoder.classes_):
        print(f"{idx} -> {label}")

    return y_encoded, encoder


# -----------------------------
# Feature Scaling
# -----------------------------
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# -----------------------------
# Train XGBoost Model
# -----------------------------
def train_model(X_train, y_train):
    print("Training XGBoost (Baseline, No GenAI)...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y_train)),
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, X_test, y_test, encoder):
    print("\nEvaluating baseline model...")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nOverall Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# -----------------------------
# Main
# -----------------------------
def main():
    print("Starting BASELINE IDS training (Before GenAI)...")

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Preprocess
    X, y = preprocess(df)
    print("Feature matrix:", X.shape)
    print("Target vector:", y.shape)

    # 3. Encode labels
    y_encoded, encoder = encode_labels(y)

    # 4. Scale features
    X_scaled, scaler = scale_features(X)

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Train set:", X_train.shape)
    print("Test set:", X_test.shape)

    # 6. Train model
    model = train_model(X_train, y_train)

    # 7. Evaluate
    evaluate_model(model, X_test, y_test, encoder)

    # -----------------------------
    # Save Baseline Artifacts
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/ids_xgboost_baseline.pkl")
    joblib.dump(encoder, "models/label_encoder_baseline.pkl")
    joblib.dump(scaler, "models/feature_scaler_baseline.pkl")

    print("\nBaseline model artifacts saved to /models/")
    print("✔ ids_xgboost_baseline.pkl")
    print("✔ label_encoder_baseline.pkl")
    print("✔ feature_scaler_baseline.pkl")


if __name__ == "__main__":
    main()
