import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

REAL_DATA_PATH = "data/processed/clean_data.csv"
SYNTHETIC_DATA_PATH = "data/genai/synthetic_attacks.csv"

def load_and_prepare():
    print("Loading real dataset...")
    real_df = pd.read_csv(REAL_DATA_PATH)
    real_df.columns = real_df.columns.str.strip()

    print("Loading synthetic attack dataset...")
    synth_df = pd.read_csv(SYNTHETIC_DATA_PATH)
    synth_df.columns = synth_df.columns.str.strip()

    print("Real dataset shape:", real_df.shape)
    print("Synthetic dataset shape:", synth_df.shape)

    # Replace synthetic label with a real rare attack label
    synth_df["Label"] = "Infiltration"

    # Merge datasets
    combined_df = pd.concat([real_df, synth_df], ignore_index=True)
    print("Combined dataset shape:", combined_df.shape)

    # Drop non-feature columns
    drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
    for col in drop_cols:
        if col in combined_df.columns:
            combined_df = combined_df.drop(columns=[col])

    X = combined_df.drop(columns=["Label", "BinaryLabel"])
    y = combined_df["Label"]

    return X, y


def main():
    print("Starting GenAI-augmented IDS retraining...")

    # 1. Load and prepare data
    X, y = load_and_prepare()

    # 2. Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("\nClass distribution after augmentation:")
    print(pd.Series(y).value_counts())

    # 3. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 5. Train XGBoost multi-class classifier
    print("\nTraining XGBoost classifier...")
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(encoder.classes_),
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    # 6. Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nOverall Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # 7. Save model
    joblib.dump(model, "models/ids_xgboost_genai.pkl")
    joblib.dump(encoder, "models/label_encoder_genai.pkl")
    joblib.dump(scaler, "models/feature_scaler_genai.pkl")

    print("\nGenAI-augmented IDS model saved to /models/")

if __name__ == "__main__":
    main()
