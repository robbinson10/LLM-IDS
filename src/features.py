import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/processed/clean_data.csv"


def load_data(path):
    print(f"Loading processed data from {path}")
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    return df


def select_features(df):
    """
    Drop non-numeric or irrelevant columns.
    Keep only numerical features for ML models.
    """
    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Columns to drop (IDs, labels)
    drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    print("Shape after dropping non-feature columns:", df.shape)
    return df


def prepare_targets(df):
    """
    Separate features (X) and targets (y).
    """
    if "BinaryLabel" not in df.columns:
        raise ValueError("BinaryLabel column not found. Run ETL first.")

    y_binary = df["BinaryLabel"]
    X = df.drop(columns=["BinaryLabel"])

    print("Features shape:", X.shape)
    print("Binary target shape:", y_binary.shape)

    return X, y_binary


def scale_features(X):
    """
    Standardize features (mean=0, std=1).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Train-test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Train set:", X_train.shape, y_train.shape)
    print("Test set:", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


def main():
    print("Starting feature engineering...")

    # 1. Load processed data
    df = load_data(DATA_PATH)

    # 2. Select useful features
    df_features = select_features(df)

    # 3. Prepare targets
    X, y_binary = prepare_targets(df_features)

    # 4. Scale features
    X_scaled = scale_features(X)

    # 5. Train-test split
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_binary)

    print("Feature engineering complete.")


if __name__ == "__main__":
    main()
