from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "graph_enhanced_xgboost.joblib"
)


def load_prediction_model():
    """
    Load the saved preprocessing + XGBoost pipeline.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(
        MODEL_PATH
    )

    if not isinstance(artifact, dict):
        raise ValueError(
            "Saved model artifact has an unexpected format."
        )

    if "pipeline" not in artifact:
        raise ValueError(
            "Saved model does not contain a pipeline."
        )

    if "feature_columns" not in artifact:
        raise ValueError(
            "Saved model does not contain feature columns."
        )

    return (
        artifact["pipeline"],
        artifact["feature_columns"],
    )


def prepare_transaction(
    transaction: dict,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Convert a transaction dictionary into the
    exact dataframe structure expected by the model.
    """

    data = pd.DataFrame(
        [transaction]
    )

    # Find features that were present during
    # training but are not supplied for this
    # demonstration transaction.
    missing_columns = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    # Add missing columns as NumPy NaN.
    if missing_columns:

        missing_data = pd.DataFrame(
            np.nan,
            index=data.index,
            columns=missing_columns,
        )

        data = pd.concat(
            [
                data,
                missing_data,
            ],
            axis=1,
        )

    # Keep exactly the same feature order
    # used during training.
    data = data[
        feature_columns
    ].copy()

    # IMPORTANT:
    # Convert every pandas missing-value marker,
    # including pd.NA, to NumPy NaN.
    data = data.astype(object)

    data = data.where(
        pd.notna(data),
        np.nan,
    )

    return data


def predict_transaction(
    transaction: dict,
    threshold: float = 0.5,
):
    """
    Predict fraud probability and risk level
    for a single transaction.
    """

    pipeline, feature_columns = (
        load_prediction_model()
    )

    data = prepare_transaction(
        transaction,
        feature_columns,
    )

    probability = float(
        pipeline.predict_proba(
            data
        )[0][1]
    )

    prediction = int(
        probability >= threshold
    )

    if probability >= 0.75:
        risk_level = "HIGH"

    elif probability >= 0.40:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return {
        "fraud_probability": probability,
        "prediction": prediction,
        "risk_level": risk_level,
    }


def main():
    """
    Demonstration prediction using a sample
    transaction.
    """

    sample_transaction = {
        "TransactionID": 999999999,
        "TransactionDT": 999999999,

        "card1": 12345,
        "addr1": 100,
        "P_emaildomain": "gmail.com",
        "DeviceInfo": "ExampleDevice",

        "graph_entity_count": 4,
        "graph_known_entity_count": 3,
        "graph_shared_entity_count": 2,
        "graph_unique_entity_types": 4,
        "graph_avg_entity_degree": 5.0,
        "graph_max_entity_degree": 10,
        "graph_connected_component_size": 20,
    }

    result = predict_transaction(
        sample_transaction
    )

    print()
    print("=" * 60)
    print("FRAUD RISK PREDICTION")
    print("=" * 60)

    print(
        f"Fraud Probability: "
        f"{result['fraud_probability']:.4f}"
    )

    print(
        f"Fraud Prediction:  "
        f"{result['prediction']}"
    )

    print(
        f"Risk Level:         "
        f"{result['risk_level']}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()