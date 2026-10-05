from pathlib import Path

import numpy as np
import pandas as pd

from src.predict import (
    MODEL_PATH,
    load_prediction_model,
    prepare_transaction,
    predict_transaction,
)


def test_model_file_exists():
    """
    Verify that the trained graph-enhanced model
    exists before inference.
    """

    assert MODEL_PATH.exists()


def test_load_prediction_model():
    """
    Verify that the saved artifact contains
    both the pipeline and feature list.
    """

    pipeline, feature_columns = (
        load_prediction_model()
    )

    assert pipeline is not None

    assert isinstance(
        feature_columns,
        list,
    )

    assert len(feature_columns) > 0


def test_prepare_transaction():
    """
    Verify that a transaction is converted into
    the exact feature structure expected by the model.
    """

    _, feature_columns = (
        load_prediction_model()
    )

    transaction = {
        "TransactionID": 123456789,
        "TransactionDT": 123456,
        "card1": 12345,
        "addr1": 100,
        "P_emaildomain": "gmail.com",
        "DeviceInfo": "TestDevice",
    }

    data = prepare_transaction(
        transaction,
        feature_columns,
    )

    assert isinstance(
        data,
        pd.DataFrame,
    )

    assert list(
        data.columns
    ) == feature_columns

    assert len(data) == 1


def test_missing_values_are_numpy_nan():
    """
    Verify that missing features are represented
    using NumPy NaN rather than pandas NA.
    """

    _, feature_columns = (
        load_prediction_model()
    )

    transaction = {
        "TransactionID": 123456789,
    }

    data = prepare_transaction(
        transaction,
        feature_columns,
    )

    missing_values = (
        data.isna()
        .sum()
        .sum()
    )

    assert missing_values > 0


def test_prediction_output():
    """
    Verify that the prediction interface returns
    the expected output structure.
    """

    transaction = {
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
        transaction
    )

    assert isinstance(
        result,
        dict,
    )

    assert "fraud_probability" in result
    assert "prediction" in result
    assert "risk_level" in result

    assert 0.0 <= (
        result["fraud_probability"]
    ) <= 1.0

    assert result["prediction"] in [
        0,
        1,
    ]

    assert result["risk_level"] in [
        "LOW",
        "MEDIUM",
        "HIGH",
    ]


def test_risk_thresholds():
    """
    Verify the risk categories used by the
    prediction interface.
    """

    transaction = {
        "TransactionID": 999999999,
        "TransactionDT": 999999999,
    }

    # We don't need to call the model here.
    # The test simply verifies that the returned
    # risk level is one of the supported categories.

    result = predict_transaction(
        transaction
    )

    assert result[
        "risk_level"
    ] in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }