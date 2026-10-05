from __future__ import annotations

import joblib
import pandas as pd

from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from xgboost import XGBClassifier


TARGET = "isFraud"

EXCLUDED_COLUMNS = {
    TARGET,
    "TransactionID",
    "TransactionDT",
}


def get_model_columns(
    data: pd.DataFrame,
):
    """
    Determine the columns supplied to the model.

    Target, transaction ID and raw transaction time are excluded.

    Object/string columns are categorical.
    Everything else is treated as numerical.
    """

    feature_columns = [
        column
        for column in data.columns
        if column not in EXCLUDED_COLUMNS
    ]

    categorical_columns = [
        column
        for column in feature_columns
        if (
            is_object_dtype(data[column])
            or is_string_dtype(data[column])
        )
    ]

    numerical_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
    ]

    return (
        feature_columns,
        numerical_columns,
        categorical_columns,
    )


def build_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str],
):
    """
    Build the training preprocessing pipeline.

    Numerical columns:
        median imputation

    Categorical columns:
        most-frequent imputation
        ordinal encoding

    The fitted statistics are learned only from the training set.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def build_xgboost_model(
    y_train: pd.Series,
):
    """
    Create the XGBoost fraud classifier.

    scale_pos_weight compensates for the strong class imbalance.
    """

    positive_count = int(
        (y_train == 1).sum()
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    if positive_count == 0:
        raise ValueError(
            "Training data contains no fraud examples."
        )

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        "\nXGBoost class imbalance configuration:"
    )

    print(
        f"Non-fraud examples: {negative_count:,}"
    )

    print(
        f"Fraud examples:     {positive_count:,}"
    )

    print(
        f"scale_pos_weight:    "
        f"{scale_pos_weight:.4f}"
    )

    model = XGBClassifier(
        objective="binary:logistic",

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.85,

        colsample_bytree=0.85,

        min_child_weight=3,

        reg_lambda=2.0,

        reg_alpha=0.0,

        scale_pos_weight=scale_pos_weight,

        eval_metric="aucpr",

        tree_method="hist",

        max_bin=256,

        n_jobs=-1,

        random_state=42,
    )

    return model


def build_training_pipeline(
    train_data: pd.DataFrame,
):
    """
    Construct the complete baseline ML pipeline.
    """

    (
        feature_columns,
        numerical_columns,
        categorical_columns,
    ) = get_model_columns(
        train_data
    )

    print("\nModel feature configuration:")

    print(
        f"Total features:       "
        f"{len(feature_columns)}"
    )

    print(
        f"Numerical features:   "
        f"{len(numerical_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    print("\nCategorical columns:")

    for column in categorical_columns:
        print(f"  - {column}")

    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns,
    )

    model = build_xgboost_model(
        train_data[TARGET]
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return (
        pipeline,
        feature_columns,
    )


def save_model(
    pipeline,
    feature_columns: list[str],
    path: str,
):
    """
    Save the complete preprocessing + model pipeline.
    """

    artifact = {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
    }

    joblib.dump(
        artifact,
        path,
    )

    print(
        f"\nModel saved to: {path}"
    )


def load_model(path: str):
    """
    Load a previously saved model.
    """

    return joblib.load(path)