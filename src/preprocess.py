from __future__ import annotations

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from src.config import TEST_SIZE
from src.data_loader import load_dataset
from src.feature_engineering import add_all_features


TARGET = "isFraud"
ID_COLUMN = "TransactionID"
TIME_COLUMN = "TransactionDT"


def basic_cleaning(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply preprocessing that does not learn statistics from the data.
    """

    data = data.copy()

    duplicate_count = (
        data[ID_COLUMN]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        print(
            f"Removing {duplicate_count:,} duplicate "
            f"TransactionID rows."
        )

        data = data.drop_duplicates(
            subset=ID_COLUMN,
            keep="first",
        )

    data = data.sort_values(
        by=[
            TIME_COLUMN,
            ID_COLUMN,
        ]
    ).reset_index(drop=True)

    if TARGET in data.columns:
        data[TARGET] = (
            data[TARGET]
            .astype("int8")
        )

    return data


def chronological_split(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
):
    """
    Split transactions chronologically.

    Older transactions -> training
    Newer transactions -> holdout
    """

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    split_index = int(
        len(data) * (1 - test_size)
    )

    train_data = (
        data.iloc[:split_index]
        .copy()
    )

    test_data = (
        data.iloc[split_index:]
        .copy()
    )

    return train_data, test_data


def get_column_groups(
    data: pd.DataFrame,
):
    """
    Separate model-input columns into numerical
    and categorical groups.
    """

    excluded = {
        TARGET,
        ID_COLUMN,
        TIME_COLUMN,
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded
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
        numerical_columns,
        categorical_columns,
    )


def print_preprocessing_report(
    full_data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
):
    numerical_columns, categorical_columns = (
        get_column_groups(train_data)
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING + FEATURE ENGINEERING REPORT")
    print("=" * 60)

    print(
        f"Full dataset:  {full_data.shape}"
    )

    print(
        f"Training set: {train_data.shape}"
    )

    print(
        f"Holdout set:  {test_data.shape}"
    )

    print("\nColumn groups:")

    print(
        f"Numerical columns:   "
        f"{len(numerical_columns)}"
    )

    print(
        f"Categorical columns: "
        f"{len(categorical_columns)}"
    )

    missing = (
        train_data
        .isna()
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print("\nMissing values:")

    print(
        f"Columns with missing values: "
        f"{(missing > 0).sum()}"
    )

    print(
        f"Columns with >95% missing: "
        f"{(missing > 0.95).sum()}"
    )

    print("\nFraud distribution:")

    for name, frame in [
        ("Training", train_data),
        ("Holdout", test_data),
    ]:

        counts = (
            frame[TARGET]
            .value_counts()
        )

        fraud_count = int(
            counts.get(1, 0)
        )

        non_fraud_count = int(
            counts.get(0, 0)
        )

        fraud_rate = (
            fraud_count / len(frame)
        )

        print(
            f"{name}: "
            f"fraud={fraud_count:,}, "
            f"non-fraud={non_fraud_count:,}, "
            f"fraud_rate={fraud_rate:.4%}"
        )

    print("\nChronological boundary:")

    print(
        f"Training last TransactionDT: "
        f"{train_data[TIME_COLUMN].max()}"
    )

    print(
        f"Holdout first TransactionDT: "
        f"{test_data[TIME_COLUMN].min()}"
    )

    print("=" * 60)


def prepare_data():
    """
    Complete preprocessing and feature-engineering pipeline.
    """

    print("Loading dataset...")

    data = load_dataset()

    print("\nApplying basic cleaning...")

    data = basic_cleaning(data)

    print(
        "\nApplying feature engineering..."
    )

    data = add_all_features(data)

    print(
        "Creating chronological "
        "train/holdout split..."
    )

    train_data, test_data = (
        chronological_split(data)
    )

    print_preprocessing_report(
        data,
        train_data,
        test_data,
    )

    return (
        train_data,
        test_data,
    )


if __name__ == "__main__":
    prepare_data()