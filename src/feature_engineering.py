from __future__ import annotations

import numpy as np
import pandas as pd


TARGET = "isFraud"
ID_COLUMN = "TransactionID"
TIME_COLUMN = "TransactionDT"


def add_transaction_amount_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create features derived from TransactionAmt.
    """

    data = data.copy()

    if "TransactionAmt" not in data.columns:
        return data

    amount = data["TransactionAmt"]

    # Log transformation reduces the effect of extremely large amounts.
    data["TransactionAmt_log"] = np.log1p(
        amount.clip(lower=0)
    )

    # Fractional/cents component.
    data["TransactionAmt_decimal"] = (
        amount - np.floor(amount)
    )

    # Whether the amount is a round integer amount.
    data["TransactionAmt_is_round"] = (
        data["TransactionAmt_decimal"] == 0
    ).astype("int8")

    return data


def add_time_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create interpretable time features from TransactionDT.

    TransactionDT is measured as elapsed seconds.
    """

    data = data.copy()

    if TIME_COLUMN not in data.columns:
        return data

    transaction_dt = data[TIME_COLUMN]

    # Seconds within a day.
    seconds_in_day = transaction_dt % 86400

    # Approximate hour of transaction.
    data["Transaction_hour"] = (
        seconds_in_day // 3600
    ).astype("int16")

    # Day index from the beginning of the dataset.
    data["Transaction_day"] = (
        transaction_dt // 86400
    ).astype("int32")

    # Day of week assuming the first TransactionDT corresponds
    # to the beginning of the first reference day.
    data["Transaction_day_of_week"] = (
        data["Transaction_day"] % 7
    ).astype("int8")

    # Simple time-of-day groups.
    data["Transaction_is_night"] = (
        (data["Transaction_hour"] < 6)
        | (data["Transaction_hour"] >= 22)
    ).astype("int8")

    data["Transaction_is_business_hours"] = (
        (data["Transaction_hour"] >= 9)
        & (data["Transaction_hour"] < 18)
    ).astype("int8")

    return data


def add_email_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create features describing purchaser/recipient email information.
    """

    data = data.copy()

    if (
        "P_emaildomain" in data.columns
        and "R_emaildomain" in data.columns
    ):
        purchaser = (
            data["P_emaildomain"]
            .fillna("__MISSING__")
            .astype(str)
        )

        recipient = (
            data["R_emaildomain"]
            .fillna("__MISSING__")
            .astype(str)
        )

        data["email_domain_match"] = (
            purchaser == recipient
        ).astype("int8")

    return data


def add_card_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create simple card-related consistency features.
    """

    data = data.copy()

    card_columns = [
        column
        for column in [
            "card1",
            "card2",
            "card3",
            "card5",
        ]
        if column in data.columns
    ]

    if card_columns:
        data["card_missing_count"] = (
            data[card_columns]
            .isna()
            .sum(axis=1)
            .astype("int8")
        )

    if "card4" in data.columns and "card6" in data.columns:
        data["card_type_missing"] = (
            data["card4"].isna()
            | data["card6"].isna()
        ).astype("int8")

    return data


def add_address_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create address-related missingness features.
    """

    data = data.copy()

    address_columns = [
        column
        for column in [
            "addr1",
            "addr2",
            "dist1",
            "dist2",
        ]
        if column in data.columns
    ]

    if address_columns:
        data["address_missing_count"] = (
            data[address_columns]
            .isna()
            .sum(axis=1)
            .astype("int8")
        )

    return data


def add_missingness_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a general missing-value count.

    This is useful because the pattern of missing fields can itself
    contain information about a transaction.
    """

    data = data.copy()

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

    data["total_missing_features"] = (
        data[feature_columns]
        .isna()
        .sum(axis=1)
        .astype("int16")
    )

    return data


def add_all_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply all Step 3 feature engineering operations.
    """

    data = data.copy()

    data = add_transaction_amount_features(data)
    data = add_time_features(data)
    data = add_email_features(data)
    data = add_card_features(data)
    data = add_address_features(data)
    data = add_missingness_features(data)

    return data