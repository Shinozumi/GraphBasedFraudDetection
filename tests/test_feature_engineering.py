import pandas as pd

from src.feature_engineering import (
    add_transaction_amount_features,
    add_time_features,
    add_email_features,
    add_card_features,
    add_missingness_features,
)


def make_test_data():

    return pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "isFraud": [0, 1, 0],
        "TransactionDT": [
            3600,
            90000,
            172800,
        ],
        "TransactionAmt": [
            100.00,
            25.50,
            999.99,
        ],
        "P_emaildomain": [
            "gmail.com",
            "yahoo.com",
            "gmail.com",
        ],
        "R_emaildomain": [
            "gmail.com",
            "gmail.com",
            "gmail.com",
        ],
        "card1": [
            1111,
            None,
            3333,
        ],
        "card2": [
            222,
            None,
            444,
        ],
        "card3": [
            150,
            150,
            None,
        ],
    })


def test_amount_features():

    data = make_test_data()

    result = add_transaction_amount_features(
        data
    )

    assert "TransactionAmt_log" in result.columns
    assert "TransactionAmt_decimal" in result.columns
    assert "TransactionAmt_is_round" in result.columns

    assert result.loc[
        0,
        "TransactionAmt_is_round"
    ] == 1

    assert result.loc[
        1,
        "TransactionAmt_is_round"
    ] == 0


def test_time_features():

    data = make_test_data()

    result = add_time_features(
        data
    )

    assert "Transaction_hour" in result.columns
    assert "Transaction_day" in result.columns
    assert "Transaction_day_of_week" in result.columns
    assert "Transaction_is_night" in result.columns
    assert "Transaction_is_business_hours" in result.columns

    assert result.loc[
        0,
        "Transaction_hour"
    ] == 1


def test_email_features():

    data = make_test_data()

    result = add_email_features(
        data
    )

    assert "email_domain_match" in result.columns

    assert result.loc[
        0,
        "email_domain_match"
    ] == 1

    assert result.loc[
        1,
        "email_domain_match"
    ] == 0


def test_card_features():

    data = make_test_data()

    result = add_card_features(
        data
    )

    assert "card_missing_count" in result.columns

    assert result.loc[
        1,
        "card_missing_count"
    ] == 2


def test_missingness_features():

    data = make_test_data()

    result = add_missingness_features(
        data
    )

    assert (
        "total_missing_features"
        in result.columns
    )