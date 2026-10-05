import pandas as pd

from src.preprocess import (
    basic_cleaning,
    chronological_split,
    get_column_groups,
)


def make_test_data():
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "isFraud": [0, 1, 0, 0, 1],
        "TransactionDT": [100, 200, 300, 400, 500],
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0],
        "ProductCD": ["W", "C", "W", "R", "W"],
    })


def test_basic_cleaning_sorts_data():
    data = make_test_data().iloc[[4, 0, 2, 1, 3]]
    cleaned = basic_cleaning(data)
    assert cleaned["TransactionDT"].tolist() == [100, 200, 300, 400, 500]


def test_chronological_split():
    data = make_test_data()
    train, test = chronological_split(data, test_size=0.4)

    assert len(train) == 3
    assert len(test) == 2
    assert train["TransactionDT"].max() < test["TransactionDT"].min()


def test_column_groups():
    data = make_test_data()
    numerical, categorical = get_column_groups(data)

    assert "TransactionAmt" in numerical
    assert "ProductCD" in categorical
    assert "isFraud" not in numerical
    assert "TransactionID" not in numerical
    assert "TransactionDT" not in numerical
