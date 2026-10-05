import pandas as pd

from src.config import TRANSACTION_FILE, IDENTITY_FILE


def load_transaction_data():
    if not TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found:\n{TRANSACTION_FILE}\n\n"
            "Please place train_transaction.csv inside data/raw/"
        )

    print("Loading transaction data...")
    transactions = pd.read_csv(TRANSACTION_FILE)
    print("Transaction dataset loaded.")
    print(f"Shape: {transactions.shape}")
    return transactions


def load_identity_data():
    if not IDENTITY_FILE.exists():
        raise FileNotFoundError(
            f"Identity dataset not found:\n{IDENTITY_FILE}\n\n"
            "Please place train_identity.csv inside data/raw/"
        )

    print("Loading identity data...")
    identity = pd.read_csv(IDENTITY_FILE)
    print("Identity dataset loaded.")
    print(f"Shape: {identity.shape}")
    return identity


def load_dataset():
    transactions = load_transaction_data()
    identity = load_identity_data()

    print("\nMerging transaction and identity data...")

    data = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    print("Merge completed.")
    print(f"Final shape: {data.shape}")
    return data


if __name__ == "__main__":
    data = load_dataset()
    print("\nFirst 5 rows:")
    print(data.head())

    print("\nTarget distribution:")
    if "isFraud" in data.columns:
        print(data["isFraud"].value_counts())
