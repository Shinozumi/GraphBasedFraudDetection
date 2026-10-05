from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TRANSACTION_FILE = RAW_DATA_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DATA_DIR / "train_identity.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
