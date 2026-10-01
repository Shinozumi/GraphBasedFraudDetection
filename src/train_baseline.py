from __future__ import annotations

from pathlib import Path

from src.config import (
    MODEL_DIR,
    REPORT_DIR,
)

from src.data_loader import load_dataset

from src.preprocess import (
    basic_cleaning,
    chronological_split,
)

from src.feature_engineering import (
    add_all_features,
)

from src.model import (
    build_training_pipeline,
    save_model,
)

from src.evaluation import (
    evaluate_model,
    save_metrics,
    save_evaluation_plots,
)


def main():

    print("=" * 60)
    print("GRAPH-BASED FINANCIAL FRAUD DETECTION")
    print("STEP 4 - BASELINE XGBOOST MODEL")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load raw data
    # ---------------------------------------------------------

    print("\n[1/7] Loading dataset...")

    data = load_dataset()

    # ---------------------------------------------------------
    # 2. Basic cleaning
    # ---------------------------------------------------------

    print("\n[2/7] Cleaning dataset...")

    data = basic_cleaning(
        data
    )

    # ---------------------------------------------------------
    # 3. Feature engineering
    # ---------------------------------------------------------

    print(
        "\n[3/7] Creating engineered features..."
    )

    data = add_all_features(
        data
    )

    # ---------------------------------------------------------
    # 4. Chronological split
    # ---------------------------------------------------------

    print(
        "\n[4/7] Creating chronological "
        "train/holdout split..."
    )

    train_data, test_data = (
        chronological_split(
            data
        )
    )

    print(
        f"Training rows: "
        f"{len(train_data):,}"
    )

    print(
        f"Holdout rows:  "
        f"{len(test_data):,}"
    )

    # ---------------------------------------------------------
    # 5. Build model
    # ---------------------------------------------------------

    print(
        "\n[5/7] Building XGBoost pipeline..."
    )

    pipeline, feature_columns = (
        build_training_pipeline(
            train_data
        )
    )

    X_train = train_data[
        feature_columns
    ]

    y_train = train_data[
        "isFraud"
    ]

    X_test = test_data[
        feature_columns
    ]

    y_test = test_data[
        "isFraud"
    ]

    # ---------------------------------------------------------
    # 6. Train
    # ---------------------------------------------------------

    print(
        "\n[6/7] Training baseline model..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "Training completed."
    )

    # ---------------------------------------------------------
    # 7. Evaluate
    # ---------------------------------------------------------

    print(
        "\n[7/7] Evaluating holdout set..."
    )

    probabilities = (
        pipeline
        .predict_proba(X_test)[:, 1]
    )

    metrics = evaluate_model(
        y_test,
        probabilities,
        threshold=0.5,
    )

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_model(
        pipeline,
        feature_columns,
        str(
            MODEL_DIR
            / "baseline_xgboost.joblib"
        ),
    )

    save_metrics(
        metrics,
        str(
            REPORT_DIR
            / "baseline_metrics.json"
        ),
    )

    save_evaluation_plots(
        y_test,
        probabilities,
        str(REPORT_DIR),
    )

    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)

    print(
        "\nSaved files:"
    )

    print(
        "models/baseline_xgboost.joblib"
    )

    print(
        "reports/baseline_metrics.json"
    )

    print(
        "reports/baseline_roc_curve.png"
    )

    print(
        "reports/baseline_precision_recall_curve.png"
    )


if __name__ == "__main__":
    main()