from __future__ import annotations

import pandas as pd

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

from src.graph_features import (
    load_graph_features,
)

from src.model import (
    build_training_pipeline,
    save_model,
)

from src.evaluation import (
    evaluate_model,
    threshold_analysis,
    save_metrics,
    save_evaluation_plots,
)

from src.error_analysis import (
    analyze_errors,
)

TRAIN_GRAPH_FEATURE_PATH = (
    MODEL_DIR
    / "training_graph_features.csv"
)

HOLDOUT_GRAPH_FEATURE_PATH = (
    MODEL_DIR
    / "holdout_graph_features.csv"
)


GRAPH_MODEL_PATH = (
    MODEL_DIR
    / "graph_enhanced_xgboost.joblib"
)

GRAPH_METRICS_PATH = (
    REPORT_DIR
    / "graph_enhanced_metrics.json"
)


GRAPH_FEATURE_COLUMNS = [
    "graph_entity_count",
    "graph_known_entity_count",
    "graph_shared_entity_count",
    "graph_unique_entity_types",
    "graph_avg_entity_degree",
    "graph_max_entity_degree",
    "graph_connected_component_size",
]


def attach_graph_features(
    data: pd.DataFrame,
    graph_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge graph features into transaction data.
    """

    result = data.merge(
        graph_features,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    for column in GRAPH_FEATURE_COLUMNS:

        result[column] = (
            result[column]
            .fillna(0)
        )

    return result


def main():

    print("=" * 60)
    print("GRAPH-BASED FINANCIAL FRAUD DETECTION")
    print("STEP 9 - THRESHOLD ANALYSIS")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load and clean data
    # ---------------------------------------------------------

    print(
        "\n[1/8] Loading dataset..."
    )

    data = load_dataset()

    data = basic_cleaning(
        data
    )

    # ---------------------------------------------------------
    # 2. Feature engineering
    # ---------------------------------------------------------

    print(
        "\n[2/8] Creating standard features..."
    )

    data = add_all_features(
        data
    )

    # ---------------------------------------------------------
    # 3. Chronological split
    # ---------------------------------------------------------

    print(
        "\n[3/8] Creating chronological split..."
    )

    train_data, holdout_data = (
        chronological_split(
            data
        )
    )

    # ---------------------------------------------------------
    # 4. Load graph features
    # ---------------------------------------------------------

    print(
        "\n[4/8] Loading graph features..."
    )

    train_graph_features = (
        load_graph_features(
            TRAIN_GRAPH_FEATURE_PATH
        )
    )

    holdout_graph_features = (
        load_graph_features(
            HOLDOUT_GRAPH_FEATURE_PATH
        )
    )

    # ---------------------------------------------------------
    # 5. Merge graph features
    # ---------------------------------------------------------

    print(
        "\n[5/8] Attaching graph features..."
    )

    train_data = attach_graph_features(
        train_data,
        train_graph_features,
    )

    holdout_data = attach_graph_features(
        holdout_data,
        holdout_graph_features,
    )

    print(
        f"Training shape: "
        f"{train_data.shape}"
    )

    print(
        f"Holdout shape: "
        f"{holdout_data.shape}"
    )

    # ---------------------------------------------------------
    # Verify graph features
    # ---------------------------------------------------------

    print(
        "\nGraph feature statistics:"
    )

    print(
        train_data[
            GRAPH_FEATURE_COLUMNS
        ].describe()
    )

    print(
        "\nHoldout graph feature statistics:"
    )

    print(
        holdout_data[
            GRAPH_FEATURE_COLUMNS
        ].describe()
    )

    # ---------------------------------------------------------
    # 6. Build and train model
    # ---------------------------------------------------------

    print(
        "\n[6/8] Building graph-enhanced XGBoost..."
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

    X_holdout = holdout_data[
        feature_columns
    ]

    y_holdout = holdout_data[
        "isFraud"
    ]

    print(
        "\nTraining graph-enhanced model..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 7. Evaluate
    # ---------------------------------------------------------

    print(
        "\n[7/8] Evaluating holdout set..."
    )

    probabilities = (
        pipeline
        .predict_proba(
            X_holdout
        )[:, 1]
    )

    metrics = evaluate_model(
        y_holdout,
        probabilities,
        threshold=0.5,
    )

    # ---------------------------------------------------------
    # Threshold analysis
    # ---------------------------------------------------------

    print(
        "\nRunning threshold analysis..."
    )

    threshold_results = threshold_analysis(
        y_holdout,
        probabilities,
        str(REPORT_DIR),
    )

    print(
        "\nRunning error analysis..."
    )

    error_summary = analyze_errors(
        holdout_data,
        probabilities,
        threshold=0.5,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print(
        "\n[8/8] Saving model and reports..."
    )

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
            GRAPH_MODEL_PATH
        ),
    )

    save_metrics(
        metrics,
        str(
            GRAPH_METRICS_PATH
        ),
    )

    save_evaluation_plots(
        y_holdout,
        probabilities,
        str(REPORT_DIR),
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "GRAPH-ENHANCED MODEL COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nModel:"
    )

    print(
        GRAPH_MODEL_PATH
    )

    print(
        "\nMetrics:"
    )

    print(
        GRAPH_METRICS_PATH
    )

    print(
        "\nThreshold analysis:"
    )

    print(
        REPORT_DIR
        / "threshold_analysis.csv"
    )

    print(
        REPORT_DIR
        / "threshold_analysis.png"
    )


if __name__ == "__main__":
    main()