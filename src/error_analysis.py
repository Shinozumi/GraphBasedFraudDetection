from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def analyze_errors(
    holdout_data: pd.DataFrame,
    probabilities,
    threshold: float = 0.5,
):
    """
    Analyze false positives and false negatives
    produced by the fraud detection model.

    This is diagnostic analysis only.
    It does not change or retrain the model.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = holdout_data.copy()

    results["fraud_probability"] = probabilities

    results["predicted_fraud"] = (
        results["fraud_probability"] >= threshold
    ).astype(int)

    # ---------------------------------------------------------
    # Identify prediction errors
    # ---------------------------------------------------------

    false_positives = results[
        (results["isFraud"] == 0)
        & (results["predicted_fraud"] == 1)
    ].copy()

    false_negatives = results[
        (results["isFraud"] == 1)
        & (results["predicted_fraud"] == 0)
    ].copy()

    true_positives = results[
        (results["isFraud"] == 1)
        & (results["predicted_fraud"] == 1)
    ]

    true_negatives = results[
        (results["isFraud"] == 0)
        & (results["predicted_fraud"] == 0)
    ]

    # ---------------------------------------------------------
    # Select useful columns for the reports
    # ---------------------------------------------------------

    preferred_columns = [
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "fraud_probability",
        "predicted_fraud",

        "card1",
        "addr1",
        "P_emaildomain",
        "DeviceInfo",

        "graph_entity_count",
        "graph_known_entity_count",
        "graph_shared_entity_count",
        "graph_unique_entity_types",
        "graph_avg_entity_degree",
        "graph_max_entity_degree",
        "graph_connected_component_size",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in results.columns
    ]

    # ---------------------------------------------------------
    # Save false positives
    # ---------------------------------------------------------

    false_positive_report = false_positives[
        available_columns
    ].sort_values(
        "fraud_probability",
        ascending=False,
    )

    false_positive_path = (
        REPORT_DIR
        / "false_positive_analysis.csv"
    )

    false_positive_report.to_csv(
        false_positive_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save false negatives
    # ---------------------------------------------------------

    false_negative_report = false_negatives[
        available_columns
    ].sort_values(
        "fraud_probability",
        ascending=True,
    )

    false_negative_path = (
        REPORT_DIR
        / "false_negative_analysis.csv"
    )

    false_negative_report.to_csv(
        false_negative_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Error statistics
    # ---------------------------------------------------------

    summary = {
        "threshold": float(threshold),

        "total_holdout_transactions": int(
            len(results)
        ),

        "actual_fraud_transactions": int(
            results["isFraud"].sum()
        ),

        "actual_non_fraud_transactions": int(
            (results["isFraud"] == 0).sum()
        ),

        "false_positives": int(
            len(false_positives)
        ),

        "false_negatives": int(
            len(false_negatives)
        ),

        "true_positives": int(
            len(true_positives)
        ),

        "true_negatives": int(
            len(true_negatives)
        ),
    }

    # ---------------------------------------------------------
    # False-positive probability statistics
    # ---------------------------------------------------------

    if len(false_positives) > 0:

        summary["false_positive_probability"] = {
            "mean": float(
                false_positives[
                    "fraud_probability"
                ].mean()
            ),
            "median": float(
                false_positives[
                    "fraud_probability"
                ].median()
            ),
            "maximum": float(
                false_positives[
                    "fraud_probability"
                ].max()
            ),
        }

    else:

        summary["false_positive_probability"] = {}

    # ---------------------------------------------------------
    # False-negative probability statistics
    # ---------------------------------------------------------

    if len(false_negatives) > 0:

        summary["false_negative_probability"] = {
            "mean": float(
                false_negatives[
                    "fraud_probability"
                ].mean()
            ),
            "median": float(
                false_negatives[
                    "fraud_probability"
                ].median()
            ),
            "minimum": float(
                false_negatives[
                    "fraud_probability"
                ].min()
            ),
        }

    else:

        summary["false_negative_probability"] = {}

    # ---------------------------------------------------------
    # Graph-feature comparison
    # ---------------------------------------------------------

    graph_columns = [
        column
        for column in [
            "graph_entity_count",
            "graph_known_entity_count",
            "graph_shared_entity_count",
            "graph_unique_entity_types",
            "graph_avg_entity_degree",
            "graph_max_entity_degree",
            "graph_connected_component_size",
        ]
        if column in results.columns
    ]

    if graph_columns:

        false_positive_graph_means = (
            false_positives[
                graph_columns
            ].mean()
            .to_dict()
        )

        false_negative_graph_means = (
            false_negatives[
                graph_columns
            ].mean()
            .to_dict()
        )

        true_positive_graph_means = (
            true_positives[
                graph_columns
            ].mean()
            .to_dict()
        )

        true_negative_graph_means = (
            true_negatives[
                graph_columns
            ].mean()
            .to_dict()
        )

        summary["graph_feature_means"] = {
            "false_positives":
                false_positive_graph_means,

            "false_negatives":
                false_negative_graph_means,

            "true_positives":
                true_positive_graph_means,

            "true_negatives":
                true_negative_graph_means,
        }

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    summary_path = (
        REPORT_DIR
        / "error_analysis_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    print(
        f"Threshold:          {threshold:.2f}"
    )

    print(
        f"False positives:    {len(false_positives):,}"
    )

    print(
        f"False negatives:    {len(false_negatives):,}"
    )

    print(
        f"True positives:     {len(true_positives):,}"
    )

    print(
        f"True negatives:     {len(true_negatives):,}"
    )

    print()

    print(
        f"False-positive report: "
        f"{false_positive_path}"
    )

    print(
        f"False-negative report: "
        f"{false_negative_path}"
    )

    print(
        f"Summary report: "
        f"{summary_path}"
    )

    print("=" * 60)

    return summary