from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def evaluate_model(
    y_true,
    probabilities,
    threshold: float = 0.5,
):
    """
    Evaluate fraud probabilities at a chosen classification threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
    )

    print("\n" + "=" * 60)
    print("GRAPH-ENHANCED MODEL EVALUATION")
    print("=" * 60)

    print(
        f"Threshold:  {threshold:.2f}"
    )

    print(
        f"Precision:  {precision:.6f}"
    )

    print(
        f"Recall:     {recall:.6f}"
    )

    print(
        f"F1-score:   {f1:.6f}"
    )

    print(
        f"ROC-AUC:    {roc_auc:.6f}"
    )

    print(
        f"PR-AUC:     {pr_auc:.6f}"
    )

    print("\nConfusion matrix:")
    print(matrix)

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    print("=" * 60)

    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": matrix.tolist(),
    }


def threshold_analysis(
    y_true,
    probabilities,
    output_directory: str,
):
    """
    Evaluate model performance across multiple
    classification thresholds.

    This is used for threshold analysis and diagnostics.
    The holdout set should not be used to permanently
    select the deployment threshold.
    """

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds = np.arange(
        0.10,
        0.91,
        0.05,
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    output_csv = (
        output_directory /
        "threshold_analysis.csv"
    )

    with open(
        output_csv,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "threshold,precision,recall,f1\n"
        )

        for row in results:

            file.write(
                f"{row['threshold']:.2f},"
                f"{row['precision']:.6f},"
                f"{row['recall']:.6f},"
                f"{row['f1']:.6f}\n"
            )

    # ---------------------------------------------------------
    # Plot threshold metrics
    # ---------------------------------------------------------

    thresholds_plot = [
        row["threshold"]
        for row in results
    ]

    precision_values = [
        row["precision"]
        for row in results
    ]

    recall_values = [
        row["recall"]
        for row in results
    ]

    f1_values = [
        row["f1"]
        for row in results
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        thresholds_plot,
        precision_values,
        marker="o",
        label="Precision",
    )

    ax.plot(
        thresholds_plot,
        recall_values,
        marker="o",
        label="Recall",
    )

    ax.plot(
        thresholds_plot,
        f1_values,
        marker="o",
        label="F1-score",
    )

    ax.set_xlabel(
        "Classification Threshold"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Fraud Detection Performance vs Classification Threshold"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    output_png = (
        output_directory /
        "threshold_analysis.png"
    )

    fig.savefig(
        output_png,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Print analysis
    # ---------------------------------------------------------

    best_f1 = max(
        results,
        key=lambda row: row["f1"],
    )

    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    print(
        f"Thresholds evaluated: "
        f"{len(results)}"
    )

    print(
        f"Highest observed F1: "
        f"{best_f1['f1']:.6f} "
        f"at threshold "
        f"{best_f1['threshold']:.2f}"
    )

    print()
    print(
        f"Threshold analysis CSV: "
        f"{output_csv}"
    )

    print(
        f"Threshold analysis plot: "
        f"{output_png}"
    )

    print("=" * 60)

    return results


def save_metrics(
    metrics: dict,
    path: str,
):
    """
    Save evaluation metrics as JSON.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        f"Metrics saved to: {path}"
    )


def save_evaluation_plots(
    y_true,
    probabilities,
    output_directory: str,
):
    """
    Save ROC and Precision-Recall curves.
    """

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # ROC curve
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
    )

    ax.set_title(
        "Graph-Enhanced XGBoost - ROC Curve"
    )

    fig.tight_layout()

    fig.savefig(
        output_directory /
        "graph_enhanced_roc_curve.png",
        dpi=150,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Precision-Recall curve
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
    )

    ax.set_title(
        "Graph-Enhanced XGBoost - Precision-Recall Curve"
    )

    fig.tight_layout()

    fig.savefig(
        output_directory /
        "graph_enhanced_precision_recall_curve.png",
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Evaluation plots saved to: "
        f"{output_directory}"
    )