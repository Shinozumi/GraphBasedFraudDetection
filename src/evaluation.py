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
    print("BASELINE MODEL EVALUATION")
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

    # ROC curve

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
    )

    ax.set_title(
        "Baseline XGBoost - ROC Curve"
    )

    fig.tight_layout()

    fig.savefig(
        output_directory / "baseline_roc_curve.png",
        dpi=150,
    )

    plt.close(fig)

    # Precision-Recall curve

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
    )

    ax.set_title(
        "Baseline XGBoost - Precision-Recall Curve"
    )

    fig.tight_layout()

    fig.savefig(
        output_directory
        / "baseline_precision_recall_curve.png",
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Evaluation plots saved to: "
        f"{output_directory}"
    )