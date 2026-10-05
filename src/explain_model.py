from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def find_model():
    """
    Find the graph-enhanced XGBoost model.
    """

    model_path = MODELS_DIR / "graph_enhanced_xgboost.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Graph-enhanced model not found: {model_path}"
        )

    return model_path


def get_classifier(model):
    """
    Extract the XGBoost classifier from the trained model.
    """

    # Direct XGBoost model
    if hasattr(model, "feature_importances_"):
        return model

    # Pipeline
    if hasattr(model, "named_steps"):
        for step in reversed(model.named_steps.values()):
            if hasattr(step, "feature_importances_"):
                return step

    # Dictionary containing the model
    if isinstance(model, dict):
        for value in model.values():
            if hasattr(value, "feature_importances_"):
                return value

            if hasattr(value, "named_steps"):
                for step in reversed(value.named_steps.values()):
                    if hasattr(step, "feature_importances_"):
                        return step

    raise ValueError(
        "Could not find feature_importances_ in the trained model."
    )


def get_feature_names(model, classifier):
    """
    Recover feature names from the preprocessing pipeline.
    """

    # Pipeline directly
    if hasattr(model, "named_steps"):

        for step in model.named_steps.values():

            if hasattr(step, "get_feature_names_out"):
                try:
                    return list(step.get_feature_names_out())
                except Exception:
                    pass

    # Dictionary containing a pipeline
    if isinstance(model, dict):

        for value in model.values():

            if hasattr(value, "named_steps"):

                for step in value.named_steps.values():

                    if hasattr(step, "get_feature_names_out"):
                        try:
                            return list(
                                step.get_feature_names_out()
                            )
                        except Exception:
                            pass

    # Fallback
    n_features = len(classifier.feature_importances_)

    return [
        f"feature_{i}"
        for i in range(n_features)
    ]


def create_feature_importance_report():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load graph-enhanced model
    # ---------------------------------------------------------

    model_path = find_model()

    print(f"Loading model: {model_path}")

    model = joblib.load(model_path)

    # ---------------------------------------------------------
    # Extract classifier
    # ---------------------------------------------------------

    classifier = get_classifier(model)

    importances = classifier.feature_importances_

    # ---------------------------------------------------------
    # Feature names
    # ---------------------------------------------------------

    feature_names = get_feature_names(
        model,
        classifier
    )

    if len(feature_names) != len(importances):

        print(
            "Warning: feature-name count does not match "
            "importance count."
        )

        feature_names = [
            f"feature_{i}"
            for i in range(len(importances))
        ]

    # ---------------------------------------------------------
    # Create feature importance dataframe
    # ---------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Save complete feature importance CSV
    # ---------------------------------------------------------

    output_csv = (
        REPORTS_DIR /
        "graph_model_feature_importance.csv"
    )

    importance_df.to_csv(
        output_csv,
        index=False
    )

    # ---------------------------------------------------------
    # Plot top 20 features
    # ---------------------------------------------------------

    top_features = importance_df.head(20)

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1]
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 20 Features - Graph-Enhanced Fraud Model"
    )

    plt.tight_layout()

    output_png = (
        REPORTS_DIR /
        "graph_model_feature_importance.png"
    )

    plt.savefig(
        output_png,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # Identify graph-related features
    # ---------------------------------------------------------

    graph_keywords = [
        "graph_",
        "component",
        "entity_degree",
        "shared_entity",
    ]

    graph_df = importance_df[
        importance_df["feature"].str.lower().apply(
            lambda feature: any(
                keyword in feature
                for keyword in graph_keywords
            )
        )
    ].copy()

    graph_df = graph_df.sort_values(
        "importance",
        ascending=False
    )

    # ---------------------------------------------------------
    # Save graph feature importance
    # ---------------------------------------------------------

    graph_csv = (
        REPORTS_DIR /
        "graph_feature_importance.csv"
    )

    graph_df.to_csv(
        graph_csv,
        index=False
    )

    # ---------------------------------------------------------
    # Create summary
    # ---------------------------------------------------------

    summary = {
        "model": str(model_path),
        "total_features": int(
            len(importance_df)
        ),
        "top_features": (
            importance_df
            .head(10)
            .to_dict(orient="records")
        ),
        "graph_features": (
            graph_df
            .to_dict(orient="records")
        ),
    }

    summary_path = (
        REPORTS_DIR /
        "graph_importance_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print(
        "Explainability report created successfully."
    )

    print()

    print(
        f"Feature importance CSV : {output_csv}"
    )

    print(
        f"Feature importance plot: {output_png}"
    )

    print(
        f"Graph importance CSV   : {graph_csv}"
    )

    print(
        f"Summary JSON           : {summary_path}"
    )

    print()

    print("Top 10 features:")

    print(
        importance_df
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    create_feature_importance_report()