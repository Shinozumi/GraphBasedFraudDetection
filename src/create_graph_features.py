from __future__ import annotations

from src.config import MODEL_DIR
from src.data_loader import load_dataset
from src.graph_builder import load_graph
from src.graph_features import (
    create_future_graph_features,
    create_graph_features,
    save_graph_features,
)
from src.preprocess import (
    basic_cleaning,
    chronological_split,
)


GRAPH_PATH = (
    MODEL_DIR
    / "transaction_entity_graph.pkl"
)

TRAIN_GRAPH_FEATURE_PATH = (
    MODEL_DIR
    / "training_graph_features.csv"
)

HOLDOUT_GRAPH_FEATURE_PATH = (
    MODEL_DIR
    / "holdout_graph_features.csv"
)


def main():

    print("=" * 60)
    print("GRAPH-BASED FINANCIAL FRAUD DETECTION")
    print("GRAPH FEATURE PREPARATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    print("\n[1/6] Loading dataset...")

    data = load_dataset()

    # ---------------------------------------------------------
    # 2. Basic cleaning
    # ---------------------------------------------------------

    print("\n[2/6] Cleaning dataset...")

    data = basic_cleaning(data)

    # ---------------------------------------------------------
    # 3. Chronological split
    # ---------------------------------------------------------

    print(
        "\n[3/6] Creating chronological split..."
    )

    train_data, holdout_data = (
        chronological_split(data)
    )

    print(
        f"Training rows: {len(train_data):,}"
    )

    print(
        f"Holdout rows: {len(holdout_data):,}"
    )

    # ---------------------------------------------------------
    # 4. Load TRAINING graph
    # ---------------------------------------------------------

    print(
        "\n[4/6] Loading training graph..."
    )

    graph = load_graph(
        GRAPH_PATH
    )

    print(
        f"Graph nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Graph edges: "
        f"{graph.number_of_edges():,}"
    )

    # ---------------------------------------------------------
    # 5. Training graph features
    # ---------------------------------------------------------

    print(
        "\n[5/6] Creating training graph features..."
    )

    train_graph_features = (
        create_graph_features(
            train_data,
            graph,
        )
    )

    save_graph_features(
        train_graph_features,
        TRAIN_GRAPH_FEATURE_PATH,
    )

    print(
        "\nTraining graph features saved."
    )

    # ---------------------------------------------------------
    # 6. Holdout graph features
    # ---------------------------------------------------------

    print(
        "\n[6/6] Creating holdout graph features..."
    )

    print(
        "IMPORTANT: Holdout transactions are NOT added "
        "to the graph."
    )

    print(
        "Their entities are looked up against the "
        "historical training graph."
    )

    holdout_graph_features = (
        create_future_graph_features(
            holdout_data,
            graph,
        )
    )

    save_graph_features(
        holdout_graph_features,
        HOLDOUT_GRAPH_FEATURE_PATH,
    )

    # ---------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("GRAPH FEATURE PREPARATION COMPLETE")
    print("=" * 60)

    print(
        "\nTraining feature shape:"
    )

    print(
        train_graph_features.shape
    )

    print(
        "\nHoldout feature shape:"
    )

    print(
        holdout_graph_features.shape
    )

    print(
        "\nFiles created:"
    )

    print(
        TRAIN_GRAPH_FEATURE_PATH
    )

    print(
        HOLDOUT_GRAPH_FEATURE_PATH
    )


if __name__ == "__main__":
    main()