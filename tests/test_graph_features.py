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
    print("STEP 7 - GRAPH FEATURE PREPARATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print(
        "\n[1/6] Loading dataset..."
    )

    data = load_dataset()

    # ---------------------------------------------------------
    # 2. Clean
    # ---------------------------------------------------------

    print(
        "\n[2/6] Cleaning dataset..."
    )

    data = basic_cleaning(
        data
    )

    # ---------------------------------------------------------
    # 3. Chronological split
    # ---------------------------------------------------------

    print(
        "\n[3/6] Creating chronological split..."
    )

    train_data, holdout_data = (
        chronological_split(
            data
        )
    )

    print(
        f"Training rows: "
        f"{len(train_data):,}"
    )

    print(
        f"Holdout rows: "
        f"{len(holdout_data):,}"
    )

    # ---------------------------------------------------------
    # 4. Load training graph
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

    # ---------------------------------------------------------
    # 6. Holdout graph features
    # ---------------------------------------------------------

    print(
        "\n[6/6] Creating holdout graph features..."
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

    print("\n" + "=" * 60)
    print("GRAPH FEATURE PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"\nTraining graph features:"
        f"\n{TRAIN_GRAPH_FEATURE_PATH}"
    )

    print(
        f"\nHoldout graph features:"
        f"\n{HOLDOUT_GRAPH_FEATURE_PATH}"
    )

    print("\nTraining feature sample:")
    print(
        train_graph_features.head()
    )

    print("\nHoldout feature sample:")
    print(
        holdout_graph_features.head()
    )


if __name__ == "__main__":
    main()