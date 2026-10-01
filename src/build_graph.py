from __future__ import annotations

from src.config import MODEL_DIR
from src.data_loader import load_dataset
from src.graph_builder import (
    build_transaction_entity_graph,
    print_graph_summary,
    save_graph,
)
from src.preprocess import (
    basic_cleaning,
    chronological_split,
)


GRAPH_PATH = MODEL_DIR / "transaction_entity_graph.pkl"


def main():

    print("=" * 60)
    print("GRAPH-BASED FINANCIAL FRAUD DETECTION")
    print("STEP 5 - GRAPH CONSTRUCTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    print("\n[1/5] Loading dataset...")

    data = load_dataset()

    # ---------------------------------------------------------
    # 2. Basic cleaning
    # ---------------------------------------------------------

    print("\n[2/5] Cleaning dataset...")

    data = basic_cleaning(
        data
    )

    # ---------------------------------------------------------
    # 3. Chronological split
    # ---------------------------------------------------------

    print(
        "\n[3/5] Creating chronological split..."
    )

    train_data, test_data = (
        chronological_split(data)
    )

    print(
        f"Training transactions: "
        f"{len(train_data):,}"
    )

    print(
        f"Holdout transactions: "
        f"{len(test_data):,}"
    )

    # ---------------------------------------------------------
    # 4. Build graph ONLY from training data
    # ---------------------------------------------------------

    print(
        "\n[4/5] Building graph from "
        "TRAINING transactions only..."
    )

    graph = build_transaction_entity_graph(
        train_data
    )

    # ---------------------------------------------------------
    # 5. Save graph
    # ---------------------------------------------------------

    print(
        "\n[5/5] Saving graph..."
    )

    print_graph_summary(
        graph
    )

    save_graph(
        graph,
        GRAPH_PATH,
    )

    print("\n" + "=" * 60)
    print("STEP 5 COMPLETE")
    print("=" * 60)

    print(
        f"\nGraph file:"
        f"\n{GRAPH_PATH}"
    )


if __name__ == "__main__":
    main()