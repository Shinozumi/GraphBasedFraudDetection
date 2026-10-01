from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


TARGET = "isFraud"

TRANSACTION_NODE_PREFIX = "transaction::"
ENTITY_NODE_PREFIX = "entity::"


# We start with a small set of meaningful relational attributes.
#
# These represent entities that can be shared across transactions.
GRAPH_ENTITY_COLUMNS = [
    "card1",
    "addr1",
    "P_emaildomain",
    "DeviceInfo",
]


def make_transaction_node(
    transaction_id,
) -> str:
    """Create a unique graph node ID for a transaction."""

    return (
        f"{TRANSACTION_NODE_PREFIX}"
        f"{transaction_id}"
    )


def make_entity_node(
    column: str,
    value,
) -> str | None:
    """
    Create a unique graph node ID for an entity.

    Prefixing the column prevents collisions between different
    entity types.

    Example:

        card1 = 12345

    becomes:

        entity::card1::12345
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return (
        f"{ENTITY_NODE_PREFIX}"
        f"{column}::"
        f"{value}"
    )


def get_available_entity_columns(
    data: pd.DataFrame,
) -> list[str]:
    """Return graph columns that exist in the dataset."""

    return [
        column
        for column in GRAPH_ENTITY_COLUMNS
        if column in data.columns
    ]


def build_transaction_entity_graph(
    data: pd.DataFrame,
) -> nx.Graph:
    """
    Build a bipartite transaction-entity graph.

    Transaction nodes:
        transaction::2987001

    Entity nodes:
        entity::card1::12345
        entity::addr1::315
        entity::DeviceInfo::Samsung...

    Edges mean:

        transaction <-> entity

    The fraud target is NOT used to construct the graph.
    """

    if "TransactionID" not in data.columns:
        raise ValueError(
            "TransactionID column is required."
        )

    entity_columns = (
        get_available_entity_columns(data)
    )

    if not entity_columns:
        raise ValueError(
            "None of the configured graph entity "
            "columns exist in the dataset."
        )

    graph = nx.Graph()

    print("\n" + "=" * 60)
    print("GRAPH CONSTRUCTION")
    print("=" * 60)

    print("\nEntity columns:")

    for column in entity_columns:
        print(f"  - {column}")

    print(
        f"\nTransactions to process: "
        f"{len(data):,}"
    )

    # ---------------------------------------------------------
    # Add transaction nodes
    # ---------------------------------------------------------

    for transaction_id in data["TransactionID"]:

        node = make_transaction_node(
            transaction_id
        )

        graph.add_node(
            node,
            node_type="transaction",
        )

    # ---------------------------------------------------------
    # Add entity nodes and transaction-entity edges
    # ---------------------------------------------------------

    for row_number, row in enumerate(
        data[
            [
                "TransactionID",
                *entity_columns,
            ]
        ].itertuples(
            index=False,
            name=None,
        ),
        start=1,
    ):

        transaction_id = row[0]

        transaction_node = (
            make_transaction_node(
                transaction_id
            )
        )

        values = row[1:]

        # Prevent duplicate connections if a transaction
        # happens to contain the same value more than once.
        seen_entities = set()

        for column, value in zip(
            entity_columns,
            values,
        ):

            entity_node = make_entity_node(
                column,
                value,
            )

            if entity_node is None:
                continue

            if entity_node in seen_entities:
                continue

            seen_entities.add(
                entity_node
            )

            if entity_node not in graph:

                graph.add_node(
                    entity_node,
                    node_type="entity",
                    entity_type=column,
                )

            graph.add_edge(
                transaction_node,
                entity_node,
            )

        if row_number % 50000 == 0:

            print(
                f"Processed "
                f"{row_number:,} / "
                f"{len(data):,} transactions | "
                f"Nodes: "
                f"{graph.number_of_nodes():,} | "
                f"Edges: "
                f"{graph.number_of_edges():,}"
            )

    print("\nGraph construction completed.")

    print(
        f"Total nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Total edges: "
        f"{graph.number_of_edges():,}"
    )

    return graph


def get_graph_summary(
    graph: nx.Graph,
) -> dict:
    """Return useful graph-level statistics."""

    transaction_nodes = [
        node
        for node, attributes
        in graph.nodes(data=True)
        if attributes.get("node_type")
        == "transaction"
    ]

    entity_nodes = [
        node
        for node, attributes
        in graph.nodes(data=True)
        if attributes.get("node_type")
        == "entity"
    ]

    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "transaction_nodes": len(
            transaction_nodes
        ),
        "entity_nodes": len(
            entity_nodes
        ),
        "connected_components": (
            nx.number_connected_components(
                graph
            )
        ),
    }


def print_graph_summary(
    graph: nx.Graph,
) -> None:
    """Print graph statistics."""

    summary = get_graph_summary(
        graph
    )

    print("\n" + "=" * 60)
    print("GRAPH SUMMARY")
    print("=" * 60)

    print(
        f"Total nodes:          "
        f"{summary['total_nodes']:,}"
    )

    print(
        f"Total edges:          "
        f"{summary['total_edges']:,}"
    )

    print(
        f"Transaction nodes:    "
        f"{summary['transaction_nodes']:,}"
    )

    print(
        f"Entity nodes:         "
        f"{summary['entity_nodes']:,}"
    )

    print(
        f"Connected components: "
        f"{summary['connected_components']:,}"
    )

    print("=" * 60)


def save_graph(
    graph: nx.Graph,
    path: str | Path,
) -> None:
    """
    Save the graph using NetworkX's current gpickle API.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # NetworkX 3.x does not expose nx.write_gpickle.
    # Use Python's pickle module directly.
    import pickle

    with open(
        path,
        "wb",
    ) as file:

        pickle.dump(
            graph,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"\nGraph saved to: {path}"
    )


def load_graph(
    path: str | Path,
) -> nx.Graph:
    """Load a graph previously saved by save_graph()."""

    import pickle

    with open(
        path,
        "rb",
    ) as file:

        graph = pickle.load(
            file
        )

    return graph