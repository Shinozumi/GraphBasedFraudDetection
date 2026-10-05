from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


TRANSACTION_NODE_PREFIX = "transaction::"
ENTITY_NODE_PREFIX = "entity::"

GRAPH_ENTITY_COLUMNS = [
    "card1",
    "addr1",
    "P_emaildomain",
    "DeviceInfo",
]


def transaction_node_id(transaction_id) -> str:
    """Return the graph node ID for a transaction."""

    return (
        f"{TRANSACTION_NODE_PREFIX}"
        f"{transaction_id}"
    )


def entity_node_id(
    column: str,
    value,
) -> str | None:
    """Return the graph node ID for an entity."""

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
    """Return graph entity columns available in the data."""

    return [
        column
        for column in GRAPH_ENTITY_COLUMNS
        if column in data.columns
    ]


def build_component_size_lookup(
    graph: nx.Graph,
) -> dict[str, int]:
    """
    Calculate connected-component size once for every node.
    """

    lookup = {}

    print(
        "\nCalculating connected-component sizes..."
    )

    for component in nx.connected_components(
        graph
    ):

        size = len(component)

        for node in component:
            lookup[node] = size

    print(
        "Connected-component lookup created."
    )

    return lookup


def calculate_transaction_graph_features(
    graph: nx.Graph,
    transaction_id,
    component_size_lookup: dict[str, int] | None = None,
) -> dict:
    """
    Calculate graph features for an existing transaction node.

    This is primarily useful for training transactions that
    already exist in the training graph.
    """

    transaction_node = (
        transaction_node_id(
            transaction_id
        )
    )

    if transaction_node not in graph:

        return {
            "graph_entity_count": 0,
            "graph_known_entity_count": 0,
            "graph_shared_entity_count": 0,
            "graph_unique_entity_types": 0,
            "graph_avg_entity_degree": 0.0,
            "graph_max_entity_degree": 0,
            "graph_connected_component_size": 0,
        }

    neighbors = list(
        graph.neighbors(
            transaction_node
        )
    )

    entity_degrees = []

    entity_types = set()

    shared_entity_count = 0

    for entity in neighbors:

        attributes = graph.nodes[
            entity
        ]

        entity_type = attributes.get(
            "entity_type"
        )

        if entity_type is not None:
            entity_types.add(
                entity_type
            )

        degree = graph.degree(
            entity
        )

        entity_degrees.append(
            degree
        )

        if degree > 1:
            shared_entity_count += 1

    if entity_degrees:

        average_degree = (
            sum(entity_degrees)
            / len(entity_degrees)
        )

        maximum_degree = max(
            entity_degrees
        )

    else:

        average_degree = 0.0
        maximum_degree = 0

    if component_size_lookup is None:

        component_size = len(
            nx.node_connected_component(
                graph,
                transaction_node,
            )
        )

    else:

        component_size = (
            component_size_lookup.get(
                transaction_node,
                0,
            )
        )

    return {
        "graph_entity_count": len(
            neighbors
        ),

        "graph_known_entity_count": len(
            neighbors
        ),

        "graph_shared_entity_count": (
            shared_entity_count
        ),

        "graph_unique_entity_types": len(
            entity_types
        ),

        "graph_avg_entity_degree": (
            average_degree
        ),

        "graph_max_entity_degree": (
            maximum_degree
        ),

        "graph_connected_component_size": (
            component_size
        ),
    }


def calculate_future_transaction_graph_features(
    graph: nx.Graph,
    transaction_row: pd.Series,
    component_size_lookup: dict[str, int],
) -> dict:
    """
    Calculate graph features for a transaction that is NOT
    present in the training graph.

    This is used for future/holdout transactions.

    The transaction itself is never added to the graph.

    Instead, its entities are looked up against the historical
    training graph.
    """

    entity_columns = (
        get_available_entity_columns(
            pd.DataFrame(
                [transaction_row]
            )
        )
    )

    entity_degrees = []

    entity_types = set()

    known_entity_count = 0

    shared_entity_count = 0

    component_sizes = []

    seen_entities = set()

    for column in entity_columns:

        value = transaction_row.get(
            column
        )

        entity_node = entity_node_id(
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
            continue

        known_entity_count += 1

        attributes = graph.nodes[
            entity_node
        ]

        entity_type = attributes.get(
            "entity_type"
        )

        if entity_type is not None:
            entity_types.add(
                entity_type
            )

        degree = graph.degree(
            entity_node
        )

        entity_degrees.append(
            degree
        )

        if degree > 1:
            shared_entity_count += 1

        component_size = (
            component_size_lookup.get(
                entity_node,
                0,
            )
        )

        if component_size > 0:
            component_sizes.append(
                component_size
            )

    if entity_degrees:

        average_degree = (
            sum(entity_degrees)
            / len(entity_degrees)
        )

        maximum_degree = max(
            entity_degrees
        )

    else:

        average_degree = 0.0
        maximum_degree = 0

    if component_sizes:

        maximum_component_size = max(
            component_sizes
        )

    else:

        maximum_component_size = 0

    return {
        "graph_entity_count": len(
            entity_columns
        ),

        "graph_known_entity_count": (
            known_entity_count
        ),

        "graph_shared_entity_count": (
            shared_entity_count
        ),

        "graph_unique_entity_types": len(
            entity_types
        ),

        "graph_avg_entity_degree": (
            average_degree
        ),

        "graph_max_entity_degree": (
            maximum_degree
        ),

        "graph_connected_component_size": (
            maximum_component_size
        ),
    }


def create_graph_features(
    data: pd.DataFrame,
    graph: nx.Graph,
) -> pd.DataFrame:
    """
    Create graph features for data whose transactions already
    exist in the graph.
    """

    component_size_lookup = (
        build_component_size_lookup(
            graph
        )
    )

    rows = []

    total = len(data)

    for index, transaction_id in enumerate(
        data["TransactionID"],
        start=1,
    ):

        features = (
            calculate_transaction_graph_features(
                graph,
                transaction_id,
                component_size_lookup,
            )
        )

        features[
            "TransactionID"
        ] = transaction_id

        rows.append(
            features
        )

        if index % 10000 == 0:

            print(
                f"Processed "
                f"{index:,} / "
                f"{total:,} transactions"
            )

    return pd.DataFrame(rows)


def create_future_graph_features(
    data: pd.DataFrame,
    graph: nx.Graph,
) -> pd.DataFrame:
    """
    Create graph features for future/holdout transactions
    using ONLY the historical training graph.
    """

    component_size_lookup = (
        build_component_size_lookup(
            graph
        )
    )

    rows = []

    total = len(data)

    print(
        "\nGenerating future transaction graph features..."
    )

    for index, (_, row) in enumerate(
        data.iterrows(),
        start=1,
    ):

        features = (
            calculate_future_transaction_graph_features(
                graph,
                row,
                component_size_lookup,
            )
        )

        features[
            "TransactionID"
        ] = row["TransactionID"]

        rows.append(
            features
        )

        if index % 10000 == 0:

            print(
                f"Processed "
                f"{index:,} / "
                f"{total:,} holdout transactions"
            )

    return pd.DataFrame(rows)


def merge_graph_features(
    data: pd.DataFrame,
    graph_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge graph features into transaction data."""

    result = data.merge(
        graph_features,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    graph_columns = [
        "graph_entity_count",
        "graph_known_entity_count",
        "graph_shared_entity_count",
        "graph_unique_entity_types",
        "graph_avg_entity_degree",
        "graph_max_entity_degree",
        "graph_connected_component_size",
    ]

    for column in graph_columns:

        if column in result.columns:

            result[column] = (
                result[column]
                .fillna(0)
            )

    return result


def save_graph_features(
    graph_features: pd.DataFrame,
    path: str | Path,
) -> None:
    """Save graph features to CSV."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph_features.to_csv(
        path,
        index=False,
    )

    print(
        f"Graph features saved to: {path}"
    )


def load_graph_features(
    path: str | Path,
) -> pd.DataFrame:
    """Load graph features from CSV."""

    return pd.read_csv(
        path
    )