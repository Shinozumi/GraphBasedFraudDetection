import pandas as pd

from src.graph_builder import (
    build_transaction_entity_graph,
    make_entity_node,
    make_transaction_node,
)


def make_test_data():

    return pd.DataFrame({
        "TransactionID": [
            1,
            2,
            3,
        ],
        "isFraud": [
            0,
            1,
            0,
        ],
        "TransactionDT": [
            100,
            200,
            300,
        ],
        "card1": [
            "CARD_A",
            "CARD_A",
            "CARD_B",
        ],
        "addr1": [
            "ADDR_A",
            "ADDR_A",
            "ADDR_B",
        ],
        "P_emaildomain": [
            "gmail.com",
            "gmail.com",
            "yahoo.com",
        ],
        "DeviceInfo": [
            "DEVICE_A",
            "DEVICE_A",
            "DEVICE_B",
        ],
    })


def test_node_id_generation():

    transaction_node = (
        make_transaction_node(123)
    )

    entity_node = (
        make_entity_node(
            "card1",
            "ABC",
        )
    )

    assert (
        transaction_node
        == "transaction::123"
    )

    assert (
        entity_node
        == "entity::card1::ABC"
    )


def test_graph_construction():

    data = make_test_data()

    graph = (
        build_transaction_entity_graph(
            data
        )
    )

    transaction_node = (
        "transaction::1"
    )

    card_node = (
        "entity::card1::CARD_A"
    )

    assert (
        transaction_node
        in graph
    )

    assert (
        card_node
        in graph
    )

    assert graph.has_edge(
        transaction_node,
        card_node,
    )


def test_target_not_used_as_graph_node():

    data = make_test_data()

    graph = (
        build_transaction_entity_graph(
            data
        )
    )

    assert "entity::isFraud::1" not in graph
    assert "entity::isFraud::0" not in graph