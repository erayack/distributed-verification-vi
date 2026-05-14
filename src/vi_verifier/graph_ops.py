from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import SupportsFloat, cast

import rustworkx as rx

from .types import Edge, NodeId, canonical_edge


@dataclass(frozen=True)
class Graph:
    rx_graph: rx.PyGraph
    node_to_index: dict[NodeId, int]
    index_to_node: dict[int, NodeId]


def _new_graph(nodes: Iterable[NodeId]) -> Graph:
    rx_graph = rx.PyGraph(multigraph=False)
    node_to_index: dict[NodeId, int] = {}
    index_to_node: dict[int, NodeId] = {}
    for node in sorted(nodes, key=repr):
        index = rx_graph.add_node(node)
        node_to_index[node] = index
        index_to_node[index] = node
    return Graph(rx_graph=rx_graph, node_to_index=node_to_index, index_to_node=index_to_node)


def _add_edge(graph: Graph, edge: Edge, weight: float | None = None) -> None:
    u, v = canonical_edge(*edge)
    graph.rx_graph.add_edge(graph.node_to_index[u], graph.node_to_index[v], weight)


def _edge_weight(weight: object) -> float:
    return 1.0 if weight is None else float(cast(SupportsFloat, weight))


def nodes(graph: Graph) -> list[NodeId]:
    return [graph.index_to_node[index] for index in graph.rx_graph.node_indices()]


def edges(graph: Graph) -> list[Edge]:
    return [
        canonical_edge(graph.index_to_node[u], graph.index_to_node[v])
        for u, v in graph.rx_graph.edge_list()
    ]


def graph_edge_set(graph: Graph) -> set[Edge]:
    return set(edges(graph))


def first_edge_by_repr(edges_iter: Iterable[Edge]) -> Edge | None:
    canonical_edges = (canonical_edge(u, v) for u, v in edges_iter)
    return min(canonical_edges, key=lambda edge: (repr(edge[0]), repr(edge[1])), default=None)


def graph_minus_edges(base_graph: Graph, removed_edges: set[Edge]) -> Graph:
    removed = {canonical_edge(*edge) for edge in removed_edges}
    minus = _new_graph(nodes(base_graph))
    for edge in edges(base_graph):
        if edge not in removed:
            _add_edge(minus, edge)
    return minus


def build_graph(input_nodes: set[NodeId], input_edges: set[Edge]) -> Graph:
    graph = _new_graph(input_nodes)
    for u, v in input_edges:
        if u not in input_nodes or v not in input_nodes:
            raise ValueError(f"edge ({u}, {v}) has endpoint not in node set")
        _add_edge(graph, canonical_edge(u, v))
    return graph


def build_subgraph(base_graph: Graph, subgraph_edges: set[Edge]) -> Graph:
    h_graph = _new_graph(nodes(base_graph))
    base_edge_set = graph_edge_set(base_graph)
    for u, v in subgraph_edges:
        edge = canonical_edge(u, v)
        if edge not in base_edge_set:
            raise ValueError(f"subgraph edge {edge} is not in base graph")
        _add_edge(h_graph, edge)
    return h_graph


def build_weighted_transform_for_mst(g_graph: Graph, h_graph: Graph) -> Graph:
    g_prime = _new_graph(nodes(g_graph))
    h_edges = graph_edge_set(h_graph)
    for edge in edges(g_graph):
        weight = 0.0 if edge in h_edges else 1.0
        _add_edge(g_prime, edge, weight=weight)
    return g_prime


def attach_edge_weights(g_graph: Graph, edge_weights: dict[Edge, float] | None) -> Graph:
    weighted = _new_graph(nodes(g_graph))
    for edge in edges(g_graph):
        weight = 1.0 if edge_weights is None else edge_weights.get(edge, 1.0)
        if weight < 0:
            raise ValueError(f"edge {edge} has negative weight")
        _add_edge(weighted, edge, weight=float(weight))
    return weighted


def mst_of_transformed_graph(g_prime: Graph) -> Graph:
    if node_count(g_prime) == 0:
        return _new_graph([])
    forest = rx.minimum_spanning_tree(g_prime.rx_graph, weight_fn=_edge_weight)
    return Graph(
        rx_graph=forest,
        node_to_index=dict(g_prime.node_to_index),
        index_to_node=dict(g_prime.index_to_node),
    )


def total_mst_weight(tree: Graph) -> int:
    return int(sum(_edge_weight(weight) for _, _, weight in tree.rx_graph.weighted_edge_list()))


def count_zero_weight_edges(tree: Graph) -> int:
    return sum(
        1 for _, _, weight in tree.rx_graph.weighted_edge_list() if _edge_weight(weight) == 0
    )


def edge_count(graph: Graph) -> int:
    return graph.rx_graph.num_edges()


def node_count(graph: Graph) -> int:
    return graph.rx_graph.num_nodes()


def degree_map(graph: Graph) -> dict[NodeId, int]:
    return {node: graph.rx_graph.degree(index) for node, index in graph.node_to_index.items()}


def incident_vertex_count(graph: Graph) -> int:
    return sum(1 for degree in degree_map(graph).values() if degree > 0)


def has_edge(graph: Graph, edge: Edge) -> bool:
    u, v = canonical_edge(*edge)
    if u not in graph.node_to_index or v not in graph.node_to_index:
        return False
    return bool(
        graph.rx_graph.edge_indices_from_endpoints(graph.node_to_index[u], graph.node_to_index[v])
    )


def is_connected_nonempty(graph: Graph) -> bool:
    return rx.is_connected(graph.rx_graph)


def has_path(graph: Graph, source: NodeId, target: NodeId) -> bool:
    if source == target:
        return source in graph.node_to_index
    return rx.has_path(graph.rx_graph, graph.node_to_index[source], graph.node_to_index[target])


def is_bipartite(graph: Graph) -> bool:
    return rx.is_bipartite(graph.rx_graph)


def induced_subgraph(graph: Graph, subgraph_nodes: Iterable[NodeId]) -> Graph:
    selected = set(subgraph_nodes)
    subgraph = _new_graph(selected)
    for edge in edges(graph):
        if edge[0] in selected and edge[1] in selected:
            _add_edge(subgraph, edge)
    return subgraph


def is_tree_nonempty(graph: Graph) -> bool:
    return (
        node_count(graph) > 0
        and is_connected_nonempty(graph)
        and edge_count(graph) == node_count(graph) - 1
    )


def single_source_dijkstra_path_lengths(graph: Graph, target: NodeId) -> dict[NodeId, float]:
    distances = rx.dijkstra_shortest_path_lengths(
        graph.rx_graph,
        graph.node_to_index[target],
        edge_cost_fn=_edge_weight,
    )
    mapped = {graph.index_to_node[index]: float(dist) for index, dist in distances.items()}
    mapped[target] = 0.0
    return mapped
