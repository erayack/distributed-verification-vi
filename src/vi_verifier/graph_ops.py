from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from .types import Edge, NodeId, canonical_edge


def graph_edge_set(graph: nx.Graph) -> set[Edge]:
    return {canonical_edge(u, v) for u, v in graph.edges()}


def first_edge_by_repr(edges: Iterable[Edge]) -> Edge | None:
    canonical_edges = (canonical_edge(u, v) for u, v in edges)
    return min(canonical_edges, key=lambda edge: (repr(edge[0]), repr(edge[1])), default=None)


def graph_minus_edges(base_graph: nx.Graph, removed_edges: set[Edge]) -> nx.Graph:
    minus = nx.Graph()
    minus.add_nodes_from(base_graph.nodes())
    for u, v in base_graph.edges():
        edge = canonical_edge(u, v)
        if edge not in removed_edges:
            minus.add_edge(*edge)
    return minus


def build_graph(nodes: set[NodeId], edges: set[Edge]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for u, v in edges:
        if u not in nodes or v not in nodes:
            raise ValueError(f"edge ({u}, {v}) has endpoint not in node set")
        graph.add_edge(*canonical_edge(u, v))
    return graph


def build_subgraph(base_graph: nx.Graph, subgraph_edges: set[Edge]) -> nx.Graph:
    h_graph = nx.Graph()
    h_graph.add_nodes_from(base_graph.nodes())
    base_edge_set = graph_edge_set(base_graph)
    for u, v in subgraph_edges:
        edge = canonical_edge(u, v)
        if edge not in base_edge_set:
            raise ValueError(f"subgraph edge {edge} is not in base graph")
        h_graph.add_edge(*edge)
    return h_graph


def build_weighted_transform_for_mst(g_graph: nx.Graph, h_graph: nx.Graph) -> nx.Graph:
    g_prime = nx.Graph()
    g_prime.add_nodes_from(g_graph.nodes())
    h_edges = graph_edge_set(h_graph)
    for u, v in g_graph.edges():
        edge = canonical_edge(u, v)
        weight = 0 if edge in h_edges else 1
        g_prime.add_edge(*edge, weight=weight)
    return g_prime


def attach_edge_weights(g_graph: nx.Graph, edge_weights: dict[Edge, float] | None) -> nx.Graph:
    weighted = nx.Graph()
    weighted.add_nodes_from(g_graph.nodes())
    for u, v in g_graph.edges():
        edge = canonical_edge(u, v)
        weight = 1.0 if edge_weights is None else edge_weights.get(edge, 1.0)
        if weight < 0:
            raise ValueError(f"edge {edge} has negative weight")
        weighted.add_edge(*edge, weight=weight)
    return weighted


def mst_of_transformed_graph(g_prime: nx.Graph) -> nx.Graph:
    if g_prime.number_of_nodes() == 0:
        return nx.Graph()
    forest = nx.minimum_spanning_tree(g_prime, weight="weight")
    return forest


def total_mst_weight(tree: nx.Graph) -> int:
    return sum(data.get("weight", 0) for _, _, data in tree.edges(data=True))


def count_zero_weight_edges(tree: nx.Graph) -> int:
    return sum(1 for _, _, data in tree.edges(data=True) if data.get("weight", 0) == 0)


def edge_count(graph: nx.Graph) -> int:
    return graph.number_of_edges()


def incident_vertex_count(graph: nx.Graph) -> int:
    return graph.number_of_nodes() - nx.number_of_isolates(graph)
