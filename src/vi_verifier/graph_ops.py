from __future__ import annotations

import networkx as nx

from .types import Edge, NodeId, canonical_edge


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
    base_edge_set = {canonical_edge(u, v) for u, v in base_graph.edges()}
    for u, v in subgraph_edges:
        edge = canonical_edge(u, v)
        if edge not in base_edge_set:
            raise ValueError(f"subgraph edge {edge} is not in base graph")
        h_graph.add_edge(*edge)
    return h_graph


def build_weighted_transform_for_mst(g_graph: nx.Graph, h_graph: nx.Graph) -> nx.Graph:
    g_prime = nx.Graph()
    g_prime.add_nodes_from(g_graph.nodes())
    h_edges = {canonical_edge(u, v) for u, v in h_graph.edges()}
    for u, v in g_graph.edges():
        edge = canonical_edge(u, v)
        weight = 0 if edge in h_edges else 1
        g_prime.add_edge(*edge, weight=weight)
    return g_prime


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
