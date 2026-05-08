from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import networkx as nx

from .graph_ops import (
    build_graph,
    build_subgraph,
    build_weighted_transform_for_mst,
    count_zero_weight_edges,
    edge_count,
    incident_vertex_count,
    mst_of_transformed_graph,
    total_mst_weight,
)
from .paper_compat import get_paper_compat_row
from .types import GraphInput, NodeId, VerificationResult, VerificationTask, canonical_edge


class Verifier:
    def _with_paper_meta(
        self,
        details: Mapping[str, object],
        predicate: str,
    ) -> dict[str, object]:
        row = get_paper_compat_row(predicate)
        merged = dict(details)
        merged["paper_section"] = row.paper_section
        merged["paper_rule"] = row.paper_rule
        merged["fidelity"] = row.status
        return merged

    def verify(self, graph_input: GraphInput, task: VerificationTask) -> VerificationResult:
        g_input = graph_input.canonicalized()
        g_graph = build_graph(g_input.nodes, g_input.edges)
        h_graph = build_subgraph(g_graph, g_input.subgraph_edges)

        method_map = {
            "spanning_connected_subgraph": self.verify_spanning_connected_subgraph,
            "spanning_tree": self.verify_spanning_tree,
            "cycle_containment": self.verify_cycle_containment,
            "connectivity": self.verify_connectivity,
            "cut": self.verify_cut,
            "st_connectivity": self.verify_st_connectivity,
            "st_cut": self.verify_st_cut,
            "edge_on_all_paths": self.verify_edge_on_all_paths,
            "e_cycle_containment": self.verify_e_cycle,
            "bipartiteness": self.verify_bipartiteness,
            "simple_path": self.verify_simple_path,
            "hamiltonian_cycle": self.verify_hamiltonian_cycle,
        }
        fn = method_map.get(task.predicate)
        if fn is None:
            valid = ", ".join(sorted(method_map.keys()))
            raise ValueError(f"unsupported predicate '{task.predicate}'. valid predicates: {valid}")
        verdict, details = fn(g_graph, h_graph, task)
        return VerificationResult(predicate=task.predicate, verdict=verdict, details=details)

    def _mst_summary(self, g_graph: nx.Graph, h_graph: nx.Graph) -> dict[str, int]:
        g_prime = build_weighted_transform_for_mst(g_graph, h_graph)
        mst = mst_of_transformed_graph(g_prime)
        return {
            "mst_weight": total_mst_weight(mst),
            "zero_weight_edges_in_mst": count_zero_weight_edges(mst),
            "h_edges": edge_count(h_graph),
            "h_incident_vertices": incident_vertex_count(h_graph),
            "n": g_graph.number_of_nodes(),
        }

    def _is_spanning_tree_graph(self, g_graph: nx.Graph, h_graph: nx.Graph) -> tuple[bool, dict[str, int]]:
        summary = self._mst_summary(g_graph, h_graph)
        h_connected = nx.is_connected(h_graph) if h_graph.number_of_nodes() > 0 else False
        is_tree = summary["mst_weight"] == 0 and summary["h_edges"] == summary["n"] - 1 and h_connected
        summary["h_connected"] = h_connected
        return is_tree, summary

    def verify_spanning_tree(
        self, g_graph: nx.Graph, h_graph: nx.Graph, _: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        is_tree, summary = self._is_spanning_tree_graph(g_graph, h_graph)
        return is_tree, self._with_paper_meta(
            summary,
            predicate="spanning_tree",
        )

    def verify_spanning_connected_subgraph(
        self, g_graph: nx.Graph, h_graph: nx.Graph, _: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        summary = self._mst_summary(g_graph, h_graph)
        all_nodes_incident = summary["h_incident_vertices"] == summary["n"]
        verdict = summary["mst_weight"] == 0 and all_nodes_incident
        summary["all_nodes_incident"] = all_nodes_incident
        return verdict, self._with_paper_meta(
            summary,
            predicate="spanning_connected_subgraph",
        )

    def verify_cycle_containment(
        self, g_graph: nx.Graph, h_graph: nx.Graph, _: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        summary = self._mst_summary(g_graph, h_graph)
        expected = summary["n"] - 1 - summary["h_edges"]
        cycle_free = summary["mst_weight"] == expected
        summary["cycle_free_weight_expected"] = expected
        summary["cycle_free"] = cycle_free
        return (not cycle_free), self._with_paper_meta(
            summary,
            predicate="cycle_containment",
        )

    def verify_connectivity(
        self, g_graph: nx.Graph, h_graph: nx.Graph, _: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        summary = self._mst_summary(g_graph, h_graph)
        expected_zero_edges = max(summary["h_incident_vertices"] - 1, 0)
        connected = summary["zero_weight_edges_in_mst"] == expected_zero_edges
        summary["expected_zero_weight_edges_in_mst"] = expected_zero_edges
        return connected, self._with_paper_meta(
            summary,
            predicate="connectivity",
        )

    def _graph_minus_h(self, g_graph: nx.Graph, h_graph: nx.Graph) -> nx.Graph:
        minus = nx.Graph()
        minus.add_nodes_from(g_graph.nodes())
        h_edges = {canonical_edge(u, v) for u, v in h_graph.edges()}
        for u, v in g_graph.edges():
            edge = canonical_edge(u, v)
            if edge not in h_edges:
                minus.add_edge(*edge)
        return minus

    def _validate_node(self, graph: nx.Graph, node: NodeId, label: str) -> None:
        if node not in graph.nodes():
            raise ValueError(f"{label}={node} is not a node in G")

    def verify_cut(
        self, g_graph: nx.Graph, h_graph: nx.Graph, _: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        minus = self._graph_minus_h(g_graph, h_graph)
        connected = nx.is_connected(minus) if minus.number_of_nodes() > 0 else True
        return (not connected), self._with_paper_meta(
            {"minus_h_connected": connected},
            predicate="cut",
        )

    def verify_st_connectivity(
        self, g_graph: nx.Graph, h_graph: nx.Graph, task: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        if task.s is None or task.t is None:
            raise ValueError("st_connectivity requires both s and t")
        self._validate_node(g_graph, task.s, "s")
        self._validate_node(g_graph, task.t, "t")
        verdict = nx.has_path(h_graph, task.s, task.t)
        return verdict, self._with_paper_meta(
            {"s": task.s, "t": task.t},
            predicate="st_connectivity",
        )

    def verify_st_cut(
        self, g_graph: nx.Graph, h_graph: nx.Graph, task: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        if task.s is None or task.t is None:
            raise ValueError("st_cut requires both s and t")
        self._validate_node(g_graph, task.s, "s")
        self._validate_node(g_graph, task.t, "t")
        minus = self._graph_minus_h(g_graph, h_graph)
        has_path = nx.has_path(minus, task.s, task.t)
        return (not has_path), self._with_paper_meta(
            {"s": task.s, "t": task.t, "minus_h_has_path": has_path},
            predicate="st_cut",
        )

    def verify_edge_on_all_paths(
        self, g_graph: nx.Graph, h_graph: nx.Graph, task: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        if task.u is None or task.v is None or task.e is None:
            raise ValueError("edge_on_all_paths requires u, v, and e")
        self._validate_node(g_graph, task.u, "u")
        self._validate_node(g_graph, task.v, "v")
        edge = canonical_edge(*task.e)
        if edge not in {canonical_edge(a, b) for a, b in h_graph.edges()}:
            raise ValueError("e must be an edge in H")
        h_minus = h_graph.copy()
        h_minus.remove_edge(*edge)
        has_path = nx.has_path(h_minus, task.u, task.v)
        return (not has_path), self._with_paper_meta(
            {"u": task.u, "v": task.v, "removed_edge": edge},
            predicate="edge_on_all_paths",
        )

    def verify_e_cycle(
        self, g_graph: nx.Graph, h_graph: nx.Graph, task: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        if task.e is None:
            raise ValueError("e_cycle_containment requires e")
        edge = canonical_edge(*task.e)
        if edge not in {canonical_edge(a, b) for a, b in h_graph.edges()}:
            raise ValueError("e must be an edge in H")
        h_minus = h_graph.copy()
        h_minus.remove_edge(*edge)
        in_cycle = nx.has_path(h_minus, edge[0], edge[1])
        return in_cycle, self._with_paper_meta(
            {"removed_edge": edge},
            predicate="e_cycle_containment",
        )

    def verify_bipartiteness(
        self, _: nx.Graph, h_graph: nx.Graph, __: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        bipartite = nx.is_bipartite(h_graph)
        return bipartite, self._with_paper_meta(
            {},
            predicate="bipartiteness",
        )

    def verify_simple_path(
        self, _: nx.Graph, h_graph: nx.Graph, __: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        degrees: dict[NodeId, int] = dict(cast("Iterable[tuple[NodeId, int]]", h_graph.degree()))
        incident_nodes = [node for node, degree in degrees.items() if degree > 0]
        deg1 = sum(1 for degree in degrees.values() if degree == 1)
        deg2 = sum(1 for degree in degrees.values() if degree == 2)
        max_degree = max(degrees.values()) if degrees else 0

        if max_degree > 2 or deg1 != 2 or len(incident_nodes) == 0:
            verdict = False
            return verdict, self._with_paper_meta(
                {
                    "deg1": deg1,
                    "deg2": deg2,
                    "max_degree": max_degree,
                    "incident_nodes": len(incident_nodes),
                    "edge_count": h_graph.number_of_edges(),
                },
                predicate="simple_path",
            )

        incident_subgraph = h_graph.subgraph(incident_nodes).copy()
        is_tree = nx.is_tree(incident_subgraph)
        verdict = is_tree
        return verdict, self._with_paper_meta(
            {
                "deg1": deg1,
                "deg2": deg2,
                "max_degree": max_degree,
                "incident_nodes": len(incident_nodes),
                "edge_count": incident_subgraph.number_of_edges(),
                "connected_incident": is_tree,
                "acyclic_incident": is_tree,
            },
            predicate="simple_path",
        )

    def verify_hamiltonian_cycle(
        self, g_graph: nx.Graph, h_graph: nx.Graph, task: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        n = g_graph.number_of_nodes()
        degrees: dict[NodeId, int] = dict(cast("Iterable[tuple[NodeId, int]]", h_graph.degree()))
        all_degree_two = all(degrees.get(node, 0) == 2 for node in g_graph.nodes())
        h_edges = h_graph.number_of_edges()

        if not all_degree_two or h_edges != n or h_edges == 0:
            return False, self._with_paper_meta(
                {
                    "all_degree_two": all_degree_two,
                    "h_edges": h_edges,
                    "n": n,
                },
                predicate="hamiltonian_cycle",
            )

        # Deterministic reduction from the paper: H is Hamiltonian cycle iff
        # every node has degree two and H\{e} is a spanning tree for any edge e.
        # Deterministic edge selection keeps results reproducible across runs.
        edge = sorted((canonical_edge(u, v) for u, v in h_graph.edges()), key=lambda x: (repr(x[0]), repr(x[1])))[0]
        h_minus = h_graph.copy()
        h_minus.remove_edge(*edge)
        spanning_tree_verdict, _ = self._is_spanning_tree_graph(g_graph, h_minus)
        return spanning_tree_verdict, self._with_paper_meta(
            {
                "all_degree_two": all_degree_two,
                "h_edges": h_edges,
                "n": n,
                "removed_edge": edge,
                "h_minus_spanning_tree": spanning_tree_verdict,
            },
            predicate="hamiltonian_cycle",
        )
