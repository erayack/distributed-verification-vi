from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import networkx as nx

from .graph_ops import (
    attach_edge_weights,
    build_graph,
    build_subgraph,
    build_weighted_transform_for_mst,
    count_zero_weight_edges,
    edge_count,
    first_edge_by_repr,
    graph_edge_set,
    graph_minus_edges,
    incident_vertex_count,
    mst_of_transformed_graph,
    total_mst_weight,
)
from .paper_compat import get_paper_compat_row
from .types import Edge, GraphInput, NodeId, VerificationResult, VerificationTask, canonical_edge

VALID_PREDICATES = (
    "bipartiteness",
    "connectivity",
    "cut",
    "cycle_containment",
    "e_cycle_containment",
    "edge_on_all_paths",
    "hamiltonian_cycle",
    "least_element_list",
    "simple_path",
    "spanning_connected_subgraph",
    "spanning_tree",
    "st_connectivity",
    "st_cut",
)


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
        verdict, details = self._verify_canonical(g_graph, h_graph, task, g_input)
        return VerificationResult(predicate=task.predicate, verdict=verdict, details=details)

    def _verify_canonical(
        self,
        g_graph: nx.Graph,
        h_graph: nx.Graph,
        task: VerificationTask,
        graph_input: GraphInput,
    ) -> tuple[bool, dict[str, object]]:
        if task.predicate == "spanning_connected_subgraph":
            return self.verify_spanning_connected_subgraph(g_graph, h_graph, task)
        if task.predicate == "spanning_tree":
            return self.verify_spanning_tree(g_graph, h_graph, task)
        if task.predicate == "cycle_containment":
            return self.verify_cycle_containment(g_graph, h_graph, task)
        if task.predicate == "connectivity":
            return self.verify_connectivity(g_graph, h_graph, task)
        if task.predicate == "cut":
            return self.verify_cut(g_graph, h_graph, task)
        if task.predicate == "st_connectivity":
            return self.verify_st_connectivity(g_graph, h_graph, task)
        if task.predicate == "st_cut":
            return self.verify_st_cut(g_graph, h_graph, task)
        if task.predicate == "edge_on_all_paths":
            return self.verify_edge_on_all_paths(g_graph, h_graph, task)
        if task.predicate == "e_cycle_containment":
            return self.verify_e_cycle(g_graph, h_graph, task)
        if task.predicate == "bipartiteness":
            return self.verify_bipartiteness(g_graph, h_graph, task)
        if task.predicate == "simple_path":
            return self.verify_simple_path(g_graph, h_graph, task)
        if task.predicate == "hamiltonian_cycle":
            return self.verify_hamiltonian_cycle(g_graph, h_graph, task)
        if task.predicate == "least_element_list":
            return self.verify_least_element_list(g_graph, h_graph, task, graph_input)
        valid = ", ".join(VALID_PREDICATES)
        raise ValueError(f"unsupported predicate '{task.predicate}'. valid predicates: {valid}")

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
        return graph_minus_edges(g_graph, graph_edge_set(h_graph))

    def _remove_h_edge(self, h_graph: nx.Graph, edge: Edge) -> nx.Graph:
        if edge not in graph_edge_set(h_graph):
            raise ValueError("e must be an edge in H")
        return graph_minus_edges(h_graph, {edge})

    def _validate_node(self, graph: nx.Graph, node: NodeId, label: str) -> None:
        if node not in graph.nodes():
            raise ValueError(f"{label}={node} is not a node in G")

    def _validate_le_list_inputs(
        self,
        graph: nx.Graph,
        task: VerificationTask,
        graph_input: GraphInput,
    ) -> tuple[NodeId, list[tuple[NodeId, float]], dict[NodeId, int], set[NodeId]]:
        if task.target is None:
            raise ValueError("least_element_list requires target")
        if task.le_list is None:
            raise ValueError("least_element_list requires le_list")

        nodes = set(cast("Iterable[NodeId]", graph.nodes()))
        if task.target not in nodes:
            raise ValueError("target must be a node in G")
        if graph_input.ranks is None:
            raise ValueError("least_element_list requires ranks in GraphInput")
        if set(graph_input.ranks.keys()) != nodes:
            raise ValueError("ranks must be provided for all nodes in G")
        if len(set(graph_input.ranks.values())) != len(graph_input.ranks):
            raise ValueError("ranks must be distinct")

        return task.target, task.le_list, graph_input.ranks, nodes

    def _expected_le_list(
        self,
        weighted_graph: nx.Graph,
        target: NodeId,
        ranks: Mapping[NodeId, int],
    ) -> dict[NodeId, float]:
        distances = nx.single_source_dijkstra_path_length(weighted_graph, target, weight="weight")
        expected: dict[NodeId, float] = {}
        best_rank_so_far: int | None = None
        for node, dist in sorted(distances.items(), key=lambda item: (item[1], repr(item[0]))):
            node_rank = ranks[node]
            if best_rank_so_far is None or node_rank < best_rank_so_far:
                expected[node] = float(dist)
                best_rank_so_far = node_rank
        return expected

    def _provided_le_list(
        self,
        le_list: Iterable[tuple[NodeId, float]],
        nodes: set[NodeId],
    ) -> dict[NodeId, float]:
        provided: dict[NodeId, float] = {}
        for node, dist in le_list:
            if node not in nodes:
                raise ValueError(f"LE-list node {node} is not in G")
            dist_f = float(dist)
            if node in provided and abs(provided[node] - dist_f) > 1e-9:
                raise ValueError(f"conflicting duplicate LE-list distance for node {node}")
            provided[node] = dist_f
        return provided

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
        h_minus = self._remove_h_edge(h_graph, edge)
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
        h_minus = self._remove_h_edge(h_graph, edge)
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

        edge = first_edge_by_repr(h_graph.edges())
        if edge is None:
            raise ValueError("hamiltonian_cycle requires at least one edge after preconditions")
        h_minus = graph_minus_edges(h_graph, {edge})
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

    def verify_least_element_list(
        self,
        g_graph: nx.Graph,
        _: nx.Graph,
        task: VerificationTask,
        graph_input: GraphInput,
    ) -> tuple[bool, dict[str, object]]:
        target, le_list, ranks, nodes = self._validate_le_list_inputs(g_graph, task, graph_input)
        weighted_graph = attach_edge_weights(g_graph, graph_input.edge_weights)
        expected = self._expected_le_list(weighted_graph, target, ranks)
        provided = self._provided_le_list(le_list, nodes)
        missing_nodes = [node for node in expected if node not in provided]
        extra_nodes = [node for node in provided if node not in expected]
        distance_mismatches = [
            node
            for node in expected
            if node in provided and abs(expected[node] - provided[node]) > 1e-9
        ]
        verdict = not missing_nodes and not extra_nodes and not distance_mismatches
        return verdict, self._with_paper_meta(
            {
                "target": target,
                "provided_count": len(provided),
                "expected_count": len(expected),
                "missing_nodes": sorted(missing_nodes, key=repr),
                "extra_nodes": sorted(extra_nodes, key=repr),
                "distance_mismatches": sorted(distance_mismatches, key=repr),
            },
            predicate="least_element_list",
        )
