from __future__ import annotations

from collections.abc import Iterable, Mapping

from .graph_ops import (
    Graph,
    attach_edge_weights,
    build_canonical_graph,
    build_canonical_subgraph,
    build_weighted_transform_for_mst,
    count_zero_weight_edges,
    degree_map,
    edge_count,
    graph_minus_edges,
    has_edge,
    has_path,
    incident_component_count,
    incident_connectivity,
    is_bipartite,
    is_connected_nonempty,
    mst_of_transformed_graph,
    node_count,
    nodes,
    single_source_dijkstra_path_lengths,
    total_mst_weight,
)
from .paper_compat import PAPER_COMPAT_BY_PREDICATE
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

_PAPER_META = {
    predicate: {
        "paper_section": row.paper_section,
        "paper_rule": row.paper_rule,
        "fidelity": row.status,
    }
    for predicate, row in PAPER_COMPAT_BY_PREDICATE.items()
}


class Verifier:
    def __init__(self) -> None:
        self._graph_cache: dict[
            int,
            tuple[tuple[object, ...], GraphInput, Graph, Graph],
        ] = {}
        self._weighted_graph_cache: dict[int, Graph] = {}
        self._le_expected_cache: dict[tuple[int, NodeId], dict[NodeId, float]] = {}
        self._minus_graph_cache: dict[tuple[int, frozenset[Edge]], Graph] = {}
        self._degree_map_cache: dict[int, dict[NodeId, int]] = {}
        self._incident_connectivity_cache: dict[int, tuple[bool, int]] = {}
        self._incident_vertex_count_cache: dict[int, int] = {}
        self._connected_cache: dict[int, bool] = {}
        self._has_path_cache: dict[tuple[int, NodeId, NodeId], bool] = {}
        self._node_set_cache: dict[int, set[NodeId]] = {}

    def _with_paper_meta(
        self,
        details: Mapping[str, object],
        predicate: str,
    ) -> dict[str, object]:
        merged = dict(details)
        merged.update(_PAPER_META[predicate])
        return merged

    def _graph_input_signature(self, graph_input: GraphInput) -> tuple[object, ...]:
        weights = None
        if graph_input.edge_weights is not None:
            weights = tuple(sorted(graph_input.edge_weights.items(), key=repr))
        ranks = None
        if graph_input.ranks is not None:
            ranks = tuple(sorted(graph_input.ranks.items(), key=repr))
        return (
            frozenset(graph_input.nodes),
            frozenset(graph_input.edges),
            frozenset(graph_input.subgraph_edges),
            weights,
            ranks,
        )

    def _compiled_graph_input(self, graph_input: GraphInput) -> tuple[GraphInput, Graph, Graph]:
        cache_key = id(graph_input)
        signature = self._graph_input_signature(graph_input)
        cached = self._graph_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2], cached[3]
        g_input = graph_input.canonicalized()
        g_graph = build_canonical_graph(g_input.nodes, g_input.edges)
        h_graph = build_canonical_subgraph(g_graph, g_input.subgraph_edges)
        self._graph_cache[cache_key] = (signature, g_input, g_graph, h_graph)
        return g_input, g_graph, h_graph

    def verify(self, graph_input: GraphInput, task: VerificationTask) -> VerificationResult:
        g_input, g_graph, h_graph = self._compiled_graph_input(graph_input)
        verdict, details = self._verify_canonical(g_graph, h_graph, task, g_input)
        return VerificationResult(predicate=task.predicate, verdict=verdict, details=details)

    def _verify_canonical(
        self,
        g_graph: Graph,
        h_graph: Graph,
        task: VerificationTask,
        graph_input: GraphInput,
    ) -> tuple[bool, dict[str, object]]:
        predicate = task.predicate
        if predicate == "spanning_connected_subgraph":
            return self.verify_spanning_connected_subgraph(g_graph, h_graph, task)
        if predicate == "spanning_tree":
            return self.verify_spanning_tree(g_graph, h_graph, task)
        if predicate == "cycle_containment":
            return self.verify_cycle_containment(g_graph, h_graph, task)
        if predicate == "connectivity":
            return self.verify_connectivity(g_graph, h_graph, task)
        if predicate == "cut":
            return self.verify_cut(g_graph, h_graph, task)
        if predicate == "st_connectivity":
            return self.verify_st_connectivity(g_graph, h_graph, task)
        if predicate == "st_cut":
            return self.verify_st_cut(g_graph, h_graph, task)
        if predicate == "edge_on_all_paths":
            return self.verify_edge_on_all_paths(g_graph, h_graph, task)
        if predicate == "e_cycle_containment":
            return self.verify_e_cycle(g_graph, h_graph, task)
        if predicate == "bipartiteness":
            return self.verify_bipartiteness(g_graph, h_graph, task)
        if predicate == "simple_path":
            return self.verify_simple_path(g_graph, h_graph, task)
        if predicate == "hamiltonian_cycle":
            return self.verify_hamiltonian_cycle(g_graph, h_graph, task)
        if predicate == "least_element_list":
            return self.verify_least_element_list(g_graph, h_graph, task, graph_input)
        valid = ", ".join(VALID_PREDICATES)
        raise ValueError(f"unsupported predicate '{predicate}'. valid predicates: {valid}")

    def _degree_map(self, graph: Graph) -> dict[NodeId, int]:
        cache_key = id(graph)
        degrees = self._degree_map_cache.get(cache_key)
        if degrees is None:
            degrees = degree_map(graph)
            self._degree_map_cache[cache_key] = degrees
        return degrees

    def _incident_vertex_count(self, graph: Graph) -> int:
        cache_key = id(graph)
        count = self._incident_vertex_count_cache.get(cache_key)
        if count is None:
            count = sum(1 for degree in self._degree_map(graph).values() if degree > 0)
            self._incident_vertex_count_cache[cache_key] = count
        return count

    def _incident_connectivity(self, graph: Graph) -> tuple[bool, int]:
        cache_key = id(graph)
        result = self._incident_connectivity_cache.get(cache_key)
        if result is None:
            result = incident_connectivity(graph)
            self._incident_connectivity_cache[cache_key] = result
        return result

    def _is_connected_nonempty(self, graph: Graph) -> bool:
        cache_key = id(graph)
        connected = self._connected_cache.get(cache_key)
        if connected is None:
            connected = is_connected_nonempty(graph)
            self._connected_cache[cache_key] = connected
        return connected

    def _has_path(self, graph: Graph, source: NodeId, target: NodeId) -> bool:
        cache_key = (id(graph), source, target) if repr(source) <= repr(target) else (id(graph), target, source)
        path_exists = self._has_path_cache.get(cache_key)
        if path_exists is None:
            path_exists = has_path(graph, source, target)
            self._has_path_cache[cache_key] = path_exists
        return path_exists

    def _node_set(self, graph: Graph) -> set[NodeId]:
        cache_key = id(graph)
        graph_nodes = self._node_set_cache.get(cache_key)
        if graph_nodes is None:
            graph_nodes = set(nodes(graph))
            self._node_set_cache[cache_key] = graph_nodes
        return graph_nodes

    def _mst_summary(self, g_graph: Graph, h_graph: Graph) -> dict[str, int]:
        g_prime = build_weighted_transform_for_mst(g_graph, h_graph)
        mst = mst_of_transformed_graph(g_prime)
        return {
            "mst_weight": total_mst_weight(mst),
            "zero_weight_edges_in_mst": count_zero_weight_edges(mst),
            "h_edges": edge_count(h_graph),
            "h_incident_vertices": self._incident_vertex_count(h_graph),
            "n": node_count(g_graph),
        }

    def _is_spanning_tree_graph(
        self, g_graph: Graph, h_graph: Graph
    ) -> tuple[bool, dict[str, int]]:
        n = node_count(g_graph)
        h_edges = edge_count(h_graph)
        h_connected = self._is_connected_nonempty(h_graph) if n > 0 else False
        is_tree = h_edges == n - 1 and h_connected
        summary = {
            "mst_weight": 0 if h_connected else 1,
            "zero_weight_edges_in_mst": n - 1 if h_connected and n > 0 else 0,
            "h_edges": h_edges,
            "h_incident_vertices": self._incident_vertex_count(h_graph),
            "n": n,
            "h_connected": h_connected,
        }
        return is_tree, summary

    def verify_spanning_tree(
        self, g_graph: Graph, h_graph: Graph, _: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        is_tree, summary = self._is_spanning_tree_graph(g_graph, h_graph)
        return is_tree, self._with_paper_meta(
            summary,
            predicate="spanning_tree",
        )

    def verify_spanning_connected_subgraph(
        self, g_graph: Graph, h_graph: Graph, _: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        n = node_count(g_graph)
        h_incident_vertices = self._incident_vertex_count(h_graph)
        all_nodes_incident = h_incident_vertices == n
        h_connected = self._is_connected_nonempty(h_graph) if n > 0 else False
        verdict = h_connected and all_nodes_incident
        summary = {
            "mst_weight": 0 if h_connected else 1,
            "zero_weight_edges_in_mst": n - 1 if h_connected and n > 0 else 0,
            "h_edges": edge_count(h_graph),
            "h_incident_vertices": h_incident_vertices,
            "n": n,
            "all_nodes_incident": all_nodes_incident,
        }
        return verdict, self._with_paper_meta(
            summary,
            predicate="spanning_connected_subgraph",
        )

    def verify_cycle_containment(
        self, g_graph: Graph, h_graph: Graph, _: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        h_incident_vertices, incident_components = incident_component_count(h_graph)
        h_edges = edge_count(h_graph)
        cycle_free = h_edges == h_incident_vertices - incident_components
        expected = node_count(g_graph) - 1 - h_edges
        summary = {
            "mst_weight": expected if cycle_free else expected + 1,
            "zero_weight_edges_in_mst": h_edges if cycle_free else h_incident_vertices - incident_components,
            "h_edges": h_edges,
            "h_incident_vertices": h_incident_vertices,
            "n": node_count(g_graph),
            "cycle_free_weight_expected": expected,
            "cycle_free": cycle_free,
        }
        return (not cycle_free), self._with_paper_meta(
            summary,
            predicate="cycle_containment",
        )

    def verify_connectivity(
        self, g_graph: Graph, h_graph: Graph, _: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        connected, h_incident_vertices = self._incident_connectivity(h_graph)
        expected_zero_edges = max(h_incident_vertices - 1, 0)
        summary = {
            "mst_weight": 0 if connected else 1,
            "zero_weight_edges_in_mst": expected_zero_edges if connected else 0,
            "h_edges": edge_count(h_graph),
            "h_incident_vertices": h_incident_vertices,
            "n": node_count(g_graph),
            "expected_zero_weight_edges_in_mst": expected_zero_edges,
        }
        return connected, self._with_paper_meta(
            summary,
            predicate="connectivity",
        )

    def _minus_graph(self, base_graph: Graph, removed_edges: frozenset[Edge]) -> Graph:
        cache_key = (id(base_graph), removed_edges)
        minus = self._minus_graph_cache.get(cache_key)
        if minus is None:
            minus = graph_minus_edges(base_graph, set(removed_edges))
            self._minus_graph_cache[cache_key] = minus
        return minus

    def _graph_minus_h(self, g_graph: Graph, h_graph: Graph) -> Graph:
        # Structural reachability graph; callers only inspect connectivity/path existence.
        return self._minus_graph(g_graph, frozenset(h_graph.edge_set))

    def _remove_h_edge(self, h_graph: Graph, edge: Edge) -> Graph:
        if not has_edge(h_graph, edge):
            raise ValueError("e must be an edge in H")
        return self._minus_graph(h_graph, frozenset((edge,)))

    def _validate_node(self, graph: Graph, node: NodeId, label: str) -> None:
        if node not in nodes(graph):
            raise ValueError(f"{label}={node} is not a node in G")

    def _validate_le_list_inputs(
        self,
        graph: Graph,
        task: VerificationTask,
        graph_input: GraphInput,
    ) -> tuple[NodeId, list[tuple[NodeId, float]], dict[NodeId, int], set[NodeId]]:
        if task.target is None:
            raise ValueError("least_element_list requires target")
        if task.le_list is None:
            raise ValueError("least_element_list requires le_list")

        graph_nodes = self._node_set(graph)
        if task.target not in graph_nodes:
            raise ValueError("target must be a node in G")
        if graph_input.ranks is None:
            raise ValueError("least_element_list requires ranks in GraphInput")
        if set(graph_input.ranks.keys()) != graph_nodes:
            raise ValueError("ranks must be provided for all nodes in G")
        if len(set(graph_input.ranks.values())) != len(graph_input.ranks):
            raise ValueError("ranks must be distinct")

        return task.target, task.le_list, graph_input.ranks, graph_nodes

    def _expected_le_list(
        self,
        weighted_graph: Graph,
        target: NodeId,
        ranks: Mapping[NodeId, int],
    ) -> dict[NodeId, float]:
        distances = single_source_dijkstra_path_lengths(weighted_graph, target)
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
        self, g_graph: Graph, h_graph: Graph, _: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        minus = self._graph_minus_h(g_graph, h_graph)
        connected = self._is_connected_nonempty(minus) if node_count(minus) > 0 else True
        return (not connected), self._with_paper_meta(
            {"minus_h_connected": connected},
            predicate="cut",
        )

    def verify_st_connectivity(
        self, g_graph: Graph, h_graph: Graph, task: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        if task.s is None or task.t is None:
            raise ValueError("st_connectivity requires both s and t")
        self._validate_node(g_graph, task.s, "s")
        self._validate_node(g_graph, task.t, "t")
        verdict = self._has_path(h_graph, task.s, task.t)
        return verdict, self._with_paper_meta(
            {"s": task.s, "t": task.t},
            predicate="st_connectivity",
        )

    def verify_st_cut(
        self, g_graph: Graph, h_graph: Graph, task: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        if task.s is None or task.t is None:
            raise ValueError("st_cut requires both s and t")
        self._validate_node(g_graph, task.s, "s")
        self._validate_node(g_graph, task.t, "t")
        minus = self._graph_minus_h(g_graph, h_graph)
        path_exists = self._has_path(minus, task.s, task.t)
        return (not path_exists), self._with_paper_meta(
            {"s": task.s, "t": task.t, "minus_h_has_path": path_exists},
            predicate="st_cut",
        )

    def verify_edge_on_all_paths(
        self, g_graph: Graph, h_graph: Graph, task: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        if task.u is None or task.v is None or task.e is None:
            raise ValueError("edge_on_all_paths requires u, v, and e")
        self._validate_node(g_graph, task.u, "u")
        self._validate_node(g_graph, task.v, "v")
        edge = canonical_edge(*task.e)
        h_minus = self._remove_h_edge(h_graph, edge)
        path_exists = self._has_path(h_minus, task.u, task.v)
        return (not path_exists), self._with_paper_meta(
            {"u": task.u, "v": task.v, "removed_edge": edge},
            predicate="edge_on_all_paths",
        )

    def verify_e_cycle(
        self, g_graph: Graph, h_graph: Graph, task: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        if task.e is None:
            raise ValueError("e_cycle_containment requires e")
        edge = canonical_edge(*task.e)
        h_minus = self._remove_h_edge(h_graph, edge)
        in_cycle = self._has_path(h_minus, edge[0], edge[1])
        return in_cycle, self._with_paper_meta(
            {"removed_edge": edge},
            predicate="e_cycle_containment",
        )

    def verify_bipartiteness(
        self, _: Graph, h_graph: Graph, __: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        bipartite = is_bipartite(h_graph)
        return bipartite, self._with_paper_meta(
            {},
            predicate="bipartiteness",
        )

    def verify_simple_path(
        self, _: Graph, h_graph: Graph, __: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        degrees = self._degree_map(h_graph)
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
                    "edge_count": edge_count(h_graph),
                },
                predicate="simple_path",
            )

        connected_incident, incident_count = self._incident_connectivity(h_graph)
        h_edges = edge_count(h_graph)
        acyclic_incident = h_edges == incident_count - 1
        verdict = connected_incident and acyclic_incident
        return verdict, self._with_paper_meta(
            {
                "deg1": deg1,
                "deg2": deg2,
                "max_degree": max_degree,
                "incident_nodes": incident_count,
                "edge_count": h_edges,
                "connected_incident": connected_incident,
                "acyclic_incident": acyclic_incident,
            },
            predicate="simple_path",
        )

    def verify_hamiltonian_cycle(
        self, g_graph: Graph, h_graph: Graph, task: VerificationTask
    ) -> tuple[bool, dict[str, object]]:
        n = node_count(g_graph)
        degrees = self._degree_map(h_graph)
        all_degree_two = all(degrees.get(node, 0) == 2 for node in nodes(g_graph))
        h_edges = edge_count(h_graph)

        if not all_degree_two or h_edges != n or h_edges == 0:
            return False, self._with_paper_meta(
                {
                    "all_degree_two": all_degree_two,
                    "h_edges": h_edges,
                    "n": n,
                },
                predicate="hamiltonian_cycle",
            )

        try:
            edge = min(h_graph.edge_set, key=lambda edge: (repr(edge[0]), repr(edge[1])))
        except ValueError:
            raise ValueError("hamiltonian_cycle requires at least one edge after preconditions") from None
        # With every node degree 2 and |E(H)| = |V|, H is a disjoint union of cycles.
        # It is Hamiltonian exactly when that 2-regular graph is connected; removing
        # any edge from a connected n-cycle would yield the spanning tree required by
        # the paper-style detail field.
        spanning_tree_verdict = self._is_connected_nonempty(h_graph)
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
        g_graph: Graph,
        _: Graph,
        task: VerificationTask,
        graph_input: GraphInput,
    ) -> tuple[bool, dict[str, object]]:
        target, le_list, ranks, nodes = self._validate_le_list_inputs(g_graph, task, graph_input)
        if graph_input.edge_weights is None:
            weighted_graph = g_graph
        else:
            weighted_cache_key = id(graph_input)
            weighted_graph = self._weighted_graph_cache.get(weighted_cache_key)
            if weighted_graph is None:
                weighted_graph = attach_edge_weights(g_graph, graph_input.edge_weights)
                self._weighted_graph_cache[weighted_cache_key] = weighted_graph
        le_cache_key = (id(graph_input), target)
        expected = self._le_expected_cache.get(le_cache_key)
        if expected is None:
            expected = self._expected_le_list(weighted_graph, target, ranks)
            self._le_expected_cache[le_cache_key] = expected
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
