from __future__ import annotations

from dataclasses import dataclass

from .types import Edge, GraphInput, NodeId, VerificationResult, VerificationTask
from .verifier import Verifier


@dataclass(frozen=True)
class EvalCase:
    name: str
    graph_input: GraphInput
    task: VerificationTask
    expected: bool


def _base_graph() -> GraphInput:
    nodes: set[NodeId] = {1, 2, 3, 4}
    edges: set[Edge] = {(1, 2), (2, 3), (3, 4), (4, 1), (2, 4)}
    return GraphInput(nodes=nodes, edges=edges, subgraph_edges=set())


def _triangle_graph() -> GraphInput:
    nodes: set[NodeId] = {1, 2, 3, 4}
    edges: set[Edge] = {(1, 2), (2, 3), (1, 3), (3, 4)}
    return GraphInput(nodes=nodes, edges=edges, subgraph_edges=set())


def _cycle_graph() -> GraphInput:
    nodes: set[NodeId] = {1, 2, 3, 4}
    edges: set[Edge] = {(1, 2), (2, 3), (3, 4), (4, 1)}
    return GraphInput(nodes=nodes, edges=edges, subgraph_edges=set())


def _two_cycle_components_graph() -> GraphInput:
    nodes: set[NodeId] = {1, 2, 3, 4, 5, 6}
    edges: set[Edge] = {(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)}
    return GraphInput(nodes=nodes, edges=edges, subgraph_edges=set())


def get_eval_cases() -> list[EvalCase]:
    base = _base_graph()
    tri = _triangle_graph()
    cyc = _cycle_graph()
    two_cyc = _two_cycle_components_graph()
    return [
        EvalCase(
            name="spanning_tree_true",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (2, 3), (3, 4)}),
            task=VerificationTask(predicate="spanning_tree"),
            expected=True,
        ),
        EvalCase(
            name="spanning_tree_false_cycle",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (2, 3), (3, 4), (4, 1)}),
            task=VerificationTask(predicate="spanning_tree"),
            expected=False,
        ),
        EvalCase(
            name="spanning_connected_subgraph_true",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (2, 3), (3, 4), (4, 1)}),
            task=VerificationTask(predicate="spanning_connected_subgraph"),
            expected=True,
        ),
        EvalCase(
            name="cycle_containment_true",
            graph_input=GraphInput(tri.nodes, tri.edges, {(1, 2), (2, 3), (3, 1)}),
            task=VerificationTask(predicate="cycle_containment"),
            expected=True,
        ),
        EvalCase(
            name="connectivity_true",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (2, 3), (3, 4)}),
            task=VerificationTask(predicate="connectivity"),
            expected=True,
        ),
        EvalCase(
            name="cut_true",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (1, 4)}),
            task=VerificationTask(predicate="cut"),
            expected=True,
        ),
        EvalCase(
            name="st_connectivity_true",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (2, 3)}),
            task=VerificationTask(predicate="st_connectivity", s=1, t=3),
            expected=True,
        ),
        EvalCase(
            name="st_cut_true",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (1, 4)}),
            task=VerificationTask(predicate="st_cut", s=1, t=3),
            expected=True,
        ),
        EvalCase(
            name="edge_on_all_paths_true",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (2, 3), (3, 4)}),
            task=VerificationTask(predicate="edge_on_all_paths", u=1, v=4, e=(2, 3)),
            expected=True,
        ),
        EvalCase(
            name="e_cycle_containment_true",
            graph_input=GraphInput(tri.nodes, tri.edges, {(1, 2), (2, 3), (3, 1)}),
            task=VerificationTask(predicate="e_cycle_containment", e=(1, 2)),
            expected=True,
        ),
        EvalCase(
            name="bipartiteness_false",
            graph_input=GraphInput(tri.nodes, tri.edges, {(1, 2), (2, 3), (3, 1)}),
            task=VerificationTask(predicate="bipartiteness"),
            expected=False,
        ),
        EvalCase(
            name="simple_path_true",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (2, 3), (3, 4)}),
            task=VerificationTask(predicate="simple_path"),
            expected=True,
        ),
        EvalCase(
            name="simple_path_false_cycle",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (2, 3), (3, 4), (4, 1)}),
            task=VerificationTask(predicate="simple_path"),
            expected=False,
        ),
        EvalCase(
            name="simple_path_false_disconnected",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (3, 4)}),
            task=VerificationTask(predicate="simple_path"),
            expected=False,
        ),
        EvalCase(
            name="hamiltonian_cycle_true",
            graph_input=GraphInput(cyc.nodes, cyc.edges, {(1, 2), (2, 3), (3, 4), (4, 1)}),
            task=VerificationTask(predicate="hamiltonian_cycle"),
            expected=True,
        ),
        EvalCase(
            name="hamiltonian_cycle_false_degree",
            graph_input=GraphInput(base.nodes, base.edges, {(1, 2), (2, 3), (3, 4)}),
            task=VerificationTask(predicate="hamiltonian_cycle"),
            expected=False,
        ),
        EvalCase(
            name="hamiltonian_cycle_false_disconnected",
            graph_input=GraphInput(
                two_cyc.nodes,
                two_cyc.edges,
                {(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)},
            ),
            task=VerificationTask(predicate="hamiltonian_cycle"),
            expected=False,
        ),
    ]


def run_eval_suite() -> list[tuple[EvalCase, VerificationResult]]:
    verifier = Verifier()
    results: list[tuple[EvalCase, VerificationResult]] = []
    for case in get_eval_cases():
        result = verifier.verify(case.graph_input, case.task)
        results.append((case, result))
    return results
