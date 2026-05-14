from __future__ import annotations

import json

import pytest

from vi_verifier.cli import main as cli_main
from vi_verifier.eval_cases import get_eval_cases
from vi_verifier.paper_compat import (
    get_implemented_rows,
    get_paper_compat_row,
    get_paper_compat_rows,
)
from vi_verifier.types import Edge, GraphInput, NodeId, VerificationTask
from vi_verifier.verifier import Verifier


def _graph_input(subgraph_edges: set[Edge]) -> GraphInput:
    nodes: set[NodeId] = {1, 2, 3, 4}
    edges: set[Edge] = {(1, 2), (2, 3), (3, 4), (4, 1), (2, 4)}
    return GraphInput(nodes=nodes, edges=edges, subgraph_edges=subgraph_edges)


def _triangle_graph_input(subgraph_edges: set[Edge]) -> GraphInput:
    nodes: set[NodeId] = {1, 2, 3, 4}
    edges: set[Edge] = {(1, 2), (2, 3), (1, 3), (3, 4)}
    return GraphInput(nodes=nodes, edges=edges, subgraph_edges=subgraph_edges)


def _cycle_graph_input(subgraph_edges: set[Edge]) -> GraphInput:
    nodes: set[NodeId] = {1, 2, 3, 4}
    edges: set[Edge] = {(1, 2), (2, 3), (3, 4), (4, 1)}
    return GraphInput(nodes=nodes, edges=edges, subgraph_edges=subgraph_edges)


def _two_cycle_components_graph_input(subgraph_edges: set[Edge]) -> GraphInput:
    nodes: set[NodeId] = {1, 2, 3, 4, 5, 6}
    edges: set[Edge] = {(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)}
    return GraphInput(nodes=nodes, edges=edges, subgraph_edges=subgraph_edges)


def _le_list_graph_input() -> GraphInput:
    return GraphInput(
        nodes={1, 2, 3},
        edges={(1, 2), (2, 3), (1, 3)},
        subgraph_edges=set(),
        edge_weights={(1, 2): 1.0, (2, 3): 1.0, (1, 3): 3.0},
        ranks={1: 5, 2: 1, 3: 3},
    )


def test_spanning_tree_true_and_false() -> None:
    verifier = Verifier()
    true_case = _graph_input({(1, 2), (2, 3), (3, 4)})
    false_case = _graph_input({(1, 2), (2, 3), (3, 4), (4, 1)})
    assert verifier.verify(true_case, VerificationTask("spanning_tree")).verdict is True
    assert verifier.verify(false_case, VerificationTask("spanning_tree")).verdict is False


def test_spanning_tree_rejects_disconnected_n_minus_one_edges() -> None:
    verifier = Verifier()
    disconnected_case = _two_cycle_components_graph_input({(1, 2), (2, 3), (4, 5), (5, 6), (6, 4)})
    assert verifier.verify(disconnected_case, VerificationTask("spanning_tree")).verdict is False


def test_spanning_connected_subgraph_true_and_false() -> None:
    verifier = Verifier()
    true_case = _graph_input({(1, 2), (2, 3), (3, 4), (4, 1)})
    false_case = GraphInput(
        nodes={1, 2, 3, 4}, edges={(1, 2), (2, 3), (3, 4)}, subgraph_edges={(1, 2)}
    )
    assert (
        verifier.verify(true_case, VerificationTask("spanning_connected_subgraph")).verdict is True
    )
    assert (
        verifier.verify(false_case, VerificationTask("spanning_connected_subgraph")).verdict
        is False
    )


def test_spanning_connected_subgraph_rejects_empty_h_with_isolated_nodes() -> None:
    graph_input = GraphInput(nodes={1, 2, 3}, edges={(1, 2), (2, 3)}, subgraph_edges=set())
    result = Verifier().verify(graph_input, VerificationTask("spanning_connected_subgraph"))
    assert result.verdict is False
    assert result.details["all_nodes_incident"] is False


def test_cycle_containment_matches_direct_cycle_check() -> None:
    verifier = Verifier()
    cycle_case = _triangle_graph_input({(1, 2), (2, 3), (3, 1)})
    acyclic_case = _triangle_graph_input({(1, 2), (2, 3)})

    assert verifier.verify(cycle_case, VerificationTask("cycle_containment")).verdict is True
    assert verifier.verify(acyclic_case, VerificationTask("cycle_containment")).verdict is False


def test_connectivity_matches_direct_check() -> None:
    verifier = Verifier()
    connected = _graph_input({(1, 2), (2, 3), (3, 4)})
    disconnected = _graph_input({(1, 2), (3, 4)})
    assert verifier.verify(connected, VerificationTask("connectivity")).verdict is True
    assert verifier.verify(disconnected, VerificationTask("connectivity")).verdict is False


def test_connectivity_empty_h_matches_incident_node_definition() -> None:
    graph_input = GraphInput(nodes={1, 2, 3}, edges={(1, 2), (2, 3)}, subgraph_edges=set())
    result = Verifier().verify(graph_input, VerificationTask("connectivity"))
    assert result.verdict is True
    assert result.details["h_incident_vertices"] == 0
    assert result.details["expected_zero_weight_edges_in_mst"] == 0


def test_cut_and_st_cut() -> None:
    verifier = Verifier()
    case = _graph_input({(1, 2), (1, 4)})
    assert verifier.verify(case, VerificationTask("cut")).verdict is True
    assert verifier.verify(case, VerificationTask("st_cut", s=1, t=3)).verdict is True


def test_st_connectivity() -> None:
    verifier = Verifier()
    case = _graph_input({(1, 2), (2, 3)})
    assert verifier.verify(case, VerificationTask("st_connectivity", s=1, t=3)).verdict is True
    assert verifier.verify(case, VerificationTask("st_connectivity", s=1, t=4)).verdict is False


def test_st_connectivity_same_node_has_trivial_path() -> None:
    case = GraphInput(nodes={1, 2}, edges={(1, 2)}, subgraph_edges=set())
    assert Verifier().verify(case, VerificationTask("st_connectivity", s=1, t=1)).verdict is True


def test_st_cut_same_node_is_not_cut() -> None:
    case = GraphInput(nodes={1, 2}, edges={(1, 2)}, subgraph_edges={(1, 2)})
    result = Verifier().verify(case, VerificationTask("st_cut", s=1, t=1))
    assert result.verdict is False
    assert result.details["minus_h_has_path"] is True


def test_edge_on_all_paths() -> None:
    verifier = Verifier()
    case = _graph_input({(1, 2), (2, 3), (3, 4)})
    assert (
        verifier.verify(
            case,
            VerificationTask("edge_on_all_paths", u=1, v=4, e=(2, 3)),
        ).verdict
        is True
    )
    assert (
        verifier.verify(
            case,
            VerificationTask("edge_on_all_paths", u=1, v=3, e=(3, 4)),
        ).verdict
        is False
    )


def test_e_cycle_containment() -> None:
    verifier = Verifier()
    cycle_case = _triangle_graph_input({(1, 2), (2, 3), (3, 1)})
    no_cycle_case = _triangle_graph_input({(1, 2), (2, 3)})
    assert (
        verifier.verify(cycle_case, VerificationTask("e_cycle_containment", e=(1, 2))).verdict
        is True
    )
    assert (
        verifier.verify(cycle_case, VerificationTask("e_cycle_containment", e=(2, 1))).verdict
        is True
    )
    assert (
        verifier.verify(no_cycle_case, VerificationTask("e_cycle_containment", e=(1, 2))).verdict
        is False
    )


def test_bipartiteness() -> None:
    verifier = Verifier()
    bip_case = _graph_input({(1, 2), (2, 3), (3, 4)})
    non_bip_case = _triangle_graph_input({(1, 2), (2, 3), (3, 1)})
    assert verifier.verify(bip_case, VerificationTask("bipartiteness")).verdict is True
    assert verifier.verify(non_bip_case, VerificationTask("bipartiteness")).verdict is False


def test_simple_path_true_and_false() -> None:
    verifier = Verifier()
    true_case = _graph_input({(1, 2), (2, 3), (3, 4)})
    false_cycle_case = _graph_input({(1, 2), (2, 3), (3, 4), (4, 1)})
    false_disconnected_case = _graph_input({(1, 2), (3, 4)})

    assert verifier.verify(true_case, VerificationTask("simple_path")).verdict is True
    assert verifier.verify(false_cycle_case, VerificationTask("simple_path")).verdict is False
    assert (
        verifier.verify(false_disconnected_case, VerificationTask("simple_path")).verdict is False
    )


def test_hamiltonian_cycle_true_and_false() -> None:
    verifier = Verifier()
    true_case = _cycle_graph_input({(1, 2), (2, 3), (3, 4), (4, 1)})
    false_degree_case = _graph_input({(1, 2), (2, 3), (3, 4)})
    false_disconnected_case = _two_cycle_components_graph_input(
        {(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)}
    )
    assert verifier.verify(true_case, VerificationTask("hamiltonian_cycle")).verdict is True
    assert (
        verifier.verify(false_degree_case, VerificationTask("hamiltonian_cycle")).verdict is False
    )
    assert (
        verifier.verify(false_disconnected_case, VerificationTask("hamiltonian_cycle")).verdict
        is False
    )


def test_hamiltonian_cycle_has_paper_metadata() -> None:
    verifier = Verifier()
    case = _cycle_graph_input({(1, 2), (2, 3), (3, 4), (4, 1)})
    details = verifier.verify(case, VerificationTask("hamiltonian_cycle")).details
    row = get_paper_compat_row("hamiltonian_cycle")
    assert details["paper_section"] == row.paper_section
    assert details["paper_rule"] == row.paper_rule
    assert details["fidelity"] == row.status


def test_least_element_list_true_and_false() -> None:
    verifier = Verifier()
    graph_input = _le_list_graph_input()
    true_task = VerificationTask("least_element_list", target=1, le_list=[(1, 0.0), (2, 1.0)])
    false_task = VerificationTask(
        "least_element_list", target=1, le_list=[(1, 0.0), (2, 1.0), (3, 2.0)]
    )
    assert verifier.verify(graph_input, true_task).verdict is True
    assert verifier.verify(graph_input, false_task).verdict is False


def test_least_element_list_accepts_duplicate_same_distance_entry() -> None:
    graph_input = _le_list_graph_input()
    task = VerificationTask("least_element_list", target=1, le_list=[(1, 0.0), (1, 0.0), (2, 1.0)])
    assert Verifier().verify(graph_input, task).verdict is True


def test_least_element_list_validation_errors() -> None:
    verifier = Verifier()
    graph_input = _le_list_graph_input()
    with pytest.raises(ValueError):
        verifier.verify(graph_input, VerificationTask("least_element_list", target=1))
    bad_ranks = GraphInput(
        nodes={1, 2},
        edges={(1, 2)},
        subgraph_edges=set(),
        ranks={1: 1, 2: 1},
    )
    with pytest.raises(ValueError):
        verifier.verify(
            bad_ranks, VerificationTask("least_element_list", target=1, le_list=[(1, 0.0)])
        )


@pytest.mark.parametrize(
    ("graph_input", "task", "message"),
    [
        (
            _le_list_graph_input(),
            VerificationTask("least_element_list", le_list=[(1, 0.0)]),
            "least_element_list requires target",
        ),
        (
            _le_list_graph_input(),
            VerificationTask("least_element_list", target=1),
            "least_element_list requires le_list",
        ),
        (
            _le_list_graph_input(),
            VerificationTask("least_element_list", target=99, le_list=[]),
            "target must be a node in G",
        ),
        (
            GraphInput(nodes={1, 2}, edges={(1, 2)}, subgraph_edges=set()),
            VerificationTask("least_element_list", target=1, le_list=[]),
            "least_element_list requires ranks in GraphInput",
        ),
        (
            GraphInput(nodes={1, 2}, edges={(1, 2)}, subgraph_edges=set(), ranks={1: 1}),
            VerificationTask("least_element_list", target=1, le_list=[]),
            "ranks must be provided for all nodes in G",
        ),
        (
            GraphInput(nodes={1, 2}, edges={(1, 2)}, subgraph_edges=set(), ranks={1: 1, 2: 1}),
            VerificationTask("least_element_list", target=1, le_list=[]),
            "ranks must be distinct",
        ),
        (
            _le_list_graph_input(),
            VerificationTask("least_element_list", target=1, le_list=[(99, 0.0)]),
            "LE-list node 99 is not in G",
        ),
        (
            _le_list_graph_input(),
            VerificationTask("least_element_list", target=1, le_list=[(1, 0.0), (1, 1.0)]),
            "conflicting duplicate LE-list distance for node 1",
        ),
    ],
)
def test_least_element_list_validation_error_messages(
    graph_input: GraphInput,
    task: VerificationTask,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        Verifier().verify(graph_input, task)


def test_least_element_list_details_key_parity() -> None:
    details = (
        Verifier()
        .verify(
            _le_list_graph_input(),
            VerificationTask("least_element_list", target=1, le_list=[(1, 0.0), (2, 1.0)]),
        )
        .details
    )
    assert set(details) == {
        "target",
        "provided_count",
        "expected_count",
        "missing_nodes",
        "extra_nodes",
        "distance_mismatches",
        "paper_section",
        "paper_rule",
        "fidelity",
    }


def test_all_predicates_include_paper_metadata() -> None:
    verifier = Verifier()
    checks = [
        (_graph_input({(1, 2), (2, 3), (3, 4)}), VerificationTask("spanning_tree")),
        (
            _graph_input({(1, 2), (2, 3), (3, 4), (4, 1)}),
            VerificationTask("spanning_connected_subgraph"),
        ),
        (_triangle_graph_input({(1, 2), (2, 3), (3, 1)}), VerificationTask("cycle_containment")),
        (_graph_input({(1, 2), (2, 3), (3, 4)}), VerificationTask("connectivity")),
        (_graph_input({(1, 2), (1, 4)}), VerificationTask("cut")),
        (_graph_input({(1, 2), (2, 3)}), VerificationTask("st_connectivity", s=1, t=3)),
        (_graph_input({(1, 2), (1, 4)}), VerificationTask("st_cut", s=1, t=3)),
        (
            _graph_input({(1, 2), (2, 3), (3, 4)}),
            VerificationTask("edge_on_all_paths", u=1, v=4, e=(2, 3)),
        ),
        (
            _triangle_graph_input({(1, 2), (2, 3), (3, 1)}),
            VerificationTask("e_cycle_containment", e=(1, 2)),
        ),
        (_graph_input({(1, 2), (2, 3), (3, 4)}), VerificationTask("bipartiteness")),
        (_graph_input({(1, 2), (2, 3), (3, 4)}), VerificationTask("simple_path")),
        (
            _cycle_graph_input({(1, 2), (2, 3), (3, 4), (4, 1)}),
            VerificationTask("hamiltonian_cycle"),
        ),
        (
            _le_list_graph_input(),
            VerificationTask("least_element_list", target=1, le_list=[(1, 0.0), (2, 1.0)]),
        ),
    ]
    for graph_input, task in checks:
        details = verifier.verify(graph_input, task).details
        assert "paper_section" in details
        assert "paper_rule" in details
        assert details.get("fidelity") in {"implemented", "approximated"}


def test_predicate_metadata_matches_paper_registry() -> None:
    verifier = Verifier()
    checks = [
        (_graph_input({(1, 2), (2, 3), (3, 4)}), VerificationTask("spanning_tree")),
        (
            _graph_input({(1, 2), (2, 3), (3, 4), (4, 1)}),
            VerificationTask("spanning_connected_subgraph"),
        ),
        (_triangle_graph_input({(1, 2), (2, 3), (3, 1)}), VerificationTask("cycle_containment")),
        (_graph_input({(1, 2), (2, 3), (3, 4)}), VerificationTask("connectivity")),
        (_graph_input({(1, 2), (1, 4)}), VerificationTask("cut")),
        (_graph_input({(1, 2), (2, 3)}), VerificationTask("st_connectivity", s=1, t=3)),
        (_graph_input({(1, 2), (1, 4)}), VerificationTask("st_cut", s=1, t=3)),
        (
            _graph_input({(1, 2), (2, 3), (3, 4)}),
            VerificationTask("edge_on_all_paths", u=1, v=4, e=(2, 3)),
        ),
        (
            _triangle_graph_input({(1, 2), (2, 3), (3, 1)}),
            VerificationTask("e_cycle_containment", e=(1, 2)),
        ),
        (_graph_input({(1, 2), (2, 3), (3, 4)}), VerificationTask("bipartiteness")),
        (_graph_input({(1, 2), (2, 3), (3, 4)}), VerificationTask("simple_path")),
        (
            _cycle_graph_input({(1, 2), (2, 3), (3, 4), (4, 1)}),
            VerificationTask("hamiltonian_cycle"),
        ),
        (
            _le_list_graph_input(),
            VerificationTask("least_element_list", target=1, le_list=[(1, 0.0), (2, 1.0)]),
        ),
    ]
    for graph_input, task in checks:
        details = verifier.verify(graph_input, task).details
        row = get_paper_compat_row(task.predicate)
        assert details["paper_section"] == row.paper_section
        assert details["paper_rule"] == row.paper_rule
        assert details["fidelity"] == row.status


def test_paper_matrix_contains_expected_rows() -> None:
    rows = get_paper_compat_rows()
    names = {row.predicate for row in rows}
    assert "simple_path" in names
    assert "spanning_tree" in names
    assert "hamiltonian_cycle" in names
    assert "least_element_list" in names
    assert any(row.status == "approximated" for row in rows)
    ham = get_paper_compat_row("hamiltonian_cycle")
    assert ham.status == "implemented"
    le_list = get_paper_compat_row("least_element_list")
    assert le_list.status == "implemented"


def test_input_validation_errors() -> None:
    verifier = Verifier()
    bad_subgraph = GraphInput(nodes={1, 2}, edges={(1, 2)}, subgraph_edges={(1, 3)})
    with pytest.raises(ValueError):
        verifier.verify(bad_subgraph, VerificationTask("connectivity"))

    good = GraphInput(nodes={1, 2, 3}, edges={(1, 2), (2, 3)}, subgraph_edges={(1, 2)})
    with pytest.raises(ValueError):
        verifier.verify(good, VerificationTask("st_connectivity", s=1))
    with pytest.raises(ValueError):
        verifier.verify(good, VerificationTask("e_cycle_containment", e=(1, 3)))
    bad_weights = GraphInput(
        nodes={1, 2}, edges={(1, 2)}, subgraph_edges=set(), edge_weights={(1, 3): 5.0}
    )
    with pytest.raises(ValueError):
        verifier.verify(bad_weights, VerificationTask("connectivity"))
    with pytest.raises(ValueError):
        verifier.verify(good, VerificationTask(predicate="not_supported"))  # type: ignore[arg-type]


def test_cli_unknown_case_exit_code() -> None:
    assert cli_main(["run-case", "--case", "does_not_exist"]) == 2


def test_cli_paper_matrix_exit_code() -> None:
    assert cli_main(["paper-matrix"]) == 0


def test_cli_reproduce_lower_bounds_exit_code() -> None:
    assert cli_main(["reproduce-lower-bounds"]) == 0


def test_cli_paper_matrix_output_shape(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["paper-matrix"]) == 0
    out = capsys.readouterr().out
    assert "predicate | paper_section | status | implementation_hook | notes" in out
    assert "spanning_tree | paper/tightness.tex | implemented" in out
    assert "simple_path | paper/prelim.tex and paper/deterministic_lb.tex | implemented" in out


def test_cli_paper_matrix_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["paper-matrix", "--format", "json"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list)
    predicates = {row["predicate"] for row in payload}
    assert "hamiltonian_cycle" in predicates


def test_non_deferred_rows_are_dispatchable_and_covered() -> None:
    verifier = Verifier()
    rows = get_implemented_rows(include_approximated=True)
    cases = {
        "spanning_connected_subgraph": _graph_input({(1, 2), (2, 3), (3, 4), (4, 1)}),
        "spanning_tree": _graph_input({(1, 2), (2, 3), (3, 4)}),
        "cycle_containment": _triangle_graph_input({(1, 2), (2, 3), (3, 1)}),
        "connectivity": _graph_input({(1, 2), (2, 3), (3, 4)}),
        "cut": _graph_input({(1, 2), (1, 4)}),
        "st_connectivity": _graph_input({(1, 2), (2, 3)}),
        "st_cut": _graph_input({(1, 2), (1, 4)}),
        "edge_on_all_paths": _graph_input({(1, 2), (2, 3), (3, 4)}),
        "e_cycle_containment": _triangle_graph_input({(1, 2), (2, 3), (3, 1)}),
        "bipartiteness": _graph_input({(1, 2), (2, 3), (3, 4)}),
        "simple_path": _graph_input({(1, 2), (2, 3), (3, 4)}),
        "hamiltonian_cycle": _cycle_graph_input({(1, 2), (2, 3), (3, 4), (4, 1)}),
        "least_element_list": _le_list_graph_input(),
    }
    tasks = {
        "st_connectivity": VerificationTask("st_connectivity", s=1, t=3),
        "st_cut": VerificationTask("st_cut", s=1, t=3),
        "edge_on_all_paths": VerificationTask("edge_on_all_paths", u=1, v=4, e=(2, 3)),
        "e_cycle_containment": VerificationTask("e_cycle_containment", e=(1, 2)),
        "least_element_list": VerificationTask(
            "least_element_list", target=1, le_list=[(1, 0.0), (2, 1.0)]
        ),
    }
    eval_case_predicates = {case.task.predicate for case in get_eval_cases()}
    for row in rows:
        predicate = row.predicate
        task = tasks.get(predicate, VerificationTask(predicate=predicate))  # type: ignore[arg-type]
        result = verifier.verify(cases[predicate], task)
        assert result.predicate == predicate
        assert predicate in eval_case_predicates
