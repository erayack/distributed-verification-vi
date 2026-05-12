from __future__ import annotations

import json

from vi_verifier.cli import main as cli_main
from vi_verifier.proof_reproduction import (
    LowerBoundInput,
    ReductionWitness,
    build_mst_approximation_reduction_input,
    check_reduction_equivalence,
    compute_deterministic_lb,
    compute_randomized_verification_lb,
    run_reduction_reproduction_suite,
)
from vi_verifier.types import GraphInput


def test_compute_deterministic_lb_domain_membership() -> None:
    valid = compute_deterministic_lb(LowerBoundInput(p=2, B=3, n=(2**5) * 2 * 3))
    invalid = compute_deterministic_lb(LowerBoundInput(p=2, B=3, n=100))
    assert valid.valid_domain is True
    assert valid.witness_base == 2
    assert invalid.valid_domain is False
    assert invalid.witness_base is None


def test_compute_randomized_verification_lb_matches_theorem_domain_shape() -> None:
    result = compute_randomized_verification_lb(LowerBoundInput(p=3, B=2, n=(3**7) * 3 * 2))
    assert result.valid_domain is True
    assert result.witness_base == 3
    assert (
        result.exponent
        == compute_deterministic_lb(LowerBoundInput(p=3, B=2, n=(3**7) * 3 * 2)).exponent
    )


def test_build_mst_approximation_reduction_input_weights_h_edges_low() -> None:
    graph_input = GraphInput(
        nodes={1, 2, 3},
        edges={(1, 2), (2, 3), (1, 3)},
        subgraph_edges={(2, 1), (2, 3)},
    )
    reduced = build_mst_approximation_reduction_input(graph_input, alpha=2.0)
    assert reduced.edge_weights == {
        (1, 2): 1.0,
        (2, 3): 1.0,
        (1, 3): 6.0,
    }


def test_reduction_reproduction_suite_passes() -> None:
    results = run_reduction_reproduction_suite()
    assert results
    assert all(result.passed for result in results)


def test_reduction_reproduction_suite_contains_reference_reductions() -> None:
    names = {
        result.reduction_name
        for result in run_reduction_reproduction_suite(include_reference_expansions=True)
    }
    assert names == {
        "hamiltonian_cycle_via_spanning_tree",
        "hamiltonian_cycle_via_simple_path",
        "spanning_connected_subgraph_via_not_cut_complement",
        "st_connectivity_via_not_st_cut_complement",
        "e_cycle_containment_via_not_edge_on_all_paths",
    }


def test_cli_reproduce_lower_bounds_json(capsys) -> None:
    assert (
        cli_main(
            [
                "reproduce-lower-bounds",
                "--format",
                "json",
                "--p",
                "2",
                "--B",
                "3",
                "--n",
                str((2**5) * 2 * 3),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "deterministic_lb" in payload
    assert "reduction_checks" in payload
    assert len(payload["reduction_checks"]) == 2
    assert payload["deterministic_lb"][0]["p"] == 2
    assert payload["deterministic_lb"][0]["B"] == 3
    assert payload["deterministic_lb"][0]["n"] == (2**5) * 2 * 3


def test_cli_reproduce_lower_bounds_rejects_partial_tuple() -> None:
    assert cli_main(["reproduce-lower-bounds", "--p", "2"]) == 2


def test_equivalence_check_reports_counterexample() -> None:
    result = check_reduction_equivalence(
        reduction_name="synthetic_equivalence",
        witnesses=[
            ReductionWitness(case_name="bad_case", graph_input=GraphInput(set(), set(), set()))
        ],
        lhs_verdict=lambda _: True,
        rhs_verdict=lambda _: False,
    )
    assert result.passed is False
    assert result.counterexample_case == "bad_case"
