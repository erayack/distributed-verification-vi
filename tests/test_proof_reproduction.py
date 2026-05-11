from __future__ import annotations

import json

from vi_verifier.cli import main as cli_main
from vi_verifier.proof_reproduction import (
    LowerBoundInput,
    ReductionWitness,
    check_reduction_equivalence,
    compute_deterministic_lb,
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


def test_reduction_reproduction_suite_passes() -> None:
    results = run_reduction_reproduction_suite()
    assert results
    assert all(result.passed for result in results)


def test_cli_reproduce_lower_bounds_json(capsys) -> None:
    assert (
        cli_main(
            ["reproduce-lower-bounds", "--format", "json", "--p", "2", "--B", "3", "--n", str((2**5) * 2 * 3)]
        )
        == 0
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "deterministic_lb" in payload
    assert "reduction_checks" in payload
    assert payload["deterministic_lb"][0]["p"] == 2
    assert payload["deterministic_lb"][0]["B"] == 3
    assert payload["deterministic_lb"][0]["n"] == (2**5) * 2 * 3


def test_cli_reproduce_lower_bounds_rejects_partial_tuple() -> None:
    assert cli_main(["reproduce-lower-bounds", "--p", "2"]) == 2


def test_equivalence_check_reports_counterexample() -> None:
    result = check_reduction_equivalence(
        reduction_name="synthetic_equivalence",
        witnesses=[ReductionWitness(case_name="bad_case", graph_input=GraphInput(set(), set(), set()))],
        lhs_verdict=lambda _: True,
        rhs_verdict=lambda _: False,
    )
    assert result.passed is False
    assert result.counterexample_case == "bad_case"
