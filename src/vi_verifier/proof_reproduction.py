from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .types import GraphInput, VerificationTask, canonical_edge
from .verifier import Verifier


@dataclass(frozen=True)
class LowerBoundInput:
    p: int
    B: int
    n: int


@dataclass(frozen=True)
class LowerBoundResult:
    valid_domain: bool
    exponent: float
    lb_value: float
    witness_base: int | None


@dataclass(frozen=True)
class ReductionCheckResult:
    reduction_name: str
    passed: bool
    counterexample_case: str | None = None


@dataclass(frozen=True)
class ReductionWitness:
    case_name: str
    graph_input: GraphInput


def _domain_witness_base(inp: LowerBoundInput) -> int | None:
    if inp.p < 1 or inp.B < 1 or inp.n <= 0:
        return None
    denom = inp.p * inp.B
    if inp.n % denom != 0:
        return None
    k_power = inp.n // denom
    exp = (2 * inp.p) + 1
    k = round(k_power ** (1.0 / exp))
    for candidate in range(max(2, k - 2), k + 3):
        if candidate**exp == k_power:
            return candidate
    return None


def compute_deterministic_lb(inp: LowerBoundInput) -> LowerBoundResult:
    exponent = 0.5 - (1.0 / (2.0 * ((2 * inp.p) + 1)))
    witness_base = _domain_witness_base(inp)
    valid_domain = witness_base is not None
    lb_value = (inp.n / (inp.p * inp.B)) ** exponent if inp.p > 0 and inp.B > 0 else 0.0
    return LowerBoundResult(
        valid_domain=valid_domain,
        exponent=exponent,
        lb_value=lb_value,
        witness_base=witness_base,
    )


def _deterministic_h_minus_edge(graph_input: GraphInput) -> GraphInput:
    edges = sorted((canonical_edge(u, v) for u, v in graph_input.subgraph_edges), key=lambda edge: (repr(edge[0]), repr(edge[1])))
    if not edges:
        return graph_input
    removed = edges[0]
    return GraphInput(
        nodes=set(graph_input.nodes),
        edges=set(graph_input.edges),
        subgraph_edges={edge for edge in graph_input.subgraph_edges if canonical_edge(*edge) != removed},
        edge_weights=None if graph_input.edge_weights is None else dict(graph_input.edge_weights),
        ranks=None if graph_input.ranks is None else dict(graph_input.ranks),
    )


def _all_degree_two(graph_input: GraphInput) -> bool:
    degree: dict[object, int] = {node: 0 for node in graph_input.nodes}
    for u, v in graph_input.subgraph_edges:
        left, right = canonical_edge(u, v)
        degree[left] += 1
        degree[right] += 1
    return all(count == 2 for count in degree.values())


def check_reduction_equivalence(
    reduction_name: str,
    witnesses: list[ReductionWitness],
    lhs_verdict: Callable[[ReductionWitness], bool],
    rhs_verdict: Callable[[ReductionWitness], bool],
) -> ReductionCheckResult:
    for witness in witnesses:
        if bool(lhs_verdict(witness)) != bool(rhs_verdict(witness)):
            return ReductionCheckResult(
                reduction_name=reduction_name,
                passed=False,
                counterexample_case=witness.case_name,
            )
    return ReductionCheckResult(reduction_name=reduction_name, passed=True, counterexample_case=None)


def _hamiltonian_witnesses() -> list[ReductionWitness]:
    return [
        ReductionWitness(
            case_name="cycle_true",
            graph_input=GraphInput(
                nodes={1, 2, 3, 4},
                edges={(1, 2), (2, 3), (3, 4), (4, 1)},
                subgraph_edges={(1, 2), (2, 3), (3, 4), (4, 1)},
            ),
        ),
        ReductionWitness(
            case_name="path_false_degree",
            graph_input=GraphInput(
                nodes={1, 2, 3, 4},
                edges={(1, 2), (2, 3), (3, 4), (4, 1), (2, 4)},
                subgraph_edges={(1, 2), (2, 3), (3, 4)},
            ),
        ),
        ReductionWitness(
            case_name="two_cycles_false_disconnected",
            graph_input=GraphInput(
                nodes={1, 2, 3, 4, 5, 6},
                edges={(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)},
                subgraph_edges={(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)},
            ),
        ),
    ]


def _reduction_hamiltonian_to_spanning_tree() -> ReductionCheckResult:
    verifier = Verifier()
    witnesses = _hamiltonian_witnesses()
    return check_reduction_equivalence(
        reduction_name="hamiltonian_cycle_via_spanning_tree",
        witnesses=witnesses,
        lhs_verdict=lambda witness: verifier.verify(witness.graph_input, VerificationTask("hamiltonian_cycle")).verdict,
        rhs_verdict=lambda witness: _all_degree_two(witness.graph_input)
        and verifier.verify(_deterministic_h_minus_edge(witness.graph_input), VerificationTask("spanning_tree")).verdict,
    )


def _reduction_hamiltonian_to_simple_path() -> ReductionCheckResult:
    verifier = Verifier()
    witnesses = _hamiltonian_witnesses()
    return check_reduction_equivalence(
        reduction_name="hamiltonian_cycle_via_simple_path",
        witnesses=witnesses,
        lhs_verdict=lambda witness: verifier.verify(witness.graph_input, VerificationTask("hamiltonian_cycle")).verdict,
        rhs_verdict=lambda witness: _all_degree_two(witness.graph_input)
        and verifier.verify(_deterministic_h_minus_edge(witness.graph_input), VerificationTask("simple_path")).verdict,
    )


def run_reduction_reproduction_suite() -> list[ReductionCheckResult]:
    return [
        _reduction_hamiltonian_to_spanning_tree(),
        _reduction_hamiltonian_to_simple_path(),
    ]
