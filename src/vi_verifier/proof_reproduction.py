from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from .graph_ops import first_edge_by_repr
from .types import GraphInput, PredicateName, VerificationTask, canonical_edge
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


def compute_randomized_verification_lb(inp: LowerBoundInput) -> LowerBoundResult:
    """Return the randomized theorem's arithmetic bound shape.

    This does not model public coins or error probability; it only mirrors the
    p/B/n domain and exponent form used by the paper's randomized verification
    theorem.
    """
    return compute_deterministic_lb(inp)


def build_mst_approximation_reduction_input(graph_input: GraphInput, alpha: float) -> GraphInput:
    if alpha < 1:
        raise ValueError("alpha must be at least 1")
    canonical = graph_input.canonicalized()
    h_edges = {canonical_edge(*edge) for edge in canonical.subgraph_edges}
    high_weight = float(len(canonical.nodes)) * float(alpha)
    return replace(
        canonical,
        edge_weights={
            edge: 1.0 if canonical_edge(*edge) in h_edges else high_weight
            for edge in canonical.edges
        },
    )


def _deterministic_h_minus_edge(graph_input: GraphInput) -> GraphInput:
    removed = first_edge_by_repr(graph_input.subgraph_edges)
    if removed is None:
        return graph_input
    return replace(
        graph_input,
        subgraph_edges={
            edge for edge in graph_input.subgraph_edges if canonical_edge(*edge) != removed
        },
    )


def _complement_h(graph_input: GraphInput) -> GraphInput:
    h_edges = {canonical_edge(*edge) for edge in graph_input.subgraph_edges}
    return replace(
        graph_input,
        subgraph_edges={
            canonical_edge(*edge)
            for edge in graph_input.edges
            if canonical_edge(*edge) not in h_edges
        },
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
    return ReductionCheckResult(
        reduction_name=reduction_name, passed=True, counterexample_case=None
    )


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


def _cut_complement_witnesses() -> list[ReductionWitness]:
    return [
        ReductionWitness(
            case_name="path_spanning_connected",
            graph_input=GraphInput(
                nodes={1, 2, 3, 4},
                edges={(1, 2), (2, 3), (3, 4), (1, 4)},
                subgraph_edges={(1, 2), (2, 3), (3, 4)},
            ),
        ),
        ReductionWitness(
            case_name="empty_not_spanning_connected",
            graph_input=GraphInput(
                nodes={1, 2, 3},
                edges={(1, 2), (2, 3)},
                subgraph_edges=set(),
            ),
        ),
        ReductionWitness(
            case_name="two_components_not_spanning_connected",
            graph_input=GraphInput(
                nodes={1, 2, 3, 4},
                edges={(1, 2), (2, 3), (3, 4), (1, 4)},
                subgraph_edges={(1, 2), (3, 4)},
            ),
        ),
    ]


def _st_witnesses() -> list[ReductionWitness]:
    return [
        ReductionWitness(
            case_name="st_connected",
            graph_input=GraphInput(
                nodes={1, 2, 3, 4},
                edges={(1, 2), (2, 3), (3, 4), (1, 4)},
                subgraph_edges={(1, 2), (2, 3)},
            ),
        ),
        ReductionWitness(
            case_name="st_disconnected",
            graph_input=GraphInput(
                nodes={1, 2, 3, 4},
                edges={(1, 2), (2, 3), (3, 4), (1, 4)},
                subgraph_edges={(1, 2), (3, 4)},
            ),
        ),
    ]


def _e_cycle_witnesses() -> list[ReductionWitness]:
    return [
        ReductionWitness(
            case_name="edge_on_cycle",
            graph_input=GraphInput(
                nodes={1, 2, 3},
                edges={(1, 2), (2, 3), (1, 3)},
                subgraph_edges={(1, 2), (2, 3), (1, 3)},
            ),
        ),
        ReductionWitness(
            case_name="bridge_edge",
            graph_input=GraphInput(
                nodes={1, 2, 3},
                edges={(1, 2), (2, 3), (1, 3)},
                subgraph_edges={(1, 2), (2, 3)},
            ),
        ),
    ]


def _reduction_hamiltonian_via(predicate: PredicateName) -> ReductionCheckResult:
    verifier = Verifier()
    target_task = VerificationTask(predicate)
    return check_reduction_equivalence(
        reduction_name=f"hamiltonian_cycle_via_{predicate}",
        witnesses=_hamiltonian_witnesses(),
        lhs_verdict=lambda witness: (
            verifier.verify(witness.graph_input, VerificationTask("hamiltonian_cycle")).verdict
        ),
        rhs_verdict=lambda witness: (
            _all_degree_two(witness.graph_input)
            and verifier.verify(
                _deterministic_h_minus_edge(witness.graph_input), target_task
            ).verdict
        ),
    )


def _reduction_hamiltonian_to_spanning_tree() -> ReductionCheckResult:
    return _reduction_hamiltonian_via("spanning_tree")


def _reduction_hamiltonian_to_simple_path() -> ReductionCheckResult:
    return _reduction_hamiltonian_via("simple_path")


def _reduction_spanning_connected_to_cut_complement() -> ReductionCheckResult:
    verifier = Verifier()
    return check_reduction_equivalence(
        reduction_name="spanning_connected_subgraph_via_not_cut_complement",
        witnesses=_cut_complement_witnesses(),
        lhs_verdict=lambda witness: (
            verifier.verify(
                witness.graph_input,
                VerificationTask("spanning_connected_subgraph"),
            ).verdict
        ),
        rhs_verdict=lambda witness: (
            not verifier.verify(
                _complement_h(witness.graph_input),
                VerificationTask("cut"),
            ).verdict
        ),
    )


def _reduction_st_connectivity_to_not_st_cut_complement() -> ReductionCheckResult:
    verifier = Verifier()
    return check_reduction_equivalence(
        reduction_name="st_connectivity_via_not_st_cut_complement",
        witnesses=_st_witnesses(),
        lhs_verdict=lambda witness: (
            verifier.verify(
                witness.graph_input,
                VerificationTask("st_connectivity", s=1, t=3),
            ).verdict
        ),
        rhs_verdict=lambda witness: (
            not verifier.verify(
                _complement_h(witness.graph_input),
                VerificationTask("st_cut", s=1, t=3),
            ).verdict
        ),
    )


def _reduction_e_cycle_to_not_edge_on_all_paths() -> ReductionCheckResult:
    verifier = Verifier()
    edge = (1, 2)
    return check_reduction_equivalence(
        reduction_name="e_cycle_containment_via_not_edge_on_all_paths",
        witnesses=_e_cycle_witnesses(),
        lhs_verdict=lambda witness: (
            verifier.verify(
                witness.graph_input,
                VerificationTask("e_cycle_containment", e=edge),
            ).verdict
        ),
        rhs_verdict=lambda witness: (
            not verifier.verify(
                witness.graph_input,
                VerificationTask("edge_on_all_paths", u=1, v=2, e=edge),
            ).verdict
        ),
    )


def run_reduction_reproduction_suite(
    include_reference_expansions: bool = False,
) -> list[ReductionCheckResult]:
    results = [
        _reduction_hamiltonian_to_spanning_tree(),
        _reduction_hamiltonian_to_simple_path(),
    ]
    if include_reference_expansions:
        results.extend(
            [
                _reduction_spanning_connected_to_cut_complement(),
                _reduction_st_connectivity_to_not_st_cut_complement(),
                _reduction_e_cycle_to_not_edge_on_all_paths(),
            ]
        )
    return results
