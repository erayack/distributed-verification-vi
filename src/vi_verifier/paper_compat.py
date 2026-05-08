from __future__ import annotations

from dataclasses import dataclass

from .types import PaperFidelity


@dataclass(frozen=True)
class PaperCompatRow:
    predicate: str
    paper_section: str
    paper_rule: str
    status: PaperFidelity
    implementation_hook: str | None
    notes: str


def get_paper_compat_rows() -> list[PaperCompatRow]:
    return [
        PaperCompatRow(
            predicate="spanning_connected_subgraph",
            paper_section="paper/tightness.tex",
            paper_rule="MST of G' has cost 0 iff H is spanning connected subgraph",
            status="implemented",
            implementation_hook="Verifier.verify_spanning_connected_subgraph",
            notes="Centralized semantics of constructive criterion.",
        ),
        PaperCompatRow(
            predicate="spanning_tree",
            paper_section="paper/tightness.tex",
            paper_rule="MST cost criterion with tree edge count",
            status="implemented",
            implementation_hook="Verifier.verify_spanning_tree",
            notes="Uses MST weight and |E(H)| = |V|-1.",
        ),
        PaperCompatRow(
            predicate="cycle_containment",
            paper_section="paper/tightness.tex",
            paper_rule="Cycle-free equivalence via MST weight n-1-|E(H)|",
            status="implemented",
            implementation_hook="Verifier.verify_cycle_containment",
            notes="Cycle containment is negation of cycle-free criterion.",
        ),
        PaperCompatRow(
            predicate="connectivity",
            paper_section="paper/tightness.tex",
            paper_rule="Zero-weight edges in MST correspond to spanning forest size",
            status="implemented",
            implementation_hook="Verifier.verify_connectivity",
            notes="Incident-node interpretation follows paper definition context.",
        ),
        PaperCompatRow(
            predicate="cut",
            paper_section="paper/tightness.tex",
            paper_rule="Verify connectivity of G after removing H",
            status="implemented",
            implementation_hook="Verifier.verify_cut",
            notes="Direct centralized reduction.",
        ),
        PaperCompatRow(
            predicate="st_connectivity",
            paper_section="paper/tightness.tex",
            paper_rule="Component-label check (paper references Thurimella labels)",
            status="approximated",
            implementation_hook="Verifier.verify_st_connectivity",
            notes="Centralized path check substitutes distributed labeling protocol.",
        ),
        PaperCompatRow(
            predicate="st_cut",
            paper_section="paper/tightness.tex",
            paper_rule="s-t disconnected in G\\H",
            status="implemented",
            implementation_hook="Verifier.verify_st_cut",
            notes="Direct centralized reduction.",
        ),
        PaperCompatRow(
            predicate="edge_on_all_paths",
            paper_section="paper/tightness.tex",
            paper_rule="u,v disconnected in H\\{e}",
            status="implemented",
            implementation_hook="Verifier.verify_edge_on_all_paths",
            notes="Direct centralized reduction.",
        ),
        PaperCompatRow(
            predicate="e_cycle_containment",
            paper_section="paper/tightness.tex",
            paper_rule="Endpoints connected in H\\{e} iff e on a cycle",
            status="implemented",
            implementation_hook="Verifier.verify_e_cycle",
            notes="Direct centralized reduction.",
        ),
        PaperCompatRow(
            predicate="bipartiteness",
            paper_section="paper/tightness.tex",
            paper_rule="Two-coloring check after component process",
            status="approximated",
            implementation_hook="Verifier.verify_bipartiteness",
            notes="Centralized bipartite check substitutes distributed tree-level process.",
        ),
        PaperCompatRow(
            predicate="simple_path",
            paper_section="paper/prelim.tex and paper/deterministic_lb.tex",
            paper_rule="Nodes degree {0,2} except exactly two degree-1; no cycle",
            status="implemented",
            implementation_hook="Verifier.verify_simple_path",
            notes="Implemented as degree constraints + connected + acyclic on incident subgraph.",
        ),
        PaperCompatRow(
            predicate="hamiltonian_cycle",
            paper_section="paper/prelim.tex and paper/deterministic_lb.tex",
            paper_rule="Hamiltonian cycle verification definition and lower-bound treatment",
            status="implemented",
            implementation_hook="Verifier.verify_hamiltonian_cycle",
            notes="Centralized check uses degree-2 plus H\\{e} spanning-tree reduction.",
        ),
        PaperCompatRow(
            predicate="least_element_list",
            paper_section="paper/prelim.tex",
            paper_rule="LE-list verification definition",
            status="deferred",
            implementation_hook=None,
            notes="Out of scope in v1/v2.",
        ),
    ]


def get_paper_compat_row(predicate: str) -> PaperCompatRow:
    for row in get_paper_compat_rows():
        if row.predicate == predicate:
            return row
    raise KeyError(f"unknown predicate in paper compatibility registry: {predicate}")


def get_implemented_rows(include_approximated: bool = True) -> list[PaperCompatRow]:
    allowed = {"implemented"}
    if include_approximated:
        allowed.add("approximated")
    return [row for row in get_paper_compat_rows() if row.status in allowed]


def render_paper_matrix_text(rows: list[PaperCompatRow]) -> str:
    header = "predicate | paper_section | status | implementation_hook | notes"
    sep = "---|---|---|---|---"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"{row.predicate} | {row.paper_section} | {row.status} | "
            f"{row.implementation_hook or '-'} | {row.notes}"
        )
    return "\n".join(lines)
