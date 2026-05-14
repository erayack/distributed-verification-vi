from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

NodeId = Union[int, str]
Edge = tuple[NodeId, NodeId]

PredicateName = Literal[
    "spanning_connected_subgraph",
    "spanning_tree",
    "cycle_containment",
    "connectivity",
    "cut",
    "st_connectivity",
    "st_cut",
    "edge_on_all_paths",
    "e_cycle_containment",
    "bipartiteness",
    "simple_path",
    "hamiltonian_cycle",
    "least_element_list",
]

PaperFidelity = Literal["implemented", "approximated", "deferred"]


def canonical_edge(u: NodeId, v: NodeId) -> Edge:
    """Return a stable undirected edge tuple."""
    u_repr = repr(u)
    v_repr = repr(v)
    if u_repr <= v_repr:
        return (u, v)
    return (v, u)


def canonicalize_edges(edges: set[Edge]) -> set[Edge]:
    return {canonical_edge(u, v) for (u, v) in edges}


@dataclass(frozen=True)
class GraphInput:
    nodes: set[NodeId]
    edges: set[Edge]
    subgraph_edges: set[Edge]
    edge_weights: dict[Edge, float] | None = None
    ranks: dict[NodeId, int] | None = None

    def canonicalized(self) -> GraphInput:
        canonical_edges = canonicalize_edges(self.edges)
        canonical_subgraph_edges = canonicalize_edges(self.subgraph_edges)
        canonical_weights: dict[Edge, float] | None = None
        if self.edge_weights is not None:
            canonical_weights = {}
            for (u, v), weight in self.edge_weights.items():
                edge = canonical_edge(u, v)
                if edge in canonical_weights and canonical_weights[edge] != weight:
                    raise ValueError(f"conflicting weights for edge {edge}")
                canonical_weights[edge] = float(weight)
            unknown_weight_edges = set(canonical_weights.keys()) - canonical_edges
            if unknown_weight_edges:
                unknown = sorted(unknown_weight_edges, key=repr)
                raise ValueError(f"edge_weights include edges not in edges: {unknown}")
        return GraphInput(
            nodes=self.nodes.copy(),
            edges=canonical_edges,
            subgraph_edges=canonical_subgraph_edges,
            edge_weights=canonical_weights,
            ranks=None if self.ranks is None else dict(self.ranks),
        )


@dataclass(frozen=True)
class VerificationTask:
    predicate: PredicateName
    s: NodeId | None = None
    t: NodeId | None = None
    u: NodeId | None = None
    v: NodeId | None = None
    e: Edge | None = None
    target: NodeId | None = None
    le_list: list[tuple[NodeId, float]] | None = None


@dataclass(frozen=True)
class VerificationResult:
    predicate: PredicateName
    verdict: bool
    details: dict[str, object] = field(default_factory=dict)
