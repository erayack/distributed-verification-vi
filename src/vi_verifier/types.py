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
]

PaperFidelity = Literal["implemented", "approximated", "deferred"]


def canonical_edge(u: NodeId, v: NodeId) -> Edge:
    """Return a stable undirected edge tuple."""
    left, right = sorted((repr(u), repr(v)))
    if left == repr(u):
        return (u, v)
    return (v, u)


def canonicalize_edges(edges: set[Edge]) -> set[Edge]:
    return {canonical_edge(u, v) for (u, v) in edges}


@dataclass(frozen=True)
class GraphInput:
    nodes: set[NodeId]
    edges: set[Edge]
    subgraph_edges: set[Edge]

    def canonicalized(self) -> GraphInput:
        return GraphInput(
            nodes=set(self.nodes),
            edges=canonicalize_edges(set(self.edges)),
            subgraph_edges=canonicalize_edges(set(self.subgraph_edges)),
        )


@dataclass(frozen=True)
class VerificationTask:
    predicate: PredicateName
    s: NodeId | None = None
    t: NodeId | None = None
    u: NodeId | None = None
    v: NodeId | None = None
    e: Edge | None = None


@dataclass(frozen=True)
class VerificationResult:
    predicate: PredicateName
    verdict: bool
    details: dict[str, object] = field(default_factory=dict)
