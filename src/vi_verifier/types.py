from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import AbstractSet, Literal, Union

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


def canonicalize_edges(edges: Iterable[Edge]) -> frozenset[Edge]:
    return frozenset(canonical_edge(u, v) for (u, v) in edges)


@dataclass(frozen=True)
class GraphInput:
    nodes: AbstractSet[NodeId]
    edges: AbstractSet[Edge]
    subgraph_edges: AbstractSet[Edge]
    edge_weights: Mapping[Edge, float] | None = None
    ranks: Mapping[NodeId, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", frozenset(self.nodes))
        object.__setattr__(self, "edges", frozenset(self.edges))
        object.__setattr__(self, "subgraph_edges", frozenset(self.subgraph_edges))
        if self.edge_weights is not None:
            object.__setattr__(self, "edge_weights", MappingProxyType(dict(self.edge_weights)))
        if self.ranks is not None:
            object.__setattr__(self, "ranks", MappingProxyType(dict(self.ranks)))

    def canonicalized(self) -> GraphInput:
        canonical_edges = canonicalize_edges(self.edges)
        canonical_subgraph_edges = canonicalize_edges(self.subgraph_edges)
        unknown_edge_endpoints = {
            edge
            for edge in canonical_edges
            if edge[0] not in self.nodes or edge[1] not in self.nodes
        }
        if unknown_edge_endpoints:
            unknown = sorted(unknown_edge_endpoints, key=repr)
            raise ValueError(f"edges include endpoints not in nodes: {unknown}")
        unknown_subgraph_edges = canonical_subgraph_edges - canonical_edges
        if unknown_subgraph_edges:
            unknown = sorted(unknown_subgraph_edges, key=repr)
            raise ValueError(f"subgraph_edges include edges not in edges: {unknown}")
        canonical_weights: dict[Edge, float] | None = None
        if self.edge_weights is not None:
            canonical_weights = {}
            for (u, v), weight in self.edge_weights.items():
                edge = canonical_edge(u, v)
                if edge in canonical_weights and canonical_weights[edge] != weight:
                    raise ValueError(f"conflicting weights for edge {edge}")
                weight_f = float(weight)
                if weight_f < 0:
                    raise ValueError(f"edge {edge} has negative weight")
                canonical_weights[edge] = weight_f
            unknown_weight_edges = set(canonical_weights.keys()) - canonical_edges
            if unknown_weight_edges:
                unknown = sorted(unknown_weight_edges, key=repr)
                raise ValueError(f"edge_weights include edges not in edges: {unknown}")
        return GraphInput(
            nodes=self.nodes,
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
    le_list: Sequence[tuple[NodeId, float]] | None = None

    def __post_init__(self) -> None:
        if self.le_list is not None:
            object.__setattr__(self, "le_list", tuple(self.le_list))


@dataclass(frozen=True)
class VerificationResult:
    predicate: PredicateName
    verdict: bool
    details: dict[str, object] = field(default_factory=dict)
