# distributed-verification-vi

This repository provides a minimal, centralized reference implementation of the verification predicates from Roger Wattenhofer's 2026 Dijkstra Prize–winning paper [Distributed Verification and Hardness of Distributed Approximation](https://arxiv.org/abs/1011.3049).

The goal is a small, well-tested codebase that mirrors the paper's predicate definitions, not a faithful CONGEST/B-model simulator. See [Scope](#scope) for what is and is not implemented.

## Install

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## CLI

```bash
vi-verify list-cases
vi-verify run-case --case spanning_tree_true
vi-verify run-suite
vi-verify paper-matrix
vi-verify paper-matrix --format json
vi-verify reproduce-lower-bounds
vi-verify reproduce-lower-bounds --format json --p 2 --B 3 --n 192
```

## Library usage

```python
from vi_verifier.types import GraphInput, VerificationTask
from vi_verifier.verifier import Verifier

gi = GraphInput(
    nodes={1, 2, 3, 4},
    edges={(1, 2), (2, 3), (3, 4), (4, 1)},
    subgraph_edges={(1, 2), (2, 3), (3, 4)},
)
result = Verifier().verify(gi, VerificationTask(predicate="spanning_tree"))
print(result.verdict, result.details)
```

## Scope

Implemented predicates:

- spanning connected subgraph
- spanning tree
- cycle containment
- connectivity
- cut
- `s-t` connectivity
- `s-t` cut
- edge-on-all-paths
- `e`-cycle containment
- bipartiteness
- simple path
- Hamiltonian cycle
- least-element-list

### Assumptions

- Single-process centralized execution (no distributed round simulation).
- Undirected simple graphs.
- `H` is represented as edge membership over base graph `G`.
- NetworkX MST/connectivity routines are used as primitive operations.

### Deferred

- Communication-complexity simulations
- CONGEST / B-model round simulation
- Large-scale benchmarks / fuzzing
