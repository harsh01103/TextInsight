"""PageRank by power iteration, written from scratch for TextRank."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TypeVar

__all__ = ["pagerank"]

Node = TypeVar("Node", bound=Hashable)


def pagerank(
    graph: Mapping[Node, Mapping[Node, float]],
    *,
    damping: float = 0.85,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    personalization: Mapping[Node, float] | None = None,
) -> dict[Node, float]:
    """Return a score per node; the scores sum to one.

    Args:
        graph: Maps each node to its neighbours and edge weights. Every neighbour
            must also be a key of ``graph``.
        damping: Probability of following an edge instead of jumping at random.
        tolerance: Stop when the total change in scores falls below this.
        max_iterations: Upper limit on the number of iterations.
        personalization: Where random jumps land, as weights that sum to one.
            Jumps are uniform when this is omitted.
    """
    if not graph:
        return {}
    size = len(graph)
    strengths = {node: sum(edges.values()) for node, edges in graph.items()}
    jumps = personalization if personalization is not None else dict.fromkeys(graph, 1.0 / size)
    scores = dict.fromkeys(graph, 1.0 / size)
    for _ in range(max_iterations):
        # Nodes without edges pass their score to the jump targets, so the total stays one.
        stranded = sum(scores[node] for node, strength in strengths.items() if not strength)
        teleport = 1.0 - damping + damping * stranded
        updated = {node: teleport * jumps[node] for node in graph}
        for node, edges in graph.items():
            if strengths[node]:
                share = damping * scores[node] / strengths[node]
                for neighbour, weight in edges.items():
                    updated[neighbour] += share * weight
        change = sum(abs(updated[node] - scores[node]) for node in graph)
        scores = updated
        if change < tolerance:
            break
    return scores
