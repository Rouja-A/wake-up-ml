"""Pareto-frontier / dominance analysis: model A dominates model B if A's validation performance is
>= B's AND A's resource cost is <= B's, with at least one STRICT improvement -- exactly the definition in
the brief. Pure Python, no plotting/torch dependency, so it is unit-testable directly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParetoPoint:
    name: str
    metric: float     # e.g. mean validation macro-F1 (higher is better)
    cost: float        # e.g. MAC/window or trainable params (lower is better)


def is_dominated(point: ParetoPoint, others: list[ParetoPoint]) -> tuple[bool, str | None]:
    """Returns (dominated?, name_of_a_dominator_if_any). `point` is dominated if some OTHER point has
    metric >= point.metric AND cost <= point.cost, with at least one strict inequality."""
    for o in others:
        if o.name == point.name:
            continue
        metric_ge = o.metric >= point.metric
        cost_le = o.cost <= point.cost
        strict = (o.metric > point.metric) or (o.cost < point.cost)
        if metric_ge and cost_le and strict:
            return True, o.name
    return False, None


def pareto_frontier(points: list[ParetoPoint]) -> dict[str, dict]:
    """Returns {name: {"dominated": bool, "dominated_by": str|None}} for every point."""
    return {p.name: dict(zip(("dominated", "dominated_by"), is_dominated(p, points))) for p in points}
