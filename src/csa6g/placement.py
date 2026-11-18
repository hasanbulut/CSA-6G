"""Isolation-aware placement (paper: isolation-aware placement rule).

    Place(u, v) = 1  <=>  r_v >= d_u  and  sum_{u' in N(v)} delta(s_u', s_u) <= beta

with delta(s', s) = 1 iff same slice type and beta the per-node co-location
budget (default 1, i.e. anti-co-location: at most one slice of a given type per
node).  Nodes are scanned in ascending load order and the first feasible node
is taken, so the cost is O(|V| log|V| + |V|.|N|) per request, including the
sort.
"""
from __future__ import annotations

from typing import Iterable

from .model import Demand, Node, Reason, SliceType
from .params import COLOCATION_BUDGET_BETA


def colocation_count(node: Node, stype: SliceType) -> int:
    """sum_{u' in N(v)} delta(s_u', s_u)."""
    return sum(1 for s in node.hosted if s == stype)


def can_place(node: Node, demand: Demand, stype: SliceType,
              beta: int = COLOCATION_BUDGET_BETA) -> bool:
    """Evaluate the placement rule for one node."""
    return node.fits(demand) and colocation_count(node, stype) <= beta - 1


def select_node(nodes: Iterable[Node], demand: Demand, stype: SliceType,
                beta: int = COLOCATION_BUDGET_BETA) -> tuple[Node | None, str]:
    """Greedy first-fit with isolation filter.

    Returns ``(node, reason)``.  The reason distinguishes the two ways the placement rule
    can fail, which matters for the evaluation: a request denied for
    ``resource_infeasible`` is a capacity result, one denied for
    ``isolation_violation`` is a security result, and conflating them would
    make the isolation constraint look free when it is not.
    """
    nodes = sorted(nodes, key=lambda n: n.load)
    any_fits = False
    for node in nodes:
        fits = node.fits(demand)
        any_fits = any_fits or fits
        if fits and colocation_count(node, stype) <= beta - 1:
            return node, Reason.OK
    return None, (Reason.ISOLATION if any_fits else Reason.INFEASIBLE)
