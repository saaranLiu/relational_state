"""
Identification filter for MCQ distractor selection.

Goal: when we build an 8-rule MCQ, we want to make sure the chosen distractors
produce x-values that are NUMERICALLY distinguishable from the gold's x-value.
Otherwise a model that correctly reasons about the rule might still pick a
different option simply because two options rounded to the same $-amount.

Margin definition
-----------------
    margin(rule_k, gold) := |x_gold - x_k| / max(F, 1e-9)

We use F (not gold) in the denominator to keep the threshold stable across
scenes: a $300 gap is "big" in a $400-baseline coffee decision but "noise" in
a $40,000-baseline car decision. F is the Langtry private floor, which scales
the whole problem.

Default thresholds
------------------
    TAU_POSITIONAL = 0.10   # d1 / d3 MCQ
    TAU_PLACEBO    = 0.08   # d2 placebo: gold=F, distractors use a "trial alpha"
    TAU_PAIRWISE   = 0.06   # eval-B: gold-vs-partner must be visibly different

These values keep distractors within a plausible narrative range while still
ensuring a readable numeric gap from the gold anchor. Callers may override
them via the `tau` argument.
"""
from __future__ import annotations

import random
from itertools import combinations
from typing import Dict, List, Sequence, Tuple


TAU_POSITIONAL = 0.10
TAU_PLACEBO = 0.08
TAU_PAIRWISE = 0.06

# Pair-wise margin between any two distractors (and also distractor-vs-gold):
# keeps options visibly distinct in display so correctness depends on rule
# judgement, not on two numbers happening to round to the same anchor.
TAU_PAIR_POSITIONAL = 0.05
TAU_PAIR_PLACEBO = 0.04
TAU_PAIR_MIN = 0.015


class NotEnoughDistractorsError(ValueError):
    """Raised when even the minimum-tau relaxation cannot yield enough distractors.

    Callers SHOULD catch this and resample the latent (keeping the same cell
    and scene) so that the parameter distribution is unchanged. Only fall
    back to padding with near-gold rules if resampling repeatedly fails.
    """

    def __init__(self, message: str, *, available: int, required: int,
                 min_tau: float, rule_x_values: Dict[str, float]) -> None:
        super().__init__(message)
        self.available = available
        self.required = required
        self.min_tau = min_tau
        self.rule_x_values = rule_x_values


def compute_margins(
    rule_x_values: Dict[str, float],
    gold_rule_id: str,
    F: float,
) -> Dict[str, float]:
    """Return {rule_id: margin_vs_gold} for every non-gold rule."""
    x_gold = rule_x_values[gold_rule_id]
    denom = max(F, 1e-9)
    return {
        rid: abs(val - x_gold) / denom
        for rid, val in rule_x_values.items()
        if rid != gold_rule_id
    }


def filter_distractors(
    rule_x_values: Dict[str, float],
    gold_rule_id: str,
    F: float,
    tau: float = TAU_POSITIONAL,
) -> List[str]:
    """Return distractor rule_ids whose predicted x is >= tau*F away from gold.

    Ordered by descending margin (the most distinguishable first) — callers
    may sample uniformly from this list, or take the top-K.
    """
    margins = compute_margins(rule_x_values, gold_rule_id, F)
    ranked = sorted(margins.items(), key=lambda kv: kv[1], reverse=True)
    return [rid for rid, m in ranked if m >= tau]


def filter_distractors_with_shortfall_relaxation(
    rule_x_values: Dict[str, float],
    gold_rule_id: str,
    F: float,
    required_count: int,
    tau: float = TAU_POSITIONAL,
    min_tau: float = 0.02,
    step: float = 0.01,
) -> Tuple[List[str], float]:
    """Try `tau`; if fewer than `required_count` distractors pass, relax to `min_tau`.

    Returns (distractor_ids, tau_used). Guarantees len >= required_count unless
    even at min_tau there are still too few separable rules (raises).
    """
    current_tau = tau
    while current_tau >= min_tau:
        ids = filter_distractors(rule_x_values, gold_rule_id, F, current_tau)
        if len(ids) >= required_count:
            return ids, current_tau
        current_tau = round(current_tau - step, 6)

    ids = filter_distractors(rule_x_values, gold_rule_id, F, min_tau)
    if len(ids) < required_count:
        raise NotEnoughDistractorsError(
            f"Cannot satisfy identification filter: need {required_count} distractors "
            f"with margin >= {min_tau}, got {len(ids)}.",
            available=len(ids),
            required=required_count,
            min_tau=min_tau,
            rule_x_values=dict(rule_x_values),
        )
    return ids, min_tau


# -----------------------------------------------------------------------------
# Stratified distractor selection (difficulty-balanced + pair-wise separable)
# -----------------------------------------------------------------------------

def _pairwise_margins(
    ids: Sequence[str], rule_x_values: Dict[str, float], F: float,
) -> List[Tuple[str, str, float]]:
    """Return [(id_i, id_j, margin_ij)] for every unordered pair in `ids`."""
    denom = max(F, 1e-9)
    out: List[Tuple[str, str, float]] = []
    for a, b in combinations(ids, 2):
        out.append((a, b, abs(rule_x_values[a] - rule_x_values[b]) / denom))
    return out


def _triple_pairwise_min(
    triple: Sequence[str], rule_x_values: Dict[str, float], F: float,
) -> float:
    margins = _pairwise_margins(triple, rule_x_values, F)
    return min((m for _, _, m in margins), default=float("inf"))


def select_stratified_distractors(
    rng: random.Random,
    rule_x_values: Dict[str, float],
    gold_rule_id: str,
    F: float,
    candidate_ids: Sequence[str],
    required_count: int = 3,
    tau_pair: float = TAU_PAIR_POSITIONAL,
    min_tau_pair: float = TAU_PAIR_MIN,
    step: float = 0.005,
) -> Tuple[List[str], float, bool, bool]:
    """Pick `required_count` distractors with (a) difficulty stratification and
    (b) pair-wise display separability.

    The caller supplies `candidate_ids` — rules that already cleared the
    gold-vs-distractor identification filter.

    Strategy
    --------
    1. Sort candidates by margin-vs-gold ASC. Split 50/50 into `near` (hardest
       to distinguish from gold; the same structural family) and `far`.
    2. Search combinations that draw at least one from `near` AND at least
       one from `far`, preferring combos with larger minimum pair-wise
       margin. Accept any combo whose min pair-wise margin >= tau_pair.
    3. If none satisfies tau_pair, relax to min_tau_pair in `step` increments.
    4. If still none, fall back to the max-min combo overall (stratified but
       possibly pair-wise-degraded).
    5. If stratification itself cannot be honoured (near and far both
       empty after overlap), return the top-3 by margin-vs-gold.

    Returns
    -------
    (picks, tau_pair_used, stratified_ok, pairwise_ok)

    `stratified_ok=False` means the near/far mix was impossible (rare — only
    when fewer than 2 distinct candidates exist). `pairwise_ok=False` means
    the final pick fell below the minimum pair-wise separability threshold
    (callers should flag this as a "selection_degraded" record).
    """
    cands = list(candidate_ids)
    if len(cands) < required_count:
        return list(cands), min_tau_pair, False, False

    denom = max(F, 1e-9)
    x_gold = rule_x_values[gold_rule_id]
    ranked = sorted(cands, key=lambda rid: abs(rule_x_values[rid] - x_gold) / denom)

    split = max(1, len(ranked) // 2)
    near = ranked[:split]
    far = ranked[split:]
    stratified_ok = len(near) >= 1 and len(far) >= 1

    # Enumerate all (required_count)-combinations.
    all_triples = list(combinations(ranked, required_count))

    def _is_stratified(tr: Sequence[str]) -> bool:
        return any(t in near for t in tr) and any(t in far for t in tr)

    def _best_for_threshold(threshold: float, triples: Sequence[Sequence[str]]):
        ok = [t for t in triples if _triple_pairwise_min(t, rule_x_values, F) >= threshold]
        if not ok:
            return None
        rng.shuffle(ok)
        return ok[0]

    stratified_triples = [t for t in all_triples if _is_stratified(t)] if stratified_ok else []

    current_tau = tau_pair
    while current_tau >= min_tau_pair:
        if stratified_triples:
            chosen = _best_for_threshold(current_tau, stratified_triples)
            if chosen is not None:
                return list(chosen), current_tau, True, True
        chosen = _best_for_threshold(current_tau, all_triples)
        if chosen is not None:
            return list(chosen), current_tau, _is_stratified(chosen), True
        current_tau = round(current_tau - step, 6)

    pool = stratified_triples if stratified_triples else all_triples
    best = max(pool, key=lambda t: _triple_pairwise_min(t, rule_x_values, F))
    effective = _triple_pairwise_min(best, rule_x_values, F)
    return list(best), effective, _is_stratified(best), False


def filter_distractors_with_hard_fallback(
    rule_x_values: Dict[str, float],
    gold_rule_id: str,
    F: float,
    required_count: int,
    tau: float = TAU_POSITIONAL,
    min_tau: float = 0.02,
    step: float = 0.01,
) -> Tuple[List[str], float, bool]:
    """Like `filter_distractors_with_shortfall_relaxation` but never raises.

    If the standard relaxation fails, pad the distractor pool with the
    non-gold rules that have the largest remaining margins (even if below
    `min_tau`). Returns (ids, tau_used, fallback_used). `fallback_used=True`
    indicates the record is "identification-degraded" and should be flagged
    downstream so eval metrics can optionally exclude it.
    """
    try:
        ids, tau_used = filter_distractors_with_shortfall_relaxation(
            rule_x_values, gold_rule_id, F, required_count, tau, min_tau, step,
        )
        return ids, tau_used, False
    except NotEnoughDistractorsError as err:
        margins = compute_margins(rule_x_values, gold_rule_id, F)
        ranked = sorted(margins.items(), key=lambda kv: kv[1], reverse=True)
        ids = [rid for rid, _ in ranked[:required_count]]
        # Use the smallest margin actually included as the effective tau.
        effective_tau = min((margins[rid] for rid in ids), default=0.0)
        return ids, effective_tau, True
