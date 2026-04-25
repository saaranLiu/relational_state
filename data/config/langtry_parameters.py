"""
Langtry (2024) parameter schema aligned with the original paper notation.

Paper reference: Langtry, "Keeping up with 'The Joneses': reference dependent
choice with social comparisons", 2024.

Notation (strictly aligned with the paper):
    x_i         : individual i's action level (own consumption / effort)
    alpha_i     : comparison intensity of individual i  (alpha_i >= 0)
    g_ij        : network / closeness weight from i to j; Sum_j g_ij = 1
    x_j         : peer j's action level
    F           : private baseline  (Langtry's (f')^{-1}(c); the no-comparison floor)
    c           : marginal cost
    f(.)        : strictly concave utility kernel (not instantiated explicitly)
    Sum_j g_ij*x_j  : social reference aggregate (scalar)
    x_i*        : optimal action  (closed form, Langtry best response):
                    x_i* = F + alpha_i * Sum_j g_ij * x_j
                  which collapses to x_i* = F when alpha_i ~= 0 (non-positional
                  scene; Placebo).

Additional notation for the two Langtry OOD tasks:
    Social decision (Eq. 1 / Prop. 4):
        u_i = f(x_i - alpha_i Sum_j g_ij x_j) - c*x_i + b_i*alpha_i*Sum_j g_ij
        x_i* = b_i / c
        Pairwise stability requires b_i/c == b_j/c.

    Career decision (Eq. 4 / Prop. 6):
        u_i = f(x_i - alpha_1i Sum_{j in friends} g_ij x_j - alpha_2i * x_bar_m)
              + b*alpha_1i*Sum_j g_ij
        Choose firm H over L iff
            (x_S_H - x_S_L) >= alpha_2i * (x_bar_H - x_bar_L)

This module is PURE configuration + sampling of latent parameters. It emits
NO natural-language text and performs NO teacher calls. Narrative / MCQ
materials live in scene_pool.py, rule_label_templates.py, vocabulary_pools.py.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# 0. Version / registry constants
# -----------------------------------------------------------------------------

LANGTRY_SCHEMA_VERSION = "langtry_v1"

# The "main" action domains (training + Eval-A + Eval-B) use the peer-weighted
# closed-form x_i* = F + alpha_i * Sum_j g_ij x_j. The placebo domain uses the
# same action space but with alpha_i clamped near zero so that gold = F.
MAIN_POSITIONAL_DOMAINS: Tuple[str, ...] = (
    "domain1_positional_consumption",
    "domain3_effort_and_human_capital",
)
PLACEBO_DOMAIN: str = "domain2_non_positional_investment"

# OOD domains (Eval-D / Eval-E): NOT used for training, only held-out test.
OOD_DOMAIN_SOCIAL: str = "domain4_social_network_formation"       # Langtry Eq. 1
OOD_DOMAIN_CAREER: str = "domain5_career_sorting"                 # Langtry Eq. 4


# -----------------------------------------------------------------------------
# 1. alpha_i buckets (comparison intensity)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BucketSpec:
    """Open-interval parameter bucket (lower, upper), exclusive at both ends."""
    label: str
    lower: float
    upper: float


# Main-task buckets. These are identical for d1 and d3 because the
# comparison-intensity psychology applies symmetrically (consumption vs effort).
ALPHA_BUCKETS: Dict[str, BucketSpec] = {
    "low":  BucketSpec(label="low",  lower=0.05, upper=0.20),
    "mid":  BucketSpec(label="mid",  lower=0.40, upper=0.60),
    "high": BucketSpec(label="high", lower=0.80, upper=0.95),
}

# Placebo override for d2: alpha_i must be effectively zero in Langtry's
# non-positional regime. We still sample in a tiny strictly-positive interval
# so that the machinery is unchanged (avoids singular g_ij normalisation) but
# the closed-form x_i* = F + alpha * Sum g_ij x_j is arithmetically
# indistinguishable from F at the formatting precision we use.
ALPHA_PLACEBO_BUCKET: BucketSpec = BucketSpec(label="placebo", lower=0.00, upper=0.05)


# -----------------------------------------------------------------------------
# 2. g_ij dispersion buckets (how spread / skewed the closeness weights are)
# -----------------------------------------------------------------------------
# "dispersion" parameterises the peer-weight distribution shape, NOT the peer
# action values. It answers: does this persona weigh one peer much more than
# the others, or roughly uniformly?
#
# We pick the top weight g_max (= g_i*, the closeness to the single closest
# peer) from a bucket, then distribute (1 - g_max) across remaining peers using
# a Dirichlet draw with concentration `rest_conc`. Lower bucket => higher
# uniformity; higher bucket => one dominant peer.
@dataclass(frozen=True)
class DispersionSpec:
    label: str
    g_max_lower: float
    g_max_upper: float
    rest_concentration: float  # Dirichlet concentration for non-top peers


DISPERSION_BUCKETS: Dict[str, DispersionSpec] = {
    # Near-uniform: closest peer only marginally above the rest.
    "low":  DispersionSpec(label="low",  g_max_lower=0.30, g_max_upper=0.40, rest_concentration=8.0),
    # Moderate skew: closest peer carries roughly half the weight.
    "mid":  DispersionSpec(label="mid",  g_max_lower=0.45, g_max_upper=0.60, rest_concentration=3.0),
    # Highly skewed: one dominant peer; the rest are noise.
    "high": DispersionSpec(label="high", g_max_lower=0.70, g_max_upper=0.85, rest_concentration=1.0),
}


# -----------------------------------------------------------------------------
# 3. x_j skew buckets (spread of peer action levels relative to F)
# -----------------------------------------------------------------------------
# "skew" parameterises where peer actions sit relative to the private baseline
# F. The persona sees N peers whose x_j are drawn from a log-normal around
# F * mean_ratio with variance shape controlled by `sigma_log`. A top-x_j
# scaling factor is applied to the single closest peer so that the gold MCQ
# distractors (top-anchoring, closest-only) are distinguishable from the
# Langtry-weighted gold.

@dataclass(frozen=True)
class SkewSpec:
    label: str
    mean_ratio_lower: float   # E[x_j] / F lower bound
    mean_ratio_upper: float   # E[x_j] / F upper bound
    sigma_log: float          # log-normal sigma on the multiplier
    top_boost_lower: float    # multiplicative boost applied to the highest-weight peer's x_j
    top_boost_upper: float


SKEW_BUCKETS: Dict[str, SkewSpec] = {
    # Peers cluster near the baseline: no obvious high roller.
    "low":  SkewSpec(label="low",  mean_ratio_lower=1.00, mean_ratio_upper=1.25,
                     sigma_log=0.08, top_boost_lower=1.05, top_boost_upper=1.15),
    # Peers sit comfortably above F; mild top-bottom spread.
    "mid":  SkewSpec(label="mid",  mean_ratio_lower=1.40, mean_ratio_upper=1.85,
                     sigma_log=0.15, top_boost_lower=1.15, top_boost_upper=1.35),
    # Clear aspirational spread: a rich peer visibly above the rest.
    "high": SkewSpec(label="high", mean_ratio_lower=2.20, mean_ratio_upper=3.20,
                     sigma_log=0.25, top_boost_lower=1.35, top_boost_upper=1.70),
}


BUCKET_ORDER: Tuple[str, ...] = ("low", "mid", "high")


# -----------------------------------------------------------------------------
# 4. Parameter-cell taxonomy (alpha x dispersion x skew = 27 cells)
# -----------------------------------------------------------------------------
# Six cells are held out from training (Eval-A held-out probe) to probe
# generalisation across the (alpha, dispersion, skew) simplex. They are chosen
# to cover each dimension at least once and avoid clustering in any corner.

HELD_OUT_CELLS: Tuple[Tuple[str, str, str], ...] = (
    ("high", "low",  "low"),   # heavy comparator facing a flat uniform peer field
    ("low",  "high", "high"),  # independent persona facing a single rich anchor
    ("mid",  "low",  "high"),  # moderate persona, uniform weights, but rich spread
    ("high", "high", "low"),   # heavy comparator fixated on one peer, peers similar
    ("low",  "mid",  "mid"),   # low-alpha safe middle; hardest to flag as positional
    ("mid",  "mid",  "high"),  # middle + spread: tests aggregation under ambiguity
)


def iter_all_cells() -> List[Tuple[str, str, str]]:
    return [
        (alpha, disp, skew)
        for alpha in BUCKET_ORDER
        for disp in BUCKET_ORDER
        for skew in BUCKET_ORDER
    ]


def iter_main_cells() -> List[Tuple[str, str, str]]:
    """21 training / Eval-A-main cells (27 - 6 held-out)."""
    held = set(HELD_OUT_CELLS)
    return [cell for cell in iter_all_cells() if cell not in held]


def is_held_out_cell(alpha_b: str, disp_b: str, skew_b: str) -> bool:
    return (alpha_b, disp_b, skew_b) in set(HELD_OUT_CELLS)


def build_cell_id(alpha_b: str, disp_b: str, skew_b: str) -> str:
    return f"alpha_{alpha_b}__disp_{disp_b}__skew_{skew_b}"


def parse_cell_id(cell_id: str) -> Tuple[str, str, str]:
    try:
        a_part, d_part, s_part = cell_id.split("__")
        alpha_b = a_part[len("alpha_"):]
        disp_b = d_part[len("disp_"):]
        skew_b = s_part[len("skew_"):]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid cell_id `{cell_id}`") from exc
    if alpha_b not in BUCKET_ORDER or disp_b not in BUCKET_ORDER or skew_b not in BUCKET_ORDER:
        raise ValueError(f"Invalid cell_id `{cell_id}`")
    return alpha_b, disp_b, skew_b


# -----------------------------------------------------------------------------
# 5. Peer count N
# -----------------------------------------------------------------------------
PEER_COUNT_CHOICES: Tuple[int, ...] = (3, 4, 5)


# -----------------------------------------------------------------------------
# 6. Elementary sampling helpers
# -----------------------------------------------------------------------------

def _open_uniform(rng: random.Random, low: float, high: float) -> float:
    """Sample from the open interval (low, high). Handles edge precision."""
    if hasattr(math, "nextafter"):
        low_open = math.nextafter(low, math.inf)
        high_open = math.nextafter(high, -math.inf)
    else:  # pragma: no cover (kept for older pythons)
        step = max(abs(low), abs(high), 1.0) * 1e-12
        low_open = low + step
        high_open = high - step
    if not low_open < high_open:
        return (low + high) / 2.0
    return rng.uniform(low_open, high_open)


def _sample_bucket_value(rng: random.Random, bucket_map: Dict[str, BucketSpec], label: str) -> float:
    spec = bucket_map[label]
    return _open_uniform(rng, spec.lower, spec.upper)


def _dirichlet(rng: random.Random, concentration: float, k: int) -> List[float]:
    """Fallback Dirichlet via Gamma(shape=conc, scale=1) normalisation."""
    if k <= 0:
        return []
    draws = [rng.gammavariate(concentration, 1.0) for _ in range(k)]
    s = sum(draws) or 1.0
    return [d / s for d in draws]


# -----------------------------------------------------------------------------
# 7. Main latent sampler (d1, d3 positional, d2 placebo)
# -----------------------------------------------------------------------------

@dataclass
class LatentLangtryRecord:
    """A single latent draw for the MAIN task (d1 / d3 / d2).

    Attributes mirror Langtry notation exactly; no extra aliases.
    """
    domain_key: str
    alpha_i: float
    F: float
    c: float
    N_peers: int
    x_j: List[float]       # peer action levels, length N_peers
    g_ij: List[float]      # closeness weights, length N_peers, sum to 1
    ref_sum: float         # Sum_j g_ij * x_j   (Langtry's reference aggregate)
    x_i_star: float        # gold action under Rule-A (peer-weighted)
    alpha_bucket: str
    dispersion_bucket: str
    skew_bucket: str
    cell_id: str
    is_held_out_cell: bool
    is_placebo: bool
    peer_rank_by_g: List[int]   # indices of peers sorted by g_ij descending
    peer_rank_by_x: List[int]   # indices of peers sorted by x_j descending


def _sample_peer_field(
    rng: random.Random,
    F: float,
    N: int,
    dispersion_bucket: str,
    skew_bucket: str,
) -> Tuple[List[float], List[float]]:
    """Sample (x_j, g_ij) for N peers.

    - g_ij distribution: one dominant entry drawn from DISPERSION_BUCKETS, the
      rest by Dirichlet with concentration = rest_concentration. Normalised
      (sum exactly 1.0).
    - x_j distribution: log-normal around F * mean_ratio; the peer that is
      already the closest (largest g_ij) receives an additional multiplicative
      top_boost so that the "top-anchoring" distractor is detectably above the
      closeness-weighted gold even after g-based averaging.

    Returns (x_j, g_ij), both of length N.
    """
    disp = DISPERSION_BUCKETS[dispersion_bucket]
    skew = SKEW_BUCKETS[skew_bucket]

    # --- g_ij ---
    g_max = _open_uniform(rng, disp.g_max_lower, disp.g_max_upper)
    g_max = max(min(g_max, 0.98), 1.0 / N)  # guardrails
    rest_total = 1.0 - g_max
    if N == 1:
        g_ij = [1.0]
    else:
        rest_ws = _dirichlet(rng, disp.rest_concentration, N - 1)
        rest_ws = [w * rest_total for w in rest_ws]
        # top peer comes first, shuffled later to avoid positional leak
        g_ij = [g_max] + rest_ws
        # Re-normalise against fp drift.
        s = sum(g_ij)
        g_ij = [w / s for w in g_ij]

    # --- x_j ---
    mean_ratio = _open_uniform(rng, skew.mean_ratio_lower, skew.mean_ratio_upper)
    x_j_raw: List[float] = []
    for _ in range(N):
        mu = math.log(max(F * mean_ratio, 1e-9))
        sample = math.exp(rng.gauss(mu, skew.sigma_log))
        x_j_raw.append(sample)
    # Apply top_boost to the peer with the largest g_ij (index 0 by construction).
    top_boost = _open_uniform(rng, skew.top_boost_lower, skew.top_boost_upper)
    x_j_raw[0] = x_j_raw[0] * top_boost

    # Shuffle so positional index of the closest peer is not always 0.
    # (Keep g_ij and x_j paired during shuffle.)
    pairs = list(zip(x_j_raw, g_ij))
    rng.shuffle(pairs)
    x_j_out = [p[0] for p in pairs]
    g_ij_out = [p[1] for p in pairs]

    return x_j_out, g_ij_out


def sample_latent_langtry(
    rng: random.Random,
    domain_key: str,
    F: float,
    alpha_bucket: str,
    dispersion_bucket: str,
    skew_bucket: str,
    N: Optional[int] = None,
    marginal_cost: float = 1.0,
) -> LatentLangtryRecord:
    """Draw one latent Langtry record for the main task.

    Args:
        rng: deterministic RNG shared across the whole dataset build.
        domain_key: one of MAIN_POSITIONAL_DOMAINS or PLACEBO_DOMAIN.
        F: private baseline (already drawn, from the scene pool / template).
        alpha_bucket, dispersion_bucket, skew_bucket: 3D cell.
        N: number of peers; if None, sampled from PEER_COUNT_CHOICES.
        marginal_cost: c (placeholder; Langtry's c is a scalar that does not
            interact with the peer-weighted closed form for our purposes, but
            we record it for the OOD tasks that do depend on it).

    Returns:
        LatentLangtryRecord with x_i_star already computed via
        x_i_star = F + alpha_i * Sum_j g_ij * x_j   for positional domains,
        and alpha_i ~ 0 (placebo bucket) with x_i_star ~ F for d2.
    """
    is_placebo = domain_key == PLACEBO_DOMAIN
    if is_placebo:
        alpha_i = _sample_bucket_value(
            rng, {"placebo": ALPHA_PLACEBO_BUCKET}, "placebo"
        )
    else:
        alpha_i = _sample_bucket_value(rng, ALPHA_BUCKETS, alpha_bucket)

    N_pick = N if N is not None else rng.choice(PEER_COUNT_CHOICES)
    x_j, g_ij = _sample_peer_field(rng, F, N_pick, dispersion_bucket, skew_bucket)

    ref_sum = sum(g * x for g, x in zip(g_ij, x_j))
    x_i_star = F + alpha_i * ref_sum

    peer_rank_by_g = sorted(range(N_pick), key=lambda i: -g_ij[i])
    peer_rank_by_x = sorted(range(N_pick), key=lambda i: -x_j[i])

    cell = build_cell_id(alpha_bucket, dispersion_bucket, skew_bucket)
    return LatentLangtryRecord(
        domain_key=domain_key,
        alpha_i=alpha_i,
        F=F,
        c=marginal_cost,
        N_peers=N_pick,
        x_j=x_j,
        g_ij=g_ij,
        ref_sum=ref_sum,
        x_i_star=x_i_star,
        alpha_bucket=alpha_bucket,
        dispersion_bucket=dispersion_bucket,
        skew_bucket=skew_bucket,
        cell_id=cell,
        is_held_out_cell=is_held_out_cell(alpha_bucket, dispersion_bucket, skew_bucket),
        is_placebo=is_placebo,
        peer_rank_by_g=peer_rank_by_g,
        peer_rank_by_x=peer_rank_by_x,
    )


# -----------------------------------------------------------------------------
# 8. OOD: social-network formation (d4 -- Langtry Eq. 1 / Prop. 4)
# -----------------------------------------------------------------------------
# Each d4 item presents the protagonist + 2 candidate peers. The decision is
# which peer is likely to stabilise into a mutual friendship.
#
# b_i / c_i is the individual's "friendship benefit-cost ratio". Pairwise
# stability requires b_i/c_i == b_j/c_j. The public record exposes only coarse
# semantic buckets for b and c; exact numeric values and ratios remain oracle-only.
#
# Narratively: b is rendered as the value from forming the tie, while c is the
# marginal cost of the same action x, consistent with the main Eval-A utility.

@dataclass
class LatentSocialOODRecord:
    """d4 OOD draw: protagonist i + candidates Mia, Jake."""
    domain_key: str                        # OOD_DOMAIN_SOCIAL
    c_i: float                             # protagonist marginal cost
    b_i: float                             # protagonist's friendship benefit
    x_i: float                             # protagonist's preferred action level (b_i / c_i)
    c_candidates: Tuple[float, float]      # candidate marginal costs
    b_candidates: Tuple[float, float]      # (Mia's b, Jake's b)
    x_candidates: Tuple[float, float]      # (Mia's x = b_Mia/c_Mia, Jake's x)
    bc_ratio_i: float                      # b_i / c_i
    bc_ratio_candidates: Tuple[float, float]
    b_bucket_i: str
    c_bucket_i: str
    b_bucket_candidates: Tuple[str, str]
    c_bucket_candidates: Tuple[str, str]
    gold_candidate_index: int              # 0 for Mia, 1 for Jake; -1 means neither
    match_distance_bucket: str             # {close, mid, far}  ← subgroup tag


SOCIAL_MATCH_DISTANCE_BUCKETS: Tuple[str, ...] = ("close", "mid", "far")
SOCIAL_B_BUCKETS: Tuple[str, ...] = ("low", "mid", "high")
SOCIAL_C_BUCKETS: Tuple[str, ...] = ("low", "mid", "high")

_SOCIAL_BUCKET_SCORE: Dict[str, float] = {"low": 1.0, "mid": 2.0, "high": 3.0}


def iter_social_cells() -> List[Tuple[str, str]]:
    return [(b, c) for b in SOCIAL_B_BUCKETS for c in SOCIAL_C_BUCKETS]


def _social_bucket_bounds(
    f_range: Tuple[float, float],
    bucket: str,
) -> Tuple[float, float]:
    """Scene-scaled numeric bounds for b and c bucket sampling."""
    lo, hi = f_range
    span = hi - lo
    if bucket == "low":
        return lo, lo + span / 3.0
    if bucket == "mid":
        return lo + span / 3.0, lo + 2.0 * span / 3.0
    if bucket == "high":
        return lo + 2.0 * span / 3.0, hi
    raise ValueError(f"Unknown social bucket `{bucket}`")


def _nominal_social_ratio(cell: Tuple[str, str]) -> float:
    b_bucket, c_bucket = cell
    return _SOCIAL_BUCKET_SCORE[b_bucket] / _SOCIAL_BUCKET_SCORE[c_bucket]


def _pick_social_candidate_cells(
    rng: random.Random,
    protagonist_cell: Tuple[str, str],
    match_distance_bucket: str,
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """Pick (gold_cell, loser_cell) from the 3x3 bucket grid."""
    cells = iter_social_cells()
    target = _nominal_social_ratio(protagonist_cell)
    ranked = sorted(
        cells,
        key=lambda cell: (abs(_nominal_social_ratio(cell) - target), rng.random()),
    )
    if match_distance_bucket == "close":
        gold_pool = ranked[:2]
        loser_pool = ranked[4:]
    elif match_distance_bucket == "mid":
        gold_pool = ranked[:3]
        loser_pool = ranked[3:6]
    elif match_distance_bucket == "far":
        gold_pool = ranked[:3]
        loser_pool = ranked[6:]
    else:
        raise ValueError(f"Unknown match_distance_bucket `{match_distance_bucket}`.")
    gold_cell = rng.choice(gold_pool)
    loser_cell = rng.choice(loser_pool or ranked[-2:])
    return gold_cell, loser_cell


def sample_latent_social_ood(
    rng: random.Random,
    f_range: Tuple[float, float],
    match_distance_bucket: str,
    protagonist_cell: Optional[Tuple[str, str]] = None,
    forced_gold_index: Optional[int] = None,
) -> LatentSocialOODRecord:
    """Sample one d4 OOD item.

    `protagonist_cell` is the visible 3x3 (b bucket, c bucket) class to balance.
    Candidate cells are selected so the gold candidate has a closer nominal b/c
    ratio than the distractor; exact values are then sampled within buckets.
    """
    if protagonist_cell is None:
        protagonist_cell = rng.choice(iter_social_cells())
    b_bucket_i, c_bucket_i = protagonist_cell
    b_i = _open_uniform(rng, *_social_bucket_bounds(f_range, b_bucket_i))
    c_i = _open_uniform(rng, *_social_bucket_bounds(f_range, c_bucket_i))
    bc_i = b_i / c_i
    x_i = bc_i

    gold_cell, loser_cell = _pick_social_candidate_cells(rng, protagonist_cell, match_distance_bucket)
    b_gold = _open_uniform(rng, *_social_bucket_bounds(f_range, gold_cell[0]))
    c_gold = _open_uniform(rng, *_social_bucket_bounds(f_range, gold_cell[1]))
    b_loser = _open_uniform(rng, *_social_bucket_bounds(f_range, loser_cell[0]))
    c_loser = _open_uniform(rng, *_social_bucket_bounds(f_range, loser_cell[1]))

    gold_index = forced_gold_index if forced_gold_index in (0, 1) else rng.choice([0, 1])
    bs_list = [0.0, 0.0]
    cs_list = [0.0, 0.0]
    b_buckets = ["", ""]
    c_buckets = ["", ""]
    bs_list[gold_index] = b_gold
    cs_list[gold_index] = c_gold
    b_buckets[gold_index], c_buckets[gold_index] = gold_cell
    bs_list[1 - gold_index] = b_loser
    cs_list[1 - gold_index] = c_loser
    b_buckets[1 - gold_index], c_buckets[1 - gold_index] = loser_cell

    ratios_list = [b / c for b, c in zip(bs_list, cs_list)]
    if abs(ratios_list[gold_index] - bc_i) > abs(ratios_list[1 - gold_index] - bc_i):
        # Within-bucket numeric draws can occasionally invert the nominal
        # ordering. Keep the requested gold slot stable by swapping contents.
        other = 1 - gold_index
        bs_list[gold_index], bs_list[other] = bs_list[other], bs_list[gold_index]
        cs_list[gold_index], cs_list[other] = cs_list[other], cs_list[gold_index]
        b_buckets[gold_index], b_buckets[other] = b_buckets[other], b_buckets[gold_index]
        c_buckets[gold_index], c_buckets[other] = c_buckets[other], c_buckets[gold_index]

    bs = tuple(bs_list)
    c_candidates = tuple(cs_list)
    ratios = tuple(b / c for b, c in zip(bs, c_candidates))
    xs = tuple(ratios)
    return LatentSocialOODRecord(
        domain_key=OOD_DOMAIN_SOCIAL,
        c_i=c_i,
        b_i=b_i,
        x_i=x_i,
        c_candidates=c_candidates,
        b_candidates=(bs[0], bs[1]),
        x_candidates=(xs[0], xs[1]),
        bc_ratio_i=bc_i,
        bc_ratio_candidates=(ratios[0], ratios[1]),
        b_bucket_i=b_bucket_i,
        c_bucket_i=c_bucket_i,
        b_bucket_candidates=(b_buckets[0], b_buckets[1]),
        c_bucket_candidates=(c_buckets[0], c_buckets[1]),
        gold_candidate_index=gold_index,
        match_distance_bucket=match_distance_bucket,
    )


# -----------------------------------------------------------------------------
# 9. OOD: career sorting (d5 -- Langtry Eq. 4 / Prop. 6)
# -----------------------------------------------------------------------------
# Each d5 item is a binary firm choice: accept high-paying firm H (where the
# protagonist ranks near the bottom of a high-pay crowd) vs low-paying firm L
# (where they rank above the firm average). Gold follows Proposition 6:
#
#     choose H iff (x_S_H - x_S_L) >= alpha_2i * (x_bar_H - x_bar_L)
#
# The three subgroup tags are the alpha_2i bucket (low / mid / high), matching
# Langtry's intuition: low alpha_2i => "big fish in big pond wins", high
# alpha_2i => "small pond with dominance wins".

ALPHA_2_BUCKETS: Dict[str, BucketSpec] = {
    "low":  BucketSpec(label="low",  lower=0.05, upper=0.20),
    "mid":  BucketSpec(label="mid",  lower=0.40, upper=0.60),
    "high": BucketSpec(label="high", lower=0.80, upper=0.95),
}


@dataclass
class LatentCareerOODRecord:
    """d5 OOD draw: binary firm choice under coworker comparison."""
    domain_key: str                        # OOD_DOMAIN_CAREER
    alpha_2i: float                        # coworker-comparison intensity
    alpha_2i_bucket: str
    x_bar_H: float                         # high-pay firm's average wage
    x_bar_L: float                         # low-pay firm's average wage
    x_S_H: float                           # protagonist's wage at H (< x_bar_H)
    x_S_L: float                           # protagonist's wage at L (> x_bar_L)
    lhs: float                             # x_S_H - x_S_L
    rhs: float                             # alpha_2i * (x_bar_H - x_bar_L)
    gold_firm: str                         # "H" or "L"


def sample_latent_career_ood(
    rng: random.Random,
    wage_anchor: float,
    alpha_2i_bucket: str,
    target_gold_firm: Optional[str] = None,
    max_retries: int = 400,
) -> LatentCareerOODRecord:
    """Sample one d5 OOD item.

    wage_anchor is roughly the median "big-pond crowd" wage (e.g. $12k/mo for
    management consulting). The caller (scene_pool.py) will pick wage_anchor
    based on the scene vertical.

    If `target_gold_firm` in {"H", "L"} is provided, the sampler resamples
    wages (and if necessary alpha_2i within the same bucket) until the
    Langtry decision rule lands on the target side, so that downstream
    `gold_firm` distribution can be enforced H/L-balanced. If after
    `max_retries` attempts the target cannot be hit within the bucket, the
    last draw is returned (logged via `gold_firm`), letting the caller
    fall back to the natural distribution for that scene/bucket.
    """
    alpha_bkt = ALPHA_2_BUCKETS[alpha_2i_bucket]

    def _one_attempt(alpha_override: Optional[float] = None) -> LatentCareerOODRecord:
        alpha_2i = (
            alpha_override
            if alpha_override is not None
            else _sample_bucket_value(rng, ALPHA_2_BUCKETS, alpha_2i_bucket)
        )

        # High-pay firm: crowd is roughly wage_anchor; protagonist sits below
        # the crowd mean (bottom of a high-pay cohort). Range widened slightly
        # from the original (0.58, 0.82) to (0.55, 0.95) so that a large
        # absolute-pay premium is attainable; a high-alpha protagonist can
        # still prefer H when the pay gap is unusually large (Prop. 6 edge).
        x_bar_H = wage_anchor * _open_uniform(rng, 1.00, 1.30)
        x_S_H = x_bar_H * _open_uniform(rng, 0.55, 0.95)

        # Low-pay firm: crowd is meaningfully below the high-pay crowd;
        # protagonist is above the crowd mean (big fish). x_bar_L lower bound
        # widened to 0.35 to allow feasibility of gold=H under high alpha.
        x_bar_L = x_bar_H * _open_uniform(rng, 0.35, 0.65)
        x_S_L = x_bar_L * _open_uniform(rng, 1.05, 1.30)

        lhs = x_S_H - x_S_L
        rhs = alpha_2i * (x_bar_H - x_bar_L)
        gold_firm = "H" if lhs >= rhs else "L"

        return LatentCareerOODRecord(
            domain_key=OOD_DOMAIN_CAREER,
            alpha_2i=alpha_2i,
            alpha_2i_bucket=alpha_2i_bucket,
            x_bar_H=x_bar_H,
            x_bar_L=x_bar_L,
            x_S_H=x_S_H,
            x_S_L=x_S_L,
            lhs=lhs,
            rhs=rhs,
            gold_firm=gold_firm,
        )

    # Unconstrained path.
    if target_gold_firm not in ("H", "L"):
        return _one_attempt()

    # Constrained path: try a direct draw first.
    last = _one_attempt()
    if last.gold_firm == target_gold_firm:
        return last

    # Strategy 1: retry wage draws at fixed alpha bucket.
    for _ in range(max_retries):
        cand = _one_attempt()
        if cand.gold_firm == target_gold_firm:
            return cand
        last = cand

    # Strategy 2: solve for an alpha_2i in-bucket that flips the decision.
    #     gold = H iff alpha_2i <= lhs / (x_bar_H - x_bar_L) = alpha_crit
    # Draw one fresh wage configuration; pick alpha_2i at the bucket boundary
    # that matches the target, if feasible.
    for _ in range(max_retries):
        base = _one_attempt()
        denom = base.x_bar_H - base.x_bar_L
        if denom <= 0:
            continue
        alpha_crit = base.lhs / denom
        if target_gold_firm == "H":
            alpha_lo = alpha_bkt.lower
            alpha_hi = min(alpha_crit, alpha_bkt.upper)
        else:  # "L"
            alpha_lo = max(alpha_crit, alpha_bkt.lower)
            alpha_hi = alpha_bkt.upper
        if alpha_hi > alpha_lo:
            forced_alpha = _open_uniform(rng, alpha_lo, alpha_hi)
            candidate = LatentCareerOODRecord(
                domain_key=OOD_DOMAIN_CAREER,
                alpha_2i=forced_alpha,
                alpha_2i_bucket=alpha_2i_bucket,
                x_bar_H=base.x_bar_H,
                x_bar_L=base.x_bar_L,
                x_S_H=base.x_S_H,
                x_S_L=base.x_S_L,
                lhs=base.lhs,
                rhs=forced_alpha * denom,
                gold_firm="H" if base.lhs >= forced_alpha * denom else "L",
            )
            if candidate.gold_firm == target_gold_firm:
                return candidate
        last = base

    # Fall back: return the last draw and let the caller keep the imbalance
    # rather than spinning. The summary will log this as a skipped target.
    return last


# -----------------------------------------------------------------------------
# 10. Public summary / description used by data_schema.md generator
# -----------------------------------------------------------------------------

def describe_schema() -> Dict[str, object]:
    return {
        "version": LANGTRY_SCHEMA_VERSION,
        "paper_reference": "Langtry (2024), Keeping up with 'The Joneses'",
        "main_closed_form": "x_i* = F + alpha_i * Sum_j g_ij * x_j",
        "placebo_closed_form": "x_i* = F  (alpha_i ~ 0)",
        "d4_closed_form": "x_i* = b_i / c  (pairwise stable iff b_i/c == b_j/c)",
        "d5_decision_rule": "choose H iff (x_S_H - x_S_L) >= alpha_2i * (x_bar_H - x_bar_L)",
        "alpha_buckets": {k: (v.lower, v.upper) for k, v in ALPHA_BUCKETS.items()},
        "alpha_placebo_bucket": (ALPHA_PLACEBO_BUCKET.lower, ALPHA_PLACEBO_BUCKET.upper),
        "dispersion_buckets": {k: (v.g_max_lower, v.g_max_upper) for k, v in DISPERSION_BUCKETS.items()},
        "skew_buckets": {k: (v.mean_ratio_lower, v.mean_ratio_upper) for k, v in SKEW_BUCKETS.items()},
        "peer_count_choices": list(PEER_COUNT_CHOICES),
        "total_cells": len(iter_all_cells()),
        "held_out_cells": [list(c) for c in HELD_OUT_CELLS],
        "main_cells_count": len(iter_main_cells()),
    }
