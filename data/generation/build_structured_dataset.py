"""
Unified structured-dataset builder.

Emits, per `--split` flag, one JSON file under `--output-dir` that contains
theory-aligned records. The teacher LLM step (write_scenarios_with_llm.py)
consumes these records to fill in `scenario_text`.

Supported splits
----------------
    train            d1+d3 positional training records (+d2 placebo)
    eval_A           d1+d3 MCQ on test scenes (held-in + held-out cells)
    eval_B           d1+d3 comparative-statics pairwise records
    placebo_test     d2 MCQ on test scenes (gold = D_pure_private)
    ood_social       d4 2-candidate MCQ (Langtry friendship Eq. 1)
    ood_career       d5 binary firm-choice MCQ (Langtry Eq. 4)
    all              run every split in sequence

Record schema (common top-level keys)
------------------------------------
    record_id, dataset_split, schema_version, generator, seed,
    scene { scene_id, domain_key, family, title, summary, action_label,
            action_unit, value_format, peer_noun_singular, peer_noun_plural,
            split },
    latent { alpha_i, F, c, N_peers, x_j, g_ij, ref_sum, x_i_star,
             alpha_bucket, dispersion_bucket, skew_bucket, cell_id,
             is_held_out_cell, is_placebo, peer_rank_by_g, peer_rank_by_x },
    peer_cards [ { id, action_value, action_display, g_ij, closeness_hint } ],
    gold { rule_id, x_value, x_display },
    scenario_text  (null — populated later by the teacher LLM),
    split-specific: mcq | pair | ood_social | ood_career
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config.langtry_parameters import (  # noqa: E402
    ALPHA_BUCKETS,
    ALPHA_2_BUCKETS,
    BUCKET_ORDER,
    HELD_OUT_CELLS,
    LANGTRY_SCHEMA_VERSION,
    MAIN_POSITIONAL_DOMAINS,
    OOD_DOMAIN_CAREER,
    OOD_DOMAIN_SOCIAL,
    PEER_COUNT_CHOICES,
    PLACEBO_DOMAIN,
    SOCIAL_MATCH_DISTANCE_BUCKETS,
    build_cell_id,
    iter_all_cells,
    iter_main_cells,
    is_held_out_cell,
    sample_latent_career_ood,
    sample_latent_langtry,
    sample_latent_social_ood,
    iter_social_cells,
)
from data.config.rule_label_templates import (  # noqa: E402
    MCQOption,
    RULE_IDS,
    RULE_SPECS,
    RuleInputs,
    build_mcq_options,
    compute_rule_x_values,
    format_value,
    gold_rule_for_domain,
)
from data.config.scene_pool import (  # noqa: E402
    Scene,
    list_scenes,
)
from data.generation.identification_filter import (  # noqa: E402
    TAU_PAIR_PLACEBO,
    TAU_PAIR_POSITIONAL,
    TAU_PAIRWISE,
    TAU_PLACEBO,
    TAU_POSITIONAL,
    NotEnoughDistractorsError,
    filter_distractors,
    filter_distractors_with_hard_fallback,
    filter_distractors_with_shortfall_relaxation,
    select_stratified_distractors,
)
from data.config.record_access import ensure_oracle  # noqa: E402


IDENT_RESAMPLE_MAX = 300  # max resamples within the same (scene, cell).
EVAL_A_GLOBAL_RESTART_MAX = 20  # restarts for hard distractor-balance search.
EVAL_A_MARGIN_HARD_MAX = 0.20
EVAL_A_MARGIN_MEDIUM_MAX = 0.50

# Tolerance (absolute L1 counts) for the joint balance optimisation before we
# raise. We only balance distractor-rule counts — difficulty tier is recorded
# on each record for downstream subgroup analysis but NOT constrained here,
# because tier availability depends on scene/cell geometry and forcing 1:1:1
# rarely has a feasible solution that is also rule-balanced.
EVAL_A_JOINT_TOLERANCE = 0
EVAL_A_CANDIDATES_PER_SLOT = 36
EVAL_A_SLOT_ATTEMPT_MULT = 16
EVAL_A_JOINT_MAX_STEPS = 600_000
EVAL_A_JOINT_STALL_PATIENCE = 10_000

PLACEBO_CANDIDATES_PER_SLOT = 36
PLACEBO_SLOT_ATTEMPT_MULT = 16
PLACEBO_JOINT_TOLERANCE = 0
PLACEBO_JOINT_MAX_STEPS = 600_000
PLACEBO_JOINT_STALL_PATIENCE = 10_000


SCHEMA_VERSION = "relstate_v3"  # v3 — physical isolation via rec["oracle"].
GENERATOR_NAME = "build_structured_dataset.py"


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split",
        required=True,
        choices=[
            "train", "eval_A", "eval_B",
            "placebo_test", "ood_social", "ood_career",
            "all",
        ],
        help="Which structured split to build.",
    )
    p.add_argument("--output-dir", default="data/structured", help="Directory to write <split>.json into.")
    p.add_argument("--seed", type=int, default=20260421)

    # Volumes (per-split knobs).
    # Default values are tuned so that each split hits the documented target
    # volume and each distribution axis is exactly balanced under round-robin
    # cycling. Change only if you know the divisibility implications.
    p.add_argument("--train-samples-per-scene", type=int, default=105,
                   help="d1+d3 positional training: samples per training scene. "
                        "Default 105 = 5 reps × 21 main cells (perfect cell balance). "
                        "21 positional scenes × 105 = 2205 records.")
    p.add_argument("--train-placebo-samples-per-scene", type=int, default=160,
                   help="d2 placebo training: samples per placebo training scene. "
                        "5 placebo scenes × 160 = 800 records (d2 ~26%% of total train). "
                        "Positional 2205 + placebo 800 = 3005 total train.")
    p.add_argument("--eval-a-samples-per-cell-per-scene", type=int, default=4,
                   help="Eval-A: samples per (test_scene, cell). "
                        "14 test scenes × 27 cells × 4 = 1512 records.")
    p.add_argument("--eval-b-pairs-per-scene", type=int, default=20,
                   help="Eval-B: pairwise comparative-statics pairs per test scene. "
                        "20 = 5 perturbations × 4 pairs; 14 scenes × 20 × 2 = 560 records.")
    p.add_argument("--placebo-test-samples-per-scene", type=int, default=84,
                   help="Placebo Eval: MCQ samples per d2 test scene. "
                        "84 = 4 × 21 (letter-balanced); 6 scenes × 84 = 504 records.")
    p.add_argument("--ood-social-samples-per-scene", type=int, default=72,
                   help="Eval-D: MCQ samples per d4 scene. "
                        "72 = 12 combos (3 match-distance × 2 gold_cand × 2 gold_letter) × 6. "
                        "8 scenes × 72 = 576 records.")
    p.add_argument("--ood-career-samples-per-scene", type=int, default=72,
                   help="Eval-E: MCQ samples per d5 scene. "
                        "72 = 12 combos (3 alpha_2i × 2 gold_firm × 2 gold_letter) × 6. "
                        "8 scenes × 72 = 576 records.")
    return p.parse_args()


# =============================================================================
# Helpers shared across splits
# =============================================================================

def _serialize_dataclass(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def _scene_summary(scene: Scene) -> Dict[str, Any]:
    return {
        "scene_id": scene.scene_id,
        "domain_key": scene.domain_key,
        "family": scene.family,
        "title": scene.title,
        "summary": scene.summary,
        "action_label": scene.action_label,
        "action_unit": scene.action_unit,
        "value_format": scene.value_format,
        "peer_noun_singular": scene.peer_noun_singular,
        "peer_noun_plural": scene.peer_noun_plural,
        "split": scene.split,
        "extras": dict(scene.extras),
    }


def _closeness_hint(g_weight: float) -> str:
    if g_weight >= 0.45:
        return "tight"
    if g_weight >= 0.18:
        return "mid"
    return "peripheral"


def _build_peer_cards(
    latent,
    scene: Scene,
) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for idx, (x_val, g_val) in enumerate(zip(latent.x_j, latent.g_ij)):
        cards.append({
            "id": f"P{idx + 1}",
            "action_value": round(x_val, 4),
            "action_display": format_value(x_val, scene.value_format, scene.action_unit),
            "g_ij": round(g_val, 4),
            "closeness_hint": _closeness_hint(g_val),
        })
    return cards


def _latent_to_dict(latent) -> Dict[str, Any]:
    d = asdict(latent)
    # Round the floaty parts to avoid noisy JSON diffs while keeping enough
    # precision for downstream computation (the raw x_j / g_ij are still the
    # authoritative values for CoT / rule replay).
    d["alpha_i"] = round(d["alpha_i"], 6)
    d["F"] = round(d["F"], 6)
    d["c"] = round(d["c"], 6)
    d["ref_sum"] = round(d["ref_sum"], 6)
    d["x_i_star"] = round(d["x_i_star"], 6)
    d["x_j"] = [round(v, 6) for v in d["x_j"]]
    d["g_ij"] = [round(v, 6) for v in d["g_ij"]]
    return d


def _pick_cell(rng: random.Random, include_held_out: bool) -> Tuple[str, str, str]:
    if include_held_out:
        return rng.choice(iter_all_cells())
    return rng.choice(iter_main_cells())


def _sample_one_positional(
    rng: random.Random,
    scene: Scene,
    cell: Tuple[str, str, str],
) -> Any:
    F_val = rng.uniform(*scene.F_range)
    alpha_b, disp_b, skew_b = cell
    return sample_latent_langtry(
        rng=rng,
        domain_key=scene.domain_key,
        F=F_val,
        alpha_bucket=alpha_b,
        dispersion_bucket=disp_b,
        skew_bucket=skew_b,
    )


def _sample_one_placebo(rng: random.Random, scene: Scene) -> Any:
    F_val = rng.uniform(*scene.F_range)
    # Placebo uses arbitrary dispersion/skew — not identified; we pick the
    # "mid" combination so the narrative peer cards still look natural.
    latent = sample_latent_langtry(
        rng=rng,
        domain_key=PLACEBO_DOMAIN,
        F=F_val,
        alpha_bucket="mid",           # ignored (placebo overrides alpha)
        dispersion_bucket="mid",
        skew_bucket="mid",
    )
    # Overwrite bucket tags so placebo records don't pollute the
    # (alpha x disp x skew) cell statistics. Any alpha_bucket value is
    # meaningless for placebo (alpha ~ 0); mark the cell as "placebo".
    latent.alpha_bucket = "placebo"
    latent.cell_id = "placebo"
    latent.is_held_out_cell = False
    return latent


def _rule_inputs_from_latent(latent) -> RuleInputs:
    closest = latent.peer_rank_by_g[0]
    return RuleInputs(
        F=latent.F,
        alpha_i=latent.alpha_i,
        x_j=latent.x_j,
        g_ij=latent.g_ij,
        closest_peer_index=closest,
    )


def _rule_inputs_with_alpha(latent, alpha_override: float) -> RuleInputs:
    closest = latent.peer_rank_by_g[0]
    return RuleInputs(
        F=latent.F,
        alpha_i=alpha_override,
        x_j=latent.x_j,
        g_ij=latent.g_ij,
        closest_peer_index=closest,
    )


def _record_base(
    record_id: str,
    dataset_split: str,
    scene: Scene,
    latent,
    gold_rule_id: str,
    gold_x: float,
    seed: int,
) -> Dict[str, Any]:
    """Build the common envelope for a structured record.

    Schema v3 — fields the student may see are at the top level; fields the
    student must never see live under `record["oracle"]`. See
    data/config/record_access.py for the contract.
    """
    rec: Dict[str, Any] = {
        # --- public (student-prompt-safe) ---
        "record_id": record_id,
        "dataset_split": dataset_split,
        "schema_version": SCHEMA_VERSION,
        "langtry_schema_version": LANGTRY_SCHEMA_VERSION,
        "generator": GENERATOR_NAME,
        "seed": seed,
        "scene": _scene_summary(scene),
        "scenario_text": None,
        # --- oracle (teacher + scorer + metrics only) ---
        "oracle": {
            "latent": _latent_to_dict(latent),
            "peer_cards": _build_peer_cards(latent, scene),
            "gold": {
                "rule_id": gold_rule_id,
                "x_value": round(gold_x, 6),
                "x_display": format_value(gold_x, scene.value_format, scene.action_unit),
            },
        },
    }
    return rec


# =============================================================================
# Split 1 — TRAIN (d1+d3 positional, d2 placebo)
# =============================================================================

def build_train_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rng = random.Random(args.seed + 1000)
    records: List[Dict[str, Any]] = []
    main_cells = iter_main_cells()

    # --- d1 + d3 positional training ---
    positional_scenes = [
        s for s in list_scenes(split="train")
        if s.domain_key in MAIN_POSITIONAL_DOMAINS
    ]
    for scene in positional_scenes:
        samples = args.train_samples_per_scene
        cells_cycle = list(main_cells)
        rng.shuffle(cells_cycle)
        for i in range(samples):
            cell = cells_cycle[i % len(cells_cycle)]
            latent = _sample_one_positional(rng, scene, cell)
            gold_rule = gold_rule_for_domain(scene.domain_key)
            gold_x = latent.x_i_star
            rec = _record_base(
                record_id=f"train__{scene.scene_id}__{latent.cell_id}__{i:04d}",
                dataset_split="train",
                scene=scene,
                latent=latent,
                gold_rule_id=gold_rule,
                gold_x=gold_x,
                seed=args.seed,
            )
            rec["train"] = {
                "is_positional": True,
                "is_placebo": False,
            }
            records.append(rec)

    # --- d2 placebo training ---
    placebo_scenes = [s for s in list_scenes(split="train") if s.domain_key == PLACEBO_DOMAIN]
    for scene in placebo_scenes:
        for i in range(args.train_placebo_samples_per_scene):
            latent = _sample_one_placebo(rng, scene)
            gold_rule = "D_pure_private"
            gold_x = latent.F
            rec = _record_base(
                record_id=f"train_placebo__{scene.scene_id}__{i:04d}",
                dataset_split="train",
                scene=scene,
                latent=latent,
                gold_rule_id=gold_rule,
                gold_x=gold_x,
                seed=args.seed,
            )
            rec["train"] = {
                "is_positional": False,
                "is_placebo": True,
            }
            records.append(rec)

    rng.shuffle(records)
    return records


# =============================================================================
# Split 2 — EVAL-A (d1+d3 MCQ on test scenes)
# =============================================================================

def _margin_vs_gold(rule_x_values: Dict[str, float], gold_rule_id: str, rid: str, F: float) -> float:
    return abs(rule_x_values[rid] - rule_x_values[gold_rule_id]) / max(F, 1e-9)


def _eval_a_margin_tier(min_margin: float) -> str:
    if min_margin < EVAL_A_MARGIN_HARD_MAX:
        return "hard"
    if min_margin < EVAL_A_MARGIN_MEDIUM_MAX:
        return "medium"
    return "easy"


def _stratified_feasible_triples(
    candidate_ids: Sequence[str],
    rule_x_values: Dict[str, float],
    gold_rule_id: str,
    F: float,
    tau_pair: float,
    classify_tier: bool = True,
) -> List[Tuple[Tuple[str, str, str], float, Optional[str], float]]:
    """Enumerate all 3-subsets of `candidate_ids` that (a) mix near/far by
    margin-vs-gold and (b) have min pair-wise margin >= tau_pair.

    Returns list of (triple, pair_min, tier_or_None, min_margin_vs_gold).
    """
    ranked = sorted(candidate_ids, key=lambda rid: _margin_vs_gold(rule_x_values, gold_rule_id, rid, F))
    split = max(1, len(ranked) // 2)
    near, far = set(ranked[:split]), set(ranked[split:])
    out: List[Tuple[Tuple[str, str, str], float, Optional[str], float]] = []
    for tr in combinations(ranked, 3):
        if not (any(r in near for r in tr) and any(r in far for r in tr)):
            continue
        pair_min = min(
            abs(rule_x_values[a] - rule_x_values[b]) / max(F, 1e-9)
            for a, b in combinations(tr, 2)
        )
        if pair_min < tau_pair:
            continue
        min_margin = min(_margin_vs_gold(rule_x_values, gold_rule_id, rid, F) for rid in tr)
        tier = _eval_a_margin_tier(min_margin) if classify_tier else None
        out.append((tuple(tr), pair_min, tier, min_margin))
    return out


def _joint_assign_local_search(
    *,
    rng: random.Random,
    candidates_by_slot: List[List[Dict[str, Any]]],
    non_gold_rules: Sequence[str],
    rule_targets: Dict[str, int],
    tier_targets: Optional[Dict[str, int]] = None,
    max_steps: int,
    stall_patience: int,
    tolerance: int = 0,
    label: str = "joint",
) -> Tuple[List[int], Dict[str, Any]]:
    """Joint local-search that balances rule-count and (optional) tier-count.

    - Initializes each slot's pick with tier round-robin priority (if tiers
      are in scope) then greedy deficit-first refinement on rule balance.
    - Runs a random 1-flip local search minimising
          L1 = sum_r |rule_count[r] - rule_targets[r]|
             + sum_t |tier_count[t] - tier_targets[t]|
      Strictly accepts any move with non-positive delta (0 moves are
      useful to hop across plateaus).
    - Returns the best-seen assignment and a stats dict.
    """
    n_slots = len(candidates_by_slot)
    have_tiers = tier_targets is not None
    tier_names: Tuple[str, ...] = ("hard", "medium", "easy") if have_tiers else tuple()

    # Greedy initialisation.
    if have_tiers:
        # Round-robin tier priority, then pick a candidate of that tier if any.
        tier_order = list(tier_names) * (n_slots // len(tier_names) + 1)
        tier_order = tier_order[:n_slots]
        rng.shuffle(tier_order)
        assignment: List[int] = []
        for si, slot_cands in enumerate(candidates_by_slot):
            preferred = [i for i, c in enumerate(slot_cands) if c["tier"] == tier_order[si]]
            if not preferred:
                preferred = list(range(len(slot_cands)))
            assignment.append(preferred[rng.randrange(len(preferred))])
    else:
        assignment = [rng.randrange(len(slot_cands)) for slot_cands in candidates_by_slot]

    rule_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    for si, ci in enumerate(assignment):
        cand = candidates_by_slot[si][ci]
        for r in cand["picks"]:
            rule_counts[r] += 1
        if have_tiers:
            tier_counts[cand["tier"]] += 1

    def _rule_err() -> int:
        return sum(abs(rule_counts[r] - rule_targets[r]) for r in non_gold_rules)

    def _tier_err() -> int:
        if not have_tiers:
            return 0
        return sum(abs(tier_counts[t] - tier_targets[t]) for t in tier_names)

    err = _rule_err() + _tier_err()
    best_err = err
    best_assignment = list(assignment)

    stall = 0
    step = 0
    while step < max_steps and err > tolerance:
        si = rng.randrange(n_slots)
        slot_cands = candidates_by_slot[si]
        if len(slot_cands) < 2:
            step += 1
            stall += 1
            continue
        curr_i = assignment[si]
        alt_i = rng.randrange(len(slot_cands))
        if alt_i == curr_i:
            step += 1
            stall += 1
            if stall > stall_patience:
                # Aggressive perturbation: randomise a random slot.
                rs = rng.randrange(n_slots)
                rc = candidates_by_slot[rs][assignment[rs]]
                nc_i = rng.randrange(len(candidates_by_slot[rs]))
                if nc_i != assignment[rs]:
                    nc = candidates_by_slot[rs][nc_i]
                    for r in rc["picks"]:
                        rule_counts[r] -= 1
                    for r in nc["picks"]:
                        rule_counts[r] += 1
                    if have_tiers:
                        tier_counts[rc["tier"]] -= 1
                        tier_counts[nc["tier"]] += 1
                    assignment[rs] = nc_i
                    err = _rule_err() + _tier_err()
                stall = 0
            continue

        curr = slot_cands[curr_i]
        alt = slot_cands[alt_i]
        curr_set = curr["picks"]
        alt_set = alt["picks"]

        # Delta on rules.
        impacted_rules = set(curr_set) | set(alt_set)
        delta_rule = 0
        for r in impacted_rules:
            old = rule_counts[r]
            new = old - (1 if r in curr_set else 0) + (1 if r in alt_set else 0)
            delta_rule += abs(new - rule_targets[r]) - abs(old - rule_targets[r])

        delta_tier = 0
        if have_tiers and curr["tier"] != alt["tier"]:
            ct = curr["tier"]; nt = alt["tier"]
            delta_tier += abs((tier_counts[ct] - 1) - tier_targets[ct]) - abs(tier_counts[ct] - tier_targets[ct])
            delta_tier += abs((tier_counts[nt] + 1) - tier_targets[nt]) - abs(tier_counts[nt] - tier_targets[nt])

        delta = delta_rule + delta_tier
        # Accept strictly improving; also accept zero-delta occasionally to
        # walk plateaus. Reject uphill.
        accept = delta < 0 or (delta == 0 and rng.random() < 0.35)
        if accept:
            for r in curr_set:
                rule_counts[r] -= 1
            for r in alt_set:
                rule_counts[r] += 1
            if have_tiers and curr["tier"] != alt["tier"]:
                tier_counts[curr["tier"]] -= 1
                tier_counts[alt["tier"]] += 1
            assignment[si] = alt_i
            err += delta
            if err < best_err:
                best_err = err
                best_assignment = list(assignment)
            stall = 0 if delta < 0 else stall + 1
        else:
            stall += 1

        step += 1

    stats = {
        "label": label,
        "total_L1_final": err,
        "total_L1_best": best_err,
        "rule_L1": _rule_err(),
        "tier_L1": _tier_err() if have_tiers else None,
        "steps": step,
        "rule_counts": dict(rule_counts),
        "tier_counts": dict(tier_counts) if have_tiers else None,
    }
    # Re-materialise counts from the returned best_assignment so callers see a
    # consistent view even if late plateau-walk moves drifted upwards.
    if best_err < err:
        rule_counts = Counter()
        tier_counts = Counter()
        for si, ci in enumerate(best_assignment):
            cand = candidates_by_slot[si][ci]
            for r in cand["picks"]:
                rule_counts[r] += 1
            if have_tiers:
                tier_counts[cand["tier"]] += 1
        stats["rule_counts"] = dict(rule_counts)
        if have_tiers:
            stats["tier_counts"] = dict(tier_counts)
        stats["total_L1_final"] = best_err
        stats["rule_L1"] = sum(abs(rule_counts[r] - rule_targets[r]) for r in non_gold_rules)
        stats["tier_L1"] = (
            sum(abs(tier_counts[t] - tier_targets[t]) for t in tier_names) if have_tiers else None
        )
    return best_assignment, stats


def build_eval_a_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rng = random.Random(args.seed + 2000)
    records: List[Dict[str, Any]] = []
    test_scenes = [s for s in list_scenes(split="test") if s.domain_key in MAIN_POSITIONAL_DOMAINS]
    all_cells = iter_all_cells()  # includes held-out
    letter_cycle = ("A", "B", "C", "D")
    non_gold_rules = [rid for rid in RULE_IDS if rid != "A_peer_weighted"]
    slots: List[Tuple[Scene, Tuple[str, str, str], int]] = []
    for scene in test_scenes:
        for cell in all_cells:
            for rep in range(args.eval_a_samples_per_cell_per_scene):
                slots.append((scene, cell, rep))

    total_distractor_slots = len(slots) * 3
    if total_distractor_slots % len(non_gold_rules) != 0:
        raise ValueError(
            "Cannot evenly split distractor slots across non-gold rules: "
            f"{total_distractor_slots} slots over {len(non_gold_rules)} rules."
        )
    target_each = total_distractor_slots // len(non_gold_rules)
    rule_targets = {rid: target_each for rid in non_gold_rules}

    ident_stats = {
        "clean": 0,
        "resample_retries": 0,
        "pairwise_resample_retries": 0,
        "degraded": 0,
    }

    # ----- Build a diverse candidate pool per slot. -----
    # For each slot we sample fresh latents until we gather up to
    # EVAL_A_CANDIDATES_PER_SLOT distinct stratified+pair-wise-feasible
    # triples. We do NOT require every slot to cover all 3 tiers — some
    # (scene, cell) geometries naturally cannot support easy tier. The joint
    # local search compensates across slots.
    candidates_by_slot: List[List[Dict[str, Any]]] = []
    candidate_per_slot = EVAL_A_CANDIDATES_PER_SLOT
    slot_attempt_max = IDENT_RESAMPLE_MAX * EVAL_A_SLOT_ATTEMPT_MULT
    print(json.dumps({"status": "eval_A_pool_start", "n_slots": len(slots),
                      "candidate_per_slot": candidate_per_slot,
                      "slot_attempt_max": slot_attempt_max}, ensure_ascii=False),
          flush=True)
    for slot_idx, (scene, cell, rep) in enumerate(slots):
        gold_rule = gold_rule_for_domain(scene.domain_key)
        slot_candidates: List[Dict[str, Any]] = []
        tier_seen: Counter[str] = Counter()
        seen_keys = set()
        latent_attempts = 0
        for _ in range(slot_attempt_max):
            if len(slot_candidates) >= candidate_per_slot:
                break
            latent_attempts += 1
            cand_latent = _sample_one_positional(rng, scene, cell)
            cand_rule_x = compute_rule_x_values(_rule_inputs_from_latent(cand_latent))
            cand_ids = filter_distractors(cand_rule_x, gold_rule, cand_latent.F, TAU_POSITIONAL)
            if len(cand_ids) < 3:
                ident_stats["resample_retries"] += 1
                continue
            triples = _stratified_feasible_triples(
                cand_ids, cand_rule_x, gold_rule, cand_latent.F,
                tau_pair=TAU_PAIR_POSITIONAL, classify_tier=True,
            )
            if not triples:
                ident_stats["pairwise_resample_retries"] += 1
                continue
            rng.shuffle(triples)
            for tr, pair_min, tier, min_margin in triples[:6]:
                key = tuple(sorted(tr))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                slot_candidates.append({
                    "latent": cand_latent,
                    "rule_x": cand_rule_x,
                    "picks": list(tr),
                    "pair_tau_used": pair_min,
                    "tier": tier,
                    "min_margin": min_margin,
                })
                tier_seen[tier] += 1
                if len(slot_candidates) >= candidate_per_slot:
                    break
        if not slot_candidates:
            raise RuntimeError(f"No feasible candidate for scene={scene.scene_id}, cell={cell}, rep={rep}.")
        candidates_by_slot.append(slot_candidates)
        if (slot_idx + 1) % 150 == 0 or slot_idx == len(slots) - 1:
            print(json.dumps({
                "status": "eval_A_pool_progress",
                "slot": slot_idx + 1,
                "n_slots": len(slots),
                "last_slot_size": len(slot_candidates),
                "last_slot_latent_attempts": latent_attempts,
                "last_slot_tiers": dict(tier_seen),
            }, ensure_ascii=False), flush=True)

    tier_coverage_stats: Counter[Tuple[str, ...]] = Counter()
    for slot_cands in candidates_by_slot:
        present = {c["tier"] for c in slot_cands}
        tier_coverage_stats[tuple(sorted(present))] += 1
    print(json.dumps({
        "status": "eval_A_pool_done",
        "tier_coverage_counts": {"|".join(k): v for k, v in tier_coverage_stats.items()},
    }, ensure_ascii=False), flush=True)

    # ----- Joint local search on distractor-rule balance only. -----
    # tier is NOT constrained here; it is recorded on each record so downstream
    # analysis can subgroup by (hard / medium / easy).
    assignment_idx, joint_stats = _joint_assign_local_search(
        rng=rng,
        candidates_by_slot=candidates_by_slot,
        non_gold_rules=non_gold_rules,
        rule_targets=rule_targets,
        tier_targets=None,
        max_steps=EVAL_A_JOINT_MAX_STEPS,
        stall_patience=EVAL_A_JOINT_STALL_PATIENCE,
        tolerance=EVAL_A_JOINT_TOLERANCE,
        label="eval_A_rule_balance",
    )
    if joint_stats["total_L1_final"] > EVAL_A_JOINT_TOLERANCE:
        raise RuntimeError(
            "eval_A distractor-rule balance did not converge within tolerance. "
            f"stats={json.dumps(joint_stats, ensure_ascii=False)}"
        )
    ident_stats["joint_assignment"] = joint_stats

    # Materialize records from assignment.
    margin_tier_stats: Counter[str] = Counter()
    distractor_usage: Counter[str] = Counter()
    _letter_counter = 0
    for slot_i, (scene, _cell, rep) in enumerate(slots):
        picked = candidates_by_slot[slot_i][assignment_idx[slot_i]]
        gold_rule = gold_rule_for_domain(scene.domain_key)
        latent = picked["latent"]
        rule_x = picked["rule_x"]
        picks = picked["picks"]
        tau_pair_used = picked["pair_tau_used"]
        forced_letter = letter_cycle[_letter_counter % 4]
        _letter_counter += 1
        options = build_mcq_options(
            rng=rng,
            gold_rule_id=gold_rule,
            candidate_distractor_ids=picks,
            rule_x_values=rule_x,
            action_label=scene.action_label,
            peer_noun_singular=scene.peer_noun_singular,
            peer_noun_plural=scene.peer_noun_plural,
            value_format=scene.value_format,
            unit=scene.action_unit,
            num_distractors=3,
            forced_gold_letter=forced_letter,
        )
        gold_letter = next(o.letter for o in options if o.rule_id == gold_rule)
        margins = {o.rule_id: abs(o.x_value - rule_x[gold_rule]) / max(latent.F, 1e-9) for o in options if o.rule_id != gold_rule}
        identification_margin = min(margins.values()) if margins else 0.0
        margin_tier = _eval_a_margin_tier(identification_margin)
        margin_tier_stats[margin_tier] += 1
        for o in options:
            if o.rule_id != gold_rule:
                distractor_usage[o.rule_id] += 1
        rec = _record_base(
            record_id=f"evalA__{scene.scene_id}__{latent.cell_id}__{rep:02d}",
            dataset_split="eval_A",
            scene=scene,
            latent=latent,
            gold_rule_id=gold_rule,
            gold_x=rule_x[gold_rule],
            seed=args.seed,
        )
        rec["mcq"] = {"options": [{"letter": o.letter, "text": o.text, "x_display": o.x_display} for o in options]}
        ensure_oracle(rec)["mcq"] = {
            "gold_letter": gold_letter,
            "gold_rule_id": gold_rule,
            "identification_tau": round(TAU_POSITIONAL, 6),
            "identification_margin": round(identification_margin, 6),
            "identification_margin_tier": margin_tier,
            "identification_degraded": False,
            "pair_tau_used": round(tau_pair_used, 6),
            "selection_stratified": True,
            "selection_pairwise_ok": True,
            "selection_degraded": False,
            "all_rule_x_values": {k: round(v, 6) for k, v in rule_x.items()},
            "letter_to_rule": {o.letter: o.rule_id for o in options},
            "options_oracle": [{"letter": o.letter, "rule_id": o.rule_id, "rule_short": o.rule_short, "x_value": round(o.x_value, 6)} for o in options],
        }
        records.append(rec)
        ident_stats["clean"] += 1

    print(json.dumps({
        "eval_a_ident_stats": ident_stats,
        "eval_a_margin_tier_stats": dict(margin_tier_stats),
        "eval_a_distractor_target_each": target_each,
        "eval_a_distractor_usage": dict(distractor_usage),
        "eval_a_tier_coverage_stats": {"|".join(k): v for k, v in tier_coverage_stats.items()},
    }, ensure_ascii=False), flush=True)
    rng.shuffle(records)
    return records


# =============================================================================
# Split 3 — EVAL-B (comparative-statics pairwise on d1+d3)
# =============================================================================
# We construct pairs (rec_A, rec_B) that share a scene and differ in ONE
# Langtry dimension. The "direction" label is the theoretical comparative
# static prediction for x_star_A vs x_star_B. Five perturbation types:
#   alpha_up       alpha_i_B > alpha_i_A       -> x_star_B > x_star_A
#   F_up           F_B > F_A                    -> x_star_B > x_star_A
#   ref_sum_up     Sum g x in B > that in A     -> x_star_B > x_star_A (if alpha>0)
#   top_weight_up  the CLOSEST peer's g rises   -> direction depends on sign of (x_top - mean)
#   peer_action_up one peer's x rises           -> x_star rises if that peer has g>0

PAIR_PERTURBATIONS: Tuple[str, ...] = (
    "alpha_up", "F_up", "ref_sum_up", "top_weight_up", "peer_action_up",
)


def _direction_sign(a: float, b: float, eps: float = 1e-6) -> str:
    if b - a > eps:
        return "B>A"
    if a - b > eps:
        return "A>B"
    return "A==B"


def _perturb_alpha_up(rng, latent_A):
    # Keep scene / F / x_j / g_ij; bump alpha from current bucket to the next.
    current = latent_A.alpha_bucket
    if current == "low":
        new_bucket = rng.choice(["mid", "high"])
    elif current == "mid":
        new_bucket = "high"
    else:
        new_bucket = "low"  # we invert the labelling; it is still a single-dim perturbation
        # In this case x_B > x_A does NOT hold; swap labels to always keep A as lower alpha.
    return ("alpha", new_bucket)


def _perturb_F_up(rng, F_range: Tuple[float, float], F_A: float) -> float:
    """Return F_B > F_A within the scene F_range. Min bump of 8% of F_A so it is legible."""
    low = max(F_A * 1.08, F_range[0])
    high = F_range[1]
    if high <= low:
        high = F_A * 1.25
    return rng.uniform(low, high)


def _sync_pair_peer_field(latent_B, latent_A) -> None:
    """Overwrite B's peer field with A's (used when perturbation is not on peers)."""
    latent_B.N_peers = latent_A.N_peers
    latent_B.x_j = list(latent_A.x_j)
    latent_B.g_ij = list(latent_A.g_ij)
    latent_B.peer_rank_by_g = list(latent_A.peer_rank_by_g)
    latent_B.peer_rank_by_x = list(latent_A.peer_rank_by_x)


def _make_paired_latent(
    rng: random.Random,
    scene: Scene,
    perturbation: str,
) -> Tuple[Any, Any, str, str]:
    """Return (latent_A, latent_B, perturbation_applied, direction).

    direction is one of "A>B", "B>A", "A==B".
    We construct A first on a random main cell, then derive B by perturbing
    the specified dimension.
    """
    F_A = rng.uniform(*scene.F_range)
    cell = rng.choice(iter_main_cells())
    latent_A = sample_latent_langtry(
        rng=rng, domain_key=scene.domain_key, F=F_A,
        alpha_bucket=cell[0], dispersion_bucket=cell[1], skew_bucket=cell[2],
    )

    if perturbation == "alpha_up":
        # Keep A's alpha bucket as "low" or "mid"; force B at a higher bucket.
        if cell[0] == "high":
            # Re-sample A with a lower alpha bucket to guarantee B goes up.
            lower_alpha = rng.choice(["low", "mid"])
            latent_A = sample_latent_langtry(
                rng=rng, domain_key=scene.domain_key, F=F_A,
                alpha_bucket=lower_alpha, dispersion_bucket=cell[1], skew_bucket=cell[2],
            )
            new_alpha_bucket = "high"
        elif cell[0] == "mid":
            new_alpha_bucket = "high"
        else:
            new_alpha_bucket = rng.choice(["mid", "high"])
        # Build B: same F, same x_j, same g_ij, same bucket labels except alpha.
        # We re-call sample_latent_langtry to get a fresh alpha draw in that bucket,
        # then overwrite the peer field with A's to keep all else equal.
        latent_B = sample_latent_langtry(
            rng=rng, domain_key=scene.domain_key, F=F_A,
            alpha_bucket=new_alpha_bucket,
            dispersion_bucket=latent_A.dispersion_bucket,
            skew_bucket=latent_A.skew_bucket,
        )
        _sync_pair_peer_field(latent_B, latent_A)
        latent_B.ref_sum = latent_A.ref_sum
        latent_B.x_i_star = latent_B.F + latent_B.alpha_i * latent_B.ref_sum
        direction = _direction_sign(latent_A.x_i_star, latent_B.x_i_star)
        return latent_A, latent_B, "alpha_i_up", direction

    if perturbation == "F_up":
        F_B = _perturb_F_up(rng, scene.F_range, F_A)
        # B uses same alpha / x_j / g_ij but different F. We rebuild the latent
        # keeping peer_field identical.
        latent_B = sample_latent_langtry(
            rng=rng, domain_key=scene.domain_key, F=F_B,
            alpha_bucket=latent_A.alpha_bucket,
            dispersion_bucket=latent_A.dispersion_bucket,
            skew_bucket=latent_A.skew_bucket,
        )
        _sync_pair_peer_field(latent_B, latent_A)
        latent_B.ref_sum = latent_A.ref_sum
        latent_B.alpha_i = latent_A.alpha_i
        latent_B.x_i_star = latent_B.F + latent_B.alpha_i * latent_B.ref_sum
        direction = _direction_sign(latent_A.x_i_star, latent_B.x_i_star)
        return latent_A, latent_B, "F_up", direction

    if perturbation == "ref_sum_up":
        # Multiply every peer's x_j by a uniform factor > 1 so ref_sum rises
        # while g_ij stays identical. Everything else unchanged.
        factor = rng.uniform(1.15, 1.45)
        latent_B = sample_latent_langtry(
            rng=rng, domain_key=scene.domain_key, F=F_A,
            alpha_bucket=latent_A.alpha_bucket,
            dispersion_bucket=latent_A.dispersion_bucket,
            skew_bucket=latent_A.skew_bucket,
        )
        latent_B.N_peers = latent_A.N_peers
        latent_B.x_j = [x * factor for x in latent_A.x_j]
        latent_B.g_ij = list(latent_A.g_ij)
        latent_B.alpha_i = latent_A.alpha_i
        latent_B.ref_sum = sum(g * x for g, x in zip(latent_B.g_ij, latent_B.x_j))
        latent_B.x_i_star = latent_B.F + latent_B.alpha_i * latent_B.ref_sum
        latent_B.peer_rank_by_g = list(latent_A.peer_rank_by_g)
        latent_B.peer_rank_by_x = sorted(range(latent_B.N_peers), key=lambda i: -latent_B.x_j[i])
        direction = _direction_sign(latent_A.x_i_star, latent_B.x_i_star)
        return latent_A, latent_B, "ref_sum_up", direction

    if perturbation == "top_weight_up":
        # Raise the CLOSEST peer's g_ij by delta, shrink the rest proportionally.
        latent_B = sample_latent_langtry(
            rng=rng, domain_key=scene.domain_key, F=F_A,
            alpha_bucket=latent_A.alpha_bucket,
            dispersion_bucket=latent_A.dispersion_bucket,
            skew_bucket=latent_A.skew_bucket,
        )
        top = latent_A.peer_rank_by_g[0]
        current_top = latent_A.g_ij[top]
        delta = min(0.15, (1.0 - current_top) * 0.6)
        if delta < 0.01:
            # Already saturated; fall back to alpha perturbation.
            return _make_paired_latent(rng, scene, "alpha_up")
        new_top = current_top + delta
        remaining = 1.0 - new_top
        rest_sum = sum(latent_A.g_ij) - current_top or 1.0
        new_g = [
            (new_top if i == top else latent_A.g_ij[i] * remaining / rest_sum)
            for i in range(latent_A.N_peers)
        ]
        latent_B.N_peers = latent_A.N_peers
        latent_B.x_j = list(latent_A.x_j)
        latent_B.g_ij = new_g
        latent_B.alpha_i = latent_A.alpha_i
        latent_B.ref_sum = sum(g * x for g, x in zip(new_g, latent_A.x_j))
        latent_B.x_i_star = latent_B.F + latent_B.alpha_i * latent_B.ref_sum
        latent_B.peer_rank_by_g = sorted(range(latent_B.N_peers), key=lambda i: -new_g[i])
        latent_B.peer_rank_by_x = list(latent_A.peer_rank_by_x)
        direction = _direction_sign(latent_A.x_i_star, latent_B.x_i_star)
        return latent_A, latent_B, "top_weight_up", direction

    if perturbation == "peer_action_up":
        # Raise the CLOSEST peer's x_j by factor, keeping g_ij identical.
        latent_B = sample_latent_langtry(
            rng=rng, domain_key=scene.domain_key, F=F_A,
            alpha_bucket=latent_A.alpha_bucket,
            dispersion_bucket=latent_A.dispersion_bucket,
            skew_bucket=latent_A.skew_bucket,
        )
        top = latent_A.peer_rank_by_g[0]
        factor = rng.uniform(1.20, 1.60)
        new_x = [(x * factor if i == top else x) for i, x in enumerate(latent_A.x_j)]
        latent_B.N_peers = latent_A.N_peers
        latent_B.x_j = new_x
        latent_B.g_ij = list(latent_A.g_ij)
        latent_B.alpha_i = latent_A.alpha_i
        latent_B.ref_sum = sum(g * x for g, x in zip(latent_B.g_ij, new_x))
        latent_B.x_i_star = latent_B.F + latent_B.alpha_i * latent_B.ref_sum
        latent_B.peer_rank_by_g = list(latent_A.peer_rank_by_g)
        latent_B.peer_rank_by_x = sorted(range(latent_B.N_peers), key=lambda i: -new_x[i])
        direction = _direction_sign(latent_A.x_i_star, latent_B.x_i_star)
        return latent_A, latent_B, "peer_action_up", direction

    raise ValueError(f"Unknown perturbation `{perturbation}`.")


EVAL_B_MAX_PRIMARY_ATTEMPTS = 200       # same-perturbation retries per slot
EVAL_B_MAX_FALLBACK_CYCLES = 5          # how many times we cycle through all perts
                                        # as fallback after primary exhaustion


def build_eval_b_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build Eval-B comparative-statics pairs with random A/B label swap.

    Hard-fill guarantee: every (scene, perturbation, k) slot is filled.
    If the primary perturbation fails to produce a distinguishable pair
    after `EVAL_B_MAX_PRIMARY_ATTEMPTS` tries (rare — happens mostly when
    `top_weight_up` saturates on an already-dominant peer), we cycle
    through the other 4 perturbations as fallback up to
    `EVAL_B_MAX_FALLBACK_CYCLES` rounds. The fallback perturbation's name
    is recorded in `pair.perturbation` (may differ from the slot's
    nominal perturbation); `pair.nominal_perturbation` records what the
    slot originally asked for so downstream analysis can still tag
    fallback pairs.

    The raw pair generator (`_make_paired_latent`) always returns the
    perturbed side as `latent_B`, which makes `direction == "B>A"` in
    >95% of cases — a trivial positional shortcut for the evaluated model.
    We flip the labels with prob ~0.5 (targeting a 50/50 direction balance
    per (scene, perturbation) cell via alternating) and record the
    `perturbed_role` for downstream subgroup analysis.
    """
    rng = random.Random(args.seed + 3000)
    records: List[Dict[str, Any]] = []
    test_scenes = [
        s for s in list_scenes(split="test")
        if s.domain_key in MAIN_POSITIONAL_DOMAINS
    ]
    perts = list(PAIR_PERTURBATIONS)
    stats = {"primary_hits": 0, "fallback_hits": 0, "dropped_slots": 0}

    def _flip_direction(d: str) -> str:
        if d == "A>B":
            return "B>A"
        if d == "B>A":
            return "A>B"
        return d  # A==B unchanged

    def _try_one(pert_choice: str) -> Optional[Tuple[Any, Any, str, str, float]]:
        """One draw; returns (latent_A, latent_B, pert_name, direction, gap)
        if it satisfies identification filter, else None."""
        latent_A, latent_B, pert_name, direction = _make_paired_latent(
            rng, scene, pert_choice
        )
        gap = abs(latent_B.x_i_star - latent_A.x_i_star) / max(latent_A.F, 1e-9)
        if direction == "A==B" or gap < TAU_PAIRWISE:
            return None
        return latent_A, latent_B, pert_name, direction, gap

    for scene in test_scenes:
        pairs_per_pert = max(args.eval_b_pairs_per_scene // len(perts), 1)
        for nominal_pert in perts:
            for k in range(pairs_per_pert):
                target_perturbed_role = "A" if (k % 2 == 0) else "B"

                outcome: Optional[Tuple[Any, Any, str, str, float]] = None
                fallback_used = False

                # --- Primary attempts on the nominal perturbation ---
                for _ in range(EVAL_B_MAX_PRIMARY_ATTEMPTS):
                    cand = _try_one(nominal_pert)
                    if cand is not None:
                        outcome = cand
                        break

                # --- Fallback: cycle through other perturbations ---
                if outcome is None:
                    fallback_pool = [p for p in perts if p != nominal_pert]
                    for _ in range(EVAL_B_MAX_FALLBACK_CYCLES):
                        # Shuffle so we don't bias toward any fallback type.
                        rng.shuffle(fallback_pool)
                        for fb_pert in fallback_pool:
                            for _ in range(EVAL_B_MAX_PRIMARY_ATTEMPTS):
                                cand = _try_one(fb_pert)
                                if cand is not None:
                                    outcome = cand
                                    fallback_used = True
                                    break
                            if outcome is not None:
                                break
                        if outcome is not None:
                            break

                if outcome is None:
                    # Extremely pathological: give up this slot.
                    stats["dropped_slots"] += 1
                    continue

                if fallback_used:
                    stats["fallback_hits"] += 1
                else:
                    stats["primary_hits"] += 1

                latent_A, latent_B, pert_name, direction, _ = outcome

                # Random A/B label flip to kill the "B is always the
                # perturbed side" shortcut.
                if target_perturbed_role == "A":
                    latent_A, latent_B = latent_B, latent_A
                    direction = _flip_direction(direction)
                perturbed_role = target_perturbed_role

                def _mk_one(tag: str, latent) -> Dict[str, Any]:
                    return _record_base(
                        record_id=f"evalB__{scene.scene_id}__{nominal_pert}__{k:02d}__{tag}",
                        dataset_split="eval_B",
                        scene=scene,
                        latent=latent,
                        gold_rule_id="A_peer_weighted",
                        gold_x=latent.x_i_star,
                        seed=args.seed,
                    )

                rec_A = _mk_one("A", latent_A)
                rec_B = _mk_one("B", latent_B)
                # ORACLE pair info — contains the answer (direction).
                pair_info = {
                    "perturbation": pert_name,
                    "nominal_perturbation": nominal_pert,
                    "fallback_used": fallback_used,
                    "perturbed_role": perturbed_role,
                    "direction": direction,
                    "gap_over_F": round(
                        abs(latent_B.x_i_star - latent_A.x_i_star) / max(latent_A.F, 1e-9), 6
                    ),
                    "cell_id_A": latent_A.cell_id,
                    "cell_id_B": latent_B.cell_id,
                    "is_held_out_cell_A": latent_A.is_held_out_cell,
                    "is_held_out_cell_B": latent_B.is_held_out_cell,
                }
                # PUBLIC pair info — only pairing metadata needed by eval_runner
                # to reconstruct the (A, B) task. No answer-bearing fields.
                rec_A["pair"] = {"role": "A", "partner_record_id": rec_B["record_id"]}
                rec_B["pair"] = {"role": "B", "partner_record_id": rec_A["record_id"]}
                ensure_oracle(rec_A)["pair"] = pair_info
                ensure_oracle(rec_B)["pair"] = pair_info
                records.append(rec_A)
                records.append(rec_B)

    print(json.dumps({"eval_b_fill_stats": stats}, ensure_ascii=False), flush=True)
    rng.shuffle(records)
    return records


# =============================================================================
# Split 4 — PLACEBO-TEST (d2 MCQ)
# =============================================================================
# Gold is D_pure_private (x = F). Distractors are computed using a TRIAL alpha
# drawn from the "mid" ALPHA bucket so that A/B/C etc. yield legibly different
# x-values, giving the identification filter something to work with. The
# narrative cue for the teacher still says "no peer audience" — the model is
# supposed to ignore the distractors and pick D.

def build_placebo_test_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rng = random.Random(args.seed + 4000)
    records: List[Dict[str, Any]] = []
    placebo_scenes = [s for s in list_scenes(split="test") if s.domain_key == PLACEBO_DOMAIN]
    placebo_letter_cycle = ("A", "B", "C", "D")
    trial_alpha_spec = ALPHA_BUCKETS["mid"]
    gold_rule = "D_pure_private"
    non_gold_rules = [rid for rid in RULE_IDS if rid != gold_rule]

    slots: List[Tuple[Scene, int]] = []
    for scene in placebo_scenes:
        for rep in range(args.placebo_test_samples_per_scene):
            slots.append((scene, rep))

    total_distractor_slots = len(slots) * 3
    if total_distractor_slots % len(non_gold_rules) != 0:
        raise ValueError(
            "Cannot evenly split placebo distractor slots across non-gold rules: "
            f"{total_distractor_slots} slots over {len(non_gold_rules)} rules."
        )
    target_each = total_distractor_slots // len(non_gold_rules)
    rule_targets = {rid: target_each for rid in non_gold_rules}

    ident_stats = {
        "clean": 0,
        "resample_retries": 0,
        "pairwise_resample_retries": 0,
        "degraded": 0,
    }

    def _one_placebo_draw(scene_ref: Scene):
        latent = _sample_one_placebo(rng, scene_ref)
        trial_alpha = rng.uniform(trial_alpha_spec.lower, trial_alpha_spec.upper)
        rule_inputs_for_distractors = _rule_inputs_with_alpha(latent, trial_alpha)
        rule_x = compute_rule_x_values(rule_inputs_for_distractors)
        rule_x["D_pure_private"] = latent.F
        return latent, rule_x, trial_alpha

    # ----- Build per-slot candidate pool. -----
    candidates_by_slot: List[List[Dict[str, Any]]] = []
    candidate_per_slot = PLACEBO_CANDIDATES_PER_SLOT
    slot_attempt_max = IDENT_RESAMPLE_MAX * PLACEBO_SLOT_ATTEMPT_MULT
    for scene, rep in slots:
        slot_candidates: List[Dict[str, Any]] = []
        seen_keys = set()
        for _ in range(slot_attempt_max):
            if len(slot_candidates) >= candidate_per_slot:
                break
            cand_latent, cand_rule_x, cand_alpha = _one_placebo_draw(scene)
            cand_ids = filter_distractors(cand_rule_x, gold_rule, cand_latent.F, TAU_PLACEBO)
            if len(cand_ids) < 3:
                ident_stats["resample_retries"] += 1
                continue
            triples = _stratified_feasible_triples(
                cand_ids, cand_rule_x, gold_rule, cand_latent.F,
                tau_pair=TAU_PAIR_PLACEBO, classify_tier=False,
            )
            if not triples:
                ident_stats["pairwise_resample_retries"] += 1
                continue
            rng.shuffle(triples)
            for tr, pair_min, _tier_none, min_margin in triples[:4]:
                key = tuple(sorted(tr))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                slot_candidates.append({
                    "latent": cand_latent,
                    "rule_x": cand_rule_x,
                    "trial_alpha": cand_alpha,
                    "picks": list(tr),
                    "pair_tau_used": pair_min,
                    "min_margin": min_margin,
                })
                if len(slot_candidates) >= candidate_per_slot:
                    break
        if not slot_candidates:
            raise RuntimeError(
                f"No feasible placebo candidate for scene={scene.scene_id}, rep={rep}."
            )
        candidates_by_slot.append(slot_candidates)

    # ----- Joint local search: rule-count only (no tier) for placebo. -----
    assignment_idx, joint_stats = _joint_assign_local_search(
        rng=rng,
        candidates_by_slot=candidates_by_slot,
        non_gold_rules=non_gold_rules,
        rule_targets=rule_targets,
        tier_targets=None,
        max_steps=PLACEBO_JOINT_MAX_STEPS,
        stall_patience=PLACEBO_JOINT_STALL_PATIENCE,
        tolerance=PLACEBO_JOINT_TOLERANCE,
        label="placebo_joint",
    )
    if joint_stats["total_L1_final"] > PLACEBO_JOINT_TOLERANCE:
        raise RuntimeError(
            "placebo_test joint assignment did not converge within tolerance. "
            f"stats={json.dumps(joint_stats, ensure_ascii=False)}"
        )
    ident_stats["joint_assignment"] = joint_stats

    _placebo_letter_counter = 0
    distractor_usage: Counter[str] = Counter()
    for slot_i, (scene, rep) in enumerate(slots):
        picked = candidates_by_slot[slot_i][assignment_idx[slot_i]]
        latent = picked["latent"]
        rule_x = picked["rule_x"]
        trial_alpha = picked["trial_alpha"]
        picks = picked["picks"]
        tau_pair_used = picked["pair_tau_used"]

        forced_letter = placebo_letter_cycle[_placebo_letter_counter % 4]
        _placebo_letter_counter += 1
        options = build_mcq_options(
            rng=rng,
            gold_rule_id=gold_rule,
            candidate_distractor_ids=picks,
            rule_x_values=rule_x,
            action_label=scene.action_label,
            peer_noun_singular=scene.peer_noun_singular or "other person",
            peer_noun_plural=scene.peer_noun_plural or "other people",
            value_format=scene.value_format,
            unit=scene.action_unit,
            num_distractors=3,
            forced_gold_letter=forced_letter,
        )
        gold_letter = next(o.letter for o in options if o.rule_id == gold_rule)
        margins = {
            o.rule_id: abs(o.x_value - rule_x[gold_rule]) / max(latent.F, 1e-9)
            for o in options if o.rule_id != gold_rule
        }
        identification_margin = min(margins.values()) if margins else 0.0
        for o in options:
            if o.rule_id != gold_rule:
                distractor_usage[o.rule_id] += 1

        rec = _record_base(
            record_id=f"placeboTest__{scene.scene_id}__{rep:03d}",
            dataset_split="placebo_test",
            scene=scene,
            latent=latent,
            gold_rule_id=gold_rule,
            gold_x=latent.F,
            seed=args.seed,
        )
        rec["mcq"] = {
            "options": [
                {"letter": o.letter, "text": o.text, "x_display": o.x_display}
                for o in options
            ],
        }
        ensure_oracle(rec)["mcq"] = {
            "gold_letter": gold_letter,
            "gold_rule_id": gold_rule,
            "identification_tau": round(TAU_PLACEBO, 6),
            "identification_margin": round(identification_margin, 6),
            "identification_degraded": False,
            "pair_tau_used": round(tau_pair_used, 6),
            "selection_stratified": True,
            "selection_pairwise_ok": True,
            "selection_degraded": False,
            "all_rule_x_values": {k: round(v, 6) for k, v in rule_x.items()},
            "trial_alpha_used_for_distractors": round(trial_alpha, 6),
            "letter_to_rule": {o.letter: o.rule_id for o in options},
            "options_oracle": [
                {
                    "letter": o.letter,
                    "rule_id": o.rule_id,
                    "rule_short": o.rule_short,
                    "x_value": round(o.x_value, 6),
                }
                for o in options
            ],
        }
        records.append(rec)
        ident_stats["clean"] += 1

    print(json.dumps({
        "placebo_ident_stats": ident_stats,
        "placebo_distractor_target_each": target_each,
        "placebo_distractor_usage": dict(distractor_usage),
    }, ensure_ascii=False), flush=True)
    rng.shuffle(records)
    return records


# =============================================================================
# Split 5 — OOD SOCIAL (d4)
# =============================================================================
# Two-candidate MCQ: pick the candidate whose b/c ratio matches the protagonist.
# Public prompts expose b and c separately; b/c itself remains oracle-only.
# We randomise match_distance_bucket across close/mid/far for subgroup analysis.

def _format_social_param(value: float) -> str:
    """Display latent b/c ingredients without making the ratio too easy."""
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _social_cost_hint(c_value: float) -> str:
    """Natural-language tier for c as financial constraint / spending pressure."""
    if c_value < 1.15:
        return "comfortable budget; extra spending does not crowd out essentials"
    if c_value < 1.85:
        return "budget-conscious; extra spending requires some tradeoffs"
    return "financially constrained; extra spending competes with important bills"


def _social_benefit_hint(bucket: str) -> str:
    if bucket == "low":
        return "mild interest; the tie would be pleasant but not central"
    if bucket == "mid":
        return "clear value; the tie would matter and be useful"
    if bucket == "high":
        return "strong value; the tie feels important and highly worthwhile"
    raise ValueError(f"Unknown benefit bucket `{bucket}`")


def _social_cost_bucket_hint(bucket: str) -> str:
    if bucket == "low":
        return "financially comfortable; the spending fits easily"
    if bucket == "mid":
        return "budget-conscious; the spending requires some tradeoffs"
    if bucket == "high":
        return "financially constrained; the spending competes with important bills"
    raise ValueError(f"Unknown cost bucket `{bucket}`")


def build_ood_social_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rng = random.Random(args.seed + 5000)
    records: List[Dict[str, Any]] = []
    scenes = [s for s in list_scenes() if s.domain_key == OOD_DOMAIN_SOCIAL]

    bucket_order = list(SOCIAL_MATCH_DISTANCE_BUCKETS)
    social_cells = iter_social_cells()
    n_cells = len(social_cells)
    for scene in scenes:
        per = args.ood_social_samples_per_scene
        for rep in range(per):
            # The default per-scene size is 72, so each of the 9 public
            # protagonist cells appears exactly 8 times per scene.
            cell = social_cells[rep % n_cells]
            mb = bucket_order[(rep // n_cells) % len(bucket_order)]
            forced_idx = (rep // (n_cells * len(bucket_order))) % 2
            target_letter = ("A", "B")[(rep // (n_cells * len(bucket_order) * 2)) % 2]
            latent = sample_latent_social_ood(
                rng=rng, f_range=scene.F_range,
                match_distance_bucket=mb,
                protagonist_cell=cell,
                forced_gold_index=forced_idx,
            )
            # Options are the two candidates. We label them Mia and Jake for
            # the teacher narrative but the structured record uses P1/P2.
            option_ids = ["P1", "P2"]
            display = lambda v: format_value(v, scene.value_format, scene.action_unit)
            options = [
                {
                    "option_id": option_ids[0],
                    "b_value": round(latent.b_candidates[0], 6),
                    "b_bucket": latent.b_bucket_candidates[0],
                    "b_hint": _social_benefit_hint(latent.b_bucket_candidates[0]),
                    "c_value": round(latent.c_candidates[0], 6),
                    "c_bucket": latent.c_bucket_candidates[0],
                    "c_hint": _social_cost_bucket_hint(latent.c_bucket_candidates[0]),
                    "x_value": round(latent.x_candidates[0], 6),
                    "x_display": display(latent.x_candidates[0]),
                    "bc_ratio": round(latent.bc_ratio_candidates[0], 6),
                },
                {
                    "option_id": option_ids[1],
                    "b_value": round(latent.b_candidates[1], 6),
                    "b_bucket": latent.b_bucket_candidates[1],
                    "b_hint": _social_benefit_hint(latent.b_bucket_candidates[1]),
                    "c_value": round(latent.c_candidates[1], 6),
                    "c_bucket": latent.c_bucket_candidates[1],
                    "c_hint": _social_cost_bucket_hint(latent.c_bucket_candidates[1]),
                    "x_value": round(latent.x_candidates[1], 6),
                    "x_display": display(latent.x_candidates[1]),
                    "bc_ratio": round(latent.bc_ratio_candidates[1], 6),
                },
            ]
            letters = ("A", "B")
            gold_opt_idx = latent.gold_candidate_index
            # Deterministically place the gold candidate under `target_letter`
            # and the loser under the other letter — this makes gold_letter
            # alternate A/B per rep (no small-sample letter imbalance).
            if target_letter == "A":
                letter_to_opt = {"A": options[gold_opt_idx], "B": options[1 - gold_opt_idx]}
            else:
                letter_to_opt = {"B": options[gold_opt_idx], "A": options[1 - gold_opt_idx]}
            gold_letter = target_letter

            scene_dict = _scene_summary(scene)
            latent_dict = {
                "domain_key": OOD_DOMAIN_SOCIAL,
                "b_i": round(latent.b_i, 6),
                "b_i_display": _format_social_param(latent.b_i),
                "b_i_bucket": latent.b_bucket_i,
                "b_i_hint": _social_benefit_hint(latent.b_bucket_i),
                "c_i": round(latent.c_i, 6),
                "c_i_display": _format_social_param(latent.c_i),
                "c_i_bucket": latent.c_bucket_i,
                "c_i_hint": _social_cost_bucket_hint(latent.c_bucket_i),
                "social_cell_id": f"b_{latent.b_bucket_i}__c_{latent.c_bucket_i}",
                "c": round(latent.c_i, 6),  # backward-compatible alias
                "x_i": round(latent.x_i, 6),
                "x_i_display": display(latent.x_i),
                "bc_ratio_i": round(latent.bc_ratio_i, 6),
                "b_candidates": [round(v, 6) for v in latent.b_candidates],
                "b_candidate_displays": [_format_social_param(v) for v in latent.b_candidates],
                "b_candidate_buckets": list(latent.b_bucket_candidates),
                "b_candidate_hints": [_social_benefit_hint(v) for v in latent.b_bucket_candidates],
                "c_candidates": [round(v, 6) for v in latent.c_candidates],
                "c_candidate_displays": [_format_social_param(v) for v in latent.c_candidates],
                "c_candidate_buckets": list(latent.c_bucket_candidates),
                "c_candidate_hints": [_social_cost_bucket_hint(v) for v in latent.c_bucket_candidates],
                "x_candidates": [round(v, 6) for v in latent.x_candidates],
                "x_candidate_displays": [display(v) for v in latent.x_candidates],
                "bc_ratio_candidates": [round(v, 6) for v in latent.bc_ratio_candidates],
                "gold_candidate_index": latent.gold_candidate_index,
                "match_distance_bucket": latent.match_distance_bucket,
            }
            # NOTE: b and c appear verbatim in scenario_text by design, but
            # x=b/c and the gold letter are oracle-only. This keeps Eval-D from
            # collapsing into a direct "match the displayed preferred action"
            # exercise.
            public_options = [
                {
                    "letter": letter,
                    "option_id": letter_to_opt[letter]["option_id"],
                    "b_bucket": letter_to_opt[letter]["b_bucket"],
                    "b_hint": letter_to_opt[letter]["b_hint"],
                    "c_bucket": letter_to_opt[letter]["c_bucket"],
                    "c_hint": letter_to_opt[letter]["c_hint"],
                }
                for letter in letters
            ]
            oracle_options = [
                {
                    "letter": letter,
                    "option_id": letter_to_opt[letter]["option_id"],
                    "b_value": letter_to_opt[letter]["b_value"],
                    "c_value": letter_to_opt[letter]["c_value"],
                    "x_value": letter_to_opt[letter]["x_value"],
                    "bc_ratio": letter_to_opt[letter]["bc_ratio"],
                }
                for letter in letters
            ]
            rec: Dict[str, Any] = {
                "record_id": f"oodSocial__{scene.scene_id}__{mb}__{rep:03d}",
                "dataset_split": "ood_social",
                "schema_version": SCHEMA_VERSION,
                "langtry_schema_version": LANGTRY_SCHEMA_VERSION,
                "generator": GENERATOR_NAME,
                "seed": args.seed,
                "scene": scene_dict,
                "scenario_text": None,
                "ood_social": {
                    "letters": list(letters),
                    "options": public_options,
                },
                "oracle": {
                    "latent": latent_dict,
                    "ood_social": {
                        "gold_letter": gold_letter,
                        "match_distance_bucket": mb,
                        "gold_candidate_index": latent.gold_candidate_index,
                        "bc_ratio_i": round(latent.bc_ratio_i, 6),
                        "options_oracle": oracle_options,
                    },
                },
            }
            records.append(rec)
    rng.shuffle(records)
    return records


# =============================================================================
# Split 6 — OOD CAREER (d5)
# =============================================================================

def build_ood_career_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build OOD-career records with balanced gold_firm H/L.

    For every scene we iterate the three `alpha_2i` buckets in round-robin
    and, within each bucket slot, alternate the target `gold_firm` between
    "H" and "L". `sample_latent_career_ood` then resamples wages / alpha
    within-bucket until the Langtry decision rule lands on the target.
    This removes the natural-sampling skew toward "L" that a prior run
    showed (85% L).
    """
    rng = random.Random(args.seed + 6000)
    records: List[Dict[str, Any]] = []
    scenes = [s for s in list_scenes() if s.domain_key == OOD_DOMAIN_CAREER]
    buckets = list(ALPHA_2_BUCKETS.keys())
    # Round-robin over all (bucket, target_firm) combos so both the alpha_2i
    # bucket and the gold_firm side balance evenly per scene. With 3 buckets
    # x 2 firms = 6 combos, any multiple of 6 yields exact H/L and per-bucket
    # balance; for per=20 we get 4·(low,H) 4·(low,L) 3·(mid,H) 3·(mid,L)
    # 3·(high,H) 3·(high,L) → 10 H + 10 L per scene.
    # Round-robin over (bucket × target_firm × gold_letter_target) so all
    # three axes balance per scene: 3 × 2 × 2 = 12 combos.
    combos: List[Tuple[str, str, str]] = [
        (b, f, letter)
        for b in buckets
        for f in ("H", "L")
        for letter in ("A", "B")
    ]
    fill_stats = {"target_hits": 0, "target_misses": 0, "outer_retries": 0}
    OOD_CAREER_OUTER_MAX = 8  # outer-loop rounds if inner sampler misses target

    for scene in scenes:
        wage_anchor = float(scene.extras.get("wage_anchor", 150000.0))
        per = args.ood_career_samples_per_scene
        for rep in range(per):
            bkt, target_firm, target_letter_career = combos[rep % len(combos)]
            latent = None
            for outer_try in range(OOD_CAREER_OUTER_MAX):
                cand = sample_latent_career_ood(
                    rng=rng,
                    wage_anchor=wage_anchor,
                    alpha_2i_bucket=bkt,
                    target_gold_firm=target_firm,
                )
                if cand.gold_firm == target_firm:
                    latent = cand
                    fill_stats["target_hits"] += 1
                    break
                fill_stats["outer_retries"] += 1
                latent = cand  # keep last as fallback
            if latent.gold_firm != target_firm:
                fill_stats["target_misses"] += 1
            display = lambda v: format_value(v, scene.value_format, scene.action_unit)
            options = [
                {
                    "firm_tag": "H",
                    "x_S": round(latent.x_S_H, 2),
                    "x_bar": round(latent.x_bar_H, 2),
                    "x_S_display": display(latent.x_S_H),
                    "x_bar_display": display(latent.x_bar_H),
                    "relative_rank": "bottom of a high-pay cohort",
                },
                {
                    "firm_tag": "L",
                    "x_S": round(latent.x_S_L, 2),
                    "x_bar": round(latent.x_bar_L, 2),
                    "x_S_display": display(latent.x_S_L),
                    "x_bar_display": display(latent.x_bar_L),
                    "relative_rank": "top of a lower-pay cohort",
                },
            ]
            letters = ("A", "B")
            # Deterministically place `gold_firm` under `target_letter_career`
            # so gold_letter balances A/B regardless of small-n variance.
            gold_idx = 0 if latent.gold_firm == "H" else 1
            if target_letter_career == "A":
                letter_to_opt = {"A": options[gold_idx], "B": options[1 - gold_idx]}
            else:
                letter_to_opt = {"B": options[gold_idx], "A": options[1 - gold_idx]}
            gold_letter = target_letter_career

            scene_dict = _scene_summary(scene)
            latent_dict = {
                "domain_key": OOD_DOMAIN_CAREER,
                "alpha_2i": round(latent.alpha_2i, 6),
                "alpha_2i_bucket": latent.alpha_2i_bucket,
                "x_bar_H": round(latent.x_bar_H, 6),
                "x_bar_L": round(latent.x_bar_L, 6),
                "x_S_H": round(latent.x_S_H, 6),
                "x_S_L": round(latent.x_S_L, 6),
                "lhs": round(latent.lhs, 6),
                "rhs": round(latent.rhs, 6),
                "gold_firm": latent.gold_firm,
            }
            # PUBLIC options keep x_S / x_bar / relative_rank cues because
            # they appear verbatim in the scenario_text (the teacher writes
            # both numbers and the rank sentence). gold_letter + gold_firm
            # are oracle.
            public_options = [
                {
                    "letter": letter,
                    "firm_tag": letter_to_opt[letter]["firm_tag"],
                    "x_S_display": letter_to_opt[letter]["x_S_display"],
                    "x_bar_display": letter_to_opt[letter]["x_bar_display"],
                    "relative_rank": letter_to_opt[letter]["relative_rank"],
                }
                for letter in letters
            ]
            rec: Dict[str, Any] = {
                "record_id": f"oodCareer__{scene.scene_id}__{bkt}__{rep:03d}",
                "dataset_split": "ood_career",
                "schema_version": SCHEMA_VERSION,
                "langtry_schema_version": LANGTRY_SCHEMA_VERSION,
                "generator": GENERATOR_NAME,
                "seed": args.seed,
                "scene": scene_dict,
                "scenario_text": None,
                "ood_career": {
                    "letters": list(letters),
                    "options": public_options,
                },
                "oracle": {
                    "latent": latent_dict,
                    "ood_career": {
                        "gold_letter": gold_letter,
                        "gold_firm": latent.gold_firm,
                        "alpha_2i_bucket": bkt,
                        "lhs": round(latent.lhs, 6),
                        "rhs": round(latent.rhs, 6),
                    },
                },
            }
            records.append(rec)
    print(json.dumps({"ood_career_fill_stats": fill_stats}, ensure_ascii=False), flush=True)
    rng.shuffle(records)
    return records


# =============================================================================
# Writers
# =============================================================================

def _records_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_domain: Counter = Counter()
    by_cell: Counter = Counter()
    by_heldout: Counter = Counter()
    by_alpha: Counter = Counter()
    by_scene: Counter = Counter()
    for r in records:
        by_domain[r["scene"]["domain_key"]] += 1
        latent = r.get("oracle", {}).get("latent", {}) or {}
        cell_id = latent.get("cell_id")
        if cell_id:
            by_cell[cell_id] += 1
        ho = latent.get("is_held_out_cell")
        if ho is not None:
            by_heldout["held_out" if ho else "main"] += 1
        ab = latent.get("alpha_bucket")
        if ab:
            by_alpha[ab] += 1
        by_scene[r["scene"]["scene_id"]] += 1
    return {
        "record_count": len(records),
        "by_domain": dict(by_domain),
        "by_cell": dict(by_cell),
        "by_heldout_cell": dict(by_heldout),
        "by_alpha_bucket": dict(by_alpha),
        "by_scene": dict(by_scene),
    }


def _write_split(output_dir: Path, split_name: str, seed: int, records: List[Dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split_name}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "langtry_schema_version": LANGTRY_SCHEMA_VERSION,
        "generator": GENERATOR_NAME,
        "split": split_name,
        "seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "records": records,
        "summary": _records_summary(records),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# =============================================================================
# Entry
# =============================================================================

SPLIT_DISPATCH = {
    "train": build_train_records,
    "eval_A": build_eval_a_records,
    "eval_B": build_eval_b_records,
    "placebo_test": build_placebo_test_records,
    "ood_social": build_ood_social_records,
    "ood_career": build_ood_career_records,
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    splits = (
        ["train", "eval_A", "eval_B", "placebo_test", "ood_social", "ood_career"]
        if args.split == "all" else [args.split]
    )
    for s in splits:
        records = SPLIT_DISPATCH[s](args)
        path = _write_split(output_dir, s, args.seed, records)
        print(json.dumps({
            "status": "split_written",
            "split": s,
            "records": len(records),
            "path": str(path),
            "summary": _records_summary(records),
        }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
