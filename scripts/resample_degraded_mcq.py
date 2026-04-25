"""
Resample ONLY the latents of records whose MCQ distractor selection is
`selection_degraded=True`, then clear their `scenario_text`/`teacher_meta`
so the teacher LLM re-writes them on the next pass.

Why this exists
---------------
After introducing the stratified + pair-wise-separability requirement for
MCQ distractors, a small number of pre-existing records in eval_A /
placebo_test could not satisfy the new constraint from their original
latents (= `selection_degraded=True`). This script:

1. Loads a structured JSON (eval_A.json or placebo_test.json).
2. Finds records with `oracle.mcq.selection_degraded == True`.
3. For each such record, re-samples a fresh `latent` (within the SAME scene
   and SAME (alpha × dispersion × skew) cell for positional records; within
   the same scene for placebo) until the identification + stratification +
   pair-wise separability checks all pass. Falls back to a best-effort
   degraded draw only if IDENT_RESAMPLE_MAX attempts fail.
4. Rebuilds `oracle.latent`, `oracle.peer_cards`, `oracle.gold`, `mcq`,
   and `oracle.mcq` from the new latent.
5. Clears `scenario_text` (sets it to `None`) and pops `teacher_meta`,
   so the normal teacher-LLM pipeline (write_scenarios_with_llm.py) will
   naturally regenerate ONLY these records on its next run (it picks up
   every record whose `scenario_text` is null).

The record_id, scene metadata, seed, split, and schema version are all
preserved. Records that were already clean are left untouched.

Usage
-----
    python -m scripts.resample_degraded_mcq \
        --input-file data/structured/eval_A.json \
        --seed 20260421

    python -m scripts.resample_degraded_mcq \
        --input-file data/structured/placebo_test.json \
        --seed 20260421

Outputs a one-line JSON summary to stdout on completion.

Note on reproducibility
-----------------------
Each degraded record gets its own RNG seeded with
`base_seed + stable_hash(record_id)`, so repeated runs are deterministic
and also independent: re-running never perturbs records outside the
degraded set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config.langtry_parameters import (  # noqa: E402
    ALPHA_BUCKETS,
    PLACEBO_DOMAIN,
    iter_all_cells,
    sample_latent_langtry,
)
from data.config.record_access import ensure_oracle  # noqa: E402
from data.config.rule_label_templates import (  # noqa: E402
    RuleInputs,
    build_mcq_options,
    compute_rule_x_values,
    format_value,
    gold_rule_for_domain,
)
from data.config.scene_pool import Scene, list_scenes  # noqa: E402
from data.generation.identification_filter import (  # noqa: E402
    TAU_PAIR_PLACEBO,
    TAU_PAIR_POSITIONAL,
    TAU_PLACEBO,
    TAU_POSITIONAL,
    NotEnoughDistractorsError,
    filter_distractors_with_hard_fallback,
    filter_distractors_with_shortfall_relaxation,
    select_stratified_distractors,
)

IDENT_RESAMPLE_MAX = 60  # a bit higher than the builder default to give the
                         # targeted retry the best chance to succeed.


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _record_seed(base_seed: int, record_id: str) -> int:
    h = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    return (base_seed ^ int(h[:12], 16)) & 0x7FFFFFFF


def _closeness_hint(g_weight: float) -> str:
    if g_weight >= 0.45:
        return "tight"
    if g_weight >= 0.18:
        return "mid"
    return "peripheral"


def _build_peer_cards(latent, scene: Scene) -> List[Dict[str, Any]]:
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
    from dataclasses import asdict
    d = asdict(latent)
    d["alpha_i"] = round(d["alpha_i"], 6)
    d["F"] = round(d["F"], 6)
    d["c"] = round(d["c"], 6)
    d["ref_sum"] = round(d["ref_sum"], 6)
    d["x_i_star"] = round(d["x_i_star"], 6)
    d["x_j"] = [round(v, 6) for v in d["x_j"]]
    d["g_ij"] = [round(v, 6) for v in d["g_ij"]]
    return d


def _rule_inputs_from_latent(latent) -> RuleInputs:
    return RuleInputs(
        F=latent.F,
        alpha_i=latent.alpha_i,
        x_j=latent.x_j,
        g_ij=latent.g_ij,
        closest_peer_index=latent.peer_rank_by_g[0],
    )


def _rule_inputs_with_alpha(latent, alpha_override: float) -> RuleInputs:
    return RuleInputs(
        F=latent.F,
        alpha_i=alpha_override,
        x_j=latent.x_j,
        g_ij=latent.g_ij,
        closest_peer_index=latent.peer_rank_by_g[0],
    )


def _index_scenes() -> Dict[str, Scene]:
    out: Dict[str, Scene] = {}
    for split in ("train", "test"):
        for s in list_scenes(split=split):
            out[s.scene_id] = s
    return out


# ---------------------------------------------------------------------------
# positional (eval_A)
# ---------------------------------------------------------------------------

def _sample_positional(
    rng: random.Random, scene: Scene, cell: Tuple[str, str, str],
):
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


def _resample_one_positional(
    rec: Dict[str, Any], scene: Scene, rng: random.Random, letter: str,
) -> Dict[str, Any]:
    oracle = rec["oracle"]
    prev_latent = oracle["latent"]
    cell = (
        prev_latent["alpha_bucket"],
        prev_latent["dispersion_bucket"],
        prev_latent["skew_bucket"],
    )
    gold_rule = oracle["gold"]["rule_id"]
    if not gold_rule or gold_rule == "D_pure_private":
        # Fall back to canonical domain mapping if the gold is somehow missing
        # or was mis-stored; never happens for genuine eval_A.
        gold_rule = gold_rule_for_domain(scene.domain_key)

    latent = None
    rule_x: Dict[str, float] = {}
    distractor_ids: List[str] = []
    picks: List[str] = []
    tau_used = TAU_POSITIONAL
    tau_pair_used = TAU_PAIR_POSITIONAL
    stratified_ok = False
    pairwise_ok = False
    degraded_ident = False

    for _ in range(IDENT_RESAMPLE_MAX):
        cand = _sample_positional(rng, scene, cell)
        cand_rule_x = compute_rule_x_values(_rule_inputs_from_latent(cand))
        try:
            cand_ids, cand_tau = filter_distractors_with_shortfall_relaxation(
                rule_x_values=cand_rule_x,
                gold_rule_id=gold_rule,
                F=cand.F,
                required_count=3,
                tau=TAU_POSITIONAL,
            )
        except NotEnoughDistractorsError:
            continue
        cand_picks, cand_tau_pair, cand_strat, cand_pair = select_stratified_distractors(
            rng=rng,
            rule_x_values=cand_rule_x,
            gold_rule_id=gold_rule,
            F=cand.F,
            candidate_ids=cand_ids,
            required_count=3,
            tau_pair=TAU_PAIR_POSITIONAL,
        )
        if not (cand_strat and cand_pair):
            continue
        latent = cand
        rule_x = cand_rule_x
        distractor_ids, tau_used = cand_ids, cand_tau
        picks = cand_picks
        tau_pair_used = cand_tau_pair
        stratified_ok, pairwise_ok = cand_strat, cand_pair
        break

    if latent is None:
        latent = _sample_positional(rng, scene, cell)
        rule_x = compute_rule_x_values(_rule_inputs_from_latent(latent))
        distractor_ids, tau_used, degraded_ident = filter_distractors_with_hard_fallback(
            rule_x_values=rule_x,
            gold_rule_id=gold_rule,
            F=latent.F,
            required_count=3,
            tau=TAU_POSITIONAL,
        )
        picks, tau_pair_used, stratified_ok, pairwise_ok = select_stratified_distractors(
            rng=rng,
            rule_x_values=rule_x,
            gold_rule_id=gold_rule,
            F=latent.F,
            candidate_ids=distractor_ids,
            required_count=3,
            tau_pair=TAU_PAIR_POSITIONAL,
        )

    options = build_mcq_options(
        rng=rng,
        gold_rule_id=gold_rule,
        candidate_distractor_ids=picks,
        rule_x_values=rule_x,
        action_label=scene.action_label,
        peer_noun_singular=scene.peer_noun_singular or "peer",
        peer_noun_plural=scene.peer_noun_plural or "peers",
        value_format=scene.value_format,
        unit=scene.action_unit,
        num_distractors=3,
        forced_gold_letter=letter,
    )
    gold_letter = next(o.letter for o in options if o.rule_id == gold_rule)
    margins = {
        o.rule_id: abs(o.x_value - rule_x[gold_rule]) / max(latent.F, 1e-9)
        for o in options if o.rule_id != gold_rule
    }
    identification_margin = min(margins.values()) if margins else 0.0
    gold_x = rule_x[gold_rule]

    oracle["latent"] = _latent_to_dict(latent)
    oracle["peer_cards"] = _build_peer_cards(latent, scene)
    oracle["gold"] = {
        "rule_id": gold_rule,
        "x_value": round(gold_x, 6),
        "x_display": format_value(gold_x, scene.value_format, scene.action_unit),
    }
    rec["mcq"] = {
        "options": [
            {"letter": o.letter, "text": o.text, "x_display": o.x_display}
            for o in options
        ],
    }
    oracle["mcq"] = {
        "gold_letter": gold_letter,
        "gold_rule_id": gold_rule,
        "identification_tau": round(tau_used, 6),
        "identification_margin": round(identification_margin, 6),
        "identification_degraded": degraded_ident,
        "pair_tau_used": round(tau_pair_used, 6),
        "selection_stratified": stratified_ok,
        "selection_pairwise_ok": pairwise_ok,
        "selection_degraded": not (stratified_ok and pairwise_ok),
        "all_rule_x_values": {k: round(v, 6) for k, v in rule_x.items()},
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
    return {
        "selection_degraded": not (stratified_ok and pairwise_ok),
        "identification_degraded": degraded_ident,
    }


# ---------------------------------------------------------------------------
# placebo (placebo_test)
# ---------------------------------------------------------------------------

def _sample_placebo(rng: random.Random, scene: Scene):
    F_val = rng.uniform(*scene.F_range)
    latent = sample_latent_langtry(
        rng=rng,
        domain_key=PLACEBO_DOMAIN,
        F=F_val,
        alpha_bucket="mid",
        dispersion_bucket="mid",
        skew_bucket="mid",
    )
    latent.alpha_bucket = "placebo"
    latent.cell_id = "placebo"
    latent.is_held_out_cell = False
    return latent


def _resample_one_placebo(
    rec: Dict[str, Any], scene: Scene, rng: random.Random, letter: str,
) -> Dict[str, Any]:
    oracle = rec["oracle"]
    gold_rule = "D_pure_private"
    trial_alpha_spec = ALPHA_BUCKETS["mid"]

    latent = None
    rule_x: Dict[str, float] = {}
    trial_alpha = 0.0
    distractor_ids: List[str] = []
    picks: List[str] = []
    tau_used = TAU_PLACEBO
    tau_pair_used = TAU_PAIR_PLACEBO
    stratified_ok = False
    pairwise_ok = False
    degraded_ident = False

    def _draw():
        la = _sample_placebo(rng, scene)
        ta = rng.uniform(trial_alpha_spec.lower, trial_alpha_spec.upper)
        rx = compute_rule_x_values(_rule_inputs_with_alpha(la, ta))
        rx["D_pure_private"] = la.F
        return la, rx, ta

    for _ in range(IDENT_RESAMPLE_MAX):
        cand, cand_rule_x, cand_alpha = _draw()
        try:
            cand_ids, cand_tau = filter_distractors_with_shortfall_relaxation(
                rule_x_values=cand_rule_x,
                gold_rule_id=gold_rule,
                F=cand.F,
                required_count=3,
                tau=TAU_PLACEBO,
            )
        except NotEnoughDistractorsError:
            continue
        cand_picks, cand_tau_pair, cand_strat, cand_pair = select_stratified_distractors(
            rng=rng,
            rule_x_values=cand_rule_x,
            gold_rule_id=gold_rule,
            F=cand.F,
            candidate_ids=cand_ids,
            required_count=3,
            tau_pair=TAU_PAIR_PLACEBO,
        )
        if not (cand_strat and cand_pair):
            continue
        latent, rule_x, trial_alpha = cand, cand_rule_x, cand_alpha
        distractor_ids, tau_used = cand_ids, cand_tau
        picks = cand_picks
        tau_pair_used = cand_tau_pair
        stratified_ok, pairwise_ok = cand_strat, cand_pair
        break

    if latent is None:
        latent, rule_x, trial_alpha = _draw()
        distractor_ids, tau_used, degraded_ident = filter_distractors_with_hard_fallback(
            rule_x_values=rule_x,
            gold_rule_id=gold_rule,
            F=latent.F,
            required_count=3,
            tau=TAU_PLACEBO,
        )
        picks, tau_pair_used, stratified_ok, pairwise_ok = select_stratified_distractors(
            rng=rng,
            rule_x_values=rule_x,
            gold_rule_id=gold_rule,
            F=latent.F,
            candidate_ids=distractor_ids,
            required_count=3,
            tau_pair=TAU_PAIR_PLACEBO,
        )

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
        forced_gold_letter=letter,
    )
    gold_letter = next(o.letter for o in options if o.rule_id == gold_rule)
    margins = {
        o.rule_id: abs(o.x_value - rule_x[gold_rule]) / max(latent.F, 1e-9)
        for o in options if o.rule_id != gold_rule
    }
    identification_margin = min(margins.values()) if margins else 0.0
    gold_x = rule_x[gold_rule]

    oracle["latent"] = _latent_to_dict(latent)
    oracle["peer_cards"] = _build_peer_cards(latent, scene)
    oracle["gold"] = {
        "rule_id": gold_rule,
        "x_value": round(gold_x, 6),
        "x_display": format_value(gold_x, scene.value_format, scene.action_unit),
    }
    rec["mcq"] = {
        "options": [
            {"letter": o.letter, "text": o.text, "x_display": o.x_display}
            for o in options
        ],
    }
    oracle["mcq"] = {
        "gold_letter": gold_letter,
        "gold_rule_id": gold_rule,
        "identification_tau": round(tau_used, 6),
        "identification_margin": round(identification_margin, 6),
        "identification_degraded": degraded_ident,
        "pair_tau_used": round(tau_pair_used, 6),
        "selection_stratified": stratified_ok,
        "selection_pairwise_ok": pairwise_ok,
        "selection_degraded": not (stratified_ok and pairwise_ok),
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
    return {
        "selection_degraded": not (stratified_ok and pairwise_ok),
        "identification_degraded": degraded_ident,
    }


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", required=True,
                   help="structured JSON (eval_A.json or placebo_test.json)")
    p.add_argument("--output-file", default=None,
                   help="destination path; defaults to in-place overwrite")
    p.add_argument("--seed", type=int, default=20260421,
                   help="rng seed base (final seed = base ^ hash(record_id))")
    p.add_argument("--dry-run", action="store_true",
                   help="Only count degraded records; do not modify the file.")
    return p.parse_args()


def _is_degraded(rec: Dict[str, Any]) -> bool:
    m = (rec.get("oracle") or {}).get("mcq") or {}
    return bool(m.get("selection_degraded"))


def main() -> None:
    args = parse_args()
    src = Path(args.input_file)
    dst = Path(args.output_file) if args.output_file else src
    payload = json.loads(src.read_text(encoding="utf-8"))
    split = payload.get("split") or src.stem
    records: List[Dict[str, Any]] = payload.get("records", [])

    is_placebo = split == "placebo_test"
    resample_fn = _resample_one_placebo if is_placebo else _resample_one_positional

    scene_index = _index_scenes()
    degraded_ids = [r["record_id"] for r in records if _is_degraded(r)]

    summary: Dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "resampled",
        "split": split,
        "input": str(src),
        "output": str(dst),
        "total_records": len(records),
        "degraded_before": len(degraded_ids),
        "resampled": 0,
        "still_degraded_after": 0,
        "cleared_scenario_text": 0,
        "dropped_teacher_meta": 0,
        "skipped_missing_scene": 0,
        "record_ids": degraded_ids,
    }

    if args.dry_run or not degraded_ids:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    letter_cycle = ("A", "B", "C", "D")
    for rec in records:
        if not _is_degraded(rec):
            continue
        ensure_oracle(rec)
        scene = scene_index.get(rec["scene"]["scene_id"])
        if scene is None:
            summary["skipped_missing_scene"] += 1
            continue
        rng = random.Random(_record_seed(args.seed, rec["record_id"]))
        # Deterministic letter: derive from record_id so A/B/C/D remain
        # roughly balanced across the small resampled set.
        letter = letter_cycle[abs(hash(rec["record_id"])) % 4]
        flags = resample_fn(rec, scene, rng, letter)
        summary["resampled"] += 1
        if flags["selection_degraded"]:
            summary["still_degraded_after"] += 1
        # Clear scenario_text + teacher_meta so write_scenarios_with_llm.py
        # picks these (and only these) up on the next run.
        if rec.get("scenario_text") is not None:
            summary["cleared_scenario_text"] += 1
        rec["scenario_text"] = None
        if rec.pop("teacher_meta", None) is not None:
            summary["dropped_teacher_meta"] += 1

    payload["records"] = records
    payload.setdefault("resample_history", []).append({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "resampled_record_ids": list(degraded_ids),
        "still_degraded_after": summary["still_degraded_after"],
    })
    payload["resample_last_run_utc"] = datetime.now(timezone.utc).isoformat()

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
