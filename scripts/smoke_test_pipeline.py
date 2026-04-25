"""
Standalone smoke test for the W1-Day-1 config files.

What this script does (NO teacher / API calls):
  1. Picks a d1 positional scene from the train-split of scene_pool.
  2. Samples a latent Langtry record using langtry_parameters.
  3. Computes the predicted x-value for each of the 8 behavioural rules in
     rule_label_templates.
  4. Assembles an Eval-A-style MCQ with a code-templated rule paraphrase per
     option.
  5. Builds a SFT-style CoT using the 4-variant taxonomy preamble in
     training/cot_templates.
  6. Runs the lexical leakage check from vocabulary_pools on the CoT.
  7. Prints the full record + CoT to stdout and writes it to
     data/smoke/smoke_d1_record.json.

Run from the repo root:
    cd /hpc2hdd/home/jliu043/relational_state
    python -m scripts.smoke_test_pipeline --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Make the repo root importable.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.config.langtry_parameters import (  # noqa: E402
    MAIN_POSITIONAL_DOMAINS,
    iter_main_cells,
    sample_latent_langtry,
)
from data.config.scene_pool import list_scenes  # noqa: E402
from data.config.rule_label_templates import (  # noqa: E402
    RULE_IDS,
    RULE_SPECS,
    RuleInputs,
    build_mcq_options,
    compute_rule_x_values,
    format_value,
    gold_rule_for_domain,
)
from data.config.vocabulary_pools import (  # noqa: E402
    find_forbidden_research_terms,
    find_mcq_rule_leaks_in_cot,
)
from training.cot_templates import (  # noqa: E402
    CoTContext,
    build_full_cot,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W1 Day 1 smoke test.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--domain", default="domain1_positional_consumption",
                   choices=list(MAIN_POSITIONAL_DOMAINS) + ["domain2_non_positional_investment"])
    p.add_argument("--output", default="data/smoke/smoke_d1_record.json")
    return p.parse_args()


def pick_scene(rng: random.Random, domain_key: str):
    pool = list_scenes(domain_key=domain_key, split="train")
    if not pool:
        raise RuntimeError(f"No training scenes for `{domain_key}`.")
    return rng.choice(pool)


def pick_cell(rng: random.Random):
    return rng.choice(iter_main_cells())


def build_identification_filter_candidates(
    gold_id: str,
    rule_x_values: Dict[str, float],
    F: float,
    tau: float = 0.05,
) -> List[str]:
    """Return rule ids (excluding gold) whose predicted x differs from gold by at least tau * F."""
    gold_x = rule_x_values[gold_id]
    candidates: List[str] = []
    for rid, x in rule_x_values.items():
        if rid == gold_id:
            continue
        gap = abs(x - gold_x)
        if F <= 0:
            continue
        if gap / F >= tau:
            candidates.append(rid)
    return candidates


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    scene = pick_scene(rng, args.domain)
    alpha_b, disp_b, skew_b = pick_cell(rng)

    F = rng.uniform(*scene.F_range)
    latent = sample_latent_langtry(
        rng=rng,
        domain_key=args.domain,
        F=F,
        alpha_bucket=alpha_b,
        dispersion_bucket=disp_b,
        skew_bucket=skew_b,
    )

    closest_j = latent.peer_rank_by_g[0]
    rule_inp = RuleInputs(
        F=latent.F,
        alpha_i=latent.alpha_i,
        x_j=latent.x_j,
        g_ij=latent.g_ij,
        closest_peer_index=closest_j,
    )
    rule_x_values = compute_rule_x_values(rule_inp)

    # Identification filter
    gold_rule = gold_rule_for_domain(args.domain)
    distractor_pool = build_identification_filter_candidates(gold_rule, rule_x_values, latent.F, tau=0.05)

    # Fallback: if too few candidates pass the filter, fall back to all 7 alternatives.
    if len(distractor_pool) < 3:
        distractor_pool = [rid for rid in RULE_IDS if rid != gold_rule]

    mcq = build_mcq_options(
        rng=rng,
        gold_rule_id=gold_rule,
        candidate_distractor_ids=distractor_pool,
        rule_x_values=rule_x_values,
        action_label=scene.action_label,
        peer_noun_singular=scene.peer_noun_singular or "peer",
        peer_noun_plural=scene.peer_noun_plural or "peers",
        value_format=scene.value_format,
        unit=scene.action_unit,
    )

    # Build a SFT-style CoT
    cot_ctx = CoTContext(
        domain_key=args.domain,
        protagonist="the protagonist",
        peer_noun_singular=scene.peer_noun_singular or "peer",
        peer_noun_plural=scene.peer_noun_plural or "peers",
        action_label=scene.action_label,
        x_display=format_value(latent.x_i_star, scene.value_format, scene.action_unit),
        tricky_positional=False,
    )
    cot_text = build_full_cot(cot_ctx, rng=rng)

    # Lexical audits
    mcq_leaks = find_mcq_rule_leaks_in_cot(cot_text)
    forbidden_hits = find_forbidden_research_terms(cot_text)
    assert not mcq_leaks, f"CoT leaks MCQ rule vocab: {mcq_leaks}"
    assert not forbidden_hits, f"CoT contains forbidden research terms: {forbidden_hits}"

    out_record: Dict[str, Any] = {
        "smoke_test_version": "w1_day1",
        "seed": args.seed,
        "scene": {
            "scene_id": scene.scene_id,
            "domain_key": scene.domain_key,
            "family": scene.family,
            "title": scene.title,
            "summary": scene.summary,
            "action_label": scene.action_label,
            "action_unit": scene.action_unit,
            "value_format": scene.value_format,
            "F_range": scene.F_range,
            "peer_noun_plural": scene.peer_noun_plural,
        },
        "latent_langtry": {
            "alpha_i": round(latent.alpha_i, 6),
            "F": round(latent.F, 4),
            "N_peers": latent.N_peers,
            "x_j": [round(v, 4) for v in latent.x_j],
            "g_ij": [round(v, 6) for v in latent.g_ij],
            "ref_sum": round(latent.ref_sum, 4),
            "x_i_star": round(latent.x_i_star, 4),
            "alpha_bucket": latent.alpha_bucket,
            "dispersion_bucket": latent.dispersion_bucket,
            "skew_bucket": latent.skew_bucket,
            "cell_id": latent.cell_id,
            "is_held_out_cell": latent.is_held_out_cell,
            "peer_rank_by_g": latent.peer_rank_by_g,
            "peer_rank_by_x": latent.peer_rank_by_x,
        },
        "rule_x_values": {rid: round(x, 4) for rid, x in rule_x_values.items()},
        "gold_rule_id": gold_rule,
        "distractor_pool_after_filter": distractor_pool,
        "mcq_options": [
            {
                "letter": o.letter,
                "rule_id": o.rule_id,
                "rule_short": o.rule_short,
                "text": o.text,
                "x_value": round(o.x_value, 4),
                "x_display": o.x_display,
            }
            for o in mcq
        ],
        "gold_choice_letter": next(o.letter for o in mcq if o.rule_id == gold_rule),
        "cot_text": cot_text,
        "audit": {
            "mcq_rule_vocab_leaks_in_cot": mcq_leaks,
            "forbidden_research_terms_in_cot": forbidden_hits,
        },
    }

    out_path = _ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out_record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
