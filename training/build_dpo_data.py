"""
Build DPO preference pairs from `data/structured/train.json` + teacher narratives.

For each training record we keep the SAME prompt (student-facing scenario) and
emit ONE chosen / rejected pair:

    chosen    = CoT ending in the gold-rule conclusion
                (A_peer_weighted for d1/d3, D_pure_private for d2 placebo)
    rejected  = CoT with the SAME taxonomy preamble and qualitative body, but
                the conclusion commits to a shortcut rule instead:
                  * d1/d3 positional : one of {B_top_anchor, C_uniform_avg}
                  * d2 placebo       : A_peer_weighted
                    (i.e. invented peer pull when there is no real audience)

The preamble / body are deliberately identical between chosen and rejected so
the preference signal is concentrated on the *rule commitment*, not on
surface form. We sample the rejected rule deterministically-but-varied
(per record_id) so each prompt has a well-defined (chosen, rejected) pair.

Lexical audit mirrors the SFT builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config.rule_label_templates import (  # noqa: E402
    RULE_SPECS,
    RuleInputs,
    compute_rule_x_values,
    format_value,
    gold_rule_for_domain,
)
from data.config.vocabulary_pools import (  # noqa: E402
    find_forbidden_research_terms,
    find_mcq_rule_leaks_in_cot,
)
from training.build_sft_data import (  # noqa: E402
    STUDENT_SYSTEM_PROMPT,
    _compose_student_user_prompt,
    _protagonist_for,
)
from training.cot_templates import (  # noqa: E402
    CoTContext,
    PreambleContext,
    build_taxonomy_preamble,
    domain_to_scene_type,
    _body_for_scene_type,
    SCENE_TYPE_NON_POSITIONAL,
    SCENE_TYPE_POSITIONAL_CONSUMPTION,
    SCENE_TYPE_POSITIONAL_HUMAN_CAPITAL,
    SCENE_TYPE_TRICKY_POSITIONAL,
)


DPO_SCHEMA_VERSION = "dpo_v1"

# Rules we prefer as "rejected" shortcuts.
POSITIONAL_SHORTCUT_RULES: Tuple[str, ...] = (
    "B_top_anchor",
    "C_uniform_avg",
)
# Fallback shortcuts for positional when the two preferred ones collapse to
# the gold under low alpha. Order reflects increasing rarity.
POSITIONAL_FALLBACK_SHORTCUTS: Tuple[str, ...] = (
    "F_median_anchor",
    "H_equal_mix",
    "E_closest_mimicry",
)
PLACEBO_SHORTCUT_RULES: Tuple[str, ...] = (
    "A_peer_weighted",
)

# For placebo records alpha_i is effectively 0; we use a trial alpha when
# computing the rejected rule's x-value so the shortcut number is legibly
# above F (matching what placebo_test does).
PLACEBO_TRIAL_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Rule-anchored conclusion templates (used for BOTH chosen and rejected
# endings). Each rule has two paraphrase conclusions that lock the commitment
# number onto that rule's predicted value.
# ---------------------------------------------------------------------------

_RULE_CONCLUSIONS: Dict[str, Tuple[str, ...]] = {
    "A_peer_weighted": (
        "Gold rule: anchor on the baseline and lift by the closeness-weighted reference, giving a target {action_label} near {x_display}.",
        "The implied commitment is the private floor plus the closeness-weighted peer reference, landing near {x_display}.",
    ),
    "D_pure_private": (
        "Gold rule: hold the private floor; the target {action_label} equals the baseline, near {x_display}.",
        "No peer lift applies; the commitment stays at the private baseline, about {x_display}.",
    ),
    "B_top_anchor": (
        "Rule of thumb: copy the single nearest relationship's level on top of the baseline, so the {action_label} lands near {x_display}.",
        "Pin the lift to just the one most close-in-life relationship; the commitment ends up around {x_display}.",
    ),
    "C_uniform_avg": (
        "Rule of thumb: give every member of the circle the same vote and lift the baseline by that flat group average, which puts the {action_label} near {x_display}.",
        "Treat the circle as a plain average above the baseline; the commitment lands around {x_display}.",
    ),
    # The following are not used as default shortcuts but are available if
    # someone later wants E/F/G/H contrast pairs.
    "E_closest_mimicry": (
        "Rule of thumb: mirror the nearest relationship's level outright, ignoring the private baseline; the {action_label} lands around {x_display}.",
        "Drop the private floor and just match the closest relationship; the commitment ends up near {x_display}.",
    ),
    "F_median_anchor": (
        "Rule of thumb: lift the baseline by the middle relationship's level; the {action_label} ends up near {x_display}.",
        "Anchor on the median of the circle above the baseline; the commitment is about {x_display}.",
    ),
    "G_counter_conformist": (
        "Rule of thumb: push below the baseline in opposition to the circle's pull; the {action_label} lands around {x_display}.",
        "Treat peer pull as a reason to pull back below the private floor; the commitment ends up near {x_display}.",
    ),
    "H_equal_mix": (
        "Rule of thumb: put equal share on the baseline and on the closeness-weighted peer aggregate; the {action_label} is near {x_display}.",
        "Put equal share between the private floor and the weighted peer aggregate; the commitment lands around {x_display}.",
    ),
}


def _rule_conclusion(rule_id: str, rng: random.Random, action_label: str, x_display: str) -> str:
    if rule_id not in _RULE_CONCLUSIONS:
        raise ValueError(f"No DPO conclusion template for rule `{rule_id}`.")
    template = rng.choice(_RULE_CONCLUSIONS[rule_id])
    return template.format(action_label=action_label or "commitment", x_display=x_display)


# ---------------------------------------------------------------------------
# CoT assembly (parametrised by which rule the conclusion commits to)
# ---------------------------------------------------------------------------

def _build_rule_anchored_cot(
    ctx: CoTContext,
    rng: random.Random,
    rule_id: str,
    x_display: str,
) -> str:
    scene_type_resolved = domain_to_scene_type(ctx.domain_key)
    preamble_scene_type = (
        SCENE_TYPE_TRICKY_POSITIONAL if ctx.tricky_positional else scene_type_resolved
    )
    preamble_ctx = PreambleContext(
        scene_type=preamble_scene_type,
        protagonist=ctx.protagonist,
        resolved_scene_type=scene_type_resolved,
        cf_rule_out_override=ctx.cf_rule_out_override,
    )
    preamble = build_taxonomy_preamble(preamble_ctx, rng=rng)
    body = _body_for_scene_type(scene_type_resolved, rng).format(
        protagonist=ctx.protagonist or "this person",
        peer_noun_singular=ctx.peer_noun_singular or "peer",
        peer_noun_plural=ctx.peer_noun_plural or "peers",
    )
    conclusion = _rule_conclusion(rule_id, rng, ctx.action_label, x_display)
    return " ".join(["Reasoning:", preamble, body, conclusion])


# ---------------------------------------------------------------------------
# Shortcut-rule selection
# ---------------------------------------------------------------------------

def _pick_shortcut_rule(record: Dict[str, Any], gold_rule: str) -> str:
    """Pick a deterministic-but-varied shortcut rule per record."""
    scene_type = domain_to_scene_type(record["scene"]["domain_key"])
    if scene_type == SCENE_TYPE_NON_POSITIONAL:
        options = PLACEBO_SHORTCUT_RULES
    else:
        options = POSITIONAL_SHORTCUT_RULES
    # Hash the record_id so the same record always gets the same rejected
    # rule, but different records spread across the options.
    digest = hashlib.blake2s(record["record_id"].encode("utf-8"), digest_size=2).digest()
    idx = int.from_bytes(digest, "big") % len(options)
    rule = options[idx]
    if rule == gold_rule:
        rule = options[(idx + 1) % len(options)]
    return rule


def _rule_inputs_from_latent(latent: Dict[str, Any]) -> RuleInputs:
    g_ij = list(latent.get("g_ij") or [])
    closest = 0
    if g_ij:
        closest = max(range(len(g_ij)), key=lambda i: g_ij[i])
    return RuleInputs(
        F=float(latent["F"]),
        alpha_i=float(latent["alpha_i"]),
        x_j=list(latent.get("x_j") or []),
        g_ij=g_ij,
        closest_peer_index=closest,
    )


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------

@dataclass
class DPOAuditStats:
    total_in: int = 0
    kept: int = 0
    dropped_no_scenario: int = 0
    dropped_audit: int = 0
    dropped_no_distinct_shortcut: int = 0


def build_pairs(
    records: Sequence[Dict[str, Any]],
    rng: random.Random,
    final_commitment_suffix: bool = True,
) -> Tuple[List[Dict[str, Any]], DPOAuditStats]:
    stats = DPOAuditStats(total_in=len(records))
    out: List[Dict[str, Any]] = []

    for rec in records:
        scenario_text = rec.get("scenario_text")
        if not scenario_text:
            stats.dropped_no_scenario += 1
            continue
        if find_forbidden_research_terms(scenario_text):
            stats.dropped_audit += 1
            continue

        scene = rec["scene"]
        latent = rec["oracle"]["latent"]
        domain_key = scene["domain_key"]
        gold_rule = gold_rule_for_domain(domain_key)
        shortcut_rule = _pick_shortcut_rule(rec, gold_rule)

        rule_inputs = _rule_inputs_from_latent(latent)
        is_placebo = domain_key == "domain2_non_positional_investment"
        if is_placebo:
            # Compute all rule values under a trial alpha so the rejected
            # shortcut reads as an AUDIENCE-GIVEN lift above F.
            trial_inputs = RuleInputs(
                F=rule_inputs.F,
                alpha_i=PLACEBO_TRIAL_ALPHA,
                x_j=rule_inputs.x_j,
                g_ij=rule_inputs.g_ij,
                closest_peer_index=rule_inputs.closest_peer_index,
            )
            all_rule_x = compute_rule_x_values(trial_inputs)
            all_rule_x["D_pure_private"] = rule_inputs.F  # gold stays at F exactly
        else:
            all_rule_x = compute_rule_x_values(rule_inputs)

        gold_x_value = rec["oracle"]["gold"]["x_value"]
        shortcut_x_value = all_rule_x[shortcut_rule]

        # Ensure rejected number is distinct from chosen.
        F_ref = max(float(latent.get("F", 1.0)), 1e-9)
        min_margin = 0.05
        if abs(shortcut_x_value - gold_x_value) / F_ref < min_margin:
            # Walk through preferred shortcuts, then fallback shortcuts.
            if is_placebo:
                candidate_rules: Tuple[str, ...] = PLACEBO_SHORTCUT_RULES
            else:
                candidate_rules = POSITIONAL_SHORTCUT_RULES + POSITIONAL_FALLBACK_SHORTCUTS
            alt = [r for r in candidate_rules if r != shortcut_rule and r != gold_rule]
            found = False
            for r in alt:
                if abs(all_rule_x[r] - gold_x_value) / F_ref >= min_margin:
                    shortcut_rule = r
                    shortcut_x_value = all_rule_x[r]
                    found = True
                    break
            if not found:
                stats.dropped_no_distinct_shortcut += 1
                continue

        chosen_display = format_value(
            gold_x_value, scene.get("value_format", "currency"), scene.get("action_unit", "")
        )
        rejected_display = format_value(
            shortcut_x_value, scene.get("value_format", "currency"), scene.get("action_unit", "")
        )

        cot_ctx = CoTContext(
            domain_key=domain_key,
            protagonist=_protagonist_for(rec),
            peer_noun_singular=scene.get("peer_noun_singular") or "peer",
            peer_noun_plural=scene.get("peer_noun_plural") or "peers",
            action_label=scene.get("action_label") or "commitment",
            x_display=chosen_display,
        )

        chosen_cot = _build_rule_anchored_cot(cot_ctx, rng, gold_rule, chosen_display)
        rejected_cot = _build_rule_anchored_cot(cot_ctx, rng, shortcut_rule, rejected_display)

        # Audit
        if any(find_mcq_rule_leaks_in_cot(t) or find_forbidden_research_terms(t)
               for t in (chosen_cot, rejected_cot)):
            stats.dropped_audit += 1
            continue

        action_label = scene.get("action_label") or "commitment"
        if final_commitment_suffix:
            chosen_response = (
                chosen_cot + f"\n\nFinal commitment: {chosen_display} for the {action_label}."
            )
            rejected_response = (
                rejected_cot + f"\n\nFinal commitment: {rejected_display} for the {action_label}."
            )
        else:
            chosen_response = chosen_cot
            rejected_response = rejected_cot

        user_prompt = _compose_student_user_prompt(rec)

        out.append({
            "dpo_id": f"dpo__{rec['record_id']}",
            "dpo_schema_version": DPO_SCHEMA_VERSION,
            "source_record_id": rec["record_id"],
            "dataset_split": rec["dataset_split"],
            "scene_id": scene["scene_id"],
            "domain_key": domain_key,
            "is_placebo": domain_key == "domain2_non_positional_investment",
            "system_prompt": STUDENT_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "chosen_rule_id": gold_rule,
            "rejected_rule_id": shortcut_rule,
            "chosen_x_value": round(gold_x_value, 6),
            "rejected_x_value": round(shortcut_x_value, 6),
            "cell_id": latent.get("cell_id"),
            "alpha_bucket": latent.get("alpha_bucket"),
            "is_held_out_cell": latent.get("is_held_out_cell", False),
        })
        stats.kept += 1
    return out, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", default="data/structured/train.json",
                   help="Structured train split with teacher-filled scenario_text.")
    p.add_argument("--output-file", default="data/dpo/train_pairs.jsonl",
                   help="Destination JSONL. One DPO preference pair per line.")
    p.add_argument("--seed", type=int, default=20260422)
    p.add_argument("--no-final-commitment", action="store_true",
                   help="Skip appending the 'Final commitment:' suffix to chosen/rejected.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed + 20000)
    payload = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    records = payload.get("records", [])

    pairs, stats = build_pairs(
        records, rng, final_commitment_suffix=not args.no_final_commitment
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for p in pairs:
            handle.write(json.dumps(p, ensure_ascii=False) + "\n")

    summary = {
        "status": "dpo_written",
        "input": args.input_file,
        "output": str(out_path),
        "total_in": stats.total_in,
        "kept": stats.kept,
        "dropped_no_scenario": stats.dropped_no_scenario,
        "dropped_audit": stats.dropped_audit,
        "dropped_no_distinct_shortcut": stats.dropped_no_distinct_shortcut,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
